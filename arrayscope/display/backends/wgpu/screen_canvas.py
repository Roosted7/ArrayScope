"""Embedded QWindow driving its own wgpu swapchain (gate-B recipe, v2).

The bitmap present path pays a 4-7 ms/frame GPU->CPU readback through
rendercanvas.  Gate B measured the native-Wayland screen path at ~0.6 ms
encode+submit / 0.08 ms present (docs/proposals/wgpu-renderer-experiment.md
Tier 1; probe `experiments/wgpu_gate_b/probe_native_wayland.py`).  This
module productionizes that probe:

* the swapchain target is a bare ``QWindow`` embedded through
  ``QWidget.createWindowContainer``.  Qt never paints it (no backing store
  is ever created), so its ``wl_surface`` belongs to wgpu alone; Qt
  committing SHM backing-store buffers to the same surface is a fatal
  compositor protocol error (explicit-sync dmabuf-only).
* the container is the load-bearing choice (2026-07-19 glitch dossier): a
  native child *widget* (``WA_PaintOnScreen``/``winId()``) makes Qt promote
  the widget's whole ancestor chain — and, without
  ``AA_DontCreateNativeWidgetSiblings``, every sibling too — into native
  windows, shattering the top-level into a soup of desynchronized
  ``wl_subsurface``s (white/hole regions, hidden overlays, resize
  old/new-size flicker; confirmed with weston headless + WAYLAND_DEBUG).
  ``createWindowContainer`` parents the QWindow directly to the top-level
  window instead: exactly one subsurface, composited above the top-level's
  Qt-painted pixels.  Overlays that must stay visible above the canvas opt
  in via ``WgpuImageView2D._prepare_display_overlay_widget``.
* the wgpu surface is created from the REAL in-process ``wl_display``
  (``QNativeInterface.QWaylandApplication.display()``) plus the embedded
  window's ``winId()`` as the ``wl_surface*``.  Both are undocumented Qt
  contracts pinned per Qt minor by
  ``tests/gpu_interaction/test_wgpu_native_wayland_pin.py`` (ring 4).
* rendercanvas is bypassed entirely: wgpu-py's ``GPUCanvasContext`` accepts
  the present-info dict directly and owns configure/acquire/present plus
  size-change reconfiguration, so no import-time env stomping ever runs.
* Fifo acquire blocks the GUI thread ~15 ms/frame (gate-B tier-1 caveat), so
  after ``configure()`` the context is re-configured for Mailbox when the
  surface advertises it; when only Fifo exists the acquire stays on-thread
  and the view's per-frame acquire timing diagnostics are the guard rail.

The same recipe drives X sessions (2026-07-31), because the readback the
screen path avoids costs the same 7 ms at 1300x650 / 26 ms at 4K there.  The
mechanics differ only in which native interface hands over the display
connection (``QX11Application`` rather than ``QWaylandApplication``) and
which wgpu surface source that selects (``WGPUSurfaceSourceXlibWindow``);
everything below — pacing, Mailbox preference, resize coalescing, teardown —
is platform-independent and unchanged.

The three Wayland-shaped choices above were re-derived for X11 rather than
assumed to transfer, since a native child there is a different animal:

* *paint-less window*: still required, for a weaker reason.  On Wayland a Qt
  SHM backing-store buffer on the swapchain's surface is a FATAL protocol
  error; on X11 it is merely two writers racing into one drawable, i.e.
  flicker.  One recipe, and the stronger constraint is the safe one.
* *window container over a native child widget*: kept, and it costs nothing.
  The pathology it was introduced for is Wayland-only — X11 has no
  subsurfaces to shatter into, and native siblings are ordinary server-side
  windows the X server clips and stacks itself.  What DOES carry over is the
  reason overlays live inside the wgpu frame: Qt cannot composite its own
  painting above a native child window on either platform (gate B measured
  exactly that under xcb).  Overlay ownership keys on the present method,
  not the platform, so it already covers X11 — confirmed on captured X11
  pixels, ``docs/redesign/wgpu-screen-on-x11-2026-07-31.md``.
* *``WindowTransparentForInput``*: on Wayland an empty input region, on X11
  an empty XShape input region.  Different mechanism, same contract, and the
  screen-present input oracles run on both.

The screen path exists only where a native surface can actually be built;
every other environment (offscreen, missing native interface) reports a loud
fallback reason and the view keeps the bitmap path.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from arrayscope.app.qt_binding import prefer_pyside6
from arrayscope.core.trace import emit_trace
from arrayscope.display.backends.wgpu.frame_timing import FrameTimingRecorder

prefer_pyside6()

import contextlib

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


@dataclass(frozen=True)
class NativePresentTarget:
    """The (windowing system, display connection) a wgpu surface is built on.

    ``platform`` is wgpu-py's present-info platform key, NOT Qt's QPA name:
    Qt calls the X11 platform ``xcb`` while wgpu-py's Xlib surface source is
    selected by ``"x11"``.  Keeping the translation in one named place stops
    the two vocabularies leaking into the canvas.
    """

    platform: str
    display: int


def _resolve_native_present_target() -> tuple[NativePresentTarget | None, str | None]:
    """Which native surface recipe applies here, or why none does.

    Exactly one of the two is ever non-``None``.  Both supported QPAs hand
    wgpu the same two ingredients — the in-process display connection and
    the window handle — through different native interfaces, so the
    per-platform knowledge is confined to this function.
    """

    app = QtWidgets.QApplication.instance()
    if app is None:
        return None, "no QApplication"
    platform = str(app.platformName())
    try:
        from PySide6.QtGui import QNativeInterface
    except Exception as exc:  # pragma: no cover - PySide6 is the pinned binding
        return None, f"QNativeInterface unavailable ({exc})"
    iface = app.nativeInterface()
    if platform == "wayland":
        if not isinstance(iface, QNativeInterface.QWaylandApplication):
            return None, "no QWaylandApplication native interface"
        display = int(iface.display() or 0)
        if not display:
            return None, "QWaylandApplication reports a null wl_display"
        return NativePresentTarget("wayland", display), None
    if platform == "xcb":
        # QX11Application is looked up rather than imported: a PySide6 built
        # without X11 support has no such attribute, and that must read as a
        # fallback reason, not an AttributeError inside setupUI.
        x11_application = getattr(QNativeInterface, "QX11Application", None)
        if x11_application is None:
            return None, "PySide6 has no QX11Application native interface"
        if not isinstance(iface, x11_application):
            return None, "no QX11Application native interface"
        display = int(iface.display() or 0)
        if not display:
            return None, "QX11Application reports a null Display*"
        return NativePresentTarget("x11", display), None
    return None, f"Qt platform {platform!r} has no native swapchain recipe"


def screen_present_unavailable_reason() -> str | None:
    """Why ``present_method="screen"`` cannot exist here, or ``None`` if it can.

    Mirrors the tier-0 probes' preconditions exactly: a live QApplication on
    a QPA whose native interface can hand over the display connection its
    windows belong to — ``wayland`` via ``QWaylandApplication``, ``xcb`` via
    ``QX11Application``.  Checked BEFORE any widget is created so the caller
    can fall back to the bitmap canvas without leaving a half-built native
    child.
    """

    _, reason = _resolve_native_present_target()
    if reason is not None:
        return reason
    try:
        import wgpu  # noqa: F401
    except Exception as exc:
        return f"wgpu unavailable ({exc})"
    return None


class _ScreenSurfaceWindow(QtGui.QWindow):
    """Bare window owning the wl_surface the swapchain presents to.

    Qt creates the platform window (and its subsurface) but never attaches a
    buffer: no backing store exists and nothing ever renders through Qt.
    Input falls through (``WindowTransparentForInput`` -> empty Wayland input
    region), so pointer events keep landing on the top-level surface where
    the transparent interaction ``graphicsView`` receives them.
    """

    def __init__(self, canvas: WgpuScreenCanvas):
        super().__init__()
        self._canvas = canvas
        self.setSurfaceType(QtGui.QSurface.SurfaceType.RasterSurface)
        self.setFlags(self.flags() | QtCore.Qt.WindowType.WindowTransparentForInput)

    def exposeEvent(self, event) -> None:  # expose/damage -> redraw
        if self.isExposed():
            self._canvas.request_draw()

    def resizeEvent(self, event) -> None:
        # A subsurface's on-screen footprint IS its latest buffer: until a
        # new frame is presented the compositor keeps compositing the
        # old-size buffer against Qt's already-resized window, which reads
        # as flicker between the old and new sizes (and spill, since
        # subsurfaces are not clipped to the parent).  X11 has the flicker
        # half of that and not the spill half — the server clips a child
        # window to its parent — but the handling is the same either way.
        # This event fires when the embedded window's real geometry changed,
        # so arm a swapchain reconfigure and let the canvas present it on the
        # next paced frame.
        super().resizeEvent(event)
        self._canvas._on_surface_resized()


class WgpuScreenCanvas(QtWidgets.QWidget):
    """A widget hosting the swapchain window Qt never paints into.

    Presents rendercanvas's minimal canvas API surface (``request_draw`` /
    ``get_physical_size`` / ``devicePixelRatio``) so ``WgpuImageView2D``
    drives both present methods through one seam.  Draw scheduling is a
    coalesced zero-timer: Qt cannot deliver paint work for the embedded
    window, so the canvas owns its own on-demand cadence exactly like
    rendercanvas's ``ondemand`` update mode.
    """

    #: Fallback pace when the platform will not say what the display does.
    fallback_draws_per_second = 60.0

    @property
    def max_draws_per_second(self) -> float:
        """Draws per second to aim for: whatever THIS display refreshes at.

        Any fixed number is wrong somewhere — 60 stutters a 144 Hz panel and
        wastes half its frames on a 30 Hz one — so the pace is read from the
        screen the surface currently sits on, live, which also means dragging
        the window to a different monitor re-paces it with no extra wiring.

        This path shipped pinned at 30, which made a pan visibly step.  The
        rationale for that cap (unpaced presents exhaust the mailbox chain
        until acquire blocks) re-measured backwards: with the embedded
        QWindow recipe and the view's draw coalescing, 30 gave an 11.23 ms
        worst acquire against 0.91 ms at 60, so the cap was causing the
        stall it existed to prevent.

        Pacing above the refresh rate is pure waste under Mailbox — the
        compositor drops the extra frames — so the display rate IS the
        optimum, not a compromise.

        The rate is cached rather than re-read per draw request: this is on
        the pacing hot path, and the two things that can change it (the
        window moving to another screen, that screen changing mode) both
        announce themselves with a signal.  Polling Qt every frame for an
        answer that changes maybe twice a session is the wrong shape, and
        the signals keep the "drag it to a 144 Hz panel and it re-paces"
        behaviour exactly as before.
        """

        override = self._max_draws_per_second_override
        if override is not None:
            return override
        cached = self._cached_draws_per_second
        if cached is not None:
            return cached
        screen = self._current_screen()
        rate = float(screen.refreshRate()) if screen is not None else 0.0
        # Qt reports 0 when the platform does not know; never divide by it.
        rate = rate if rate >= 1.0 else self.fallback_draws_per_second
        self._cached_draws_per_second = rate
        self._watch_screen(screen)
        return rate

    def _current_screen(self):
        """The screen the surface sits on, or the primary as a stand-in."""

        window = self._surface_window
        screen = window.screen() if window is not None else None
        if screen is not None:
            return screen
        app = QtWidgets.QApplication.instance()
        return app.primaryScreen() if app is not None else None

    def _watch_screen(self, screen) -> None:
        """Re-arm the invalidation signals for whichever screen we just read.

        A mode change on the *current* screen and a move to a *different*
        screen are separate signals, so both are tracked; the old screen's
        connection is dropped so a background monitor's mode change cannot
        keep invalidating a cache it no longer describes.
        """

        if screen is self._watched_screen:
            return
        previous = self._watched_screen
        if previous is not None:
            with contextlib.suppress(Exception, RuntimeError):
                previous.refreshRateChanged.disconnect(self._invalidate_pace_cache)
        self._watched_screen = screen
        if screen is not None:
            with contextlib.suppress(Exception):
                screen.refreshRateChanged.connect(self._invalidate_pace_cache)

    def _invalidate_pace_cache(self, *_args) -> None:
        """Next ``request_draw`` re-reads the display rate."""

        self._cached_draws_per_second = None

    @max_draws_per_second.setter
    def max_draws_per_second(self, value) -> None:
        """Pin the pace (benchmarks and tests); ``None`` restores the display.

        A non-positive pin is rejected rather than silently substituted: it
        has no meaning as a pace, and the interval maths divides by it.  The
        old divide-site `or 30.0` guard hid exactly this mistake behind a
        number nobody chose.
        """

        if value is None:
            self._max_draws_per_second_override = None
            return
        rate = float(value)
        if rate <= 0.0:
            raise ValueError(f"max_draws_per_second must be positive, got {rate!r}")
        self._max_draws_per_second_override = rate

    def __init__(self, parent=None):
        super().__init__(parent)
        # Belt and braces: nothing in this recipe makes a sibling widget
        # native, but if anything else ever forces one (QWindowContainer
        # goes native inside scroll areas, for example), this attribute
        # keeps Qt from shattering the rest of the window into subsurfaces.
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setAttribute(QtCore.Qt.ApplicationAttribute.AA_DontCreateNativeWidgetSiblings, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._surface_window = _ScreenSurfaceWindow(self)
        container = QtWidgets.QWidget.createWindowContainer(self._surface_window, self)
        container.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        container.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        # Geometry is managed by hand in resizeEvent: a layout would give
        # this canvas a valid sizeHint derived from the embedded window's
        # default size, and the surrounding StackAll layout would inflate
        # the whole main window from it (observed: startup ballooned to
        # 1960x1680).  The old paint-less canvas had no hint; keep that.
        self._container = container
        container.setGeometry(self.rect())
        self._draw_callback = None
        self._draw_scheduled = False
        self._last_draw_started = float("-inf")
        self._context = None
        self._context_error: str = ""
        self._configured_format: str | None = None
        self._present_mode: str = ""
        self._present_platform: str = ""
        self._present_modes_available: tuple[str, ...] = ()
        self._max_draws_per_second_override: float | None = None
        self._cached_draws_per_second: float | None = None
        self._watched_screen = None
        self._resize_pending = False
        # Pacing lives here, so the timings do too: relating a slipped wakeup
        # to a late frame needs both halves under one owner.
        self.frame_timing = FrameTimingRecorder()
        # A move between monitors changes which display the pace must match.
        with contextlib.suppress(Exception):
            self._surface_window.screenChanged.connect(self._invalidate_pace_cache)

    # ---- draw scheduling (rendercanvas request_draw seam) -------------------

    def request_draw(self, callback=None) -> None:
        if callback is not None:
            self._draw_callback = callback
        if self._draw_scheduled or self._draw_callback is None:
            return
        self._draw_scheduled = True
        # No `or <n>` guard: the property already floors the rate at 1.0 via
        # fallback_draws_per_second.  The old `or 30.0` was a leftover from
        # the pinned-at-30 era and could only ever have masked a zero the
        # property cannot return.
        interval = 1000.0 / float(self.max_draws_per_second)
        now = perf_counter()
        elapsed_ms = (now - self._last_draw_started) * 1000.0
        delay_ms = 0 if elapsed_ms >= interval else round(interval - elapsed_ms)
        # Record the wakeup this pacer INTENDED, so the gap against the one
        # it got is attributable to the platform rather than to us.
        self.frame_timing.note_scheduled(now + delay_ms / 1000.0)
        QtCore.QTimer.singleShot(delay_ms, self, self._invoke_draw)

    def _invoke_draw(self) -> None:
        self._draw_scheduled = False
        callback = self._draw_callback
        if callback is None:
            return
        # No visibility bail: a draw that runs hidden must still reach the
        # callback, because the view's draw acknowledgement lives inside it
        # ("the ack still publishes so the presentation gate can never stay
        # armed forever on a hidden or headless surface") and the view
        # coalesces its rerequests against that ack.  Dropping the draw here
        # left no wakeup armed anywhere — the requested>drawn latch of
        # 2026-07-31, ground rule 11's forbidden "a later show/expose will
        # pick it up".  Pixels while hidden are ensure_context's refusal to
        # make, not this pacer's.
        self._apply_pending_resize()
        self._last_draw_started = perf_counter()
        self.frame_timing.note_draw_started(self._last_draw_started)
        callback()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.request_draw()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._container.setGeometry(self.rect())

    def _on_surface_resized(self) -> None:
        """Fold this resize into the next paced frame.

        A compositor delivers resize events during an interactive drag far
        faster than the display refreshes — and Qt emits several per drag
        step.  Reconfiguring the swapchain and presenting synchronously on
        each one measured 249 resize events -> 736 draws (184 fps against a
        60 Hz pace) with 2380 ms of a 4008 ms drag spent *inside* the resize
        events themselves: the drag could not keep up with the pointer, which
        is the sluggishness.

        Coalescing onto the normal paced path fixes that (428 events -> 186
        draws, 4.0 ms p50 inside the event), and
        ``test_resize_coalesces_into_one_paced_reconfigure`` pins it.

        KNOWN TRADE, not yet adjudicated on pixels: the synchronous draw was
        added to stop stale-buffer spill — a subsurface is NOT clipped by its
        parent, so while shrinking, an old larger buffer hangs outside the
        window until a correctly-sized one is presented.  Coalescing widens
        that window from ~0 to a measured 16.8 ms mean / 29.0 ms max, i.e.
        one to two compositor frames.  Whether that is visible has NOT been
        settled with a bitmap diff on a windowed, decorated compositor.  A
        conservative variant that keeps the shrink direction synchronous was
        measured too — spill 6.3 ms mean / 17.1 ms max, but 551 draws instead
        of 186, so it gives back most of the responsiveness win.

        That trade is WAYLAND-ONLY, measured 2026-07-31: driven 300 px past
        its top-level, this canvas put 312,000 pixels outside the window on
        Wayland and exactly zero on X11 (the X server clips a child window to
        its parent).  So on an X session the same coalescing costs only the
        flicker half — old-size content for 22.7 ms mean / 33.3 ms max
        against Wayland's 8.2 / 15.1 in the same sweep — and cannot spill.
        The handling stays shared: the conservative variant would trade the
        same responsiveness for a smaller flicker window on both.
        """

        self._resize_pending = True
        if self._context is None:
            # No swapchain yet: this is the show/expose-time resize that
            # establishes the surface size, not an interactive drag.  Keyed
            # on the context rather than the draw clock because draws now
            # run (and stamp the clock) on hidden canvases without ever
            # configuring a chain — the context is the truth about whether
            # the establish already happened.
            # Configuring the swapchain and taking the first (cold) acquire
            # here keeps that one-off ~15 ms cost off the first *content*
            # frame — deferring it measured 16-26 ms on the commit that
            # follows, against 0.9 ms warm.  There is no pacing concern
            # before the first frame, and nothing to coalesce with.
            self._invoke_draw()
            return
        self.request_draw()

    def _apply_pending_resize(self) -> None:
        """Reconfigure the swapchain for the latest size, once, at draw time.

        ``set_physical_size`` marks the chain for reconfiguration, so calling
        it per event rebuilt the image chain repeatedly mid-drag.  Every
        resize event since the last frame collapses into this one call.

        A hidden canvas leaves the resize armed rather than reconfiguring
        against a size no compositor is showing.  Unlike the draw itself
        (whose acknowledgement nothing else can publish), a deferred resize
        HAS a standing owner: ``showEvent`` requests a draw unconditionally,
        and that draw applies it before the first visible frame.
        """

        if not self._resize_pending or not self.isVisible():
            return
        self._resize_pending = False
        if self._context is not None:
            self._context.set_physical_size(*self.get_physical_size())

    # ---- physical geometry ---------------------------------------------------

    def get_physical_size(self) -> tuple[int, int]:
        window = self._surface_window
        ratio = float(window.devicePixelRatio() or 1.0)
        return (
            max(0, round(window.width() * ratio)),
            max(0, round(window.height() * ratio)),
        )

    # ---- swapchain context ---------------------------------------------------

    def ensure_context(self, device):
        """Lazily create + configure the swapchain context, or return ``None``.

        ``None`` means "not yet": the embedded window is not created/exposed
        or has zero pixels.  A show/resize/expose event schedules another
        draw, so returning ``None`` never strands the frame.  A hard failure
        is recorded in ``context_error`` and re-raised so the view's
        draw-error diagnostics surface it.
        """

        if self._context is not None:
            # A cached swapchain is only handed out while the widget can
            # present: draws now run on hidden canvases (so their ack can
            # publish), and a Fifo acquire against an unmapped surface can
            # block the GUI thread until the compositor feels like mapping
            # it again.
            return self._context if self.isVisible() else None
        if not self.isVisible() or not self._surface_window.isExposed():
            return None
        width, height = self.get_physical_size()
        if width < 1 or height < 1:
            return None
        # The platform window (and its wl_surface / X11 Window) only exists
        # once winId() is first resolved.
        winid = int(self._surface_window.winId())
        target, reason = _resolve_native_present_target()
        if target is None:
            # setupUI resolved the same question before building this canvas,
            # so reaching here means the platform changed underneath us.
            self._context_error = f"no native present target: {reason}"
            raise RuntimeError(self._context_error)
        try:
            from wgpu.backends.wgpu_native._api import GPUCanvasContext

            context = GPUCanvasContext(
                {
                    "method": "screen",
                    "platform": target.platform,
                    # What winId() means is the per-platform contract the
                    # ring-4 pins cover: under the wayland QPA it is this
                    # embedded window's wl_surface* (undocumented), under xcb
                    # it is its X11 Window XID (documented, long-standing).
                    "window": winid,
                    "display": target.display,
                    "vsync": True,
                }
            )
            context.set_physical_size(width, height)
            fmt = str(context.get_preferred_format(device.adapter))
            fmt = fmt.removesuffix("-srgb")
            context.configure(device=device, format=fmt)
            self._present_modes_available = tuple(
                context._get_capabilities(device.adapter).get("present_modes", ())
            )
            self._present_mode = self._prefer_mailbox(context)
        except Exception as exc:
            self._context_error = f"{type(exc).__name__}: {exc}"
            raise
        self._context = context
        self._configured_format = fmt
        self._present_platform = target.platform
        # Timings taken before the chain existed (or under a different
        # present mode) do not describe this swapchain; start clean.
        self.frame_timing.reset()
        emit_trace(
            "wgpu_screen_swapchain_configured",
            format=fmt,
            # Which windowing system the surface was built on: a screen-path
            # timing that cannot be attributed to a platform is unusable, and
            # both now exist in the field.
            platform=self._present_platform,
            present_mode=self._present_mode,
            present_modes=self._present_modes_available,
            physical_size=(width, height),
        )
        return context

    def _prefer_mailbox(self, context) -> str:
        """Re-configure for Mailbox when available; report the mode in use.

        wgpu-py's vsync heuristic always picks Fifo when present, and Fifo
        acquire blocks the GUI thread ~15 ms/frame (gate-B tier 1).  Mailbox
        keeps vsync pacing without the blocking acquire.  The override
        touches wgpu-py private config state; any API drift downgrades to
        Fifo loudly instead of failing the frame.
        """

        if "mailbox" not in self._present_modes_available:
            return "fifo"
        try:
            from wgpu.backends.wgpu_native._ffi import lib

            config = context._wgpu_config
            config.presentMode = lib.WGPUPresentMode_Mailbox
            context._configure_screen_real()
            return "mailbox"
        except Exception as exc:
            emit_trace(
                "wgpu_screen_mailbox_unavailable",
                error=f"{type(exc).__name__}: {exc}",
            )
            return "fifo"

    @property
    def configured_format(self) -> str | None:
        return self._configured_format

    @property
    def context_error(self) -> str:
        return self._context_error

    @property
    def present_mode(self) -> str:
        return self._present_mode

    @property
    def present_platform(self) -> str:
        """``"wayland"``/``"x11"`` once configured, ``""`` before that."""

        return self._present_platform

    @property
    def present_modes_available(self) -> tuple[str, ...]:
        return self._present_modes_available

    def frame_timing_snapshot(self) -> dict[str, object]:
        """Frame cadence against the period this canvas is pacing to.

        The period comes from the same property the pacer uses, so the phase
        statistics are always read against the rate actually being targeted
        — including a benchmark's pinned override, where measuring phase
        against the physical display instead would be meaningless.
        """

        return self.frame_timing.snapshot(period_ms=1000.0 / float(self.max_draws_per_second))

    def teardown(self) -> None:
        context, self._context = self._context, None
        self._draw_callback = None
        if context is not None:
            with contextlib.suppress(Exception):
                context.unconfigure()
            # Release the wgpu surface NOW, while Qt's wl_display connection
            # is still alive.  Leaving it to interpreter-exit GC releases the
            # surface after the display connection is gone and dumps core.
            with contextlib.suppress(Exception):
                context._release()
