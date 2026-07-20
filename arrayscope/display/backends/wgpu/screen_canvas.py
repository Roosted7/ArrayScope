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

The screen path exists only on a live Wayland session; every other
environment (offscreen, xcb, missing native interface) reports a loud
fallback reason and the view keeps the bitmap path.
"""

from __future__ import annotations

from time import perf_counter

from arrayscope.app.qt_binding import prefer_pyside6
from arrayscope.core.trace import emit_trace

prefer_pyside6()

import contextlib

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


def screen_present_unavailable_reason() -> str | None:
    """Why ``present_method="screen"`` cannot exist here, or ``None`` if it can.

    Mirrors the tier-0 probe's preconditions exactly: a live QApplication on
    the ``wayland`` QPA with a working ``QWaylandApplication`` native
    interface.  Checked BEFORE any widget is created so the caller can fall
    back to the bitmap canvas without leaving a half-built native child.
    """

    app = QtWidgets.QApplication.instance()
    if app is None:
        return "no QApplication"
    platform = str(app.platformName())
    if platform != "wayland":
        return f"Qt platform {platform!r} is not wayland"
    try:
        from PySide6.QtGui import QNativeInterface
    except Exception as exc:  # pragma: no cover - PySide6 is the pinned binding
        return f"QNativeInterface unavailable ({exc})"
    iface = app.nativeInterface()
    if not isinstance(iface, QNativeInterface.QWaylandApplication):
        return "no QWaylandApplication native interface"
    if not int(iface.display() or 0):
        return "QWaylandApplication reports a null wl_display"
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
        # subsurfaces are not clipped to the parent).  This event fires when
        # the embedded window's real geometry changed, so reconfigure and
        # present at the new size NOW, bypassing the draw-rate cap.
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

    #: Draw-rate cap, mirroring rendercanvas's ``max_fps=30`` default that the
    #: bitmap canvas runs under.  Without a cap, a 60 fps camera glide requests
    #: a present per input event, exhausts the mailbox swapchain, and every
    #: subsequent acquire blocks the GUI thread — measured as consistently
    #: ~1.5-2x worse zoompan event-loop gaps than bitmap (2026-07-19 paired
    #: controls) until this pacing landed.
    max_draws_per_second = 30.0

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
        self._present_modes_available: tuple[str, ...] = ()

    # ---- draw scheduling (rendercanvas request_draw seam) -------------------

    def request_draw(self, callback=None) -> None:
        if callback is not None:
            self._draw_callback = callback
        if self._draw_scheduled or self._draw_callback is None:
            return
        self._draw_scheduled = True
        interval = 1000.0 / float(self.max_draws_per_second or 30.0)
        elapsed_ms = (perf_counter() - self._last_draw_started) * 1000.0
        delay_ms = 0 if elapsed_ms >= interval else round(interval - elapsed_ms)
        QtCore.QTimer.singleShot(delay_ms, self, self._invoke_draw)

    def _invoke_draw(self) -> None:
        self._draw_scheduled = False
        callback = self._draw_callback
        if callback is None or not self.isVisible():
            return
        self._last_draw_started = perf_counter()
        callback()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.request_draw()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._container.setGeometry(self.rect())

    def _on_surface_resized(self) -> None:
        if self._context is not None:
            self._context.set_physical_size(*self.get_physical_size())
        self._invoke_draw()

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
            return self._context
        if not self.isVisible() or not self._surface_window.isExposed():
            return None
        width, height = self.get_physical_size()
        if width < 1 or height < 1:
            return None
        # The platform window (and its wl_surface) only exists once winId()
        # is first resolved.
        winid = int(self._surface_window.winId())
        app = QtWidgets.QApplication.instance()
        from PySide6.QtGui import QNativeInterface

        iface = app.nativeInterface()
        assert isinstance(iface, QNativeInterface.QWaylandApplication)
        try:
            from wgpu.backends.wgpu_native._api import GPUCanvasContext

            context = GPUCanvasContext(
                {
                    "method": "screen",
                    "platform": "wayland",
                    # Qt contract pinned by the ring-4 test: under the wayland
                    # QPA winId() IS the wl_surface* of this embedded window.
                    "window": winid,
                    "display": int(iface.display()),
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
        emit_trace(
            "wgpu_screen_swapchain_configured",
            format=fmt,
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
    def present_modes_available(self) -> tuple[str, ...]:
        return self._present_modes_available

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
