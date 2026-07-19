"""Paint-less native child driving its own wgpu swapchain (gate-B recipe).

The bitmap present path pays a 4-7 ms/frame GPU->CPU readback through
rendercanvas.  Gate B measured the native-Wayland screen path at ~0.6 ms
encode+submit / 0.08 ms present (docs/proposals/wgpu-renderer-experiment.md
Tier 1; probe `experiments/wgpu_gate_b/probe_native_wayland.py`).  This
module productionizes that probe:

* the widget is a NATIVE child Qt never paints into (``WA_PaintOnScreen`` +
  null ``paintEngine``): its ``wl_surface`` belongs to wgpu alone.  Qt
  committing SHM backing-store buffers to the same surface is a fatal
  compositor protocol error (explicit-sync dmabuf-only).
* the wgpu surface is created from the REAL in-process ``wl_display``
  (``QNativeInterface.QWaylandApplication.display()``) plus
  ``QWidget.winId()`` as the ``wl_surface*``.  Both are undocumented Qt
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

from arrayscope.app.qt_binding import prefer_pyside6
from arrayscope.core.trace import emit_trace

prefer_pyside6()

from pyqtgraph.Qt import QtCore, QtWidgets


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


class WgpuScreenCanvas(QtWidgets.QWidget):
    """A widget Qt never paints into: its wl_surface belongs to wgpu alone.

    Presents rendercanvas's minimal canvas API surface (``request_draw`` /
    ``get_physical_size`` / ``devicePixelRatio``) so ``WgpuImageView2D``
    drives both present methods through one seam.  Draw scheduling is a
    coalesced zero-timer: Qt cannot deliver paint work for a paint-less
    widget, so the canvas owns its own on-demand cadence exactly like
    rendercanvas's ``ondemand`` update mode.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_PaintOnScreen, True)  # implies WA_NativeWindow
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._draw_callback = None
        self._draw_scheduled = False
        self._context = None
        self._context_error: str = ""
        self._configured_format: str | None = None
        self._present_mode: str = ""
        self._present_modes_available: tuple[str, ...] = ()

    # Qt must never attach a paint engine to this window: a single SHM
    # backing-store commit onto the swapchain's wl_surface kills the
    # compositor connection (explicit-sync protocol error, gate-B tier 0).
    def paintEngine(self):
        return None

    # ---- draw scheduling (rendercanvas request_draw seam) -------------------

    def request_draw(self, callback=None) -> None:
        if callback is not None:
            self._draw_callback = callback
        if self._draw_scheduled or self._draw_callback is None:
            return
        self._draw_scheduled = True
        QtCore.QTimer.singleShot(0, self._invoke_draw)

    def _invoke_draw(self) -> None:
        self._draw_scheduled = False
        callback = self._draw_callback
        if callback is None or not self.isVisible():
            return
        callback()

    def paintEvent(self, event) -> None:  # expose/damage -> redraw, never paint
        self.request_draw()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.request_draw()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._context is not None:
            self._context.set_physical_size(*self.get_physical_size())
        self.request_draw()

    # ---- physical geometry ---------------------------------------------------

    def get_physical_size(self) -> tuple[int, int]:
        ratio = float(self.devicePixelRatio() or 1.0)
        return (
            max(0, int(round(self.width() * ratio))),
            max(0, int(round(self.height() * ratio))),
        )

    # ---- swapchain context ---------------------------------------------------

    def ensure_context(self, device):
        """Lazily create + configure the swapchain context, or return ``None``.

        ``None`` means "not yet": the native window is not created/exposed or
        has zero pixels.  A show/resize/paint event schedules another draw,
        so returning ``None`` never strands the frame.  A hard failure is
        recorded in ``context_error`` and re-raised so the view's draw-error
        diagnostics surface it.
        """

        if self._context is not None:
            return self._context
        if not self.isVisible():
            return None
        width, height = self.get_physical_size()
        if width < 1 or height < 1:
            return None
        # Force native-window creation: WA_PaintOnScreen implies
        # WA_NativeWindow, but the QWindow (and its wl_surface) only exists
        # once winId() is first resolved.
        winid = int(self.winId())
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
                    # QPA winId() IS the wl_surface* of this native child.
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
            try:
                context.unconfigure()
            except Exception:
                pass
            # Release the wgpu surface NOW, while Qt's wl_display connection
            # is still alive.  Leaving it to interpreter-exit GC releases the
            # surface after the display connection is gone and dumps core.
            try:
                context._release()
            except Exception:
                pass
