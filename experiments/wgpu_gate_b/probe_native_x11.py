"""Tier-0 probe: the same native-swapchain question as the Wayland probe, on X11.

``probe_native_wayland.py`` established that a wgpu surface can be built on
Qt's own native Wayland window from pure Python.  This is its X sibling, and
it exists because the two ingredients are *different objects* under the xcb
QPA even though the recipe reads the same:

  1. ``Display*``: ``QNativeInterface.QX11Application.display()``.  A wgpu
     Xlib surface must be built on the SAME display connection that owns the
     window, exactly as on Wayland — the X server would accept a second
     connection, but the window ID would then be looked up on a connection
     that never receives Qt's geometry/expose traffic.
  2. ``Window`` (XID): ``QWidget.winId()`` under the xcb QPA.  Unlike the
     Wayland ``winId() == wl_surface*`` behaviour this pins, the X11 identity
     is a documented, long-standing Qt contract — this probe pins it anyway,
     because the *pairing* with wgpu's ``WGPUSurfaceSourceXlibWindow`` is
     what the screen path depends on, not either half alone.

Success criterion is identical to the Wayland probe: capabilities return
formats, and configure + get-current-texture + clear + present complete with
SUCCESS status for N frames, on a native child widget, a top-level, and the
production ``createWindowContainer`` shape.

One X11-specific fact this probe also records, because it decides whether
the Wayland recipe's *reasoning* transfers: ``present_modes``.  Mailbox is
what keeps the Fifo acquire off the GUI thread (gate-B tier-1 caveat), and
its availability on X11 is a driver/WSI property rather than a compositor
one — so it is observed here rather than assumed from the Wayland run.

The other X11 difference that matters, child windows being server-clipped to
their parent (a Wayland subsurface is not), is deliberately NOT asserted
here: any check this probe could make would be Qt's own arithmetic agreeing
with itself.  It is settled on captured pixels instead — see the resize arm
of ``x11_screen_evidence.py``.

Run:  DISPLAY=:N QT_QPA_PLATFORM=xcb python probe_native_x11.py out.json
"""

import json
import sys
import time

import wgpu
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QNativeInterface
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from wgpu.backends.wgpu_native import _helpers
from wgpu.backends.wgpu_native._ffi import ffi, lib
from wgpu.backends.wgpu_native.extras import set_instance_extras

# Same instance masking as the Wayland probe: ArrayScope pins Vulkan on
# Linux, and letting the GL backend initialize alongside it has already cost
# one fatal protocol error on the Wayland side.  Keeping the two probes on
# one backend also keeps their numbers comparable.
set_instance_extras(backends=["Vulkan"])

RESULT = {"probe": "native-x11-surface", "qt_platform": None, "cases": {}}


def _create_surface(display_ptr, window_xid):
    # "x11" selects WGPUSurfaceSourceXlibWindow in wgpu-py 0.31.x (it is also
    # the default platform value on Linux).  The "xcb" variant exists but is
    # upstream-marked untested, so the Xlib source is the one we build on.
    return _helpers.get_surface_id_from_info(
        {"platform": "x11", "window": int(window_xid), "display": int(display_ptr)}
    )


def _present_mode_names(caps, count):
    names = {
        int(lib.WGPUPresentMode_Fifo): "fifo",
        int(lib.WGPUPresentMode_FifoRelaxed): "fiforelaxed",
        int(lib.WGPUPresentMode_Immediate): "immediate",
        int(lib.WGPUPresentMode_Mailbox): "mailbox",
    }
    return [
        names.get(int(caps.presentModes[i]), str(int(caps.presentModes[i]))) for i in range(count)
    ]


def _drive_surface(surface_id, adapter, device, width, height, frames=30):
    """Configure, render `frames` clears, present.  Returns dict of evidence."""
    out = {"formats": [], "frames_presented": 0, "statuses": {}}

    caps = ffi.new("WGPUSurfaceCapabilities *")
    status = lib.wgpuSurfaceGetCapabilities(surface_id, adapter._internal, caps)
    out["get_capabilities_status"] = int(status)
    out["formats"] = [int(caps.formats[i]) for i in range(caps.formatCount)]
    out["present_modes"] = _present_mode_names(caps, caps.presentModeCount)
    if not out["formats"]:
        return out
    fmt = out["formats"][0]

    config = ffi.new("WGPUSurfaceConfiguration *")
    config.device = device._internal
    config.format = fmt
    config.usage = lib.WGPUTextureUsage_RenderAttachment
    config.width = max(8, width)
    config.height = max(8, height)
    config.presentMode = lib.WGPUPresentMode_Fifo
    config.alphaMode = caps.alphaModes[0] if caps.alphaModeCount else 0
    lib.wgpuSurfaceConfigure(surface_id, config)

    acquire_ms = []
    present_ms = []
    for frame in range(frames):
        st = ffi.new("WGPUSurfaceTexture *")
        started = time.perf_counter()
        lib.wgpuSurfaceGetCurrentTexture(surface_id, st)
        acquire_ms.append((time.perf_counter() - started) * 1000.0)
        status = int(st.status)
        out["statuses"][status] = out["statuses"].get(status, 0) + 1
        if status not in (
            lib.WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal,
            lib.WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal,
        ):
            break
        view = lib.wgpuTextureCreateView(st.texture, ffi.NULL)
        enc = lib.wgpuDeviceCreateCommandEncoder(device._internal, ffi.NULL)
        color = ffi.new("WGPURenderPassColorAttachment *")
        color.view = view
        color.loadOp = lib.WGPULoadOp_Clear
        color.storeOp = lib.WGPUStoreOp_Store
        t = frame / max(1, frames - 1)
        color.clearValue.r, color.clearValue.g = 0.1 + 0.8 * t, 0.4
        color.clearValue.b, color.clearValue.a = 0.9 - 0.8 * t, 1.0
        color.depthSlice = lib.WGPU_DEPTH_SLICE_UNDEFINED
        rp_desc = ffi.new("WGPURenderPassDescriptor *")
        rp_desc.colorAttachmentCount = 1
        rp_desc.colorAttachments = color
        rp = lib.wgpuCommandEncoderBeginRenderPass(enc, rp_desc)
        lib.wgpuRenderPassEncoderEnd(rp)
        lib.wgpuRenderPassEncoderRelease(rp)
        cb = lib.wgpuCommandEncoderFinish(enc, ffi.NULL)
        lib.wgpuCommandEncoderRelease(enc)
        arr = ffi.new("WGPUCommandBuffer[]", [cb])
        lib.wgpuQueueSubmit(device.queue._internal, 1, arr)
        lib.wgpuCommandBufferRelease(cb)
        lib.wgpuTextureViewRelease(view)
        started = time.perf_counter()
        present_status = lib.wgpuSurfacePresent(surface_id)
        present_ms.append((time.perf_counter() - started) * 1000.0)
        out.setdefault("present_statuses", {})
        out["present_statuses"][int(present_status)] = (
            out["present_statuses"].get(int(present_status), 0) + 1
        )
        lib.wgpuTextureRelease(st.texture)
        out["frames_presented"] += 1
        QApplication.processEvents()
        time.sleep(0.016)
    out["acquire_ms_p50"] = (
        round(sorted(acquire_ms)[len(acquire_ms) // 2], 3) if acquire_ms else None
    )
    out["present_ms_p50"] = (
        round(sorted(present_ms)[len(present_ms) // 2], 3) if present_ms else None
    )
    return out


class PaintlessWidget(QWidget):
    """A widget Qt never paints into, so the swapchain owns its pixels alone.

    On Wayland this is load-bearing for protocol survival (Qt's SHM buffer on
    a dmabuf explicit-sync surface is a fatal error).  On X11 the failure is
    milder — Qt's expose repaints would simply race the swapchain's presents
    into the same drawable and flicker — but the widget shape is kept
    identical so the two probes measure one recipe, not two.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_PaintOnScreen, True)  # implies WA_NativeWindow
        self.setAttribute(Qt.WA_NoSystemBackground, True)

    def paintEngine(self):
        return None


def main():
    app = QApplication(sys.argv)
    RESULT["qt_platform"] = app.platformName()
    assert RESULT["qt_platform"] == "xcb", "run with QT_QPA_PLATFORM=xcb"

    iface = app.nativeInterface()
    assert isinstance(iface, QNativeInterface.QX11Application)
    display_ptr = int(iface.display())
    RESULT["x_display"] = hex(display_ptr)

    adapter = wgpu.gpu.request_adapter_sync(power_preference="low-power")
    device = adapter.request_device_sync()
    RESULT["adapter"] = adapter.info["device"]

    # Case 1: paint-less native child inside a normal layout.  On X11 this is
    # an ordinary child X window — server-clipped to its parent, which is the
    # structural difference from the Wayland subsurface.
    host = QWidget()
    host.setWindowTitle("wgpu native-x11 probe (native child)")
    layout = QVBoxLayout(host)
    layout.addWidget(QLabel("Qt label above the canvas"))
    child = PaintlessWidget()
    child.setMinimumSize(320, 200)
    layout.addWidget(child)
    host.resize(400, 300)
    host.show()
    app.processEvents()
    child_winid = int(child.winId())
    RESULT["cases"]["native_child"] = {"winid": hex(child_winid)}
    surface_id = _create_surface(display_ptr, child_winid)
    RESULT["cases"]["native_child"].update(
        _drive_surface(surface_id, adapter, device, child.width(), child.height())
    )
    lib.wgpuSurfaceRelease(surface_id)

    # Case 2: paint-less top-level window (control case).
    top = PaintlessWidget()
    top.setWindowTitle("wgpu native-x11 probe (top-level)")
    top.resize(320, 240)
    top.show()
    deadline = time.time() + 5.0
    while time.time() < deadline and not (
        top.windowHandle() is not None and top.windowHandle().isExposed()
    ):
        app.processEvents()
        time.sleep(0.01)
    winid = int(top.winId())
    RESULT["cases"]["top_level"] = {"winid": hex(winid)}
    surface_id = _create_surface(display_ptr, winid)
    RESULT["cases"]["top_level"].update(
        _drive_surface(surface_id, adapter, device, top.width(), top.height())
    )
    lib.wgpuSurfaceRelease(surface_id)

    # Case 3: bare QWindow embedded through QWidget.createWindowContainer —
    # the production shape (display/backends/wgpu/screen_canvas.py).
    from PySide6.QtGui import QSurface, QWindow

    holder = QWidget()
    holder.setWindowTitle("wgpu native-x11 probe (window container)")
    holder_layout = QVBoxLayout(holder)
    holder_layout.addWidget(QLabel("Qt label above the canvas"))
    embedded = QWindow()
    embedded.setSurfaceType(QSurface.SurfaceType.RasterSurface)
    container = QWidget.createWindowContainer(embedded, holder)
    container.setMinimumSize(320, 200)
    holder_layout.addWidget(container)
    holder.resize(400, 300)
    holder.show()
    deadline = time.time() + 5.0
    while time.time() < deadline and not embedded.isExposed():
        app.processEvents()
        time.sleep(0.01)
    embedded_winid = int(embedded.winId())
    RESULT["cases"]["window_container"] = {"winid": hex(embedded_winid)}
    surface_id = _create_surface(display_ptr, embedded_winid)
    RESULT["cases"]["window_container"].update(
        _drive_surface(surface_id, adapter, device, embedded.width(), embedded.height())
    )

    lib.wgpuSurfaceRelease(surface_id)

    print(json.dumps(RESULT, indent=2))
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump(RESULT, f, indent=2)
    QTimer.singleShot(0, app.quit)
    app.exec()


if __name__ == "__main__":
    main()
