"""Tier-0 probe: can a wgpu surface be created on Qt's NATIVE Wayland window
from pure Python (no C++ shim)?

Background: rendercanvas's Qt backend hard-disables its Wayland branch
(`if False:` in `_get_surface_ids`) and force-sets `QT_QPA_PLATFORM=xcb` at
import time on Wayland systems, so its `present_method="screen"` always goes
through XWayland.  Its own comment says the blocker is obtaining the real
`wl_display` used by Qt (their alt `wl_display_connect()` segfaults, because
a wl_surface is only valid on the connection that created it).

This probe tests the two ingredients directly:
  1. `wl_display`: PySide6 6.11 binds `QNativeInterface.QWaylandApplication`
     whose `.display()` returns the REAL in-process display pointer.
  2. `wl_surface`: candidate = `QWidget.winId()` under the wayland platform
     (pointer-sized on this Qt; if it is the wl_surface, surface creation,
     configure, and present must succeed).

Success criterion: wgpuSurfaceGetCapabilities returns formats, configure +
get-current-texture + clear-render + present complete with SUCCESS status
for N frames, on both a top-level window and a NATIVE CHILD widget inside a
layout (the ArrayScope case; on Wayland a native child is a wl_subsurface).

Run:  QT_QPA_PLATFORM=wayland python probe_native_wayland.py out.json
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

# The instance's GL backend re-initializes EGL on Qt's wl_display when a
# surface is created ("Re-initializing Gles context due to Wayland window"),
# which triggered a fatal wp_linux_drm_syncobj explicit-sync protocol error
# on this compositor.  Masking the instance to Vulkan avoids the EGL path
# entirely; ArrayScope wants Vulkan on Linux anyway.
set_instance_extras(backends=["Vulkan"])

RESULT = {"probe": "native-wayland-surface", "qt_platform": None, "cases": {}}


def _create_surface(display_ptr, surface_ptr):
    return _helpers.get_surface_id_from_info(
        {"platform": "wayland", "window": int(surface_ptr), "display": int(display_ptr)}
    )


def _drive_surface(surface_id, adapter, device, width, height, frames=30):
    """Configure, render `frames` clears, present.  Returns dict of evidence."""
    out = {"formats": [], "frames_presented": 0, "statuses": {}}

    caps = ffi.new("WGPUSurfaceCapabilities *")
    status = lib.wgpuSurfaceGetCapabilities(surface_id, adapter._internal, caps)
    out["get_capabilities_status"] = int(status)
    out["formats"] = [int(caps.formats[i]) for i in range(caps.formatCount)]
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

    for frame in range(frames):
        st = ffi.new("WGPUSurfaceTexture *")
        lib.wgpuSurfaceGetCurrentTexture(surface_id, st)
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
        present_status = lib.wgpuSurfacePresent(surface_id)
        out.setdefault("present_statuses", {})
        out["present_statuses"][int(present_status)] = (
            out["present_statuses"].get(int(present_status), 0) + 1
        )
        lib.wgpuTextureRelease(st.texture)
        out["frames_presented"] += 1
        QApplication.processEvents()
        time.sleep(0.016)
    return out


class PaintlessWidget(QWidget):
    """A widget Qt never paints into: its wl_surface belongs to wgpu alone.

    Without this, Qt commits SHM backing-store buffers to the SAME wl_surface
    that the Vulkan swapchain drives, and this compositor kills the
    connection with `wp_linux_drm_syncobj_surface_v1 error 2: Explicit Sync
    only supported on dmabuf buffers`.  This mirrors what rendercanvas's
    screen mode does (WA_PaintOnScreen + null paintEngine).
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
    assert RESULT["qt_platform"] == "wayland", "run with QT_QPA_PLATFORM=wayland"

    iface = app.nativeInterface()
    assert isinstance(iface, QNativeInterface.QWaylandApplication)
    display_ptr = int(iface.display())
    RESULT["wl_display"] = hex(display_ptr)

    adapter = wgpu.gpu.request_adapter_sync(power_preference="low-power")
    device = adapter.request_device_sync()
    RESULT["adapter"] = adapter.info["device"]

    # Case 1: paint-less native child inside a normal layout — the
    # ArrayScope shape (viewer widget docked in a larger Qt window; on
    # Wayland the native child becomes a wl_subsurface).
    host = QWidget()
    host.setWindowTitle("wgpu native-wayland probe (native child)")
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
    top.setWindowTitle("wgpu native-wayland probe (top-level)")
    top.resize(320, 240)
    top.show()
    # Present only after the initial xdg_surface configure/ack: committing a
    # buffer to an unconfigured toplevel is a fatal protocol error on strict
    # compositors (weston; GNOME happened to win the race, 2026-07-20).
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
    # the production shape since the 2026-07-19 subsurface-soup fix
    # (display/backends/wgpu/screen_canvas.py).  Unlike a native child
    # WIDGET, the embedded window is parented directly to the top-level
    # window: exactly one wl_subsurface, no native ancestor chain, and Qt
    # never attaches a backing-store buffer to it.
    from PySide6.QtGui import QSurface, QWindow

    holder = QWidget()
    holder.setWindowTitle("wgpu native-wayland probe (window container)")
    holder_layout = QVBoxLayout(holder)
    holder_layout.addWidget(QLabel("Qt label above the canvas"))
    embedded = QWindow()
    embedded.setSurfaceType(QSurface.SurfaceType.RasterSurface)
    container = QWidget.createWindowContainer(embedded, holder)
    container.setMinimumSize(320, 200)
    holder_layout.addWidget(container)
    holder.resize(400, 300)
    holder.show()
    # Mirror production (ensure_context waits for exposure): presenting to
    # the embedded window before the holder's xdg_surface got its initial
    # configure is a fatal protocol error on strict compositors (weston).
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
