"""Tier-1 harness: Qt presentation x overlay gate for the wgpu experiment.

Runs ONE (mode, session) cell per process and writes evidence JSON:

    QT_QPA_PLATFORM=wayland python run_gate_b.py --mode bitmap        --out ...
    QT_QPA_PLATFORM=wayland python run_gate_b.py --mode screen-native --out ...
    QT_QPA_PLATFORM=xcb     python run_gate_b.py --mode bitmap        --out ...
    QT_QPA_PLATFORM=xcb     python run_gate_b.py --mode screen-stock  --out ...

Modes:
  bitmap        rendercanvas QRenderWidget, present_method="bitmap"
                (GPU render -> readback -> QImage -> QPainter).
  screen-stock  rendercanvas QRenderWidget, present_method="screen"
                (stock rendercanvas: X11 handles; xcb sessions only).
  screen-native our paint-less native child + raw wgpu Wayland surface
                (the Tier-0 "patched rendercanvas" path; wayland only).

Layout mirrors the ArrayScope shape used in Datoviz gate A: the canvas is
a fixed central tab (never floated); two dock panels float/re-dock around
it; Qt overlay labels (magenta, tile-truth stand-ins) sit OVER the canvas.

Journey phases (per-frame GUI-thread ms recorded for each):
  steady(120) -> dock_float(60) -> dock_redock(60) -> tab_away(30) ->
  tab_back(60) -> resize(60, cycling sizes) -> second_window(60) ->
  overlay capture -> (bitmap only) readback price microbench at
  1300x650 and 3840x2160.

Overlay oracle: count pure-magenta pixels over the canvas rect in
(a) the QWidget backing store (host.grab()) and (b) a compositor-side
capture: QScreen.grabWindow on xcb; on Wayland, `grim` or `spectacle -b`
if available (recorded as unavailable otherwise).
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

REAL_SIZE = (1300, 650)
STRESS_SIZE = (3840, 2160)
MAGENTA = (255, 0, 255)


def stats(ms_list):
    if not ms_list:
        return None
    s = sorted(ms_list)
    return {
        "n": len(s),
        "mean_ms": round(statistics.fmean(s), 3),
        "p50_ms": round(s[len(s) // 2], 3),
        "p95_ms": round(s[min(len(s) - 1, int(len(s) * 0.95))], 3),
        "max_ms": round(s[-1], 3),
    }


TRIANGLE_WGSL = """
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) color: vec3<f32> };
struct U { v: vec4<f32> };
@group(0) @binding(0) var<uniform> u: U;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VOut {
    var pts = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.6), vec2<f32>(-0.6, -0.6), vec2<f32>(0.6, -0.6));
    var cols = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.2, 0.1), vec3<f32>(0.1, 1.0, 0.2), vec3<f32>(0.2, 0.3, 1.0));
    var out: VOut;
    let a = u.v.x;
    let p = pts[i];
    out.pos = vec4<f32>(p.x * cos(a) - p.y * sin(a), p.x * sin(a) + p.y * cos(a), 0.0, 1.0);
    out.color = cols[i];
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"""


class Renderer:
    """Device + triangle pipeline, shared by every canvas in the process."""

    def __init__(self):
        import wgpu

        self.wgpu = wgpu
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="low-power")
        self.device = self.adapter.request_device_sync()
        self.shader = self.device.create_shader_module(code=TRIANGLE_WGSL)
        self.ubo = self.device.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self.bgl = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        self.bind_group = self.device.create_bind_group(
            layout=self.bgl,
            entries=[{"binding": 0, "resource": {"buffer": self.ubo, "offset": 0, "size": 16}}],
        )
        self.layout = self.device.create_pipeline_layout(bind_group_layouts=[self.bgl])
        self._pipelines = {}

    def pipeline(self, fmt):
        if fmt not in self._pipelines:
            self._pipelines[fmt] = self.device.create_render_pipeline(
                layout=self.layout,
                vertex={"module": self.shader, "entry_point": "vs_main"},
                primitive={"topology": "triangle-list"},
                fragment={
                    "module": self.shader,
                    "entry_point": "fs_main",
                    "targets": [{"format": fmt}],
                },
            )
        return self._pipelines[fmt]

    def draw_to_view(self, view, fmt, t):
        import struct

        self.device.queue.write_buffer(self.ubo, 0, struct.pack("4f", t, 0, 0, 0))
        enc = self.device.create_command_encoder()
        rp = enc.begin_render_pass(
            color_attachments=[
                {
                    "view": view,
                    "load_op": "clear",
                    "store_op": "store",
                    "clear_value": (0.05, 0.08, 0.12, 1.0),
                }
            ]
        )
        rp.set_pipeline(self.pipeline(fmt))
        rp.set_bind_group(0, self.bind_group)
        rp.draw(3)
        rp.end()
        self.device.queue.submit([enc.finish()])


class RendercanvasCanvas:
    """bitmap / screen-stock cells via rendercanvas QRenderWidget."""

    def __init__(self, renderer, parent, present_method):
        from rendercanvas.pyside6 import QRenderWidget

        self.renderer = renderer
        self.t = 0.0
        self.widget = QRenderWidget(
            parent=parent, present_method=present_method, update_mode="ondemand"
        )
        self.context = self.widget.get_context("wgpu")
        self.fmt = None
        self.widget.request_draw(self._draw)

    def _draw(self):
        if self.fmt is None:
            self.fmt = self.context.get_preferred_format(self.renderer.adapter)
            self.context.configure(device=self.renderer.device, format=self.fmt)
        tex = self.context.get_current_texture()
        self.renderer.draw_to_view(tex.create_view(), self.fmt, self.t)

    def frame(self, t):
        self.t = t
        self.widget.force_draw()
        return {}


class NativeWaylandCanvas:
    """screen-native cell: paint-less native child + raw Wayland surface."""

    def __init__(self, renderer, parent):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QNativeInterface
        from PySide6.QtWidgets import QApplication, QWidget
        from wgpu.backends.wgpu_native import _helpers
        from wgpu.backends.wgpu_native._ffi import ffi, lib

        self.renderer = renderer
        self.ffi, self.lib = ffi, lib

        class Paintless(QWidget):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setAttribute(Qt.WA_PaintOnScreen, True)
                self.setAttribute(Qt.WA_NoSystemBackground, True)

            def paintEngine(self):
                return None

        self.widget = Paintless(parent)
        self._configured_size = None
        app = QApplication.instance()
        iface = app.nativeInterface()
        assert isinstance(iface, QNativeInterface.QWaylandApplication)
        self._display_ptr = int(iface.display())
        self._helpers = _helpers
        self._surface_id = None
        self.statuses = {}
        self.errors = 0

    def _ensure_surface(self):
        if self._surface_id is None:
            winid = int(self.widget.winId())
            self._surface_id = self._helpers.get_surface_id_from_info(
                {"platform": "wayland", "window": winid, "display": self._display_ptr}
            )
        w = max(8, int(self.widget.width() * self.widget.devicePixelRatioF()))
        h = max(8, int(self.widget.height() * self.widget.devicePixelRatioF()))
        if self._configured_size != (w, h):
            ffi, lib = self.ffi, self.lib
            caps = ffi.new("WGPUSurfaceCapabilities *")
            lib.wgpuSurfaceGetCapabilities(self._surface_id, self.renderer.adapter._internal, caps)
            self._fmt_int = caps.formats[0]
            config = ffi.new("WGPUSurfaceConfiguration *")
            config.device = self.renderer.device._internal
            config.format = self._fmt_int
            config.usage = lib.WGPUTextureUsage_RenderAttachment
            config.width, config.height = w, h
            config.presentMode = lib.WGPUPresentMode_Fifo
            config.alphaMode = caps.alphaModes[0] if caps.alphaModeCount else 0
            lib.wgpuSurfaceConfigure(self._surface_id, config)
            self._configured_size = (w, h)

    def frame(self, t):
        import struct

        ffi, lib = self.ffi, self.lib
        self._ensure_surface()
        sub = {}
        t0 = time.perf_counter()
        st = ffi.new("WGPUSurfaceTexture *")
        lib.wgpuSurfaceGetCurrentTexture(self._surface_id, st)
        sub["acquire_ms"] = (time.perf_counter() - t0) * 1000
        status = int(st.status)
        self.statuses[status] = self.statuses.get(status, 0) + 1
        ok = status in (
            lib.WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal,
            lib.WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal,
        )
        if not ok:
            self.errors += 1
            self._configured_size = None  # force reconfigure next frame
            return sub
        # Render via the raw texture (mirror Renderer.draw_to_view at C level).
        t1 = time.perf_counter()
        dev = self.renderer.device
        dev.queue.write_buffer(self.renderer.ubo, 0, struct.pack("4f", t, 0, 0, 0))
        view = lib.wgpuTextureCreateView(st.texture, ffi.NULL)
        enc = lib.wgpuDeviceCreateCommandEncoder(dev._internal, ffi.NULL)
        color = ffi.new("WGPURenderPassColorAttachment *")
        color.view = view
        color.loadOp = lib.WGPULoadOp_Clear
        color.storeOp = lib.WGPUStoreOp_Store
        color.clearValue.r, color.clearValue.g, color.clearValue.b, color.clearValue.a = (
            0.05,
            0.08,
            0.12,
            1.0,
        )
        color.depthSlice = lib.WGPU_DEPTH_SLICE_UNDEFINED
        rp_desc = ffi.new("WGPURenderPassDescriptor *")
        rp_desc.colorAttachmentCount = 1
        rp_desc.colorAttachments = color
        rp = lib.wgpuCommandEncoderBeginRenderPass(enc, rp_desc)
        # Raw draw with the Python-side pipeline object's internal handle.
        fmt_name = _surface_format_name(self._fmt_int)
        pipeline = self.renderer.pipeline(fmt_name)
        lib.wgpuRenderPassEncoderSetPipeline(rp, pipeline._internal)
        bg = self.renderer.bind_group
        lib.wgpuRenderPassEncoderSetBindGroup(rp, 0, bg._internal, 0, ffi.NULL)
        lib.wgpuRenderPassEncoderDraw(rp, 3, 1, 0, 0)
        lib.wgpuRenderPassEncoderEnd(rp)
        lib.wgpuRenderPassEncoderRelease(rp)
        cb = lib.wgpuCommandEncoderFinish(enc, ffi.NULL)
        lib.wgpuCommandEncoderRelease(enc)
        arr = ffi.new("WGPUCommandBuffer[]", [cb])
        lib.wgpuQueueSubmit(dev.queue._internal, 1, arr)
        lib.wgpuCommandBufferRelease(cb)
        lib.wgpuTextureViewRelease(view)
        sub["encode_submit_ms"] = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        lib.wgpuSurfacePresent(self._surface_id)
        lib.wgpuTextureRelease(st.texture)
        sub["present_ms"] = (time.perf_counter() - t2) * 1000
        return sub


_FMT_NAMES = None


def _surface_format_name(fmt_int):
    """Map a WGPUTextureFormat int to the wgpu-py enum string."""
    global _FMT_NAMES
    if _FMT_NAMES is None:
        from wgpu.backends.wgpu_native._mappings import enummap

        _FMT_NAMES = {
            v: k.split(".")[-1] for k, v in enummap.items() if k.startswith("TextureFormat.")
        }
    return _FMT_NAMES[fmt_int]


def count_magenta(qimage, rect=None, tol=12):
    """Count near-magenta pixels (tolerant of capture color conversion)."""
    from PySide6.QtGui import QImage

    if qimage.isNull():
        return -1
    img = qimage.convertToFormat(QImage.Format_RGBA8888)
    if rect is not None:
        img = img.copy(rect)
    w, h = img.width(), img.height()
    buf = bytes(img.constBits())[: w * h * 4]
    count = 0
    r0, g0, b0 = MAGENTA
    for i in range(0, len(buf), 4):
        if abs(buf[i] - r0) <= tol and abs(buf[i + 1] - g0) <= tol and abs(buf[i + 2] - b0) <= tol:
            count += 1
    return count


def compositor_capture(session, host, out_png):
    """Best-effort compositor-side capture; returns (tool, ok)."""
    if session == "xcb":
        # Grab the WINDOW, not the root: rootless XWayland's root is black.
        # XGetImage on the window region includes native-child X windows'
        # on-screen pixels, so this is the compositor-side truth for xcb.
        from PySide6.QtGui import QGuiApplication

        screen = QGuiApplication.primaryScreen()
        pm = screen.grabWindow(int(host.winId()))
        ok = not pm.toImage().isNull() and pm.width() > 0
        if ok:
            pm.toImage().save(out_png)
        return "grabWindow(winId)", ok
    for tool, cmd in (
        ("grim", ["grim", out_png]),
        ("spectacle", ["spectacle", "-b", "-n", "-o", out_png]),
    ):
        if shutil.which(tool):
            try:
                subprocess.run(cmd, timeout=15, check=True, capture_output=True)
                return tool, os.path.exists(out_png) and os.path.getsize(out_png) > 0
            except Exception:
                return tool, False
    return None, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["bitmap", "screen-stock", "screen-native"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=120)
    args = ap.parse_args()

    # Vulkan-only instance BEFORE any adapter work (see Tier-0 notes).
    from wgpu.backends.wgpu_native.extras import set_instance_extras

    set_instance_extras(backends=["Vulkan"])

    from PySide6.QtCore import QRect, Qt
    from PySide6.QtWidgets import (
        QApplication,
        QDockWidget,
        QLabel,
        QMainWindow,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    app = QApplication(sys.argv)
    session = app.platformName()
    evidence = {
        "harness": "wgpu-gate-b-tier1",
        "mode": args.mode,
        "qt_platform": session,
        "phases": {},
        "notes": [],
    }
    import rendercanvas
    import wgpu

    evidence["versions"] = {
        "wgpu": wgpu.__version__,
        "rendercanvas": rendercanvas.__version__,
    }

    renderer = Renderer()
    evidence["adapter"] = renderer.adapter.info["device"]

    # --- ArrayScope-shaped window: central tabs + docks + overlays.
    win = QMainWindow()
    win.setWindowTitle(f"wgpu gate B [{args.mode}/{session}]")
    tabs = QTabWidget()
    win.setCentralWidget(tabs)

    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(4, 4, 4, 4)

    if args.mode == "bitmap":
        canvas = RendercanvasCanvas(renderer, page, "bitmap")
    elif args.mode == "screen-stock":
        canvas = RendercanvasCanvas(renderer, page, "screen")
    else:
        canvas = NativeWaylandCanvas(renderer, page)
    canvas.widget.setMinimumSize(400, 300)
    layout.addWidget(canvas.widget)
    tabs.addTab(page, "Viewer")
    tabs.addTab(QTextEdit("second tab (canvas hidden)"), "Other")

    for name in ("Inspection", "Operations"):
        dock = QDockWidget(name, win)
        dock.setWidget(QTextEdit(f"{name} panel"))
        win.addDockWidget(Qt.RightDockWidgetArea, dock)

    # Overlay stand-ins OVER the canvas (tile-truth/ROI style): siblings in
    # the page, positioned above the canvas rect, magenta for the oracle.
    overlays = []
    for i, text in enumerate(("TILE 42 TRUTH", "ROI 1")):
        ov = QLabel(text, page)
        ov.setStyleSheet("background: #FF00FF; color: black; padding: 2px;")
        ov.move(40 + i * 160, 60)
        ov.raise_()
        ov.show()
        overlays.append(ov)

    win.resize(*REAL_SIZE)
    win.show()
    app.processEvents()
    for ov in overlays:
        ov.raise_()

    def run_phase(name, n, setup=None):
        if setup:
            setup()
            app.processEvents()
        times = []
        subtimes = {}
        for i in range(n):
            t0 = time.perf_counter()
            sub = canvas.frame(i / 60.0) or {}
            times.append((time.perf_counter() - t0) * 1000)
            for k, v in sub.items():
                subtimes.setdefault(k, []).append(v)
            app.processEvents()
        entry = {"frame": stats(times)}
        for k, v in subtimes.items():
            entry[k.replace("_ms", "")] = stats(v)
        evidence["phases"][name] = entry

    docks = win.findChildren(QDockWidget)
    run_phase("steady", args.frames)
    run_phase("dock_float", 60, lambda: [d.setFloating(True) for d in docks])
    run_phase("dock_redock", 60, lambda: [d.setFloating(False) for d in docks])

    # Tab away: canvas hidden; keep ticking the event loop, then draw again.
    tabs.setCurrentIndex(1)
    app.processEvents()
    time.sleep(0.3)
    hidden_errors_before = getattr(canvas, "errors", 0)
    run_phase("tab_back", 60, lambda: tabs.setCurrentIndex(0))
    evidence["notes"].append(
        f"errors_while_hidden_delta={getattr(canvas, 'errors', 0) - hidden_errors_before}"
    )

    def resize_cycle():
        pass

    times = []
    for i in range(60):
        if i % 20 == 0:
            w, h = (1000 + (i // 20) * 180, 520 + (i // 20) * 90)
            win.resize(w, h)
            app.processEvents()
        t0 = time.perf_counter()
        canvas.frame(i / 60.0)
        times.append((time.perf_counter() - t0) * 1000)
        app.processEvents()
    evidence["phases"]["resize"] = {"frame": stats(times)}
    win.resize(*REAL_SIZE)
    app.processEvents()

    # Second window with its own canvas, same device.
    win2 = QMainWindow()
    win2.setWindowTitle("gate B second window")
    page2 = QWidget()
    lay2 = QVBoxLayout(page2)
    if args.mode == "screen-native":
        canvas2 = NativeWaylandCanvas(renderer, page2)
    else:
        canvas2 = RendercanvasCanvas(
            renderer, page2, "bitmap" if args.mode == "bitmap" else "screen"
        )
    canvas2.widget.setMinimumSize(300, 200)
    lay2.addWidget(canvas2.widget)
    win2.setCentralWidget(page2)
    win2.resize(500, 350)
    win2.show()
    app.processEvents()
    times = []
    for i in range(60):
        t0 = time.perf_counter()
        canvas.frame(i / 60.0)
        canvas2.frame(i / 30.0)
        times.append((time.perf_counter() - t0) * 1000)
        app.processEvents()
    evidence["phases"]["two_windows_both"] = {"frame": stats(times)}
    win2.close()
    app.processEvents()

    # --- Overlay oracle.
    for ov in overlays:
        ov.raise_()
    for _ in range(5):
        canvas.frame(1.0)
        app.processEvents()
    canvas_rect_in_page = canvas.widget.geometry()
    backing = win.grab().toImage()
    # Map canvas rect to window coords for the crop.
    top_left = canvas.widget.mapTo(win, canvas.widget.rect().topLeft())
    crop = QRect(
        top_left.x(), top_left.y(), canvas_rect_in_page.width(), canvas_rect_in_page.height()
    )
    backing_count = count_magenta(backing, crop)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    backing.save(os.path.join(out_dir, f"{args.mode}-{session}-backing.png"))
    comp_png = os.path.join(out_dir, f"{args.mode}-{session}-compositor.png")
    tool, comp_ok = compositor_capture(session, win, comp_png)
    comp_count = None
    if comp_ok:
        from PySide6.QtGui import QImage

        comp_count = count_magenta(QImage(comp_png))  # whole screen: magenta only from us
    evidence["overlay"] = {
        "backing_store_magenta_px": backing_count,
        "compositor_tool": tool,
        "compositor_capture_ok": comp_ok,
        "compositor_magenta_px": comp_count,
    }
    if hasattr(canvas, "statuses"):
        evidence["surface_statuses"] = {str(k): v for k, v in canvas.statuses.items()}
        evidence["surface_errors"] = canvas.errors

    # --- Bitmap readback price microbench (offscreen; both sizes).
    if args.mode == "bitmap":
        bench = {}
        for label, (w, h) in (("real_1300x650", REAL_SIZE), ("stress_3840x2160", STRESS_SIZE)):
            tex = renderer.device.create_texture(
                size=(w, h, 1),
                format="rgba8unorm",
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
            )
            view = tex.create_view()
            renderer.draw_to_view(view, "rgba8unorm", 0.5)
            times_rb = []
            for _ in range(20):
                t0 = time.perf_counter()
                renderer.device.queue.read_texture(
                    {"texture": tex},
                    {"bytes_per_row": w * 4, "rows_per_image": h},
                    (w, h, 1),
                )
                times_rb.append((time.perf_counter() - t0) * 1000)
            bench[label] = {
                "readback": stats(times_rb),
                "mb_per_frame": round(w * h * 4 / 1e6, 1),
            }
            tex.destroy()
        evidence["readback_price"] = bench

    with open(args.out, "w") as f:
        json.dump(evidence, f, indent=2)
    print(json.dumps(evidence, indent=2))
    win.close()
    app.processEvents()


if __name__ == "__main__":
    main()
