"""What the screen present path actually puts on screen, per windowing system.

The X widening (2026-07-31) rests on two claims that were inference until
this script measured them.  Every arm runs identically under ``wayland`` and
``xcb`` so each answer is a CONTRAST between two platforms rather than a
number with nothing to be wrong against; the compositor is the same managed
headless Weston either way (its own XWayland serves the xcb arm).

**Arm 1 — overlays reach the screen.**  Gate B measured the screen path under
xcb as journey-clean but with overlays missing: Qt cannot composite its own
painting above a native child window, so ROI outlines, tile-truth labels and
floating chips were in the backing store and not on screen (5,416 px vs 0).
ArrayScope answered that by drawing overlays INSIDE the wgpu frame, keyed on
the present method rather than on the platform — which makes X11 support
plausible, not proven.  This arm proves it on pixels, with three oracles that
can each fail:

  * each overlay family is counted by its own distinctive colour in a capture
    of the real compositor output;
  * the same scene is captured with the overlays removed, so a count only
    counts as evidence if it collapses in the control;
  * Qt's own ``grab()`` of the same window is captured too.  It is the gate-B
    contrast inverted: the pixels must now be on the screen and NOT in the
    backing store.

**Arm 2 — can canvas pixels land outside their window.**  A Wayland
subsurface is not clipped by its parent, which is what makes coalescing a
resize into the next paced frame a trade (``screen_canvas._on_surface_resized``):
while shrinking, an old, larger buffer hangs outside the window until a
correctly-sized one is presented.  The X11 analogue was never characterised.
Rather than race a ~17 ms transient with a ~100 ms screenshot, this arm holds
the pathological state still: a production ``WgpuScreenCanvas`` is forced to a
geometry reaching well outside its top-level and driven with a flat, garish
clear colour, so counting that colour against the area the window could
legitimately own answers the question in steady state.  A clear colour rather
than the app's scene, because the question is a windowing-system property of
the canvas and nothing else should be able to move the number.

**Arm 3 — resize behaviour in the real view.**  Events, draws and how long
the stale window stays open, in the same shape as the Wayland numbers quoted
in that comment.

Run (either arm; the launcher owns the compositor):

    tools/env python -m arrayscope.tools.headless_display --qt-platform xcb \\
        -- python experiments/wgpu_gate_b/x11_screen_evidence.py out.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.tools.headless_display import capture_output

# Overlay colours chosen to be unmistakable against the scene, the Qt chrome
# and the compositor background, so a colour count needs no segmentation.
ROI_COLOR = (255, 0, 255)  # magenta ROI outline
CHIP_COLOR = (0, 200, 255)  # cyan chip background
TRUTH_BORDER_COLOR = (34, 211, 238)  # the app's own DRAW-state label border
TRUTH_TEXT_COLOR = (165, 243, 252)  # the app's own label ink
OVERHANG_CLEAR_COLOR = (255, 0, 128)  # arm 2's flat clear

WINDOW_SIZE = (760, 560)
SCENE_VALUE = 0.35  # a mid grey: far from every colour counted above


def _settle(app, seconds=0.35):
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.005)


def _capture_array(tag, scratch):
    from PIL import Image

    path = capture_output(Path(scratch) / f"{tag}.png")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.int16)


def _color_pixels(pixels, color, tolerance=26):
    """Pixels within ``tolerance`` of ``color`` on every channel.

    A tolerance rather than an equality: the chip is translucent over the
    scene, outlines and glyphs are antialiased, and the capture round-trips
    through the compositor's own format conversion.
    """

    target = np.asarray(color, dtype=np.int16)
    return int(np.count_nonzero(np.all(np.abs(pixels - target) <= tolerance, axis=-1)))


# ---- arm 1: overlays -------------------------------------------------------


def _build_view(app):
    from arrayscope.display.wgpu_imageview2d import WgpuImageView2D

    view = WgpuImageView2D(present_method="screen")
    view.resize(*WINDOW_SIZE)
    view.show()
    for _ in range(120):
        app.processEvents()
        time.sleep(0.01)
        if view.wgpuPresentMethod() == "screen":
            break
    if view.wgpuPresentMethod() != "screen":
        raise SystemExit(
            f"screen present path did not activate: {view.wgpuPresentMethodFallbackReason()}"
        )
    return view


def _commit_scene(view):
    from tests.display.test_imageview2d import _present_tiled

    canvas = np.full((96, 128), SCENE_VALUE, dtype=np.float32)
    return _present_tiled(
        view,
        canvas,
        histogramData=canvas.copy(),
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
    )


def _add_overlays(app, view):
    """ROI outline + tile-truth label + floating chip: the three families."""

    view.createRoi("rectangle", rect=(16.0, 16.0, 64.0, 48.0), color=ROI_COLOR, line_width=3.0)
    view.setTileTruthOverlayRows(
        ({"tile": 0, "drawable": True, "tile_rect": (0.0, 0.0, 128.0, 96.0)},)
    )
    chip = QtWidgets.QFrame(view)
    chip.setObjectName("EvidenceChip")
    chip.setStyleSheet(
        "QFrame#EvidenceChip {"
        f" background: rgb({CHIP_COLOR[0]}, {CHIP_COLOR[1]}, {CHIP_COLOR[2]});"
        " border: 1px solid #202020; border-radius: 4px; }"
    )
    chip.setGeometry(48, 48, 180, 48)
    chip.show()
    view._prepare_display_overlay_widget(chip)
    _settle(app, 0.5)
    return chip


def _overlay_counts(pixels):
    return {
        "roi_outline_px": _color_pixels(pixels, ROI_COLOR),
        "chip_px": _color_pixels(pixels, CHIP_COLOR),
        "truth_border_px": _color_pixels(pixels, TRUTH_BORDER_COLOR),
        "truth_text_px": _color_pixels(pixels, TRUTH_TEXT_COLOR, tolerance=18),
    }


def arm_overlays(app, view, scratch):
    """Overlay pixels on screen vs in Qt's backing store, with a control."""

    from PIL import Image

    chip = _add_overlays(app, view)
    _settle(app, 0.7)
    on_screen = _overlay_counts(_capture_array("overlays_screen", scratch))

    # Qt's own render of the same window: the gate-B backing store.  A grab
    # never involves the compositor, so this is what Qt WOULD have shown if
    # it could composite above the native child.
    grab_path = Path(scratch) / "overlays_backing_store.png"
    view.grab().save(str(grab_path))
    with Image.open(grab_path) as image:
        backing = np.asarray(image.convert("RGB"), dtype=np.int16)
    in_backing_store = _overlay_counts(backing)

    view.clearRois()
    view.setTileTruthOverlayRows(())
    chip.hide()
    chip.setParent(None)
    chip.deleteLater()
    _settle(app, 0.7)
    control = _overlay_counts(_capture_array("overlays_control", scratch))
    return {
        "on_screen": on_screen,
        "in_backing_store": in_backing_store,
        "control_no_overlays": control,
    }


# ---- arm 2: can canvas pixels land outside their window --------------------


def arm_geometry_overhang(app, scratch):
    """Drive a production canvas past its window and count what reaches screen.

    ``ceiling_px`` is every pixel the canvas could own if the window clipped
    it — the top-level's whole area, which is generous (the canvas is inset
    inside it).  Anything beyond that came from outside the window, and needs
    no knowledge of where the window sits on the output, which matters
    because a Wayland client cannot ask for its own position.
    """

    from arrayscope.display.backends.wgpu.screen_canvas import WgpuScreenCanvas
    from arrayscope.display.wgpu_imageview2d import _shared_wgpu_device

    # The app's own device, not a fresh one: the wgpu instance already exists
    # by the time arm 1 has run, and re-masking it to Vulkan would raise.
    # Sharing it also keeps this arm on the same adapter as the rest.
    device = _shared_wgpu_device()

    host = QtWidgets.QWidget()
    host.setWindowTitle("overhang")
    width, height = 420, 320
    host.resize(width, height)
    canvas = WgpuScreenCanvas(parent=host)
    canvas.setGeometry(0, 0, width, height)
    host.show()
    _settle(app, 0.6)

    red, green, blue = (value / 255.0 for value in OVERHANG_CLEAR_COLOR)

    def draw_clear():
        context = canvas.ensure_context(device)
        if context is None:
            return False
        texture = context.get_current_texture()
        encoder = device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": texture.create_view(),
                    "load_op": "clear",
                    "store_op": "store",
                    "clear_value": (red, green, blue, 1.0),
                }
            ]
        )
        render_pass.end()
        device.queue.submit([encoder.finish()])
        context.present()
        return True

    canvas.request_draw(draw_clear)
    for _ in range(20):
        _settle(app, 0.05)
    baseline = _color_pixels(_capture_array("overhang_before", scratch), OVERHANG_CLEAR_COLOR)

    # Reach well outside the top-level in both directions.
    overhang = 300
    canvas._container.setGeometry(0, 0, width + overhang, height + overhang)
    canvas._surface_window.setGeometry(0, 0, width + overhang, height + overhang)
    canvas.request_draw()
    for _ in range(20):
        _settle(app, 0.05)
    after = _color_pixels(_capture_array("overhang_after", scratch), OVERHANG_CLEAR_COLOR)

    result = {
        "window_size": [width, height],
        "canvas_requested_size": [width + overhang, height + overhang],
        "canvas_physical_size": list(canvas.get_physical_size()),
        "ceiling_px": width * height,
        "clear_px_before": baseline,
        "clear_px_after": after,
        "spilled_px": max(0, after - width * height),
    }
    canvas.teardown()
    host.close()
    _settle(app, 0.2)
    return result


# ---- arm 3: resize behaviour in the real view ------------------------------


def arm_resize_timing(app, view):
    """Resize events, draws, and how long the stale window stays open.

    Same shape as the Wayland numbers quoted in ``_on_surface_resized``: a
    scripted shrink/grow sweep, counting the resize events Qt delivered
    against the draws actually taken, and timing each resize event to the
    next draw that answered it.
    """

    canvas = view._wgpu_canvas
    events = {"count": 0}
    draws = {"count": 0}
    stale_windows_ms = []
    pending = {"at": None}

    original_on_resized = canvas._on_surface_resized
    original_invoke = canvas._invoke_draw

    def counting_on_resized():
        events["count"] += 1
        if pending["at"] is None:
            pending["at"] = time.perf_counter()
        original_on_resized()

    def timing_invoke():
        original_invoke()
        draws["count"] += 1
        if pending["at"] is not None:
            stale_windows_ms.append((time.perf_counter() - pending["at"]) * 1000.0)
            pending["at"] = None

    canvas._on_surface_resized = counting_on_resized
    canvas._invoke_draw = timing_invoke
    presents_before = int(view.wgpuPresentationDiagnostics()["wgpu_screen_presents"])

    try:
        base_w, base_h = WINDOW_SIZE
        started = time.perf_counter()
        for step in range(60):
            shrink = step if step < 30 else 59 - step
            view.resize(base_w - 6 * shrink, base_h - 4 * shrink)
            app.processEvents()
            time.sleep(0.008)
        _settle(app, 0.5)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    finally:
        canvas._on_surface_resized = original_on_resized
        canvas._invoke_draw = original_invoke

    presents_after = int(view.wgpuPresentationDiagnostics()["wgpu_screen_presents"])
    ordered = sorted(stale_windows_ms)
    return {
        "resize_events": events["count"],
        "draws": draws["count"],
        "presents": presents_after - presents_before,
        "sweep_ms": round(elapsed_ms, 1),
        "stale_window_ms_mean": round(float(np.mean(ordered)), 2) if ordered else None,
        "stale_window_ms_max": round(float(max(ordered)), 2) if ordered else None,
        "stale_window_samples": len(ordered),
    }


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    scratch = (out_path.parent if out_path is not None else Path.cwd()) / "captures"
    scratch.mkdir(parents=True, exist_ok=True)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    result = {"qt_platform": str(app.platformName())}

    def record(name, work):
        # One arm failing must not throw away the others' numbers: a partial
        # evidence file that says WHICH arm broke is worth more than a stack
        # trace and nothing.
        try:
            result[name] = work()
        except Exception as exc:
            result[name] = {"error": f"{type(exc).__name__}: {exc}"}

    view = _build_view(app)
    try:
        canvas = view._wgpu_canvas
        _commit_scene(view)
        _settle(app, 0.6)
        # Read AFTER the first real frame: the swapchain is created lazily
        # inside the draw, so asking at construction time reports nothing.
        result["present_mode"] = canvas.present_mode
        result["present_platform"] = canvas.present_platform
        result["present_modes_available"] = list(canvas.present_modes_available)
        record("overlays", lambda: arm_overlays(app, view, scratch))
        record("resize", lambda: arm_resize_timing(app, view))
    finally:
        view.close()
        view.teardown_surface()
        for _ in range(20):
            app.processEvents()

    record("overhang", lambda: arm_geometry_overhang(app, scratch))

    print(json.dumps(result, indent=2))
    if out_path is not None:
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    QtCore.QTimer.singleShot(0, app.quit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
