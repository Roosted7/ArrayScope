"""Live wgpu preview: real data through the renderer command protocol.

The smallest end-to-end demonstration of the gate-B GO verdict inside a real
Qt widget: load an array, chunk it into the executor's page pool via
``EnsureChunkResident``, and drive mode/levels/pan interactions purely as
protocol commands into a bitmap-presented ``QRenderWidget`` (the Tier-1
default mode: every Qt overlay keeps working; here a plain Qt status label
sits over the canvas as the standing witness).

Keys: m/p/r/i = magnitude/phase/real/imag · [ ] = levels window · arrows =
pan (descriptor-only; zero uploads once resident) · 0/1 = requested LOD.
The window title live-reports uploads-per-frame so the zero-upload behavior
is visible while interacting.

Usage:
    python -m arrayscope.tools.wgpu_preview [path.nii|.npy]  [--slice N]
    python -m arrayscope.tools.wgpu_preview path --smoke out.png   # offscreen

Experimental: this tool exercises the protocol seam; it is not a product
viewer and changes no live rendering path.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def _load_plane(path: str, slice_index: int | None) -> np.ndarray:
    if path.endswith(".npy"):
        data = np.load(path)
    else:
        import nibabel

        data = np.asanyarray(nibabel.load(path).dataobj)
    data = np.squeeze(data)
    while data.ndim > 2:
        axis = int(np.argmin(data.shape))
        index = data.shape[axis] // 2 if slice_index is None else slice_index
        data = np.take(data, index, axis=axis)
    return np.ascontiguousarray(data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help=".nii/.npy source")
    ap.add_argument("--slice", type=int, default=None)
    ap.add_argument("--smoke", metavar="PNG", help="offscreen: one frame, save, exit")
    args = ap.parse_args(argv)

    if args.smoke:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

    app = QApplication(sys.argv[:1])

    from rendercanvas.pyside6 import QRenderWidget  # AFTER QApplication: platform stays ours

    from arrayscope.gpu.command_protocol import (
        BindContentPlanes,
        ContentPlane,
        DisplayMapping,
        EnsureChunkResident,
        FrameSubmission,
        PresentGeneration,
        SetDisplayMapping,
        TileInstance,
        UpdateTileInstances,
    )
    from arrayscope.gpu.keys import COMPLEX_RG32F
    from arrayscope.gpu.wgpu_executor import PAGE, WgpuPlaneExecutor, plane_chunk_key

    plane = _load_plane(args.path, args.slice)
    is_complex = np.iscomplexobj(plane)
    h, w = plane.shape
    grid_h, grid_w = -(-h // PAGE), -(-w // PAGE)

    executor = WgpuPlaneExecutor(
        (grid_h * PAGE, grid_w * PAGE), max_lod=1, pool_layers=grid_w * grid_h * 2 + 8
    )

    # Residency: pad to page multiples, L1 (2x2 mean, pinned) then L0.
    padded = np.zeros((grid_h * PAGE, grid_w * PAGE), np.complex64 if is_complex else np.float32)
    padded[:h, :w] = plane
    stacked = np.stack(
        [padded.real.astype(np.float32), padded.imag.astype(np.float32)]
        if is_complex
        else [padded.astype(np.float32), np.zeros_like(padded, np.float32)],
        axis=-1,
    )
    l1 = (stacked[0::2, 0::2] + stacked[1::2, 0::2] + stacked[0::2, 1::2] + stacked[1::2, 1::2]) / 4

    commands = [
        BindContentPlanes(
            (
                ContentPlane(
                    "preview",
                    "identity",
                    (grid_h * PAGE, grid_w * PAGE),
                    max_lod=1,
                    representation=COMPLEX_RG32F,
                ),
            )
        )
    ]
    for cy in range(-(-grid_h // 2)):
        for cx in range(-(-grid_w // 2)):
            page = np.zeros((PAGE, PAGE, 2), np.float32)
            block = l1[cy * PAGE : (cy + 1) * PAGE, cx * PAGE : (cx + 1) * PAGE]
            page[: block.shape[0], : block.shape[1]] = block
            commands.append(
                EnsureChunkResident(
                    plane_chunk_key("preview", "identity", 1, cx, cy), page, pinned=True
                )
            )
    commands.extend(
        EnsureChunkResident(
            plane_chunk_key("preview", "identity", 0, cx, cy),
            stacked[cy * PAGE : (cy + 1) * PAGE, cx * PAGE : (cx + 1) * PAGE],
        )
        for cy in range(grid_h)
        for cx in range(grid_w)
    )
    t0 = time.perf_counter()
    report = executor.submit(FrameSubmission(0, commands))
    report.wait_completed()
    print(
        f"resident: {report.uploads} pages ({report.uploads * PAGE * PAGE * 8 / 1e6:.1f} MB) "
        f"in {(time.perf_counter() - t0) * 1e3:.1f} ms"
    )

    # View state (mutated by keys; every change is a protocol submission).
    finite = np.abs(plane[np.isfinite(plane)]) if is_complex else plane[np.isfinite(plane)]
    lo, hi = (
        (0.0, float(np.percentile(finite, 99.5)))
        if is_complex
        else (
            float(np.percentile(finite, 0.5)),
            float(np.percentile(finite, 99.5)),
        )
    )
    # A smooth non-gray LUT (toggled with 'v') to demonstrate that LUT swaps
    # are pure mapping state — zero uploads, like mode/levels changes.
    ramp = np.arange(256)
    color_lut = (
        np.stack(
            [ramp, (ramp * ramp) // 255, np.sqrt(ramp / 255.0) * 255, np.full(256, 255)],
            axis=-1,
        )
        .astype(np.uint8)
        .tobytes()
    )

    state = {
        "mode": "magnitude" if is_complex else "real",
        "lo": lo,
        "hi": max(hi, lo + 1e-6),
        "origin": [0.0, 0.0],
        "size": [float(w), float(h)],
        "lod": 0,
        "generation": 0,
        "lut": None,
    }

    win = QWidget()
    win.setWindowTitle("ArrayScope wgpu preview")
    layout = QVBoxLayout(win)
    layout.setContentsMargins(2, 2, 2, 2)
    canvas = QRenderWidget(parent=win, present_method="bitmap", update_mode="ondemand")
    canvas.setMinimumSize(640, 640)
    layout.addWidget(canvas)
    overlay = QLabel("Qt overlay ✓ (bitmap composition)", win)
    overlay.setStyleSheet("background: rgba(255,0,255,200); color: black; padding: 3px;")
    overlay.move(16, 16)
    overlay.raise_()

    context = canvas.get_context("wgpu")
    configured = {}

    def draw():
        if "fmt" not in configured:
            adapter_fmt = context.get_preferred_format(None)
            fmt = adapter_fmt.removesuffix("-srgb")
            context.configure(device=executor.device, format=fmt)
            configured["fmt"] = fmt
        state["generation"] += 1
        tiles = (
            TileInstance(
                (0.0, 0.0, 1.0, 1.0),
                tuple(state["origin"]),
                tuple(state["size"]),
                state["lod"],
            ),
        )
        t0 = time.perf_counter()
        report = executor.submit(
            FrameSubmission(
                state["generation"],
                (
                    SetDisplayMapping(
                        DisplayMapping(state["mode"], state["lo"], state["hi"], lut=state["lut"])
                    ),
                    UpdateTileInstances(tiles),
                    PresentGeneration(state["generation"]),
                ),
            ),
            present_to=context.get_current_texture().create_view(),
            present_format=configured["fmt"],
        )
        dt = (time.perf_counter() - t0) * 1e3
        win.setWindowTitle(
            f"wgpu preview · {state['mode']} [{state['lo']:.3g},{state['hi']:.3g}] "
            f"lod{state['lod']} · {dt:.1f} ms · uploads {report.uploads}"
        )

    canvas.request_draw(draw)

    def keys(event):
        k, text = event.key(), event.text()
        span = state["hi"] - state["lo"]
        step = state["size"][0] * 0.05
        if text == "m":
            state["mode"] = "magnitude"
        elif text == "p" and is_complex:
            state["mode"] = "phase"
        elif text == "r":
            state["mode"] = "real"
        elif text == "i" and is_complex:
            state["mode"] = "imag"
        elif text == "[":
            state["hi"] = state["lo"] + span * 0.8
        elif text == "]":
            state["hi"] = state["lo"] + span * 1.25
        elif text == "v":
            state["lut"] = color_lut if state["lut"] is None else None
        elif text == "0":
            state["lod"] = 0
        elif text == "1":
            state["lod"] = 1
        elif k == Qt.Key_Left:
            state["origin"][0] -= step
        elif k == Qt.Key_Right:
            state["origin"][0] += step
        elif k == Qt.Key_Up:
            state["origin"][1] -= step
        elif k == Qt.Key_Down:
            state["origin"][1] += step
        else:
            return
        canvas.request_draw(draw)

    win.keyPressEvent = keys
    win.resize(720, 760)
    win.show()

    if args.smoke:
        canvas.force_draw()
        # Exercise a descriptor-only pan + mode/levels change, then capture.
        state["origin"][0] += 32
        state["hi"] *= 0.8
        canvas.force_draw()
        app.processEvents()
        ok = win.grab().save(args.smoke)
        print(f"smoke frame saved={ok} -> {args.smoke}")
        return 0

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
