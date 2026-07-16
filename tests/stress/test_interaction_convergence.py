"""Live interaction-convergence harness (field stall 2026-07-16, session 148).

Drives the user's real dataset through a seeded ~12 s human-shaped gesture mix
— zoom glides in/out, axis-0 range-window shifts including an explicit
full-range spelling and its ``None`` reset, and montage-axis slice pokes — on
a full-axis FFT-chain complex montage (VisPy, resident quality policy), then
stops interacting and requires the session to converge on its own.

Before the 2026-07-16 fixes this stalled deterministically on a real display:
retained payloads minted under the explicit-full-range spelling could never
satisfy targets minted under the ``None`` spelling, every commit silently
rejected their upserts, the shared first-pass barrier never opened, and the
pipeline replanned forever with ~90-110 required tiles empty or stale.

Run explicitly (serial, real display, real data):

    ARRAYSCOPE_STRESS=1 QT_QPA_PLATFORM=wayland \
        pytest tests/stress/test_interaction_convergence.py -n 0

Offscreen runs historically do NOT reproduce the scheduling shape (see
docs/redesign/black-tiles-and-priority.md ground rules); the deterministic
unit gates live in tests/core/test_view_state.py,
tests/window/test_montage_lod_residency.py, and
tests/display/test_vispy_physical_presentation.py.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "_WIPDelRec-tT2_20260223150234_14.nii"

pytestmark = pytest.mark.skipif(
    os.environ.get("ARRAYSCOPE_STRESS") != "1"
    or os.environ.get("QT_QPA_PLATFORM", "") == "offscreen"
    or not DATA_PATH.exists(),
    reason="needs ARRAYSCOPE_STRESS=1, a real display, and the local NIfTI dataset",
)

_FILL_TIMEOUT_S = 120
_CONVERGE_TIMEOUT_S = 45


def _settled(win) -> bool:
    session = getattr(win.renderer, "_frame_session", None)
    if session is None:
        return False
    return bool(
        session.visible_plan_complete()
        and not win.montage_tile_evaluation_controller.is_busy()
        and session.required_target_settled()
    )


def _pump(qtbot, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qtbot.wait(20)


def test_interaction_churn_converges_on_real_data(qtbot):
    from tests.ui.helpers import make_backend_window, restore_default_backend, use_vispy_backend

    settings = use_vispy_backend(extra_settings={"montage_quality_policy": "resident"})
    from arrayscope.io.file_interpreters import load_file
    from arrayscope.operations.pipeline import CenteredFFT, CenteredIFFT, FFTShift

    loaded = load_file(str(DATA_PATH))
    data = np.asarray(loaded.data if hasattr(loaded, "data") else loaded)

    win = make_backend_window(qtbot, data)
    win.resize(1200, 900)
    try:
        win.show()
        win.operation_coordinator.load_operations(
            (CenteredFFT(axis=2), FFTShift(axis=2), CenteredIFFT(axis=2))
        )
        win._set_document(win.operation_coordinator.document)
        win._coerce_channel_for_current_dtype()

        n = int(data.shape[2])
        state = win.view_state.with_montage_axis(
            2, columns=None, indices=tuple(range(n)), text=f"0:{n}"
        )
        win._set_view_state(state)
        win.update_image_view()
        qtbot.waitUntil(lambda: _settled(win), timeout=_FILL_TIMEOUT_S * 1000)

        view = win.img_view.getView()
        (x0, x1), (y0, y1) = view.viewRange()[0], view.viewRange()[1]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = (x1 - x0), (y1 - y0)
        rng = np.random.default_rng(7)
        ax_len = int(data.shape[0])

        for step in range(48):
            t = step / 47.0
            mode = step % 4
            factor = (
                0.30 + 0.10 * float(rng.random())
                if mode == 0
                else 1.4 + 0.8 * t
                if mode == 1
                else 0.6
                if mode == 2
                else 2.2
            )
            jx = float(rng.uniform(-0.15, 0.15)) * w
            jy = float(rng.uniform(-0.15, 0.15)) * h
            half_w, half_h = w * factor / 2, h * factor / 2
            view.setRange(
                xRange=(cx + jx - half_w, cx + jx + half_w),
                yRange=(cy + jy - half_h, cy + jy + half_h),
                padding=0,
            )
            if step % 6 == 2:
                phase = (step // 6) % 4
                if phase == 0:
                    lo = 60 + (step * 3) % 80
                    indices, text = tuple(range(lo, lo + 128)), f"{lo}:{lo + 128}"
                elif phase == 1:
                    lo = 40 + (step * 5) % 60
                    indices, text = tuple(range(lo, lo + 160)), f"{lo}:{lo + 160}"
                elif phase == 2:
                    indices, text = tuple(range(ax_len)), f"0:{ax_len}"
                else:
                    indices, text = None, None
                win._apply_slice_state(
                    0,
                    win.view_state.with_axis_range(0, indices=indices, text=text),
                    reason="slice-range",
                    interactive=True,
                    immediate_axis_only=False,
                )
            if step % 5 == 0:
                idx = int(rng.integers(low=n // 4, high=3 * n // 4))
                win._apply_slice_state(
                    2,
                    win.view_state.with_slice(2, idx),
                    reason="slice",
                    interactive=True,
                    immediate_axis_only=True,
                )
            _pump(qtbot, 0.09)

        qtbot.waitUntil(lambda: _settled(win), timeout=_CONVERGE_TIMEOUT_S * 1000)
        session = win.renderer._frame_session
        assert session.required_target_unsettled_tiles() == ()
        assert not session.pending_payload_upserts
    finally:
        win.close()
        restore_default_backend(settings)
