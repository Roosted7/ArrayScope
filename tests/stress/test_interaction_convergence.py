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

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_S



# PAL-relaxed LUT[0]: the color of zero-magnitude complex texels drawn
# without their phase mapping (a_mode 3 instead of 4).  Real phase-color
# content contains a few hundred legitimately near-orange pixels; the
# 2026-07-16 floor-mapping defect produced 6,000-19,000 per frame.
_PAL_RELAXED_ORANGE = (249, 127, 16)
_ORANGE_TOLERANCE = 16
_ORANGE_DEFECT_THRESHOLD = 3000


def _orange_pixel_count(win) -> int:
    frame = np.asarray(win.img_view._vispy_canvas.render())[..., :3].astype(np.int16)
    reference = np.array(_PAL_RELAXED_ORANGE, dtype=np.int16)
    return int(np.count_nonzero(np.all(np.abs(frame - reference) <= _ORANGE_TOLERANCE, axis=-1)))


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


def _dump_convergence_state(win, label: str) -> None:
    """On failure: the owner-chain numbers a stall diagnosis starts from."""

    session = getattr(win.renderer, "_frame_session", None)
    if session is None:
        print(f"[{label}] no frame session")
        return
    unsettled = session.required_target_unsettled_tiles()
    print(
        f"[{label}] session={session.session_id} unsettled={len(unsettled)} "
        f"pending={len(session.pending_tiles)} "
        f"active={len(session.active_tile_requests)} "
        f"evaluating={len(session.lifecycle.evaluating_tiles)} "
        f"dirty={len(session.dirty_payloads)} "
        f"upserts={len(session.pending_payload_upserts)} "
        f"loading={len(session.loading_tiles)} "
        f"flush={session.flush_pending}/{session.final_commit_pending} "
        f"visible_busy={win.montage_tile_evaluation_controller.is_busy()} "
        f"plan_complete={session.visible_plan_complete()}"
    )
    print(f"[{label}] unsettled sample: {tuple(unsettled)[:24]}")
    fan_in = session.stage_fan_in
    print(
        f"[{label}] stage_deferred={getattr(session, 'stage_planning_deferred', None)} "
        f"async={getattr(session, 'stage_planning_async', None)} "
        f"deferred_missing={len(tuple(getattr(session, 'deferred_missing_tiles', ()) or ()))} "
        f"fanin_active={len(getattr(fan_in, 'active_requests', ()) or ())} "
        f"fanin_attached={len(getattr(fan_in, 'attached_requests', ()) or ())} "
        f"fanin_stage_keys={len(getattr(fan_in, 'tile_stage_keys', {}) or {})} "
        f"viewport_pending={getattr(win, '_montage_viewport_update_pending', None)} "
        f"viewport_interaction={getattr(win, '_viewport_interaction_active', None)} "
        f"coordinator_interactive={getattr(getattr(win, 'render_coordinator', None), 'interactive_active', None)} "
        f"native_deferred={getattr(getattr(getattr(session, 'pipeline', None), 'counters', None), 'interactive_native_deferred', None)}"
    )
    kernel = getattr(win, "kernel", None)
    quotas = getattr(kernel, "_lane_quotas", None)
    print(f"[{label}] lane quotas: {quotas}")
    for attr in ("_queues", "_pending", "_heap", "_tasks"):
        value = getattr(kernel, attr, None)
        if value is not None:
            try:
                print(f"[{label}] kernel.{attr}: size={len(value)}")
            except TypeError:
                pass
    coordinator = getattr(win, "render_coordinator", None)
    print(f"[{label}] coordinator pending_render={getattr(coordinator, 'has_pending_render', None)} "
          f"backpressure_skips={getattr(coordinator, 'presentation_backpressure_skips', None)}")
    renderer = win.renderer
    print(
        f"[{label}] commit outcome={getattr(renderer, '_last_montage_commit_outcome', None)} "
        f"counts={getattr(renderer, '_montage_commit_outcome_counts', None)} "
        f"gate_no_progress={getattr(renderer, '_montage_gate_no_progress', None)} "
        f"delta_upserts={getattr(renderer, '_last_montage_commit_delta_upserts', None)} "
        f"fast_reject={getattr(renderer, '_last_montage_atomic_fast_reject_reason', None)!r}"
    )
    print(
        f"[{label}] source_window_pending={getattr(session, 'source_window_changed_pending', None)} "
        f"atomic_committed={session.atomic_source_successor_committed() if hasattr(session, 'atomic_source_successor_committed') else None} "
        f"residency_deferred={getattr(session, '_interactive_residency_deferred', None)} "
        f"prepared_atomic={getattr(session, '_atomic_prepared_transaction', None) is not None}"
    )
    view = getattr(win, "img_view", None)
    draw_pending = getattr(view, "presentationDrawPending", None)
    print(f"[{label}] presentationDrawPending={draw_pending() if callable(draw_pending) else None}")
    # Discriminator: does one manual retarget unstick the session?
    win.renderer.retarget_frame_pipeline(session)
    print(f"[{label}] after manual retarget: pending={len(session.pending_tiles)} "
          f"active={len(session.active_tile_requests)} "
          f"fanin_active={len(getattr(fan_in, 'active_requests', ()) or ())}")


def _build_fft_montage_window(qtbot):
    from tests.ui.helpers import make_backend_window, use_vispy_backend

    # Doctrine (docs/testing/stress-and-trace-strategy.md): every harness run
    # records a complete trace. The bounded watchdog ring only covers the
    # FIRST stall; the compound churn stalls need the full stream.
    from arrayscope.core.trace import configure_trace

    trace_path = Path(
        os.environ.get("ARRAYSCOPE_ARTIFACT_DIR", "/tmp")
    ) / f"arrayscope-churn-{os.getpid()}.trace.jsonl"
    configure_trace(trace_path)
    print(f"[harness] full trace: {trace_path}")

    settings = use_vispy_backend(extra_settings={"montage_quality_policy": "resident"})
    from arrayscope.io.file_interpreters import load_file
    from arrayscope.operations.pipeline import CenteredFFT, CenteredIFFT, FFTShift

    loaded = load_file(str(DATA_PATH))
    data = np.asarray(loaded.data if hasattr(loaded, "data") else loaded)

    win = make_backend_window(qtbot, data)
    win.resize(1200, 900)
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
    qtbot.waitUntil(lambda: _settled(win), timeout=INTERACTION_SETTLE_HARD_LIMIT_S * 1000)
    return win, settings, data, n


def test_montage_window_change_presents_mapped_complex_tiles(qtbot):
    """Field defect 2026-07-16 09:14: entering tiles of a montage window
    change presented resident complex floors WITHOUT their phase mapping and
    flashed PAL-relaxed LUT[0] orange until exact payloads replaced them
    (pre-fix probe: 54 orange frames, up to 19,300 orange pixels)."""

    from tests.ui.helpers import restore_default_backend

    win, settings, _data, n = _build_fft_montage_window(qtbot)
    try:
        for indices in (tuple(range(n // 3, n // 3 + 60)), tuple(range(n))):
            win._apply_slice_state(
                2,
                win.view_state.with_montage_axis(
                    2, columns=None, indices=indices,
                    text=f"{indices[0]}:{indices[-1] + 1}",
                ),
                reason="slice-range",
                interactive=True,
                immediate_axis_only=False,
            )
            deadline = time.monotonic() + INTERACTION_SETTLE_HARD_LIMIT_S
            while time.monotonic() < deadline and not _settled(win):
                _pump(qtbot, 0.08)
                orange = _orange_pixel_count(win)
                assert orange < _ORANGE_DEFECT_THRESHOLD, (
                    f"montage window change shows {orange} PAL-relaxed LUT[0] "
                    "orange pixels while filling (2026-07-16 floor-mapping "
                    "defect class)"
                )
            assert _settled(win)
    finally:
        win.close()
        restore_default_backend(settings)


def test_interaction_churn_converges_on_real_data(qtbot):
    """Closed 2026-07-16 (was an xfail net). Members 4 and 5 of the
    deferred-stage lost-wakeup family: the stage-plan/stage-value completions
    discarded session-current results on a stale render-generation stamp
    (discard/resubmit livelock, 5,200 plan computations per churn run), and
    the shared exact pass filtered out tiles holding non-exact payloads at
    the target level (38 tiles parked with open targets). Post-fix the churn
    converges in seconds, 3/3 runs. Dossier:
    docs/redesign/stale-empty-tiles-2026-07-16.md."""
    from tests.ui.helpers import restore_default_backend

    win, settings, data, n = _build_fft_montage_window(qtbot)
    try:

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
            orange = _orange_pixel_count(win)
            assert orange < _ORANGE_DEFECT_THRESHOLD, (
                f"frame during churn step {step} shows {orange} PAL-relaxed "
                "LUT[0] orange pixels: complex texels are being drawn without "
                "their phase mapping (2026-07-16 floor-mapping defect class)"
            )

        try:
            qtbot.waitUntil(lambda: _settled(win), timeout=INTERACTION_SETTLE_HARD_LIMIT_S * 1000)
        except Exception:
            _dump_convergence_state(win, "post-churn-timeout")
            raise

        session = win.renderer._frame_session
        assert session.required_target_unsettled_tiles() == ()
        assert not session.pending_payload_upserts
    finally:
        win.close()
        restore_default_backend(settings)
