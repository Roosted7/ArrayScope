"""Montage index-window scroll must settle to target quality both directions.

Regression net for the trace-proven idle stall (2026-07-15,
``/tmp/arrayscope-stall-18-1.trace.jsonl`` seq 9703-9708 and
``/tmp/arrayscope-stall-65-2.trace.jsonl`` seq 37576-37585, decoded in
``docs/redesign/coverage-stall-2026-07-15.md``): after an index-window
retarget the shared-transform DESIRED pass is barred behind the reset
``first_pass_histogram_published`` flag, and when the rough level evidence
for the newly scrolled sources completed AFTER the last acknowledgement
commit, no hook remained to publish it — a closed wait cycle with an idle
kernel and ``required_target_unsettled_tiles()`` non-empty until the
watchdog asserted.  The deterministic half of the fix is unit-tested in
``tests/window/test_montage_backend.py`` (late first-pass evidence arms the
publication flush); this test drives the live shape offscreen: an
FFT-over-montage-axis pipeline (shared-transform owner) on the VisPy shader
backend, scrolled down and back up, must reach ``required_target_settled``
in both directions with zero watchdog stall assertions.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pytest

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import (
    frame_session_settled,
    make_backend_window,
    restore_default_backend,
    use_pyqtgraph_backend,
    use_vispy_backend,
    use_wgpu_backend,
)


def _window_settled(win, indices) -> bool:
    session = getattr(win.renderer, "_frame_session", None)
    if session is None or session.plan is None:
        return False
    plan_sources = tuple(int(tile.source_index) for tile in session.plan.tiles)
    if plan_sources != tuple(indices):
        return False
    return frame_session_settled(win)


def _window_first_pixels_presented(win, indices) -> bool:
    session = getattr(win.renderer, "_frame_session", None)
    if session is None or session.plan is None:
        return False
    plan_sources = tuple(int(tile.source_index) for tile in session.plan.tiles)
    return bool(plan_sources == tuple(indices) and session.required_first_pixels_presented())


def _scroll_to(win, qtbot, indices) -> None:
    state = win.view_state.with_montage_axis(
        2,
        columns=5,
        indices=tuple(indices),
        text=f"{indices[0]}:{indices[-1] + 1}",
    )
    win._set_view_state(state)
    win.update_image_view()
    qtbot.waitUntil(lambda: _window_settled(win, indices), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)


def test_fft_montage_scroll_down_then_up_settles_required_target(qtbot):
    settings = use_vispy_backend(extra_settings={"montage_quality_policy": "resident"})
    from arrayscope.operations.pipeline import CenteredFFT

    rng = np.random.default_rng(20260715)
    data = rng.standard_normal((96, 96, 36), dtype=np.float32)

    win = make_backend_window(qtbot, data)
    win.resize(520, 420)
    try:
        win.show()
        # FFT over the montage axis: non-commuting for display LOD, so the
        # shared-transform fanout owns every tile's producer — the exact
        # scheduling shape of the stalled traces.
        win.operation_coordinator.load_operations((CenteredFFT(axis=2),))
        win._set_document(win.operation_coordinator.document)
        win._coerce_channel_for_current_dtype()

        initial = tuple(range(20))
        state = win.view_state.with_montage_axis(
            2,
            columns=5,
            indices=initial,
            text="0:20",
        )
        win._set_view_state(state)
        win.update_image_view()
        qtbot.waitUntil(
            lambda: _window_settled(win, initial), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS
        )

        # Scroll down (new sources appended), then back up (new sources
        # prepended) — both retarget directions of the live repro.
        _scroll_to(win, qtbot, tuple(range(6, 26)))
        _scroll_to(win, qtbot, tuple(range(2, 22)))

        session = win.renderer._frame_session
        assert session.required_target_unsettled_tiles() == ()
        # The watchdog must never have asserted: the kernel may not sit idle
        # while required tiles are unsettled (dossier exit gate).
        assert int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0) == 0
    finally:
        win.close()
        restore_default_backend(settings)


def test_wgpu_scalar_scroll_back_settles_retained_fallbacks_to_exact(qtbot):
    """Offscreen ring-1 pin for the 2026-07-18 fallback-forever livelock.

    A scroll back reuses a complete set of physically presented wgpu floor
    payloads after ``retarget_index_window`` resets first-pass evidence.  Those
    fallback pixels must remain visible, but they must neither acknowledge the
    exact target nor consume its producer.  Before the fix the retained report
    left ``first_pass_quality`` unset, so ``coverage_evidence_ready`` never
    fired and all target tiles parked with an idle kernel.
    """

    settings = use_wgpu_backend(extra_settings={"montage_quality_policy": "resident"})
    slice_values = np.arange(104, dtype=np.float32)
    data = np.broadcast_to(slice_values, (336, 336, 104))

    win = make_backend_window(qtbot, data, backend="wgpu", require_gpu_atlas=True)
    win.resize(1200, 900)
    try:
        win.show()
        initial = tuple(range(40, 100))
        state = win.view_state.with_montage_axis(
            2,
            columns=7,
            indices=initial,
            text="40:100",
        )
        win._set_view_state(state)
        win.update_image_view()
        qtbot.waitUntil(
            lambda: _window_first_pixels_presented(win, initial),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )

        for indices in (tuple(range(41, 101)), initial):
            state = win.view_state.with_montage_axis(
                2,
                columns=7,
                indices=indices,
                text=f"{indices[0]}:{indices[-1] + 1}",
            )
            win._set_view_state(state)
            win.update_image_view()
            qtbot.waitUntil(
                lambda indices=indices: _window_first_pixels_presented(win, indices),
                timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
            )

        qtbot.waitUntil(
            lambda: _window_settled(win, initial),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )

        session = win.renderer._frame_session
        assert session.required_target_unsettled_tiles() == ()
        assert int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0) == 0
    finally:
        win.close()
        restore_default_backend(settings)


@pytest.mark.parametrize("backend", ["wgpu", "pyqtgraph"])
def test_cropped_display_axis_scroll_keeps_complete_montage(qtbot, backend):
    """A rapid displayed-axis crop retarget must retain all 50 montage tiles."""

    configure = use_wgpu_backend if backend == "wgpu" else use_pyqtgraph_backend
    settings = configure(extra_settings={"montage_quality_policy": "resident"})
    data = np.broadcast_to(
        np.arange(336 * 336, dtype=np.float32).reshape(336, 336, 1),
        (336, 336, 50),
    )
    win = make_backend_window(
        qtbot,
        data,
        backend=backend,
        require_gpu_atlas=backend == "wgpu",
    )
    win.resize(1200, 900)
    try:
        win.show()
        montage_indices = tuple(range(50))
        state = (
            win.view_state.with_image_axes(1, 0)
            .with_axis_flipped(1, True)
            .with_montage_axis(2, columns=10, indices=montage_indices, text=":")
        )
        win._set_view_state(state)
        win.update_image_view()
        qtbot.waitUntil(
            lambda: _window_settled(win, montage_indices),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        full_uploads = (
            int(win.img_view.wgpuPresentationDiagnostics()["wgpu_uploads_total"])
            if backend == "wgpu"
            else None
        )
        full_global_l0 = (
            sum(
                key.document_generation[0] == "wgpu-source-plane" and int(key.lod.level) == 0
                for key in win.img_view._wgpu_executor.page_table.resident_keys()
            )
            if backend == "wgpu"
            else 0
        )

        initial_indices = tuple(range(97, 197))
        uncropped_session_id = int(win.renderer._frame_session.session_id)
        win._apply_slice_state(
            0,
            win.view_state.with_axis_range(
                0,
                indices=initial_indices,
                text="97:197",
            ),
            reason="slice-range",
            interactive=True,
            immediate_axis_only=False,
        )
        qtbot.waitUntil(
            lambda: (
                int(win.renderer._frame_session.session_id) != uncropped_session_id
                and _window_settled(win, montage_indices)
            ),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        crop_uploads = (
            int(win.img_view.wgpuPresentationDiagnostics()["wgpu_uploads_total"])
            if backend == "wgpu"
            else None
        )
        physical_tile_counts = [len(win.img_view.tileTruthPhysicalRows())]

        # Start three one-pixel successor generations without waiting for the
        # preceding generation to settle. Waiting only for a new FrameSession
        # keeps this deterministic while reproducing the attached scrollbar
        # trace's overlapping hidden atomic handoffs.
        for start in (96, 95, 94):
            previous_session_id = int(win.renderer._frame_session.session_id)
            indices = tuple(range(start, start + 100))
            win._apply_slice_state(
                0,
                win.view_state.with_axis_range(
                    0,
                    indices=indices,
                    text=f"{start}:{start + 100}",
                ),
                reason="slice-range",
                interactive=True,
                immediate_axis_only=False,
            )
            qtbot.waitUntil(
                lambda previous_session_id=previous_session_id: (
                    int(win.renderer._frame_session.session_id) != previous_session_id
                ),
                timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
            )
            cadence_stop_at = perf_counter() + 0.12
            while perf_counter() < cadence_stop_at:
                qtbot.wait(1)
                physical_tile_counts.append(len(win.img_view.tileTruthPhysicalRows()))
        try:

            def cropped_target_settled():
                session = win.renderer._frame_session
                return (
                    _window_settled(win, montage_indices)
                    and not session.atomic_successor_pending
                    and set(win.img_view.tileTruthPhysicalRows()) == set(montage_indices)
                )

            qtbot.waitUntil(
                cropped_target_settled,
                timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
            )
        except Exception:
            session = win.renderer._frame_session
            report = win.renderer._display_committer().last_tile_commit_report
            pytest.fail(
                "cropped-axis montage did not settle: "
                f"backend={backend}, "
                f"unsettled={session.required_target_unsettled_tiles()}, "
                f"first_pixels={session.required_first_pixels_presented()}, "
                f"busy={win.montage_tile_evaluation_controller.is_busy()}, "
                f"physical={sorted(win.img_view.tileTruthPhysicalRows())}, "
                f"dirty={sorted(session.dirty_payloads)}, "
                f"pending={sorted(session.pending_payload_upserts)}, "
                f"draw_pending={win.img_view.presentationDrawPending()}, "
                f"mode={win.img_view.montageDisplayMode()}, "
                f"report_presented={getattr(win.renderer, '_last_montage_report_presented', None)}, "
                f"report_committed={getattr(win.renderer, '_last_montage_report_committed', None)}, "
                f"report_stale={getattr(win.renderer, '_last_montage_report_stale', None)}, "
                f"atomic_pending={session.atomic_successor_pending}, "
                f"warmed={len(getattr(session, '_atomic_warmed_identities', ()))}, "
                f"identity_rejected={getattr(report, 'identity_rejected_tiles', None)}, "
                f"atomic_fast_reject={getattr(session, '_atomic_fast_reject_reason', None)}, "
                f"outcome={getattr(win.renderer, '_last_montage_commit_outcome', None)}"
            )
        session = win.renderer._frame_session
        assert session.required_target_unsettled_tiles() == ()
        assert session.atomic_successor_pending is False
        assert min(physical_tile_counts) == len(montage_indices)
        assert set(win.img_view.tileTruthPhysicalRows()) == set(montage_indices)
        assert int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0) == 0
        uploads_before_axis_swap = (
            int(win.img_view.wgpuPresentationDiagnostics()["wgpu_uploads_total"])
            if backend == "wgpu"
            else None
        )
        for image_axes in ((0, 1), (1, 0)):
            previous_session_id = int(win.renderer._frame_session.session_id)
            win._set_view_state(win.view_state.with_image_axes(*image_axes))
            win.update_image_view()
            qtbot.waitUntil(
                lambda previous_session_id=previous_session_id: (
                    int(win.renderer._frame_session.session_id) != previous_session_id
                    and _window_settled(win, montage_indices)
                ),
                timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
            )
            assert set(win.img_view.tileTruthPhysicalRows()) == set(montage_indices)
            assert int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0) == 0
        if backend == "wgpu":
            final = win.img_view.wgpuPresentationDiagnostics()
            assert full_global_l0 == 4 * len(montage_indices)
            assert crop_uploads == full_uploads
            assert uploads_before_axis_swap == crop_uploads
            assert int(final["wgpu_uploads_total"]) == uploads_before_axis_swap
            assert final["wgpu_last_pool_exhaustion"] == ""
    finally:
        win.close()
        restore_default_backend(settings)
