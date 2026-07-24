"""Offscreen WGPU proof for refined single-slice level evidence.

The semantic evidence owner used to require ``montage_axis`` even though the
tiled frame pipeline also owns ordinary one-tile image sessions. Those
sessions published rough resident evidence, then stayed there forever.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytestqt.exceptions import TimeoutError as QtBotTimeoutError

from arrayscope.display.model.montage_levels import (
    PROVISIONAL_TILE_SAMPLE_LIMIT,
    LevelEvidenceQuality,
)
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import (
    frame_session_settled,
    make_backend_window,
    restore_default_backend,
    use_wgpu_backend,
)


def test_wgpu_single_slice_reaches_refined_levels_and_histogram(qtbot):
    pytest.importorskip("wgpu")
    settings = use_wgpu_backend(extra_settings={"montage_quality_policy": "resident"})
    yy, xx = np.mgrid[0:96, 0:128].astype(np.float32)
    planes = tuple((index + 1.0) * yy + (3.0 - index) * xx for index in range(3))
    data = np.stack(planes, axis=2).astype(np.float32)
    win = make_backend_window(qtbot, data, backend="wgpu", require_gpu_atlas=True)
    try:
        win.resize(720, 600)
        win.show()
        qtbot.waitExposed(win)
        win._set_view_state(win.view_state.with_image_axes(0, 1).with_slice(2, 1))
        win.render(reason="test-wgpu-single-slice-refined-evidence")

        def refined_published() -> bool:
            session = getattr(win.renderer, "_frame_session", None)
            if session is None or not frame_session_settled(win):
                return False
            summary = win.renderer._montage_level_tracker().summary_for(session.level_key)
            source = getattr(session, "applied_level_source", None)
            histogram = getattr(win.img_view, "histogramPlotSource", None)
            return bool(
                summary is not None
                and summary.evidence_quality == LevelEvidenceQuality.REFINED
                and bool(summary.refined)
                and int(getattr(source, "evidence_quality", 0) or 0)
                == int(LevelEvidenceQuality.REFINED)
                and histogram is not None
                and np.asarray(histogram).size > PROVISIONAL_TILE_SAMPLE_LIMIT
            )

        try:
            qtbot.waitUntil(refined_published, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        except QtBotTimeoutError:
            session = getattr(win.renderer, "_frame_session", None)
            summary = (
                None
                if session is None
                else win.renderer._montage_level_tracker().summary_for(session.level_key)
            )
            cached_histogram = (
                None
                if session is None
                else win.renderer._montage_level_tracker().cached_histogram_data(session.level_key)
            )
            histogram = getattr(win.img_view, "histogramPlotSource", None)
            pytest.fail(
                "single-slice refinement did not publish: "
                f"settled={frame_session_settled(win)} "
                f"diagnostics={None if session is None else session.semantic_level_evidence_diagnostics()} "
                f"summary={summary!r} "
                f"applied={None if session is None else session.applied_level_source!r} "
                f"histogram_size={None if histogram is None else np.asarray(histogram).size} "
                f"cached_histogram_size="
                f"{None if cached_histogram is None else np.asarray(cached_histogram).size} "
                f"histogram_pending="
                f"{None if session is None else session.histogram_metadata_pending} "
                f"histogram_inflight="
                f"{None if session is None else session.histogram_aggregate_inflight} "
                f"schedule_verdict="
                f"{None if session is None else session.scheduling_policy.verdict!r} "
                f"level_decision={getattr(win.renderer, '_last_montage_level_decision', None)!r}"
            )

        session = win.renderer._frame_session
        summary = win.renderer._montage_level_tracker().summary_for(session.level_key)
        diagnostics = session.semantic_level_evidence_diagnostics()
        source = session.applied_level_source
        assert session.montage_axis is None
        assert diagnostics["target_population"] == 1
        assert diagnostics["covered_source_count"] == 1
        assert diagnostics["blocking_reason"] == "ready"
        assert summary.source_indices == frozenset({0})
        assert source.source_count == source.expected_count == 1
        assert win.img_view.getLevels() == pytest.approx(source.levels)
        assert np.asarray(win.img_view.histogramPlotSource).size <= 8192
    finally:
        win.close()
        restore_default_backend(settings)
