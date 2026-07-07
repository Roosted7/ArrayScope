import time

import numpy as np
import pytest

from tests.ui.helpers import (
    assert_panel_invariants as _assert_panel_invariants,
    assert_size_close as _assert_size_close,
    clear_arrayscope_settings as _clear_arrayscope_settings,
    panel_body as _panel_body,
    process_events as _process_events,
    view_action as _view_action,
    wait_for_panel_preserve as _wait_for_panel_preserve,
)


def test_over_budget_view_skips_tiles_without_clearing_previous_image(qtbot, monkeypatch):
    """Memory protection now happens through the tiled render budgets.

    A view whose tiles exceed the visible render budget must be skipped with a
    warning while the previously committed presentation stays on screen.
    """

    _clear_arrayscope_settings()
    from dataclasses import replace as dataclass_replace

    from pyqtgraph.Qt import QtWidgets
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    warnings = []
    try:
        _process_events(qtbot, count=20)
        qtbot.waitUntil(lambda: getattr(win, "_committed_display_frame", None) is not None, timeout=3000)
        qtbot.waitUntil(lambda: not win.montage_tile_evaluation_controller.is_busy(), timeout=3000)
        previous_image = win.img_view.image.copy()
        previous_frame = win._committed_display_frame
        win.operation_evaluator.clear_cache()
        # Force a fresh session so the budget decision is actually re-evaluated.
        win.renderer._montage_session = None
        tiny_policy = dataclass_replace(win.renderer._memory_policy(), single_tile_budget_bytes=1)
        monkeypatch.setattr(win.renderer, "_refresh_memory_policy", lambda *args, **kwargs: tiny_policy)
        monkeypatch.setattr(win.renderer, "_memory_policy", lambda *args, **kwargs: tiny_policy)
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda _parent, _title, message: warnings.append(message))

        win.update_image_view()
        _process_events(qtbot, count=10)

        np.testing.assert_array_equal(win.img_view.image, previous_image)
        assert win._committed_display_frame is previous_frame
        assert win._montage_session.skipped_tiles
        assert warnings
        assert "over the visible render budget" in warnings[0]
    finally:
        win.close()


def test_visible_render_cancels_prefetch_queue(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(16 * 16 * 3, dtype=float).reshape(16, 16, 3))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        started = win.prefetch_evaluation_controller.start_prefetch(lambda: time.sleep(0.05), key="prefetch")
        assert started.scheduled
        win.operation_evaluator.clear_cache()
        win.update_image_view()
        _process_events(qtbot, count=5)

        assert not win.prefetch_evaluation_controller._prefetch_keys
    finally:
        win.close()


def test_prefetch_never_runs_during_montage(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win._active_slice_axis = 2
        state = win.view_state.with_montage_axis(2, indices=(0, 1, 2), text=":")
        before = win.operation_evaluator.display_cache_diagnostics()
        win.renderer._prefetch_nearby_slices(state, None)
        _process_events(qtbot, count=20)
        after = win.operation_evaluator.display_cache_diagnostics()

        assert after.prefetch_scheduled == before.prefetch_scheduled
        assert after.prefetch_skipped > before.prefetch_skipped
    finally:
        win.close()


def test_compute_policy_configures_stage_and_montage_lanes(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.core.compute_policy import ComputeLane
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        montage_workers = win.montage_tile_evaluation_controller.diagnostics().max_workers
        assert 1 <= montage_workers <= win.compute_policy.workers_for_lane(ComputeLane.MONTAGE_TILE)
        assert win.stage_evaluation_controller.diagnostics().max_workers == win.compute_policy.workers_for_lane(ComputeLane.STAGE)
        assert win.histogram_evaluation_controller.diagnostics().max_workers == win.compute_policy.workers_for_lane(ComputeLane.HISTOGRAM)
        assert win.compute_policy.fft_workers_for_lane(ComputeLane.MONTAGE_TILE) == 1
        assert win.compute_policy.fft_workers_for_lane(ComputeLane.STAGE) >= 1
        assert win.compute_policy.fft_workers_for_lane(ComputeLane.HISTOGRAM) == 1
    finally:
        win.close()


def test_histogram_background_work_uses_histogram_priority_not_prefetch(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.core.scheduler import EvalPriority, WorkStart
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    histogram_calls = []
    prefetch_calls = []

    def start_histogram(fn, **kwargs):
        histogram_calls.append(kwargs)
        return 1

    monkeypatch.setattr(win.histogram_evaluation_controller, "start_active_plus_latest", start_histogram)
    monkeypatch.setattr(
        win.prefetch_evaluation_controller,
        "start_prefetch",
        lambda *args, **kwargs: prefetch_calls.append(kwargs) or WorkStart(False, "wrong-controller"),
    )
    try:
        result = win._submit_histogram_background_task(lambda: "hist", on_done=lambda _value: None, key=("histogram_plot", "source"))

        assert result.scheduled
        assert histogram_calls
        assert histogram_calls[-1]["priority"] == EvalPriority.HISTOGRAM
        assert histogram_calls[-1]["replace_group"] == "histogram-plot"
        assert callable(histogram_calls[-1]["on_reuse_stale"])
        assert prefetch_calls == []
    finally:
        win.close()


def test_stale_tile_result_does_not_clear_updating_overlay(qtbot):
    """A result for a superseded render must not clear the updating overlay.

    Visible rendering flows through the montage tile lane; the stale path is
    the tile-done callback of a superseded session.
    """

    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9 * 10, dtype=float).reshape(8, 9, 10))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        qtbot.waitUntil(lambda: not win.montage_tile_evaluation_controller.is_busy(), timeout=3000)
        win.operation_evaluator.clear_cache()
        win.renderer._retained_tiled_payload_store().clear_for_document_or_context_change("test-cold-start")
        win.renderer._montage_session = None
        frame = getattr(win, "_committed_display_frame", None)
        payloads = getattr(getattr(frame, "value_source", None), "payloads", None)
        if isinstance(payloads, dict):
            payloads.clear()
        win.update_image_view()
        stale_session = win._montage_session
        win.renderer.show_montage_session_slow_overlay(stale_session)
        assert win.img_view._evaluation_overlay is not None
        assert win.img_view._evaluation_overlay.isVisible()

        # Supersede the render, then let the old work report slow/stale.
        win._set_view_state(win.view_state.with_slice(2, 2))
        win.update_image_view()
        win.img_view.setEvaluationOverlay(True, "Updating image frame...")
        win.renderer.show_montage_session_slow_overlay(stale_session)
        win.renderer._settle_montage_visible_plan_if_complete(stale_session)

        assert win.img_view._evaluation_overlay.isVisible()
    finally:
        win.close()
