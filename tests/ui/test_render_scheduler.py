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


def test_visible_render_controller_uses_latest_only_group(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9 * 10, dtype=float).reshape(8, 9, 10))
    qtbot.addWidget(win)
    calls = []
    original_start_latest = win.visible_evaluation_controller.start_latest

    def recording_start_latest(fn, **kwargs):
        calls.append(kwargs.get("replace_group"))

        def slow_fn(*args):
            time.sleep(0.02)
            return fn(*args)

        return original_start_latest(slow_fn, **kwargs)

    monkeypatch.setattr(win.visible_evaluation_controller, "start_latest", recording_start_latest)
    try:
        _process_events(qtbot, count=20)
        win._set_view_state(win.view_state.with_slice(2, 1))
        win.render(reason="test-1")
        win._set_view_state(win.view_state.with_slice(2, 2))
        win.render(reason="test-2")
        _process_events(qtbot, count=40)

        assert "visible-image" in calls
        assert win.visible_evaluation_controller.pool.maxThreadCount() == 1
    finally:
        win.close()


def test_visible_render_uses_cost_decision_refuse_without_clearing_previous_image(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    import arrayscope.window.render as render_module
    from arrayscope.operations.render_plan import RenderDecision, RenderDecisionKind
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        previous = win.img_view.image.copy()
        win.operation_evaluator.clear_cache()
        monkeypatch.setattr(
            render_module,
            "choose_visible_render_decision",
            lambda _context: RenderDecision(
                RenderDecisionKind.REFUSE,
                "test refuse",
                should_show_overlay=True,
                overlay_text="Exact view over budget",
                status_text="refused",
            ),
        )

        win.update_image_view()
        _process_events(qtbot, count=5)

        np.testing.assert_array_equal(win.img_view.image, previous)
        assert win.operation_evaluator.image_cache_diagnostics().refused_evaluations == 1
    finally:
        win.close()


def test_degraded_preview_commits_with_overlay_and_not_exact_cache(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    import arrayscope.window.render as render_module
    from arrayscope.operations.render_plan import RenderDecision, RenderDecisionKind
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(16 * 16, dtype=float).reshape(16, 16))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win.operation_evaluator.clear_cache()
        monkeypatch.setattr(
            render_module,
            "choose_visible_render_decision",
            lambda _context: RenderDecision(
                RenderDecisionKind.DEGRADED_PREVIEW,
                "preview",
                should_show_overlay=True,
                overlay_text="Preview only: exact view exceeds budget",
                status_text="preview",
                degraded_factor=2,
            ),
        )

        win.update_image_view()
        qtbot.waitUntil(lambda: getattr(win, "_last_render_was_degraded", False), timeout=3000)

        assert win.img_view._evaluation_overlay.isVisible()
        assert win.operation_evaluator.cached_image(win.view_state) is None
    finally:
        win.close()


def test_next_exact_render_clears_degraded_overlay(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    import arrayscope.window.render as render_module
    from arrayscope.operations.render_plan import RenderDecision, RenderDecisionKind
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(16 * 16, dtype=float).reshape(16, 16))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win.operation_evaluator.clear_cache()
        monkeypatch.setattr(
            render_module,
            "choose_visible_render_decision",
            lambda _context: RenderDecision(RenderDecisionKind.DEGRADED_PREVIEW, "preview", overlay_text="Preview only", degraded_factor=2),
        )
        win.update_image_view()
        qtbot.waitUntil(lambda: getattr(win, "_last_render_was_degraded", False), timeout=3000)
        win.operation_evaluator.clear_cache()
        monkeypatch.setattr(
            render_module,
            "choose_visible_render_decision",
            lambda _context: RenderDecision(RenderDecisionKind.ASYNC_EXACT, "exact"),
        )

        win.update_image_view()
        qtbot.waitUntil(lambda: not getattr(win, "_last_render_was_degraded", False), timeout=3000)

        assert not win.img_view._evaluation_overlay.isVisible()
    finally:
        win.close()


def test_chunked_visible_render_commits_exact_result(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    import arrayscope.window.render as render_module
    from arrayscope.operations.render_plan import RenderDecision, RenderDecisionKind
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(16 * 16, dtype=float).reshape(16, 16))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win.operation_evaluator.clear_cache()
        monkeypatch.setattr(
            render_module,
            "choose_visible_render_decision",
            lambda _context: RenderDecision(RenderDecisionKind.ASYNC_CHUNKED, "chunk", chunk_axis=0, chunk_size=4),
        )

        win.update_image_view()
        qtbot.waitUntil(lambda: win.operation_evaluator.cached_image(win.view_state) is not None, timeout=3000)

        assert win.operation_evaluator.image_cache_diagnostics().chunked_evaluations >= 1
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
        before = win.operation_evaluator.image_cache_diagnostics()
        win._prefetch_nearby_slices(state, None)
        _process_events(qtbot, count=20)
        after = win.operation_evaluator.image_cache_diagnostics()

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
        assert win.montage_tile_evaluation_controller.pool.maxThreadCount() == win.compute_policy.workers_for_lane(ComputeLane.MONTAGE_TILE)
        assert win.stage_evaluation_controller.pool.maxThreadCount() == win.compute_policy.workers_for_lane(ComputeLane.STAGE)
        assert win.compute_policy.fft_workers_for_lane(ComputeLane.MONTAGE_TILE) == 1
        assert win.compute_policy.fft_workers_for_lane(ComputeLane.STAGE) >= 1
    finally:
        win.close()


def test_stale_visible_result_does_not_clear_updating_overlay(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9 * 10, dtype=float).reshape(8, 9, 10))
    qtbot.addWidget(win)
    captured = {}

    def capture_start_latest(_fn, **kwargs):
        captured.update(kwargs)
        kwargs["on_slow"]()
        return 1

    monkeypatch.setattr(win.visible_evaluation_controller, "start_latest", capture_start_latest)
    try:
        _process_events(qtbot, count=20)
        win.operation_evaluator.clear_cache()
        win.update_image_view()
        assert win.img_view._evaluation_overlay is not None
        assert win.img_view._evaluation_overlay.isVisible()

        captured["on_stale"]()
        assert win.img_view._evaluation_overlay.isVisible()
    finally:
        win.close()
