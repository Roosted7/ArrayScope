import os
import time

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _process_events(qtbot, count=8):
    for _ in range(count):
        qtbot.wait(10)


def _clear_arrayscope_settings():
    from pyqtgraph.Qt import QtCore

    settings = QtCore.QSettings()
    settings.clear()
    settings.sync()


def test_nearby_slice_prefetch_uses_prefetch_state_keys(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win._active_slice_axis = 2
        win.renderer._schedule_prefetch_nearby_slices(win.view_state.with_slice(2, 2), None)
        _process_events(qtbot, count=40)

        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled > 0
        assert win.operation_evaluator.display_cache_diagnostics().prefetch_stored > 0
        assert win.operation_evaluator.cached_display_tile(win.view_state.with_slice(2, 1)) is not None
    finally:
        win.close()


def test_nearby_slice_prefetch_skips_when_setting_disabled(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._active_slice_axis = 2
        before = win.operation_evaluator.display_cache_diagnostics()
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 1), None)
        _process_events(qtbot, count=20)
        after = win.operation_evaluator.display_cache_diagnostics()

        assert after.prefetch_scheduled == before.prefetch_scheduled
        assert after.prefetch_skipped > before.prefetch_skipped
    finally:
        win.close()


def test_prefetch_dispatch_is_queued_but_not_timer_admitted(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win._active_slice_axis = 2
        win.renderer._schedule_prefetch_nearby_slices(win.view_state.with_slice(2, 2), None)
        assert not hasattr(win, "_prefetch_idle_timer")
        _process_events(qtbot, count=10)
        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled > 0
    finally:
        win.close()


def test_prefetch_limits_to_two_neighbors(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.core.prefetch_policy import SliceScrubMomentum
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 8, dtype=float).reshape(3, 4, 8))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win._active_slice_axis = 2
        momentum = SliceScrubMomentum()
        momentum.observe(2, now=0.0)
        win.renderer._prefetch_momentum = momentum
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 4), None)
        _process_events(qtbot, count=40)

        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled <= 2
    finally:
        win.close()


def test_prefetch_deepens_with_sustained_directional_scrub(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.core.prefetch_policy import SliceScrubMomentum
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 16, dtype=float).reshape(3, 4, 16))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win._active_slice_axis = 2
        momentum = SliceScrubMomentum()
        base = time.monotonic() - 0.3
        for step, index in enumerate((2, 3, 4, 5)):
            momentum.observe(index, now=base + 0.1 * step)
        win.renderer._prefetch_momentum = momentum
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 6), None)
        _process_events(qtbot, count=40)

        scheduled = win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled
        assert scheduled > 2, "sustained same-direction scrubbing should warm deeper ahead of the motion"
    finally:
        win.close()


def test_montage_prefetch_denial_does_not_cap_slice_prefetch(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.core.prefetch_policy import SliceScrubMomentum
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 16, dtype=float).reshape(3, 4, 16))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win._active_slice_axis = 2
        win.prefetch_evaluation_controller.set_max_prefetch(32)

        # With no montage stage ready, the montage-prefetch decision denies
        # montage speculation. That local decision must not shrink the shared
        # prefetch controller and accidentally throttle ordinary slice prefetch.
        win._apply_resource_governor_decisions()
        assert win.prefetch_evaluation_controller._max_prefetch == 32

        momentum = SliceScrubMomentum()
        base = time.monotonic() - 0.3
        for step, index in enumerate((2, 3, 4, 5)):
            momentum.observe(index, now=base + 0.1 * step)
        win.renderer._prefetch_momentum = momentum
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 6), None)
        _process_events(qtbot, count=40)

        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled > 2
    finally:
        win.close()


def test_montage_prefetch_completion_uses_real_orchestrator_staleness_guard(
    qtbot,
    monkeypatch,
):
    _clear_arrayscope_settings()
    from arrayscope.core.frame_targets import WorkStart
    from arrayscope.window import ArrayScopeWindow
    from arrayscope.window import montage_prefetch

    win = ArrayScopeWindow(np.arange(3 * 4 * 8, dtype=float).reshape(3, 4, 8))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.request_operation("reverse", 0)
        state = win.view_state.with_montage_axis(
            2,
            columns=4,
            indices=tuple(range(8)),
            text=":",
        )
        win._set_view_state(state)
        win.render(reason="test-prefetch-completion-guard")
        qtbot.waitUntil(
            lambda: win.renderer._frame_session is not None,
            timeout=5000,
        )
        session = win.renderer._frame_session
        assert session.document.enabled_operations

        captured = {}

        def capture_prefetch(_evaluate, *, on_done, **_kwargs):
            captured["done"] = on_done
            return WorkStart(True)

        monkeypatch.setattr(montage_prefetch, "_interaction_active", lambda _owner: False)
        monkeypatch.setattr(montage_prefetch, "_busy", lambda _owner, _session: False)
        monkeypatch.setattr(
            montage_prefetch,
            "_candidate_tiles",
            lambda _session: (session.plan.tiles[-1],),
        )
        monkeypatch.setattr(
            montage_prefetch,
            "_stage_for_tile",
            lambda _owner, _session, _tile: (object(), object(), object()),
        )
        monkeypatch.setattr(
            win.operation_evaluator,
            "cached_montage_tile",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            win.prefetch_evaluation_controller,
            "start_prefetch",
            capture_prefetch,
        )

        decisions = montage_prefetch.schedule_near_viewport_montage_prefetch(
            win.renderer,
            session,
            max_tiles=1,
        )
        assert decisions[0].decision == "scheduled"
        assert callable(captured.get("done"))

        win._set_view_state(win.view_state.with_slice(2, 1))
        win.render(reason="test-prefetch-completion-superseded")
        qtbot.waitUntil(
            lambda: win.renderer._frame_session is not session,
            timeout=5000,
        )
        stale_before = win.operation_evaluator.display_cache_diagnostics().prefetch_stale

        captured["done"](object())

        assert (
            win.operation_evaluator.display_cache_diagnostics().prefetch_stale
            == stale_before + 1
        )
    finally:
        win.close()


def test_prefetch_skips_while_visible_controller_busy(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.core.frame_targets import FrameTarget
    from arrayscope.kernel import Lane as WorkLane, WorkItem
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win._active_slice_axis = 2
        win.visible_evaluation_controller.start_latest(
            lambda: time.sleep(0.08),
            key="busy",
            priority=0,
            replace_group="visible",
            frame_target=FrameTarget("busy", None, "presentation", "exact-visible"),
            supersession_key="visible-image",
            supersession_value="busy",
            work_item=WorkItem(
                key=("visible", "busy"),
                lane=WorkLane.VISIBLE_MATERIALIZATION,
                frame_target=FrameTarget("busy", None, "presentation", "exact-visible"),
                supersession_key="visible-image",
                supersession_value="busy",
            ),
            on_done=lambda _value: None,
        )
        before_scheduled = win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 2), None)

        # The gate decision is synchronous: blocked, nothing scheduled, and
        # (post re-arm fix) the request is retained for the drain retry
        # rather than dropped.
        assert win.prefetch_evaluation_controller.diagnostics().prefetch_visible_busy_blocked >= 1
        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled == before_scheduled
        assert getattr(win.renderer, "_pending_prefetch_request", None) is not None
    finally:
        win.close()


def _pin_visible_busy(win):
    """Deterministically close the visible-busy gate like a live scrub does.

    In the field the gate closes because every scrub step starts a new
    visible evaluation milliseconds before the speculative dispatch lands;
    reproducing that timing with real work is racy offscreen (the kernel
    also parks the speculative batch behind real visible work), so pin the
    predicate itself and release it explicitly.
    """

    state = {"busy": True}
    original = win.visible_evaluation_controller.is_busy
    win.visible_evaluation_controller.is_busy = lambda: state["busy"]

    def release():
        state["busy"] = False
        win.visible_evaluation_controller.is_busy = original
        # The retained request re-arms on the next completion drain; give the
        # shared bridge one completed task to drain, exactly like the visible
        # frame whose completion would wake the retry in a live session.
        from arrayscope.core.frame_targets import FrameTarget
        from arrayscope.kernel import Lane as WorkLane, WorkItem

        win.visible_evaluation_controller.start_latest(
            lambda: None,
            key="drain",
            priority=0,
            replace_group="visible",
            frame_target=FrameTarget("drain", None, "presentation", "exact-visible"),
            supersession_key="visible-image",
            supersession_value="drain",
            work_item=WorkItem(
                key=("visible", "drain"),
                lane=WorkLane.VISIBLE_MATERIALIZATION,
                frame_target=FrameTarget("drain", None, "presentation", "exact-visible"),
                supersession_key="visible-image",
                supersession_value="drain",
            ),
            on_done=lambda _value: None,
        )

    return release


def test_prefetch_gated_by_busy_visible_runs_after_drain(qtbot):
    """A visible-busy dispatch retains the request and re-arms on drain.

    Regression gate: the speculative dispatch lands milliseconds after the
    slice change that armed it — while that change's own visible evaluation
    is still in flight — so the busy gate closed on ~every scrub step,
    CONSUMED the pending request, and never re-armed: 43/43 dispatches
    skipped, prefetch_scheduled=0 across a whole scrub session.
    """

    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win._active_slice_axis = 2
        # Let startup evaluation drain first so the speculative dispatch is
        # not parked behind real work under parallel test load.
        qtbot.waitUntil(
            lambda: not win.visible_evaluation_controller.is_busy(),
            timeout=10000,
        )
        release_busy = _pin_visible_busy(win)

        state = win.view_state.with_slice(2, 2)
        win.renderer._schedule_prefetch_nearby_slices(state, None)

        # The dispatch must actually hit the closed gate first...
        qtbot.waitUntil(
            lambda: win.prefetch_evaluation_controller.diagnostics().prefetch_visible_busy_blocked >= 1,
            timeout=10000,
        )
        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled == 0
        assert getattr(win.renderer, "_pending_prefetch_request", None) is not None, (
            "the gated dispatch must retain the request, not consume it"
        )

        # ...and the retained request must run once the visible work drains.
        release_busy()
        qtbot.waitUntil(
            lambda: win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled >= 1,
            timeout=5000,
        )
        qtbot.waitUntil(
            lambda: win.operation_evaluator.cached_display_tile(win.view_state.with_slice(2, 1)) is not None
            or win.operation_evaluator.cached_display_tile(win.view_state.with_slice(2, 3)) is not None,
            timeout=5000,
        )
    finally:
        win.close()


def test_prefetch_burst_coalesces_and_momentum_observes_despite_gating(qtbot):
    """Rapid gated scrub steps collapse to one latest-wins retry.

    Momentum observes at SCHEDULE time, so the gated steps still feed the
    direction model (pre-fix: observe only ran inside the gated body, so a
    busy scrub session left momentum with no data at all).
    """

    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 16, dtype=float).reshape(3, 4, 16))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win._active_slice_axis = 2
        qtbot.waitUntil(
            lambda: not win.visible_evaluation_controller.is_busy(),
            timeout=10000,
        )
        release_busy = _pin_visible_busy(win)

        for index in (2, 3, 4, 5):
            win.renderer._schedule_prefetch_nearby_slices(win.view_state.with_slice(2, index), None)
        _process_events(qtbot, count=10)

        momentum = getattr(win.renderer, "_prefetch_momentum", None)
        assert momentum is not None, "gated scrub steps must still feed the momentum model"
        assert momentum.direction == 1
        assert momentum.streak >= 3
        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled == 0

        release_busy()
        qtbot.waitUntil(
            lambda: win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled >= 1,
            timeout=5000,
        )
        _process_events(qtbot, count=40)
        diagnostics = win.operation_evaluator.display_cache_diagnostics()
        # Latest-wins coalescing: one retained request, one momentum-planned
        # fan-out (max_depth=4) — not one prefetch per gated step.
        assert diagnostics.prefetch_scheduled <= 4
        assert getattr(win.renderer, "_pending_prefetch_request", None) is None
    finally:
        win.close()


def test_cost_aware_prefetch_allows_cheap_operation_backed_stack(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win.request_operation("reverse", 0)
        _process_events(qtbot, count=20)
        win._active_slice_axis = 2
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 2), None)
        _process_events(qtbot, count=40)
        after = win.operation_evaluator.display_cache_diagnostics()

        assert after.prefetch_scheduled > 0
        assert after.prefetch_stored > 0
    finally:
        win.close()


def test_cost_aware_prefetch_blocks_expensive_fft_stack(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((96, 96, 96), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True, render_memory_budget_mb=128)
        win.request_operation("centered_fft", 2)
        _process_events(qtbot, count=20)
        win._active_slice_axis = 2
        before = win.operation_evaluator.display_cache_diagnostics()
        before_blocked = win.prefetch_evaluation_controller.diagnostics().prefetch_cost_blocked
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 1), None)
        _process_events(qtbot, count=20)
        after = win.operation_evaluator.display_cache_diagnostics()

        assert after.prefetch_scheduled == before.prefetch_scheduled
        assert win.prefetch_evaluation_controller.diagnostics().prefetch_cost_blocked > before_blocked
    finally:
        win.close()


def test_slice_change_schedules_nearby_prefetch_when_enabled(qtbot):
    """The opt-in slice prefetcher rides every live slice change.

    Regression guard: the scheduler call was severed when the legacy
    normal-image update path was deleted, leaving the setting silently dead.
    """

    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 4 * 6, dtype=np.float32).reshape(4, 4, 6))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(
            theme=win.app_settings.theme, prefetch_nearby_slices=True
        )
        before = win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled
        win._apply_slice_state(
            2,
            win.view_state.with_slice(2, 3),
            reason="test-slice-prefetch-wiring",
            interactive=True,
            immediate_axis_only=True,
        )
        qtbot.waitUntil(
            lambda: win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled > before,
            timeout=5000,
        )
    finally:
        win.close()


def test_slice_change_without_setting_skips_prefetch(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 4 * 6, dtype=np.float32).reshape(4, 4, 6))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        before = win.operation_evaluator.display_cache_diagnostics().prefetch_skipped
        win._apply_slice_state(
            2,
            win.view_state.with_slice(2, 2),
            reason="test-slice-prefetch-off",
            interactive=True,
            immediate_axis_only=True,
        )
        qtbot.waitUntil(
            lambda: win.operation_evaluator.display_cache_diagnostics().prefetch_skipped > before,
            timeout=5000,
        )
    finally:
        win.close()


def test_prefetch_menu_action_exists_and_toggles_setting(qtbot):
    """The Performance-menu toggle must stay wired to the live setting.

    Regression guard: the handler survived the legacy-path removal but the
    menu action itself was lost, leaving the setting unreachable in the UI.
    """

    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        action = getattr(win, "_prefetch_nearby_slices_action", None)
        assert action is not None, "Performance menu lost the prefetch toggle"
        assert action.isCheckable()
        assert not action.isChecked()
        assert not win.app_settings.prefetch_nearby_slices
        action.setChecked(True)
        _process_events(qtbot)
        assert win.app_settings.prefetch_nearby_slices
        action.setChecked(False)
        _process_events(qtbot)
        assert not win.app_settings.prefetch_nearby_slices
    finally:
        win.close()


def test_prefetch_menu_action_exists_and_toggles_setting(qtbot):
    """The Performance-menu toggle must stay wired to the live setting.

    Regression guard: the handler survived the legacy-path removal but the
    menu action itself was lost, leaving the setting unreachable in the UI.
    """

    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        action = getattr(win, "_prefetch_nearby_slices_action", None)
        assert action is not None, "Performance menu lost the prefetch toggle"
        assert action.isCheckable()
        assert not action.isChecked()
        assert not win.app_settings.prefetch_nearby_slices
        action.setChecked(True)
        _process_events(qtbot)
        assert win.app_settings.prefetch_nearby_slices
        action.setChecked(False)
        _process_events(qtbot)
        assert not win.app_settings.prefetch_nearby_slices
    finally:
        win.close()
