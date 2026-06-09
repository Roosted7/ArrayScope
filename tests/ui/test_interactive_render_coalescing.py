import numpy as np

from tests.ui.helpers import clear_arrayscope_settings as _clear_arrayscope_settings, process_events as _process_events


def test_slice_text_updates_immediately_while_render_is_coalesced(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=float).reshape(4, 5, 8))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win, "render", lambda **kwargs: calls.append((kwargs, win.view_state.slice_indices[2])))
    try:
        _process_events(qtbot, count=2)
        win._on_slice_index_changed(2, 3)

        assert win.view_state.slice_indices[2] == 3
        assert win.dimension_strip.chip(2).slice_edit.text() == "3"
        assert win.widgets["spins"]["slice_indices"][2].value() == 3
        assert calls == []

        _process_events(qtbot, count=3)
        assert len(calls) == 1
        assert calls[0][0]["reason"] == "slice"
        assert calls[0][0]["defer_side_panels"] is True
        assert calls[0][1] == 3
    finally:
        win.close()


def test_rapid_slice_burst_renders_only_latest_state(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=float).reshape(4, 5, 8))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win, "render", lambda **kwargs: calls.append((kwargs["reason"], win.view_state.slice_indices[2])))
    try:
        _process_events(qtbot, count=2)
        win._on_slice_index_changed(2, 1)
        win._on_slice_index_changed(2, 2)
        win._on_slice_index_changed(2, 3)

        _process_events(qtbot, count=3)

        assert calls == [("slice", 3)]
        assert win.view_state.slice_indices[2] == 3
        assert win.render_coordinator.requested == 3
        assert win.render_coordinator.coalesced >= 2
    finally:
        win.close()


def test_coalescer_throttles_without_starving_continuous_bursts(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 12, dtype=float).reshape(4, 5, 12))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win, "render", lambda **kwargs: calls.append(win.view_state.slice_indices[2]))
    try:
        _process_events(qtbot, count=2)
        for index in range(1, 8):
            win._on_slice_index_changed(2, index)
            qtbot.wait(5)
        _process_events(qtbot, count=4)

        assert 1 <= len(calls) < 7
        assert calls[-1] == 7
    finally:
        win.close()


def test_interactive_slice_cancels_stale_render_dependent_work(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=float).reshape(4, 5, 8))
    qtbot.addWidget(win)
    cleared = []
    cancelled_prefetch = []

    def record_clear(controller_name):
        return lambda group: cleared.append((controller_name, group))

    monkeypatch.setattr(win.visible_evaluation_controller, "clear_group", record_clear("visible"))
    monkeypatch.setattr(win.profile_evaluation_controller, "clear_group", record_clear("profile"))
    monkeypatch.setattr(win.roi_evaluation_controller, "clear_group", record_clear("roi"))
    monkeypatch.setattr(win.pixel_evaluation_controller, "clear_group", record_clear("pixel"))
    monkeypatch.setattr(win.prefetch_evaluation_controller, "cancel_prefetch", lambda: cancelled_prefetch.append(True))
    monkeypatch.setattr(win, "render", lambda **_kwargs: None)
    try:
        _process_events(qtbot, count=2)
        win._on_slice_index_changed(2, 1)

        assert ("visible", "visible-image") in cleared
        assert ("visible", "visible-montage") in cleared
        assert ("profile", "profile-plot") in cleared
        assert ("profile", "live-profile") in cleared
        assert ("roi", "roi-inspection") in cleared
        assert ("pixel", "pixel") in cleared
        assert cancelled_prefetch == [True]
    finally:
        win.close()


def test_deferred_side_panels_refresh_once_after_interaction_quiet(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=float).reshape(4, 5, 8))
    qtbot.addWidget(win)
    calls = {"operation": 0, "inspection": 0, "profile": 0}
    monkeypatch.setattr(win, "_update_operation_dock", lambda: calls.__setitem__("operation", calls["operation"] + 1))
    monkeypatch.setattr(win, "_refresh_inspection_dock", lambda: calls.__setitem__("inspection", calls["inspection"] + 1))
    monkeypatch.setattr(win, "update_line_plot", lambda: calls.__setitem__("profile", calls["profile"] + 1))
    try:
        _process_events(qtbot, count=2)
        calls.update({"operation": 0, "inspection": 0, "profile": 0})
        win._on_slice_index_changed(2, 1)
        _process_events(qtbot, count=3)

        assert calls == {"operation": 0, "inspection": 0, "profile": 0}
        assert win._deferred_side_panel_refresh_pending

        _process_events(qtbot, count=12)
        assert calls["operation"] == 1
        assert calls["inspection"] == 1
        assert calls["profile"] == 0
        assert not win._deferred_side_panel_refresh_pending
    finally:
        win.close()


def test_direct_render_still_refreshes_side_panels(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5, dtype=float).reshape(4, 5))
    qtbot.addWidget(win)
    calls = {"operation": 0, "inspection": 0}
    monkeypatch.setattr(win, "_update_operation_dock", lambda: calls.__setitem__("operation", calls["operation"] + 1))
    monkeypatch.setattr(win, "_refresh_inspection_dock", lambda: calls.__setitem__("inspection", calls["inspection"] + 1))
    try:
        _process_events(qtbot, count=2)
        calls.update({"operation": 0, "inspection": 0})
        win.render(reason="normal")

        assert calls["operation"] >= 1
        assert calls["inspection"] == 1
    finally:
        win.close()
