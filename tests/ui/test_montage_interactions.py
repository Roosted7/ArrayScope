import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.display.slice_engine import DisplayImage
from arrayscope.operations.evaluator import EvaluationResult
from tests.ui.helpers import (
    assert_panel_invariants as _assert_panel_invariants,
    assert_size_close as _assert_size_close,
    clear_arrayscope_settings as _clear_arrayscope_settings,
    panel_body as _panel_body,
    process_events as _process_events,
    view_action as _view_action,
    wait_for_panel_preserve as _wait_for_panel_preserve,
)


def _tile_result(tile, value):
    image = np.full((tile.height, tile.width), value, dtype=np.float32)
    return EvaluationResult(DisplayImage(image, histogram_data=image.copy()), 0.0, image.shape, int(image.nbytes))


def _tile_for_callback(win, call):
    requested_index = int(call["key"][-1])
    return next(
        tile
        for tile in win._montage_session.plan.tiles
        if int(tile.montage_index) == requested_index
    )


def _committed_tile_payload(win, tile):
    frame = getattr(win, "_committed_display_frame", None)
    value_source = getattr(frame, "value_source", None)
    payloads = getattr(value_source, "payloads", {})
    return payloads.get(int(tile.montage_index))


def _committed_tile_has_value(win, tile, value):
    payload = _committed_tile_payload(win, tile)
    if payload is None:
        return False
    expected = np.full((int(tile.height), int(tile.width)), value, dtype=np.float32)
    return np.array_equal(np.asarray(payload.image), expected)


def _display_levels(win):
    return tuple(float(value) for value in win.img_view.getLevels())


def _assert_view_contains_applied_montage_plan(win):
    plan = win._montage_session.plan
    height, width = tuple(int(value) for value in plan.display_shape[:2])
    view_range = win.img_view.getView().viewRange()
    assert view_range[0][0] <= 0.0
    assert view_range[0][1] >= float(width)
    assert view_range[1][0] <= 0.0
    assert view_range[1][1] >= float(height)


def _assert_committed_tile_value(win, tile, value):
    payload = _committed_tile_payload(win, tile)
    assert payload is not None
    expected = np.full((int(tile.height), int(tile.width)), value, dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(payload.image), expected)


def _reset_warm_tile_state(win):
    """Return the window to a semantically cold tile state.

    The tiled pipeline renders plain slices through the montage tile lane and
    keeps their payloads resident (evaluator cache, retained payload store,
    committed frame).  Tests that pin the cold montage contract (loading
    placeholders, progressive level warm-up) must drop that warm state first,
    as a document/context change would.
    """

    win.operation_evaluator.clear_cache()
    win.renderer._retained_tiled_payload_store().clear_for_document_or_context_change("test-cold-start")
    # The slice render's session would re-acknowledge its payloads on the next
    # presentation commit; drop it so no warm tile state can be resurrected.
    win.renderer._montage_session = None
    frame = getattr(win, "_committed_display_frame", None)
    payloads = getattr(getattr(frame, "value_source", None), "payloads", None)
    if isinstance(payloads, dict):
        payloads.clear()


def _use_slice_zero(win, qtbot):
    win._set_view_state(win.view_state.with_slice(2, 0))
    win.render(reason="test-initial-slice")
    _process_events(qtbot, count=20)
    qtbot.waitUntil(lambda: not win.montage_tile_evaluation_controller.is_busy(), timeout=3000)
    _reset_warm_tile_state(win)


def _settle_initial_render(win, qtbot):
    """Wait for the startup render (which flows through the montage tile lane
    since the tiled pipeline handles single slices too) and drop its warm
    tiles, so tests control exactly which tiles are warm and which calls the
    patched montage controller sees."""

    qtbot.waitUntil(lambda: getattr(win, "_committed_display_frame", None) is not None, timeout=3000)
    qtbot.waitUntil(lambda: not win.montage_tile_evaluation_controller.is_busy(), timeout=3000)
    _reset_warm_tile_state(win)


def test_montage_status_does_not_remain_computing(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.core.cache_status import CacheStatus
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, indices=(0, 1), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)
        assert win.operation_evaluator.last_status.status != CacheStatus.COMPUTING
    finally:
        win.close()


def test_montage_visible_subset_hover_uses_source_index_not_local_tile_zero(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((2, 3, 20), dtype=float)
    for index in range(data.shape[2]):
        data[:, :, index] = index
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(20)), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)
        win.img_view.getView().setRange(xRange=(0, 2), yRange=(7, 8), padding=0)
        win.update_image_view()
        tile_10 = win._montage_session.plan.tiles[10]
        qtbot.waitUntil(
            lambda: win.display_geometry.context_for_view_point(
                tile_10.x0 + 1,
                tile_10.y0 + 1,
            )
            is not None,
            timeout=3000,
        )
        context = win.display_geometry.context_for_view_point(tile_10.x0 + 1, tile_10.y0 + 1)

        assert context is not None
        assert context.context_text.endswith("d2=10")
    finally:
        win.close()


def test_panned_montage_hover_reads_committed_display_coordinates(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((2, 3, 20), dtype=np.float32)
    for index in range(data.shape[2]):
        data[:, :, index] = index
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(20)), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)
        win.img_view.getView().setRange(xRange=(0, 2), yRange=(7, 8), padding=0)
        win.update_image_view()

        tile_10 = win._montage_session.plan.tiles[10]
        qtbot.waitUntil(
            lambda: win.display_geometry.context_for_view_point(tile_10.x0 + 1, tile_10.y0 + 1)
            is not None,
            timeout=3000,
        )
        context = win.display_geometry.context_for_view_point(tile_10.x0 + 1, tile_10.y0 + 1)

        assert context is not None
        assert context.mapping.local_x == 1
        assert context.mapping.local_y == 1
        assert win.renderer._hover_value_from_display(context.mapping) == pytest.approx(10.0)
    finally:
        win.close()


def test_montage_update_after_shifted_origin_preserves_world_view_range(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((2, 3, 20), dtype=np.float32)
    for index in range(data.shape[2]):
        data[:, :, index] = index
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(20)), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)

        win.img_view.getView().setRange(xRange=(0, 2), yRange=(6, 8), padding=0)
        win.update_image_view()
        before = win.img_view.getView().viewRange()

        win.update_image_view()
        _process_events(qtbot, count=20)

        after = win.img_view.getView().viewRange()
        tile_10 = win._montage_session.plan.tiles[10]
        assert after[0] == pytest.approx(before[0])
        assert after[1] == pytest.approx(before[1])
        assert win.display_geometry.context_for_view_point(tile_10.x0 + 1, tile_10.y0 + 1) is not None
    finally:
        win.close()


def test_montage_tile_count_increase_preserves_manual_zoom_when_not_near_auto(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((2, 3, 20), dtype=np.float32)
    for index in range(data.shape[2]):
        data[:, :, index] = index
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(5)), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)
        win.img_view.getView().setRange(xRange=(0, 2), yRange=(0, 3), padding=0)
        before = win.img_view.getView().viewRange()

        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(20)), text=":"))
        win.render(reason="test-montage-more-tiles")
        _process_events(qtbot, count=80)

        view_range = win.img_view.getView().viewRange()
        assert view_range[0] == pytest.approx(before[0], abs=0.03)
        assert view_range[1] == pytest.approx(before[1], abs=0.03)
    finally:
        win.close()


def test_switching_to_larger_montage_auto_fits_when_tiles_would_be_hidden(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtWidgets
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((2, 3, 20), dtype=np.float32)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win.img_view.getView().setRange(xRange=(0, 3), yRange=(0, 2), padding=0)

        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(20)), text=":"))
        win.render(reason="test-switch-to-montage")
        _process_events(qtbot, count=80)

        _assert_view_contains_applied_montage_plan(win)
        assert win.statusBar().findChild(QtWidgets.QLabel, "ArrayScopeStatusActionLabel") is not None
    finally:
        win.close()


def test_montage_tile_count_increase_auto_adjusts_when_near_auto(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtWidgets
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((2, 3, 12), dtype=np.float32)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(10)), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)
        action = win.statusBar().findChild(QtWidgets.QLabel, "ArrayScopeStatusActionLabel")
        if action is not None:
            action.linkActivated.emit("action")
            _process_events(qtbot, count=5)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(12)), text=":"))
        win.render(reason="test-montage-few-more-tiles")
        _process_events(qtbot, count=80)

        _assert_view_contains_applied_montage_plan(win)
    finally:
        win.close()


def test_montage_manual_resize_is_single_camera_transaction(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.display.viewport import ViewportMode
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((20, 20, 30), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        win.resize(900, 700)
        win.show()
        _process_events(qtbot, count=20)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=None, indices=tuple(range(30)), text=":"))
        win.render(reason="test-montage-resize")
        _process_events(qtbot, count=80)
        view = win.img_view.getView()
        view.setRange(xRange=(-100.0, 300.0), yRange=(-100.0, 300.0), padding=0)
        _process_events(qtbot, count=5)
        win.img_view.viewport_controller.mode = ViewportMode.USER
        before_size = win.img_view.graphicsView.viewport().size()
        before = view.viewRange()
        range_changes = []
        view.sigRangeChanged.connect(lambda *_args: range_changes.append(view.viewRange()))

        win.resize(500, 700)
        _process_events(qtbot, count=20)

        after_size = win.img_view.graphicsView.viewport().size()
        after = view.viewRange()
        before_x_units = (before[0][1] - before[0][0]) / before_size.width()
        before_y_units = (before[1][1] - before[1][0]) / before_size.height()
        after_x_units = (after[0][1] - after[0][0]) / after_size.width()
        after_y_units = (after[1][1] - after[1][0]) / after_size.height()
        assert after_x_units == pytest.approx(before_x_units)
        assert after_y_units == pytest.approx(before_y_units)
        assert range_changes == []
    finally:
        win.close()


def test_montage_auto_fit_skips_when_fit_mode_is_enabled(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtWidgets
    from arrayscope.display.viewport import ViewportMode
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((2, 3, 20), dtype=np.float32)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(5)), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)
        action = win.statusBar().findChild(QtWidgets.QLabel, "ArrayScopeStatusActionLabel")
        if action is not None:
            action.linkActivated.emit("action")
            _process_events(qtbot, count=5)
        win.display_toolbar.fit_action.trigger()
        _process_events(qtbot, count=10)
        assert win.img_view.viewport_controller.mode == ViewportMode.FIT

        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(20)), text=":"))
        win.render(reason="test-montage-fit-mode")
        _process_events(qtbot, count=80)

        assert win.img_view.viewport_controller.mode == ViewportMode.FIT
        assert win.statusBar().findChild(QtWidgets.QLabel, "ArrayScopeStatusActionLabel") is None
    finally:
        win.close()


def test_montage_interactive_render_commits_tiled_frame(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 8, dtype=float).reshape(2, 3, 8))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(8)), text=":"))
        win.render(reason="test-montage")
        qtbot.waitUntil(lambda: getattr(getattr(win, "_committed_display_frame", None), "scene", None) is not None, timeout=3000)

        frame = win._committed_display_frame
        assert frame.is_tiled
        assert frame.scene.resident_region_ids
    finally:
        win.close()


def test_montage_commits_cached_tiles_immediately_with_loading_placeholders(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.display.montage import MontageTileState, make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        _settle_initial_render(win, qtbot)
        state = win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":")
        win._set_view_state(state)
        plan = make_montage_plan(state, axis=2, indices=(0, 1, 2), tile_shape=(2, 2), columns=3)
        win.operation_evaluator.store_montage_tile_result(plan.tiles[0], montage_axis=2, colormap_lut=None, result=_tile_result(plan.tiles[0], 10))
        monkeypatch.setattr(win.renderer, "retarget_montage_pipeline", lambda _session: None)

        win.update_image_view()

        states = win._montage_session.ensure_tile_states()
        assert states[0] == MontageTileState.LOADED
        # R4 scopes visible admission to current visible tiles. Coverage/near
        # tiles do not masquerade as loading work when no owner admitted them.
        assert states[1] == MontageTileState.UNLOADED
        _assert_committed_tile_value(win, plan.tiles[0], 10)
    finally:
        win.close()


def test_montage_loading_overlay_clears_after_final_delayed_commit(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.widgets["buttons"]["display"]["window_absolute"].setChecked(True)
        win.widgets["buttons"]["display"]["window_relative"].setChecked(False)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_image_view()
        win.renderer._show_montage_session_loading_overlay(win._montage_session)
        assert win.img_view._evaluation_overlay.isVisible()

        qtbot.waitUntil(
            lambda: win.img_view._evaluation_overlay is None or not win.img_view._evaluation_overlay.isVisible(),
            timeout=3000,
        )
        assert getattr(win.img_view, "_montage_tile_overlay_items", []) == []
        assert win._montage_session.visible_plan_complete()
    finally:
        win.close()


def test_montage_ready_display_payloads_commit_immediately(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    session = SimpleNamespace(
        session_id=999,
        key=("test-session",),
        render_generation=win.renderer._capture_render_generation(),
        stage_fan_in=StageFanInState(),
        final_commit_pending=False,
        flush_pending=False,
        pending_removals=set(),
        dirty_payloads={1: None},
    )
    calls = []
    try:
        _process_events(qtbot)
        _settle_initial_render(win, qtbot)
        monkeypatch.setattr(
            win.renderer,
            "_is_current_frame_session",
            lambda session_id, key: session_id == 999 and key == ("test-session",),
        )
        monkeypatch.setattr(
            win.renderer,
            "commit_frame_session_presentation",
            lambda _session: calls.append(
                (
                    bool(session.final_commit_pending),
                    bool(session.flush_pending),
                )
            ),
        )
        win.renderer._frame_session = session

        win.renderer.apply_ready_montage_display(session)

        assert session.final_commit_pending
        assert session.flush_pending
        assert calls
        assert calls[0] == (True, True)
    finally:
        win.close()


def test_montage_active_tiles_are_all_accounted_for(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.display.montage import MontageTileState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 12, dtype=np.float32).reshape(2, 2, 12))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(12)), text=":"))
        win.update_image_view()
        qtbot.waitUntil(lambda: win._montage_session is not None, timeout=3000)

        active_ids = set(win._montage_session.visible_tile_numbers)
        qtbot.waitUntil(
            lambda: all(
                win._montage_session.ensure_tile_states()[int(tile.montage_index)] != MontageTileState.UNLOADED
                for tile in win._montage_session.plan.tiles
                if int(tile.montage_index) in active_ids
            ),
            timeout=3000,
        )
        states = win._montage_session.ensure_tile_states()
        for tile in win._montage_session.plan.tiles:
            if int(tile.montage_index) in active_ids:
                assert states[tile.montage_index] in {
                    MontageTileState.LOADED,
                    MontageTileState.LOADING,
                    MontageTileState.SKIPPED,
                }
    finally:
        win.close()


def test_montage_pan_schedules_viewport_update(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 12, dtype=np.float32).reshape(2, 2, 12))
    qtbot.addWidget(win)
    calls = []
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(12)), text=":"))
        win.update_image_view()
        qtbot.waitUntil(lambda: getattr(win._montage_session, "display_committed", False), timeout=3000)
        monkeypatch.setattr(win.renderer, "retarget_montage_viewport", lambda: calls.append("retargeted"))

        win.img_view.getView().setRange(xRange=(6, 9), yRange=(0, 2), padding=0)

        qtbot.waitUntil(lambda: bool(calls), timeout=1000)
    finally:
        win.close()


def test_cached_montage_tile_rebinds_to_current_layout(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        old_state = ViewState.from_shape((2, 2, 3)).with_montage_axis(2, columns=1, indices=(0, 1, 2), text=":")
        old_plan = make_montage_plan(old_state, axis=2, indices=(0, 1, 2), tile_shape=(2, 2), columns=1)
        win.operation_evaluator.store_montage_tile_result(old_plan.tiles[1], montage_axis=2, colormap_lut=None, result=_tile_result(old_plan.tiles[1], 11))
        monkeypatch.setattr(win.renderer, "retarget_montage_pipeline", lambda _session: None)

        new_state = win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":")
        win._set_view_state(new_state)
        win.update_image_view()

        new_tile = win._montage_session.plan.tiles[1]
        _assert_committed_tile_value(win, new_tile, 11)
    finally:
        win.close()


def test_montage_skipped_tiles_show_detailed_warning(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    import arrayscope.window.render as render_module
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((2, 2, 2), dtype=np.float32))
    qtbot.addWidget(win)
    messages = []
    monkeypatch.setattr(render_module.Qt.QtWidgets.QMessageBox, "warning", lambda _parent, _title, message: messages.append(message))
    try:
        _process_events(qtbot)
        win.renderer._warn_montage_tiles_skipped(
            skipped_count=2,
            tile_bytes=2048,
            budget_bytes=1024,
            tile_shape=(16, 16),
        )

        assert messages
        assert "skipped 2 tile" in messages[0]
        assert "over the visible render budget" in messages[0]
        assert "Tile shape is (16, 16)" in messages[0]
    finally:
        win.close()


def test_montage_auto_window_button_applies_current_semantic_bounds_immediately(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=80)
        expected = (1.0, 11.0)
        original_session = win._montage_session

        win.img_view.setLevels(2.0, 8.0)
        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (2.0, 8.0)

        win.auto_window_levels()

        assert win._montage_session is original_session
        assert win.renderer._explicit_user_level_source is None
        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == expected
        _process_events(qtbot, count=20)
        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == expected
    finally:
        win.close()


def test_montage_zoom_in_does_not_shrink_level_source_coverage(qtbot, monkeypatch):
    from arrayscope.window.montage_commit import MontagePipelineEffects

    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    tile_count = 6
    win = ArrayScopeWindow(np.arange(2 * 2 * tile_count, dtype=np.float32).reshape(2, 2, tile_count))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=6, indices=tuple(range(tile_count)), text=":"))
        win.update_image_view()

        session = win._montage_session
        effects = MontagePipelineEffects(win.renderer, session)
        for index, tile in enumerate(session.plan.tiles):
            effects.admit_tile_result(tile, _tile_result(tile, 100 * (index + 1)))
        win.renderer.apply_montage_presentation(session)
        _process_events(qtbot)

        before_levels = tuple(round(float(value), 6) for value in win.img_view.getLevels())
        before_histogram = tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds())
        assert session.applied_level_source.source_count == tile_count
        assert session.applied_level_source.expected_count == tile_count

        win.img_view.getView().setRange(xRange=(0.0, 2.0), yRange=(0.0, 2.0), padding=0)
        win.renderer.apply_montage_viewport_retarget()
        _process_events(qtbot)

        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == before_levels
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == before_histogram
        assert win._montage_session.applied_level_source.source_count == tile_count
        assert win._montage_session.applied_level_source.expected_count == tile_count
    finally:
        win.close()


def test_enabling_montage_with_cached_tile_preserves_relative_window_fractions(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((4, 5, 3), dtype=np.float32)
    for index in range(3):
        data[:, :, index] = index * 100.0 + np.arange(20, dtype=np.float32).reshape(4, 5)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        _use_slice_zero(win, qtbot)
        win.img_view.setLevels(5.0, 15.0)
        state = win.view_state.with_montage_axis(2, columns=3, indices=(1, 2), text="1:3")
        plan = make_montage_plan(state, axis=2, indices=(1, 2), tile_shape=(4, 5), columns=3)
        tile = plan.tiles[0]
        win.operation_evaluator.store_montage_tile_result(tile, montage_axis=2, colormap_lut=None, result=_tile_result(tile, 100.0))
        monkeypatch.setattr(win.renderer, "retarget_montage_pipeline", lambda _session: None)

        win._set_view_state(state)
        win.update_image_view()
        _process_events(qtbot, count=10)

        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (99.526316, 100.578947)
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (99.0, 101.0)
    finally:
        win.close()


def test_shifting_montage_range_preserves_relative_window_fractions(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((4, 5, 3), dtype=np.float32)
    for index in range(3):
        data[:, :, index] = index * 100.0 + np.arange(20, dtype=np.float32).reshape(4, 5)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)

    def result_for(tile, image):
        image = np.asarray(image, dtype=np.float32)
        return EvaluationResult(DisplayImage(image, histogram_data=image.copy()), 0.0, image.shape, int(image.nbytes))

    try:
        _process_events(qtbot)
        _use_slice_zero(win, qtbot)
        win.img_view.setLevels(5.0, 15.0)
        first_state = win.view_state.with_montage_axis(2, columns=2, indices=(0, 1), text="0:2")
        first_plan = make_montage_plan(first_state, axis=2, indices=(0, 1), tile_shape=(4, 5), columns=2)
        for tile in first_plan.tiles:
            win.operation_evaluator.store_montage_tile_result(tile, montage_axis=2, colormap_lut=None, result=result_for(tile, data[:, :, tile.source_index]))
        win._set_view_state(first_state)
        win.update_image_view()
        qtbot.waitUntil(
            lambda: tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (0.0, 119.0),
            timeout=3000,
        )
        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (31.315789, 93.947368)
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (0.0, 119.0)

        shifted_state = win.view_state.with_montage_axis(2, columns=2, indices=(1, 2), text="1:3")
        win._set_view_state(shifted_state)
        win.update_image_view()
        qtbot.waitUntil(
            lambda: tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (131.315789, 193.947368),
            timeout=3000,
        )

        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (131.315789, 193.947368)
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (100.0, 219.0)
    finally:
        win.close()


def test_cached_montage_commit_uses_all_loaded_tiles_for_initial_histogram(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    tile_count = 6
    win = ArrayScopeWindow(np.zeros((2, 2, tile_count), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        state = win.view_state.with_montage_axis(2, columns=tile_count, indices=tuple(range(tile_count)), text=":")
        plan = make_montage_plan(state, axis=2, indices=tuple(range(tile_count)), tile_shape=(2, 2), columns=tile_count)
        values = (0.0, 0.0, 0.0, 0.0, 1000.0, 1200.0)
        for tile, value in zip(plan.tiles, values):
            win.operation_evaluator.store_montage_tile_result(
                tile,
                montage_axis=2,
                colormap_lut=None,
                result=_tile_result(tile, value),
            )
        monkeypatch.setattr(win.renderer, "retarget_montage_pipeline", lambda _session: None)

        win.img_view.getView().setRange(xRange=(0, 20), yRange=(0, 2), padding=0)
        win._set_view_state(state)
        win.update_image_view()
        _process_events(qtbot, count=10)

        bounds = tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds())
        assert bounds[1] > 1000.0
        assert win._montage_session.applied_level_source.source_count == tile_count
        assert win._montage_session.applied_level_source.expected_count == tile_count
    finally:
        win.close()


def test_fft_montage_stage_cache_hit_keeps_per_tile_slab_plans(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.operations.pipeline import CenteredFFT
    from arrayscope.operations.regions import region_text
    from arrayscope.operations.stage_cache import StageValue
    from arrayscope.operations.stage_materialization import StageMaterializationResult
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4))
    qtbot.addWidget(win)
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **_kwargs: 1)
    try:
        _process_events(qtbot)
        win.operation_coordinator.load_operations((CenteredFFT(axis=2),))
        win._set_document(win.operation_coordinator.document)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=(0, 1, 2, 3), text=":"))

        materializer = win.operation_evaluator.stage_materializer

        def cached_stage(document_key, candidate):
            key = materializer.key_for_candidate(document_key, candidate)
            data = np.zeros(tuple(candidate.shape), dtype=np.dtype(candidate.dtype))
            value = StageValue(
                data=data,
                region=candidate.region,
                stage_index=int(candidate.stage_index),
                nbytes=int(data.nbytes),
                priority=str(candidate.priority),
            )
            return StageMaterializationResult("hit", key, value=value)

        monkeypatch.setattr(materializer, "request_stage", cached_stage)

        win.update_image_view()

        session = win._montage_session
        planned_regions = [
            region_text(session.stage_fan_in.tile_stage_plans[index].region_plan.final_region)
            for index in range(4)
        ]

        assert len(set(planned_regions)) == 4
    finally:
        win.close()


def test_operation_backed_complex_montage_tile_layer_rewindows_rgb_from_histogram_levels(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.operations.pipeline import CenteredFFT
    from arrayscope.window import ArrayScopeWindow

    data = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.operation_coordinator.load_operations((CenteredFFT(axis=0),))
        win._set_document(win.operation_coordinator.document)
        win._coerce_channel_for_current_dtype()
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_image_view()

        qtbot.waitUntil(lambda: getattr(win._montage_session, "display_committed", False), timeout=3000)
        qtbot.waitUntil(lambda: win.img_view.montageDisplayMode() == "tile_layer", timeout=3000)
        assert any(state.rgb_base is not None for state in win.img_view._montage_tile_layer.states.values())

        low, high = win.img_view.getHistogramDataBounds()
        desired = ((float(low) + float(high)) / 2.0, float(high))
        # A user command must supersede automatic level work still attached to
        # an otherwise committed progressive montage session.
        win._montage_session.force_auto = True
        win.img_view.setLevels(*desired)
        assert win._montage_session.force_auto is False
        assert win._montage_session.level_generation.target_levels == desired
        qtbot.waitUntil(
            lambda: all(tuple(state.levels) == desired for state in win.img_view._montage_tile_layer.states.values()),
            timeout=1000,
        )
        assert win._montage_session.has_stale_level_presentations() is False
    finally:
        win.close()


def test_large_complex_montage_auto_uses_tile_layer(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.ones((840, 840, 3), dtype=np.complex64)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_image_view()

        qtbot.waitUntil(lambda: getattr(win._montage_session, "display_committed", False), timeout=5000)

        assert win.img_view.montageDisplayMode() == "tile_layer"
        assert getattr(win.renderer, "_last_montage_backend_actual", None) == "tile_layer"
    finally:
        win.close()


def test_large_complex_montage_tile_layer_histogram_drag_does_not_update_base_image_item(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.ones((840, 840, 3), dtype=np.complex64)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_image_view()
        qtbot.waitUntil(lambda: win.img_view.montageDisplayMode() == "tile_layer", timeout=5000)
        qtbot.waitUntil(lambda: bool(win.img_view._montage_tile_layer.states), timeout=5000)
        qtbot.waitUntil(
            lambda: (
                not getattr(win._montage_session, "dirty_payloads", {})
                and not getattr(win._montage_session, "pending_payload_upserts", {})
                and not getattr(win._montage_session, "pending_removals", set())
                and not getattr(win._montage_session, "active_tile_requests", set())
            ),
            timeout=5000,
        )

        def fail_base_image_item_update(*args, **kwargs):
            raise AssertionError("base ImageItem update during tile-layer histogram drag")

        monkeypatch.setattr(win.img_view.imageItem, "setImage", fail_base_image_item_update)
        low, high = win.img_view.getHistogramDataBounds()
        win.img_view.histogram.setLevels((float(low) + float(high)) / 2.0, float(high))
        qtbot.wait(50)
        win.img_view._on_histogram_level_change_finished()

        timing = win.img_view.lastImageUploadTiming()
        assert timing.mode == "tile_layer"
        assert timing.tile_layer_visible_items > 0
        assert timing.tile_layer_items_updated <= timing.tile_layer_visible_items
        assert timing.tile_layer_texture_uploads == 0
    finally:
        win.close()


def test_visible_render_budget_uses_app_setting():
    from types import SimpleNamespace

    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window.render import RenderOrchestrator

    renderer = RenderOrchestrator.__new__(RenderOrchestrator)
    renderer.win = SimpleNamespace(app_settings=AppSettingsState(render_memory_budget_mb=256))

    assert renderer._visible_render_budget_bytes() == 256 * 1024 * 1024
