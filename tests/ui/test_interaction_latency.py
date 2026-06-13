import numpy as np

from arrayscope.display.slice_engine import DisplayImage
from arrayscope.operations.evaluator import EvaluationResult
from tests.ui.helpers import clear_arrayscope_settings, process_events


def _tile_result(tile, value):
    image = np.full((tile.height, tile.width), value, dtype=np.float32)
    return EvaluationResult(DisplayImage(image, histogram_data=image.copy()), 0.0, image.shape, int(image.nbytes))


def test_rapid_slice_burst_is_coalesced_and_latest(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=np.float32).reshape(4, 5, 8))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win, "render", lambda **kwargs: calls.append((kwargs, win.view_state.slice_indices[2])))
    try:
        win._on_slice_index_changed(2, 1)
        win._on_slice_index_changed(2, 2)
        win._on_slice_index_changed(2, 3)

        assert calls == []
        qtbot.waitUntil(lambda: bool(calls), timeout=500)

        assert len(calls) < 3
        assert calls[-1][1] == 3
    finally:
        win.close()


def test_hot_cached_montage_schedules_no_tile_evaluation(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        state = win.view_state.with_montage_axis(2, columns=2, indices=(0, 1), text=":")
        plan = make_montage_plan(state, axis=2, indices=(0, 1), tile_shape=(2, 2), columns=2)
        for tile in plan.tiles:
            win.operation_evaluator.store_montage_tile_result(
                tile,
                montage_axis=2,
                colormap_lut=None,
                result=_tile_result(tile, int(tile.source_index) + 1),
            )
        calls = []
        monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))

        win._set_view_state(state)
        win.update_montage_view()

        qtbot.waitUntil(lambda: win._current_montage_canvas is not None, timeout=250)
        assert calls == []
        assert win._montage_cached_tiles_last_session == 2
        assert win._montage_missing_tiles_last_session == 0
    finally:
        win.close()


def test_cold_montage_tile_patches_without_side_panel_refresh(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    calls = []
    operation_refreshes = []
    inspection_refreshes = []
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
    monkeypatch.setattr(win, "_update_operation_dock", lambda: operation_refreshes.append("operation"))
    monkeypatch.setattr(win, "_refresh_inspection_dock", lambda: inspection_refreshes.append("inspection"))
    try:
        process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_montage_view(defer_side_panels=True)
        operation_refreshes.clear()
        inspection_refreshes.clear()

        tile = win._montage_session.plan.tiles[0]
        calls[0]["on_done"](_tile_result(tile, 9))
        qtbot.waitUntil(lambda: np.array_equal(win._current_montage_canvas.data[0:2, 0:2], np.full((2, 2), 9, dtype=np.float32)), timeout=250)

        assert operation_refreshes == []
        assert inspection_refreshes == []
    finally:
        win.close()
