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


def test_roi_statistics_refresh_is_debounced(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(40 * 40, dtype=float).reshape(40, 40))
    qtbot.addWidget(win)
    calls = []
    original = win._compute_roi_inspection_snapshot
    monkeypatch.setattr(
        win,
        "_compute_roi_inspection_snapshot",
        lambda *args, **kwargs: (calls.append(args[0]), original(*args, **kwargs))[1],
        )
    try:
        _process_events(qtbot, count=20)
        win.layout_manager.set_managed_dock_visible(win.inspection_dock, True, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=10)
        calls.clear()
        win.img_view.createRoi("rectangle", rect=(2, 2, 6, 6))
        win.img_view.createRoi("rectangle", rect=(4, 4, 6, 6))
        win.img_view.createRoi("rectangle", rect=(6, 6, 6, 6))
        _process_events(qtbot, count=20)

        assert len(calls) == 1
        assert win.inspection_dock.roi_model.rowCount() == 3
    finally:
        win.close()


def test_hidden_inspection_panel_updates_overlay_without_dock_work(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(40 * 40, dtype=float).reshape(40, 40))
    qtbot.addWidget(win)
    calls = []
    original = win._compute_roi_inspection_snapshot
    monkeypatch.setattr(
        win,
        "_compute_roi_inspection_snapshot",
        lambda *args, **kwargs: (calls.append(args[0]), original(*args, **kwargs))[1],
    )
    try:
        _process_events(qtbot, count=20)
        win.layout_manager.set_managed_dock_visible(win.inspection_dock, False, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=10)
        calls.clear()
        win.img_view.createRoi("rectangle", rect=(2, 2, 6, 6))
        _process_events(qtbot, count=20)

        assert calls == []
        assert getattr(win, "_inspection_stale", False)
        assert win.inspection_dock.roi_model.rowCount() == 0
        assert win.img_view._roi_info_panel is not None
        assert "n=36" in win.img_view._roi_info_panel.text()
        assert "mean=184.5" in win.img_view._roi_info_panel.text()
    finally:
        win.close()


def test_hidden_inspection_panel_uses_tiled_frame_payloads_and_opening_populates_dock(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.display.frame_planner import FramePlanner
    from arrayscope.window import ArrayScopeWindow

    data = np.arange(8 * 8, dtype=float).reshape(8, 8)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    calls = []
    original = win._compute_roi_inspection_snapshot
    monkeypatch.setattr(
        win,
        "_compute_roi_inspection_snapshot",
        lambda *args, **kwargs: (calls.append(args[0]), original(*args, **kwargs))[1],
    )
    try:
        win._frame_planner_instance = FramePlanner(internal_tile_shape=(4, 4), max_raster_pixels=16)
        win.render(reason="test-tiled-roi")
        _process_events(qtbot, count=30)
        assert getattr(win._committed_display_frame, "is_tiled", False)

        win.layout_manager.set_managed_dock_visible(win.inspection_dock, False, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=10)
        calls.clear()
        win.img_view.createRoi("rectangle", rect=(1, 1, 3, 3))
        _process_events(qtbot, count=20)

        assert calls == []
        assert getattr(win, "_inspection_stale", False)
        assert win.inspection_dock.roi_model.rowCount() == 0
        assert win.img_view._roi_info_panel is not None
        assert "n=9" in win.img_view._roi_info_panel.text()
        assert "mean=18" in win.img_view._roi_info_panel.text()

        win._show_inspection_dock()
        qtbot.waitUntil(lambda: win.inspection_dock.roi_model.rowCount() == 1, timeout=3000)
        _process_events(qtbot, count=10)

        model = win.inspection_dock.roi_model
        assert model.data(model.index(0, 2)) == "9"
        assert len(win.inspection_dock.histogram_plot.listDataItems()) == 1
        assert not getattr(win, "_inspection_stale", False)
    finally:
        win.close()


def test_hidden_montage_roi_overlay_does_not_sample_loading_placeholder(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from dataclasses import replace
    from arrayscope.display.slice_engine import DisplayImage
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.evaluator import EvaluationResult
    from arrayscope.window import ArrayScopeWindow

    data = np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(
        win.montage_tile_evaluation_controller,
        "start_latest",
        lambda _fn, **kwargs: calls.append(kwargs) or len(calls),
    )
    try:
        _process_events(qtbot, count=20)
        first_state = win.view_state.with_montage_axis(2, columns=2, indices=(0, 1), text="0:2")
        first_plan = make_montage_plan(first_state, axis=2, indices=(0, 1), tile_shape=(2, 2), columns=2)
        for tile in first_plan.tiles:
            value = 10.0 + float(tile.source_index)
            image = np.full((2, 2), value, dtype=np.float32)
            win.operation_evaluator.store_montage_tile_result(
                tile,
                montage_axis=2,
                colormap_lut=None,
                result=EvaluationResult(DisplayImage(image, histogram_data=image.copy()), 0.0, image.shape, int(image.nbytes)),
            )

        win._set_view_state(first_state)
        win.update_montage_view()
        qtbot.waitUntil(lambda: getattr(win._montage_session, "display_committed", False), timeout=1000)
        win.layout_manager.set_managed_dock_visible(win.inspection_dock, False, reason="test", preserve_canvas=False)
        win.img_view.createRoi("rectangle", rect=(0, 0, 2, 2))
        _process_events(qtbot, count=20)
        assert "mean=10" in win.img_view._roi_info_panel.text()
        truthful_text = win.img_view._roi_info_panel.text()

        calls.clear()
        second_state = win.view_state.with_axis_range(2, indices=(2, 3), text="2:4")
        win._set_view_state(second_state)
        win.update_montage_view()
        win._refresh_inspection_dock()
        _process_events(qtbot, count=20)

        assert calls
        assert not win._montage_session.display_committed
        assert win.img_view._roi_info_panel.text() == truthful_text
        assert "mean=0" not in win.img_view._roi_info_panel.text()
    finally:
        win.close()


def test_vispy_hidden_inspection_panel_uses_tiled_frame_payloads(qtbot):
    _clear_arrayscope_settings()
    pytest.importorskip("vispy")
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.display.backend_contract import image_view_backend_capabilities
    from arrayscope.display.frame_planner import FramePlanner
    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.VISPY.value)
    settings.sync()

    data = np.arange(8 * 8, dtype=float).reshape(8, 8)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        if image_view_backend_capabilities(win.img_view).name != "vispy":
            pytest.skip("VisPy backend unavailable in this Qt environment")
        win._frame_planner_instance = FramePlanner(internal_tile_shape=(4, 4), max_raster_pixels=16)
        win.render(reason="test-vispy-tiled-roi")
        _process_events(qtbot, count=30)
        assert getattr(win._committed_display_frame, "is_tiled", False)

        win.layout_manager.set_managed_dock_visible(win.inspection_dock, False, reason="test", preserve_canvas=False)
        win.img_view.createRoi("rectangle", rect=(1, 1, 3, 3))
        _process_events(qtbot, count=20)

        assert win.img_view._roi_info_panel is not None
        assert "n=9" in win.img_view._roi_info_panel.text()
        assert "mean=18" in win.img_view._roi_info_panel.text()

        win._show_inspection_dock()
        qtbot.waitUntil(lambda: win.inspection_dock.roi_model.rowCount() == 1, timeout=3000)
    finally:
        win.close()
        settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.PYQTGRAPH.value)
        settings.sync()


def test_detached_inspection_panel_refreshes_roi_statistics_and_histogram(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window.panels import PanelLocation
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(40 * 40, dtype=float).reshape(40, 40))
    qtbot.addWidget(win)
    calls = []
    original = win._compute_roi_inspection_snapshot
    monkeypatch.setattr(
        win,
        "_compute_roi_inspection_snapshot",
        lambda *args, **kwargs: (calls.append(args[0]), original(*args, **kwargs))[1],
    )
    try:
        _process_events(qtbot, count=20)
        win.layout_manager.set_managed_dock_visible(win.inspection_dock, True, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=10)
        win.layout_manager.detach_managed_dock(win.inspection_dock, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=10)
        assert win.panel_manager.location("inspection") == PanelLocation.DETACHED
        assert not win.inspection_dock.isVisible()

        calls.clear()
        win.img_view.createRoi("rectangle", rect=(2, 2, 6, 6))
        qtbot.waitUntil(lambda: len(calls) == 1, timeout=3000)
        _process_events(qtbot, count=10)

        model = win.inspection_dock.roi_model
        assert model.rowCount() == 1
        assert model.data(model.index(0, 2)) == "36"
        assert len(win.inspection_dock.histogram_plot.listDataItems()) == 1
        assert not getattr(win, "_inspection_stale", False)
    finally:
        win.close()


def test_render_with_hidden_profile_and_live_profile_off_skips_line_plot(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(20 * 30, dtype=float).reshape(20, 30))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win.layout_manager.set_managed_dock_visible(win.profile_dock, False, reason="test", preserve_canvas=False)
        win.widgets["buttons"]["display"]["live_profile"].setChecked(False)
        _process_events(qtbot, count=10)
        monkeypatch.setattr(win, "update_line_plot", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("hidden profile work ran")))

        win.render(reason="hidden-profile-test")
        _process_events(qtbot, count=10)
    finally:
        win.close()


def test_render_refreshes_inspection_once_on_image_commit(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(20 * 30, dtype=float).reshape(20, 30))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        calls = []
        original = win._refresh_inspection_dock
        monkeypatch.setattr(win, "_refresh_inspection_dock", lambda *args, **kwargs: (calls.append(1), original(*args, **kwargs))[1])

        win.render(reason="inspection-refresh-test")
        _process_events(qtbot, count=10)

        assert len(calls) == 1
    finally:
        win.close()


def test_montage_viewport_updates_do_not_recompute_same_roi_stats(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((4, 4, 8), dtype=np.float32)
    for index in range(data.shape[2]):
        data[:, :, index] = index
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    calls = []
    original = win._compute_roi_inspection_snapshot
    monkeypatch.setattr(win, "_compute_roi_inspection_snapshot", lambda *args, **kwargs: (calls.append(args[0]), original(*args, **kwargs))[1])
    try:
        _process_events(qtbot, count=20)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(8)), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=40)
        win.layout_manager.set_managed_dock_visible(win.inspection_dock, True, reason="test", preserve_canvas=False)
        win.img_view.createRoi("rectangle", rect=(1, 1, 2, 2))
        qtbot.waitUntil(lambda: len(calls) == 1, timeout=3000)
        _process_events(qtbot, count=10)

        win.img_view.getView().setRange(xRange=(0, 3), yRange=(3, 6), padding=0)
        win.update_montage_view()
        _process_events(qtbot, count=40)
        win.update_montage_view()
        _process_events(qtbot, count=40)

        assert len(calls) == 1
    finally:
        win.close()


def test_empty_inspection_dock_does_not_rewrite_table_on_montage_viewport_updates(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((4, 4, 8), dtype=np.float32)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    table_updates = []
    histogram_updates = []
    overlay_updates = []
    original_set_statistics = win.inspection_dock.set_statistics
    original_set_histograms = win.inspection_dock.set_histograms
    original_update_overlay = win._update_roi_info_overlay
    monkeypatch.setattr(win.inspection_dock, "set_statistics", lambda stats: (table_updates.append(dict(stats)), original_set_statistics(stats))[1])
    monkeypatch.setattr(
        win.inspection_dock,
        "set_histograms",
        lambda histograms: (histogram_updates.append(tuple(histograms)), original_set_histograms(histograms))[1],
    )
    monkeypatch.setattr(
        win,
        "_update_roi_info_overlay",
        lambda stats: (overlay_updates.append(dict(stats)), original_update_overlay(stats))[1],
    )
    try:
        _process_events(qtbot, count=20)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(8)), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=40)
        win.layout_manager.set_managed_dock_visible(win.inspection_dock, True, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=20)
        table_updates.clear()
        histogram_updates.clear()
        overlay_updates.clear()

        win.img_view.getView().setRange(xRange=(0, 3), yRange=(3, 6), padding=0)
        win.update_montage_view()
        _process_events(qtbot, count=40)
        win.update_montage_view()
        _process_events(qtbot, count=40)

        assert table_updates == []
        assert histogram_updates == []
        assert overlay_updates == []
        assert win.inspection_dock.roi_model.rowCount() == 0
    finally:
        win.close()
