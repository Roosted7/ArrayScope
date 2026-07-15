import time
from types import SimpleNamespace

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


def test_hidden_montage_roi_stats_use_semantic_demand_not_presented_payloads(monkeypatch):
    from arrayscope.core.roi import RoiGeometry, RoiKind, RoiSelection
    from arrayscope.core.roi_store import RoiStore
    from arrayscope.core.view_state import ViewState
    from arrayscope.operations.pipeline import ArrayDocument
    from arrayscope.window.inspection import InspectionWorkflowMixin

    class FakeRoiController:
        submissions = []

        def start_latest(self, fn, **kwargs):
            self.submissions.append((fn, kwargs))
            return len(self.submissions)

    selection = RoiSelection("roi-1", "ROI 1", RoiGeometry(RoiKind.RECTANGLE, rect=(50.0, 50.0, 4.0, 4.0)))
    win = InspectionWorkflowMixin()
    win.roi_store = RoiStore(selections=(selection,))
    win.document = ArrayDocument(np.zeros((4, 4, 8), dtype=np.float32))
    win.img_view = SimpleNamespace(roiSelections=lambda: (selection,))
    win.inspection_dock = SimpleNamespace(set_rois=lambda _selections: None, isVisible=lambda: False)
    win.view_state = ViewState.from_shape((4, 4, 8)).with_montage_axis(2, indices=tuple(range(8)), text=":")
    win._montage_roi_values_pending = lambda: False
    win._gui_callback_budget_decision = lambda *args, **kwargs: None
    win._hidden_roi_statistics = lambda _selections: (_ for _ in ()).throw(AssertionError("presented payloads are not authoritative"))
    win.roi_evaluation_controller = FakeRoiController()

    win._schedule_refresh_inspection_dock("file-session-restore")

    assert len(win.roi_evaluation_controller.submissions) == 1
    _, kwargs = win.roi_evaluation_controller.submissions[0]
    assert kwargs["priority"].name == "HIDDEN_ROI"


def test_montage_roi_waits_for_canonical_visible_plan_completion():
    from arrayscope.window.inspection import InspectionWorkflowMixin

    win = InspectionWorkflowMixin()
    state = object()
    win.view_state = state
    session = SimpleNamespace(
        view_state=state,
        visible_plan_complete=lambda: False,
    )
    win.renderer = SimpleNamespace(_frame_session=session)

    assert win._montage_roi_values_pending()

    session.visible_plan_complete = lambda: True
    assert not win._montage_roi_values_pending()

    session.view_state = object()
    assert win._montage_roi_values_pending()


def _render_committed_tiled_frame(win, qtbot, *, reason: str) -> None:
    win.render(reason=reason)
    qtbot.waitUntil(lambda: getattr(getattr(win, "_committed_display_frame", None), "is_tiled", False), timeout=3000)


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
        _render_committed_tiled_frame(win, qtbot, reason="test-roi-inspection")
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
        _render_committed_tiled_frame(win, qtbot, reason="test-hidden-roi-overlay")
        win.layout_manager.set_managed_dock_visible(win.inspection_dock, False, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=10)
        calls.clear()
        win.img_view.createRoi("rectangle", rect=(2, 2, 6, 6))
        _process_events(qtbot, count=20)

        assert len(calls) == 1
        assert win._roi_inspection_priority.name == "HIDDEN_ROI"
        assert getattr(win, "_inspection_stale", False)
        assert win.inspection_dock.roi_model.rowCount() == 0
        assert win.img_view._roi_info_panel is not None
        assert "n=36" in win.img_view._roi_info_panel.text()
        assert "µ=184.5" in win.img_view._roi_info_panel.text()
    finally:
        win.close()


def test_hidden_single_image_timed_roi_refresh_updates_overlay(qtbot, monkeypatch):
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
        _render_committed_tiled_frame(win, qtbot, reason="test-hidden-timed-roi-overlay")
        win.layout_manager.set_managed_dock_visible(win.inspection_dock, False, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=10)

        win.img_view.createRoi("rectangle", rect=(2, 2, 6, 6))
        win.img_view.setRoiInfoText("")
        calls.clear()
        win._refresh_inspection_dock_now()

        qtbot.waitUntil(lambda: win.img_view._roi_info_panel is not None, timeout=3000)
        assert len(calls) == 1
        assert win._roi_inspection_priority.name == "HIDDEN_ROI"
        assert getattr(win, "_inspection_stale", False)
        assert win.inspection_dock.roi_model.rowCount() == 0
        assert "n=36" in win.img_view._roi_info_panel.text()
        assert "µ=184.5" in win.img_view._roi_info_panel.text()
    finally:
        win.close()


def test_hidden_roi_overlay_refreshes_when_tiled_frame_commits(qtbot, monkeypatch):
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
        win._committed_display_frame = None
        win.renderer._committed_display_request_key = None

        win.img_view.createRoi("rectangle", rect=(2, 2, 6, 6))
        win.img_view.setRoiInfoText("")
        calls.clear()

        _render_committed_tiled_frame(win, qtbot, reason="test-hidden-roi-overlay-after-commit")
        _process_events(qtbot, count=20)

        assert len(calls) == 1
        assert win._roi_inspection_priority.name == "HIDDEN_ROI"
        assert getattr(win, "_inspection_stale", False)
        assert win.inspection_dock.roi_model.rowCount() == 0
        assert win.img_view._roi_info_panel is not None
        assert "n=36" in win.img_view._roi_info_panel.text()
        assert "µ=184.5" in win.img_view._roi_info_panel.text()
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
        win.renderer._frame_planner_instance = FramePlanner(internal_tile_shape=(4, 4))
        win.render(reason="test-tiled-roi")
        _process_events(qtbot, count=30)
        assert getattr(win._committed_display_frame, "is_tiled", False)

        win.layout_manager.set_managed_dock_visible(win.inspection_dock, False, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=10)
        calls.clear()
        win.img_view.createRoi("rectangle", rect=(1, 1, 3, 3))
        _process_events(qtbot, count=20)

        assert len(calls) == 1
        assert win._roi_inspection_priority.name == "HIDDEN_ROI"
        assert getattr(win, "_inspection_stale", False)
        assert win.inspection_dock.roi_model.rowCount() == 0
        assert win.img_view._roi_info_panel is not None
        assert "n=9" in win.img_view._roi_info_panel.text()
        assert "µ=18" in win.img_view._roi_info_panel.text()

        win._show_inspection_dock()
        qtbot.waitUntil(lambda: win.inspection_dock.roi_model.rowCount() == 1, timeout=3000)
        _process_events(qtbot, count=10)

        model = win.inspection_dock.roi_model
        # Column 0 is the color swatch; Count sits at column 3.
        assert model.data(model.index(0, 3)) == "9"
        assert len(win.inspection_dock.histogram_plot.listDataItems()) == 1
        assert not getattr(win, "_inspection_stale", False)
    finally:
        win.close()


@pytest.mark.parametrize("backend", ("pyqtgraph", "vispy"))
def test_hidden_montage_roi_overlay_does_not_sample_loading_placeholder(
    qtbot,
    backend,
):
    if backend == "vispy":
        pytest.importorskip("vispy")
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore
    from arrayscope.display.slice_engine import DisplayImage
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.evaluator import EvaluationResult
    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", backend)
    settings.sync()
    data = np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
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
                shader_display=backend == "vispy",
            )

        win._set_view_state(first_state)
        win.update_image_view()
        qtbot.waitUntil(lambda: getattr(win.renderer._frame_session, "display_committed", False), timeout=1000)
        win.layout_manager.set_managed_dock_visible(win.inspection_dock, False, reason="test", preserve_canvas=False)
        win.img_view.createRoi("rectangle", rect=(0, 0, 2, 2))
        _process_events(qtbot, count=20)
        # The committed cached payload owns the visible frame's value
        # semantics. Preserve that truthful value (10 here) while the
        # successor is incomplete; never sample a black loading placeholder.
        assert "µ=10" in win.img_view._roi_info_panel.text()
        truthful_text = win.img_view._roi_info_panel.text()

        second_state = win.view_state.with_axis_range(2, indices=(2, 3), text="2:4")
        win._set_view_state(second_state)
        # The semantic state now leads the still-committed frame session.
        # Refreshing inspection at this boundary must retain the committed
        # values until update_image_view installs and completes a successor.
        win._refresh_inspection_dock()
        _process_events(qtbot, count=20)

        assert win._montage_roi_values_pending()
        assert win.img_view._roi_info_panel.text() == truthful_text
        assert "µ=0" not in win.img_view._roi_info_panel.text()
    finally:
        win.close()
        settings.setValue("image_rendering_backend", "pyqtgraph")
        settings.sync()


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
        win.renderer._frame_planner_instance = FramePlanner(internal_tile_shape=(4, 4))
        win.render(reason="test-vispy-tiled-roi")
        _process_events(qtbot, count=30)
        assert getattr(win._committed_display_frame, "is_tiled", False)

        win.layout_manager.set_managed_dock_visible(win.inspection_dock, False, reason="test", preserve_canvas=False)
        win.img_view.createRoi("rectangle", rect=(1, 1, 3, 3))
        _process_events(qtbot, count=20)

        assert win.img_view._roi_info_panel is not None
        assert "n=9" in win.img_view._roi_info_panel.text()
        assert "µ=18" in win.img_view._roi_info_panel.text()

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
        assert model.data(model.index(0, 3)) == "36"
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

        win.operation_evaluator.clear_cache()
        win.render(reason="inspection-refresh-test")
        _process_events(qtbot, count=10)

        assert len(calls) == 0
    finally:
        win.close()


def test_montage_viewport_updates_recompute_roi_stats_only_when_layout_changes(qtbot, monkeypatch):
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
        win.update_image_view()
        qtbot.waitUntil(lambda: len(calls) >= 2, timeout=3000)
        _process_events(qtbot, count=40)
        calls_after_layout = len(calls)
        win.update_image_view()
        _process_events(qtbot, count=40)

        assert len(calls) == calls_after_layout
        assert calls[0][0][3].columns != calls[-1][0][3].columns
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
        win.update_image_view()
        _process_events(qtbot, count=40)
        win.update_image_view()
        _process_events(qtbot, count=40)

        assert table_updates == []
        assert histogram_updates == []
        assert overlay_updates == []
        assert win.inspection_dock.roi_model.rowCount() == 0
    finally:
        win.close()
