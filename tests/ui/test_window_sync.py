"""Linked-window sync between two real windows over the local-socket bus.

Both windows live in one process; the transport is the same QLocalServer/
QLocalSocket path that separately started ArrayScope processes use, so this
exercises the full publish -> broker -> apply chain including the sync
toggle buttons, origin/revision echo suppression, and clamping.
"""

import uuid

import numpy as np
import pytest

from tests.ui.helpers import clear_arrayscope_settings as _clear_arrayscope_settings

pytest.importorskip("pytestqt")


@pytest.fixture
def sync_group(monkeypatch):
    name = f"arrayscope-sync-uitest-{uuid.uuid4().hex[:12]}"
    monkeypatch.setenv("ARRAYSCOPE_SYNC_NAME", name)
    return name


@pytest.fixture
def make_window(qtbot, sync_group):
    windows = []

    def make(data):
        from arrayscope.window import ArrayScopeWindow

        win = ArrayScopeWindow(data)
        windows.append(win)
        qtbot.addWidget(win)
        return win

    _clear_arrayscope_settings()
    yield make
    for win in windows:
        win.close()


def _enable_all_facets(win):
    win.display_toolbar.sync_window_action.setChecked(True)
    win.sync_dims_button.setChecked(True)
    win.operation_dock.sync_button.setChecked(True)
    win.inspection_dock.sync_button.setChecked(True)


def _settled(qtbot, predicate, timeout=4000):
    qtbot.waitUntil(lambda: bool(predicate()), timeout=timeout)


def test_sync_buttons_toggle_controller_facets(qtbot, make_window):
    win = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    controller = win.sync_controller
    assert not any(controller.facet_enabled(facet) for facet in ("levels", "dims", "operations", "rois"))
    _enable_all_facets(win)
    assert all(controller.facet_enabled(facet) for facet in ("levels", "dims", "operations", "rois"))
    assert controller.bus.is_running()
    win.display_toolbar.sync_window_action.setChecked(False)
    win.sync_dims_button.setChecked(False)
    win.operation_dock.sync_button.setChecked(False)
    win.inspection_dock.sync_button.setChecked(False)
    assert not controller.bus.is_running()


def test_dimension_indexing_syncs_between_windows(qtbot, make_window):
    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.sync_dims_button.setChecked(True)
    win_b.sync_dims_button.setChecked(True)

    win_a.widgets["spins"]["slice_indices"][2].setValue(3)
    _settled(qtbot, lambda: win_b.view_state.slice_indices[2] == 3)
    # The mirrored spinbox follows the applied state.
    assert win_b.widgets["spins"]["slice_indices"][2].value() == 3


def test_dimension_sync_clamps_for_smaller_arrays_and_ignores_extra_dims(qtbot, make_window):
    win_a = make_window(np.arange(9 * 6 * 4, dtype=float).reshape(9, 6, 4))
    win_b = make_window(np.arange(5 * 3, dtype=float).reshape(5, 3))
    win_a.sync_dims_button.setChecked(True)
    win_b.sync_dims_button.setChecked(True)

    win_a.widgets["spins"]["slice_indices"][0].setValue(8)
    _settled(qtbot, lambda: win_b.view_state.slice_indices[0] == 4)  # clamped to size 5
    assert win_b.view_state.ndim == 2


def test_window_levels_sync_between_windows(qtbot, make_window):
    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.display_toolbar.sync_window_action.setChecked(True)
    win_b.display_toolbar.sync_window_action.setChecked(True)

    win_a.renderer._apply_display_level_override((10.0, 90.0), emit_user=True)
    _settled(
        qtbot,
        lambda: tuple(round(float(v), 3) for v in win_b.img_view.getLevels()) == (10.0, 90.0),
    )

    index = win_a.display_toolbar.window_combo.findData("absolute")
    win_a.display_toolbar.window_combo.setCurrentIndex(index)
    _settled(qtbot, lambda: win_b._current_window_mode() == "absolute")


def test_operations_sync_between_windows(qtbot, make_window):
    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.operation_dock.sync_button.setChecked(True)
    win_b.operation_dock.sync_button.setChecked(True)

    assert win_a.request_operation("mean", 2)
    _settled(qtbot, lambda: len(win_b.document.steps) == 1)
    operation = win_b.document.steps[0].operation
    assert type(operation).__name__.lower().startswith("mean")
    assert int(getattr(operation, "axis")) == 2

    win_a.clear_operations()
    _settled(qtbot, lambda: len(win_b.document.steps) == 0)


def test_incompatible_operation_recipe_is_skipped_without_error(qtbot, make_window):
    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(5 * 3, dtype=float).reshape(5, 3))
    win_a.operation_dock.sync_button.setChecked(True)
    win_b.operation_dock.sync_button.setChecked(True)

    # Axis 2 does not exist on the 2-D receiver; it must skip, not crash.
    assert win_a.request_operation("mean", 2)
    _settled(qtbot, lambda: len(win_a.document.steps) == 1)
    qtbot.wait(600)
    assert len(win_b.document.steps) == 0


def test_rois_sync_between_windows(qtbot, make_window):
    from arrayscope.core.roi import RoiKind

    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.inspection_dock.sync_button.setChecked(True)
    win_b.inspection_dock.sync_button.setChecked(True)

    selection = win_a.img_view.createRoi(RoiKind.RECTANGLE, rect=(1.0, 1.0, 2.0, 2.0))
    _settled(qtbot, lambda: len(win_b.roi_store.selections) == 1)
    mirrored = win_b.roi_store.selections[0]
    assert mirrored.id == selection.id
    assert mirrored.geometry.kind == RoiKind.RECTANGLE

    win_a._clear_rois()
    _settled(qtbot, lambda: len(win_b.roi_store.selections) == 0)


def test_joining_window_pulls_group_state(qtbot, make_window):
    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.sync_dims_button.setChecked(True)
    win_a.widgets["spins"]["slice_indices"][1].setValue(5)
    qtbot.wait(300)

    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    assert win_b.view_state.slice_indices[1] != 5
    win_b.sync_dims_button.setChecked(True)
    # Joining requests the group's state instead of pushing its own.
    _settled(qtbot, lambda: win_b.view_state.slice_indices[1] == 5)
    assert win_a.view_state.slice_indices[1] == 5


def test_sync_does_not_feedback_loop(qtbot, make_window):
    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.sync_dims_button.setChecked(True)
    win_b.sync_dims_button.setChecked(True)

    win_a.widgets["spins"]["slice_indices"][0].setValue(6)
    _settled(qtbot, lambda: win_b.view_state.slice_indices[0] == 6)
    qtbot.wait(400)
    revisions_a = dict(win_a.sync_controller._revisions)
    revisions_b = dict(win_b.sync_controller._revisions)
    qtbot.wait(600)
    assert dict(win_a.sync_controller._revisions) == revisions_a
    assert dict(win_b.sync_controller._revisions) == revisions_b
    assert win_a.view_state.slice_indices[0] == 6
    assert win_b.view_state.slice_indices[0] == 6
