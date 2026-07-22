"""Dock lifecycle regressions: detach survives sync (BUG 1), reopen restores
float state (BUG 2), and closing the inspection dock quiesces the ROI
pipeline (BUG 3)."""

import numpy as np
import pytest

from tests.ui.helpers import (
    clear_arrayscope_settings as _clear_arrayscope_settings,
)
from tests.ui.helpers import (
    give_generous_work_area as _give_generous_work_area,
)
from tests.ui.helpers import (
    process_events as _process_events,
)
from tests.ui.helpers import (
    view_action as _view_action,
)

# name (panel_manager key), View-menu text, window dock attribute.
_MANAGED_DOCKS = [
    ("profile", "Profile", "profile_dock"),
    ("inspection", "Inspection", "inspection_dock"),
    ("operations", "Operations", "operation_dock"),
]


def _open_dock(win, dock):
    win.layout_manager.set_managed_dock_visible(dock, True, reason="test", preserve_canvas=False)


# --- BUG 1: detaching a dock must survive render/progressive-sync ------------


@pytest.mark.parametrize(("name", "action_text", "dock_attr"), _MANAGED_DOCKS)
def test_detached_dock_survives_render_and_progressive_sync(qtbot, name, action_text, dock_attr):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow
    from arrayscope.window.panels import PanelLocation

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    _give_generous_work_area(win)
    try:
        _process_events(qtbot, count=15)
        dock = getattr(win, dock_attr)
        _open_dock(win, dock)
        _process_events(qtbot, count=15)
        win.layout_manager.detach_managed_dock(dock, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=15)

        assert win.panel_manager.location(name) == PanelLocation.DETACHED
        action = _view_action(win, action_text)
        assert action.isChecked()

        # The renderer and the progressive-sync policy run constantly; neither
        # may tear down a floating panel behind the user's back.
        for _ in range(4):
            win.render(reason="bug1-detach-survives")
            win.layout_manager.sync_progressive_docks()
            _process_events(qtbot, count=15)

        assert win.panel_manager.location(name) == PanelLocation.DETACHED
        panel = win.panel_manager.panel_for_dock(dock)
        assert panel.dialog is not None
        assert not panel.dialog.isHidden()
        assert action.isChecked()
    finally:
        win.close()


# --- BUG 2: reopening restores the last OPEN presentation + geometry ---------


@pytest.mark.parametrize(("name", "action_text", "dock_attr"), _MANAGED_DOCKS)
def test_reopen_restores_float_state_and_geometry(qtbot, name, action_text, dock_attr):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    from arrayscope.window import ArrayScopeWindow
    from arrayscope.window.panels import PanelLocation

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    _give_generous_work_area(win)
    try:
        _process_events(qtbot, count=15)
        dock = getattr(win, dock_attr)
        action = _view_action(win, action_text)
        if not action.isChecked():
            action.trigger()
            _process_events(qtbot, count=15)

        win.layout_manager.detach_managed_dock(dock, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=15)
        panel = win.panel_manager.panel_for_dock(dock)
        assert panel.dialog is not None

        panel.dialog.setGeometry(QtCore.QRect(140, 110, 360, 280))
        _process_events(qtbot, count=10)
        target = QtCore.QRect(panel.dialog.geometry())

        panel.dialog.close()
        _process_events(qtbot, count=15)
        assert win.panel_manager.location(name) == PanelLocation.HIDDEN

        action.trigger()
        _process_events(qtbot, count=15)

        assert win.panel_manager.location(name) == PanelLocation.DETACHED
        restored = win.panel_manager.panel_for_dock(dock).dialog
        assert restored is not None
        geo = restored.geometry()
        assert abs(geo.x() - target.x()) <= 4
        assert abs(geo.y() - target.y()) <= 4
        assert abs(geo.width() - target.width()) <= 4
        assert abs(geo.height() - target.height()) <= 4
    finally:
        win.close()


def test_float_state_is_not_persisted_across_sessions(qtbot):
    # Float state lives only in memory on ManagedPanel; a fresh window must
    # open every dock DOCKED, never restoring a previous session's float.
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    from arrayscope.window import ArrayScopeWindow
    from arrayscope.window.panels import PanelLocation

    first = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(first)
    _give_generous_work_area(first)
    try:
        _process_events(qtbot, count=15)
        action = _view_action(first, "Inspection")
        action.trigger()
        _process_events(qtbot, count=15)
        first.layout_manager.detach_managed_dock(
            first.inspection_dock, reason="test", preserve_canvas=False
        )
        _process_events(qtbot, count=15)
        panel = first.panel_manager.panel_for_dock(first.inspection_dock)
        panel.dialog.setGeometry(QtCore.QRect(140, 110, 360, 280))
        _process_events(qtbot, count=10)
        panel.dialog.close()
        _process_events(qtbot, count=15)
        first.close()
        _process_events(qtbot, count=10)
    finally:
        if first.isVisible():
            first.close()

    # Persisted settings carry no float geometry (only window_state /
    # viewport_session are written).
    settings = QtCore.QSettings()
    keys = list(settings.allKeys())
    assert not any("float" in key.lower() for key in keys)

    second = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(second)
    _give_generous_work_area(second)
    try:
        _process_events(qtbot, count=15)
        assert (
            second.panel_manager._panels_by_name["inspection"].last_open_location
            == PanelLocation.DOCKED
        )
        action = _view_action(second, "Inspection")
        action.trigger()
        _process_events(qtbot, count=15)
        assert second.panel_manager.location("inspection") == PanelLocation.DOCKED
    finally:
        second.close()
        _clear_arrayscope_settings()


# --- BUG 3: closing the inspection dock quiesces the ROI pipeline ------------


def test_closing_inspection_dock_bounds_and_quiesces_roi_refresh(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(40 * 40, dtype=float).reshape(40, 40))
    qtbot.addWidget(win)
    _give_generous_work_area(win)

    controller = win.roi_evaluation_controller
    submissions = []
    cleared = []
    original_start = controller.start_latest
    original_clear = controller.clear_group
    controller.start_latest = lambda *a, **k: (
        submissions.append(k.get("key")),
        original_start(*a, **k),
    )[1]
    controller.clear_group = lambda group: (cleared.append(group), original_clear(group))[1]

    # A live montage/tiled source produces a fresh dedup key on every commit,
    # so the request-level dedup never holds -- the exact condition that made
    # the ROI worker submit one job per commit. Simulate that churn so the test
    # exercises the runaway rather than a coincidentally-stable key.
    key_counter = {"n": 0}

    def _churning_key(_selections):
        key_counter["n"] += 1
        return ("montage-demand-sim", key_counter["n"])

    win._roi_inspection_key = _churning_key

    try:
        _process_events(qtbot, count=20)
        win.layout_manager.set_managed_dock_visible(
            win.inspection_dock, True, reason="test", preserve_canvas=False
        )
        _process_events(qtbot, count=10)
        win.img_view.createRoi("rectangle", rect=(2, 2, 6, 6))
        _process_events(qtbot, count=20)
        assert any(selection.enabled for selection in win.img_view.roiSelections())

        cleared.clear()
        win.layout_manager.set_inspection_dock_visible_from_user(False)
        _process_events(qtbot, count=15)

        # Closing the dock cancels the ROI evaluation group.
        assert "roi-inspection" in cleared

        # A live source that keeps committing must not saturate the ROI lane.
        # Each commit lands in its own event-loop turn (the case a refining
        # montage produces); without coalescing that is one ROI job per commit.
        # The single-shot timer collapses the whole burst to at most one job.
        submissions.clear()
        for _ in range(12):
            win._refresh_hidden_roi_overlay_from_committed_frame()
            qtbot.wait(5)
        _process_events(qtbot, count=25)

        assert len(submissions) <= 1
    finally:
        controller.start_latest = original_start
        controller.clear_group = original_clear
        win.close()
