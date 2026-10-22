"""Operation history through real actions, rank changes, and linked windows."""

from __future__ import annotations

import gc
import uuid

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.operations.pipeline import ReverseAxis, SplitAxis
from arrayscope.operations.recipes import save_recipe
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.settle import settle_window
from tests.ui.helpers import frame_session_settled


@pytest.fixture
def windows(qtbot, monkeypatch):
    from shiboken6 import isValid

    monkeypatch.setenv("ARRAYSCOPE_SYNC_NAME", f"arrayscope-history-{uuid.uuid4().hex[:12]}")
    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", "pyqtgraph")
    tracked = []

    def track(win):
        tracked.append(win)
        win.show()
        _settle(win)
        return win

    yield track

    for win in tracked:
        if isValid(win):
            win.close()
            win.deleteLater()
    QtWidgets.QApplication.processEvents()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def _settle(win):
    settle_window(win, lambda: frame_session_settled(win), label="operation history frame settles")


def _new_window(windows, data=None, **kwargs):
    from arrayscope.window import ArrayScopeWindow

    if data is None:
        data = np.arange(4 * 6, dtype=np.float64).reshape(4, 6)
    return windows(ArrayScopeWindow(data, **kwargs))


def _key(qtbot, widget, standard):
    combination = QtGui.QKeySequence.keyBindings(standard)[0][0]
    qtbot.keyClick(widget, combination.key(), modifier=combination.keyboardModifiers())


def _wait(qtbot, predicate):
    qtbot.waitUntil(predicate, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)


def _link_operations(qtbot, *linked):
    for win in linked:
        win.operation_dock.sync_button.setChecked(True)
    _wait(
        qtbot,
        lambda: (
            all(win.sync_controller.bus.is_running() for win in linked)
            and max(win.sync_controller.bus.peer_count for win in linked) >= len(linked) - 1
        ),
    )


def test_keyboard_history_restores_stack_but_text_editor_undo_stays_local(windows, qtbot):
    win = _new_window(windows)
    win.request_operation("reverse", 0)
    _settle(win)
    edited = win.document.steps
    assert win.undo_operation_action.isEnabled()
    assert not win.redo_operation_action.isEnabled()
    gc.collect()
    QtWidgets.QApplication.processEvents()
    edit_menu = win.operation_edit_menu
    edit_action = edit_menu.menuAction()
    qtbot.mouseClick(
        win.menuBar(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=win.menuBar().actionGeometry(edit_action).center(),
    )
    _wait(qtbot, edit_menu.isVisible)
    assert win.undo_operation_action in edit_menu.actions()
    edit_menu.hide()
    win.activateWindow()
    win.setFocus()
    _wait(qtbot, win.isActiveWindow)

    _key(qtbot, win, QtGui.QKeySequence.StandardKey.Undo)

    assert win.document.steps == ()
    assert win.redo_operation_action.isEnabled()
    _key(qtbot, win, QtGui.QKeySequence.StandardKey.Redo)
    assert win.document.steps == edited
    win.request_operation("reverse", 1)
    win.undo_last_operation()
    assert win.document.steps == edited
    assert win.operation_coordinator.can_undo
    assert win.operation_coordinator.can_redo

    editor = QtWidgets.QLineEdit(win)
    editor.setText("before")
    editor.show()
    editor.raise_()
    editor.setFocus()
    qtbot.keyClick(editor, QtCore.Qt.Key.Key_End)
    qtbot.keyClicks(editor, " after")
    assert editor.text() == "before after"

    _key(qtbot, editor, QtGui.QKeySequence.StandardKey.Undo)

    assert editor.text() == "before"
    assert win.document.steps == edited
    assert win.operation_coordinator.can_undo
    assert win.operation_coordinator.can_redo
    _key(qtbot, editor, QtGui.QKeySequence.StandardKey.Redo)
    assert editor.text() == "before after"
    assert win.document.steps == edited
    assert win.operation_coordinator.can_undo
    assert win.operation_coordinator.can_redo


def test_rank_growing_sibling_can_undo_its_new_step_without_changing_origin(windows):
    origin = _new_window(windows)
    origin.request_operation("reverse", 0)
    _settle(origin)
    original_steps = origin.document.steps

    sibling = windows(origin._commit_operation("split_axis", 1, {"factor": 2}))

    assert sibling.document.current_shape == (4, 3, 2)
    sibling.undo_last_operation()
    _settle(sibling)
    assert sibling.document.steps == original_steps
    np.testing.assert_array_equal(sibling.document.materialize(), origin.base_data[::-1])
    assert origin.document.steps == original_steps
    sibling.redo_last_operation()
    _settle(sibling)
    np.testing.assert_array_equal(
        sibling.document.materialize(), origin.base_data[::-1].reshape(4, 3, 2)
    )
    assert sibling.per_dimension_widget_capacity() == 3
    assert not getattr(origin, "_closing", False)


def test_rank_growing_recipe_handoff_keeps_undo_and_never_renders_closed_origin(
    windows, tmp_path, monkeypatch
):
    from arrayscope.window import ArrayScopeWindow

    origin = _new_window(windows)
    origin.request_operation("reverse", 0)
    _settle(origin)
    previous = origin.document.steps
    path = tmp_path / "rank.json"
    save_recipe(path, (SplitAxis(1, 2),))
    before = set(QtWidgets.QApplication.topLevelWidgets())

    def refuse_origin_render(*_args, **_kwargs):
        pytest.fail("recipe history rendered the origin after handing off its rank")

    monkeypatch.setattr(origin, "render", refuse_origin_render)

    assert origin.load_operation_recipe_from_path(path)

    targets = [
        win
        for win in QtWidgets.QApplication.topLevelWidgets()
        if win not in before and isinstance(win, ArrayScopeWindow)
    ]
    assert len(targets) == 1
    target = windows(targets[0])
    assert origin._closing
    assert target.operation_coordinator.can_undo
    target.undo_last_operation()
    _settle(target)
    assert target.document.steps == previous
    np.testing.assert_array_equal(target.document.materialize(), target.base_data[::-1])
    target.redo_last_operation()
    _settle(target)
    np.testing.assert_array_equal(target.document.materialize(), target.base_data.reshape(4, 3, 2))


def test_external_rank_handoff_keeps_incoming_source_epoch_and_retires_old_history(windows):
    origin = _new_window(windows)
    origin.request_operation("reverse", 0)
    origin.request_operation("reverse", 1)
    origin.undo_last_operation()
    assert origin.operation_coordinator.can_undo
    assert origin.operation_coordinator.can_redo
    origin.base_data[:] += 100
    incoming = origin.document.mark_base_data_changed().with_operation(SplitAxis(1, 2))

    target = windows(origin._set_document(incoming))

    assert target.document.revision == incoming.revision
    assert target.document.steps == incoming.steps
    assert not target.operation_coordinator.can_undo
    assert not target.operation_coordinator.can_redo
    assert target.undo_last_operation() is False
    np.testing.assert_array_equal(
        target.document.materialize(), target.base_data[::-1].reshape(4, 3, 2)
    )


def test_remote_stack_is_one_undoable_transition_and_undo_publishes_back(windows, qtbot):
    first = _new_window(windows)
    second = _new_window(windows)
    _link_operations(qtbot, first, second)
    initial_epoch = second.document.revision
    received_states = []
    second.sync_controller.bus.messageReceived.connect(
        lambda message: received_states.append(message)
    )
    first.operation_coordinator.load_operations((ReverseAxis(0), ReverseAxis(1)))
    first._set_document(first.operation_coordinator.document)
    first.render(reason="test-history-remote-stack")
    _wait(qtbot, lambda: len(second.document.steps) == 2)
    _settle(second)
    assert second.operation_coordinator.undo_description == "Replace operation stack"
    assert second.document.revision == initial_epoch
    assert second.sync_controller._revisions["operations"] == 0

    second.undo_last_operation()

    _wait(qtbot, lambda: first.document.steps == ())
    _settle(first)
    assert second.document.steps == ()
    assert second.operation_coordinator.can_redo
    np.testing.assert_array_equal(first.document.materialize(), first.base_data)
    assert second.sync_controller._revisions["operations"] == 1
    assert any(message.get("facet") == "operations" for message in received_states)


def test_rank_growing_remote_stack_hands_off_without_rendering_closed_origin(
    windows, qtbot, monkeypatch
):
    from arrayscope.window import ArrayScopeWindow

    origin = _new_window(windows)
    before = set(QtWidgets.QApplication.topLevelWidgets())
    source = _new_window(windows, steps=(SplitAxis(1, 2),))
    before.add(source)
    original_render = origin.render
    closed_render_attempts = []

    def refuse_closed_render(*args, **kwargs):
        if getattr(origin, "_closing", False):
            # Sync catches apply errors and displays a status message, so a
            # raised assertion here alone would be swallowed by production.
            closed_render_attempts.append(kwargs.get("reason"))
            return None
        return original_render(*args, **kwargs)

    monkeypatch.setattr(origin, "render", refuse_closed_render)
    source.operation_dock.sync_button.setChecked(True)
    origin.operation_dock.sync_button.setChecked(True)
    _wait(qtbot, lambda: bool(getattr(origin, "_closing", False)))
    targets = [
        win
        for win in QtWidgets.QApplication.topLevelWidgets()
        if win not in before and isinstance(win, ArrayScopeWindow)
    ]
    assert len(targets) == 1
    target = windows(targets[0])
    assert closed_render_attempts == []
    assert target.document.current_shape == (4, 3, 2)
    assert target.operation_coordinator.can_undo
    target.undo_last_operation()
    _settle(target)
    assert target.document.steps == ()
    np.testing.assert_array_equal(target.document.materialize(), target.base_data)


@pytest.mark.parametrize("change", ["notify", "replace"])
def test_source_epoch_reset_disables_both_history_directions_in_the_ui(windows, change):
    win = _new_window(windows)
    win.request_operation("reverse", 0)
    win.request_operation("reverse", 1)
    win.undo_last_operation()
    assert win.undo_operation_action.isEnabled()
    assert win.redo_operation_action.isEnabled()
    epoch = win.document.revision

    if change == "notify":
        win.notify_data_changed()
    else:
        win._reset_data(win.base_data.copy())
    _settle(win)

    assert win.document.revision > epoch
    assert not win.undo_operation_action.isEnabled()
    assert not win.redo_operation_action.isEnabled()
    assert not win.operation_dock.undo_button.isEnabled()
    assert not win.operation_dock.redo_button.isEnabled()
    assert "reset" in win.undo_operation_action.toolTip().lower()
