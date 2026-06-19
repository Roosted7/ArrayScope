import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def process_events(qtbot, count=8):
    for _ in range(count):
        qtbot.wait(10)


def clear_arrayscope_settings():
    from pyqtgraph.Qt import QtCore

    settings = QtCore.QSettings()
    settings.clear()
    settings.sync()


def view_action(win, text):
    for action in win.menuBar().actions():
        if action.text() == "View":
            for child in action.menu().actions():
                if child.text() == text:
                    return child
    raise AssertionError(f"View action not found: {text}")


def panel_body(panel):
    return panel.body


def assert_panel_invariants(win, name, expected_location):
    from pyqtgraph.Qt import QtWidgets
    from arrayscope.window.panels import PanelLocation

    panel = win.panel_manager._panels_by_name[name]
    assert panel.location == expected_location
    if expected_location == PanelLocation.HIDDEN:
        assert panel.dialog is None
        assert not panel.dock.isVisible()
        assert panel.body is not None
    elif expected_location == PanelLocation.DETACHED:
        assert panel.dialog is not None
        assert panel.dialog.findChild(type(panel.body)) is panel.body or panel.body.parent() is not None
        assert not panel.dock.isVisible()
    elif expected_location == PanelLocation.DOCKED:
        assert panel.dialog is None
        assert panel.dock.isVisible()
        assert QtWidgets.QDockWidget.widget(panel.dock) is panel.body


def wait_for_panel_preserve(qtbot):
    process_events(qtbot, count=50)


def assert_size_close(actual, expected, tolerance=1):
    assert abs(actual.width() - expected.width()) <= tolerance
    assert abs(actual.height() - expected.height()) <= tolerance
