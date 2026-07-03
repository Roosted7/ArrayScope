from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.ui.toasts import show_status_action, show_status_message


class _Signal:
    def connect(self, _callback):
        pass


def test_plain_status_message_is_right_aligned(qtbot):
    win = QtWidgets.QMainWindow()
    qtbot.addWidget(win)
    win.resize(420, 120)
    win.show()
    win.statusBar().showMessage("old native left status", 5000)

    label = show_status_message(win, "Image rendering backend: pyqtgraph", timeout=0)
    qtbot.wait(10)

    rect = win.statusBar().contentsRect()
    assert win.statusBar().currentMessage() == ""
    assert label.objectName() == "ArrayScopeStatusMessageLabel"
    assert label.alignment() & QtCore.Qt.AlignmentFlag.AlignRight
    assert label.geometry().right() == rect.right()


def test_right_status_elides_after_full_action_when_lanes_do_not_fit(qtbot):
    win = QtWidgets.QMainWindow()
    qtbot.addWidget(win)
    win.resize(300, 120)
    win.show()

    action = show_status_action(
        win,
        "Restored saved view",
        "Revert",
        lambda: None,
        timeout=5000,
    )
    status = show_status_message(
        win,
        "Image rendering backend: pyqtgraph | offscreen Qt platform",
        timeout=0,
    )
    qtbot.wait(10)

    assert action.geometry().right() < status.geometry().left()
    assert "Restored saved view" in action.text()
    assert "..." not in action.text()
    assert "Revert" in action.text()
    assert status.text().endswith("...")
    assert status.alignment() & QtCore.Qt.AlignmentFlag.AlignRight
    assert status.geometry().right() == win.statusBar().contentsRect().right()


def test_display_builder_routes_backend_notify_to_right_status_lane(qtbot, monkeypatch):
    import arrayscope.ui.display_controls as display_controls

    class _FakeScene:
        sigMouseMoved = _Signal()

    class _FakeView:
        sigRangeChanged = _Signal()

        def scene(self):
            return _FakeScene()

    class _FakeGraphicsView:
        def __init__(self):
            self._viewport = QtWidgets.QWidget()

        def viewport(self):
            return self._viewport

    class _FakeImageView(QtWidgets.QWidget):
        roiCreated = _Signal()
        roiChanged = _Signal()
        roiDeleted = _Signal()
        imageContextMenuRequested = _Signal()
        userLevelsChanged = _Signal()
        autoWindowRequested = _Signal()

        def __init__(self):
            super().__init__()
            self.graphicsView = _FakeGraphicsView()

        def getView(self):
            return _FakeView()

        def setHudWidget(self, _widget):
            pass

        def set_profile_marker_callback(self, _callback):
            pass

    class _Window(display_controls.DisplayControlBuildMixin, QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.layouts = {"topDown": QtWidgets.QVBoxLayout()}
            self.app_settings = None
            self.resource_governor = None

        def _ui_work_decision(self, *_args, **_kwargs):
            return None

        def _submit_histogram_background_task(self, *_args, **_kwargs):
            return None

        def _on_level_presentation_changed(self, *_args, **_kwargs):
            pass

        def _on_image_mouse_moved(self, *_args, **_kwargs):
            pass

        def _on_profile_marker_moved(self, *_args, **_kwargs):
            pass

        def _on_roi_created(self, *_args, **_kwargs):
            pass

        def _on_roi_changed(self, *_args, **_kwargs):
            pass

        def _on_roi_deleted(self, *_args, **_kwargs):
            pass

        def _show_image_context_menu(self, *_args, **_kwargs):
            pass

        def _on_display_levels_changed(self, *_args, **_kwargs):
            pass

        def auto_window_levels(self, *_args, **_kwargs):
            pass

        def _on_view_range_changed(self, *_args, **_kwargs):
            pass

        def on_tab_changed(self, *_args, **_kwargs):
            pass

        def eventFilter(self, _obj, _event):
            return False

    def fake_create_image_view(_settings, *, notify):
        notify("Image rendering backend: pyqtgraph | offscreen Qt platform", timeout=5000)
        return _FakeImageView()

    monkeypatch.setattr(display_controls, "create_image_view", fake_create_image_view)
    win = _Window()
    qtbot.addWidget(win)
    win.resize(420, 120)
    win.show()

    win._build_main_canvas()
    qtbot.wait(10)

    label = win.statusBar().findChild(QtWidgets.QLabel, "ArrayScopeStatusMessageLabel")
    assert label is not None
    assert win.statusBar().currentMessage() == ""
    assert label.alignment() & QtCore.Qt.AlignmentFlag.AlignRight
    assert label.geometry().right() == win.statusBar().contentsRect().right()
