import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from arrayscope.window.file_view_session import FileViewSessionMixin


class _FakeFileSessionWindow(FileViewSessionMixin):
    def __init__(self, path, data, settings):
        self._filepath = path
        self._dataset_path = None
        self._selector_class_name = None
        self.base_data = data
        self._settings = settings


def test_file_view_session_disable_uses_existing_filename_key(qt_app, tmp_path, monkeypatch):
    from pyqtgraph.Qt import QtCore

    import arrayscope.window.file_view_session as file_view_session
    from arrayscope.core.view_session import settings_key_for_metadata

    path = tmp_path / "scan.npy"
    data = np.zeros((4, 5), dtype=np.float32)
    np.save(path, data)
    settings = QtCore.QSettings()
    window = _FakeFileSessionWindow(path, data, settings)
    key = settings_key_for_metadata(window._current_file_session_metadata())
    settings.setValue(key, "scan--abcdef123456.json")

    window._disable_current_file_view_session_persistence()

    assert settings.contains(key)
    assert settings.value(key) is None
    monkeypatch.setattr(
        file_view_session,
        "save_session_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not save")),
    )
    window._save_file_view_session_on_close()
    assert settings.value(key) is None

    window._enable_file_view_session_persistence()

    assert not settings.contains(key)


def test_disabled_file_view_session_load_shows_enable_action(qt_app, tmp_path, monkeypatch):
    from pyqtgraph.Qt import QtCore

    import arrayscope.window.file_view_session as file_view_session
    from arrayscope.core.view_session import settings_key_for_metadata

    path = tmp_path / "scan.npy"
    data = np.zeros((4, 5), dtype=np.float32)
    np.save(path, data)
    settings = QtCore.QSettings()
    window = _FakeFileSessionWindow(path, data, settings)
    key = settings_key_for_metadata(window._current_file_session_metadata())
    settings.setValue(key, None)
    actions = []
    monkeypatch.setattr(
        file_view_session,
        "show_status_action",
        lambda _window, message, action_text, on_action, **_kwargs: actions.append((message, action_text, on_action)),
    )

    restored = window._restore_file_view_session_if_available()

    assert not restored
    assert actions
    assert actions[0][0] == "Saved view disabled for this file."
    assert actions[0][1] == "Enable"
    actions[0][2]()
    assert not settings.contains(key)


def test_file_view_session_config_dir_is_next_to_qsettings(qt_app):
    from pyqtgraph.Qt import QtCore

    from arrayscope.window.file_view_session import _file_view_session_config_dir

    settings = QtCore.QSettings()

    assert _file_view_session_config_dir() == Path(settings.fileName()).parent


def test_restored_file_session_uses_restore_render_path(qtbot, monkeypatch):
    from arrayscope.window.main import ArrayScopeWindow

    calls = []
    monkeypatch.setattr(ArrayScopeWindow, "_restore_file_view_session_if_available", lambda self: True)
    monkeypatch.setattr(ArrayScopeWindow, "_pending_display_levels_for_render", lambda self: (0.0, 1.0))
    monkeypatch.setattr(
        ArrayScopeWindow,
        "render",
        lambda self, *, reason=None, force_autolevel=False: calls.append((reason, force_autolevel)),
    )

    window = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(window)

    assert calls == [("file-view-session-restore", False)]


def test_restored_montage_viewport_schedules_retarget_after_set_range(qt_app, monkeypatch):
    import arrayscope.window.file_view_session as file_view_session
    from arrayscope.core.view_session import ViewportSession

    class FakeView:
        def __init__(self):
            self.ranges = []

        def setRange(self, *, xRange, yRange, padding):
            self.ranges.append((tuple(xRange), tuple(yRange), padding))

    view = FakeView()
    scheduled = []
    single_shots = []
    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5, 6), dtype=np.float32), None)
    window.view_state = SimpleNamespace(montage_axis=2)
    window._montage_session = SimpleNamespace(display_committed=True)
    window._current_montage_geometry = object()
    window.img_view = SimpleNamespace(
        _viewport_applying=False,
        getView=lambda: view,
        viewport_controller=SimpleNamespace(mode=None),
    )
    window._pending_file_session_viewport = ViewportSession(
        mode="user",
        view_range=((10.0, 20.0), (30.0, 40.0)),
    )
    window._schedule_montage_viewport_update = lambda *, delay_ms=None: scheduled.append(delay_ms)
    monkeypatch.setattr(file_view_session, "show_revert_action", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(file_view_session.Qt.QtCore.QTimer, "singleShot", lambda delay, callback: single_shots.append((delay, callback)))

    window._apply_pending_file_session_viewport()

    assert view.ranges == [((10.0, 20.0), (30.0, 40.0), 0)]
    assert window._skip_next_montage_viewport_remap is True
    assert not scheduled
    assert len(single_shots) == 1
    assert single_shots[0][0] == 0
    single_shots[0][1]()
    assert scheduled == [0]


def test_restored_viewport_waits_until_window_is_visible(qt_app, monkeypatch):
    import arrayscope.window.file_view_session as file_view_session
    from arrayscope.core.view_session import ViewportSession

    class FakeView:
        def __init__(self):
            self.ranges = []

        def setRange(self, *, xRange, yRange, padding):
            self.ranges.append((tuple(xRange), tuple(yRange), padding))

    view = FakeView()
    visible = False
    single_shots = []
    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5), dtype=np.float32), None)
    window.isVisible = lambda: visible
    window.view_state = SimpleNamespace(montage_axis=None)
    window.img_view = SimpleNamespace(
        _viewport_applying=False,
        getView=lambda: view,
        viewport_controller=SimpleNamespace(mode=None),
    )
    window._pending_file_session_viewport = ViewportSession(
        mode="user",
        view_range=((1.0, 2.0), (3.0, 4.0)),
    )
    monkeypatch.setattr(file_view_session, "show_revert_action", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(file_view_session.Qt.QtCore.QTimer, "singleShot", lambda delay, callback: single_shots.append((delay, callback)))

    window._apply_pending_file_session_viewport()

    assert view.ranges == []
    assert single_shots == []
    visible = True
    window._apply_pending_file_session_viewport()
    assert view.ranges == [((1.0, 2.0), (3.0, 4.0), 0)]


def test_restored_montage_viewport_waits_for_committed_presentation(qt_app, monkeypatch):
    import arrayscope.window.file_view_session as file_view_session
    from arrayscope.core.view_session import ViewportSession

    class FakeView:
        def __init__(self):
            self.ranges = []

        def setRange(self, *, xRange, yRange, padding):
            self.ranges.append((tuple(xRange), tuple(yRange), padding))

    view = FakeView()
    single_shots = []
    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5, 6), dtype=np.float32), None)
    window.view_state = SimpleNamespace(montage_axis=2)
    window._montage_session = SimpleNamespace(display_committed=False)
    window._current_montage_geometry = None
    window.img_view = SimpleNamespace(
        _viewport_applying=False,
        getView=lambda: view,
        viewport_controller=SimpleNamespace(mode=None),
    )
    window._pending_file_session_viewport = ViewportSession(
        mode="user",
        view_range=((10.0, 20.0), (30.0, 40.0)),
    )
    monkeypatch.setattr(file_view_session, "show_revert_action", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(file_view_session.Qt.QtCore.QTimer, "singleShot", lambda delay, callback: single_shots.append((delay, callback)))

    window._apply_pending_file_session_viewport()

    assert view.ranges == []
    assert single_shots == []

    window._montage_session.display_committed = True
    window._current_montage_geometry = object()
    window._schedule_pending_file_session_viewport_restore()

    assert len(single_shots) == 1
    single_shots[0][1]()
    assert view.ranges == [((10.0, 20.0), (30.0, 40.0), 0)]


def test_file_session_viewport_size_overrides_current_window_size(qt_app):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_session import ViewportSession

    class FakeViewport:
        def __init__(self):
            self.size_value = QtCore.QSize(300, 200)

        def size(self):
            return QtCore.QSize(self.size_value)

    viewport = FakeViewport()
    refreshes = []
    resizes = []
    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5), dtype=np.float32), None)
    window._pending_file_session_viewport = ViewportSession(
        mode="user",
        view_range=((0.0, 1.0), (0.0, 1.0)),
        viewport_shape=(260, 420),
    )
    window.img_view = SimpleNamespace(
        graphicsView=SimpleNamespace(viewport=lambda: viewport),
    )
    window.isMaximized = lambda: False
    window.isFullScreen = lambda: False
    window.width = lambda: 640
    window.height = lambda: 480
    window.minimumSize = lambda: QtCore.QSize(320, 240)
    window.resize = lambda width, height: resizes.append((int(width), int(height)))
    window.layout_manager = SimpleNamespace(refresh_view_geometry=lambda: refreshes.append(True))

    assert window._apply_pending_file_session_viewport_size()

    assert resizes == [(760, 540)]
    assert refreshes == [True]
