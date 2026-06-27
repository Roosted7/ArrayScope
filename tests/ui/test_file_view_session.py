import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from arrayscope.window.file_view_session import FileSessionRestoreTransaction, FileViewSessionMixin


class _FakeFileSessionWindow(FileViewSessionMixin):
    def __init__(self, path, data, settings):
        self._filepath = path
        self._dataset_path = None
        self._selector_class_name = None
        self.base_data = data
        self._settings = settings


def _restore_transaction(*, viewport=None, window_size=None, profile_visible=False):
    return FileSessionRestoreTransaction(
        viewport=viewport,
        window_size=window_size,
        profile_visible=bool(profile_visible),
        defaults={"recipe": SimpleNamespace(display=SimpleNamespace(profile_visible=False))},
    )


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


def test_restored_roi_session_schedules_semantic_stats_refresh(qt_app, tmp_path):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.roi import RoiGeometry, RoiKind, RoiSelection
    from arrayscope.core.roi_store import RoiStore

    path = tmp_path / "scan.npy"
    data = np.zeros((4, 5), dtype=np.float32)
    np.save(path, data)
    window = _FakeFileSessionWindow(path, data, QtCore.QSettings())
    selection = RoiSelection("roi-1", "ROI 1", RoiGeometry(RoiKind.RECTANGLE, rect=(50.0, 50.0, 4.0, 4.0)))
    refresh_reasons = []
    rows = []
    window.img_view = SimpleNamespace(
        setRoiSelections=lambda selections, *, selected_id=None: None,
        roiSelections=lambda: (selection,),
    )
    window.roi_store = RoiStore()
    window.inspection_dock = SimpleNamespace(set_rois=lambda selections: rows.append(tuple(selections)))
    window._schedule_refresh_inspection_dock = lambda reason: refresh_reasons.append(reason)

    window._restore_roi_session((selection,), selected_id=selection.id)

    assert rows[-1] == (selection,)
    assert refresh_reasons == ["file-session-restore"]


def test_restored_file_session_uses_restore_render_path(qtbot, monkeypatch):
    from arrayscope.window.main import ArrayScopeWindow

    calls = []
    monkeypatch.setattr(ArrayScopeWindow, "_restore_file_view_session_if_available", lambda self: True)
    monkeypatch.setattr(ArrayScopeWindow, "_pending_display_levels_for_render", lambda self: (0.0, 1.0))
    monkeypatch.setattr(
        ArrayScopeWindow,
        "render",
        lambda self, *, reason=None, force_autolevel=False, defer_side_panels=False: calls.append(
            (reason, force_autolevel, defer_side_panels)
        ),
    )

    window = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(window)

    assert calls == [("file-view-session-restore", False, True)]


def test_restored_file_session_seeds_initial_size_from_session_window_size(qtbot, monkeypatch):
    from arrayscope.core.view_session import ViewportSession
    from arrayscope.window.layout_controller import WindowLayoutManager
    from arrayscope.window.main import ArrayScopeWindow

    calls = []

    def restore_session(self):
        self._file_session_restore = _restore_transaction(
            viewport=ViewportSession(
                mode="user",
                view_range=((0.0, 1.0), (0.0, 1.0)),
                viewport_shape=(222, 333),
            ),
            window_size=(700, 500),
        )
        return True

    monkeypatch.setattr(ArrayScopeWindow, "_restore_file_view_session_if_available", restore_session)
    monkeypatch.setattr(ArrayScopeWindow, "_pending_display_levels_for_render", lambda self: (0.0, 1.0))
    monkeypatch.setattr(
        ArrayScopeWindow,
        "render",
        lambda self, *, reason=None, force_autolevel=False, defer_side_panels=False: None,
    )
    monkeypatch.setattr(WindowLayoutManager, "_restore_saved_base_window_size", lambda self: calls.append("settings"))
    monkeypatch.setattr(
        WindowLayoutManager,
        "resize_to_dockless_window_size",
        lambda self, size: calls.append(("session", tuple(size))) or True,
    )

    window = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(window)

    assert calls == [("session", (700, 500))]


def test_restored_file_session_without_window_size_uses_viewport_shape(qtbot, monkeypatch):
    from arrayscope.core.view_session import ViewportSession
    from arrayscope.window.layout_controller import WindowLayoutManager
    from arrayscope.window.main import ArrayScopeWindow

    calls = []

    def restore_session(self):
        self._file_session_restore = _restore_transaction(
            viewport=ViewportSession(
                mode="user",
                view_range=((0.0, 1.0), (0.0, 1.0)),
                viewport_shape=(222, 333),
            )
        )
        return True

    monkeypatch.setattr(ArrayScopeWindow, "_restore_file_view_session_if_available", restore_session)
    monkeypatch.setattr(ArrayScopeWindow, "_pending_display_levels_for_render", lambda self: (0.0, 1.0))
    monkeypatch.setattr(
        ArrayScopeWindow,
        "render",
        lambda self, *, reason=None, force_autolevel=False, defer_side_panels=False: None,
    )
    monkeypatch.setattr(WindowLayoutManager, "_restore_saved_base_window_size", lambda self: calls.append("settings"))
    monkeypatch.setattr(
        WindowLayoutManager,
        "resize_to_dockless_window_size",
        lambda self, size: calls.append(("session", tuple(size))) or True,
    )
    monkeypatch.setattr(
        WindowLayoutManager,
        "resize_to_dockless_viewport_shape",
        lambda self, shape: calls.append(("viewport", tuple(shape))) or True,
    )

    window = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(window)

    assert calls
    assert calls[0] == ("viewport", (222, 333))
    assert "settings" not in calls


def test_restored_file_session_defers_progressive_docks_until_window_is_visible(qtbot, monkeypatch):
    from arrayscope.core.view_session import ViewportSession
    from arrayscope.window.layout_controller import WindowLayoutManager
    from arrayscope.window.main import ArrayScopeWindow

    sync_calls = []

    def restore_session(self):
        self._file_session_restore = _restore_transaction(
            viewport=ViewportSession(
                mode="user",
                view_range=((0.0, 1.0), (0.0, 1.0)),
                viewport_shape=(222, 333),
            )
        )
        return True

    def sync_progressive_docks(self, *, preserve_canvas=True):
        if bool(getattr(self.window, "_suspend_progressive_dock_sync", False)):
            return
        sync_calls.append((bool(preserve_canvas), bool(self.window.isVisible())))

    monkeypatch.setattr(ArrayScopeWindow, "_restore_file_view_session_if_available", restore_session)
    monkeypatch.setattr(ArrayScopeWindow, "_pending_display_levels_for_render", lambda self: (0.0, 1.0))
    monkeypatch.setattr(
        ArrayScopeWindow,
        "render",
        lambda self, *, reason=None, force_autolevel=False, defer_side_panels=False: None,
    )
    monkeypatch.setattr(WindowLayoutManager, "resize_to_dockless_window_size", lambda self, size: True)
    monkeypatch.setattr(WindowLayoutManager, "sync_progressive_docks", sync_progressive_docks)

    window = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(window)

    assert sync_calls
    assert all(call == (True, True) for call in sync_calls)


def test_restored_file_session_does_not_run_default_dock_resize_after_show(qtbot, monkeypatch):
    from arrayscope.core.view_session import ViewportSession
    from arrayscope.window.layout_controller import WindowLayoutManager
    from arrayscope.window.main import ArrayScopeWindow

    default_resize_calls = []

    def restore_session(self):
        self._file_session_restore = _restore_transaction(
            viewport=ViewportSession(
                mode="user",
                view_range=((0.0, 1.0), (0.0, 1.0)),
                viewport_shape=(222, 333),
            )
        )
        return True

    monkeypatch.setattr(ArrayScopeWindow, "_restore_file_view_session_if_available", restore_session)
    monkeypatch.setattr(ArrayScopeWindow, "_pending_display_levels_for_render", lambda self: (0.0, 1.0))
    monkeypatch.setattr(
        ArrayScopeWindow,
        "render",
        lambda self, *, reason=None, force_autolevel=False, defer_side_panels=False: None,
    )
    monkeypatch.setattr(WindowLayoutManager, "resize_to_dockless_window_size", lambda self, size: True)
    monkeypatch.setattr(WindowLayoutManager, "resize_default_docks", lambda self: default_resize_calls.append(True))

    window = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(window)
    qtbot.wait(20)

    assert default_resize_calls == []


def test_restored_file_session_defers_saved_profile_dock_visibility(qtbot, monkeypatch):
    from arrayscope.core.view_session import ViewportSession
    from arrayscope.window.layout_controller import WindowLayoutManager
    from arrayscope.window.main import ArrayScopeWindow
    from arrayscope.window.operation_actions import DisplaySettings

    dock_calls = []
    original_set_visible = WindowLayoutManager.set_managed_dock_visible

    def restore_session(self):
        self._file_session_restore = _restore_transaction(
            viewport=ViewportSession(
                mode="user",
                view_range=((0.0, 1.0), (0.0, 1.0)),
                viewport_shape=(222, 333),
            ),
            profile_visible=True,
        )
        settings = DisplaySettings(
            channel="real",
            scale="linear",
            aspect_mode="square_pixels",
            window_mode="relative",
            levels=None,
            profile_visible=True,
            live_profile=False,
        )
        self._apply_display_settings(settings)
        return True

    def set_visible(self, dock, visible, *, reason, preserve_canvas=True, raise_dock=True):
        dock_calls.append((dock.objectName(), bool(visible), str(reason), bool(self.window.isVisible())))
        return original_set_visible(
            self,
            dock,
            visible,
            reason=reason,
            preserve_canvas=preserve_canvas,
            raise_dock=raise_dock,
        )

    monkeypatch.setattr(ArrayScopeWindow, "_restore_file_view_session_if_available", restore_session)
    monkeypatch.setattr(ArrayScopeWindow, "_pending_display_levels_for_render", lambda self: (0.0, 1.0))
    monkeypatch.setattr(ArrayScopeWindow, "render", lambda self, *, reason=None, force_autolevel=False, defer_side_panels=False: None)
    monkeypatch.setattr(WindowLayoutManager, "resize_to_dockless_window_size", lambda self, size: True)
    monkeypatch.setattr(WindowLayoutManager, "set_managed_dock_visible", set_visible)

    window = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(window)
    qtbot.wait(20)

    assert ("ProfileDock", True, "view-recipe", False) not in dock_calls
    assert window._profile_dock_user_visible is True


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
    window._montage_session = SimpleNamespace(plan=object(), display_committed=False)
    window._current_montage_geometry = None
    window.img_view = SimpleNamespace(
        _viewport_applying=False,
        getView=lambda: view,
        viewport_controller=SimpleNamespace(mode=None),
    )
    window._file_session_restore = _restore_transaction(
        viewport=ViewportSession(
            mode="user",
            view_range=((10.0, 20.0), (30.0, 40.0)),
        )
    )
    window._schedule_montage_viewport_update = lambda *, delay_ms=None: scheduled.append(delay_ms)
    monkeypatch.setattr(file_view_session, "show_revert_action", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(file_view_session.Qt.QtCore.QTimer, "singleShot", lambda delay, callback: single_shots.append((delay, callback)))

    window._apply_file_session_viewport_when_ready()

    assert view.ranges == [((10.0, 20.0), (30.0, 40.0), 0)]
    assert window._skip_next_montage_viewport_remap is True
    assert not scheduled
    assert len(single_shots) == 1
    assert single_shots[0][0] == 0
    single_shots[0][1]()
    assert scheduled == [0]


def test_restored_montage_auto_range_reopens_as_user_camera(qt_app, monkeypatch):
    import arrayscope.window.file_view_session as file_view_session
    from arrayscope.core.view_session import ViewportSession
    from arrayscope.display.viewport import ViewportMode

    class FakeView:
        def __init__(self):
            self.ranges = []

        def setRange(self, *, xRange, yRange, padding):
            self.ranges.append((tuple(xRange), tuple(yRange), padding))

    view = FakeView()
    controller = SimpleNamespace(mode=None, last_auto_view_range=None)
    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5, 6), dtype=np.float32), None)
    window.view_state = SimpleNamespace(montage_axis=2)
    window._montage_session = SimpleNamespace(plan=object(), display_committed=False)
    window._current_montage_geometry = None
    window.img_view = SimpleNamespace(
        _viewport_applying=False,
        getView=lambda: view,
        viewport_controller=controller,
    )
    window._file_session_restore = _restore_transaction(
        viewport=ViewportSession(
            mode="auto_untouched",
            view_range=((10.0, 20.0), (30.0, 40.0)),
        )
    )
    window._schedule_file_session_viewport_retarget = lambda: None
    monkeypatch.setattr(file_view_session, "show_revert_action", lambda *_args, **_kwargs: None)

    window._apply_file_session_viewport_when_ready()

    assert view.ranges == [((10.0, 20.0), (30.0, 40.0), 0)]
    assert controller.mode is ViewportMode.USER
    assert controller.last_auto_view_range is None


def test_locked_restored_camera_is_saved_instead_of_live_aspect_adjusted_range(qt_app):
    from arrayscope.core.view_session import ViewportSession

    restored_viewport = ViewportSession(
        mode="user",
        view_range=((10.0, 200.0), (-50.0, 80.0)),
        viewport_shape=(120, 180),
        montage_columns=3,
    )
    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5, 6), dtype=np.float32), None)
    window._file_session_restore = _restore_transaction(viewport=restored_viewport)

    assert window._current_viewport_session() is restored_viewport


def test_programmatic_range_change_does_not_release_restored_camera_lock(qt_app, monkeypatch):
    from pyqtgraph.Qt import QtCore

    import arrayscope.window.viewport_bridge as viewport_bridge
    from arrayscope.window.viewport_bridge import ViewportBridge

    released = []
    owner = SimpleNamespace(
        img_view=SimpleNamespace(_viewport_applying=False),
        _release_file_session_restore_camera_lock=lambda: released.append(True),
        _note_viewport_interaction=lambda _reason: None,
        _update_display_group_title=lambda: None,
        view_state=SimpleNamespace(montage_axis=None),
    )
    monkeypatch.setattr(
        viewport_bridge.Qt.QtWidgets.QApplication,
        "mouseButtons",
        lambda: QtCore.Qt.MouseButton.NoButton,
    )

    ViewportBridge(owner).on_view_range_changed()

    assert released == []


def test_tiled_single_scene_range_change_schedules_tiled_viewport_update(qt_app, monkeypatch):
    from pyqtgraph.Qt import QtCore

    import arrayscope.window.viewport_bridge as viewport_bridge
    from arrayscope.display.scene import DisplayScene, DisplayLayout
    from arrayscope.window.viewport_bridge import ViewportBridge

    scheduled = []
    owner = SimpleNamespace(
        img_view=SimpleNamespace(_viewport_applying=False),
        _release_file_session_restore_camera_lock=lambda: None,
        _note_viewport_interaction=lambda _reason: None,
        _update_display_group_title=lambda: None,
        _committed_display_frame=SimpleNamespace(
            scene=DisplayScene(
                geometry=object(),
                layout=DisplayLayout.SINGLE,
                regions=(),
                bounds=(0.0, 0.0, 1.0, 1.0),
            )
        ),
        _schedule_tiled_viewport_update=lambda: scheduled.append("tiled"),
        view_state=SimpleNamespace(montage_axis=None),
    )
    monkeypatch.setattr(
        viewport_bridge.Qt.QtWidgets.QApplication,
        "mouseButtons",
        lambda: QtCore.Qt.MouseButton.NoButton,
    )

    ViewportBridge(owner).on_view_range_changed()

    assert scheduled == ["tiled"]


def test_restored_viewport_waits_until_frame_committed(qt_app, monkeypatch):
    import arrayscope.window.file_view_session as file_view_session
    from arrayscope.core.view_session import ViewportSession

    class FakeView:
        def __init__(self):
            self.ranges = []

        def setRange(self, *, xRange, yRange, padding):
            self.ranges.append((tuple(xRange), tuple(yRange), padding))

    view = FakeView()
    single_shots = []
    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5), dtype=np.float32), None)
    window.view_state = SimpleNamespace(montage_axis=None)
    window._committed_display_frame = None
    window._is_committed_display_frame_current = lambda frame: bool(frame == "current")
    window.img_view = SimpleNamespace(
        _viewport_applying=False,
        getView=lambda: view,
        viewport_controller=SimpleNamespace(mode=None),
    )
    window._file_session_restore = _restore_transaction(
        viewport=ViewportSession(
            mode="user",
            view_range=((1.0, 2.0), (3.0, 4.0)),
        )
    )
    monkeypatch.setattr(file_view_session, "show_revert_action", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(file_view_session.Qt.QtCore.QTimer, "singleShot", lambda delay, callback: single_shots.append((delay, callback)))

    window._apply_file_session_viewport_when_ready()

    assert view.ranges == []
    assert single_shots == []
    window._committed_display_frame = "stale"
    window._apply_file_session_viewport_when_ready()
    assert view.ranges == []
    window._committed_display_frame = "current"
    window._apply_file_session_viewport_when_ready()
    assert view.ranges == [((1.0, 2.0), (3.0, 4.0), 0)]


def test_restored_montage_viewport_waits_for_plan_not_tile_completion(qt_app, monkeypatch):
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
    window._montage_session = SimpleNamespace(plan=None, display_committed=False)
    window._current_montage_geometry = None
    window.img_view = SimpleNamespace(
        _viewport_applying=False,
        getView=lambda: view,
        viewport_controller=SimpleNamespace(mode=None),
    )
    window._file_session_restore = _restore_transaction(
        viewport=ViewportSession(
            mode="user",
            view_range=((10.0, 20.0), (30.0, 40.0)),
        )
    )
    monkeypatch.setattr(file_view_session, "show_revert_action", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(file_view_session.Qt.QtCore.QTimer, "singleShot", lambda delay, callback: single_shots.append((delay, callback)))

    window._apply_file_session_viewport_when_ready()

    assert view.ranges == []
    assert single_shots == []

    window._montage_session.plan = object()
    window._schedule_file_session_viewport_when_ready()

    assert len(single_shots) == 1
    single_shots[0][1]()
    assert view.ranges == [((10.0, 20.0), (30.0, 40.0), 0)]


def test_file_session_persists_dockless_window_size(qt_app):
    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5), dtype=np.float32), None)
    window._current_file_session_metadata = lambda: {"path": "scan.npy"}
    window._file_view_session_persistence_disabled = lambda _metadata: False
    window._current_view_recipe = lambda: SimpleNamespace()
    window._current_viewport_session = lambda: None
    window.roi_store = SimpleNamespace(selections=(), selected_id=None)
    window.layout_manager = SimpleNamespace(window_size_for_file_session=lambda: (700, 500))

    assert window._current_file_view_session().window_size == (700, 500)
