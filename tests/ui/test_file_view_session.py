import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from arrayscope.window.file_view_session import FileViewSessionMixin
from arrayscope.window.viewport_continuity import ViewportContinuityTransaction
from tests.ui.helpers import clear_arrayscope_settings as _clear_arrayscope_settings


class _FakeFileSessionWindow(FileViewSessionMixin):
    def __init__(self, path, data, settings):
        self._filepath = path
        self._dataset_path = None
        self._selector_class_name = None
        self.base_data = data
        self._settings = settings

    def width(self):
        return 800

    def height(self):
        return 600

    def isMaximized(self):
        return False


def _restore_transaction(*, viewport=None, profile_visible=False):
    return ViewportContinuityTransaction(
        viewport=viewport,
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
        lambda _window, message, action_text, on_action, **_kwargs: actions.append(
            (message, action_text, on_action)
        ),
    )

    restored = window._restore_file_view_session_if_available()

    assert not restored
    assert actions
    assert actions[0][0] == "Saved view disabled for this file."
    assert actions[0][1] == "Enable"
    actions[0][2]()
    assert not settings.contains(key)


def test_nonproduction_file_view_sessions_are_scoped_by_application_name(qt_app):
    from pyqtgraph.Qt import QtCore

    from arrayscope.window.file_view_session import _file_view_session_config_dir

    previous_organization = qt_app.organizationName()
    previous_application = qt_app.applicationName()
    try:
        qt_app.setOrganizationName("ArrayScope")
        qt_app.setApplicationName("ArrayScopeProfileMontage")
        settings = QtCore.QSettings()

        assert _file_view_session_config_dir() == (
            Path(settings.fileName()).parent / "ArrayScopeProfileMontage"
        )
    finally:
        qt_app.setOrganizationName(previous_organization)
        qt_app.setApplicationName(previous_application)


def test_production_file_view_session_directory_remains_compatible(qt_app):
    from pyqtgraph.Qt import QtCore

    from arrayscope.window.file_view_session import _file_view_session_config_dir

    previous_organization = qt_app.organizationName()
    previous_application = qt_app.applicationName()
    try:
        qt_app.setOrganizationName("ArrayScope")
        qt_app.setApplicationName("ArrayScope")
        settings = QtCore.QSettings()

        assert _file_view_session_config_dir() == Path(settings.fileName()).parent
    finally:
        qt_app.setOrganizationName(previous_organization)
        qt_app.setApplicationName(previous_application)


def test_restored_roi_session_schedules_semantic_stats_refresh(qt_app, tmp_path):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.roi import RoiGeometry, RoiKind, RoiSelection
    from arrayscope.core.roi_store import RoiStore

    path = tmp_path / "scan.npy"
    data = np.zeros((4, 5), dtype=np.float32)
    np.save(path, data)
    window = _FakeFileSessionWindow(path, data, QtCore.QSettings())
    selection = RoiSelection(
        "roi-1", "ROI 1", RoiGeometry(RoiKind.RECTANGLE, rect=(50.0, 50.0, 4.0, 4.0))
    )
    refresh_reasons = []
    rows = []
    window.img_view = SimpleNamespace(
        setRoiSelections=lambda selections, *, selected_id=None: None,
        roiSelections=lambda: (selection,),
    )
    window.roi_store = RoiStore()
    window.inspection_dock = SimpleNamespace(
        set_rois=lambda selections: rows.append(tuple(selections))
    )
    window._schedule_refresh_inspection_dock = lambda reason: refresh_reasons.append(reason)

    window._restore_roi_session((selection,), selected_id=selection.id)

    assert rows[-1] == (selection,)
    assert refresh_reasons == ["file-session-restore"]


def test_restored_file_session_viewport_releases_continuity_after_apply(qt_app, tmp_path):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_session import ViewportSession
    from arrayscope.display.viewport import ViewportMode

    path = tmp_path / "scan.npy"
    data = np.zeros((4, 5), dtype=np.float32)
    np.save(path, data)
    window = _FakeFileSessionWindow(path, data, QtCore.QSettings())
    calls = []

    class View:
        def setRange(self, *, xRange, yRange, padding=0):
            calls.append((tuple(xRange), tuple(yRange), padding))

    controller = SimpleNamespace(
        mode=ViewportMode.USER,
        last_display_rect=(0.0, 0.0, 5.0, 4.0),
        fit=lambda _view: calls.append("fit"),
    )
    window.img_view = SimpleNamespace(
        getView=lambda: View(),
        viewport_controller=controller,
        _viewport_applying=False,
    )
    window.view_state = SimpleNamespace(montage_axis=None)
    window._committed_display_frame = SimpleNamespace(
        geometry=SimpleNamespace(display_shape=(4, 5))
    )
    window._is_committed_display_frame_current = lambda _frame: True
    window._suppress_montage_autofit_revert_message = True
    window._schedule_viewport_continuity_retarget = lambda: None
    window._viewport_continuity = _restore_transaction(
        viewport=ViewportSession(
            mode="user",
            view_range=((1.0, 3.0), (1.0, 3.0)),
            viewport_shape=(200, 300),
        )
    )
    window._viewport_continuity.message_enabled = False
    window._viewport_continuity.shape_settled = True

    window._apply_viewport_continuity_when_ready()
    window._apply_viewport_continuity_when_ready()

    assert calls == [((1.0, 3.0), (1.0, 3.0), 0)]
    assert window._viewport_continuity.applied
    assert window._viewport_continuity.released


def test_restored_file_session_viewport_rejects_invalid_range(qt_app, tmp_path):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_session import ViewportSession
    from arrayscope.display.viewport import ViewportMode

    path = tmp_path / "scan.npy"
    data = np.zeros((4, 5), dtype=np.float32)
    np.save(path, data)
    window = _FakeFileSessionWindow(path, data, QtCore.QSettings())
    calls = []
    view = SimpleNamespace(setRange=lambda **_kwargs: calls.append("range"))
    controller = SimpleNamespace(
        mode=ViewportMode.USER,
        last_display_rect=(0.0, 0.0, 5.0, 4.0),
        fit=lambda _view: calls.append("fit"),
    )
    window.img_view = SimpleNamespace(
        getView=lambda: view,
        viewport_controller=controller,
        _viewport_applying=False,
    )
    window.view_state = SimpleNamespace(montage_axis=None)
    window._committed_display_frame = SimpleNamespace(
        geometry=SimpleNamespace(display_shape=(4, 5))
    )
    window._is_committed_display_frame_current = lambda _frame: True
    window._suppress_montage_autofit_revert_message = True
    window._viewport_continuity = _restore_transaction(
        viewport=ViewportSession(
            mode="user",
            view_range=((1.0, 1.0), (0.0, 4.0)),
            viewport_shape=(200, 300),
        )
    )
    window._viewport_continuity.message_enabled = False

    window._apply_viewport_continuity_when_ready()

    assert calls == ["fit"]
    assert window._viewport_continuity.applied
    assert window._viewport_continuity.released


def test_restored_file_session_uses_restore_render_path(qtbot, monkeypatch):
    from arrayscope.window.main import ArrayScopeWindow

    calls = []
    monkeypatch.setattr(
        ArrayScopeWindow, "_restore_file_view_session_if_available", lambda self: True
    )
    monkeypatch.setattr(
        ArrayScopeWindow, "_pending_display_levels_for_render", lambda self: (0.0, 1.0)
    )
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


def test_restored_file_session_uses_viewport_shape_authority(qtbot, monkeypatch):
    from arrayscope.core.view_session import ViewportSession
    from arrayscope.window.layout_controller import WindowLayoutManager
    from arrayscope.window.main import ArrayScopeWindow

    calls = []

    def restore_session(self):
        self._viewport_continuity = _restore_transaction(
            viewport=ViewportSession(
                mode="user",
                view_range=((0.0, 1.0), (0.0, 1.0)),
                viewport_shape=(222, 333),
            )
        )
        return True

    monkeypatch.setattr(
        ArrayScopeWindow, "_restore_file_view_session_if_available", restore_session
    )
    monkeypatch.setattr(
        ArrayScopeWindow, "_pending_display_levels_for_render", lambda self: (0.0, 1.0)
    )
    monkeypatch.setattr(
        ArrayScopeWindow,
        "render",
        lambda self, *, reason=None, force_autolevel=False, defer_side_panels=False: None,
    )
    monkeypatch.setattr(
        WindowLayoutManager,
        "_restore_saved_viewport_session",
        lambda self: calls.append("settings"),
    )
    monkeypatch.setattr(
        WindowLayoutManager,
        "resize_to_dockless_viewport_shape",
        lambda self, shape: calls.append(("viewport", tuple(shape))) or True,
    )

    window = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(window)
    qtbot.wait(20)

    assert calls
    assert calls[0] == ("viewport", (222, 333))
    assert "settings" not in calls


def test_restored_file_session_uses_viewport_shape_without_general_settings(qtbot, monkeypatch):
    from arrayscope.core.view_session import ViewportSession
    from arrayscope.window.layout_controller import WindowLayoutManager
    from arrayscope.window.main import ArrayScopeWindow

    calls = []

    def restore_session(self):
        self._viewport_continuity = _restore_transaction(
            viewport=ViewportSession(
                mode="user",
                view_range=((0.0, 1.0), (0.0, 1.0)),
                viewport_shape=(222, 333),
            )
        )
        return True

    monkeypatch.setattr(
        ArrayScopeWindow, "_restore_file_view_session_if_available", restore_session
    )
    monkeypatch.setattr(
        ArrayScopeWindow, "_pending_display_levels_for_render", lambda self: (0.0, 1.0)
    )
    monkeypatch.setattr(
        ArrayScopeWindow,
        "render",
        lambda self, *, reason=None, force_autolevel=False, defer_side_panels=False: None,
    )
    monkeypatch.setattr(
        WindowLayoutManager,
        "_restore_saved_viewport_session",
        lambda self: calls.append("settings"),
    )
    monkeypatch.setattr(
        WindowLayoutManager,
        "resize_to_dockless_viewport_shape",
        lambda self, shape: calls.append(("viewport", tuple(shape))) or True,
    )

    window = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(window)
    qtbot.wait(20)

    assert calls
    assert calls[0] == ("viewport", (222, 333))
    assert "settings" not in calls


def test_settings_viewport_session_uses_viewport_continuity_transaction(qtbot, monkeypatch):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_session import ViewportSession, viewport_to_mapping
    from arrayscope.window.layout_controller import WindowLayoutManager
    from arrayscope.window.main import ArrayScopeWindow

    _clear_arrayscope_settings()
    settings = QtCore.QSettings()
    settings.setValue(
        "viewport_session",
        viewport_to_mapping(
            ViewportSession(
                mode="user",
                view_range=((1.0, 2.0), (3.0, 4.0)),
                viewport_shape=(222, 333),
                montage_columns=3,
            )
        ),
    )
    settings.sync()
    calls = []
    monkeypatch.setattr(
        WindowLayoutManager,
        "resize_to_dockless_viewport_shape",
        lambda self, shape: (
            calls.append((tuple(shape), getattr(self.window._viewport_continuity, "reason", None)))
            or True
        ),
    )

    window = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(window)
    qtbot.wait(20)

    assert calls
    assert calls[0] == ((222, 333), "settings-restore")
    _clear_arrayscope_settings()


def test_global_viewport_setting_stores_size_without_content_position(qtbot):
    """The app-wide viewport setting carries the render viewport SIZE only.

    ``view_range``/``mode``/``montage_columns`` locate a camera inside ONE
    dataset's world coordinates.  Replayed onto the next file they are
    meaningless, and a single bad capture poisons every file opened after it.
    The per-file view session already stores them, per file, where they mean
    something.
    """

    from pyqtgraph.Qt import QtCore

    from arrayscope.window import ArrayScopeWindow

    _clear_arrayscope_settings()
    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        qtbot.wait(20)
        win.layout_manager.save_window_settings()
    finally:
        win.close()

    stored = QtCore.QSettings().value("viewport_session")
    assert stored is not None
    assert stored["viewport_shape"] is not None, "the viewport size IS the global setting"
    assert stored["view_range"] is None
    assert stored["montage_columns"] is None
    _clear_arrayscope_settings()


def test_poisoned_global_viewport_range_never_reaches_the_camera(qtbot):
    """A stored no-content default must not park the camera on pixel (0, 0).

    Field defect (2026-07-21): x(-0.32, 1.32) y(0, 1) -- a ViewBox's pristine
    range before any content gives it bounds -- reached the global setting and
    was replayed onto a 336x336 slice.  The camera landed on data pixels
    (0,0)-(1,1), which are background, so the window rendered black on every
    backend and every present method.  Nothing was desynchronized: the hover
    readout said ``(0,0) = 0.0`` and the tile overlay sat where that camera
    put it, both truthfully.  Only the restored range was wrong.
    """

    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_session import ViewportSession, viewport_to_mapping
    from arrayscope.window import ArrayScopeWindow

    _clear_arrayscope_settings()
    settings = QtCore.QSettings()
    settings.setValue(
        "viewport_session",
        viewport_to_mapping(
            ViewportSession(
                mode="user",
                view_range=((-0.32274011299435035, 1.3227401129943503), (0.0, 1.0)),
                viewport_shape=(708, 1165),
                montage_columns=None,
            )
        ),
    )
    settings.sync()

    data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        qtbot.wait(200)
        (x0, x1), (y0, y1) = win.img_view.getView().viewRange()
    finally:
        win.close()
        _clear_arrayscope_settings()

    # The collapse showed 2 of 64 columns.  Anything that still frames the
    # data spans a large part of it; the exact fit depends on window aspect.
    assert (x1 - x0) > 16.0, f"camera collapsed to {(x1 - x0):.3f} of 64 columns"
    assert (y1 - y0) > 16.0, f"camera collapsed to {(y1 - y0):.3f} of 64 rows"


def test_viewport_session_omits_range_when_the_view_has_no_content_bounds(qt_app):
    """A ViewBox with nothing in it still reports a range; never store it.

    That placeholder range is what a saved viewport must never be made of --
    it is how the collapse above got recorded as though a user had chosen it.
    """

    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5), dtype=np.float32), None)
    window.img_view = SimpleNamespace(
        getView=lambda: SimpleNamespace(
            viewRange=lambda: [[-0.32274011299435035, 1.3227401129943503], [0.0, 1.0]]
        ),
    )
    window._committed_display_frame = None

    assert window._viewport_continuity_content_rect() is None
    assert window._current_viewport_session().view_range is None


def test_restored_file_session_defers_progressive_docks_until_window_is_visible(qtbot, monkeypatch):
    from arrayscope.core.view_session import ViewportSession
    from arrayscope.window.layout_controller import WindowLayoutManager
    from arrayscope.window.main import ArrayScopeWindow

    sync_calls = []

    def restore_session(self):
        self._viewport_continuity = _restore_transaction(
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

    monkeypatch.setattr(
        ArrayScopeWindow, "_restore_file_view_session_if_available", restore_session
    )
    monkeypatch.setattr(
        ArrayScopeWindow, "_pending_display_levels_for_render", lambda self: (0.0, 1.0)
    )
    monkeypatch.setattr(
        ArrayScopeWindow,
        "render",
        lambda self, *, reason=None, force_autolevel=False, defer_side_panels=False: None,
    )
    monkeypatch.setattr(
        WindowLayoutManager, "resize_to_dockless_viewport_shape", lambda self, shape: True
    )
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
        self._viewport_continuity = _restore_transaction(
            viewport=ViewportSession(
                mode="user",
                view_range=((0.0, 1.0), (0.0, 1.0)),
                viewport_shape=(222, 333),
            )
        )
        return True

    monkeypatch.setattr(
        ArrayScopeWindow, "_restore_file_view_session_if_available", restore_session
    )
    monkeypatch.setattr(
        ArrayScopeWindow, "_pending_display_levels_for_render", lambda self: (0.0, 1.0)
    )
    monkeypatch.setattr(
        ArrayScopeWindow,
        "render",
        lambda self, *, reason=None, force_autolevel=False, defer_side_panels=False: None,
    )
    monkeypatch.setattr(
        WindowLayoutManager, "resize_to_dockless_viewport_shape", lambda self, shape: True
    )
    monkeypatch.setattr(
        WindowLayoutManager, "resize_default_docks", lambda self: default_resize_calls.append(True)
    )

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
        self._viewport_continuity = _restore_transaction(
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
        dock_calls.append(
            (dock.objectName(), bool(visible), str(reason), bool(self.window.isVisible()))
        )
        return original_set_visible(
            self,
            dock,
            visible,
            reason=reason,
            preserve_canvas=preserve_canvas,
            raise_dock=raise_dock,
        )

    monkeypatch.setattr(
        ArrayScopeWindow, "_restore_file_view_session_if_available", restore_session
    )
    monkeypatch.setattr(
        ArrayScopeWindow, "_pending_display_levels_for_render", lambda self: (0.0, 1.0)
    )
    monkeypatch.setattr(
        ArrayScopeWindow,
        "render",
        lambda self, *, reason=None, force_autolevel=False, defer_side_panels=False: None,
    )
    monkeypatch.setattr(
        WindowLayoutManager, "resize_to_dockless_viewport_shape", lambda self, shape: True
    )
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
    window.isVisible = lambda: True
    window.view_state = SimpleNamespace(montage_axis=2)
    window._frame_session = SimpleNamespace(plan=object(), display_committed=False)
    window._current_montage_geometry = None
    window.img_view = SimpleNamespace(
        _viewport_applying=False,
        getView=lambda: view,
        viewport_controller=SimpleNamespace(mode=None),
    )
    window._viewport_continuity = _restore_transaction(
        viewport=ViewportSession(
            mode="user",
            view_range=((10.0, 20.0), (30.0, 40.0)),
        )
    )
    window.retarget_montage_viewport = lambda: scheduled.append("retargeted")
    monkeypatch.setattr(file_view_session, "show_revert_action", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        file_view_session.Qt.QtCore.QTimer,
        "singleShot",
        lambda delay, receiver, callback: single_shots.append((delay, callback)),
    )

    window._apply_viewport_continuity_when_ready()

    assert view.ranges == [((10.0, 20.0), (30.0, 40.0), 0)]
    assert window._viewport_continuity.range_applied
    assert not window._viewport_continuity.released
    assert not scheduled
    assert len(single_shots) == 1
    assert single_shots[0][0] == 0
    single_shots[0][1]()
    assert scheduled == ["retargeted"]


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
    window._frame_session = SimpleNamespace(plan=object(), display_committed=False)
    window._current_montage_geometry = None
    window.img_view = SimpleNamespace(
        _viewport_applying=False,
        getView=lambda: view,
        viewport_controller=controller,
    )
    window._viewport_continuity = _restore_transaction(
        viewport=ViewportSession(
            mode="auto_untouched",
            view_range=((10.0, 20.0), (30.0, 40.0)),
        )
    )
    window._schedule_viewport_continuity_retarget = lambda: None
    monkeypatch.setattr(file_view_session, "show_revert_action", lambda *_args, **_kwargs: None)

    window._apply_viewport_continuity_when_ready()

    assert view.ranges == [((10.0, 20.0), (30.0, 40.0), 0)]
    assert controller.mode is ViewportMode.USER
    assert controller.last_auto_view_range is None


def test_restored_viewport_continuity_survives_pending_viewport_shape(qt_app, monkeypatch):
    import arrayscope.window.file_view_session as file_view_session
    from arrayscope.core.view_session import ViewportSession

    class FakeView:
        def __init__(self):
            self.ranges = []

        def setRange(self, *, xRange, yRange, padding):
            self.ranges.append((tuple(xRange), tuple(yRange), padding))

    view = FakeView()
    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5, 6), dtype=np.float32), None)
    window.isVisible = lambda: True
    window.view_state = SimpleNamespace(montage_axis=2)
    window._frame_session = SimpleNamespace(plan=object(), display_committed=False)
    window._current_montage_geometry = None
    window.img_view = SimpleNamespace(
        _viewport_applying=False,
        getView=lambda: view,
        viewport_controller=SimpleNamespace(mode=None),
    )
    window._viewport_continuity = _restore_transaction(
        viewport=ViewportSession(
            mode="user",
            view_range=((10.0, 20.0), (30.0, 40.0)),
            viewport_shape=(222, 333),
        )
    )
    window._viewport_continuity.shape_settled = False
    window._schedule_viewport_continuity_retarget = lambda: None
    monkeypatch.setattr(file_view_session, "show_revert_action", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(window, "_viewport_continuity_shape_matches", lambda _shape: True)

    window._apply_viewport_continuity_when_ready()

    assert view.ranges == [((10.0, 20.0), (30.0, 40.0), 0)]
    assert window._viewport_continuity.applied
    assert not window._viewport_continuity.released

    window._viewport_continuity.shape_settled = True
    window._apply_viewport_continuity_when_ready()
    window._complete_viewport_continuity_if_settled()

    assert view.ranges == [
        ((10.0, 20.0), (30.0, 40.0), 0),
        ((10.0, 20.0), (30.0, 40.0), 0),
    ]
    assert window._viewport_continuity.released


def test_restored_viewport_shape_retry_reapplies_range_after_continuity_release(
    qt_app, monkeypatch
):
    import arrayscope.window.file_view_session as file_view_session
    from arrayscope.core.view_session import ViewportSession

    class FakeView:
        def __init__(self):
            self.ranges = []

        def setRange(self, *, xRange, yRange, padding):
            self.ranges.append((tuple(xRange), tuple(yRange), padding))

    view = FakeView()
    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5, 6), dtype=np.float32), None)
    window.isVisible = lambda: True
    window.view_state = SimpleNamespace(montage_axis=2)
    window._frame_session = SimpleNamespace(plan=object(), display_committed=True)
    window._current_montage_geometry = None
    window.img_view = SimpleNamespace(
        _viewport_applying=False,
        getView=lambda: view,
        viewport_controller=SimpleNamespace(mode=None),
    )
    window._viewport_continuity = _restore_transaction(
        viewport=ViewportSession(
            mode="user",
            view_range=((10.0, 20.0), (30.0, 40.0)),
            viewport_shape=(222, 333),
        )
    )
    window._viewport_continuity.applied = True
    window._viewport_continuity.released = True
    window._viewport_continuity.shape_settled = False
    window._schedule_viewport_continuity_retarget = lambda: None
    monkeypatch.setattr(file_view_session, "show_revert_action", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(window, "_viewport_continuity_shape_matches", lambda _shape: True)

    window._restore_viewport_continuity_shape_step(
        (222, 333),
        attempts=1,
        generation=window._viewport_continuity.generation,
    )

    assert view.ranges == [((10.0, 20.0), (30.0, 40.0), 0)]


def test_restored_viewport_shape_does_not_settle_in_same_turn_as_resize(qt_app, monkeypatch):
    import arrayscope.window.file_view_session as file_view_session
    from arrayscope.core.view_session import ViewportSession

    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5, 6), dtype=np.float32), None)
    window.isVisible = lambda: True
    window._viewport_continuity = _restore_transaction(
        viewport=ViewportSession(
            mode="user",
            view_range=None,
            viewport_shape=(222, 333),
        )
    )
    resize_results = iter((True, False))
    window.layout_manager = SimpleNamespace(
        resize_to_dockless_viewport_shape=lambda _shape: next(resize_results)
    )
    match_calls = []
    monkeypatch.setattr(
        window,
        "_viewport_continuity_shape_matches",
        lambda shape: match_calls.append(tuple(shape)) or True,
    )
    single_shots = []
    monkeypatch.setattr(
        file_view_session.Qt.QtCore.QTimer,
        "singleShot",
        lambda delay, receiver, callback: single_shots.append((delay, callback)),
    )

    window._restore_viewport_continuity_shape_step(
        (222, 333),
        attempts=3,
        generation=window._viewport_continuity.generation,
    )

    assert not window._viewport_continuity.shape_settled
    assert match_calls == []
    assert len(single_shots) == 1

    single_shots.pop()[1]()

    assert window._viewport_continuity.shape_settled
    assert window._viewport_continuity.released
    assert match_calls == [(222, 333)]


def test_viewport_continuity_reopens_shape_settle_after_dock_layout_change(qt_app, monkeypatch):
    import arrayscope.window.file_view_session as file_view_session
    from arrayscope.core.view_session import ViewportSession

    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5, 6), dtype=np.float32), None)
    window._viewport_continuity = _restore_transaction(
        viewport=ViewportSession(
            mode="user",
            view_range=((10.0, 20.0), (30.0, 40.0)),
            viewport_shape=(222, 333),
        )
    )
    window._viewport_continuity.shape_settled = True
    matches = [False]
    single_shots = []
    monkeypatch.setattr(
        window, "_viewport_continuity_shape_matches", lambda _shape: bool(matches[-1])
    )
    monkeypatch.setattr(
        file_view_session.Qt.QtCore.QTimer,
        "singleShot",
        lambda delay, receiver, callback: single_shots.append((delay, callback)),
    )

    window._restore_viewport_continuity_shape_after_layout()

    assert not window._viewport_continuity.shape_settled
    assert len(single_shots) == 1


def test_viewport_continuity_settled_shape_does_not_resize_for_pending_range(qt_app, monkeypatch):
    import arrayscope.window.file_view_session as file_view_session
    from arrayscope.core.view_session import ViewportSession

    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5, 6), dtype=np.float32), None)
    window._viewport_continuity = _restore_transaction(
        viewport=ViewportSession(
            mode="user",
            view_range=((10.0, 20.0), (30.0, 40.0)),
            viewport_shape=(222, 333),
        )
    )
    window._viewport_continuity.shape_settled = True
    single_shots = []
    monkeypatch.setattr(window, "_viewport_continuity_shape_matches", lambda _shape: True)
    monkeypatch.setattr(
        file_view_session.Qt.QtCore.QTimer,
        "singleShot",
        lambda delay, receiver, callback: single_shots.append((delay, callback)),
    )

    window._restore_viewport_continuity_shape_after_layout()

    assert window._viewport_continuity.shape_settled
    assert single_shots == []


def test_active_restored_viewport_is_saved_instead_of_live_aspect_adjusted_range(qt_app):
    from arrayscope.core.view_session import ViewportSession

    restored_viewport = ViewportSession(
        mode="user",
        view_range=((10.0, 200.0), (-50.0, 80.0)),
        viewport_shape=(120, 180),
        montage_columns=3,
    )
    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5, 6), dtype=np.float32), None)
    window._viewport_continuity = _restore_transaction(viewport=restored_viewport)

    assert window._current_viewport_session() == restored_viewport


def test_applied_viewport_continuity_does_not_lock_future_resize(qt_app):
    from arrayscope.core.view_session import ViewportSession

    restored_viewport = ViewportSession(
        mode="user",
        view_range=((10.0, 200.0), (-50.0, 80.0)),
        viewport_shape=(120, 180),
        montage_columns=3,
    )
    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5, 6), dtype=np.float32), None)
    tx = _restore_transaction(viewport=restored_viewport)
    tx.range_applied = True
    tx.shape_settled = True
    tx.released = False
    window._viewport_continuity = tx
    window._viewport_continuity_shape_matches = lambda _shape: False

    assert window._pending_viewport_continuity_range() is None
    assert window._pending_viewport_continuity_columns() is None
    assert window._active_viewport_continuity_range() is None


def test_montage_viewport_continuity_readiness_uses_canonical_frame_session(qt_app):
    window = _FakeFileSessionWindow(
        "unused.npy",
        np.zeros((4, 5, 6), dtype=np.float32),
        None,
    )
    window.view_state = SimpleNamespace(montage_axis=2)
    window._frame_session = SimpleNamespace(plan=object())

    assert window._viewport_continuity_ready()


@pytest.mark.parametrize("backend", ["pyqtgraph", "vispy"])
def test_manual_main_window_resize_releases_settled_pending_viewport_restore(
    qtbot,
    backend,
):
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.core.view_session import ViewportSession
    from arrayscope.window import ArrayScopeWindow

    _clear_arrayscope_settings()
    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", backend)
    settings.sync()
    window = ArrayScopeWindow(np.zeros((12, 10, 8), dtype=np.float32))
    qtbot.addWidget(window)
    try:
        window.show()
        qtbot.waitUntil(window.isVisible, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        viewport = window.img_view.graphicsView.viewport()
        view_range = window.img_view.getView().viewRange()
        tx = ViewportContinuityTransaction(
            reason="settings-restore",
            viewport=ViewportSession(
                mode="user",
                view_range=(tuple(view_range[0]), tuple(view_range[1])),
                viewport_shape=(int(viewport.height()), int(viewport.width())),
            ),
        )
        tx.shape_settled = True
        window._viewport_continuity = tx

        window.resize(max(window.minimumWidth(), window.width() - 40), window.height())

        qtbot.waitUntil(lambda: tx.released, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
    finally:
        window.close()
        settings.setValue(
            "image_rendering_backend",
            ImageRenderingBackendChoice.PYQTGRAPH.value,
        )
        settings.sync()


def test_programmatic_range_change_does_not_release_viewport_continuity(qt_app, monkeypatch):
    from pyqtgraph.Qt import QtCore

    import arrayscope.window.viewport_bridge as viewport_bridge
    from arrayscope.window.viewport_bridge import ViewportBridge

    released = []
    title_updates = []
    owner = SimpleNamespace(
        img_view=SimpleNamespace(_viewport_applying=False),
        _release_viewport_continuity=lambda: released.append(True),
        _note_viewport_interaction=lambda _reason: None,
        _update_display_group_title=lambda: title_updates.append(True),
        view_state=SimpleNamespace(montage_axis=None),
    )
    monkeypatch.setattr(
        viewport_bridge.Qt.QtWidgets.QApplication,
        "mouseButtons",
        lambda: QtCore.Qt.MouseButton.NoButton,
    )

    owner.win = owner
    ViewportBridge(owner).on_view_range_changed()

    assert released == []
    assert title_updates == []


def test_tiled_single_scene_range_change_schedules_frame_viewport_update(qt_app, monkeypatch):
    from pyqtgraph.Qt import QtCore

    import arrayscope.window.viewport_bridge as viewport_bridge
    from arrayscope.display.scene import DisplayLayout, DisplayScene
    from arrayscope.window.viewport_bridge import ViewportBridge

    scheduled = []
    owner = SimpleNamespace(
        img_view=SimpleNamespace(_viewport_applying=False),
        _release_viewport_continuity=lambda: None,
        _note_viewport_interaction=lambda _reason: None,
        _update_display_group_title=lambda: None,
        _committed_display_frame=SimpleNamespace(
            scene=DisplayScene(
                geometry=object(),
                layout=DisplayLayout.SINGLE,
                regions=(),
                bounds=(0.0, 0.0, 1.0, 1.0),
            ),
            value_source=SimpleNamespace(payloads={}),
        ),
        _schedule_frame_viewport_update=lambda *, delay_ms=None: scheduled.append(
            ("frame", delay_ms)
        ),
        view_state=SimpleNamespace(montage_axis=None),
    )
    monkeypatch.setattr(
        viewport_bridge.Qt.QtWidgets.QApplication,
        "mouseButtons",
        lambda: QtCore.Qt.MouseButton.NoButton,
    )

    owner.win = owner
    ViewportBridge(owner).on_view_range_changed()

    assert scheduled == [("frame", 0)]


def test_uncommitted_montage_range_defers_retarget_to_commit_teardown(qt_app, monkeypatch):
    """Pre-commit camera intent is deferred, never dropped.

    The montage-entry auto-fit range change arrives before any committed
    tiled frame; discarding it froze session.view_range (and LOD demand) at
    the stale entry fit until an unrelated retarget rescued it — the
    2026-07-19 pyqtgraph cold_fill demand-freshness red.
    """

    from pyqtgraph.Qt import QtCore

    import arrayscope.window.viewport_bridge as viewport_bridge
    from arrayscope.window.viewport_bridge import ViewportBridge

    retargeted = []
    owner = SimpleNamespace(
        img_view=SimpleNamespace(_viewport_applying=False),
        _release_viewport_continuity=lambda: None,
        _note_viewport_interaction=lambda _reason: None,
        _update_display_group_title=lambda: None,
        _committed_display_frame=None,
        _frame_session=SimpleNamespace(display_committed=True),
        _frame_viewport_retarget_after_commit=False,
        retarget_montage_viewport=lambda: retargeted.append(True),
        view_state=SimpleNamespace(montage_axis=2),
    )
    monkeypatch.setattr(
        viewport_bridge.Qt.QtWidgets.QApplication,
        "mouseButtons",
        lambda: QtCore.Qt.MouseButton.NoButton,
    )

    owner.win = owner
    ViewportBridge(owner).on_view_range_changed()

    assert retargeted == []
    assert owner._frame_viewport_retarget_after_commit is True


def test_uncommitted_single_view_range_records_no_montage_obligation(qt_app, monkeypatch):
    from pyqtgraph.Qt import QtCore

    import arrayscope.window.viewport_bridge as viewport_bridge
    from arrayscope.window.viewport_bridge import ViewportBridge

    owner = SimpleNamespace(
        img_view=SimpleNamespace(_viewport_applying=False),
        _release_viewport_continuity=lambda: None,
        _note_viewport_interaction=lambda _reason: None,
        _update_display_group_title=lambda: None,
        _committed_display_frame=None,
        _frame_viewport_retarget_after_commit=False,
        view_state=SimpleNamespace(montage_axis=None),
    )
    monkeypatch.setattr(
        viewport_bridge.Qt.QtWidgets.QApplication,
        "mouseButtons",
        lambda: QtCore.Qt.MouseButton.NoButton,
    )

    owner.win = owner
    ViewportBridge(owner).on_view_range_changed()

    assert owner._frame_viewport_retarget_after_commit is False


def test_wheel_range_change_uses_interactive_viewport_cadence(qt_app, monkeypatch):
    from pyqtgraph.Qt import QtCore

    import arrayscope.window.viewport_bridge as viewport_bridge
    from arrayscope.window.viewport_bridge import ViewportBridge

    scheduled = []
    image_view = SimpleNamespace(
        _viewport_applying=False,
        _viewport_wheel_range_pending=True,
    )
    owner = SimpleNamespace(
        img_view=image_view,
        _release_viewport_continuity=lambda: None,
        _note_viewport_interaction=lambda _reason: None,
        _committed_display_frame=SimpleNamespace(
            scene=object(),
            value_source=SimpleNamespace(payloads={}),
        ),
        _frame_session=SimpleNamespace(display_committed=True),
        _schedule_interactive_montage_viewport_update=lambda: scheduled.append(16),
        retarget_montage_viewport=lambda: None,
        view_state=SimpleNamespace(montage_axis=2),
    )
    owner.win = owner
    monkeypatch.setattr(
        viewport_bridge.Qt.QtWidgets.QApplication,
        "mouseButtons",
        lambda: QtCore.Qt.MouseButton.NoButton,
    )

    ViewportBridge(owner).on_view_range_changed()

    assert scheduled == [16]
    assert not image_view._viewport_wheel_range_pending


def test_montage_range_change_during_commit_joins_post_commit_retarget(qt_app, monkeypatch):
    from pyqtgraph.Qt import QtCore

    import arrayscope.window.viewport_bridge as viewport_bridge
    from arrayscope.window.viewport_bridge import ViewportBridge

    retargeted = []
    owner = SimpleNamespace(
        img_view=SimpleNamespace(_viewport_applying=False),
        _release_viewport_continuity=lambda: None,
        _note_viewport_interaction=lambda _reason: None,
        _committed_display_frame=SimpleNamespace(
            scene=object(),
            value_source=SimpleNamespace(payloads={}),
        ),
        _montage_presentation_commit_active=True,
        _frame_viewport_retarget_after_commit=False,
        retarget_montage_viewport=lambda: retargeted.append(True),
        view_state=SimpleNamespace(montage_axis=2),
    )
    owner.win = owner
    monkeypatch.setattr(
        viewport_bridge.Qt.QtWidgets.QApplication,
        "mouseButtons",
        lambda: QtCore.Qt.MouseButton.NoButton,
    )

    ViewportBridge(owner).on_view_range_changed()

    assert owner._frame_viewport_retarget_after_commit is True
    assert retargeted == []


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
    window._viewport_continuity = _restore_transaction(
        viewport=ViewportSession(
            mode="user",
            view_range=((1.0, 2.0), (3.0, 4.0)),
        )
    )
    monkeypatch.setattr(file_view_session, "show_revert_action", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        file_view_session.Qt.QtCore.QTimer,
        "singleShot",
        lambda delay, receiver, callback: single_shots.append((delay, callback)),
    )

    window._apply_viewport_continuity_when_ready()

    assert view.ranges == []
    assert single_shots == []
    window._committed_display_frame = "stale"
    window._apply_viewport_continuity_when_ready()
    assert view.ranges == []
    window._committed_display_frame = "current"
    window._apply_viewport_continuity_when_ready()
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
    window._frame_session = SimpleNamespace(plan=None, display_committed=False)
    window._current_montage_geometry = None
    window.img_view = SimpleNamespace(
        _viewport_applying=False,
        getView=lambda: view,
        viewport_controller=SimpleNamespace(mode=None),
    )
    window._viewport_continuity = _restore_transaction(
        viewport=ViewportSession(
            mode="user",
            view_range=((10.0, 20.0), (30.0, 40.0)),
        )
    )
    monkeypatch.setattr(file_view_session, "show_revert_action", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        file_view_session.Qt.QtCore.QTimer,
        "singleShot",
        lambda delay, receiver, callback: single_shots.append((delay, callback)),
    )

    window._apply_viewport_continuity_when_ready()

    assert view.ranges == []
    assert single_shots == []

    window._frame_session.plan = object()
    window._schedule_viewport_continuity_when_ready()

    assert len(single_shots) == 1
    single_shots[0][1]()
    assert view.ranges == [((10.0, 20.0), (30.0, 40.0), 0)]


def test_file_session_persists_viewport_session_only(qt_app):
    window = _FakeFileSessionWindow("unused.npy", np.zeros((4, 5), dtype=np.float32), None)
    window._current_file_session_metadata = lambda: {"path": "scan.npy"}
    window._file_view_session_persistence_disabled = lambda _metadata: False
    window._current_view_recipe = lambda: SimpleNamespace()
    window._current_viewport_session = lambda: None
    window.roi_store = SimpleNamespace(selections=(), selected_id=None)

    assert window._current_file_view_session().viewport is None
    assert window._current_file_view_session().panels.window_size == (800, 600)
