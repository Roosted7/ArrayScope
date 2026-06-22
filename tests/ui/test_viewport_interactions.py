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


def test_render_preserves_viewport_for_same_display_shape(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        view = win.img_view.getView()
        view.setRange(xRange=(1, 3), yRange=(1, 3), padding=0)
        before = view.viewRange()

        win._on_slice_index_changed(2, 1)
        _process_events(qtbot, count=20)

        np.testing.assert_allclose(view.viewRange(), before, atol=1e-9)
    finally:
        win.close()


def test_toolbar_fit_and_one_to_one_are_viewport_commands(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.display.viewport import ViewportMode
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(20 * 30, dtype=float).reshape(20, 30))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        def fail_render(*args, **kwargs):
            raise AssertionError("Fit/1:1 must not render")

        win.render = fail_render
        win.display_toolbar.one_to_one_action.trigger()
        _process_events(qtbot, count=20)
        assert win.img_view.viewport_controller.mode == ViewportMode.USER
        win.display_toolbar.fit_action.trigger()
        _process_events(qtbot, count=20)
        assert win.img_view.viewport_controller.mode == ViewportMode.FIT
        assert not hasattr(win.display_toolbar, "aspect_combo")
    finally:
        win.close()


def test_fit_mode_pan_zoom_reminder_is_transient(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.display.viewport import ViewportMode
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(20 * 30, dtype=float).reshape(20, 30))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win.display_toolbar.fit_action.trigger()
        _process_events(qtbot, count=10)
        assert win.img_view.viewport_controller.mode == ViewportMode.FIT

        win.img_view._show_fit_mode_interaction_reminder()

        assert "Fit mode is enabled" in win.statusBar().currentMessage()
        qtbot.wait(1200)
        assert win.statusBar().currentMessage() == ""
    finally:
        win.close()


def test_one_to_one_is_one_shot_and_slice_updates_preserve_user_view(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.display.viewport import ViewportMode
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(10 * 12 * 5, dtype=float).reshape(10, 12, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win.display_toolbar.one_to_one_action.trigger()
        _process_events(qtbot, count=10)
        assert win.img_view.viewport_controller.mode == ViewportMode.USER

        view = win.img_view.getView()
        view.setRange(xRange=(2.0, 7.0), yRange=(3.0, 8.0), padding=0)
        before = view.viewRange()

        win._on_slice_index_changed(2, 3)
        _process_events(qtbot, count=30)

        np.testing.assert_allclose(view.viewRange(), before, atol=1e-9)
    finally:
        win.close()


def test_vispy_axis_direction_changes_sync_camera_orientation(qtbot):
    pytest.importorskip("vispy")

    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore
    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.window import ArrayScopeWindow

    win = None
    try:
        settings = QtCore.QSettings()
        settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.VISPY.value)
        settings.sync()

        win = ArrayScopeWindow(np.arange(20 * 30, dtype=np.float32).reshape(20, 30))
        qtbot.addWidget(win)
        _process_events(qtbot, count=20)
        assert win.img_view.rendering_backend_name == "vispy"
        y_dim, x_dim = win.view_state.image_axes

        win._set_view_state(win.view_state.with_axis_flipped(y_dim, True).with_axis_flipped(x_dim, True))
        win.apply_axis_flips()
        _process_events(qtbot)
        assert win.img_view.getView().state["xInverted"] is True
        assert win.img_view.getView().state["yInverted"] is False
        assert win.img_view._vispy_view.camera.flip == (True, False, False)

        win._set_view_state(win.view_state.with_axis_flipped(y_dim, False).with_axis_flipped(x_dim, False))
        win.apply_axis_flips()
        _process_events(qtbot)
        assert win.img_view.getView().state["xInverted"] is False
        assert win.img_view.getView().state["yInverted"] is True
        assert win.img_view._vispy_view.camera.flip == (False, True, False)
    finally:
        if win is not None:
            win.close()
        _clear_arrayscope_settings()


def test_dimension_axis_flip_is_view_transform_only(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        y_axis = int(win.view_state.image_axes[0])
        monkeypatch.setattr(win, "render", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("flip must not render synchronously")))
        monkeypatch.setattr(win, "request_render", lambda **kwargs: (_ for _ in ()).throw(AssertionError("flip must not request render")))

        win.set_dimension_role("y", y_axis)

        assert win.view_state.axis_flipped[y_axis] is True
    finally:
        win.close()
