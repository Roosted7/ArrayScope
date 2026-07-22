import numpy as np
import pytest

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import (
    clear_arrayscope_settings as _clear_arrayscope_settings,
)
from tests.ui.helpers import (
    process_events as _process_events,
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


@pytest.mark.parametrize("span", [40.0, 600.0], ids=["zoomed-in", "zoomed-out"])
def test_single_image_manual_resize_preserves_content_scale(qtbot, span):
    # A manual (USER) single-image view must keep its content scale (world
    # units per viewport pixel) across a window resize -- only the amount of
    # content revealed changes. Regression: the montage retarget was running
    # for non-montage images too and re-applied a conflicting camera, shrinking
    # the content. Covered zoomed IN and OUT because the bug hit both.
    _clear_arrayscope_settings()
    from arrayscope.display.viewport import ViewportMode
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.random.default_rng(0).random((200, 200)).astype(np.float32))
    qtbot.addWidget(win)
    try:
        win.resize(1000, 800)
        win.show()
        qtbot.waitExposed(win)
        _process_events(qtbot, count=30)
        view = win.img_view.getView()
        center = 100.0
        view.setRange(
            xRange=(center - span / 2, center + span / 2),
            yRange=(center - span / 2, center + span / 2),
            padding=0,
        )
        _process_events(qtbot, count=8)
        win.img_view.viewport_controller.mode = ViewportMode.USER

        def units_per_pixel():
            vp = win.img_view.graphicsView.viewport().size()
            r = view.viewRange()
            return (
                (r[0][1] - r[0][0]) / max(1, vp.width()),
                (r[1][1] - r[1][0]) / max(1, vp.height()),
            )

        before = units_per_pixel()
        win.resize(600, 780)
        _process_events(qtbot, count=30)
        after = units_per_pixel()

        assert after[0] == pytest.approx(before[0], rel=0.02)
        assert after[1] == pytest.approx(before[1], rel=0.02)
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
    from pyqtgraph.Qt import QtWidgets

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

        label = win.statusBar().findChild(QtWidgets.QLabel, "ArrayScopeStatusMessageLabel")
        assert label is not None
        assert "Fit mode is enabled" in label.text()
        qtbot.waitUntil(
            lambda: (
                win.statusBar().findChild(QtWidgets.QLabel, "ArrayScopeStatusMessageLabel") is None
            ),
            timeout=min(2000, INTERACTION_SETTLE_HARD_LIMIT_MS),
        )
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
        assert win.img_view.surface.capabilities.name == "vispy"
        y_dim, x_dim = win.view_state.image_axes

        win._set_view_state(
            win.view_state.with_axis_flipped(y_dim, True).with_axis_flipped(x_dim, True)
        )
        win.apply_axis_flips()
        _process_events(qtbot)
        assert win.img_view.getView().state["xInverted"] is True
        assert win.img_view.getView().state["yInverted"] is False
        assert win.img_view._vispy_view.camera.flip == (True, False, False)

        win._set_view_state(
            win.view_state.with_axis_flipped(y_dim, False).with_axis_flipped(x_dim, False)
        )
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
        monkeypatch.setattr(
            win,
            "render",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("flip must not render synchronously")
            ),
        )
        monkeypatch.setattr(
            win,
            "request_render",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("flip must not request render")),
        )

        win.set_dimension_role("y", y_axis)

        assert win.view_state.axis_flipped[y_axis] is True
    finally:
        win.close()
