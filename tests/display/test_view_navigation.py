import os

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from arrayscope.display.view_navigation import (
    begin_pan,
    drag_zoom_view_range,
    pan_view_range,
    pinch_zoom_view_range,
    scale_zoom_view_range,
    scroll_pan_view_range,
    wheel_zoom_view_range,
)


def test_pan_view_range_preserves_span_and_tracks_pixel_delta():
    gesture = begin_pan((100.0, 50.0), ((0.0, 200.0), (10.0, 110.0)), (400.0, 200.0))

    view_range = pan_view_range(gesture, (140.0, 70.0))

    assert view_range[0] == pytest.approx((-20.0, 180.0))
    assert view_range[1] == pytest.approx((0.0, 100.0))


def test_pan_view_range_honors_axis_inversion():
    x_flipped = begin_pan(
        (100.0, 50.0),
        ((0.0, 200.0), (10.0, 110.0)),
        (400.0, 200.0),
        x_inverted=True,
        y_inverted=True,
    )
    y_unflipped = begin_pan(
        (100.0, 50.0),
        ((0.0, 200.0), (10.0, 110.0)),
        (400.0, 200.0),
        x_inverted=False,
        y_inverted=False,
    )

    assert pan_view_range(x_flipped, (140.0, 70.0))[0] == pytest.approx((20.0, 220.0))
    assert pan_view_range(y_unflipped, (140.0, 70.0))[1] == pytest.approx((20.0, 120.0))


def test_wheel_zoom_view_range_preserves_focus_position():
    view_range = wheel_zoom_view_range(((0.0, 100.0), (0.0, 100.0)), (25.0, 75.0), 1.0)

    assert view_range[0] == pytest.approx((2.5, 92.5))
    assert view_range[1] == pytest.approx((7.5, 97.5))


def test_scale_zoom_view_range_keeps_focus_fixed():
    view_range = scale_zoom_view_range(((0.0, 100.0), (0.0, 100.0)), (50.0, 50.0), 0.5)

    assert view_range[0] == pytest.approx((25.0, 75.0))
    assert view_range[1] == pytest.approx((25.0, 75.0))


def test_drag_zoom_is_smooth_and_directional():
    start = ((0.0, 100.0), (0.0, 100.0))

    upward = drag_zoom_view_range(start, (50.0, 50.0), -20.0)
    downward = drag_zoom_view_range(start, (50.0, 50.0), 20.0)

    assert _span(upward)[0] < 100.0
    assert _span(downward)[0] > 100.0
    assert sum(upward[0]) / 2.0 == pytest.approx(50.0)


def test_pinch_zoom_magnifies_content_for_positive_value():
    # Fingers apart (value > 0) magnifies: the world span shrinks by 1/(1+value)
    # about the pinch point, which stays put.
    view_range = pinch_zoom_view_range(((0.0, 100.0), (0.0, 100.0)), (50.0, 50.0), 0.25)

    assert view_range[0] == pytest.approx((10.0, 90.0))
    assert view_range[1] == pytest.approx((10.0, 90.0))


def test_pinch_zoom_shrinks_content_for_negative_value():
    view_range = pinch_zoom_view_range(((0.0, 100.0), (0.0, 100.0)), (50.0, 50.0), -0.2)

    assert view_range[0] == pytest.approx((-12.5, 112.5))
    assert view_range[1] == pytest.approx((-12.5, 112.5))


def test_pinch_zoom_clamps_pathological_value():
    # value == -1 would divide by zero; the driver must stay finite.
    view_range = pinch_zoom_view_range(((0.0, 100.0), (0.0, 100.0)), (50.0, 50.0), -1.0)

    assert view_range[0] == pytest.approx((-200.0, 300.0))  # PINCH_SCALE_MAX == 5.0


def test_scroll_pan_matches_drag_direction_and_span():
    view_range = ((0.0, 200.0), (0.0, 100.0))

    # A two-finger scroll of (dx, dy) pixels pans exactly like dragging the
    # canvas by the same pixel vector, and never changes the span (no zoom).
    scrolled = scroll_pan_view_range(view_range, (40.0, -20.0), (400.0, 200.0))
    dragged = pan_view_range(begin_pan((0.0, 0.0), view_range, (400.0, 200.0)), (40.0, -20.0))

    assert scrolled[0] == pytest.approx(dragged[0])
    assert scrolled[1] == pytest.approx(dragged[1])
    assert scrolled[0][1] - scrolled[0][0] == pytest.approx(200.0)
    assert scrolled[1][1] - scrolled[1][0] == pytest.approx(100.0)


def _make_seeded_view(qt_app):
    """A base pyqtgraph view, seeded and unlocked from fit, ready for gestures.

    Touchpad gestures are handled in the shared ``ImageViewShell`` driver, so the
    lightweight pyqtgraph backend exercises the same code path as wgpu
    without a GPU.
    """

    from tests.display.test_imageview2d import _seed_displayed_image, _view_class

    view = _view_class("pyqtgraph")()
    view.resize(400, 400)
    _seed_displayed_image(view, np.zeros((64, 64), dtype=np.float32))
    view.show()
    qt_app.processEvents()
    view.setFitLocked(False)
    return view


def _viewport_center(view):
    from pyqtgraph.Qt import QtCore

    viewport = view.graphicsView.viewport()
    return QtCore.QPointF(viewport.width() / 2.0, viewport.height() / 2.0)


def _span(view_range):
    return (view_range[0][1] - view_range[0][0], view_range[1][1] - view_range[1][0])


def _touchpad_device(qt_app, system_id):
    from pyqtgraph.Qt import QtGui

    return QtGui.QPointingDevice(
        "test touchpad",
        system_id,
        QtGui.QInputDevice.DeviceType.TouchPad,
        QtGui.QPointingDevice.PointerType.Finger,
        QtGui.QInputDevice.Capability.Scroll | QtGui.QInputDevice.Capability.PixelScroll,
        10,
        0,
        parent=qt_app,
    )


def test_two_finger_scroll_pans_without_zooming(qt_app):
    from pyqtgraph.Qt import QtCore, QtGui

    view = _make_seeded_view(qt_app)
    try:
        viewport = view.graphicsView.viewport()
        before = view.getView().viewRange()
        emitted = []
        view.getView().scene().sigMouseMoved.connect(emitted.append)

        local = _viewport_center(view)
        cursor_global = viewport.mapToGlobal(local.toPoint())
        view._global_cursor_position = lambda: cursor_global
        gesture_local = local + QtCore.QPointF(-18.0, 31.0)
        device = _touchpad_device(qt_app, 101)
        event = QtGui.QWheelEvent(
            gesture_local,
            viewport.mapToGlobal(gesture_local.toPoint()),
            QtCore.QPoint(40, -20),  # pixelDelta -> a real touchpad scroll
            QtCore.QPoint(0, -5),  # fine-grained finger scroll, not a wheel notch
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.ScrollPhase.ScrollUpdate,
            False,
            QtCore.Qt.MouseEventSource.MouseEventNotSynthesized,
            device,
        )

        assert view.eventFilter(viewport, event) is True

        after = view.getView().viewRange()
        # The camera translated (pan) ...
        assert after[0][0] != pytest.approx(before[0][0])
        # ... but the span is unchanged (a scroll must not zoom).
        assert _span(after) == pytest.approx(_span(before))
        # Gesture-local coordinates may describe a finger centroid and can be
        # displaced from the mouse. The stationary physical cursor remains the
        # hover owner after the camera commit.
        assert len(emitted) == 1
        cursor_viewport = viewport.mapFromGlobal(cursor_global)
        expected = view.graphicsView.mapToScene(cursor_viewport)
        assert (emitted[0].x(), emitted[0].y()) == pytest.approx((expected.x(), expected.y()))
    finally:
        view.close()


def test_angle_only_momentum_uses_tuned_scroll_fallback(qt_app):
    """Keep angle-only touchpads usable without inventing gesture state.

    The event magnitude already carries platform acceleration/inertia; the
    compatibility path only maps one wheel step to the manually tuned 240 px.
    """

    from pyqtgraph.Qt import QtCore, QtGui

    from arrayscope.display.view_navigation_driver import _is_touchpad_scroll, _scroll_pixels

    local = QtCore.QPointF(10.0, 10.0)
    device = _touchpad_device(qt_app, 102)
    event = QtGui.QWheelEvent(
        local,
        local,
        QtCore.QPoint(),
        QtCore.QPoint(120, -60),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.ScrollMomentum,
        False,
        QtCore.Qt.MouseEventSource.MouseEventNotSynthesized,
        device,
    )

    assert _is_touchpad_scroll(event) is True
    assert _scroll_pixels(event) == pytest.approx((240.0, -120.0))


def test_calibrated_angle_boost_is_bounded_without_losing_pixel_only_motion(qt_app):
    """Wayland's two delta forms must not cause a large speed discontinuity."""

    from pyqtgraph.Qt import QtCore, QtGui

    from arrayscope.display.view_navigation_driver import _scroll_pixels

    local = QtCore.QPointF(10.0, 10.0)
    event = QtGui.QWheelEvent(
        local,
        local,
        QtCore.QPoint(7, -3),
        QtCore.QPoint(0, -5),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.ScrollUpdate,
        False,
    )

    # Horizontal has no angle step, so retain the native pixel delta. Vertical
    # calibrates to -10 px but is capped at 2x the native -3 px baseline.
    assert _scroll_pixels(event) == pytest.approx((7.0, -6.0))


def test_ctrl_two_finger_scroll_zooms_about_cursor(qt_app):
    from pyqtgraph.Qt import QtCore, QtGui

    view = _make_seeded_view(qt_app)
    try:
        viewport = view.graphicsView.viewport()
        before_span = _span(view.getView().viewRange())

        local = _viewport_center(view)
        device = _touchpad_device(qt_app, 103)
        event = QtGui.QWheelEvent(
            local,
            view.mapToGlobal(local.toPoint()),
            QtCore.QPoint(0, 30),
            QtCore.QPoint(0, 12),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.ControlModifier,
            QtCore.Qt.ScrollPhase.ScrollUpdate,
            False,
            QtCore.Qt.MouseEventSource.MouseEventNotSynthesized,
            device,
        )

        assert view.eventFilter(viewport, event) is True

        after_span = _span(view.getView().viewRange())
        assert after_span[0] < before_span[0]
        assert after_span[1] < before_span[1]
    finally:
        view.close()


def test_pinch_gesture_zooms_the_view(qt_app):
    from pyqtgraph.Qt import QtCore, QtGui

    view = _make_seeded_view(qt_app)
    try:
        viewport = view.graphicsView.viewport()
        before_span = _span(view.getView().viewRange())

        local = _viewport_center(view)
        device = QtGui.QPointingDevice.primaryPointingDevice()
        event = QtGui.QNativeGestureEvent(
            QtCore.Qt.NativeGestureType.ZoomNativeGesture,
            device,
            2,
            local,
            local,
            view.mapToGlobal(local.toPoint()),
            0.25,  # fingers apart -> magnify
            QtCore.QPointF(0.0, 0.0),
        )

        assert view.eventFilter(viewport, event) is True

        after_span = _span(view.getView().viewRange())
        assert after_span[0] < before_span[0]
        assert after_span[1] < before_span[1]
    finally:
        view.close()


def test_plain_mouse_wheel_is_left_to_the_wheel_zoom_path(qt_app):
    """A classic mouse wheel (no pixelDelta, no scroll phase) must fall through
    the gesture driver so the existing wheel-zoom owner keeps handling it."""

    from pyqtgraph.Qt import QtCore, QtGui

    view = _make_seeded_view(qt_app)
    try:
        local = _viewport_center(view)
        event = QtGui.QWheelEvent(
            local,
            view.mapToGlobal(local.toPoint()),
            QtCore.QPoint(0, 0),  # no pixelDelta -> not a touchpad
            QtCore.QPoint(0, 120),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.ScrollPhase.NoScrollPhase,
            False,
        )

        assert view._gesture_navigation.handle_event(event) is False
    finally:
        view.close()


def test_phase_tagged_mouse_wheel_stays_with_existing_zoom_owner(qt_app):
    """Wayland phase and pixel deltas are not, by themselves, a touchpad."""

    from pyqtgraph.Qt import QtCore, QtGui

    view = _make_seeded_view(qt_app)
    try:
        local = _viewport_center(view)
        event = QtGui.QWheelEvent(
            local,
            view.mapToGlobal(local.toPoint()),
            QtCore.QPoint(0, 15),
            QtCore.QPoint(0, 120),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.ScrollPhase.ScrollUpdate,
            False,
        )

        assert view._gesture_navigation.handle_event(event) is False
    finally:
        view.close()


def test_discrete_mouse_notch_wins_over_misreported_touchpad_identity(qt_app):
    """Wayland may attach a touchpad identity to a classic wheel event."""

    from pyqtgraph.Qt import QtCore, QtGui

    from arrayscope.display.view_navigation_driver import _is_touchpad_scroll

    local = QtCore.QPointF(10.0, 10.0)
    device = _touchpad_device(qt_app, 104)
    event = QtGui.QWheelEvent(
        local,
        local,
        QtCore.QPoint(0, 15),
        QtCore.QPoint(0, 120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.ScrollUpdate,
        False,
        QtCore.Qt.MouseEventSource.MouseEventNotSynthesized,
        device,
    )

    assert _is_touchpad_scroll(event) is False


def test_fit_locked_gesture_is_consumed_without_moving_camera(qt_app):
    from pyqtgraph.Qt import QtCore, QtGui

    view = _make_seeded_view(qt_app)
    try:
        view.setFitLocked(True)
        viewport = view.graphicsView.viewport()
        before = view.getView().viewRange()

        local = _viewport_center(view)
        device = QtGui.QPointingDevice.primaryPointingDevice()
        event = QtGui.QNativeGestureEvent(
            QtCore.Qt.NativeGestureType.ZoomNativeGesture,
            device,
            2,
            local,
            local,
            view.mapToGlobal(local.toPoint()),
            0.25,
            QtCore.QPointF(0.0, 0.0),
        )

        # Fit lock swallows the gesture (reminder path) and leaves the camera put.
        assert view.eventFilter(viewport, event) is True
        after = view.getView().viewRange()
        assert after[0] == pytest.approx(before[0])
        assert after[1] == pytest.approx(before[1])
    finally:
        view.close()


def _send_middle_mouse(view, event_type, pos, *, buttons):
    from pyqtgraph.Qt import QtCore, QtGui

    middle = QtCore.Qt.MouseButton.MiddleButton
    button = (
        middle if event_type != QtCore.QEvent.Type.MouseMove else QtCore.Qt.MouseButton.NoButton
    )
    event = QtGui.QMouseEvent(
        event_type,
        QtCore.QPointF(pos),
        button,
        buttons,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    return view.eventFilter(view.graphicsView.viewport(), event)


def test_middle_vertical_drag_zooms_without_dimension_steps(qt_app):
    from pyqtgraph.Qt import QtCore

    view = _make_seeded_view(qt_app)
    steps = []
    active = []
    view.setMiddleDragNavigationHandlers(
        target_provider=lambda: 2,
        step_handler=lambda axis, delta: steps.append((axis, delta)),
        active_handler=active.append,
    )
    try:
        middle = QtCore.Qt.MouseButton.MiddleButton
        start = _viewport_center(view)
        before_span = _span(view.getView().viewRange())

        assert _send_middle_mouse(view, QtCore.QEvent.Type.MouseButtonPress, start, buttons=middle)
        sample_scene = view.graphicsView.mapToScene(int(start.x()), int(start.y()))
        assert view._sampling_marker.isVisible()
        marker_geometry = view._sampling_marker.geometry()
        assert _send_middle_mouse(
            view,
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(start.x(), start.y() - 60.0),
            buttons=middle,
        )
        # The shared screen overlay stays at the press-time sample rather than
        # following either camera motion or the physical middle-drag cursor.
        assert view._sampling_marker.geometry() == marker_geometry
        expected_center = view._map_scene_to_display_overlay(sample_scene)
        assert view._sampling_marker.geometry().center() == expected_center
        assert _send_middle_mouse(
            view,
            QtCore.QEvent.Type.MouseButtonRelease,
            QtCore.QPointF(start.x(), start.y() - 60.0),
            buttons=QtCore.Qt.MouseButton.NoButton,
        )

        assert _span(view.getView().viewRange())[0] < before_span[0]
        assert steps == []
        assert active == [False]
        assert not view._sampling_marker.isVisible()
    finally:
        view.close()


def test_middle_horizontal_drag_steps_target_without_zooming(qt_app):
    from pyqtgraph.Qt import QtCore

    view = _make_seeded_view(qt_app)
    steps = []
    active = []
    view.setMiddleDragNavigationHandlers(
        target_provider=lambda: 2,
        step_handler=lambda axis, delta: steps.append((axis, delta)),
        active_handler=active.append,
    )
    try:
        middle = QtCore.Qt.MouseButton.MiddleButton
        start = _viewport_center(view)
        before = view.getView().viewRange()

        assert _send_middle_mouse(view, QtCore.QEvent.Type.MouseButtonPress, start, buttons=middle)
        assert _send_middle_mouse(
            view,
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(start.x() + 55.0, start.y()),
            buttons=middle,
        )
        assert _send_middle_mouse(
            view,
            QtCore.QEvent.Type.MouseButtonRelease,
            QtCore.QPointF(start.x() + 55.0, start.y()),
            buttons=QtCore.Qt.MouseButton.NoButton,
        )

        assert steps == [(2, 5)]
        assert active == [True, False]
        assert view.getView().viewRange()[0] == pytest.approx(before[0])
        assert view.getView().viewRange()[1] == pytest.approx(before[1])
    finally:
        view.close()


def test_middle_horizontal_lock_resets_provisional_zoom(qt_app):
    from pyqtgraph.Qt import QtCore

    view = _make_seeded_view(qt_app)
    steps = []
    view.setMiddleDragNavigationHandlers(
        target_provider=lambda: 2,
        step_handler=lambda axis, delta: steps.append((axis, delta)),
    )
    try:
        middle = QtCore.Qt.MouseButton.MiddleButton
        start = _viewport_center(view)
        before = view.getView().viewRange()
        before_mode = view.viewport_controller.mode

        assert _send_middle_mouse(view, QtCore.QEvent.Type.MouseButtonPress, start, buttons=middle)
        # Below the 6 px direction threshold, vertical motion responds at once.
        assert _send_middle_mouse(
            view,
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(start.x() + 1.0, start.y() - 3.0),
            buttons=middle,
        )
        provisional = view.getView().viewRange()
        assert provisional[0] != pytest.approx(before[0])

        # Horizontal motion wins, restores the exact press-time camera, then
        # crosses one calibrated 10 px index step.
        assert _send_middle_mouse(
            view,
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(start.x() + 12.0, start.y() - 3.0),
            buttons=middle,
        )
        assert view.getView().viewRange()[0] == pytest.approx(before[0])
        assert view.getView().viewRange()[1] == pytest.approx(before[1])
        assert view.viewport_controller.mode == before_mode
        assert steps == [(2, 1)]
    finally:
        view.close()


def test_pan_move_reemits_hover_at_moved_position(qt_app):
    """A pan MouseMove is consumed by QtViewNavigationDriver (accept + return
    True), so pyqtgraph's GraphicsScene never emits sigMouseMoved and the HUD
    hover readout would freeze at pan-start.  The driver must re-emit the hover
    at the moved position via the owner's _notify_pointer_drag_moved hook, the
    same fix already carried by the ROI/profile pointer driver.

    The base ImageView2D does not install _view_navigation, so drive the wgpu
    view (offscreen bitmap), which owns both the nav driver and the HUD wiring.
    """

    pytest.importorskip("wgpu")
    from pyqtgraph.Qt import QtCore, QtGui

    from tests.display.test_imageview2d import (
        _seed_displayed_image,
        _view_class,
        _viewport_pos_for_image_point,
    )

    try:
        from arrayscope.display.wgpu_imageview2d import import_qrenderwidget

        import_qrenderwidget()
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"rendercanvas unavailable: {exc}")

    view = _view_class("wgpu")()
    try:
        view.resize(400, 400)
        _seed_displayed_image(view, np.zeros((64, 64), dtype=np.float32))
        view.show()
        qt_app.processEvents()
        # Panning requires the camera to be unlocked from fit.
        view.setFitLocked(False)

        driver = view._view_navigation
        assert driver is not None

        emitted: list[tuple[float, float]] = []
        view.getView().scene().sigMouseMoved.connect(lambda pt: emitted.append((pt.x(), pt.y())))

        viewport = view.graphicsView.viewport()
        left = QtCore.Qt.MouseButton.LeftButton
        none_button = QtCore.Qt.MouseButton.NoButton

        def send(event_type, pos, *, button, buttons):
            event = QtGui.QMouseEvent(
                event_type,
                QtCore.QPointF(pos),
                button,
                buttons,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )
            return view.eventFilter(viewport, event)

        # Press at the image centre so the point lands inside the content rect
        # and the pan gesture actually begins.
        p0 = _viewport_pos_for_image_point(view, 32.0, 32.0)
        assert send(QtCore.QEvent.Type.MouseButtonPress, p0, button=left, buttons=left) is True
        assert driver.is_active() is True

        # Two pan moves to distinct viewport pixels.  The viewport->scene map is
        # the QGraphicsView's own transform (independent of the ViewBox data
        # range that the pan shifts), so each move's emitted scene point is a
        # stable function of its viewport pixel.
        p1 = QtCore.QPointF(p0.x() + 9.0, p0.y() + 5.0)
        p2 = QtCore.QPointF(p0.x() + 23.0, p0.y() + 17.0)
        assert send(QtCore.QEvent.Type.MouseMove, p1, button=none_button, buttons=left) is True
        assert send(QtCore.QEvent.Type.MouseMove, p2, button=none_button, buttons=left) is True

        # The pan is still active (guards against the test silently passing on a
        # non-pan code path) ...
        assert driver.is_active() is True
        # ... and the hover was re-emitted for BOTH moves, the last one at the
        # MOVED (p2) position, not frozen at pan-start (p0).
        assert len(emitted) == 2
        expected_last = view.graphicsView.mapToScene(int(p2.x()), int(p2.y()))
        assert emitted[-1] == pytest.approx((expected_last.x(), expected_last.y()))
        start_scene = view.graphicsView.mapToScene(int(p0.x()), int(p0.y()))
        assert emitted[-1] != pytest.approx((start_scene.x(), start_scene.y()))
    finally:
        view.close()


def _send_wheel(qt_app, view, angle_y: float) -> bool:
    """Deliver a classic mouse-wheel notch burst the way the platform does.

    Through ``sendEvent``, not the shell's filter directly: on this backend the
    filter declines a plain wheel and pyqtgraph's ViewBox does the zoom, so a
    direct ``eventFilter`` call would move no camera at all.
    """

    from pyqtgraph.Qt import QtCore, QtGui

    local = _viewport_center(view)
    event = QtGui.QWheelEvent(
        local,
        view.mapToGlobal(local.toPoint()),
        QtCore.QPoint(0, 0),  # no pixelDelta -> a classic wheel, not a touchpad
        QtCore.QPoint(0, int(angle_y)),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    return bool(qt_app.sendEvent(view.graphicsView.viewport(), event))


def _connect_montage_viewport_bridge(view):
    """Wire the production bridge to the real range signal over a montage owner.

    Returns the call record: how many range signals arrived, how many took the
    synchronous whole-plan retarget, how many took the coalesced interactive
    path, and how each was classified.
    """

    from types import SimpleNamespace

    from arrayscope.window.viewport_bridge import ViewportBridge

    calls = {"emitted": 0, "retarget": 0, "scheduled": 0, "reasons": []}
    owner = SimpleNamespace(
        img_view=view,
        _release_viewport_continuity=lambda: None,
        _note_viewport_interaction=calls["reasons"].append,
        _committed_display_frame=SimpleNamespace(
            value_source=SimpleNamespace(payloads={}),
        ),
        _frame_session=SimpleNamespace(display_committed=True),
        _schedule_interactive_montage_viewport_update=lambda: calls.__setitem__(
            "scheduled", calls["scheduled"] + 1
        ),
        retarget_montage_viewport=lambda: calls.__setitem__("retarget", calls["retarget"] + 1),
        view_state=SimpleNamespace(montage_axis=2),
    )
    owner.win = owner
    bridge = ViewportBridge(owner)

    def on_range_changed(*_args):
        calls["emitted"] += 1
        bridge.on_view_range_changed()

    view.getView().sigRangeChanged.connect(on_range_changed)
    calls["disconnect"] = lambda: view.getView().sigRangeChanged.disconnect(on_range_changed)
    return calls


# A zoom-out far past the constraint's maximum span, so the viewport clamp has
# something to correct within the one wheel event.
_UNCONSTRAINABLE_ZOOM_OUT_ANGLE = -3600


def test_wheel_range_changes_never_retarget_synchronously(qt_app):
    """Every range change a wheel causes is classified as the gesture it is.

    A wheel produces more than one sigRangeChanged: the camera move, plus the
    viewport constraint correcting it from inside that first emission. The
    one-shot wheel flag was consumed by the first, and
    QApplication.mouseButtons() is empty for wheel input, so the rest read as
    programmatic replay and took the synchronous whole-plan retarget — 92 ms
    median, up to 542 ms, inside wheel dispatch.
    """

    view = _make_seeded_view(qt_app)
    calls = _connect_montage_viewport_bridge(view)
    try:
        assert _send_wheel(qt_app, view, _UNCONSTRAINABLE_ZOOM_OUT_ANGLE) is True

        # More than one range signal really arrived (otherwise this test proves
        # nothing about the nested ones) ...
        assert calls["emitted"] >= 2
        # ... and none of them entered the synchronous path.
        assert calls["retarget"] == 0
        assert calls["scheduled"] == calls["emitted"]
        assert set(calls["reasons"]) == {"range-pointer"}
    finally:
        calls["disconnect"]()
        view.close()


def test_wheel_zoom_driver_clamps_before_it_sets_the_range():
    """One gesture step is one setRange: the clamp folds into the driver.

    The shell's constraint enforcement corrects an out-of-bounds range from
    inside the resulting sigRangeChanged, which is a second range signal for
    one gesture step. Pre-applying the same clamp leaves it nothing to correct.
    """

    from types import SimpleNamespace

    from arrayscope.display.view_navigation_driver import QtViewNavigationDriver

    clamped = ((-10.0, 10.0), (-10.0, 10.0))
    applied = []
    owner = SimpleNamespace(
        image=object(),
        viewport_controller=SimpleNamespace(is_fit_locked=lambda: False),
        view=SimpleNamespace(
            viewRange=lambda: ((0.0, 64.0), (0.0, 64.0)),
            setRange=lambda **kwargs: applied.append(kwargs),
        ),
        constrained_view_range=lambda _view_range: clamped,
        window=lambda: None,
        _event_display_point=lambda _event: (32.0, 32.0),
    )
    event = SimpleNamespace(
        angleDelta=lambda: SimpleNamespace(y=lambda: -3600, isNull=lambda: False)
    )

    assert QtViewNavigationDriver(owner)._apply_wheel_zoom(event) is True

    assert applied == [{"xRange": clamped[0], "yRange": clamped[1], "padding": 0}]


def test_programmatic_range_change_still_retargets_synchronously(qt_app):
    """No gesture scope, no buttons: the synchronous obligation is unchanged.

    Deferring the genuinely programmatic branch is refuted — it costs a whole
    montage republication per return-leg pan
    (docs/redesign/programmatic-viewport-replay-2026-07-31.md).
    """

    from pyqtgraph.Qt import QtCore, QtTest

    view = _make_seeded_view(qt_app)
    calls = _connect_montage_viewport_bridge(view)
    try:
        assert view.in_camera_gesture is False
        # The bridge reads QApplication.mouseButtons(), which is process-global
        # state an earlier test's unreleased press leaves stuck — observed as
        # this test passing alone and misclassifying in a full run. Releasing
        # here makes the test's own precondition ("no buttons") true by
        # construction instead of by suite ordering.
        stuck = qt_app.mouseButtons()
        if stuck != QtCore.Qt.MouseButton.NoButton:
            for button in QtCore.Qt.MouseButton:
                if button != QtCore.Qt.MouseButton.NoButton and stuck & button:
                    QtTest.QTest.mouseRelease(view.graphicsView.viewport(), button)
        assert qt_app.mouseButtons() == QtCore.Qt.MouseButton.NoButton
        current = view.getView().viewRange()
        view.getView().setRange(
            xRange=(current[0][0] - 3.0, current[0][1] - 3.0),
            yRange=current[1],
            padding=0,
        )

        assert calls["emitted"] >= 1
        assert calls["retarget"] >= 1
        assert calls["scheduled"] == 0
        assert set(calls["reasons"]) == {"range-programmatic"}
    finally:
        calls["disconnect"]()
        view.close()


def test_camera_gesture_scope_is_balanced_when_a_handler_raises(qt_app):
    """A raising handler must not leave every later range change a "gesture"."""

    from pyqtgraph.Qt import QtCore, QtGui

    def boom(_event):
        raise RuntimeError("handler failure")

    view = _make_seeded_view(qt_app)
    try:
        local = _viewport_center(view)
        event = QtGui.QNativeGestureEvent(
            QtCore.Qt.NativeGestureType.ZoomNativeGesture,
            QtGui.QPointingDevice.primaryPointingDevice(),
            2,
            local,
            local,
            view.mapToGlobal(local.toPoint()),
            0.25,
            QtCore.QPointF(0.0, 0.0),
        )
        view._handle_context_menu_event = boom
        try:
            with pytest.raises(RuntimeError):
                view.eventFilter(view.graphicsView.viewport(), event)
        finally:
            # Qt keeps sending this widget events while it closes.
            del view._handle_context_menu_event

        assert view.in_camera_gesture is False
    finally:
        view.close()
