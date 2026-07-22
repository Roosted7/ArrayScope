import os

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from arrayscope.display.view_navigation import begin_pan, pan_view_range, wheel_zoom_view_range


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
        view.getView().scene().sigMouseMoved.connect(
            lambda pt: emitted.append((pt.x(), pt.y()))
        )

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
        assert (
            send(QtCore.QEvent.Type.MouseButtonPress, p0, button=left, buttons=left)
            is True
        )
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
