import pytest

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
