from types import SimpleNamespace

import pytest

from arrayscope.display.viewport import (
    MIN_VIEWPORT_CONTENT_FRACTION,
    ViewportController,
    ViewportIntent,
    ViewportMode,
    ViewportPolicy,
    coerce_viewport_policy,
    constrain_view_range,
)


class FakeViewBox:
    def __init__(self):
        self.range = [[0.0, 1.0], [0.0, 1.0]]
        self.fit_count = 0

    def autoRange(self, padding=0):
        self.fit_count += 1
        self.range = [[0.0, 10.0], [0.0, 8.0]]

    def viewRange(self):
        return [list(self.range[0]), list(self.range[1])]

    def setRange(self, *, xRange, yRange, padding=0):
        self.range = [list(xRange), list(yRange)]


def _size(width, height):
    return SimpleNamespace(width=lambda: width, height=lambda: height)


def test_viewport_controller_fits_first_image():
    controller = ViewportController()
    view = FakeViewBox()

    controller.apply_after_image(view, (8, 10), _size(100, 80), policy=ViewportPolicy.PRESERVE)

    assert controller.mode == ViewportMode.AUTO_UNTOUCHED
    assert view.fit_count == 0
    assert view.viewRange() == [[0.0, 10.0], [0.0, 8.0]]


def test_viewport_controller_one_to_one_uses_viewport_pixels():
    controller = ViewportController()
    view = FakeViewBox()
    view.setRange(xRange=(4, 6), yRange=(3, 5), padding=0)

    controller.one_to_one(view, (20, 30), _size(12, 8))

    assert controller.mode == ViewportMode.USER
    assert view.viewRange()[0] == pytest.approx([-1, 11])
    assert view.viewRange()[1] == pytest.approx([0, 8])


def test_viewport_controller_preserve_ignores_origin_only_changes():
    controller = ViewportController()
    view = FakeViewBox()
    controller.apply_after_image(view, (8, 10), _size(100, 80), policy=ViewportPolicy.PRESERVE, display_rect=(0, 0, 9, 7))
    view.setRange(xRange=(20, 30), yRange=(40, 50), padding=0)
    controller.note_user_range_changed()

    controller.apply_after_image(view, (8, 10), _size(100, 80), policy=ViewportPolicy.PRESERVE, display_rect=(0, 100, 9, 107))

    assert view.fit_count == 0
    assert view.viewRange() == [[20, 30], [40, 50]]


def test_viewport_controller_fit_uses_exact_display_rect():
    controller = ViewportController()
    view = FakeViewBox()
    controller.apply_after_image(
        view,
        (8, 10),
        _size(100, 80),
        policy=ViewportPolicy.FIT_ONCE,
        display_rect=(5.0, 7.0, 14.0, 20.0),
    )

    assert controller.mode == ViewportMode.FIT
    assert view.viewRange() == [[5.0, 14.0], [7.0, 20.0]]
    assert view.fit_count == 0


def test_viewport_controller_one_to_one_does_not_reapply_after_image_change():
    controller = ViewportController()
    view = FakeViewBox()
    controller.apply_after_image(view, (20, 30), _size(12, 8), policy=ViewportPolicy.PRESERVE)
    view.setRange(xRange=(4, 6), yRange=(3, 5), padding=0)

    controller.one_to_one(view, (20, 30), _size(12, 8))
    view.setRange(xRange=(100, 120), yRange=(200, 220), padding=0)
    controller.apply_after_image(view, (20, 30), _size(12, 8), policy=ViewportPolicy.PRESERVE)

    assert controller.mode == ViewportMode.USER
    assert view.viewRange() == [[100, 120], [200, 220]]


def test_fit_lock_survives_shape_reset_and_tracks_new_full_bounds():
    controller = ViewportController()
    view = FakeViewBox()
    controller.apply_after_image(view, (8, 10), _size(100, 80), policy=ViewportPolicy.PRESERVE)
    controller.set_fit_locked(view, True)

    controller.apply_after_image(
        view,
        (12, 20),
        _size(100, 80),
        policy=ViewportPolicy.RESET_FOR_NEW_SHAPE,
        display_rect=(5.0, 7.0, 25.0, 19.0),
    )

    assert controller.mode == ViewportMode.FIT
    assert controller.is_fit_locked()
    assert view.viewRange() == [[5.0, 25.0], [7.0, 19.0]]


def test_fit_lock_tracks_origin_only_display_rect_changes():
    controller = ViewportController()
    view = FakeViewBox()
    controller.apply_after_image(
        view,
        (8, 10),
        _size(100, 80),
        policy=ViewportPolicy.FIT_ONCE,
        display_rect=(0.0, 0.0, 10.0, 8.0),
    )

    controller.apply_after_image(
        view,
        (8, 10),
        _size(100, 80),
        policy=ViewportPolicy.PRESERVE,
        display_rect=(0.0, 100.0, 10.0, 108.0),
    )

    assert controller.mode == ViewportMode.FIT
    assert view.viewRange() == [[0.0, 10.0], [100.0, 108.0]]


def test_legacy_auto_range_is_coerced_identically_for_every_backend():
    assert coerce_viewport_policy(ViewportPolicy.PRESERVE, True) is ViewportPolicy.FIT_ONCE
    assert coerce_viewport_policy(ViewportPolicy.FIT_ONCE, False) is ViewportPolicy.PRESERVE


def test_semantic_viewport_intent_survives_canonical_coercion():
    assert coerce_viewport_policy(ViewportIntent.FIT) is ViewportIntent.FIT
    assert coerce_viewport_policy("reset_for_new_shape") is ViewportPolicy.RESET_FOR_NEW_SHAPE
    assert coerce_viewport_policy("one_to_one") is ViewportIntent.ONE_TO_ONE


def test_constrain_view_range_caps_zoom_out_to_minimum_content_fraction():
    constrained = constrain_view_range(
        ((-1070.0, 1330.0), (-850.0, 950.0)),
        (0.0, 0.0, 100.0, 40.0),
    )

    assert constrained[0][1] - constrained[0][0] == pytest.approx(100.0 / MIN_VIEWPORT_CONTENT_FRACTION)
    assert constrained[1][1] - constrained[1][0] == pytest.approx(40.0 / MIN_VIEWPORT_CONTENT_FRACTION)
    assert (constrained[0][0] + constrained[0][1]) * 0.5 == pytest.approx(130.0)
    assert (constrained[1][0] + constrained[1][1]) * 0.5 == pytest.approx(50.0)


def test_constrain_view_range_uses_previous_center_when_zoom_hits_limit():
    constrained = constrain_view_range(
        ((-1070.0, 1330.0), (-850.0, 950.0)),
        (0.0, 0.0, 100.0, 40.0),
        previous_view_range=((10.0, 110.0), (-20.0, 60.0)),
    )

    assert constrained[0] == pytest.approx((-940.0, 1060.0))
    assert constrained[1] == pytest.approx((-380.0, 420.0))


def test_constrain_view_range_keeps_minimum_overlap_after_pan():
    constrained = constrain_view_range(
        ((200.0, 300.0), (10.0, 20.0)),
        (0.0, 0.0, 100.0, 40.0),
    )

    assert constrained[0] == pytest.approx((95.0, 195.0))
    assert constrained[1] == pytest.approx((10.0, 20.0))


def test_constrain_view_range_clamps_axes_independently():
    constrained = constrain_view_range(
        ((95.0, 195.0), (100.0, 110.0)),
        (0.0, 0.0, 100.0, 40.0),
    )

    assert constrained[0] == pytest.approx((95.0, 195.0))
    assert constrained[1] == pytest.approx((39.5, 49.5))


def test_constrain_view_range_allows_zoomed_in_edges():
    constrained = constrain_view_range(
        ((150.0, 152.0), (0.0, 2.0)),
        (0.0, 0.0, 100.0, 40.0),
    )

    assert constrained[0] == pytest.approx((99.9, 101.9))
    assert constrained[1] == pytest.approx((0.0, 2.0))
