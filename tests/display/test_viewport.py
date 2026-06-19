from types import SimpleNamespace

import pytest

from arrayscope.display.viewport import ViewportController, ViewportMode, ViewportPolicy


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
    assert view.viewRange() == [[0.0, 9.0], [0.0, 7.0]]


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
