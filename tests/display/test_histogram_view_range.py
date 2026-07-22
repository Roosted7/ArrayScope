import pytest

from arrayscope.display.histogram_view_range import HistogramViewRangePolicy


def test_non_negative_range_keeps_zero_floor_with_asymmetric_margin():
    policy = HistogramViewRangePolicy()

    assert policy.update((20.0, 100.0)) == pytest.approx((-1.0, 110.0))


def test_signed_range_uses_equal_margins():
    policy = HistogramViewRangePolicy()

    assert policy.update((-100.0, 50.0)) == pytest.approx((-115.0, 65.0))


def test_unchanged_bounds_and_manual_navigation_do_not_emit_ranges():
    policy = HistogramViewRangePolicy()
    policy.update((0.0, 100.0))

    assert policy.update((0.0, 100.0)) is None
    policy.note_manual_navigation()
    assert policy.update((0.0, 200.0)) is None
    assert policy.update((0.0, 200.0), force=True) == pytest.approx((-2.0, 220.0))


def test_hysteresis_updates_only_the_endpoint_that_crossed_its_gate():
    policy = HistogramViewRangePolicy()
    initial = policy.update((-100.0, 100.0))

    assert policy.update((-80.0, 80.0)) is None
    lower = policy.update((-70.0, 80.0))
    assert lower == pytest.approx((-85.0, initial[1]))

    upper = policy.update((-70.0, 118.0))
    assert upper == pytest.approx((lower[0], 136.8))
