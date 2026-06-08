import numpy as np

from arrayscope.display.levels import finite_bounds


def test_finite_bounds_ignores_nan_without_copying_contract():
    data = np.array([[np.nan, 1.0], [2.0, np.inf]])

    assert finite_bounds(data) is None


def test_finite_bounds_returns_exact_small_bounds():
    data = np.array([[np.nan, 1.0], [2.0, 3.0]])

    assert finite_bounds(data) == (1.0, 3.0)


def test_finite_bounds_samples_large_arrays():
    data = np.arange(10_000, dtype=float).reshape(100, 100)

    bounds = finite_bounds(data, exact_limit=100, max_samples=100)

    assert bounds is not None
    assert bounds[0] == 0.0
    assert bounds[1] <= float(data.max())
