import numpy as np

from arrayscope.display.levels import finite_bounds


def test_finite_bounds_masks_inf_to_finite_structure():
    # ±Inf must not blow the window to the float range or void the bounds;
    # the non-finite entries are masked (a copy is made only in this rare
    # path, and only of the already-sampled array).
    data = np.array([[np.nan, 1.0], [2.0, np.inf]])

    assert finite_bounds(data) == (1.0, 2.0)


def test_finite_bounds_all_nonfinite_returns_none():
    data = np.array([[np.nan, np.inf], [-np.inf, np.nan]])

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
