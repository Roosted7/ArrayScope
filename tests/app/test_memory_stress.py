import numpy as np

from arrayscope.core.memory_budget import MONTAGE_BUDGET_BYTES, estimate_montage_bytes
from arrayscope.display.levels import finite_bounds


def test_memory_estimate_blocks_large_montage_without_allocation():
    nbytes = estimate_montage_bytes((8192, 8192), 128, np.float32, histogram=True, columns=16)

    assert nbytes > MONTAGE_BUDGET_BYTES


def test_large_level_bounds_uses_sampling():
    data = np.arange(250_000, dtype=float).reshape(500, 500)

    bounds = finite_bounds(data, exact_limit=1_000, max_samples=10_000)

    assert bounds is not None
    assert bounds[0] == 0.0
    assert bounds[1] <= float(data.max())
