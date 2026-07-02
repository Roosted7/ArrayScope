import importlib.util
import sys
from pathlib import Path

import pytest
import arrayscope.core.axis_utils as axis_utils


AXIS_UTILS_PATH = Path(__file__).parents[2] / "arrayscope" / "core" / "axis_utils.py"


def test_validate_axis_accepts_shape_or_ndim():
    assert axis_utils.validate_axis((3, 4), 1) == 1
    assert axis_utils.validate_axis(2, "0") == 0


def test_validate_axis_rejects_out_of_bounds_with_label():
    with pytest.raises(ValueError, match="line axis 2 is out of bounds for 2D data"):
        axis_utils.validate_axis((3, 4), 2, label="line axis")


def test_validate_distinct_axes_checks_count_and_duplicates():
    assert axis_utils.validate_distinct_axes((3, 4, 5), (2, 0), count=2) == (2, 0)

    with pytest.raises(ValueError, match="exactly 2"):
        axis_utils.validate_distinct_axes((3, 4), (0,), count=2)

    with pytest.raises(ValueError, match="distinct"):
        axis_utils.validate_distinct_axes((3, 4), (1, 1), count=2)


def test_clamp_index_and_non_singleton_axes():
    assert axis_utils.clamp_index((3, 4), 0, -5) == 0
    assert axis_utils.clamp_index((3, 4), 1, 99) == 3
    assert axis_utils.non_singleton_axes((1, 3, 1, 4)) == (1, 3)
