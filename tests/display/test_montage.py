import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[2]
PACKAGE = types.ModuleType("arrayscope")
PACKAGE.__path__ = [str(ROOT / "arrayscope")]
sys.modules.setdefault("arrayscope", PACKAGE)

MONTAGE_PATH = ROOT / "arrayscope" / "display" / "montage.py"
SPEC = importlib.util.spec_from_file_location("arrayscope.display.montage", MONTAGE_PATH)
montage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = montage
SPEC.loader.exec_module(montage)


def test_make_montage_tiles_scalar_images_with_gap():
    images = [np.full((2, 3), value, dtype=float) for value in (1, 2, 3)]

    result = montage.make_montage(images, columns=2, gap=1).image

    assert result.data.shape == (5, 7)
    np.testing.assert_array_equal(result.data[0:2, 0:3], np.ones((2, 3)))
    np.testing.assert_array_equal(result.data[0:2, 4:7], np.full((2, 3), 2.0))
    np.testing.assert_array_equal(result.data[3:5, 0:3], np.full((2, 3), 3.0))


def test_make_montage_returns_geometry_matching_assembly():
    images = [np.full((2, 3), value, dtype=float) for value in (1, 2, 3)]

    rendered = montage.make_montage(images, columns=2, gap=1, indices=(4, 5, 7))

    assert rendered.image.data.shape == (5, 7)
    assert rendered.geometry.indices == (4, 5, 7)
    assert rendered.geometry.tile_shape == (2, 3)
    assert rendered.geometry.columns == 2
    assert rendered.geometry.rows == 2
    assert rendered.geometry.tile_height == 2
    assert rendered.geometry.tile_width == 3


def test_make_montage_preserves_histogram_data():
    images = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (10, 20)]
    hist = [np.full((2, 2), value, dtype=float) for value in (1, 2)]

    result = montage.make_montage(images, histogram_images=hist, columns=2, gap=0).image

    assert result.data.shape == (2, 4, 3)
    assert result.histogram_data.shape == (2, 4)
    np.testing.assert_array_equal(result.histogram_data[:, :2], np.ones((2, 2)))
    np.testing.assert_array_equal(result.histogram_data[:, 2:], np.full((2, 2), 2.0))


def test_make_montage_histogram_marks_gaps_as_nan_for_roi_stats():
    images = [np.full((2, 2), value, dtype=float) for value in (1, 2)]

    result = montage.make_montage(images, histogram_images=images, columns=2, gap=1).image

    assert np.isnan(result.histogram_data[:, 2]).all()
    assert np.isfinite(result.histogram_data[:, :2]).all()
    assert np.isfinite(result.histogram_data[:, 3:]).all()


def test_optimal_montage_columns_match_viewport_shape():
    wide = montage.optimal_montage_columns(8, (10, 10), (100, 240))
    tall = montage.optimal_montage_columns(8, (10, 10), (240, 100))

    assert wide == 5
    assert tall == 2
    assert wide > tall


def test_optimal_montage_columns_maximizes_fitted_viewport_area():
    columns = montage.optimal_montage_columns(8, (10, 10), (100, 240), gap=1)

    assert columns == 5


def test_montage_plan_display_shape_matches_grid():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 3, 5)).with_montage_axis(2, indices=(0, 1, 2, 3, 4), text=":")

    plan = montage.make_montage_plan(state, axis=2, indices=(0, 1, 2, 3, 4), tile_shape=(2, 3), columns=2, gap=1)

    assert plan.grid_shape == (3, 2)
    assert plan.display_shape == (8, 7)


def test_montage_plan_visible_tiles_intersect_view_range():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 3, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    plan = montage.make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(2, 3), columns=3, gap=1)

    visible = plan.tiles_intersecting(((4.0, 8.0), (0.0, 2.0)), margin_tiles=0)

    assert tuple(tile.source_index for tile in visible) == (1, 2)


def test_montage_plan_preserves_source_indices():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 3, 10)).with_montage_axis(2, indices=(2, 4, 8), text="2:2:10")
    plan = montage.make_montage_plan(state, axis=2, indices=(2, 4, 8), tile_shape=(2, 3), columns=2)

    assert tuple(tile.source_index for tile in plan.tiles) == (2, 4, 8)
    assert plan.geometry.indices == (2, 4, 8)
    assert tuple(tile.view_state.slice_indices[2] for tile in plan.tiles) == (2, 4, 8)
