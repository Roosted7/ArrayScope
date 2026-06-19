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


def test_montage_plan_empty_view_range_does_not_fallback_to_tile_zero():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 3, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    plan = montage.make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(2, 3), columns=3, gap=1)

    visible = plan.tiles_intersecting(((10_000.0, 10_100.0), (10_000.0, 10_100.0)), margin_tiles=0)

    assert visible == ()


def test_montage_plan_preserves_source_indices():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 3, 10)).with_montage_axis(2, indices=(2, 4, 8), text="2:2:10")
    plan = montage.make_montage_plan(state, axis=2, indices=(2, 4, 8), tile_shape=(2, 3), columns=2)

    assert tuple(tile.source_index for tile in plan.tiles) == (2, 4, 8)
    assert plan.geometry.indices == (2, 4, 8)
    assert tuple(tile.view_state.slice_indices[2] for tile in plan.tiles) == (2, 4, 8)


def test_montage_plan_tile_at_returns_source_tile_and_ignores_gap():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 3, 9)).with_montage_axis(2, indices=tuple(range(9)), text=":")
    plan = montage.make_montage_plan(state, axis=2, indices=tuple(range(9)), tile_shape=(2, 3), columns=3, gap=1)

    tile = plan.tile_at(4, 4)

    assert tile is not None
    assert tile.source_index == 4
    assert plan.tile_at(3, 0) is None


def test_montage_viewport_canvas_preserves_global_tile_positions():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 3, 20)).with_montage_axis(2, indices=tuple(range(20)), text=":")
    plan = montage.make_montage_plan(state, axis=2, indices=tuple(range(20)), tile_shape=(2, 3), columns=5, gap=1)
    rendered = tuple(
        montage.RenderedTile(
            tile=plan.tiles[index],
            image=np.full((2, 3), plan.tiles[index].source_index, dtype=np.float32),
            histogram_data=np.full((2, 3), plan.tiles[index].source_index, dtype=np.float32),
            eval_ms=0.0,
            slab_shape=(2, 3),
            slab_nbytes=24,
        )
        for index in (10, 11, 12)
    )

    canvas = montage.make_montage_viewport_canvas(
        plan,
        rendered,
        view_range=((0, 12), (6, 8)),
        budget_bytes=1024 * 1024,
    )

    assert canvas.origin_y != 0
    for rendered_tile in rendered:
        x0 = rendered_tile.tile.x0 - canvas.origin_x
        y0 = rendered_tile.tile.y0 - canvas.origin_y
        np.testing.assert_array_equal(canvas.data[y0 : y0 + 2, x0 : x0 + 3], np.full((2, 3), rendered_tile.tile.source_index))


def test_montage_viewport_canvas_shape_does_not_shrink_to_loaded_tiles():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 3, 9)).with_montage_axis(2, indices=tuple(range(9)), text=":")
    plan = montage.make_montage_plan(state, axis=2, indices=tuple(range(9)), tile_shape=(2, 3), columns=3, gap=1)
    rendered = (
        montage.RenderedTile(
            tile=plan.tiles[4],
            image=np.full((2, 3), 4, dtype=np.float32),
            histogram_data=np.full((2, 3), 4, dtype=np.float32),
            eval_ms=0.0,
            slab_shape=(2, 3),
            slab_nbytes=24,
        ),
    )

    canvas = montage.make_montage_viewport_canvas(
        plan,
        rendered,
        view_range=((0, 11), (0, 8)),
        viewport_shape=(8, 11),
        budget_bytes=1024 * 1024,
    )

    assert canvas.data.shape == (8, 11)
    assert canvas.canvas_rect == (0, 0, 11, 8)


def test_montage_viewport_canvas_origin_stays_stable_while_tiles_load():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 3, 20)).with_montage_axis(2, indices=tuple(range(20)), text=":")
    plan = montage.make_montage_plan(state, axis=2, indices=tuple(range(20)), tile_shape=(2, 3), columns=5, gap=1)

    def rendered(index):
        return montage.RenderedTile(
            tile=plan.tiles[index],
            image=np.full((2, 3), index, dtype=np.float32),
            histogram_data=np.full((2, 3), index, dtype=np.float32),
            eval_ms=0.0,
            slab_shape=(2, 3),
            slab_nbytes=24,
        )

    first = montage.make_montage_viewport_canvas(plan, (rendered(10),), view_range=((0, 12), (6, 8)), budget_bytes=1024 * 1024)
    second = montage.make_montage_viewport_canvas(plan, (rendered(12),), view_range=((0, 12), (6, 8)), budget_bytes=1024 * 1024)

    assert (first.origin_x, first.origin_y, first.data.shape) == (second.origin_x, second.origin_y, second.data.shape)


def test_montage_viewport_canvas_expands_to_complete_intersecting_tiles():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 3, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    plan = montage.make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(2, 3), columns=3, gap=1)
    rendered = (
        montage.RenderedTile(plan.tiles[1], np.ones((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32), 0.0, (2, 3), 24),
    )

    canvas = montage.make_montage_viewport_canvas(plan, rendered, view_range=((5, 6), (0, 1)), viewport_shape=(2, 3), budget_bytes=1024 * 1024)

    assert canvas.canvas_rect[0] <= plan.tiles[1].x0
    assert canvas.canvas_rect[2] >= plan.tiles[1].x0 + plan.tiles[1].width
    assert canvas.canvas_rect[1] <= plan.tiles[1].y0
    assert canvas.canvas_rect[3] >= plan.tiles[1].y0 + plan.tiles[1].height


def test_montage_viewport_canvas_marks_loaded_loading_skipped_unloaded_tiles():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 2, 4)).with_montage_axis(2, indices=(0, 1, 2, 3), text=":")
    plan = montage.make_montage_plan(state, axis=2, indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=4, gap=1)
    rendered = (
        montage.RenderedTile(plan.tiles[0], np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32), 0.0, (2, 2), 16),
    )

    canvas = montage.make_montage_viewport_canvas(
        plan,
        rendered,
        view_range=((0, 11), (0, 2)),
        budget_bytes=1024 * 1024,
        loading_tiles=(plan.tiles[1],),
        skipped_tiles=(plan.tiles[2],),
    )

    assert canvas.tile_states == (
        montage.MontageTileState.LOADED,
        montage.MontageTileState.LOADING,
        montage.MontageTileState.SKIPPED,
        montage.MontageTileState.UNLOADED,
    )


def test_montage_viewport_canvas_histogram_keeps_unloaded_loading_skipped_as_nan():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 2, 4)).with_montage_axis(2, indices=(0, 1, 2, 3), text=":")
    plan = montage.make_montage_plan(state, axis=2, indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=4, gap=1)
    rendered = (
        montage.RenderedTile(plan.tiles[0], np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32), 0.0, (2, 2), 16),
    )

    canvas = montage.make_montage_viewport_canvas(
        plan,
        rendered,
        view_range=((0, 11), (0, 2)),
        budget_bytes=1024 * 1024,
        loading_tiles=(plan.tiles[1],),
        skipped_tiles=(plan.tiles[2],),
    )

    assert np.isfinite(canvas.histogram_data[:, 0:2]).all()
    assert np.isnan(canvas.histogram_data[:, 3:]).all()


def test_montage_tile_status_at_global_point_distinguishes_tile_and_gap():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 2, 3)).with_montage_axis(2, indices=(0, 1, 2), text=":")
    plan = montage.make_montage_plan(state, axis=2, indices=(0, 1, 2), tile_shape=(2, 2), columns=3, gap=1)
    states = (montage.MontageTileState.LOADED, montage.MontageTileState.LOADING, montage.MontageTileState.SKIPPED)

    assert montage.tile_status_at_global_point(plan, states, 1, 1).state == montage.MontageTileState.LOADED
    assert montage.tile_status_at_global_point(plan, states, 4, 1).state == montage.MontageTileState.LOADING
    assert montage.tile_status_at_global_point(plan, states, 7, 1).state == montage.MontageTileState.SKIPPED
    assert montage.tile_status_at_global_point(plan, states, 2, 1) is None


def test_montage_viewport_canvas_histogram_marks_gaps_and_unloaded_as_nan():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 2, 3)).with_montage_axis(2, indices=(0, 1, 2), text=":")
    plan = montage.make_montage_plan(state, axis=2, indices=(0, 1, 2), tile_shape=(2, 2), columns=3, gap=1)
    rendered = tuple(
        montage.RenderedTile(
            tile=plan.tiles[index],
            image=np.full((2, 2), index, dtype=np.float32),
            histogram_data=np.full((2, 2), index, dtype=np.float32),
            eval_ms=0.0,
            slab_shape=(2, 2),
            slab_nbytes=16,
        )
        for index in (0, 2)
    )

    canvas = montage.make_montage_viewport_canvas(plan, rendered, view_range=((0, 8), (0, 2)), budget_bytes=1024 * 1024)

    assert np.isnan(canvas.histogram_data[:, 2]).all()
    assert np.isnan(canvas.histogram_data[:, 3:5]).all()
    assert np.isfinite(canvas.histogram_data[:, :2]).all()
    assert np.isfinite(canvas.histogram_data[:, 6:8]).all()


def test_montage_viewport_canvas_handles_scalar_tile_after_rgb_tile():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 2, 2)).with_montage_axis(2, indices=(0, 1), text=":")
    plan = montage.make_montage_plan(state, axis=2, indices=(0, 1), tile_shape=(2, 2), columns=2, gap=1)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 1] = 7
    scalar = np.full((2, 2), 5, dtype=np.uint8)
    rendered = (
        montage.RenderedTile(plan.tiles[0], rgb, np.ones((2, 2), dtype=np.float32), 0.0, rgb.shape, rgb.nbytes),
        montage.RenderedTile(plan.tiles[1], scalar, scalar.astype(np.float32), 0.0, scalar.shape, scalar.nbytes),
    )

    canvas = montage.make_montage_viewport_canvas(plan, rendered, view_range=None, budget_bytes=1024 * 1024)

    assert canvas.data.shape == (2, 5, 3)
    np.testing.assert_array_equal(canvas.data[:, 3:5], np.full((2, 2, 3), 5, dtype=np.uint8))
    assert np.isfinite(canvas.histogram_data[:, 3:5]).all()


def test_montage_viewport_canvas_handles_rgb_tile_after_scalar_tile():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 2, 2)).with_montage_axis(2, indices=(0, 1), text=":")
    plan = montage.make_montage_plan(state, axis=2, indices=(0, 1), tile_shape=(2, 2), columns=2, gap=1)
    scalar = np.full((2, 2), 5, dtype=np.uint8)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 2] = 9
    rendered = (
        montage.RenderedTile(plan.tiles[0], scalar, scalar.astype(np.float32), 0.0, scalar.shape, scalar.nbytes),
        montage.RenderedTile(plan.tiles[1], rgb, np.ones((2, 2), dtype=np.float32), 0.0, rgb.shape, rgb.nbytes),
    )

    canvas = montage.make_montage_viewport_canvas(plan, rendered, view_range=None, budget_bytes=1024 * 1024)

    assert canvas.data.shape == (2, 5, 3)
    np.testing.assert_array_equal(canvas.data[:, :2], np.full((2, 2, 3), 5, dtype=np.uint8))
    np.testing.assert_array_equal(canvas.data[:, 3:5], rgb)


def test_montage_viewport_canvas_rejects_over_budget_before_allocation():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((10_000, 10_000, 1)).with_montage_axis(2, indices=(0,), text=":")
    plan = montage.make_montage_plan(state, axis=2, indices=(0,), tile_shape=(10_000, 10_000), columns=1, gap=1)
    rendered = (
        montage.RenderedTile(
            tile=plan.tiles[0],
            image=np.zeros((1, 1), dtype=np.float32),
            histogram_data=np.zeros((1, 1), dtype=np.float32),
            eval_ms=0.0,
            slab_shape=(1, 1),
            slab_nbytes=4,
        ),
    )

    with np.testing.assert_raises(MemoryError):
        montage.make_montage_viewport_canvas(plan, rendered, view_range=None, budget_bytes=1024)
