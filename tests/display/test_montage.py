import arrayscope.display.montage as montage


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

    plan = montage.make_montage_plan(
        state, axis=2, indices=(0, 1, 2, 3, 4), tile_shape=(2, 3), columns=2, gap=1
    )

    assert plan.grid_shape == (3, 2)
    assert plan.display_shape == (8, 7)


def test_montage_plan_visible_tiles_intersect_view_range():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 3, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    plan = montage.make_montage_plan(
        state, axis=2, indices=tuple(range(6)), tile_shape=(2, 3), columns=3, gap=1
    )

    visible = plan.tiles_intersecting(((4.0, 8.0), (0.0, 2.0)), margin_tiles=0)

    # Tile 2 begins exactly at x=8. Pixel-center geometry makes that boundary
    # sample visible, so it belongs to the same required set as admission and
    # completion (V1); excluding it recreated the persistent-black-tile bug.
    assert tuple(tile.source_index for tile in visible) == (1, 2)


def test_montage_plan_empty_view_range_does_not_fallback_to_tile_zero():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 3, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    plan = montage.make_montage_plan(
        state, axis=2, indices=tuple(range(6)), tile_shape=(2, 3), columns=3, gap=1
    )

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
    plan = montage.make_montage_plan(
        state, axis=2, indices=tuple(range(9)), tile_shape=(2, 3), columns=3, gap=1
    )

    tile = plan.tile_at(4, 4)

    assert tile is not None
    assert tile.source_index == 4
    assert plan.tile_at(3, 0) is None


def test_montage_tile_status_at_global_point_distinguishes_tile_and_gap():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 2, 3)).with_montage_axis(2, indices=(0, 1, 2), text=":")
    plan = montage.make_montage_plan(
        state, axis=2, indices=(0, 1, 2), tile_shape=(2, 2), columns=3, gap=1
    )
    states = (
        montage.MontageTileState.LOADED,
        montage.MontageTileState.LOADING,
        montage.MontageTileState.SKIPPED,
    )

    assert (
        montage.tile_status_at_global_point(plan, states, 1, 1).state
        == montage.MontageTileState.LOADED
    )
    assert (
        montage.tile_status_at_global_point(plan, states, 4, 1).state
        == montage.MontageTileState.LOADING
    )
    assert (
        montage.tile_status_at_global_point(plan, states, 7, 1).state
        == montage.MontageTileState.SKIPPED
    )
    assert montage.tile_status_at_global_point(plan, states, 2, 1) is None


def _field_montage_plan():
    """The 272-tile field montage, at a camera covering only a few tiles.

    The size matters. On a montage small enough to fit the viewport, every
    ring covers every tile and a halo assertion passes no matter what the
    radii are -- ``margin=0`` and ``margin=3`` both answer "all of them". The
    tests below therefore assert their own regime before asserting anything
    about the halo.
    """

    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((64, 64, 272)).with_montage_axis(
        2, columns=17, indices=tuple(range(272)), text="0:272"
    )
    plan = montage.make_montage_plan(
        state, axis=2, indices=tuple(range(272)), tile_shape=(64, 64), columns=17
    )
    # A window onto roughly two tiles near the middle of the grid.
    tile = plan.tiles[136]
    cx = float(tile.x0) + float(tile.width) * 0.5
    cy = float(tile.y0) + float(tile.height) * 0.5
    view_range = ((cx - 64.0, cx + 64.0), (cy - 64.0, cy + 64.0))
    return plan, view_range, (800, 800)


def _ring(plan, view_range, viewport_shape, margin):
    return {
        int(tile.montage_index)
        for tile in montage.viewport_tiles(
            plan, view_range=view_range, viewport_shape=viewport_shape, margin_tiles=margin
        )
    }


def test_viewport_rings_are_strictly_nested_and_a_genuine_subset():
    plan, view_range, viewport_shape = _field_montage_plan()

    visible = _ring(plan, view_range, viewport_shape, montage.VISIBLE_MARGIN_TILES)
    coverage = _ring(plan, view_range, viewport_shape, montage.COVERAGE_MARGIN_TILES)
    retention = _ring(plan, view_range, viewport_shape, montage.RETENTION_HALO_TILES)
    everything = {int(tile.montage_index) for tile in plan.tiles}

    # Regime first: if the retention halo covered the montage this test could
    # not fail, and neither could any other halo test written at this size.
    assert 0 < len(visible) < len(coverage) < len(retention) < len(everything)
    assert visible < coverage < retention < everything


def test_retention_halo_survives_the_pan_step_that_coverage_only_absorbs():
    """Why retention is two rings and coverage is one.

    A pan of one tile step turns the old ring 1 into the new drawn set. A
    one-ring halo is therefore spent exactly by the move it exists to absorb:
    after it, the newly adjacent tiles are as cold as if there had been no
    halo. At two rings the old ring 2 is still ahead of the new viewport.
    """

    plan, view_range, viewport_shape = _field_montage_plan()
    (x0, x1), y = view_range
    stride = plan.tile_shape[1] + plan.gap
    panned = ((x0 + stride, x1 + stride), y)

    before_coverage = _ring(plan, view_range, viewport_shape, montage.COVERAGE_MARGIN_TILES)
    before_retention = _ring(plan, view_range, viewport_shape, montage.RETENTION_HALO_TILES)
    after_visible = _ring(plan, panned, viewport_shape, montage.VISIBLE_MARGIN_TILES)
    after_coverage = _ring(plan, panned, viewport_shape, montage.COVERAGE_MARGIN_TILES)

    assert after_visible, "the pan must land on tiles, or this proves nothing"
    # One ring covers what the pan draws...
    assert after_visible <= before_coverage
    # ...but leaves the next step's coverage already outside what was protected.
    assert not (after_coverage <= before_coverage)
    # Two rings still lead the pan.
    assert after_coverage <= before_retention
