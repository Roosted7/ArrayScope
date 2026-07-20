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
