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


def test_montage_tile_at_global_point_distinguishes_tile_and_gap():
    from arrayscope.core.view_state import ViewState

    state = ViewState.from_shape((2, 2, 3)).with_montage_axis(2, indices=(0, 1, 2), text=":")
    plan = montage.make_montage_plan(
        state, axis=2, indices=(0, 1, 2), tile_shape=(2, 2), columns=3, gap=1
    )

    assert plan.tile_at(1, 1).montage_index == 0
    assert plan.tile_at(4, 1).montage_index == 1
    assert plan.tile_at(7, 1).montage_index == 2
    # The one-pixel gap between tiles belongs to no tile.
    assert plan.tile_at(2, 1) is None


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
    return plan, view_range


def _ring(plan, view_range, margin):
    return {
        int(tile.montage_index)
        for tile in montage.viewport_tiles(plan, view_range=view_range, margin_tiles=margin)
    }


def test_a_camera_edge_that_only_touches_a_tile_still_requires_it():
    """The closed boundary, at margin 0 where nothing else can supply the tile.

    This used to be asserted through ``FramePlanner`` on a 2x3 montage, where
    the tile in question is inside every ring the plan carries, so nothing but
    a coincidence was keeping it honest. Here it is asserted at margin 0, where
    there is no ring to supply the answer. The camera below ends exactly on the
    first pixel row of the second montage row -- one pixel of it is on screen,
    so it is a render obligation, not speculation.
    """

    from arrayscope.core.view_state import ViewState

    state = (
        ViewState.from_shape((10, 10, 6))
        .with_image_axes(0, 1)
        .with_montage_axis(2, columns=3, indices=tuple(range(6)))
    )
    plan = montage.make_montage_plan(
        state, axis=2, indices=tuple(range(6)), tile_shape=(10, 10), columns=3
    )
    stride = plan.tile_shape[0] + plan.gap

    touching = _ring(plan, ((0.0, 10.0), (0.0, float(stride))), 0)
    short_of_it = _ring(plan, ((0.0, 10.0), (0.0, stride - 1.0)), 0)

    assert touching == {0, 3}
    assert short_of_it == {0}


def test_snapping_a_rect_to_tile_bounds_cannot_change_which_tiles_it_touches():
    """The rect convention, stated as the one property it has to have.

    ``montage_rect_for_viewport`` snaps outward to whole-tile bounds, and for a
    long time that snap solved a strict overlap test while ``tiles_intersecting``
    solved an inclusive one. Composing them was therefore not a fixed point,
    and the error term was a whole ring: a rect edge landing on a multiple of
    the stride lost the tile it was touching. That is what made "raw rect" and
    "snapped rect" two different answers to one question, which is the only
    reason ``FrameRegion.active`` was entitled to a second convention.

    Both now read one span computation, so the property below holds by
    construction rather than by two formulas agreeing. The sweep steps by one
    pixel through several strides, which is the resolution the defect lived at:
    it was invisible at any coarser step.
    """

    plan, _view_range = _field_montage_plan()
    stride = plan.tile_shape[1] + plan.gap
    checked = boundary_cases = 0

    for x0 in range(4 * stride):
        for width in (1, stride - 1, stride, stride + 1, 2 * stride):
            rect = (x0, x0 // 2, x0 + width, x0 // 2 + width)
            raw = _tiles_for_rect(plan, rect)
            snapped = _tiles_for_rect(plan, montage._expand_rect_to_tile_bounds(plan, rect))
            assert raw == snapped, f"snap changed the tile set for {rect}"
            checked += 1
            if rect[2] % stride == 0:
                boundary_cases += 1

    # Regime: a sweep that never lands an edge on a boundary tests nothing.
    assert boundary_cases > 0
    assert checked > 100


def _tiles_for_rect(plan, rect):
    return {
        int(tile.montage_index)
        for tile in plan.tiles_intersecting(
            ((rect[0], rect[2]), (rect[1], rect[3])), margin_tiles=0
        )
    }


def test_viewport_rings_are_strictly_nested_and_a_genuine_subset():
    plan, view_range = _field_montage_plan()

    visible = _ring(plan, view_range, montage.VISIBLE_MARGIN_TILES)
    coverage = _ring(plan, view_range, montage.COVERAGE_MARGIN_TILES)
    retention = _ring(plan, view_range, montage.RETENTION_HALO_TILES)
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

    plan, view_range = _field_montage_plan()
    (x0, x1), y = view_range
    stride = plan.tile_shape[1] + plan.gap
    panned = ((x0 + stride, x1 + stride), y)

    before_coverage = _ring(plan, view_range, montage.COVERAGE_MARGIN_TILES)
    before_retention = _ring(plan, view_range, montage.RETENTION_HALO_TILES)
    after_visible = _ring(plan, panned, montage.VISIBLE_MARGIN_TILES)
    after_coverage = _ring(plan, panned, montage.COVERAGE_MARGIN_TILES)

    assert after_visible, "the pan must land on tiles, or this proves nothing"
    # One ring covers what the pan draws...
    assert after_visible <= before_coverage
    # ...but leaves the next step's coverage already outside what was protected.
    assert not (after_coverage <= before_coverage)
    # Two rings still lead the pan.
    assert after_coverage <= before_retention
