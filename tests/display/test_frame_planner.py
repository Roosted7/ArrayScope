from __future__ import annotations

from arrayscope.core.frame_targets import FrameTarget
from arrayscope.core.view_state import ViewState
from arrayscope.display.backend_contract import PYQTGRAPH_CAPABILITIES
from arrayscope.display.frame_planner import MONTAGE_REGION_CACHE_SIZE, FramePlanner
from arrayscope.display.scene import DisplayLayout

# The field montage: 272 tiles, the size the wgpu stall dossiers were written
# against. A four-tile montage cannot tell a reusing planner from a rebuilding
# one, so every reuse assertion below runs at this size.
FIELD_TILES = 272
FIELD_COLUMNS = 17
FIELD_TILE = 16
FIELD_STRIDE = FIELD_TILE + 1  # montage gap is 1


def _target(semantic="sem", viewport=None):
    return FrameTarget(
        semantic_key=semantic,
        viewport_key=viewport,
        presentation_key=("levels", 0.0, 1.0),
        quality="exact-visible",
    )


def _field_state(*, tiles=FIELD_TILES, columns=FIELD_COLUMNS, crop=None):
    state = ViewState.from_shape((FIELD_TILE, FIELD_TILE, tiles)).with_image_axes(0, 1)
    if crop is not None:
        state = state.with_axis_range(0, crop)
    return state.with_montage_axis(2, columns=columns, indices=tuple(range(tiles)))


def _field_plan(planner, state, view_range, *, viewport=None):
    return planner.plan(
        target=_target("field", viewport=viewport if viewport is not None else view_range),
        view_state=state,
        display_shape=(FIELD_COLUMNS * FIELD_STRIDE, FIELD_COLUMNS * FIELD_STRIDE),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
        viewport_shape=(256, 256),
        view_range=view_range,
    )


def _pan_sequence(steps=24, span_tiles=4):
    """A camera walking the montage a half-tile at a time."""

    span = span_tiles * FIELD_STRIDE
    ranges = []
    for step in range(steps):
        x0 = step * FIELD_STRIDE * 0.5
        y0 = (step % 7) * FIELD_STRIDE * 0.5
        ranges.append(((x0, x0 + span), (y0, y0 + span)))
    return ranges


def test_small_frame_plans_one_tile_region():
    state = ViewState.from_shape((32, 48)).with_image_axes(0, 1)

    plan = FramePlanner(internal_tile_shape=(64, 64)).plan(
        target=_target(),
        view_state=state,
        display_shape=(32, 48),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
    )

    assert plan.layout is DisplayLayout.SINGLE
    assert plan.tile_shape == (32, 48)
    assert plan.active_region_ids == (0,)
    assert plan.regions[0].bounds == (0.0, 0.0, 47.0, 31.0)
    assert plan.regions[0].data_slices == (slice(0, 32), slice(0, 48))


def test_huge_single_plane_plans_internal_tiles():
    state = ViewState.from_shape((40, 60)).with_image_axes(0, 1)

    plan = FramePlanner(internal_tile_shape=(16, 16)).plan(
        target=_target(),
        view_state=state,
        display_shape=(40, 60),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
    )

    assert plan.layout is DisplayLayout.SINGLE
    assert plan.tile_shape == (16, 16)
    assert len(plan.regions) == 12
    assert plan.regions[0].bounds == (0.0, 0.0, 15.0, 15.0)
    assert plan.regions[-1].bounds == (48.0, 32.0, 59.0, 39.0)
    assert plan.active_region_ids == tuple(range(12))


def test_one_tile_montage_matches_single_region_geometry():
    normal = ViewState.from_shape((12, 14, 1)).with_image_axes(0, 1)
    montage = normal.with_montage_axis(2, columns=1, indices=(0,))
    planner = FramePlanner(internal_tile_shape=(16, 16))

    frame_plan = planner.plan(
        target=_target("same-semantic"),
        view_state=normal,
        display_shape=(12, 14),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
    )
    montage_plan = planner.plan(
        target=_target("same-semantic"),
        view_state=montage,
        display_shape=(12, 14),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
    )

    assert frame_plan.semantic_key == montage_plan.semantic_key
    assert frame_plan.active_region_ids == montage_plan.active_region_ids == (0,)
    assert frame_plan.regions[0].bounds == montage_plan.regions[0].bounds
    assert frame_plan.tile_shape == montage_plan.tile_shape == (12, 14)


def test_multi_tile_montage_marks_active_and_near_regions_from_viewport():
    state = (
        ViewState.from_shape((10, 10, 6))
        .with_image_axes(0, 1)
        .with_montage_axis(2, columns=3, indices=tuple(range(6)))
    )

    plan = FramePlanner().plan(
        target=_target(),
        view_state=state,
        display_shape=(21, 32),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
        viewport_shape=(10, 10),
        view_range=((11.0, 20.0), (0.0, 9.0)),
    )

    assert plan.layout is DisplayLayout.MONTAGE
    assert plan.planned_region_ids == tuple(range(6))
    assert plan.active_region_ids == (1,)
    assert set(plan.near_region_ids).issuperset({0, 1, 2, 4})
    assert plan.regions[1].source_index == 1


def test_montage_requires_tiles_that_touch_the_viewport_boundary():
    state = (
        ViewState.from_shape((10, 10, 6))
        .with_image_axes(0, 1)
        .with_montage_axis(2, columns=3, indices=tuple(range(6)))
    )

    plan = FramePlanner().plan(
        target=_target(),
        view_state=state,
        display_shape=(21, 32),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
        viewport_shape=(10, 10),
        view_range=((0.0, 10.0), (0.0, 11.0)),
    )

    assert plan.active_region_ids == (0, 3)


def test_camera_only_retarget_changes_active_regions_not_materialization_key():
    state = ViewState.from_shape((40, 60)).with_image_axes(0, 1)
    planner = FramePlanner(internal_tile_shape=(16, 16))

    left = planner.plan(
        target=_target("semantic", viewport="left"),
        view_state=state,
        display_shape=(40, 60),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
        view_range=((0.0, 15.0), (0.0, 15.0)),
    )
    right = planner.plan(
        target=_target("semantic", viewport="right"),
        view_state=state,
        display_shape=(40, 60),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
        view_range=((32.0, 59.0), (16.0, 39.0)),
    )

    assert left.semantic_key == right.semantic_key == "semantic"
    assert left.materialization_key == right.materialization_key
    assert left.active_region_ids != right.active_region_ids


def test_field_montage_pan_reuses_the_region_skeleton_it_did_not_change():
    """The hit rate itself is the assertion.

    A pass/fail on plan *values* cannot distinguish reuse from a rebuild —
    both produce the same plan. What has to be true is that the second and
    every later pan finds the skeleton already built, so the counter is what
    gets asserted, at the montage size where rebuilding is actually expensive.
    """

    state = _field_state()
    planner = FramePlanner()
    ranges = _pan_sequence()

    for view_range in ranges:
        _field_plan(planner, state, view_range)

    assert planner.montage_region_cache_misses == 1
    assert planner.montage_region_cache_hits == len(ranges) - 1


def test_sub_tile_nudge_reuses_the_whole_region_tuple():
    """A camera move too small to change the snapped tile sets costs nothing.

    ``tiles_intersecting`` snaps outward to whole tiles, so a half-pixel drag
    step resolves to the identical active/near sets. That case must not walk
    the montage at all: the previous regions tuple is the answer.
    """

    state = _field_state()
    planner = FramePlanner()
    span = 4 * FIELD_STRIDE

    first = _field_plan(planner, state, ((0.0, span), (0.0, span)), viewport="a")
    nudged = _field_plan(planner, state, ((0.5, span + 0.5), (0.5, span + 0.5)), viewport="b")

    assert first.active_region_ids == nudged.active_region_ids
    assert first.regions is nudged.regions
    assert first.scene_region_signature is nudged.scene_region_signature
    assert planner.montage_region_set_hits == 1
    assert planner.montage_region_set_misses == 1


def test_pan_rebuilds_only_the_regions_whose_flags_changed():
    """The tiles the camera did not touch are the *same objects* afterwards.

    This is the assertion a rebuilding planner cannot pass: it allocates a
    fresh FrameRegion for all 272 tiles, so nothing is identical across pans
    even though everything is equal.
    """

    state = _field_state()
    planner = FramePlanner()
    span = 4 * FIELD_STRIDE

    first = _field_plan(planner, state, ((0.0, span), (0.0, span)), viewport="a")
    panned = _field_plan(
        planner,
        state,
        ((3 * FIELD_STRIDE, 3 * FIELD_STRIDE + span), (0.0, span)),
        viewport="b",
    )

    assert first.active_region_ids != panned.active_region_ids
    reused = sum(
        1 for before, after in zip(first.regions, panned.regions, strict=True) if before is after
    )
    changed = len(first.regions) - reused
    # Only the tiles that entered or left the active/near bands are new.
    assert 0 < changed < len(first.regions) // 4
    assert reused > len(first.regions) * 0.75


def test_reused_montage_plans_equal_the_unmemoized_planner_across_every_input():
    """Value equality against a planner that provably cannot reuse anything.

    ``montage_region_cache_size=0`` is the same code path with the skeleton
    lookup switched off, so any divergence the cache introduces — a stale
    region, a wrong derived tuple, a key that conflates two montages — shows
    up here as an inequality rather than as a plausible-looking plan.
    """

    memoized = FramePlanner()
    reference = FramePlanner(montage_region_cache_size=0)

    states = [
        ("camera", _field_state()),
        # A different montage window: same axis, fewer sources.
        ("window", _field_state(tiles=120)),
        # A different plan for the same window: relayout to another column count.
        ("plan", _field_state(tiles=120, columns=10)),
        # A different view state entirely: back to the full montage, cropped.
        ("view-state", _field_state(crop=(2, FIELD_TILE - 1))),
    ]

    compared = 0
    for label, state in states:
        for view_range in _pan_sequence(steps=6):
            got = _field_plan(memoized, state, view_range)
            want = _field_plan(reference, state, view_range)
            assert got.regions == want.regions, label
            assert got.active_region_ids == want.active_region_ids, label
            assert got.planned_region_ids == want.planned_region_ids, label
            assert got.near_region_ids == want.near_region_ids, label
            assert got.scene_region_signature == want.scene_region_signature, label
            assert got.materialization_key == want.materialization_key, label
            assert got == want, label
            compared += 1

    assert compared == len(states) * 6
    assert memoized.montage_region_cache_hits > 0
    assert reference.montage_region_cache_hits == 0


def test_montage_region_cache_is_bounded():
    planner = FramePlanner()

    for tiles in range(60, 60 + MONTAGE_REGION_CACHE_SIZE + 3):
        _field_plan(planner, _field_state(tiles=tiles), ((0.0, 68.0), (0.0, 68.0)))

    assert len(planner._montage_skeletons) == MONTAGE_REGION_CACHE_SIZE


def test_unhashable_semantic_key_still_plans_without_reuse():
    state = _field_state(tiles=6, columns=3)

    plan = FramePlanner().plan(
        target=_target(semantic=["unhashable"]),
        view_state=state,
        display_shape=(51, 51),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
        viewport_shape=(64, 64),
        view_range=((0.0, 17.0), (0.0, 17.0)),
    )

    assert plan.layout is DisplayLayout.MONTAGE
    assert plan.planned_region_ids == tuple(range(6))


def test_montage_near_regions_are_the_one_retention_halo():
    """The planner and the frame session must name the same set.

    Both fill the same two downstream fields -- ``near_tiles`` on the
    presentation delta and ``near_tile_numbers`` on the LOD admission scope --
    and which of them answered depended on the code path. They disagreed
    twice over: margin 1 against margin 2, and the raw camera rect against the
    rect snapped outward to whole-tile bounds. On the 272-tile field montage
    that was 15 tiles called near where 45 were being protected.
    """

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import (
        RETENTION_HALO_TILES,
        make_montage_plan,
        viewport_tiles,
    )

    state = ViewState.from_shape((64, 64, 272)).with_montage_axis(
        2, columns=17, indices=tuple(range(272)), text="0:272"
    )
    montage_plan = make_montage_plan(
        state, axis=2, indices=tuple(range(272)), tile_shape=(64, 64), columns=17
    )
    tile = montage_plan.tiles[136]
    cx = float(tile.x0) + float(tile.width) * 0.5
    cy = float(tile.y0) + float(tile.height) * 0.5
    view_range = ((cx - 64.0, cx + 64.0), (cy - 64.0, cy + 64.0))
    viewport_shape = (800, 800)

    frame_plan = FramePlanner(internal_tile_shape=(1024, 1024)).plan(
        target=_target("halo", view_range),
        view_state=state,
        display_shape=montage_plan.display_shape,
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
        viewport_shape=viewport_shape,
        view_range=view_range,
        montage_plan=montage_plan,
    )

    halo = {
        int(t.montage_index)
        for t in viewport_tiles(
            montage_plan,
            view_range=view_range,
            viewport_shape=viewport_shape,
            margin_tiles=RETENTION_HALO_TILES,
        )
    }
    # Regime: a halo covering the whole montage would make the equality below
    # true for any radius and any rect convention.
    assert 0 < len(halo) < len(montage_plan.tiles)
    assert set(frame_plan.near_region_ids) == halo
