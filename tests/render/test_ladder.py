"""Unified LOD ladder semantics.

These pin the redesign's quality-progression contract: coarse before fine,
no recomputation of anything already committable, native-only collapses the
ladder, and cross-tile floor-first fill.
"""

from __future__ import annotations

from arrayscope.display.lod import LodDemand
from arrayscope.kernel.task import Lane, Priority
from arrayscope.render.ladder import LadderPolicy, LodLadder, Rung, TileLodState


def demand(level: int, acceptable=None) -> LodDemand:
    return LodDemand(
        desired_level=level,
        desired_factor=2**level,
        desired_factor_xy=(2**level, 2**level),
        acceptable_levels=tuple(acceptable) if acceptable is not None else (max(0, level - 1), level, level + 1),
        source_texels_per_pixel_xy=(float(2**level), float(2**level)),
        reason="test demand",
    )


def rungs(steps):
    return [(step.rung, step.level) for step in steps]


def test_cold_tile_climbs_floor_preview_desired():
    ladder = LodLadder(LadderPolicy(floor_level=4, preview_level=2))
    steps = ladder.plan_tile(TileLodState(tile_number=3), demand(1))
    assert rungs(steps) == [(Rung.FLOOR, 4), (Rung.PREVIEW, 2), (Rung.DESIRED, 1)]
    assert steps[0].priority == Priority.INTERACTIVE
    assert steps[0].lane == Lane.DISPLAY_PREVIEW
    assert steps[2].lane == Lane.DISPLAY_PREPARATION


def test_coarse_demand_skips_redundant_preview_rung():
    ladder = LodLadder(LadderPolicy(floor_level=4, preview_level=2))
    steps = ladder.plan_tile(TileLodState(tile_number=0), demand(3))
    # preview level clamps to the demand (3) and DESIRED covers it.
    assert rungs(steps) == [(Rung.FLOOR, 4), (Rung.DESIRED, 3)]


def test_converged_tile_plans_nothing():
    ladder = LodLadder()
    state = TileLodState(tile_number=0, presented_level=1, resident_levels=(1, 2))
    assert ladder.plan_tile(state, demand(1)) == ()


def test_zoom_in_refines_progressively():
    ladder = LodLadder(LadderPolicy(preview_level=2))
    state = TileLodState(tile_number=0, presented_level=4, resident_levels=(4,))
    steps = ladder.plan_tile(state, demand(1, acceptable=(0, 1, 2)))
    # Presented 4 is outside the acceptable window: preview gives fast
    # improvement, desired completes it; the floor never reruns.
    assert rungs(steps) == [(Rung.PREVIEW, 2), (Rung.DESIRED, 1)]
    assert steps[1].priority == Priority.VISIBLE_IMAGE  # visibly wrong level


def test_unpresented_native_source_still_plans_demanded_display_level():
    ladder = LodLadder()
    # Resident source data alone is not proof that the backend is presenting
    # anything current; a black slot still needs a display payload.
    state = TileLodState(tile_number=0, resident_levels=(0,))
    steps = ladder.plan_tile(state, demand(2))
    assert rungs(steps) == [(Rung.DESIRED, 2)]


def test_zoom_out_keeps_presented_finer_level():
    ladder = LodLadder(LadderPolicy(preview_level=2))
    state = TileLodState(tile_number=0, presented_level=0, resident_levels=(0,))
    steps = ladder.plan_tile(state, demand(3, acceptable=(2, 3, 4)))
    assert steps == ()


def test_native_demand_ends_exact_without_duplicate_step():
    ladder = LodLadder(LadderPolicy(floor_level=4, preview_level=2))
    steps = ladder.plan_tile(TileLodState(tile_number=0), demand(0, acceptable=(0, 1)))
    assert rungs(steps) == [(Rung.FLOOR, 4), (Rung.PREVIEW, 2), (Rung.DESIRED, 0)]
    assert all(step.rung != Rung.EXACT for step in steps)  # DESIRED==native


def test_exact_requested_appends_native_rung():
    ladder = LodLadder()
    state = TileLodState(tile_number=0, presented_level=2, resident_levels=(2,), exact_requested=True)
    steps = ladder.plan_tile(state, demand(2))
    assert rungs(steps) == [(Rung.EXACT, 0)]
    assert steps[0].lane == Lane.VISIBLE_MATERIALIZATION


def test_native_only_policy_collapses_ladder():
    ladder = LodLadder(LadderPolicy(mode="native-only"))
    steps = ladder.plan_tile(TileLodState(tile_number=0), demand(3))
    assert rungs(steps) == [(Rung.EXACT, 0)]
    assert ladder.plan_tile(TileLodState(tile_number=0, presented_level=0), demand(3)) == ()


def test_without_reduced_input_pre_native_rungs_reduce_from_native():
    ladder = LodLadder(LadderPolicy(reduced_input_available=False))
    steps = ladder.plan_tile(TileLodState(tile_number=0), demand(1))
    assert all(step.reduce_from_native for step in steps)


def test_cross_tile_floor_first_fill_ordering():
    ladder = LodLadder(LadderPolicy(floor_level=4, preview_level=2))
    states = (TileLodState(tile_number=0), TileLodState(tile_number=1))
    steps = ladder.plan(states, demand(1))
    assert [(step.rung, step.tile_number) for step in steps] == [
        (Rung.FLOOR, 0),
        (Rung.FLOOR, 1),
        (Rung.PREVIEW, 0),
        (Rung.PREVIEW, 1),
        (Rung.DESIRED, 0),
        (Rung.DESIRED, 1),
    ]
