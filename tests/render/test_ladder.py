"""Unified LOD ladder semantics.

These pin the redesign's quality-progression contract: coarse before fine,
no recomputation of anything already committable, native-only collapses the
ladder, and cross-tile floor-first fill.
"""

from __future__ import annotations

import pytest

from arrayscope.display.lod import LodDemand
from arrayscope.kernel.task import Lane, Priority
from arrayscope.render.ladder import LadderPolicy, LodLadder, Rung, TileLodState
from arrayscope.render.lod import round_preview_level


def demand(level: int, acceptable=None) -> LodDemand:
    return LodDemand(
        desired_level=level,
        desired_factor=2**level,
        desired_factor_xy=(2**level, 2**level),
        acceptable_levels=tuple(acceptable)
        if acceptable is not None
        else (max(0, level - 1), level, level + 1),
        source_texels_per_pixel_xy=(float(2**level), float(2**level)),
        reason="test demand",
    )


def demand_at_scale(level: int, source_texels_per_pixel: float) -> LodDemand:
    factor = 2**level
    return LodDemand(
        desired_level=level,
        desired_factor=factor,
        desired_factor_xy=(factor, factor),
        acceptable_levels=(max(0, level - 1), level, level + 1),
        source_texels_per_pixel_xy=(
            float(source_texels_per_pixel),
            float(source_texels_per_pixel),
        ),
        reason="continuous-scale test demand",
    )


def rungs(steps):
    return [(step.rung, step.level) for step in steps]


def plan_tile(
    ladder,
    state,
    current_demand,
    verdict=None,
    *,
    preview_level=4,
    target_level=None,
):
    """Spell the required round floors at every direct ladder unit call."""

    return ladder.plan_tile(
        state,
        current_demand,
        verdict,
        preview_level=preview_level,
        target_level=(
            int(current_demand.desired_level) if target_level is None else int(target_level)
        ),
    )


def test_cold_tile_climbs_coarse_then_desired():
    ladder = LodLadder(LadderPolicy())
    steps = plan_tile(ladder, TileLodState(tile_number=3), demand(1))
    assert rungs(steps) == [(Rung.FLOOR, 4), (Rung.DESIRED, 1)]
    assert steps[0].priority == Priority.INTERACTIVE
    assert steps[0].lane == Lane.DISPLAY_PREVIEW
    assert steps[1].lane == Lane.DISPLAY_PREPARATION


def test_preview_disabled_goes_directly_to_desired_target():
    ladder = LodLadder(LadderPolicy())

    steps = plan_tile(
        ladder,
        TileLodState(tile_number=3, allow_preview=False),
        demand(1),
    )

    assert rungs(steps) == [(Rung.DESIRED, 1)]
    assert steps[0].lane == Lane.DISPLAY_PREVIEW


def test_explicit_target_only_arm_omits_floor_without_changing_target_evaluation():
    ladder = LodLadder(
        LadderPolicy(
            reduced_input_available=True,
            coarse_rung_enabled=False,
        )
    )

    steps = plan_tile(ladder, TileLodState(tile_number=3), demand(1))

    assert rungs(steps) == [(Rung.DESIRED, 1)]
    assert steps[0].lane == Lane.DISPLAY_PREVIEW
    assert steps[0].reduce_from_native is False


def test_coarse_demand_still_gets_round_preview_floor_before_target():
    ladder = LodLadder(LadderPolicy())
    steps = plan_tile(ladder, TileLodState(tile_number=0), demand(3), preview_level=5)
    assert rungs(steps) == [(Rung.FLOOR, 5), (Rung.DESIRED, 3)]
    assert steps[0].lane == Lane.DISPLAY_PREVIEW


def test_preview_stays_two_levels_coarser_as_target_moves_coarse():
    for target, preview in ((0, 2), (2, 4), (3, 5), (6, 8)):
        level = round_preview_level(demand=demand(target), retention_level=0)
        assert level == preview
        assert level >= target + 2


def test_round_planner_preview_tracks_smooth_screen_scale_within_three_to_six_pixels():
    observed = []
    for source_texels_per_pixel in (4.0, 5.2, 5.4, 7.58):
        current_demand = demand_at_scale(2, source_texels_per_pixel)
        level = round_preview_level(demand=current_demand, retention_level=0)
        screen_pixels_per_preview_texel = (2**level) / source_texels_per_pixel
        assert 3.0 <= screen_pixels_per_preview_texel <= 6.0
        assert level >= current_demand.desired_level + 2
        observed.append(level)

    assert observed == [4, 4, 5, 5]


def test_round_planner_retention_hint_cannot_exceed_screen_density_cap():
    current_demand = demand_at_scale(2, 7.58)

    level = round_preview_level(demand=current_demand, retention_level=9)

    assert level == 5
    assert (2**level) / 7.58 <= 6.0


def test_target_finer_than_retention_floor_gets_one_coarse_rung():
    ladder = LodLadder(LadderPolicy())
    steps = plan_tile(ladder, TileLodState(tile_number=0), demand(2))
    assert rungs(steps) == [(Rung.FLOOR, 4), (Rung.DESIRED, 2)]
    assert steps[0].lane == Lane.DISPLAY_PREVIEW


def test_desired_refinement_after_presented_preview_stays_preparation():
    ladder = LodLadder(LadderPolicy())
    steps = plan_tile(
        ladder,
        TileLodState(
            tile_number=0,
            presented_level=3,
            resident_levels=(3,),
            presented_quality="preview",
        ),
        demand(3),
    )

    assert rungs(steps) == [(Rung.DESIRED, 3)]
    assert steps[0].lane == Lane.DISPLAY_PREPARATION


def test_converged_tile_plans_nothing():
    ladder = LodLadder()
    state = TileLodState(tile_number=0, presented_level=1, resident_levels=(1, 2))
    assert plan_tile(ladder, state, demand(1)) == ()


def test_ready_unacknowledged_preview_is_not_recomputed_during_commit_gap():
    ladder = LodLadder(LadderPolicy())
    floor_ready = TileLodState(tile_number=0, ready_level=4, ready_quality="fallback")
    assert rungs(plan_tile(ladder, floor_ready, demand(1))) == [(Rung.DESIRED, 1)]


def test_ready_unacknowledged_target_is_converged_for_admission():
    ladder = LodLadder()
    state = TileLodState(tile_number=0, ready_level=1, ready_quality="exact")

    assert plan_tile(ladder, state, demand(1)) == ()


def test_zoom_in_refines_progressively():
    ladder = LodLadder(LadderPolicy())
    state = TileLodState(tile_number=0, presented_level=4, resident_levels=(4,))
    steps = plan_tile(ladder, state, demand(1, acceptable=(0, 1, 2)))
    # The retained coarse rung never reruns; desired completes the refinement.
    assert rungs(steps) == [(Rung.DESIRED, 1)]
    assert steps[0].priority == Priority.VISIBLE_IMAGE  # visibly wrong level


def test_unpresented_native_source_still_plans_demanded_display_level():
    ladder = LodLadder()
    # Resident source data alone is not proof that the backend is presenting
    # anything current; a black slot still needs a display payload.
    state = TileLodState(tile_number=0, resident_levels=(0,))
    steps = plan_tile(ladder, state, demand(2))
    assert rungs(steps) == [(Rung.DESIRED, 2)]


def test_zoom_out_keeps_presented_finer_level():
    ladder = LodLadder(LadderPolicy())
    state = TileLodState(tile_number=0, presented_level=0, resident_levels=(0,))
    steps = plan_tile(ladder, state, demand(3, acceptable=(2, 3, 4)))
    assert steps == ()


def test_native_demand_ends_exact_without_duplicate_step():
    ladder = LodLadder(LadderPolicy())
    steps = plan_tile(
        ladder,
        TileLodState(tile_number=0),
        demand(0, acceptable=(0, 1)),
        preview_level=2,
    )
    assert rungs(steps) == [(Rung.FLOOR, 2), (Rung.DESIRED, 0)]
    assert all(step.rung != Rung.EXACT for step in steps)  # DESIRED==native


def test_exact_requested_appends_native_rung():
    ladder = LodLadder()
    state = TileLodState(
        tile_number=0, presented_level=2, resident_levels=(2,), exact_requested=True
    )
    steps = plan_tile(ladder, state, demand(2))
    assert rungs(steps) == [(Rung.EXACT, 0)]
    assert steps[0].lane == Lane.VISIBLE_MATERIALIZATION


def test_native_only_policy_collapses_ladder():
    ladder = LodLadder(LadderPolicy(mode="native-only"))
    steps = plan_tile(ladder, TileLodState(tile_number=0), demand(3))
    assert rungs(steps) == [(Rung.EXACT, 0)]
    assert plan_tile(ladder, TileLodState(tile_number=0, presented_level=0), demand(3)) == ()


def test_without_reduced_input_pre_native_rungs_reduce_from_native():
    ladder = LodLadder(LadderPolicy(reduced_input_available=False))
    steps = plan_tile(ladder, TileLodState(tile_number=0), demand(1))
    assert all(step.reduce_from_native for step in steps)


def test_cross_tile_floor_first_fill_ordering():
    ladder = LodLadder(LadderPolicy())
    states = (TileLodState(tile_number=0), TileLodState(tile_number=1))
    steps = ladder.plan(states, demand(1), preview_level=4, target_level=1)
    assert [(step.rung, step.tile_number) for step in steps] == [
        (Rung.FLOOR, 0),
        (Rung.FLOOR, 1),
        (Rung.DESIRED, 0),
        (Rung.DESIRED, 1),
    ]


def test_round_floors_are_required_at_the_ladder_boundary():
    ladder = LodLadder()

    with pytest.raises(TypeError):
        ladder.plan((TileLodState(tile_number=0),), demand(1))


def test_round_preview_floor_is_passed_once_across_heterogeneous_tile_state():
    """Progressive contract R2b: retention changes skips, never the round floor."""

    ladder = LodLadder(LadderPolicy())
    states = (
        TileLodState(tile_number=0),
        TileLodState(tile_number=1, resident_levels=(0,), presented_level=0),
        TileLodState(tile_number=2, floor_available=True),
    )

    steps = ladder.plan(states, demand(2), preview_level=5, target_level=2)

    assert {step.level for step in steps if step.rung is Rung.FLOOR} == {5}


def test_foreign_retained_levels_are_reused_without_a_third_production_rung():
    """Progressive contract R1/R2: bound production, not visible reuse."""

    ladder = LodLadder(LadderPolicy())
    states = (
        TileLodState(tile_number=0, resident_levels=(0,), presented_level=0),
        TileLodState(tile_number=1, resident_levels=(6,), presented_level=6),
        TileLodState(tile_number=2),
    )

    steps = ladder.plan(states, demand(2), preview_level=5, target_level=2)

    assert not any(step.tile_number == 0 for step in steps)
    assert {(step.tile_number, step.rung, step.level) for step in steps} == {
        (1, Rung.DESIRED, 2),
        (2, Rung.FLOOR, 5),
        (2, Rung.DESIRED, 2),
    }
    assert {step.level for step in steps} <= {2, 5}


def test_unpresented_finer_residency_plans_only_a_presentation_step():
    """R2 skips production, but physical first pixels still need an owner."""

    ladder = LodLadder(LadderPolicy())
    state = TileLodState(
        tile_number=0,
        resident_levels=(0,),
        presented_level=None,
        floor_available=True,
        allow_preview=True,
    )

    steps = ladder.plan((state,), demand(4), preview_level=5, target_level=4)

    assert len(steps) == 2
    assert steps[0].rung == Rung.FLOOR
    assert steps[0].level == 5
    assert steps[0].presentation_only is True
    assert steps[1].rung == Rung.DESIRED
    assert steps[1].level == 4
    assert ladder.coarse_rung_refusal(state, demand(4)) == ""


def test_ladder_carries_canonical_tile_rank_across_every_rung():
    ladder = LodLadder(LadderPolicy())
    steps = plan_tile(
        ladder,
        TileLodState(tile_number=7, scheduling_rank=3),
        demand(1),
    )

    assert {step.scheduling_rank for step in steps} == {3}


# ---- coarse-rung refusal reporting -------------------------------------------
#
# "The ladder planned no coarse rung" is an absence, and an absence names no
# cause. The 2026-07-26 preview-LOD work attributed a missing FFT preview to
# `allow_preview` by reading the source, and the attribution was wrong: the
# real gate fires two clauses earlier. These pin the reported reason to what
# `plan_tile` actually does, so the two cannot drift.

_REFUSAL_CASES = (
    ("native_only", LadderPolicy(mode="native-only"), TileLodState(tile_number=0), 2),
    (
        "preview_not_allowed",
        LadderPolicy(),
        TileLodState(tile_number=0, allow_preview=False),
        2,
    ),
    (
        "already_covered",
        LadderPolicy(),
        TileLodState(tile_number=0, resident_levels=(4,), presented_level=4),
        2,
    ),
    (
        "cold_tile_gets_one",
        LadderPolicy(),
        TileLodState(tile_number=0),
        2,
    ),
    (
        "retained_floor_without_reduced_input",
        LadderPolicy(reduced_input_available=False),
        TileLodState(tile_number=0, floor_available=True),
        2,
    ),
)


def test_coarse_rung_refusal_agrees_with_what_plan_tile_does():
    """A reported reason must be empty exactly when a coarse rung is planned."""

    for name, policy, state, level in _REFUSAL_CASES:
        ladder = LodLadder(policy)
        steps = plan_tile(ladder, state, demand(level))
        planned = any(step.rung == Rung.FLOOR for step in steps)
        reason = ladder.coarse_rung_refusal(state, demand(level))
        assert planned == (reason == ""), (
            f"{name}: planned={planned} but reason={reason!r} "
            f"(rungs={[step.rung for step in steps]})"
        )


def test_coarse_rung_refusal_names_the_gate_that_actually_fired():
    from arrayscope.render import ladder as ladder_module

    def reason_for(name):
        policy, state, level = next(
            (policy, state, level) for case, policy, state, level in _REFUSAL_CASES if case == name
        )
        return LodLadder(policy).coarse_rung_refusal(state, demand(level))

    assert reason_for("native_only") == ladder_module.COARSE_RUNG_NATIVE_ONLY
    assert reason_for("preview_not_allowed") == ladder_module.COARSE_RUNG_PREVIEW_NOT_ALLOWED
    assert reason_for("already_covered") == ladder_module.COARSE_RUNG_ALREADY_COVERED
    assert reason_for("cold_tile_gets_one") == ladder_module.COARSE_RUNG_PLANNED
    # A retained floor still earns a coarse rung with no reduced input at all.
    assert reason_for("retained_floor_without_reduced_input") == ladder_module.COARSE_RUNG_PLANNED


def test_non_reducible_pipeline_keeps_its_native_output_preview_pass():
    """Progressive contract R4: reduced input is not preview admission.

    A genuinely non-reducible pipeline evaluates natively for FLOOR and
    reduces only the output.  That native result is finer than both round
    floors, so R2 requires the later target pass to reuse it.
    """

    ladder = LodLadder(LadderPolicy(reduced_input_available=False))
    state = TileLodState(tile_number=0, allow_preview=True, floor_available=False)

    steps = plan_tile(ladder, state, demand(2))
    assert [step.rung for step in steps] == [Rung.FLOOR, Rung.DESIRED]
    assert all(step.reduce_from_native for step in steps)
    assert ladder.coarse_rung_refusal(state, demand(2)) == ""


def test_raw_montage_has_one_coarse_rung_then_refines():
    """ADR 0059 removes the FLOOR/PREVIEW self-shadowing state."""

    from arrayscope.render import ladder as ladder_module

    ladder = LodLadder(LadderPolicy())
    cold = TileLodState(tile_number=0)
    assert [step.rung for step in plan_tile(ladder, cold, demand(2))] == [
        Rung.FLOOR,
        Rung.DESIRED,
    ]

    floored = TileLodState(tile_number=0, resident_levels=(4,), presented_level=4)
    assert [step.rung for step in plan_tile(ladder, floored, demand(2))] == [Rung.DESIRED]
    assert (
        ladder.coarse_rung_refusal(floored, demand(2)) == ladder_module.COARSE_RUNG_ALREADY_COVERED
    )


def test_fallback_at_the_desired_level_still_plans_a_desired_step():
    """2026-07-16 churn starvation, member 5 of the deferred-stage/commit
    lost-wakeup family (docs/redesign/stale-empty-tiles-2026-07-16.md).

    A tile presenting a retained FALLBACK floor AT the (re-coarsened) desired
    level has correct-looking pixels and an open exact target. Filtering it out
    of refinement parked 38 such tiles with the kernel idle, immune to
    retargets. ADR 0059 retired the shared exact pass that used to carry this
    invariant, so it is pinned here on the owner that carries it now: a
    non-exact payload never satisfies the demand, whatever its level.
    """

    ladder = LodLadder(LadderPolicy())
    for quality in ("preview", "fallback"):
        state = TileLodState(
            tile_number=0,
            resident_levels=(2,),
            presented_level=2,
            presented_quality=quality,
            current_presentation_quality=quality,
        )
        rungs = [step.rung for step in plan_tile(ladder, state, demand(2))]
        assert Rung.DESIRED in rungs, f"{quality} at the desired level must still refine"

    # The exact payload at the same level is what settles it — otherwise this
    # test would pass just as well against a ladder that never converges.
    settled = TileLodState(
        tile_number=0,
        resident_levels=(2,),
        presented_level=2,
        presented_quality="exact",
        current_presentation_quality="exact",
    )
    assert [step.rung for step in plan_tile(ladder, settled, demand(2))] == []
