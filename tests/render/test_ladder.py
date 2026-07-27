"""Unified LOD ladder semantics.

These pin the redesign's quality-progression contract: coarse before fine,
no recomputation of anything already committable, native-only collapses the
ladder, and cross-tile floor-first fill.
"""

from __future__ import annotations

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


def test_cold_tile_climbs_coarse_then_desired():
    ladder = LodLadder(LadderPolicy(floor_level=4))
    steps = ladder.plan_tile(TileLodState(tile_number=3), demand(1))
    assert rungs(steps) == [(Rung.FLOOR, 4), (Rung.DESIRED, 1)]
    assert steps[0].priority == Priority.INTERACTIVE
    assert steps[0].lane == Lane.DISPLAY_PREVIEW
    assert steps[1].lane == Lane.DISPLAY_PREPARATION


def test_preview_disabled_goes_directly_to_desired_target():
    ladder = LodLadder(LadderPolicy(floor_level=4))

    steps = ladder.plan_tile(
        TileLodState(tile_number=3, allow_preview=False),
        demand(1),
    )

    assert rungs(steps) == [(Rung.DESIRED, 1)]
    assert steps[0].lane == Lane.DISPLAY_PREVIEW


def test_explicit_target_only_arm_omits_floor_without_changing_target_evaluation():
    ladder = LodLadder(
        LadderPolicy(
            floor_level=4,
            reduced_input_available=True,
            coarse_rung_enabled=False,
        )
    )

    steps = ladder.plan_tile(TileLodState(tile_number=3), demand(1))

    assert rungs(steps) == [(Rung.DESIRED, 1)]
    assert steps[0].lane == Lane.DISPLAY_PREVIEW
    assert steps[0].reduce_from_native is False


def test_coarse_demand_still_gets_round_preview_floor_before_target():
    ladder = LodLadder(LadderPolicy(floor_level=5))
    steps = ladder.plan_tile(TileLodState(tile_number=0), demand(3))
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
    ladder = LodLadder(LadderPolicy(floor_level=4))
    steps = ladder.plan_tile(TileLodState(tile_number=0), demand(2))
    assert rungs(steps) == [(Rung.FLOOR, 4), (Rung.DESIRED, 2)]
    assert steps[0].lane == Lane.DISPLAY_PREVIEW


def test_desired_refinement_after_presented_preview_stays_preparation():
    ladder = LodLadder(LadderPolicy(floor_level=4))
    steps = ladder.plan_tile(
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
    assert ladder.plan_tile(state, demand(1)) == ()


def test_ready_unacknowledged_preview_is_not_recomputed_during_commit_gap():
    ladder = LodLadder(LadderPolicy(floor_level=4))
    floor_ready = TileLodState(tile_number=0, ready_level=4, ready_quality="fallback")
    assert rungs(ladder.plan_tile(floor_ready, demand(1))) == [(Rung.DESIRED, 1)]


def test_ready_unacknowledged_target_is_converged_for_admission():
    ladder = LodLadder()
    state = TileLodState(tile_number=0, ready_level=1, ready_quality="exact")

    assert ladder.plan_tile(state, demand(1)) == ()


def test_zoom_in_refines_progressively():
    ladder = LodLadder(LadderPolicy(floor_level=4))
    state = TileLodState(tile_number=0, presented_level=4, resident_levels=(4,))
    steps = ladder.plan_tile(state, demand(1, acceptable=(0, 1, 2)))
    # The retained coarse rung never reruns; desired completes the refinement.
    assert rungs(steps) == [(Rung.DESIRED, 1)]
    assert steps[0].priority == Priority.VISIBLE_IMAGE  # visibly wrong level


def test_unpresented_native_source_still_plans_demanded_display_level():
    ladder = LodLadder()
    # Resident source data alone is not proof that the backend is presenting
    # anything current; a black slot still needs a display payload.
    state = TileLodState(tile_number=0, resident_levels=(0,))
    steps = ladder.plan_tile(state, demand(2))
    assert rungs(steps) == [(Rung.DESIRED, 2)]


def test_zoom_out_keeps_presented_finer_level():
    ladder = LodLadder(LadderPolicy(floor_level=4))
    state = TileLodState(tile_number=0, presented_level=0, resident_levels=(0,))
    steps = ladder.plan_tile(state, demand(3, acceptable=(2, 3, 4)))
    assert steps == ()


def test_native_demand_ends_exact_without_duplicate_step():
    ladder = LodLadder(LadderPolicy(floor_level=2))
    steps = ladder.plan_tile(TileLodState(tile_number=0), demand(0, acceptable=(0, 1)))
    assert rungs(steps) == [(Rung.FLOOR, 2), (Rung.DESIRED, 0)]
    assert all(step.rung != Rung.EXACT for step in steps)  # DESIRED==native


def test_exact_requested_appends_native_rung():
    ladder = LodLadder()
    state = TileLodState(
        tile_number=0, presented_level=2, resident_levels=(2,), exact_requested=True
    )
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
    ladder = LodLadder(LadderPolicy(floor_level=4))
    states = (TileLodState(tile_number=0), TileLodState(tile_number=1))
    steps = ladder.plan(states, demand(1))
    assert [(step.rung, step.tile_number) for step in steps] == [
        (Rung.FLOOR, 0),
        (Rung.FLOOR, 1),
        (Rung.DESIRED, 0),
        (Rung.DESIRED, 1),
    ]


def test_round_preview_floor_is_passed_once_across_heterogeneous_tile_state():
    """Progressive contract R2b: retention changes skips, never the round floor."""

    ladder = LodLadder(LadderPolicy(floor_level=9))
    states = (
        TileLodState(tile_number=0),
        TileLodState(tile_number=1, resident_levels=(0,), presented_level=0),
        TileLodState(tile_number=2, floor_available=True),
    )

    steps = ladder.plan(states, demand(2), preview_level=5)

    assert {step.level for step in steps if step.rung is Rung.FLOOR} == {5}


def test_foreign_retained_levels_are_reused_without_a_third_production_rung():
    """Progressive contract R1/R2: bound production, not visible reuse."""

    ladder = LodLadder(LadderPolicy(floor_level=9))
    states = (
        TileLodState(tile_number=0, resident_levels=(0,), presented_level=0),
        TileLodState(tile_number=1, resident_levels=(6,), presented_level=6),
        TileLodState(tile_number=2),
    )

    steps = ladder.plan(states, demand(2), preview_level=5)

    assert not any(step.tile_number == 0 for step in steps)
    assert {(step.tile_number, step.rung, step.level) for step in steps} == {
        (1, Rung.DESIRED, 2),
        (2, Rung.FLOOR, 5),
        (2, Rung.DESIRED, 2),
    }
    assert {step.level for step in steps} <= {2, 5}


def test_ladder_carries_canonical_tile_rank_across_every_rung():
    ladder = LodLadder(LadderPolicy(floor_level=4))
    steps = ladder.plan_tile(
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
        LadderPolicy(floor_level=4),
        TileLodState(tile_number=0, allow_preview=False),
        2,
    ),
    (
        "no_reduced_input",
        LadderPolicy(floor_level=4, reduced_input_available=False),
        TileLodState(tile_number=0, floor_available=False),
        2,
    ),
    (
        "already_covered",
        LadderPolicy(floor_level=4),
        TileLodState(tile_number=0, resident_levels=(4,), presented_level=4),
        2,
    ),
    (
        "cold_tile_gets_one",
        LadderPolicy(floor_level=4),
        TileLodState(tile_number=0),
        2,
    ),
    (
        "retained_floor_without_reduced_input",
        LadderPolicy(floor_level=4, reduced_input_available=False),
        TileLodState(tile_number=0, floor_available=True),
        2,
    ),
)


def test_coarse_rung_refusal_agrees_with_what_plan_tile_does():
    """A reported reason must be empty exactly when a coarse rung is planned."""

    for name, policy, state, level in _REFUSAL_CASES:
        ladder = LodLadder(policy)
        steps = ladder.plan_tile(state, demand(level))
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
    assert reason_for("no_reduced_input") == ladder_module.COARSE_RUNG_NO_REDUCED_INPUT
    assert reason_for("already_covered") == ladder_module.COARSE_RUNG_ALREADY_COVERED
    assert reason_for("cold_tile_gets_one") == ladder_module.COARSE_RUNG_PLANNED
    # A retained floor still earns a coarse rung with no reduced input at all.
    assert reason_for("retained_floor_without_reduced_input") == ladder_module.COARSE_RUNG_PLANNED


def test_measured_fft_state_is_refused_for_no_reduced_input_not_allow_preview():
    """The montage-axis FFT case, as measured on the 272-tile stage.

    `allow_preview` is True and the tile is genuinely blank; the ladder still
    plans nothing coarse because `reduced_input_available` is False (the FFT is
    not tile-local). Both candidates that were guessed from the source —
    `allow_preview` and the `preview_level < finest_available()` collapse — are
    downstream of this one, which is why the plan carries no `rung=0` either.
    """

    ladder = LodLadder(LadderPolicy(floor_level=4, reduced_input_available=False))
    state = TileLodState(tile_number=0, allow_preview=True, floor_available=False)

    from arrayscope.render import ladder as ladder_module

    steps = ladder.plan_tile(state, demand(2))
    assert [step.rung for step in steps] == [Rung.DESIRED]
    assert (
        ladder.coarse_rung_refusal(state, demand(2)) == ladder_module.COARSE_RUNG_NO_REDUCED_INPUT
    )


def test_raw_montage_has_one_coarse_rung_then_refines():
    """ADR 0059 removes the FLOOR/PREVIEW self-shadowing state."""

    from arrayscope.render import ladder as ladder_module

    ladder = LodLadder(LadderPolicy(floor_level=4))
    cold = TileLodState(tile_number=0)
    assert [step.rung for step in ladder.plan_tile(cold, demand(2))] == [
        Rung.FLOOR,
        Rung.DESIRED,
    ]

    floored = TileLodState(tile_number=0, resident_levels=(4,), presented_level=4)
    assert [step.rung for step in ladder.plan_tile(floored, demand(2))] == [Rung.DESIRED]
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

    ladder = LodLadder(LadderPolicy(floor_level=4))
    for quality in ("preview", "fallback"):
        state = TileLodState(
            tile_number=0,
            resident_levels=(2,),
            presented_level=2,
            presented_quality=quality,
            current_presentation_quality=quality,
        )
        rungs = [step.rung for step in ladder.plan_tile(state, demand(2))]
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
    assert [step.rung for step in ladder.plan_tile(settled, demand(2))] == []
