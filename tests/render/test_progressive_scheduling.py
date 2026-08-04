"""Qt-free contract tests for the COVERAGE -> REFINE and R6 policy owners."""

from arrayscope.kernel.task import Lane
from arrayscope.presentation import TileLifecycle, TileTarget
from arrayscope.render.progressive_scheduling import (
    ProgressiveSchedulingPolicy,
    RoundLoadShedding,
    SchedulingPhase,
    SchedulingWork,
    ShedLevel,
    is_refinement_lane,
)


def _targets(*sources: int) -> dict[int, TileTarget]:
    return {
        tile: TileTarget(
            tile_number=tile,
            source_index=source,
            semantic_source_id=("source", source),
        )
        for tile, source in enumerate(sources)
    }


def _present(lifecycle: TileLifecycle, targets: dict[int, TileTarget]) -> None:
    lifecycle.backend_presented_snapshot(
        {tile: target.semantic_source_id for tile, target in targets.items()}
    )
    lifecycle.presentation_confirmed(tuple(targets))


def test_coverage_closes_once_from_lifecycle_first_pixel_truth():
    lifecycle = TileLifecycle()
    targets = _targets(10, 11)
    lifecycle.retarget(targets)
    policy = ProgressiveSchedulingPolicy()
    policy.retarget(tuple(targets.values()), tuple(targets), progressive=True)
    replans = []

    assert policy.verdict.phase is SchedulingPhase.COVERAGE
    assert policy.observe(lifecycle, on_refinement_replan=lambda: replans.append("refine")) is False

    _present(lifecycle, {0: targets[0]})
    assert policy.observe(lifecycle, on_refinement_replan=lambda: replans.append("refine")) is False

    _present(lifecycle, targets)
    assert policy.observe(lifecycle, on_refinement_replan=lambda: replans.append("refine")) is True
    assert policy.verdict.phase is SchedulingPhase.REFINE
    assert replans == ["refine"]

    assert (
        policy.observe(lifecycle, on_refinement_replan=lambda: replans.append("duplicate")) is False
    )
    assert replans == ["refine"]


def test_coverage_waits_for_owner_registered_rough_evidence():
    lifecycle = TileLifecycle()
    targets = _targets(10, 11)
    lifecycle.retarget(targets)
    policy = ProgressiveSchedulingPolicy()
    policy.retarget(tuple(targets.values()), tuple(targets), progressive=True)
    _present(lifecycle, targets)

    assert policy.set_coverage_evidence_pending(True) is True
    assert policy.observe(lifecycle) is False
    assert policy.verdict.phase is SchedulingPhase.COVERAGE

    assert policy.set_coverage_evidence_pending(False) is True
    assert policy.observe(lifecycle) is True
    assert policy.verdict.phase is SchedulingPhase.REFINE


def test_coverage_owner_can_require_preview_ack_beyond_generic_first_pixels():
    lifecycle = TileLifecycle()
    targets = _targets(10, 11)
    lifecycle.retarget(targets)
    _present(lifecycle, targets)
    first_pass_ready = False

    class CoverageOwner:
        def first_pass_pixels_presented(self):
            return first_pass_ready

    policy = ProgressiveSchedulingPolicy()
    policy.retarget(tuple(targets.values()), tuple(targets), progressive=True)

    assert lifecycle.first_pixels_presented(tuple(targets))
    assert policy.observe(CoverageOwner()) is False

    first_pass_ready = True
    assert policy.observe(CoverageOwner()) is True
    assert policy.verdict.phase is SchedulingPhase.REFINE


def test_same_slots_with_new_sources_open_a_new_scope_generation():
    lifecycle = TileLifecycle()
    first = _targets(10, 11)
    lifecycle.retarget(first)
    policy = ProgressiveSchedulingPolicy()
    policy.retarget(tuple(first.values()), tuple(first), progressive=True)
    _present(lifecycle, first)
    assert policy.observe(lifecycle) is True
    first_generation = policy.verdict.generation

    second = _targets(20, 21)
    lifecycle.retarget(second)
    policy.retarget(tuple(second.values()), tuple(second), progressive=True)

    assert policy.verdict.generation == first_generation + 1
    assert policy.verdict.phase is SchedulingPhase.COVERAGE
    assert policy.verdict.admits(SchedulingWork.COVERAGE)
    assert not policy.verdict.admits(SchedulingWork.REFINEMENT)
    assert not policy.verdict.admits_lane(Lane.DISPLAY_PREPARATION)
    assert policy.verdict.admits_lane(Lane.DISPLAY_PREVIEW)


def test_an_already_covered_scope_opens_in_refine_and_owes_no_wakeup():
    """A scope both observe owners would close at once never opens COVERAGE.

    Trace 5 phase 4: a pan across a tile boundary on a montage that was never
    uncovered opened COVERAGE, and the next verdict *inside the same retarget*
    closed it again and fired the replan wakeup -- a second full retarget for
    a scope that had been complete the whole time.

    The flag is only ever passed as the conjunction of both owners, so the
    phase reached here is the phase the first observation would have reached.
    Nothing is refused, so nothing is owed a wakeup -- and none is fired.
    """

    lifecycle = TileLifecycle()
    targets = _targets(10, 11)
    lifecycle.retarget(targets)
    _present(lifecycle, targets)

    policy = ProgressiveSchedulingPolicy()
    generation = policy.verdict.generation
    replans: list[str] = []

    assert (
        policy.retarget(
            tuple(targets.values()),
            tuple(targets),
            progressive=True,
            already_covered=True,
        )
        is True
    )

    assert policy.verdict.generation == generation + 1
    assert policy.verdict.required_tiles == tuple(targets)
    assert policy.verdict.phase is SchedulingPhase.REFINE
    assert policy.verdict.admits(SchedulingWork.COVERAGE)
    assert policy.verdict.admits(SchedulingWork.REFINEMENT)
    for lane in (
        Lane.DISPLAY_PREVIEW,
        Lane.DISPLAY_PREPARATION,
        Lane.VISIBLE_MATERIALIZATION,
        Lane.HISTOGRAM_REFINEMENT,
        Lane.SPECULATIVE_RESIDENCY,
    ):
        assert policy.verdict.admits_lane(lane), lane

    assert policy.observe(lifecycle, on_refinement_replan=lambda: replans.append("refine")) is False
    assert replans == []


def test_a_covered_scope_still_opens_coverage_when_the_flag_is_omitted():
    """The shortcut is opt-in per call; the default may never flip.

    Every other caller of ``retarget`` -- and every future one -- must keep
    getting COVERAGE plus the close wakeup, however complete the montage
    already is.  Only a caller that has itself computed the conjunction of
    both observe owners is allowed to skip that.
    """

    lifecycle = TileLifecycle()
    targets = _targets(10, 11)
    lifecycle.retarget(targets)
    _present(lifecycle, targets)

    policy = ProgressiveSchedulingPolicy()
    replans: list[str] = []

    assert policy.retarget(tuple(targets.values()), tuple(targets), progressive=True) is True

    assert policy.verdict.phase is SchedulingPhase.COVERAGE
    assert not policy.verdict.admits(SchedulingWork.REFINEMENT)
    assert policy.observe(lifecycle, on_refinement_replan=lambda: replans.append("refine")) is True
    assert replans == ["refine"]


def test_single_pass_scope_starts_in_refine():
    policy = ProgressiveSchedulingPolicy()
    targets = _targets(10, 11)

    policy.retarget(tuple(targets.values()), tuple(targets), progressive=False)

    assert policy.verdict.phase is SchedulingPhase.REFINE
    assert policy.verdict.admits(SchedulingWork.COVERAGE)
    assert policy.verdict.admits(SchedulingWork.REFINEMENT)
    assert policy.verdict.admits_lane(Lane.DISPLAY_PREPARATION)
    assert policy.verdict.admits_lane(Lane.HISTOGRAM_REFINEMENT)


def test_quality_retarget_does_not_create_a_new_required_scope_generation():
    policy = ProgressiveSchedulingPolicy()
    scope = ((0, 10, ("source", 10)), (1, 11, ("source", 11)))

    policy.retarget(scope, (0, 1), progressive=True)
    generation = policy.verdict.generation

    assert policy.retarget(scope, (0, 1), progressive=True) is False
    assert policy.verdict.generation == generation
    assert policy.verdict.phase is SchedulingPhase.COVERAGE


def test_shedding_escalates_only_while_rounds_deliver_nothing():
    shedding = RoundLoadShedding()

    first = shedding.open_round(predecessor_delivered=True)
    behind = shedding.open_round(predecessor_delivered=False)
    further_behind = shedding.open_round(predecessor_delivered=False)

    assert first.level is ShedLevel.NONE
    assert behind.level is ShedLevel.TARGET_PASS
    assert behind.drops_target_pass is True
    # R6 orders the two steps: the target pass goes before the montage is
    # allowed to show a partially replaced window.
    assert behind.drops_atomic_handoff is False
    assert further_behind.level is ShedLevel.RETAIN_ONLY
    assert further_behind.drops_atomic_handoff is True
    assert shedding.rounds_shed == 2


def test_a_round_that_reached_the_backend_clears_the_shedding_streak():
    shedding = RoundLoadShedding()

    shedding.open_round(predecessor_delivered=False)
    shedding.open_round(predecessor_delivered=False)
    recovered = shedding.open_round(predecessor_delivered=True)

    assert recovered.level is ShedLevel.NONE
    assert recovered.streak == 0


def test_release_hands_the_live_round_its_target_pass_back():
    """R6 step 3: shedding postpones the target pass, it never cancels it."""

    shedding = RoundLoadShedding()
    shedding.open_round(predecessor_delivered=False)
    shedding.open_round(predecessor_delivered=False)

    assert shedding.verdict.drops_atomic_handoff is True
    assert shedding.release() is True
    assert shedding.verdict.level is ShedLevel.NONE
    # Idempotent: nothing to relax means no replan is owed.
    assert shedding.release() is False
    # Releasing does not forget that the scheduler was behind: the next round
    # is still opening on a predecessor that reached the backend not once.
    assert shedding.open_round(predecessor_delivered=False).level is ShedLevel.RETAIN_ONLY


def test_refinement_lane_classification_has_one_owner():
    assert is_refinement_lane(Lane.DISPLAY_PREPARATION) is True
    assert is_refinement_lane(Lane.VISIBLE_MATERIALIZATION) is True
    assert is_refinement_lane(Lane.DISPLAY_PREVIEW) is False
