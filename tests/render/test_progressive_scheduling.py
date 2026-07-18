"""Qt-free contract tests for the COVERAGE -> REFINE policy owner."""

from arrayscope.kernel.task import Lane
from arrayscope.presentation import TileLifecycle, TileTarget
from arrayscope.render.progressive_scheduling import (
    ProgressiveSchedulingPolicy,
    SchedulingPhase,
    SchedulingWork,
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
        {
            tile: target.semantic_source_id
            for tile, target in targets.items()
        }
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

    assert policy.observe(lifecycle, on_refinement_replan=lambda: replans.append("duplicate")) is False
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
