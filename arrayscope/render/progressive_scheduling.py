"""The one owner of COVERAGE -> REFINE scheduling policy.

The machine is scoped to the lifecycle target generation, not to a planning
wave or backend transaction.  Consumers receive immutable verdicts and never
reconstruct phase from ladder rows, queued work, histogram flags, or commit
state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from arrayscope.core.trace import emit_trace
from arrayscope.kernel.task import Lane


class SchedulingPhase(str, Enum):
    COVERAGE = "coverage"
    REFINE = "refine"


class SchedulingWork(str, Enum):
    COVERAGE = "coverage"
    REFINEMENT = "refinement"
    UNPHASED = "unphased"


_REFINEMENT_LANES = frozenset(
    {
        Lane.VISIBLE_MATERIALIZATION,
        Lane.DISPLAY_PREPARATION,
        Lane.HISTOGRAM_REFINEMENT,
        Lane.SPECULATIVE_RESIDENCY,
    }
)


@dataclass(frozen=True)
class SchedulingVerdict:
    generation: int
    phase: SchedulingPhase
    required_tiles: tuple[int, ...]

    @property
    def coverage_open(self) -> bool:
        return self.phase is SchedulingPhase.COVERAGE

    @property
    def refinement_admissible(self) -> bool:
        return self.phase is SchedulingPhase.REFINE

    def admits_lane(self, lane: Lane) -> bool:
        """Whether work on ``lane`` belongs in the current compute phase."""

        return bool(self.phase is SchedulingPhase.REFINE or Lane(lane) not in _REFINEMENT_LANES)

    def admits(self, work: SchedulingWork) -> bool:
        """Admit prerequisites always; admit refinements only after coverage.

        ``REFINE`` is a one-way permission edge, not a ban on a late or
        single-pass correctness producer.  This asymmetry keeps CPU-LUT
        sessions (which start in REFINE) from stranding their first-frame
        level evidence while still making phase-2 work impossible during
        COVERAGE.
        """

        work = SchedulingWork(work)
        return bool(
            work in {SchedulingWork.UNPHASED, SchedulingWork.COVERAGE}
            or (work is SchedulingWork.REFINEMENT and self.refinement_admissible)
        )


class ProgressiveSchedulingPolicy:
    """Per-required-scope-generation COVERAGE -> REFINE state machine."""

    def __init__(self) -> None:
        self._scope_signature: object = None
        self._generation = 0
        self._required_tiles: tuple[int, ...] = ()
        self._phase = SchedulingPhase.REFINE
        self._coverage_evidence_pending = False

    @property
    def verdict(self) -> SchedulingVerdict:
        return SchedulingVerdict(
            generation=int(self._generation),
            phase=self._phase,
            required_tiles=self._required_tiles,
        )

    def retarget(
        self,
        scope_signature: object,
        required_tiles,
        *,
        progressive: bool,
    ) -> bool:
        """Adopt one required lifecycle scope; return whether it changed."""

        required = tuple(dict.fromkeys(int(tile) for tile in tuple(required_tiles or ())))
        signature = (scope_signature, required, bool(progressive))
        if signature == self._scope_signature:
            return False
        self._scope_signature = signature
        self._generation += 1
        self._required_tiles = required
        self._coverage_evidence_pending = False
        self._phase = (
            SchedulingPhase.COVERAGE
            if bool(progressive) and bool(required)
            else SchedulingPhase.REFINE
        )
        emit_trace(
            "scheduling_phase",
            event="scope_started",
            generation=int(self._generation),
            phase=str(self._phase.value),
            required_tiles=len(required),
        )
        return True

    def set_coverage_evidence_pending(self, pending: bool) -> bool:
        """Own the phase-1 evidence barrier for the current generation."""

        pending = bool(pending and self._phase is SchedulingPhase.COVERAGE)
        if pending == self._coverage_evidence_pending:
            return False
        self._coverage_evidence_pending = pending
        emit_trace(
            "scheduling_phase",
            event="coverage_evidence_pending" if pending else "coverage_evidence_ready",
            generation=int(self._generation),
            phase=str(self._phase.value),
            required_tiles=len(self._required_tiles),
        )
        return True

    def observe(
        self,
        coverage_owner,
        *,
        on_refinement_replan: Callable[[], None] | None = None,
    ) -> bool:
        """Advance on acknowledged first-pass truth and own the close wakeup."""

        if self._phase is not SchedulingPhase.COVERAGE:
            return False
        if self._coverage_evidence_pending:
            return False
        first_pass_presented = getattr(coverage_owner, "first_pass_pixels_presented", None)
        if callable(first_pass_presented):
            coverage_complete = bool(first_pass_presented())
        else:
            coverage_complete = bool(coverage_owner.first_pixels_presented(self._required_tiles))
        if not coverage_complete:
            return False
        self._phase = SchedulingPhase.REFINE
        emit_trace(
            "scheduling_phase",
            event="coverage_closed",
            generation=int(self._generation),
            phase=str(self._phase.value),
            required_tiles=len(self._required_tiles),
        )
        if on_refinement_replan is not None:
            on_refinement_replan()
        return True


__all__ = [
    "ProgressiveSchedulingPolicy",
    "SchedulingPhase",
    "SchedulingVerdict",
    "SchedulingWork",
]
