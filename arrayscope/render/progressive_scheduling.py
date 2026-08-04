"""The one owner of COVERAGE -> REFINE scheduling policy and of R6 shedding.

The machine is scoped to the lifecycle target generation, not to a planning
wave or backend transaction.  Consumers receive immutable verdicts and never
reconstruct phase from ladder rows, queued work, histogram flags, or commit
state.

``RoundLoadShedding`` is the second machine here, at the altitude the
progressive-render contract assigns it ("Load shedding | scheduler | per
round").  It decides once per round, from whether the previous round reached
the backend at all, how much of the round to give up so the montage keeps
moving.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, IntEnum

from arrayscope.core.mutation_clock import ClockedState
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


def is_refinement_lane(lane: Lane) -> bool:
    """Whether ``lane`` carries the round's target pass rather than its first."""

    return Lane(lane) in _REFINEMENT_LANES


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

        return bool(self.phase is SchedulingPhase.REFINE or not is_refinement_lane(lane))

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


class ProgressiveSchedulingPolicy(ClockedState):
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
        already_covered: bool = False,
        session_id: int = 0,
        round_id: str = "",
    ) -> bool:
        """Adopt one required lifecycle scope; return whether it changed.

        ``already_covered`` is the caller's answer to "would the very first
        :meth:`observe` of this generation have closed coverage immediately?"
        It is a **conjunction of both ``observe`` coverage owners** — the
        lifecycle's ``first_pixels_presented`` over the adopted scope *and* the
        session's ``first_pass_pixels_presented`` over that same scope — because
        the two live call sites pass different owners and either of them may be
        the first to observe this generation.  Only when both already hold is
        the answer independent of which one wins the race.

        That conjunction is what makes opening in ``REFINE`` here safe rather
        than merely cheaper.  The phase reached is the phase the first
        observation would have reached anyway; no work is refused that today's
        code would have admitted, so **no wakeup is owed** and none is emitted.
        (There is no backstop for a lost wakeup: the montage watchdog observes
        and emits evidence only, so a phase edge that nobody was told about
        would strand the round permanently.)

        The default is ``False``, which is exactly today's behaviour: a caller
        that cannot cheaply prove both halves must let COVERAGE open and let
        ``observe`` close it.
        """

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
            if bool(progressive) and bool(required) and not bool(already_covered)
            else SchedulingPhase.REFINE
        )
        emit_trace(
            "scheduling_phase",
            event="scope_started",
            generation=int(self._generation),
            phase=str(self._phase.value),
            required_tiles=len(required),
            required_tile_numbers=required,
            already_covered=bool(already_covered),
            session_id=int(session_id),
            round_id=str(round_id or ""),
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


class ShedLevel(IntEnum):
    """How much of a round is given up so the montage keeps moving (R6)."""

    NONE = 0
    TARGET_PASS = 1
    RETAIN_ONLY = 2


@dataclass(frozen=True)
class RoundShedding:
    level: ShedLevel
    streak: int

    @property
    def drops_target_pass(self) -> bool:
        """R6 step 1: this round produces its preview pass and nothing more."""

        return self.level >= ShedLevel.TARGET_PASS

    @property
    def drops_atomic_handoff(self) -> bool:
        """R6 step 2: present what exists now instead of one complete swap."""

        return self.level >= ShedLevel.RETAIN_ONLY


NO_SHEDDING = RoundShedding(level=ShedLevel.NONE, streak=0)


class RoundLoadShedding(ClockedState):
    """R6 owner: one shedding verdict per round, relaxed when input settles.

    The "falling behind" signal is the product's own commit counter, not a
    clock: a round that opens while its predecessor never reached the backend
    at all is one the scheduler did not keep up with.  Consecutive such rounds
    escalate — first the target pass goes (R6 step 1), then the all-or-nothing
    successor handoff that is what actually freezes the montage (R6 step 2).

    Within a round the verdict may only relax, through ``release`` (R6 step
    3).  Re-arming inside a round would let an already-planned round lose work
    it had been promised.
    """

    TARGET_PASS_AFTER = 1
    RETAIN_ONLY_AFTER = 2

    def __init__(self) -> None:
        self._streak = 0
        self._verdict = NO_SHEDDING
        self.rounds_shed = 0

    @property
    def verdict(self) -> RoundShedding:
        return self._verdict

    def open_round(
        self,
        *,
        round_id: str = "",
        predecessor_delivered: bool,
        session_id: int = 0,
    ) -> RoundShedding:
        """Decide this round's shedding from whether the last one delivered."""

        self._streak = 0 if predecessor_delivered else self._streak + 1
        if self._streak >= self.RETAIN_ONLY_AFTER:
            level = ShedLevel.RETAIN_ONLY
        elif self._streak >= self.TARGET_PASS_AFTER:
            level = ShedLevel.TARGET_PASS
        else:
            level = ShedLevel.NONE
        self._verdict = RoundShedding(level=level, streak=int(self._streak))
        if level is not ShedLevel.NONE:
            self.rounds_shed += 1
        emit_trace(
            "round_load_shedding",
            event="round_opened",
            level=str(level.name).lower(),
            streak=int(self._streak),
            round_id=str(round_id or ""),
            session_id=int(session_id),
        )
        return self._verdict

    def release(self) -> bool:
        """Hand the live round its target pass back (R6 step 3).

        Shedding may postpone the target pass; it may never cancel it, and a
        round nothing supersedes has to reach target quality or the montage is
        stranded at preview forever.  The two callers are the edges that prove
        the round has stopped racing input: a plan that finds nothing left to
        shed, and the end of the interaction.

        The streak is deliberately untouched.  Whether the scheduler is still
        behind is re-decided when the next round opens, from whether this one
        reached the backend — not from here.
        """

        if self._verdict.level is ShedLevel.NONE:
            return False
        self._verdict = NO_SHEDDING
        emit_trace("round_load_shedding", event="released")
        return True


__all__ = [
    "NO_SHEDDING",
    "ProgressiveSchedulingPolicy",
    "RoundLoadShedding",
    "RoundShedding",
    "SchedulingPhase",
    "SchedulingVerdict",
    "SchedulingWork",
    "ShedLevel",
    "is_refinement_lane",
]
