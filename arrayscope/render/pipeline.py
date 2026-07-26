"""FramePipeline: ladder plans → kernel tasks → commit batches.

One modular chunk per stage, one owner per piece of state:

    stage                 owner of state              runs on
    -----                 --------------              -------
    intent snapshot       RenderIntent (immutable)    GUI thread (cheap)
    ladder planning       LodLadder (stateless)       kernel task
    materialize/reduce    kernel (task records)       worker threads
    tile bookkeeping      TileLifecycle               GUI thread (drain)
    commit batching       this pipeline (queue only)  GUI thread (drain)
    GPU/CPU application   backend adapter             GUI thread (bounded)
    acknowledgement       TileLifecycle               GUI thread (drain)

The pipeline never owns tile state (TileLifecycle does), never runs work
(the kernel does), and never paces with timers (completions drive it; the
bridge's fallback is the only safety net). R2 supplies concrete effects for
worker evaluation, stage dependencies, commit batching, and backend
acknowledgement.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Protocol

from arrayscope.kernel import Kernel, Lane, Priority, Supersession, TaskSpec
from arrayscope.render.ladder import LodLadder, Rung, RungStep, TileLodState
from arrayscope.render.progressive_scheduling import SchedulingVerdict
from arrayscope.render.stages import (
    CommitBatch,
    LodAdmissionScope,
    PipelineCounters,
    RenderIntent,
    RungEvaluationTimings,
)


class PipelineEffects(Protocol):
    """What the pipeline needs from the outside world, nothing more.

    Implementations must be side-effect-only at their boundary: evaluation
    functions run on worker threads and must not touch Qt; `apply_commit`
    runs on the GUI thread and is the ONLY place backend adapters are
    reached.
    """

    def evaluate_rung(self, intent: RenderIntent, step: RungStep) -> Callable[[], Any]:
        """Return the worker-thread callable producing this rung's payload.

        R2 ports exact and preview evaluation into ``render.effects``; the
        concrete effect bridge selects the callable for this rung.
        """
        ...

    def apply_commit(self, batch: CommitBatch) -> None:
        """Apply one bounded batch through the backend adapter (GUI thread).

        The concrete effect consumes ``CommitBatch`` rows, builds the shared
        tiled-presentation delta, presents through the surface contract, and
        feeds backend acknowledgement back to ``TileLifecycle``.
        """
        ...

    def tile_states(
        self,
        intent: RenderIntent,
        demand,
        scope: LodAdmissionScope,
    ) -> tuple[TileLodState, ...]:
        """Snapshot per-tile lod state from TileLifecycle claims.

        Implementations read ``TileLifecycle`` records and pyramid residency
        without mutating either source of truth.
        """
        ...

    def prepare_rung(self, intent: RenderIntent, step: RungStep) -> bool:
        """Prepare cheap, reversible state before kernel admission."""
        ...

    def rung_admitted(self, intent: RenderIntent, step: RungStep, task_key: object) -> None:
        """Record that the kernel accepted this rung task."""
        ...

    def rung_deps(self, intent: RenderIntent, step: RungStep) -> tuple[object, ...]:
        """Kernel task keys this rung waits for."""
        ...

    def rung_dropped(self, intent: RenderIntent, step: RungStep) -> None:
        """Release lifecycle state when a submitted rung cannot deliver."""
        ...

    def retained_native_source_available(self, intent: RenderIntent, step: RungStep) -> bool:
        """Return whether a native rung can use retained/staged source data."""
        ...

    def scheduling_verdict(self) -> SchedulingVerdict:
        """Return the current required-scope phase verdict."""

        ...


@dataclass(frozen=True)
class _RungKey:
    """Kernel task key for one tile rung under one semantic target.

    Deliberately viewport-free: camera-only changes must never restart
    materialization (core invariant). Viewport moves change *which* steps
    the ladder plans (levels, priorities, ordering) — an unchanged step
    resubmits the same key and the kernel dedupes it.
    """

    semantic_key: object
    tile_number: int
    source_id: object
    rung: int
    level: int


class FramePipeline:
    """Schedules quality progression for regions of any image frame.

    Supersession contract (the part that kept breaking pre-redesign, now in
    exactly one place):

    - Semantic changes (`intent.semantic_key`) clear the whole scope: no
      task from an older document/selection may deliver.
    - Viewport changes supersede per-tile *rung* families — running
      reusable evaluations may finish into caches, but only current-target
      results commit.
    - Presentation-only changes never cancel materialization (they are not
      rung work at all; levels/LUT ride commit metadata).
    """

    SCOPE = "frame"
    ADMISSION_CHUNK = 24

    def __init__(
        self,
        kernel: Kernel,
        effects: PipelineEffects,
        ladder: LodLadder | None = None,
        *,
        commit_max_items: int = 8,
    ) -> None:
        self.kernel = kernel
        self.effects = effects
        self.ladder = ladder or LodLadder()
        self.counters = PipelineCounters()
        self.rung_timings = RungEvaluationTimings()
        self._commit_max_items = max(1, int(commit_max_items))
        self._current_intent: RenderIntent | None = None
        self._ready_upserts: list = []
        self._pending_admissions: deque[tuple[RenderIntent, RungStep]] = deque()
        self._admission_generation = 0
        self._admission_continuation_armed = False
        self._admission_continuation_sequence = 0

    # ----------------------------------------------------------- lifecycle

    def retarget(self, intent: RenderIntent, demand, scope: LodAdmissionScope) -> int:
        """Adopt a new intent; schedule exactly the missing rung steps.

        Returns the number of submitted kernel tasks. Safe to call on every
        interaction event: convergence means zero submissions, and stale
        work is superseded by family, never hunted down by ad-hoc flags.
        """

        previous = self._current_intent
        self._current_intent = intent
        self._admission_generation += 1
        admission_generation = int(self._admission_generation)
        self._admission_continuation_armed = False
        self._pending_admissions.clear()
        self.counters.intents += 1
        if previous is not None and previous.semantic_key != intent.semantic_key:
            # Semantic change: everything under the old target is stale.
            self.kernel.clear_scope(self._scope(previous.semantic_key))
            for queued_intent, step, _payload in tuple(self._ready_upserts):
                if queued_intent.semantic_key != intent.semantic_key:
                    self.effects.rung_dropped(queued_intent, step)
            self._ready_upserts.clear()

        verdict = self.effects.scheduling_verdict()
        states = self.effects.tile_states(intent, demand, scope)
        session = getattr(self.effects, "session", None)
        session_id = int(getattr(session, "session_id", 0) or 0)
        if session_id > 0:
            self.kernel.rerank_unstarted_tile_tasks(
                session_id=session_id,
                scheduling_ranks={
                    int(state.tile_number): int(state.scheduling_rank) for state in states
                },
            )
        steps = self.ladder.plan(states, demand, verdict)
        self.last_plan_states = tuple(
            (
                int(state.tile_number),
                tuple(int(level) for level in state.resident_levels),
                state.presented_level,
                state.ready_level,
                bool(state.allow_preview),
            )
            for state in states
        )
        self.last_plan_steps = tuple(
            (int(step.tile_number), int(step.rung), int(step.level)) for step in steps
        )
        self.counters.ladder_plans += 1
        submitted = 0
        presented_preview_tiles = {
            int(state.tile_number)
            for state in states
            if str(getattr(state, "presented_quality", "exact") or "exact") == "preview"
        }
        # Cross-rung/cross-tile ordering comes from priorities plus this
        # submission order (the kernel heap is FIFO within equal priority).
        # NEVER express ordering through `deps`: dependencies fail-propagate,
        # so a skipped floor would park its tile's exact work forever.
        for step in steps:
            if self._defer_native_quality_during_interaction(
                intent, step, presented_preview_tiles
            ) and not self.effects.retained_native_source_available(intent, step):
                self.counters.interactive_native_deferred += 1
                continue
            self._pending_admissions.append((intent, step))
        submitted += self._drain_pending_admissions(admission_generation)
        if self._pending_admissions:
            self._arm_admission_continuation(admission_generation)
        self._flush_ready()
        return submitted

    def close(self) -> None:
        self._admission_generation += 1
        self._pending_admissions.clear()
        self._admission_continuation_armed = False
        if self._current_intent is not None:
            self.kernel.clear_scope(self._scope(self._current_intent.semantic_key))

    # ------------------------------------------------------------ internal

    def _scope(self, semantic_key: object) -> str:
        return f"{self.SCOPE}:{semantic_key!r}"

    def _rung_key(self, intent: RenderIntent, step: RungStep) -> _RungKey:
        return _RungKey(
            semantic_key=intent.semantic_key,
            tile_number=int(step.tile_number),
            source_id=intent.source_id_for_tile(int(step.tile_number)),
            rung=int(step.rung),
            level=int(step.level),
        )

    def _defer_native_quality_during_interaction(
        self,
        intent: RenderIntent,
        step: RungStep,
        presented_preview_tiles: set[int],
    ) -> bool:
        """Keep active gestures on correctness rungs; admit native with proof.

        Native evaluation is the expensive quality rung in both spellings:
        explicit EXACT, and DESIRED(level=0) when the viewport is zoomed to
        native.  For opaque pipelines, DESIRED(reduce_from_native=True) also
        means "run native, then reduce".  During active viewport movement,
        cold native tasks are usually superseded before they can present.
        Floor, preview, resident remaps, already-presented payloads, and
        stage-backed extraction from retained source data are the correctness
        path; cold stage planning is submitted through the kernel and
        superseded by newer retargets.
        """

        if not bool(getattr(intent, "interactive", False)):
            return False
        if int(step.tile_number) in presented_preview_tiles:
            return False
        if step.rung == Rung.EXACT:
            return True
        if step.rung != Rung.DESIRED:
            return False
        return int(step.level) <= 0 or bool(step.reduce_from_native)

    def _submit_step(
        self,
        intent: RenderIntent,
        step: RungStep,
        *,
        step_key: _RungKey,
    ) -> bool:
        if not self.effects.prepare_rung(intent, step):
            return False
        session = getattr(self.effects, "session", None)
        coverage_pass_open = bool(self.effects.scheduling_verdict().coverage_open)
        # Phase follows the work's role, not its historical rung name.
        # DESIRED on a blank tile runs in DISPLAY_PREVIEW and is phase-1
        # coverage; the same rung on an already-covered tile runs in
        # DISPLAY_PREPARATION and is phase-2 refinement. Exact work admitted
        # while another tile still lacks first pixels is refinement too.
        presentation_phase = (
            2
            if step.lane == Lane.DISPLAY_PREPARATION
            or (step.rung == Rung.EXACT and coverage_pass_open)
            else 1
        )
        spec = TaskSpec(
            key=step_key,
            fn=self._timed_rung_evaluation(step, self.effects.evaluate_rung(intent, step)),
            lane=step.lane,
            priority=step.priority,
            scheduling_rank=int(step.scheduling_rank),
            presentation_phase=presentation_phase,
            coverage_pass_open=coverage_pass_open,
            session_id=int(getattr(session, "session_id", 0) or 0),
            tile_number=int(step.tile_number),
            # Ladder provenance for the trace only; identity stays `step_key`.
            rung=int(step.rung),
            level=int(step.level),
            scope=self._scope(intent.semantic_key),
            deps=self.effects.rung_deps(intent, step),
            # Latest-only per tile+rung: a *demand/level* change replaces the
            # target; camera-only changes keep the same value and therefore
            # never invalidate running or queued materialization. Reusable
            # results may still land in caches when superseded.
            supersession=Supersession(
                family=("rung", intent.semantic_key, step.tile_number, int(step.rung)),
                value=(intent.source_id_for_tile(int(step.tile_number)), int(step.level)),
            ),
            reusable=True,
            pass_token=True,
        )
        handle = self.kernel.submit(
            spec,
            on_done=lambda payload, intent=intent, step=step: self._on_rung_done(
                intent, step, payload
            ),
            on_error=lambda exc, intent=intent, step=step: self._on_rung_error(intent, step, exc),
            on_stale=lambda intent=intent, step=step: self._on_rung_stale(intent, step),
            on_reuse=lambda payload, intent=intent, step=step: self._on_rung_reusable(
                intent, step, payload
            ),
        )
        if handle is not None:
            self.effects.rung_admitted(intent, step, step_key)
            self.counters.tasks_submitted += 1
            return True
        self.effects.rung_dropped(intent, step)
        return False

    def _timed_rung_evaluation(self, step: RungStep, evaluate: Callable[..., Any]):
        """Wrap a rung's worker function so its cost lands in ``rung_timings``.

        The wrapper is the whole evaluation and nothing else: no per-tile
        allocation, one clock read on each side, and the record happens in a
        ``finally`` so a cancelled or failed rung still reports the work it
        burned.
        """

        rung = int(step.rung)
        level = int(step.level)
        timings = self.rung_timings

        def timed(*args, **kwargs):
            started_ns = perf_counter_ns()
            try:
                return evaluate(*args, **kwargs)
            finally:
                timings.record(rung, level, perf_counter_ns() - started_ns)

        return timed

    def _drain_pending_admissions(self, generation: int) -> int:
        """Submit one bounded chunk of already-planned visible rung work."""

        if int(generation) != int(self._admission_generation):
            return 0
        submitted = 0
        inspected = 0
        while self._pending_admissions and inspected < int(self.ADMISSION_CHUNK):
            queued_intent, step = self._pending_admissions.popleft()
            inspected += 1
            current = self._current_intent
            if current is None or queued_intent is not current:
                continue
            if self._submit_step(queued_intent, step, step_key=self._rung_key(queued_intent, step)):
                submitted += 1
        return submitted

    def _arm_admission_continuation(self, generation: int) -> None:
        if int(generation) != int(self._admission_generation) or not self._pending_admissions:
            return
        if self._admission_continuation_armed:
            return
        intent = self._current_intent
        if intent is None:
            return
        self._admission_continuation_armed = True
        self._admission_continuation_sequence += 1
        sequence = int(self._admission_continuation_sequence)

        def done(_value=None, generation=generation):
            if int(generation) != int(self._admission_generation):
                return
            self._admission_continuation_armed = False
            self._drain_pending_admissions(generation)
            if self._pending_admissions:
                self._arm_admission_continuation(generation)

        def stale(generation=generation):
            if int(generation) == int(self._admission_generation):
                self._admission_continuation_armed = False

        def failed(exc, generation=generation):
            if int(generation) == int(self._admission_generation):
                self._admission_continuation_armed = False
            raise exc

        handle = self.kernel.submit(
            TaskSpec(
                key=("frame-admission", id(self), generation, sequence),
                fn=lambda: True,
                lane=Lane.VISIBLE_PLANNING,
                priority=Priority.VISIBLE_IMAGE,
                scope=self._scope(intent.semantic_key),
                supersession=Supersession(("frame-admission", id(self)), generation),
                reusable=False,
                pass_token=False,
            ),
            on_done=done,
            on_stale=stale,
            on_error=failed,
        )
        if handle is None:
            self._admission_continuation_armed = False

    # Handlers run on the GUI thread (kernel bridge drain).

    def _on_rung_done(self, intent: RenderIntent, step: RungStep, payload) -> None:
        current = self._current_intent
        if not self._intent_step_matches_current(intent, step, current):
            self.effects.rung_dropped(intent, step)
            return
        if payload is None:
            self.effects.rung_dropped(intent, step)
            return
        self._ready_upserts.append((intent, step, payload))
        self._flush_ready()

    def _on_rung_reusable(self, intent: RenderIntent, step: RungStep, payload) -> None:
        # Stale-but-reusable: worker side may have populated caches, but this
        # rung must never commit. It still owns lifecycle claims from
        # prepare_rung(), so release those with the preparing intent.
        self.effects.rung_dropped(intent, step)

    @staticmethod
    def _intent_step_matches_current(intent: RenderIntent, step: RungStep, current) -> bool:
        if current is None or intent.semantic_key != current.semantic_key:
            return False
        previous_source = intent.source_id_for_tile(int(step.tile_number))
        current_source = current.source_id_for_tile(int(step.tile_number))
        return previous_source == current_source

    def _on_rung_stale(self, intent: RenderIntent, step: RungStep) -> None:
        self.effects.rung_dropped(intent, step)

    def _on_rung_error(self, intent: RenderIntent, step: RungStep, exc: BaseException) -> None:
        self.effects.rung_dropped(intent, step)
        raise exc

    def _flush_ready(self) -> None:
        """Hand every ready payload to the effects for admission.

        ``apply_commit`` is admission-only (cheap bookkeeping); the heavy
        presentation commit is coalesced behind the effects' presentation
        gate, one bounded commit per event-loop turn. Holding payloads back
        here would only add a second queue with its own lost-wakeup risk.
        """

        intent = self._current_intent
        if intent is None or not self._ready_upserts:
            return
        queued, self._ready_upserts = self._ready_upserts, []
        batch_items = []
        for queued_intent, step, payload in queued:
            if self._intent_step_matches_current(queued_intent, step, intent):
                batch_items.append((step, payload))
            else:
                self.effects.rung_dropped(queued_intent, step)
        if not batch_items:
            return
        batch = CommitBatch(
            semantic_key=intent.semantic_key,
            presentation_key=intent.presentation_key,
            upserts=tuple(batch_items),
            max_items=self._commit_max_items,
        )
        self.counters.commit_batches += 1
        self.effects.apply_commit(batch)


__all__ = ["FramePipeline", "PipelineEffects"]
