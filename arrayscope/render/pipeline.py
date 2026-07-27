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

from collections import Counter, deque
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


@dataclass(frozen=True)
class _PreviewBatchKey:
    """One worker task covering a complete, immutable coarse montage scope."""

    semantic_key: object
    source_ids: tuple[object, ...]
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
    PREVIEW_FANOUT_MIN_ITEMS = 256
    PREVIEW_FANOUT_MAX_ITEMS = 512

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
        # Cumulative over the session's plans, in tile-plans (one tile in one
        # plan), NOT tiles.  Deliberately not last-plan-only: the last plan of
        # a settled fill is the converged one, and its refusal ('allow_preview
        # false: covered') is the opposite of the cold-fill answer someone
        # asking "why was there no preview" needs.  The trace keeps the
        # per-plan, time-resolved view.
        self._coarse_rung_refusals: Counter[str] = Counter()
        self.last_coarse_rung_refusals: tuple[tuple[str, int], ...] = ()

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
        # TODO(R2b): this reads the floor back off the ladder policy it is
        # about to plan with, and `plan` falls back to that same value when the
        # argument is omitted -- so passing it is currently a no-op round trip
        # that documents ownership without enforcing it. The value does now
        # originate in one place (`render.lod.round_preview_level`, via
        # `selected_lod_factor`), but it reaches here by `frame_runtime`
        # mutating `ladder.policy` before every plan. Closing this needs a
        # round identity to pin the floor to; see the contract's R2b.
        preview_level = max(0, int(self.ladder.policy.floor_level))
        steps = self.ladder.plan(
            states,
            demand,
            verdict,
            preview_level=preview_level,
        )
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
        # Why the coarse rungs are absent, aggregated per plan.  A plan with no
        # rung=0/rung=1 step says nothing about why on its own, and reading the
        # code to guess got the FFT answer wrong once already.  One Counter bump
        # per coarse-less tile, and only for tiles that actually lack one.
        coarse_tiles = {int(step.tile_number) for step in steps if step.rung == Rung.FLOOR}
        refusals: Counter[str] = Counter()
        for state in states:
            if int(state.tile_number) in coarse_tiles:
                continue
            refusals[self.ladder.coarse_rung_refusal(state, demand, verdict)] += 1
        self.last_coarse_rung_refusals = tuple(sorted(refusals.items()))
        self._coarse_rung_refusals.update(refusals)
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
        if self._submit_preview_batch(admission_generation, verdict, states):
            submitted += 1
        else:
            submitted += self._drain_pending_admissions(admission_generation)
        if self._pending_admissions:
            self._arm_admission_continuation(admission_generation)
        self._flush_ready()
        return submitted

    def coarse_rung_refusals(self) -> tuple[tuple[str, int], ...]:
        """Cumulative "why no coarse rung", in tile-plans, commonest first."""

        return tuple(sorted(self._coarse_rung_refusals.items(), key=lambda row: (-row[1], row[0])))

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
        scheduling_verdict = self.effects.scheduling_verdict()
        coverage_pass_open = bool(scheduling_verdict.coverage_open)
        # Phase follows the work's role, not its historical rung name.
        # DESIRED runs in DISPLAY_PREVIEW only when it is the pipeline's
        # first-and-only presentable rung. A target behind FLOOR is always
        # DISPLAY_PREPARATION and therefore cannot be submitted while
        # required-set coverage is open. Exact work admitted while another
        # tile still lacks first pixels is refinement too.
        presentation_phase = (
            2
            if step.lane == Lane.DISPLAY_PREPARATION
            or (step.rung == Rung.EXACT and coverage_pass_open)
            else 1
        )
        # One mutable flag per submission, written by the worker wrapper and
        # read by the GUI-thread callbacks.  A dropped rung is only *spent*
        # work if its evaluation actually ran: the kernel also supersedes
        # queued tasks that never started, and counting those as discarded
        # priced 32 discards against 8 evaluations -- a 45.9 s waste inside a
        # 5.4 s stage, which is how the conflation announced itself.
        evaluated = [False]
        spec = TaskSpec(
            key=step_key,
            fn=self._timed_rung_evaluation(
                step, self.effects.evaluate_rung(intent, step), evaluated
            ),
            lane=step.lane,
            priority=step.priority,
            scheduling_rank=int(step.scheduling_rank),
            presentation_phase=presentation_phase,
            coverage_pass_open=coverage_pass_open,
            session_id=int(getattr(session, "session_id", 0) or 0),
            tile_number=int(step.tile_number),
            scheduling_generation=int(scheduling_verdict.generation),
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
            on_done=lambda payload, intent=intent, step=step, ran=evaluated: self._on_rung_done(
                intent, step, payload, ran
            ),
            on_error=lambda exc, intent=intent, step=step, ran=evaluated: self._on_rung_error(
                intent, step, exc, ran
            ),
            on_stale=lambda intent=intent, step=step, ran=evaluated: self._on_rung_stale(
                intent, step, ran
            ),
            on_reuse=lambda payload, intent=intent, step=step, ran=evaluated: (
                self._on_rung_reusable(intent, step, payload, ran)
            ),
        )
        if handle is not None:
            self.effects.rung_admitted(intent, step, step_key)
            self.counters.tasks_submitted += 1
            return True
        self.effects.rung_dropped(intent, step)
        return False

    def _timed_rung_evaluation(
        self, step: RungStep, evaluate: Callable[..., Any], evaluated: list[bool]
    ):
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
            evaluated[0] = True
            try:
                return evaluate(*args, **kwargs)
            finally:
                timings.record(rung, level, perf_counter_ns() - started_ns)

        return timed

    def _submit_preview_batch(
        self,
        generation: int,
        verdict: SchedulingVerdict,
        states: tuple[TileLodState, ...],
    ) -> bool:
        """Submit one complete coarse montage as one worker task.

        The task consumes one ``DISPLAY_PREVIEW`` worker slot, but its internal
        256–512-tile work is not chunked or elapsed-time-governed. Lifecycle
        claims and physical acknowledgements remain one per required tile.
        """

        if int(generation) != int(self._admission_generation):
            return False
        pending = tuple(self._pending_admissions)
        count = len(pending)
        evaluate_batch = getattr(self.effects, "evaluate_preview_batch", None)
        if (
            not callable(evaluate_batch)
            or not bool(verdict.coverage_open)
            or not (self.PREVIEW_FANOUT_MIN_ITEMS <= count <= self.PREVIEW_FANOUT_MAX_ITEMS)
            or not pending
            or any(intent is not self._current_intent for intent, _step in pending)
            or any(step.rung != Rung.FLOOR for _intent, step in pending)
        ):
            return False
        intent = pending[0][0]
        steps = tuple(step for _intent, step in pending)
        tile_numbers = tuple(int(step.tile_number) for step in steps)
        covered_tiles = {
            int(state.tile_number)
            for state in states
            if (
                state.presented_level is not None
                or state.ready_level is not None
                or bool(state.resident_levels)
            )
        }
        required_tiles = {int(tile) for tile in verdict.required_tiles}
        if (
            len(tile_numbers) != len(set(tile_numbers))
            or set(tile_numbers) | covered_tiles != required_tiles
        ):
            return False
        if len({int(step.level) for step in steps}) != 1:
            return False

        prepared = []
        for step in steps:
            if not self.effects.prepare_rung(intent, step):
                for claimed in prepared:
                    self.effects.rung_dropped(intent, claimed)
                return False
            prepared.append(step)

        source_ids = tuple(intent.source_id_for_tile(int(step.tile_number)) for step in steps)
        batch_key = _PreviewBatchKey(
            semantic_key=intent.semantic_key,
            source_ids=source_ids,
            level=int(steps[0].level),
        )
        session = getattr(self.effects, "session", None)
        evaluated = [False]
        deps = tuple(
            dict.fromkeys(
                dependency for step in steps for dependency in self.effects.rung_deps(intent, step)
            )
        )
        spec = TaskSpec(
            key=batch_key,
            fn=self._timed_rung_evaluation(
                steps[0],
                evaluate_batch(intent, steps),
                evaluated,
            ),
            lane=Lane.DISPLAY_PREVIEW,
            priority=min((step.priority for step in steps), default=Priority.INTERACTIVE),
            scheduling_rank=min((int(step.scheduling_rank) for step in steps), default=0),
            presentation_phase=1,
            coverage_pass_open=True,
            session_id=int(getattr(session, "session_id", 0) or 0),
            tile_number=-1,
            scheduling_generation=int(verdict.generation),
            rung=int(Rung.FLOOR),
            level=int(steps[0].level),
            scope=self._scope(intent.semantic_key),
            deps=deps,
            supersession=Supersession(
                family=("preview-batch", intent.semantic_key),
                value=(source_ids, int(steps[0].level)),
            ),
            reusable=True,
            pass_token=True,
        )
        handle = self.kernel.submit(
            spec,
            on_done=lambda payload, intent=intent, steps=steps, ran=evaluated: (
                self._on_preview_batch_done(intent, steps, payload, ran)
            ),
            on_error=lambda exc, intent=intent, steps=steps, ran=evaluated: (
                self._on_preview_batch_error(intent, steps, exc, ran)
            ),
            on_stale=lambda intent=intent, steps=steps, ran=evaluated: self._drop_preview_batch(
                intent, steps, ran
            ),
            on_reuse=lambda _payload, intent=intent, steps=steps, ran=evaluated: (
                self._drop_preview_batch(intent, steps, ran)
            ),
        )
        if handle is None:
            for step in steps:
                self.effects.rung_dropped(intent, step)
            return False
        self._pending_admissions.clear()
        for step in steps:
            self.effects.rung_admitted(intent, step, batch_key)
        self.counters.tasks_submitted += 1
        return True

    def _drain_pending_admissions(
        self,
        generation: int,
        *,
        max_inspected: int | None = None,
    ) -> int:
        """Submit one bounded chunk of already-planned visible rung work."""

        if int(generation) != int(self._admission_generation):
            return 0
        inspection_limit = max(
            1,
            int(self.ADMISSION_CHUNK if max_inspected is None else max_inspected),
        )
        submitted = 0
        inspected = 0
        while self._pending_admissions and inspected < inspection_limit:
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

    def _on_rung_done(
        self, intent: RenderIntent, step: RungStep, payload, evaluated: list[bool] | None = None
    ) -> None:
        current = self._current_intent
        if not self._intent_step_matches_current(intent, step, current):
            self._discard_rung(intent, step, evaluated)
            return
        if payload is None:
            self._discard_rung(intent, step, evaluated)
            return
        self._ready_upserts.append((intent, step, payload))
        self._flush_ready()

    def _on_preview_batch_done(
        self,
        intent: RenderIntent,
        steps: tuple[RungStep, ...],
        payload,
        evaluated: list[bool] | None = None,
    ) -> None:
        current = self._current_intent
        if any(not self._intent_step_matches_current(intent, step, current) for step in steps):
            self._drop_preview_batch(intent, steps, evaluated)
            return
        rows = tuple(payload or ())
        by_tile = {
            int(row[0]): tuple(row[1:]) for row in rows if isinstance(row, tuple) and len(row) >= 2
        }
        expected = {int(step.tile_number) for step in steps}
        if set(by_tile) != expected:
            self._drop_preview_batch(intent, steps, evaluated)
            return
        self._ready_upserts.extend((intent, step, by_tile[int(step.tile_number)]) for step in steps)
        self._flush_ready()

    def _drop_preview_batch(
        self,
        intent: RenderIntent,
        steps: tuple[RungStep, ...],
        evaluated: list[bool] | None = None,
    ) -> None:
        discarded_recorded = False
        for step in steps:
            if evaluated is not None and evaluated[0] and not discarded_recorded:
                self.rung_timings.record_discarded(int(step.rung), int(step.level))
                evaluated[0] = False
                discarded_recorded = True
            self.effects.rung_dropped(intent, step)

    def _on_preview_batch_error(
        self,
        intent: RenderIntent,
        steps: tuple[RungStep, ...],
        exc: BaseException,
        evaluated: list[bool] | None = None,
    ) -> None:
        self._drop_preview_batch(intent, steps, evaluated)
        raise exc

    def _on_rung_reusable(
        self, intent: RenderIntent, step: RungStep, payload, evaluated: list[bool] | None = None
    ) -> None:
        # Stale-but-reusable: worker side may have populated caches, but this
        # rung must never commit. It still owns lifecycle claims from
        # prepare_rung(), so release those with the preparing intent.
        self._discard_rung(intent, step, evaluated)

    def _discard_rung(
        self, intent: RenderIntent, step: RungStep, evaluated: list[bool] | None = None
    ) -> None:
        """Release a rung that can never commit; count it only if it ran.

        A rung whose evaluation ran is spent work, and counting it beside the
        cost of the rung that spent it turns "the stage took 5.5 s" into "8 s
        of level-1 FFT was computed and thrown away".  A rung superseded while
        still queued spent nothing and must not be counted, or the waste reads
        larger than the stage that contains it.
        """

        if evaluated is not None and evaluated[0]:
            # Consume the flag: one submission's evaluation is at most one
            # discard however many delivery callbacks it reaches (a superseded
            # reusable task can be told twice).  This bounds `discarded` by
            # `calls` structurally rather than by hoping the paths line up.
            evaluated[0] = False
            self.rung_timings.record_discarded(int(step.rung), int(step.level))
        self.effects.rung_dropped(intent, step)

    @staticmethod
    def _intent_step_matches_current(intent: RenderIntent, step: RungStep, current) -> bool:
        if current is None or intent.semantic_key != current.semantic_key:
            return False
        previous_source = intent.source_id_for_tile(int(step.tile_number))
        current_source = current.source_id_for_tile(int(step.tile_number))
        return previous_source == current_source

    def _on_rung_stale(
        self, intent: RenderIntent, step: RungStep, evaluated: list[bool] | None = None
    ) -> None:
        self._discard_rung(intent, step, evaluated)

    def _on_rung_error(
        self,
        intent: RenderIntent,
        step: RungStep,
        exc: BaseException,
        evaluated: list[bool] | None = None,
    ) -> None:
        self._discard_rung(intent, step, evaluated)
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
