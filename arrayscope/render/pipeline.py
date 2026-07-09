"""MontagePipeline: ladder plans → kernel tasks → commit batches.

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

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from arrayscope.kernel import Kernel, Supersession, TaskSpec
from arrayscope.render.ladder import LodLadder, Rung, RungStep, TileLodState
from arrayscope.render.stages import CommitBatch, LodAdmissionScope, PipelineCounters, RenderIntent


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
        """Claim lifecycle state before a rung task is submitted."""
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
    rung: int
    level: int


class MontagePipeline:
    """Schedules montage quality progression on the kernel.

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

    SCOPE = "montage"

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
        self._commit_max_items = max(1, int(commit_max_items))
        self._current_intent: RenderIntent | None = None
        self._ready_upserts: list = []

    # ----------------------------------------------------------- lifecycle

    def retarget(self, intent: RenderIntent, demand, scope: LodAdmissionScope) -> int:
        """Adopt a new intent; schedule exactly the missing rung steps.

        Returns the number of submitted kernel tasks. Safe to call on every
        interaction event: convergence means zero submissions, and stale
        work is superseded by family, never hunted down by ad-hoc flags.
        """

        previous = self._current_intent
        self._current_intent = intent
        self.counters.intents += 1
        if previous is not None and previous.semantic_key != intent.semantic_key:
            # Semantic change: everything under the old target is stale.
            self.kernel.clear_scope(self._scope(previous.semantic_key))
            for queued_intent, step, _payload in tuple(self._ready_upserts):
                if queued_intent.semantic_key != intent.semantic_key:
                    self.effects.rung_dropped(queued_intent, step)
            self._ready_upserts.clear()

        states = self.effects.tile_states(intent, demand, scope)
        steps = self.ladder.plan(states, demand)
        self.counters.ladder_plans += 1
        submitted = 0
        first_pixel_tiles = {
            int(step.tile_number)
            for step in steps
            if step.rung in (Rung.FLOOR, Rung.PREVIEW)
        }
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
            if self._defer_quality_behind_first_pixel(step, first_pixel_tiles):
                self._supersede_deferred_step(intent, step)
                self.counters.first_pixel_quality_deferred += 1
                continue
            if (
                self._defer_native_quality_during_interaction(intent, step, presented_preview_tiles)
                and not self.effects.retained_native_source_available(intent, step)
            ):
                self.counters.interactive_native_deferred += 1
                continue
            if self._submit_step(intent, step, step_key=self._rung_key(intent, step)):
                submitted += 1
        self._flush_ready()
        return submitted

    def close(self) -> None:
        if self._current_intent is not None:
            self.kernel.clear_scope(self._scope(self._current_intent.semantic_key))

    # ------------------------------------------------------------ internal

    def _scope(self, semantic_key: object) -> str:
        return f"{self.SCOPE}:{semantic_key!r}"

    def _rung_key(self, intent: RenderIntent, step: RungStep) -> _RungKey:
        return _RungKey(
            semantic_key=intent.semantic_key,
            tile_number=int(step.tile_number),
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

    def _defer_quality_behind_first_pixel(
        self,
        step: RungStep,
        first_pixel_tiles: set[int],
    ) -> bool:
        """Do not race exact/native quality against this plan's first pixels."""

        if int(step.tile_number) not in first_pixel_tiles:
            return False
        if step.rung == Rung.EXACT:
            return True
        if step.rung != Rung.DESIRED:
            return False
        return True

    def _submit_step(
        self,
        intent: RenderIntent,
        step: RungStep,
        *,
        step_key: _RungKey,
    ) -> bool:
        if not self.effects.prepare_rung(intent, step):
            return False
        spec = TaskSpec(
            key=step_key,
            fn=self.effects.evaluate_rung(intent, step),
            lane=step.lane,
            priority=step.priority,
            scope=self._scope(intent.semantic_key),
            deps=self.effects.rung_deps(intent, step),
            # Latest-only per tile+rung: a *demand/level* change replaces the
            # target; camera-only changes keep the same value and therefore
            # never invalidate running or queued materialization. Reusable
            # results may still land in caches when superseded.
            supersession=Supersession(
                family=("rung", intent.semantic_key, step.tile_number, int(step.rung)),
                value=(int(step.level),),
            ),
            reusable=True,
            pass_token=True,
        )
        handle = self.kernel.submit(
            spec,
            on_done=lambda payload, intent=intent, step=step: self._on_rung_done(intent, step, payload),
            on_error=lambda exc, intent=intent, step=step: self._on_rung_error(intent, step, exc),
            on_stale=lambda intent=intent, step=step: self._on_rung_stale(intent, step),
            on_reuse=lambda payload, intent=intent, step=step: self._on_rung_reusable(intent, step, payload),
        )
        if handle is not None:
            self.counters.tasks_submitted += 1
            return True
        self.effects.rung_dropped(intent, step)
        return False

    def _supersede_deferred_step(self, intent: RenderIntent, step: RungStep) -> None:
        """Invalidate older work for a rung we intentionally delay."""

        self.kernel.supersede(
            ("rung", intent.semantic_key, step.tile_number, int(step.rung)),
            ("deferred", int(step.level), int(self.counters.intents)),
        )

    # Handlers run on the GUI thread (kernel bridge drain).

    def _on_rung_done(self, intent: RenderIntent, step: RungStep, payload) -> None:
        current = self._current_intent
        if current is None or intent.semantic_key != current.semantic_key:
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
            if queued_intent.semantic_key == intent.semantic_key:
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


__all__ = ["MontagePipeline", "PipelineEffects"]
