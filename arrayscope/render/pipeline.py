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
bridge's fallback is the only safety net).

STATUS (redesign R2): the scheduling skeleton below is real and unit-tested
against the kernel; the evaluation/commit effect implementations are
integration points that R2 fills in by porting logic OUT of
`window/frame_renderer.py` (see docs/redesign/frame-renderer-map.md for the
method-by-method destination table).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from arrayscope.kernel import Kernel, Supersession, TaskSpec
from arrayscope.render.ladder import LodLadder, RungStep, TileLodState
from arrayscope.render.stages import CommitBatch, PipelineCounters, RenderIntent


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

        TODO(redesign R2): port `_commit_montage_session_tile_layer` /
        `_commit_montage_tile_delta_direct` batching into CommitBatch
        consumption; acknowledgement flows back via TileLifecycle events,
        not return values.
        """
        ...

    def tile_states(self, intent: RenderIntent, demand) -> tuple[TileLodState, ...]:
        """Snapshot per-tile lod state from TileLifecycle claims.

        TODO(redesign R2): derive from `TileLifecycle` records +
        `PyramidCache` residency; keep it a pure read.
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


@dataclass(frozen=True)
class _RungKey:
    """Kernel task key for one tile rung under one semantic target."""

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

    def retarget(self, intent: RenderIntent, demand) -> int:
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
            self._ready_upserts.clear()

        states = self.effects.tile_states(intent, demand)
        steps = self.ladder.plan(states, demand)
        self.counters.ladder_plans += 1
        submitted = 0
        for step in steps:
            if self._submit_step(intent, step):
                submitted += 1
        self._flush_ready()
        return submitted

    def close(self) -> None:
        if self._current_intent is not None:
            self.kernel.clear_scope(self._scope(self._current_intent.semantic_key))

    # ------------------------------------------------------------ internal

    def _scope(self, semantic_key: object) -> str:
        return f"{self.SCOPE}:{semantic_key!r}"

    def _submit_step(self, intent: RenderIntent, step: RungStep) -> bool:
        if not self.effects.prepare_rung(intent, step):
            return False
        key = _RungKey(
            semantic_key=intent.semantic_key,
            tile_number=step.tile_number,
            rung=int(step.rung),
            level=int(step.level),
        )
        spec = TaskSpec(
            key=key,
            fn=self.effects.evaluate_rung(intent, step),
            lane=step.lane,
            priority=step.priority,
            scope=self._scope(intent.semantic_key),
            deps=self.effects.rung_deps(intent, step),
            # Latest-only per tile+rung: a viewport/demand change replaces
            # the level target; reusable results may still land in caches.
            supersession=Supersession(
                family=("rung", intent.semantic_key, step.tile_number, int(step.rung)),
                value=(intent.viewport_key, int(step.level)),
            ),
            reusable=True,
            pass_token=True,
        )
        handle = self.kernel.submit(
            spec,
            on_done=lambda payload, step=step: self._on_rung_done(step, payload),
            on_error=lambda exc, step=step: self._on_rung_error(step, exc),
            on_reuse=lambda payload, step=step: self._on_rung_reusable(step, payload),
        )
        if handle is not None:
            self.counters.tasks_submitted += 1
            return True
        self.effects.rung_dropped(intent, step)
        return False

    # Handlers run on the GUI thread (kernel bridge drain).

    def _on_rung_done(self, step: RungStep, payload) -> None:
        self._ready_upserts.append((step, payload))
        # TODO(redesign R2): feed TileLifecycle `level_materialized` /
        # residency-claim events here (single owner), then flush.
        self._flush_ready()

    def _on_rung_reusable(self, step: RungStep, payload) -> None:
        # Stale-but-reusable: cache only, never commit.
        # TODO(redesign R2): store into PyramidCache/RetainedTiledPayloadStore.
        if self._current_intent is not None:
            self.effects.rung_dropped(self._current_intent, step)

    def _on_rung_error(self, step: RungStep, exc: BaseException) -> None:
        # TODO(redesign R2): route through app error handling + lifecycle
        # `level_failed` so the machine can re-plan; silence is forbidden.
        if self._current_intent is not None:
            self.effects.rung_dropped(self._current_intent, step)
        raise exc

    def _flush_ready(self) -> None:
        intent = self._current_intent
        if intent is None or not self._ready_upserts:
            return
        batch_items = self._ready_upserts[: self._commit_max_items]
        del self._ready_upserts[: len(batch_items)]
        batch = CommitBatch(
            semantic_key=intent.semantic_key,
            presentation_key=intent.presentation_key,
            upserts=tuple(batch_items),
            max_items=self._commit_max_items,
        )
        while not batch.empty:
            self.counters.commit_batches += 1
            self.effects.apply_commit(batch)
            if not self._ready_upserts:
                break
            batch_items = self._ready_upserts[: self._commit_max_items]
            del self._ready_upserts[: len(batch_items)]
            batch = CommitBatch(
                semantic_key=intent.semantic_key,
                presentation_key=intent.presentation_key,
                upserts=tuple(batch_items),
                max_items=self._commit_max_items,
            )


__all__ = ["MontagePipeline", "PipelineEffects"]
