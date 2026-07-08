"""MontagePipeline scheduling skeleton against the real kernel.

Pins: retarget submits exactly the missing rung work, semantic changes clear
the scope, viewport changes supersede per-tile rung families, and commit
batches stay bounded. Effects are stubbed here so the tests isolate the
pipeline/kernel contract from the concrete montage backend bridge.
"""

from __future__ import annotations

from arrayscope.display.lod import LodDemand
from arrayscope.kernel import InlineWorkerBackend, Kernel, Lane, TaskSpec
from arrayscope.render.ladder import LadderPolicy, LodLadder, TileLodState
from arrayscope.render.pipeline import MontagePipeline
from arrayscope.render.stages import RenderIntent


class ManualBackend:
    workers = 4

    def attach(self, kernel) -> None:
        self.kernel = kernel

    def wake(self) -> None:
        pass

    def shutdown(self, timeout: float = 5.0) -> None:
        pass

    def run_next(self) -> bool:
        record = self.kernel._take_next(block=False)
        if record is None:
            return False
        self.kernel._execute(record)
        return True


class StubEffects:
    def __init__(self, tiles=2):
        self.tiles = tiles
        self.evaluated = []
        self.batches = []
        self.states = {}
        self.prepared = []
        self.dropped = []
        self.drop_intents = []
        self.deps = {}
        self.retained_native = set()
        self.last_intent = None
        self.last_demand = None

    def evaluate_rung(self, intent, step):
        self.last_intent = intent

        def run(_token=None, step=step):
            self.evaluated.append((step.tile_number, int(step.rung), step.level))
            return ("payload", step.tile_number, step.level)

        return run

    def apply_commit(self, batch):
        self.batches.append(batch)

    def tile_states(self, intent, demand):
        self.last_intent = intent
        self.last_demand = demand
        return tuple(
            self.states.get(number, TileLodState(tile_number=number))
            for number in range(self.tiles)
        )

    def prepare_rung(self, intent, step):
        # Mirror the real effects' in-flight/admitted dedupe: an identical
        # (tile, rung, level) is prepared at most once per target — that is
        # the guard that makes camera-only replans free.
        marker = (intent.semantic_key, step.tile_number, int(step.rung), step.level)
        if marker in self.prepared:
            return False
        self.prepared.append(marker)
        return True

    def rung_deps(self, intent, step):
        self.last_intent = intent
        return tuple(self.deps.get(step.tile_number, ()))

    def rung_dropped(self, intent, step):
        self.last_intent = intent
        self.drop_intents.append(intent)
        self.dropped.append((step.tile_number, int(step.rung), step.level))

    def retained_native_source_available(self, intent, step):
        return int(step.tile_number) in self.retained_native


def make_pipeline(tiles=2, **policy_kwargs):
    kernel = Kernel(InlineWorkerBackend(), handler_error_hook=lambda ctx, exc: None)
    effects = StubEffects(tiles=tiles)
    ladder = LodLadder(LadderPolicy(**policy_kwargs)) if policy_kwargs else LodLadder()
    pipeline = MontagePipeline(kernel, effects, ladder)
    return kernel, effects, pipeline


def make_manual_pipeline(tiles=2, **policy_kwargs):
    backend = ManualBackend()
    kernel = Kernel(backend, handler_error_hook=lambda ctx, exc: None)
    effects = StubEffects(tiles=tiles)
    ladder = LodLadder(LadderPolicy(**policy_kwargs)) if policy_kwargs else LodLadder()
    pipeline = MontagePipeline(kernel, effects, ladder)
    return kernel, backend, effects, pipeline


class CaptureKernel:
    def __init__(self):
        self.specs = []
        self.callbacks = []
        self.cleared_scopes = []
        self.superseded = []

    def submit(self, spec, **callbacks):
        self.specs.append(spec)
        self.callbacks.append(callbacks)
        return object()

    def clear_scope(self, scope):
        self.cleared_scopes.append(scope)

    def supersede(self, family, value):
        self.superseded.append((family, value))


def drain(kernel):
    while True:
        event = kernel.completions.pop()
        if event is None:
            return
        kernel.dispatch_event(event)


def intent(semantic="doc-v1", viewport="vp-1", *, interactive=False):
    return RenderIntent(
        semantic_key=semantic,
        viewport_key=viewport,
        presentation_key="pres-1",
        view_range=((0.0, 10.0), (0.0, 10.0)),
        viewport_shape=(100, 100),
        interactive=bool(interactive),
    )


def demand(level: int) -> LodDemand:
    return LodDemand(
        desired_level=level,
        desired_factor=2**level,
        desired_factor_xy=(2**level, 2**level),
        acceptable_levels=(max(0, level - 1), level, level + 1),
        source_texels_per_pixel_xy=(float(2**level), float(2**level)),
        reason="test",
    )


def test_retarget_submits_ladder_work_and_commits_batches():
    kernel, effects, pipeline = make_pipeline(tiles=2)
    submitted = pipeline.retarget(intent(), demand(1))
    assert submitted == 4  # 2 tiles x (floor, preview); desired waits for first pixels
    drain(kernel)
    assert len(effects.evaluated) == 4
    # Floor-first across tiles: both floors evaluated before any preview.
    assert [entry[1] for entry in effects.evaluated[:2]] == [0, 0]
    assert effects.batches, "ready results must flush as commit batches"
    total_upserts = sum(len(batch.upserts) for batch in effects.batches)
    assert total_upserts == 4
    assert all(len(batch.upserts) <= batch.max_items for batch in effects.batches)
    assert pipeline.counters.first_pixel_quality_deferred == 2


def test_converged_retarget_submits_nothing():
    kernel, effects, pipeline = make_pipeline(tiles=1)
    effects.states[0] = TileLodState(tile_number=0, presented_level=1, resident_levels=(1,))
    assert pipeline.retarget(intent(), demand(1)) == 0
    drain(kernel)
    assert effects.evaluated == []


def test_semantic_change_clears_previous_scope():
    kernel, effects, pipeline = make_pipeline(tiles=1)
    pipeline.retarget(intent(semantic="doc-v1"), demand(1))
    drain(kernel)
    evaluated_before = len(effects.evaluated)
    pipeline.retarget(intent(semantic="doc-v2"), demand(1))
    drain(kernel)
    # Old scope cleared, new work ran; nothing half-committed from doc-v1.
    assert len(effects.evaluated) > evaluated_before
    assert all(batch.semantic_key in ("doc-v1", "doc-v2") for batch in effects.batches)


def test_camera_only_retarget_never_invalidates_rung_work():
    """Core invariant: camera-only changes do not restart evaluation.

    Same demand, new viewport: the replan resubmits identical step keys and
    the kernel/effects guards must dedupe them — nothing superseded,
    nothing re-evaluated.
    """

    kernel, effects, pipeline = make_pipeline(tiles=1)
    pipeline.retarget(intent(viewport="vp-1"), demand(2))
    drain(kernel)
    evaluated_before = len(effects.evaluated)
    pipeline.retarget(intent(viewport="vp-2"), demand(2))
    drain(kernel)
    assert len(effects.evaluated) == evaluated_before
    lanes = kernel.diagnostics().lanes
    superseded = sum(counters.get("superseded", 0) for counters in lanes.values())
    assert superseded == 0


def test_demand_level_change_supersedes_stale_rung_targets():
    kernel, effects, pipeline = make_pipeline(tiles=1)
    effects.states[0] = TileLodState(tile_number=0, presented_level=4, resident_levels=(4,))
    pipeline.retarget(intent(viewport="vp-1"), demand(2))
    # Before the GUI drains, the demand moves to a finer level.
    pipeline.retarget(intent(viewport="vp-2"), demand(1))
    drain(kernel)
    lanes = kernel.diagnostics().lanes
    invalidated = sum(
        counters.get("superseded", 0) + counters.get("stale", 0) + counters.get("stale_reused", 0)
        for counters in lanes.values()
    )
    assert invalidated >= 1, "older level targets must not survive"
    # The old desired-level-2 target existed but was classified stale even
    # though the new desired-level-1 refinement is deferred behind preview.
    committed = sorted(
        step.level for batch in effects.batches for (step, _payload) in batch.upserts
    )
    assert committed == [2]
    assert (0, 2, 2) in effects.dropped


def test_interactive_native_demand_defers_cold_native_until_noninteractive_replan():
    kernel, effects, pipeline = make_pipeline(tiles=1)

    assert pipeline.retarget(intent(interactive=True), demand(0)) == 2
    drain(kernel)

    # Native demand plans FLOOR, PREVIEW, DESIRED(0); DESIRED waits until the
    # first-pixel rungs have had a chance to present.
    assert effects.evaluated == [(0, 0, 4), (0, 1, 2)]
    assert pipeline.counters.first_pixel_quality_deferred == 1

    effects.states[0] = TileLodState(tile_number=0, presented_level=2, resident_levels=(2,), presented_quality="preview")
    assert pipeline.retarget(intent(interactive=False), demand(0)) == 1
    drain(kernel)
    assert (0, 2, 0) in effects.evaluated


def test_interactive_opaque_desired_rung_defers_reduce_from_native_work():
    kernel, effects, pipeline = make_pipeline(tiles=1, reduced_input_available=False)

    assert pipeline.retarget(intent(interactive=True), demand(1)) == 0
    assert effects.evaluated == []
    assert pipeline.counters.interactive_native_deferred == 1

    assert pipeline.retarget(intent(interactive=False), demand(1)) == 1
    drain(kernel)
    assert effects.evaluated == [(0, 2, 1)]


def test_interactive_retained_native_source_is_correctness_work():
    kernel, effects, pipeline = make_pipeline(tiles=1, reduced_input_available=False)
    effects.retained_native.add(0)

    assert pipeline.retarget(intent(interactive=True), demand(1)) == 1
    drain(kernel)

    assert effects.evaluated == [(0, 2, 1)]
    assert pipeline.counters.interactive_native_deferred == 0


def test_zoom_out_over_presented_native_submits_no_display_demotions():
    kernel, effects, pipeline = make_pipeline(tiles=1)
    effects.states[0] = TileLodState(tile_number=0, presented_level=0, resident_levels=(0,))

    assert pipeline.retarget(intent(interactive=True), demand(6)) == 0
    assert pipeline.retarget(intent(interactive=False), demand(6)) == 0
    drain(kernel)

    assert effects.evaluated == []
    assert effects.batches == []


def test_preview_at_non_native_demand_level_satisfies_display_demand():
    kernel, effects, pipeline = make_pipeline(tiles=1)
    effects.states[0] = TileLodState(
        tile_number=0,
        presented_level=3,
        resident_levels=(3,),
        presented_quality="preview",
    )

    assert pipeline.retarget(intent(interactive=False), demand(3)) == 0
    drain(kernel)

    assert effects.evaluated == []


def test_stale_done_from_previous_semantic_is_dropped_not_committed():
    kernel = CaptureKernel()
    effects = StubEffects(tiles=1)
    pipeline = MontagePipeline(kernel, effects, LodLadder())

    pipeline.retarget(intent(semantic="window-1"), demand(1))
    old_done = kernel.callbacks[0]["on_done"]
    old_step = kernel.specs[0].key

    pipeline.retarget(intent(semantic="window-2"), demand(1))
    old_done(("payload", old_step.tile_number, old_step.level))

    assert effects.batches == []
    assert any(drop.semantic_key == "window-1" for drop in effects.drop_intents)
    assert (old_step.tile_number, old_step.rung, old_step.level) in effects.dropped


def test_none_completion_releases_prepared_rung_claim():
    kernel = CaptureKernel()
    effects = StubEffects(tiles=1)
    pipeline = MontagePipeline(kernel, effects, LodLadder())

    pipeline.retarget(intent(semantic="window-1"), demand(1))
    done = kernel.callbacks[0]["on_done"]
    step = kernel.specs[0].key

    done(None)

    assert effects.batches == []
    assert any(drop.semantic_key == "window-1" for drop in effects.drop_intents)
    assert (step.tile_number, step.rung, step.level) in effects.dropped


def test_dropped_queued_rung_releases_prepared_state():
    kernel, _backend, effects, pipeline = make_manual_pipeline(tiles=1)
    effects.states[0] = TileLodState(tile_number=0, presented_level=4, resident_levels=(4,))
    assert pipeline.retarget(intent(viewport="vp-1"), demand(2)) == 1

    # Supersede the queued desired-level rung before it can run.  The effects
    # already prepared lifecycle ownership; the drop callback must release it.
    assert pipeline.retarget(intent(viewport="vp-2"), demand(1)) == 1
    drain(kernel)

    assert (0, 2, 2) in effects.dropped
    assert any(drop.viewport_key == "vp-1" for drop in effects.drop_intents)


def test_counters_track_pipeline_activity():
    kernel, effects, pipeline = make_pipeline(tiles=1)
    pipeline.retarget(intent(), demand(1))
    drain(kernel)
    counts = pipeline.counters.as_dict()
    assert counts["intents"] == 1
    assert counts["tasks_submitted"] >= 2
    assert counts["commit_batches"] >= 1


def test_rung_dependencies_park_tasks_until_stage_key_completes():
    kernel, effects, pipeline = make_pipeline(tiles=1)
    effects.deps[0] = ("stage-key",)

    assert pipeline.retarget(intent(), demand(1)) == 2
    assert effects.evaluated == []
    assert kernel.diagnostics().parked_deps == 2

    kernel.submit(TaskSpec(key="stage-key", fn=lambda: "stage", lane=Lane.STAGE_MATERIALIZATION))
    drain(kernel)

    assert len(effects.evaluated) == 2


def test_pipeline_never_expresses_rung_ordering_through_deps():
    """Deps fail-propagate (a skipped floor would park exact work forever);
    ordering must come from priorities + submission order only. Deps are
    reserved for real data dependencies (stage keys)."""

    kernel = CaptureKernel()
    effects = StubEffects(tiles=2)
    pipeline = MontagePipeline(kernel, effects, LodLadder())

    assert pipeline.retarget(intent(), demand(1)) == 4

    assert all(spec.deps == () for spec in kernel.specs)
    # Coarse rungs are submitted before fine rungs across all tiles.
    rungs_in_order = [spec.key.rung for spec in kernel.specs]
    assert rungs_in_order == sorted(rungs_in_order)
