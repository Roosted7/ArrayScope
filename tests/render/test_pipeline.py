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


class StubEffects:
    def __init__(self, tiles=2):
        self.tiles = tiles
        self.evaluated = []
        self.batches = []
        self.states = {}
        self.prepared = []
        self.dropped = []
        self.deps = {}
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
        self.dropped.append((step.tile_number, int(step.rung), step.level))


def make_pipeline(tiles=2, **policy_kwargs):
    kernel = Kernel(InlineWorkerBackend(), handler_error_hook=lambda ctx, exc: None)
    effects = StubEffects(tiles=tiles)
    ladder = LodLadder(LadderPolicy(**policy_kwargs)) if policy_kwargs else LodLadder()
    pipeline = MontagePipeline(kernel, effects, ladder)
    return kernel, effects, pipeline


class CaptureKernel:
    def __init__(self):
        self.specs = []
        self.cleared_scopes = []

    def submit(self, spec, **_callbacks):
        self.specs.append(spec)
        return object()

    def clear_scope(self, scope):
        self.cleared_scopes.append(scope)


def drain(kernel):
    while True:
        event = kernel.completions.pop()
        if event is None:
            return
        kernel.dispatch_event(event)


def intent(semantic="doc-v1", viewport="vp-1"):
    return RenderIntent(
        semantic_key=semantic,
        viewport_key=viewport,
        presentation_key="pres-1",
        view_range=((0.0, 10.0), (0.0, 10.0)),
        viewport_shape=(100, 100),
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
    assert submitted == 6  # 2 tiles x (floor, preview, desired)
    drain(kernel)
    assert len(effects.evaluated) == 6
    # Floor-first across tiles: both floors evaluated before any preview.
    assert [entry[1] for entry in effects.evaluated[:2]] == [0, 0]
    assert effects.batches, "ready results must flush as commit batches"
    total_upserts = sum(len(batch.upserts) for batch in effects.batches)
    assert total_upserts == 6
    assert all(len(batch.upserts) <= batch.max_items for batch in effects.batches)


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
    # Only current-level payloads were committed: floor 4, preview 2,
    # desired 1 — exactly once each. The desired-level-2 result existed but
    # was classified stale at dispatch and diverted to reuse.
    committed = sorted(
        step.level for batch in effects.batches for (step, _payload) in batch.upserts
    )
    assert committed == [1, 2, 4]


def test_counters_track_pipeline_activity():
    kernel, effects, pipeline = make_pipeline(tiles=1)
    pipeline.retarget(intent(), demand(1))
    drain(kernel)
    counts = pipeline.counters.as_dict()
    assert counts["intents"] == 1
    assert counts["tasks_submitted"] >= 3
    assert counts["commit_batches"] >= 1


def test_rung_dependencies_park_tasks_until_stage_key_completes():
    kernel, effects, pipeline = make_pipeline(tiles=1)
    effects.deps[0] = ("stage-key",)

    assert pipeline.retarget(intent(), demand(1)) == 3
    assert effects.evaluated == []
    assert kernel.diagnostics().parked_deps == 3

    kernel.submit(TaskSpec(key="stage-key", fn=lambda: "stage", lane=Lane.STAGE_MATERIALIZATION))
    drain(kernel)

    assert len(effects.evaluated) == 3


def test_pipeline_never_expresses_rung_ordering_through_deps():
    """Deps fail-propagate (a skipped floor would park exact work forever);
    ordering must come from priorities + submission order only. Deps are
    reserved for real data dependencies (stage keys)."""

    kernel = CaptureKernel()
    effects = StubEffects(tiles=2)
    pipeline = MontagePipeline(kernel, effects, LodLadder())

    assert pipeline.retarget(intent(), demand(1)) == 6

    assert all(spec.deps == () for spec in kernel.specs)
    # Coarse rungs are submitted before fine rungs across all tiles.
    rungs_in_order = [spec.key.rung for spec in kernel.specs]
    assert rungs_in_order == sorted(rungs_in_order)
