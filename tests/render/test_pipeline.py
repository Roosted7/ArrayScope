"""MontagePipeline scheduling skeleton against the real kernel.

Pins: retarget submits exactly the missing rung work, semantic changes clear
the scope, viewport changes supersede per-tile rung families, and commit
batches stay bounded. Effects are stubbed — evaluation/commit porting is
redesign R2 (see pipeline.py TODOs).
"""

from __future__ import annotations

from arrayscope.display.lod import LodDemand
from arrayscope.kernel import InlineWorkerBackend, Kernel
from arrayscope.render.ladder import LadderPolicy, LodLadder, TileLodState
from arrayscope.render.pipeline import MontagePipeline
from arrayscope.render.stages import RenderIntent


class StubEffects:
    def __init__(self, tiles=2):
        self.tiles = tiles
        self.evaluated = []
        self.batches = []
        self.states = {}

    def evaluate_rung(self, intent, step):
        def run(step=step):
            self.evaluated.append((step.tile_number, int(step.rung), step.level))
            return ("payload", step.tile_number, step.level)

        return run

    def apply_commit(self, batch):
        self.batches.append(batch)

    def tile_states(self, _intent, _demand):
        return tuple(
            self.states.get(number, TileLodState(tile_number=number))
            for number in range(self.tiles)
        )


def make_pipeline(tiles=2, **policy_kwargs):
    kernel = Kernel(InlineWorkerBackend(), handler_error_hook=lambda ctx, exc: None)
    effects = StubEffects(tiles=tiles)
    ladder = LodLadder(LadderPolicy(**policy_kwargs)) if policy_kwargs else LodLadder()
    pipeline = MontagePipeline(kernel, effects, ladder)
    return kernel, effects, pipeline


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


def test_viewport_change_supersedes_stale_rung_targets():
    kernel, effects, pipeline = make_pipeline(tiles=1)
    first = intent(viewport="vp-1")
    second = intent(viewport="vp-2")
    pipeline.retarget(first, demand(2))
    # Before the GUI drains, the viewport moves and demands a finer level.
    pipeline.retarget(second, demand(1))
    drain(kernel)
    lanes = kernel.diagnostics().lanes
    invalidated = sum(
        counters.get("superseded", 0) + counters.get("stale", 0) + counters.get("stale_reused", 0)
        for counters in lanes.values()
    )
    assert invalidated >= 2, "older viewport rung targets must not survive"
    # Only the current plan's payloads were committed: floor 4, preview 2,
    # desired 1 — exactly once each. The vp-1 floor/desired results existed
    # but were classified stale at dispatch and diverted to reuse.
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
