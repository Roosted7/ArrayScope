"""The hot-cached stage-plan match loop yields to the async planner.

`hot_cached_stage_fan_in_plan`'s per-tile fallback is O(missing_tiles x
cache_entries) region matching on the GUI thread.  Unbounded, it owned the
event loop for seconds on a 272-tile cold entry (rank-1 in
docs/redesign/gui-callback-owners-2026-07-31.md).  The loop now runs under the
window's GUI-callback budget: when the budget expires it returns the deferred
placeholder — the shape every caller already pairs with
`submit_deferred_stage_fan_in_plan` — and reports what it spent.
"""

import numpy as np

from arrayscope.core.gui_callback_budget import GuiCallbackBudget
from arrayscope.core.view_state import ViewState
from arrayscope.display.montage import make_montage_plan
from arrayscope.operations.evaluator import stage_document_key
from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.operations.regions import AxisRegion, AxisRegionKind, RegionSpec
from arrayscope.window.frame_effects import hot_cached_stage_fan_in_plan


class _StageCache:
    def __init__(self, items):
        self._items = tuple(items)

    def resident_items(self):
        return list(self._items)


class _Entry:
    def __init__(self, *, region, stage_index=1, nbytes=1024):
        self.region = region
        self.stage_index = stage_index
        self.nbytes = nbytes
        self.prefetch_only = False
        self.visible_reuse = True


class _EntryKey:
    def __init__(self, *, document_key, shape):
        self.document_key = document_key
        self.shape = tuple(int(size) for size in shape)
        self.operation_prefix = ()


class _Win:
    def __init__(self, cache):
        import types

        self.operation_evaluator = types.SimpleNamespace(stage_cache=cache)
        self._viewport_interaction_active = True


class _Renderer:
    def __init__(self, cache, *, budget_ms):
        self.win = _Win(cache)
        self._budget_ms = float(budget_ms)
        self.recorded_observations = []

    def _montage_callback_budget(
        self,
        channel,
        *,
        interactive,
        work_class,
        item_cap=None,
        byte_cap=None,
        target_ms=None,
    ):
        budget = GuiCallbackBudget(
            channel=channel,
            work_class=work_class,
            item_cap=1 if item_cap is None else int(item_cap),
        )
        budget.target_ms = self._budget_ms
        return budget

    def _record_gui_budget(self, budget):
        self.recorded_observations.append(budget.observation())


def _setup(*, budget_ms):
    data = np.arange(4 * 4 * 8, dtype=float).reshape(4, 4, 8)
    indices = tuple(range(6))
    state = (
        ViewState.from_shape(data.shape)
        .with_image_axes(0, 1)
        .with_montage_axis(2, columns=3, indices=indices, text=":")
    )
    plan = make_montage_plan(state, axis=2, indices=indices, tile_shape=(4, 4), columns=3, gap=1)
    document = ArrayDocument(data)
    # A partial-region entry: shape-matched so the per-tile match loop (not the
    # whole-volume fast path) is what answers, and containing every planned
    # montage index so an unbudgeted scan would match every tile.
    region = RegionSpec(
        (
            AxisRegion(AxisRegionKind.ALL),
            AxisRegion(AxisRegionKind.ALL),
            AxisRegion(AxisRegionKind.INDICES, indices),
        )
    )
    key = _EntryKey(document_key=stage_document_key(document), shape=data.shape)
    cache = _StageCache([(key, _Entry(region=region))])
    renderer = _Renderer(cache, budget_ms=budget_ms)
    return renderer, document, tuple(plan.tiles), key


def test_generous_budget_attaches_resident_stage_values():
    renderer, document, tiles, key = _setup(budget_ms=1e9)

    plan = hot_cached_stage_fan_in_plan(renderer, document, tiles)

    assert plan["retained_stage_decision"] == "cached-hot-stage"
    assert set(plan["tile_stage_keys"]) == {int(tile.montage_index) for tile in tiles}
    assert key in plan["stage_values"]
    assert renderer.recorded_observations
    observation = renderer.recorded_observations[-1]
    assert observation.channel == "montage_stage_plan"
    assert observation.processed_items == len(tiles)


def test_expired_budget_defers_to_the_async_planner():
    renderer, document, tiles, _key = _setup(budget_ms=0.0)

    plan = hot_cached_stage_fan_in_plan(renderer, document, tiles)

    # The deferred placeholder is exactly what every caller pairs with
    # submit_deferred_stage_fan_in_plan: no waiting tiles, no stage values.
    assert plan["retained_stage_decision"] == "hot-stage-budget-deferred"
    assert plan["tile_stage_keys"] == {}
    assert plan["stage_values"] == {}
    assert plan["waiting_indices"] == set()
    assert renderer._hot_stage_budget_deferrals == 1
    observation = renderer.recorded_observations[-1]
    assert observation.channel == "montage_stage_plan"
    # Guaranteed progress: the first tile is always processed before the
    # deadline is consulted, so a zero budget still scans one tile.
    assert observation.processed_items == 1


def test_expired_budget_keeps_the_match_cache_warm_for_the_next_call():
    renderer, document, tiles, _key = _setup(budget_ms=0.0)

    hot_cached_stage_fan_in_plan(renderer, document, tiles)
    first_misses = renderer._hot_stage_match_cache_misses

    renderer._budget_ms = 1e9
    plan = hot_cached_stage_fan_in_plan(renderer, document, tiles)

    assert plan["retained_stage_decision"] == "cached-hot-stage"
    # The bailed slice's matches were remembered: the follow-up call resumes
    # ahead instead of rescanning tile one.
    assert renderer._hot_stage_match_cache_hits >= 1
    assert renderer._hot_stage_match_cache_misses == first_misses + len(tiles) - 1
