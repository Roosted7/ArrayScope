"""Qt-free fan-in state for reusable stage materialization."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StageActivationBatch:
    key: object
    tiles: tuple[object, ...]
    complete: bool


@dataclass(frozen=True)
class StageReleaseBatch:
    key: object
    tiles: tuple[object, ...]
    complete: bool


@dataclass
class StageFanInState:
    active_requests: set[object] = field(default_factory=set)
    attached_requests: set[object] = field(default_factory=set)
    values: dict[object, object] = field(default_factory=dict)
    tile_stage_keys: dict[int, object] = field(default_factory=dict)
    tile_stage_plans: dict[int, object] = field(default_factory=dict)
    tile_stage_candidates: dict[int, object] = field(default_factory=dict)
    lead_warmups: dict[int, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.active_requests = set(self.active_requests or ())
        self.attached_requests = set(self.attached_requests or ())
        self.values = dict(self.values or {})
        self.tile_stage_keys = {
            int(key): value for key, value in dict(self.tile_stage_keys or {}).items()
        }
        self.tile_stage_plans = {
            int(key): value for key, value in dict(self.tile_stage_plans or {}).items()
        }
        self.tile_stage_candidates = {
            int(key): value for key, value in dict(self.tile_stage_candidates or {}).items()
        }
        self.lead_warmups = {
            int(key): value for key, value in dict(self.lead_warmups or {}).items()
        }

    def merge_plan(self, plan: dict) -> None:
        self.tile_stage_keys.update(
            {int(key): value for key, value in dict(plan.get("tile_stage_keys", {})).items()}
        )
        self.tile_stage_plans.update(
            {int(key): value for key, value in dict(plan.get("tile_stage_plans", {})).items()}
        )
        self.tile_stage_candidates.update(
            {int(key): value for key, value in dict(plan.get("tile_stage_candidates", {})).items()}
        )
        self.attached_requests.update(plan.get("attached_stage_keys", ()) or ())
        self.values.update(dict(plan.get("stage_values", {}) or {}))
        self.lead_warmups.update(
            {int(key): value for key, value in dict(plan.get("lead_stage_warmups", {})).items()}
        )

    def activate_value(self, key, value, *, max_items: int | None = None) -> StageActivationBatch:
        self.active_requests.discard(key)
        self.attached_requests.discard(key)
        self.values[key] = value
        activated = tuple(
            tile for tile, stage_key in tuple(self.tile_stage_keys.items()) if stage_key == key
        )
        for tile in activated:
            self.tile_stage_keys.pop(_tile_index(tile), None)
        return StageActivationBatch(key=key, tiles=activated, complete=True)

    def release_missing(self, key, *, max_items: int | None = None) -> StageReleaseBatch:
        self.active_requests.discard(key)
        self.attached_requests.discard(key)
        released = tuple(
            tile for tile, stage_key in tuple(self.tile_stage_keys.items()) if stage_key == key
        )
        for tile in released:
            self.tile_stage_keys.pop(_tile_index(tile), None)
        return StageReleaseBatch(key=key, tiles=released, complete=True)

    def fail(self, key) -> tuple[object, ...]:
        self.active_requests.discard(key)
        self.attached_requests.discard(key)
        failed_tiles = tuple(
            tile for tile, stage_key in tuple(self.tile_stage_keys.items()) if stage_key == key
        )
        for tile in failed_tiles:
            self.tile_stage_keys.pop(_tile_index(tile), None)
        return failed_tiles

    def detach_unbound_requests(self) -> None:
        bound = set(self.tile_stage_keys.values())
        for key in tuple(self.attached_requests):
            if key not in bound:
                self.attached_requests.discard(key)

    def has_waiting(self) -> bool:
        return bool(self.tile_stage_keys)


def _take_batch(waiting, *, max_items: int | None) -> tuple[tuple[object, ...], bool]:
    limit = len(waiting) if max_items is None else max(0, int(max_items))
    if isinstance(waiting, list):
        batch = tuple(waiting[:limit])
        del waiting[:limit]
        return batch, not waiting
    # Priority-ordered queues (e.g. MontageTilePriorityQueue) release the
    # highest-priority tiles first, so budget-capped activation batches follow
    # the viewport/focus order instead of the plan's row-major order.
    batch = []
    while waiting and len(batch) < limit:
        tile = waiting.pop()
        if tile is None:
            break
        batch.append(tile)
    return tuple(batch), not waiting


def _tile_index(tile_or_index) -> int:
    try:
        return int(tile_or_index.montage_index)
    except AttributeError:
        return int(tile_or_index)
