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
    waiting_tiles: dict[object, list[object]] = field(default_factory=dict)
    active_requests: set[object] = field(default_factory=set)
    attached_requests: set[object] = field(default_factory=set)
    values: dict[object, object] = field(default_factory=dict)
    tile_stage_keys: dict[int, object] = field(default_factory=dict)
    tile_stage_plans: dict[int, object] = field(default_factory=dict)
    tile_stage_candidates: dict[int, object] = field(default_factory=dict)
    lead_warmups: dict[int, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.waiting_tiles = {
            key: list(value or ())
            for key, value in dict(self.waiting_tiles or {}).items()
        }
        self.active_requests = set(self.active_requests or ())
        self.attached_requests = set(self.attached_requests or ())
        self.values = dict(self.values or {})
        self.tile_stage_keys = {int(key): value for key, value in dict(self.tile_stage_keys or {}).items()}
        self.tile_stage_plans = {int(key): value for key, value in dict(self.tile_stage_plans or {}).items()}
        self.tile_stage_candidates = {int(key): value for key, value in dict(self.tile_stage_candidates or {}).items()}
        self.lead_warmups = {int(key): value for key, value in dict(self.lead_warmups or {}).items()}

    def merge_plan(self, plan: dict) -> None:
        self.tile_stage_keys.update({int(key): value for key, value in dict(plan.get("tile_stage_keys", {})).items()})
        self.tile_stage_plans.update({int(key): value for key, value in dict(plan.get("tile_stage_plans", {})).items()})
        self.tile_stage_candidates.update({int(key): value for key, value in dict(plan.get("tile_stage_candidates", {})).items()})
        for key, waiting in dict(plan.get("stage_waiting_tiles", {})).items():
            existing = self.waiting_tiles.setdefault(key, [])
            existing_numbers = {_tile_index(tile) for tile in existing}
            existing.extend(
                tile
                for tile in tuple(waiting or ())
                if _tile_index(tile) not in existing_numbers
            )
        self.attached_requests.update(plan.get("attached_stage_keys", ()) or ())
        self.values.update(dict(plan.get("stage_values", {}) or {}))
        self.lead_warmups.update({int(key): value for key, value in dict(plan.get("lead_stage_warmups", {})).items()})

    def activate_value(self, key, value, *, max_items: int | None = None) -> StageActivationBatch:
        self.active_requests.discard(key)
        self.attached_requests.discard(key)
        self.values[key] = value
        waiting = self.waiting_tiles.get(key)
        if not waiting:
            self.waiting_tiles.pop(key, None)
            return StageActivationBatch(key=key, tiles=(), complete=True)
        batch, complete = _take_batch(waiting, max_items=max_items)
        if complete:
            self.waiting_tiles.pop(key, None)
        else:
            self.waiting_tiles[key] = waiting
        return StageActivationBatch(key=key, tiles=batch, complete=complete)

    def release_missing(self, key, *, max_items: int | None = None) -> StageReleaseBatch:
        self.active_requests.discard(key)
        self.attached_requests.discard(key)
        waiting = self.waiting_tiles.get(key)
        if not waiting:
            self.waiting_tiles.pop(key, None)
            return StageReleaseBatch(key=key, tiles=(), complete=True)
        batch, complete = _take_batch(waiting, max_items=max_items)
        for tile in batch:
            self.tile_stage_keys.pop(_tile_index(tile), None)
        if complete:
            self.waiting_tiles.pop(key, None)
        else:
            self.waiting_tiles[key] = waiting
        return StageReleaseBatch(key=key, tiles=batch, complete=complete)

    def fail(self, key) -> tuple[object, ...]:
        self.active_requests.discard(key)
        self.attached_requests.discard(key)
        waiting = tuple(self.waiting_tiles.pop(key, ()) or ())
        for tile in waiting:
            self.tile_stage_keys.pop(_tile_index(tile), None)
        return waiting

    def has_waiting(self) -> bool:
        return any(bool(value) for value in self.waiting_tiles.values())


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
