"""Qt-free dynamic priority queues for montage tile scheduling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from heapq import heappop, heappush
from itertools import count
from math import isfinite


class TilePriorityBand(IntEnum):
    """Coarse scheduling bands for pending montage work."""

    VISIBLE = 0
    NEAR = 1
    WAITING = 2


@dataclass(frozen=True)
class TilePriorityContext:
    """Transient viewport/hover state used only for scheduling metadata."""

    view_range: object = None
    focus: tuple[float, float] | None = None
    visible_tiles: frozenset[int] = field(default_factory=frozenset)
    near_tiles: frozenset[int] = field(default_factory=frozenset)
    priority_tiles: tuple[int, ...] = ()

    @classmethod
    def from_tiles(
        cls,
        *,
        view_range=None,
        focus=None,
        visible_tiles=(),
        near_tiles=(),
        priority_tiles=(),
    ) -> "TilePriorityContext":
        return cls(
            view_range=view_range,
            focus=_normalize_focus(focus),
            visible_tiles=frozenset(int(tile) for tile in visible_tiles),
            near_tiles=frozenset(int(tile) for tile in near_tiles),
            priority_tiles=tuple(dict.fromkeys(int(tile) for tile in priority_tiles)),
        )


class MontageTilePriorityQueue:
    """Mutable indexed queue with lazy priority updates.

    The queue deliberately separates cheap event callbacks from scheduling
    updates.  Mouse movement can request a retarget, while the timer-driven
    scheduler updates a bounded number of heap entries and lets stale heap
    entries fall out lazily.
    """

    def __init__(
        self,
        tiles=(),
        *,
        context: TilePriorityContext | None = None,
        aging_after: int = 8,
    ) -> None:
        self._tiles: dict[int, object] = {}
        self._versions: dict[int, int] = {}
        self._sequence: dict[int, int] = {}
        self._heap: list[tuple[object, ...]] = []
        self._serial = count()
        self._retarget_order: deque[int] = deque()
        self.context = context or TilePriorityContext()
        self.aging_after = max(1, int(aging_after))
        self._preferred_pops = 0
        self.retargeted_last = 0
        self.stale_entries_discarded = 0
        self.fairness_pops = 0
        for tile in tuple(tiles or ()):
            self.append(tile)

    def __bool__(self) -> bool:
        return bool(self._tiles)

    def __len__(self) -> int:
        return len(self._tiles)

    def __contains__(self, tile) -> bool:
        return _tile_index(tile) in self._tiles

    def __iter__(self):
        yield from self.insertion_tiles()

    def clear(self) -> None:
        self._tiles.clear()
        self._versions.clear()
        self._sequence.clear()
        self._heap.clear()
        self._retarget_order.clear()
        self._preferred_pops = 0

    def append(self, tile) -> None:
        index = _tile_index(tile)
        if index in self._tiles:
            return
        self._tiles[index] = tile
        self._versions[index] = 0
        self._sequence[index] = next(self._serial)
        self._retarget_order.append(index)
        self._push(index)

    def extend(self, tiles) -> None:
        for tile in tuple(tiles or ()):
            self.append(tile)

    def discard(self, tile_or_index) -> bool:
        index = _tile_index(tile_or_index)
        if index not in self._tiles:
            return False
        self._tiles.pop(index, None)
        self._versions.pop(index, None)
        self._sequence.pop(index, None)
        return True

    def prune(self, keep: set[int] | frozenset[int]) -> int:
        keep = {int(index) for index in keep}
        removed = 0
        for index in tuple(self._tiles):
            if int(index) not in keep:
                self.discard(index)
                removed += 1
        return int(removed)

    def set_context(self, context: TilePriorityContext, *, max_items: int | None = None) -> int:
        self.context = context
        limit = len(self._tiles) if max_items is None else max(0, int(max_items))
        processed = 0
        for index in tuple(context.priority_tiles):
            if processed >= limit:
                break
            if int(index) not in self._tiles:
                continue
            self._push(int(index))
            processed += 1
        if processed < limit:
            processed += self.retarget(max_items=limit - processed)
        self.retargeted_last = int(processed)
        return int(processed)

    def retarget(self, *, max_items: int | None = None) -> int:
        if not self._tiles:
            self.retargeted_last = 0
            return 0
        limit = len(self._tiles) if max_items is None else max(0, int(max_items))
        processed = 0
        while processed < limit and self._retarget_order:
            index = self._retarget_order.popleft()
            if index not in self._tiles:
                continue
            self._retarget_order.append(index)
            self._push(index)
            processed += 1
        self.retargeted_last = int(processed)
        return int(processed)

    def pop(self, position=None):
        self.last_pop_position = position
        if not self._tiles:
            return None
        fair = self._fair_tile()
        if fair is not None:
            return fair
        while self._heap:
            key = heappop(self._heap)
            index = int(key[-2])
            version = int(key[-1])
            if index not in self._tiles or version != self._versions.get(index):
                self.stale_entries_discarded += 1
                continue
            tile = self._tiles.pop(index)
            self._versions.pop(index, None)
            self._sequence.pop(index, None)
            self._preferred_pops += 1
            return tile
        # Heap entries can all be stale after heavy pruning. Rebuild lazily.
        for index in tuple(self._tiles):
            self._push(index)
        if self._heap:
            return self.pop()
        return None

    def ordered_tiles(self) -> tuple[object, ...]:
        return tuple(
            self._tiles[index]
            for index in sorted(
                self._tiles,
                key=lambda index: self._priority_key(index),
            )
        )

    def insertion_tiles(self) -> tuple[object, ...]:
        return tuple(
            self._tiles[index]
            for index in sorted(
                self._tiles,
                key=lambda index: (int(self._sequence.get(int(index), 0)), int(index)),
            )
        )

    def _push(self, index: int) -> None:
        if index not in self._tiles:
            return
        version = int(self._versions.get(index, 0)) + 1
        self._versions[index] = version
        heappush(self._heap, (*self._priority_key(index), int(index), version))

    def _priority_key(self, index: int) -> tuple[object, ...]:
        tile = self._tiles[int(index)]
        band = _band_for_index(int(index), self.context)
        return (
            int(band),
            _distance_score(tile, self.context),
            int(self._sequence.get(int(index), 0)),
            int(index),
        )

    def _fair_tile(self):
        if self._preferred_pops < self.aging_after or len(self._tiles) <= 1:
            return None
        candidates = (
            index
            for index in self._tiles
            if _band_for_index(int(index), self.context) in {TilePriorityBand.VISIBLE, TilePriorityBand.NEAR}
        )
        try:
            index = min(candidates, key=lambda value: (self._sequence.get(int(value), 0), int(value)))
        except ValueError:
            return None
        tile = self._tiles.pop(int(index))
        self._versions.pop(int(index), None)
        self._sequence.pop(int(index), None)
        self._preferred_pops = 0
        self.fairness_pops += 1
        return tile


def tile_numbers(tiles) -> tuple[int, ...]:
    return tuple(int(_tile_index(tile)) for tile in tuple(tiles or ()))


def _tile_index(tile_or_index) -> int:
    if hasattr(tile_or_index, "montage_index"):
        return int(tile_or_index.montage_index)
    return int(tile_or_index)


def _band_for_index(index: int, context: TilePriorityContext) -> TilePriorityBand:
    if int(index) in context.visible_tiles:
        return TilePriorityBand.VISIBLE
    if int(index) in context.near_tiles:
        return TilePriorityBand.NEAR
    return TilePriorityBand.WAITING


def _distance_score(tile, context: TilePriorityContext) -> float:
    ranges = _normalize_view_range(context.view_range)
    if ranges is None:
        return 0.0
    (x0, x1), (y0, y1) = ranges
    span_x = max(1.0, abs(float(x1) - float(x0)))
    span_y = max(1.0, abs(float(y1) - float(y0)))
    focus = context.focus
    if focus is None:
        focus_x = (float(x0) + float(x1)) * 0.5
        focus_y = (float(y0) + float(y1)) * 0.5
    else:
        focus_x, focus_y = focus
    center_x = float(getattr(tile, "x0", 0.0)) + float(getattr(tile, "width", 1.0)) * 0.5
    center_y = float(getattr(tile, "y0", 0.0)) + float(getattr(tile, "height", 1.0)) * 0.5
    dx = (center_x - focus_x) / span_x
    dy = (center_y - focus_y) / span_y
    return float(dx * dx + dy * dy)


def _normalize_focus(focus) -> tuple[float, float] | None:
    if focus is None:
        return None
    try:
        x = float(focus[0])
        y = float(focus[1])
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if not isfinite(x) or not isfinite(y):
        return None
    return (x, y)


def _normalize_view_range(view_range) -> tuple[tuple[float, float], tuple[float, float]] | None:
    try:
        x_range, y_range = view_range
        x0, x1 = float(x_range[0]), float(x_range[1])
        y0, y1 = float(y_range[0]), float(y_range[1])
    except Exception:
        return None
    if not all(isfinite(value) for value in (x0, x1, y0, y1)):
        return None
    return (x0, x1), (y0, y1)
