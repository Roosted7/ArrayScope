"""Qt-free dynamic priority queues for montage tile scheduling."""

from __future__ import annotations

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
    """Mutable indexed queue that always pops under the current context.

    The context can be supplied as a value (``set_context``) or, preferably,
    as a ``context_provider`` callable owned by whoever decides scheduling
    (the montage session). With a provider the queue never holds a stale
    context copy: every pop resolves the provider, and the heap is rebuilt
    when the resolved context object changed since the keys were computed.
    An earlier design gave each queue its own context copy and re-keyed a
    bounded batch per retarget, so with several actors retargeting (hover,
    viewport restores, stage activation) the pop order depended on which
    context happened to be current when each tile was pushed.
    """

    def __init__(
        self,
        tiles=(),
        *,
        context: TilePriorityContext | None = None,
        context_provider=None,
    ) -> None:
        self._tiles: dict[int, object] = {}
        self._versions: dict[int, int] = {}
        self._sequence: dict[int, int] = {}
        self._heap: list[tuple[object, ...]] = []
        self._serial = count()
        self._context_provider = context_provider
        self._context = context or TilePriorityContext()
        self._keyed_context: TilePriorityContext | None = self.context
        self.retargeted_last = 0
        self.stale_entries_discarded = 0
        for tile in tuple(tiles or ()):
            self.append(tile)

    @property
    def context(self) -> TilePriorityContext:
        if self._context_provider is not None:
            provided = self._context_provider()
            if provided is not None:
                return provided
        return self._context

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

    def append(self, tile) -> None:
        index = _tile_index(tile)
        if index in self._tiles:
            return
        self._tiles[index] = tile
        self._versions[index] = 0
        self._sequence[index] = next(self._serial)
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
        """Adopt a new scheduling context; takes effect on every tile.

        Re-keying happens lazily on the next ``pop``, so this is O(1)
        regardless of queue size. ``max_items`` is accepted for backward
        compatibility and ignored. When a ``context_provider`` is set it
        takes precedence over the value stored here.
        """
        del max_items
        self._context = context
        self.retargeted_last = len(self._tiles)
        return int(self.retargeted_last)

    def pop(self, position=None):
        # Strictly priority-ordered under the CURRENT context: the heap is
        # rebuilt on the first pop after the resolved context changed, so a
        # retarget always takes full effect. (An earlier "fairness aging"
        # variant popped the oldest-inserted tile after every few priority
        # pops; any bulk drain therefore degenerated to insertion order,
        # visibly corrupting the montage fill order. Every queue here is
        # drained completely, so priority order cannot starve a tile — it
        # only decides when it completes.)
        self.last_pop_position = position
        if not self._tiles:
            return None
        if self.context is not self._keyed_context:
            self._rebuild_heap()
        while self._heap:
            key = heappop(self._heap)
            version = int(key[-1])
            index = int(key[-2])
            if index not in self._tiles or version != self._versions.get(index):
                self.stale_entries_discarded += 1
                continue
            tile = self._tiles.pop(index)
            self._versions.pop(index, None)
            self._sequence.pop(index, None)
            return tile
        # Heap entries can all be stale after heavy pruning. Rebuild lazily.
        self._rebuild_heap()
        if self._heap:
            return self.pop()
        return None

    def _rebuild_heap(self) -> None:
        self._keyed_context = self.context
        self._heap.clear()
        for index in tuple(self._tiles):
            self._push(index)

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

def tile_numbers(tiles) -> tuple[int, ...]:
    return tuple(int(_tile_index(tile)) for tile in tuple(tiles or ()))


def prioritize_tile_numbers(tiles, *, plan_tiles, context: TilePriorityContext) -> tuple[int, ...]:
    """Return tile numbers in the same priority order as the queue.

    This is for one-shot commit/admission ordering.  It deliberately avoids
    constructing a mutable queue when the caller already owns the candidate
    set and only needs a sorted view.
    """

    requested = tuple(dict.fromkeys(int(tile) for tile in tuple(tiles or ())))
    if len(requested) <= 1:
        return requested
    plan_tiles = tuple(plan_tiles or ())
    requested_set = set(requested)
    sequence = {int(tile): offset for offset, tile in enumerate(requested)}
    valid = tuple(tile for tile in requested if 0 <= int(tile) < len(plan_tiles))
    if not valid:
        return requested

    distance_score = _distance_scorer(context)

    def priority_key(tile_number: int) -> tuple[object, ...]:
        index = int(tile_number)
        tile = plan_tiles[index]
        return (
            int(_band_for_index(index, context)),
            distance_score(tile),
            int(sequence.get(index, 0)),
            index,
        )

    ordered = sorted(valid, key=priority_key)
    ordered_set = set(ordered)
    ordered.extend(tile for tile in requested if tile not in ordered_set)
    return tuple(int(tile) for tile in ordered if int(tile) in requested_set)


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
    return _distance_scorer(context)(tile)


def _distance_scorer(context: TilePriorityContext):
    ranges = _normalize_view_range(context.view_range)
    if ranges is None:
        return lambda _tile: 0.0
    (x0, x1), (y0, y1) = ranges
    span_x = max(1.0, abs(float(x1) - float(x0)))
    span_y = max(1.0, abs(float(y1) - float(y0)))
    focus = context.focus
    if focus is None:
        focus_x = (float(x0) + float(x1)) * 0.5
        focus_y = (float(y0) + float(y1)) * 0.5
    else:
        focus_x, focus_y = focus

    def score(tile) -> float:
        center_x = float(getattr(tile, "x0", 0.0)) + float(getattr(tile, "width", 1.0)) * 0.5
        center_y = float(getattr(tile, "y0", 0.0)) + float(getattr(tile, "height", 1.0)) * 0.5
        dx = (center_x - focus_x) / span_x
        dy = (center_y - focus_y) / span_y
        return float(dx * dx + dy * dy)

    return score


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
