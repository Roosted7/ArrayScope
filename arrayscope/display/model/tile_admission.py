"""Qt-free tile admission for bounded presentation work."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from arrayscope.display.model.tile_priority import MontageTilePriorityQueue, TilePriorityContext, tile_numbers


@dataclass(frozen=True)
class TileAdmissionDecision:
    admitted: tuple[int, ...]
    deferred: tuple[int, ...]
    active: tuple[int, ...]
    admitted_bytes: int = 0


@dataclass
class TileAdmissionQueue:
    """Owns ordering and admission caps without semantic rendering meaning."""

    context: TilePriorityContext = field(default_factory=TilePriorityContext)

    def order(self, candidates) -> tuple[int, ...]:
        queue = MontageTilePriorityQueue(tuple(candidates or ()), context=self.context)
        ordered = []
        while queue:
            tile = queue.pop()
            if tile is None:
                break
            ordered.append(_tile_index(tile))
        return tuple(ordered)

    def admit(
        self,
        candidates,
        *,
        retained=(),
        cost_fn=None,
        max_items: int | None = None,
        max_bytes: int | None = None,
        deadline_ms: float | None = None,
    ) -> TileAdmissionDecision:
        ordered = self.order(candidates)
        item_cap = None if max_items is None else max(0, int(max_items))
        byte_cap = None if max_bytes is None else max(0, int(max_bytes))
        if item_cap == 0 or byte_cap == 0:
            return TileAdmissionDecision((), ordered, tuple(dict.fromkeys(int(tile) for tile in tuple(retained or ()))))
        started = perf_counter()
        admitted: list[int] = []
        deferred: list[int] = []
        used_bytes = 0
        for tile in ordered:
            tile = int(tile)
            cost = 0 if cost_fn is None else max(0, int(cost_fn(tile) or 0))
            if item_cap is not None and len(admitted) >= item_cap:
                deferred.append(tile)
                continue
            if byte_cap is not None and admitted and used_bytes + cost > byte_cap:
                deferred.append(tile)
                continue
            if deadline_ms is not None and admitted and (perf_counter() - started) * 1000.0 >= float(deadline_ms):
                deferred.append(tile)
                continue
            admitted.append(tile)
            used_bytes += cost
        active = tuple(dict.fromkeys((*tuple(int(tile) for tile in tuple(retained or ())), *admitted)))
        return TileAdmissionDecision(
            admitted=tuple(admitted),
            deferred=tuple(deferred),
            active=active,
            admitted_bytes=int(used_bytes),
        )


def _tile_index(tile_or_index) -> int:
    try:
        return int(tile_or_index.montage_index)
    except AttributeError:
        return int(tile_or_index)


__all__ = [
    "TileAdmissionDecision",
    "TileAdmissionQueue",
    "TilePriorityContext",
    "tile_numbers",
]
