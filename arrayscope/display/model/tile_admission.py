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
        free_fn=None,
        item_free_fn=None,
        max_item_free: int | None = None,
        max_items: int | None = None,
        max_bytes: int | None = None,
        deadline_ms: float | None = None,
    ) -> TileAdmissionDecision:
        ordered = self.order(candidates)
        item_cap = None if max_items is None else max(0, int(max_items))
        byte_cap = None if max_bytes is None else max(0, int(max_bytes))
        if (item_cap == 0 or byte_cap == 0) and free_fn is None:
            return TileAdmissionDecision((), ordered, tuple(dict.fromkeys(int(tile) for tile in tuple(retained or ()))))
        started = perf_counter()
        admitted: list[int] = []
        deferred: list[int] = []
        used_bytes = 0
        costed_admitted = 0
        item_free_admitted = 0
        item_free_cap = None if max_item_free is None else max(0, int(max_item_free))
        for tile in ordered:
            tile = int(tile)
            cost = 0 if cost_fn is None else max(0, int(cost_fn(tile) or 0))
            # `free_fn` declares an item genuinely instant for the backend
            # (e.g. a VisPy resident atlas remap): it bypasses every cap.
            # `item_free_fn` is weaker: it bypasses the item count but still
            # pays bytes/time. Everything else is paced because "not an
            # upload" is not "not work" on backends that rebuild items.
            free = free_fn is not None and bool(free_fn(tile))
            item_free_candidate = item_free_fn is not None and bool(item_free_fn(tile))
            item_free = free or item_free_candidate
            if item_free_candidate and not free and item_free_cap is not None and item_free_admitted >= item_free_cap:
                deferred.append(tile)
                continue
            if not free:
                if (
                    not item_free
                    and item_cap is not None
                    and (item_cap == 0 or costed_admitted >= item_cap)
                ):
                    deferred.append(tile)
                    continue
                if byte_cap is not None and (
                    byte_cap == 0 or (admitted and used_bytes + cost > byte_cap)
                ):
                    deferred.append(tile)
                    continue
                if deadline_ms is not None and admitted and (perf_counter() - started) * 1000.0 >= float(deadline_ms):
                    deferred.append(tile)
                    continue
            admitted.append(tile)
            used_bytes += cost
            if not free and not item_free:
                costed_admitted += 1
            elif item_free and not free:
                item_free_admitted += 1
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
