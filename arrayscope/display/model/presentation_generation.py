"""Qt-free presentation generation tracking.

The tracker owns the semantic convergence state for global presentation
commands.  Backends may converge through very different physical work, but
they acknowledge the same target revision here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose


LEVEL_REL_TOL = 1e-12
LEVEL_ABS_TOL = 1e-9


@dataclass(frozen=True)
class LevelPresentationTarget:
    revision: int
    levels: tuple[float, float]
    source: object | None = None
    semantic_key: object | None = None
    active_tiles: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "revision", int(self.revision))
        object.__setattr__(self, "levels", _normalize_levels(self.levels))
        object.__setattr__(self, "active_tiles", frozenset(int(tile) for tile in self.active_tiles))


@dataclass(frozen=True)
class PresentationGenerationSnapshot:
    revision: int
    target_levels: tuple[float, float] | None
    stale_count: int
    pending_count: int
    settled: bool
    active_tile_count: int
    active_presented_tile_count: int


@dataclass
class PresentationGenerationTracker:
    """Mutable convergence model for one latest presentation target."""

    revision: int = 0
    target_levels: tuple[float, float] | None = None
    target_source: object | None = None
    semantic_key: object | None = None
    active_tiles: frozenset[int] = field(default_factory=frozenset)
    tile_revisions: dict[int, int] = field(default_factory=dict)
    tile_values: dict[int, tuple[float, float]] = field(default_factory=dict)
    stale_active_tiles: set[int] = field(default_factory=set)

    @property
    def target(self) -> LevelPresentationTarget | None:
        if self.target_levels is None:
            return None
        return LevelPresentationTarget(
            revision=int(self.revision),
            levels=self.target_levels,
            source=self.target_source,
            semantic_key=self.semantic_key,
            active_tiles=self.active_tiles,
        )

    def begin_target(self, levels, source: object | None = None, active_tiles=()) -> bool:
        target = _normalize_levels(levels)
        self.active_tiles = frozenset(int(tile) for tile in tuple(active_tiles or ()))
        same_target = levels_match(self.target_levels, target)
        self.target_levels = target
        self.target_source = source
        self.semantic_key = getattr(source, "semantic_key", self.semantic_key)
        if not same_target:
            self.revision = int(self.revision) + 1
        self._refresh_stale_active_tiles()
        return bool(self.stale_active_tiles)

    def set_active_tiles(self, tile_ids) -> None:
        previous = self.active_tiles
        active = frozenset(int(tile) for tile in tuple(tile_ids or ()))
        if active == previous:
            return
        self.active_tiles = active
        if self.target_levels is None:
            self.stale_active_tiles.clear()
            return
        removed = previous.difference(active)
        if removed:
            self.stale_active_tiles.difference_update(removed)
        for tile in active.difference(previous):
            tile = int(tile)
            if self._tile_matches_target_value(tile):
                self.tile_revisions[tile] = int(self.revision)
                self.stale_active_tiles.discard(tile)
            else:
                self.stale_active_tiles.add(tile)

    def forget_tiles(self, tile_ids) -> None:
        for tile in tuple(tile_ids or ()):
            self.forget_tile(int(tile))

    def forget_tile(self, tile: int) -> None:
        tile = int(tile)
        self.tile_revisions.pop(tile, None)
        self.tile_values.pop(tile, None)
        self.active_tiles = frozenset(active for active in self.active_tiles if int(active) != tile)
        self.stale_active_tiles.discard(tile)

    def stale_tiles(self, priority_order=None) -> tuple[int, ...]:
        if self.target_levels is None:
            return ()
        stale = set(int(tile) for tile in self.stale_active_tiles)
        if priority_order is None:
            return tuple(sorted(stale))
        ordered = []
        seen = set()
        for tile in tuple(priority_order or ()):
            tile = int(tile)
            if tile in stale and tile not in seen:
                ordered.append(tile)
                seen.add(tile)
        ordered.extend(tile for tile in sorted(stale) if tile not in seen)
        return tuple(ordered)

    def pending_tiles(self, pending_upserts=(), priority_order=None) -> tuple[int, ...]:
        pending = set(self.stale_tiles(priority_order=priority_order))
        if self.target_levels is not None:
            pending.update(
                int(tile)
                for tile in tuple(pending_upserts or ())
                if int(tile) in self.active_tiles and not self.tile_matches_target(int(tile))
            )
        if priority_order is None:
            return tuple(sorted(pending))
        ordered = []
        seen = set()
        for tile in tuple(priority_order or ()):
            tile = int(tile)
            if tile in pending and tile not in seen:
                ordered.append(tile)
                seen.add(tile)
        ordered.extend(tile for tile in sorted(pending) if tile not in seen)
        return tuple(ordered)

    def acknowledge_upserts(self, target_revision: int, accepted_tiles, levels=None) -> None:
        if int(target_revision) != int(self.revision):
            return
        committed = self.target_levels if levels is None else _normalize_levels(levels)
        for tile in tuple(accepted_tiles or ()):
            tile = int(tile)
            self.tile_values[tile] = committed
            self.tile_revisions[tile] = int(target_revision)
            if tile in self.active_tiles:
                if self.tile_matches_target(tile):
                    self.stale_active_tiles.discard(tile)
                else:
                    self.stale_active_tiles.add(tile)

    def acknowledge_uniform(self, target_revision: int, active_tiles, levels=None) -> None:
        if int(target_revision) != int(self.revision):
            return
        committed = self.target_levels if levels is None else _normalize_levels(levels)
        if committed is None:
            return
        self.set_active_tiles(active_tiles)
        for tile in self.active_tiles:
            self.tile_values[int(tile)] = committed
            self.tile_revisions[int(tile)] = int(target_revision)
        self._refresh_stale_active_tiles()

    def tile_matches_target(self, tile: int) -> bool:
        if self.target_levels is None:
            return True
        return self._tile_matches_target_value(int(tile))

    def _tile_matches_target_value(self, tile: int) -> bool:
        return (
            levels_match(self.tile_values.get(int(tile)), self.target_levels)
            and self.tile_revisions.get(int(tile)) == int(self.revision)
        )

    def value_counts(self, tile_ids=None) -> dict[tuple[float, float], int]:
        scope = self.active_tiles if tile_ids is None else frozenset(int(tile) for tile in tuple(tile_ids or ()))
        counts: dict[tuple[float, float], int] = {}
        for tile in scope:
            value = self.tile_values.get(int(tile))
            if value is not None:
                counts[value] = int(counts.get(value, 0)) + 1
        return counts

    def snapshot(self, *, pending_upserts=(), active_tile_count: int | None = None) -> PresentationGenerationSnapshot:
        stale_count = 0 if self.target_levels is None else len(self.stale_active_tiles)
        pending_count = stale_count
        if self.target_levels is not None:
            pending_count = len(
                set(self.stale_active_tiles).union(
                    int(tile)
                    for tile in tuple(pending_upserts or ())
                    if int(tile) in self.active_tiles and not self.tile_matches_target(int(tile))
                )
            )
        return PresentationGenerationSnapshot(
            revision=int(self.revision),
            target_levels=self.target_levels,
            stale_count=int(stale_count),
            pending_count=int(pending_count),
            settled=stale_count == 0 and pending_count == 0,
            active_tile_count=len(self.active_tiles) if active_tile_count is None else max(0, int(active_tile_count)),
            active_presented_tile_count=len(self.active_tiles),
        )

    def _refresh_stale_active_tiles(self) -> None:
        if self.target_levels is None:
            self.stale_active_tiles.clear()
            return
        stale: set[int] = set()
        for tile in self.active_tiles:
            tile = int(tile)
            if levels_match(self.tile_values.get(tile), self.target_levels):
                self.tile_revisions[tile] = int(self.revision)
            if not self._tile_matches_target_value(tile):
                stale.add(tile)
        self.stale_active_tiles = stale


def _normalize_levels(levels) -> tuple[float, float]:
    low, high = levels
    return (float(low), float(high))


def levels_match(first, second) -> bool:
    """Return whether two level pairs are the same presentation target."""

    if first is None or second is None:
        return first is None and second is None
    try:
        first_low, first_high = first
        second_low, second_high = second
        return isclose(
            float(first_low),
            float(second_low),
            rel_tol=LEVEL_REL_TOL,
            abs_tol=LEVEL_ABS_TOL,
        ) and isclose(
            float(first_high),
            float(second_high),
            rel_tol=LEVEL_REL_TOL,
            abs_tol=LEVEL_ABS_TOL,
        )
    except Exception:
        return False
