"""Backend-specific level convergence strategies behind one semantic contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from arrayscope.display.model.presentation_generation import PresentationGenerationTracker


class LevelConvergenceStrategy(Protocol):
    def begin(
        self, tracker: PresentationGenerationTracker, levels, *, source=None, active_tiles=()
    ) -> bool: ...


@dataclass(frozen=True)
class ProgressiveTileLevelConvergence:
    """PyQtGraph-style convergence through bounded per-tile upserts."""

    def begin(
        self, tracker: PresentationGenerationTracker, levels, *, source=None, active_tiles=()
    ) -> bool:
        return tracker.begin_target(levels, source=source, active_tiles=active_tiles)

    def stale_tiles(
        self, tracker: PresentationGenerationTracker, priority_order=None
    ) -> tuple[int, ...]:
        return tracker.stale_tiles(priority_order=priority_order)

    def acknowledge(
        self,
        tracker: PresentationGenerationTracker,
        *,
        target_revision: int,
        accepted_tiles,
        levels=None,
    ) -> None:
        tracker.acknowledge_upserts(target_revision, accepted_tiles, levels=levels)


@dataclass(frozen=True)
class UniformLevelConvergence:
    """Shader/uniform convergence for compatible tiled surfaces."""

    def begin(
        self, tracker: PresentationGenerationTracker, levels, *, source=None, active_tiles=()
    ) -> bool:
        tracker.begin_target(levels, source=source, active_tiles=active_tiles)
        tracker.acknowledge_uniform(int(tracker.revision), active_tiles, levels=levels)
        return False

    def acknowledge(
        self,
        tracker: PresentationGenerationTracker,
        *,
        target_revision: int,
        active_tiles,
        levels=None,
    ) -> None:
        tracker.acknowledge_uniform(target_revision, active_tiles, levels=levels)
