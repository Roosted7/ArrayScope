"""Pure helpers and state model for image window/level decisions.

This module intentionally has no Qt or pyqtgraph imports.  The rest of the
application should treat it as the single source of truth for automatic vs
user-locked display levels.  Renderers provide candidate data bounds; widgets
only apply the resulting state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from typing import Optional, Tuple

import numpy as np


Levels = Tuple[float, float]


class LevelMode(str, Enum):
    """How automatic levels should relate to the previous semantic source."""

    AUTO = "auto"
    RELATIVE = "relative"
    ABSOLUTE = "absolute"
    USER_LOCKED = "user_locked"


class LevelSourceRank(IntEnum):
    """Ranking of level sources."""

    NONE = 0
    FALLBACK = 1
    PREVIOUS_COMMITTED = 2
    MONTAGE_VISIBLE_SUBSET = 3
    MONTAGE_COMPLETE = 4
    MONTAGE_SAMPLED_FULL = 5
    EXPLICIT_USER = 6


@dataclass(frozen=True)
class WindowLevelDecision:
    auto_levels: bool
    levels: Optional[Levels]


@dataclass(frozen=True)
class LevelSource:
    levels: Levels
    histogram_range: Levels
    rank: LevelSourceRank
    source_count: int = 0
    expected_count: int = 0
    semantic_key: object | None = None
    mode: LevelMode = LevelMode.USER_LOCKED

    @property
    def user_locked(self) -> bool:
        return self.rank == LevelSourceRank.EXPLICIT_USER and self.mode != LevelMode.RELATIVE


@dataclass(frozen=True)
class WindowLevelState:
    semantic_key: object | None
    display_levels: Levels
    histogram_range: Levels
    source_rank: LevelSourceRank
    source_count: int = 0
    expected_count: int = 0
    user_locked: bool = False
    mode: LevelMode = LevelMode.RELATIVE

    def as_level_source(self) -> LevelSource:
        return LevelSource(
            levels=self.display_levels,
            histogram_range=self.histogram_range,
            rank=LevelSourceRank.EXPLICIT_USER if self.user_locked else self.source_rank,
            source_count=self.source_count,
            expected_count=self.expected_count,
            semantic_key=self.semantic_key,
            mode=self.mode,
        )


def normalize_levels(levels) -> Optional[Levels]:
    return normalize_bounds(levels)


def normalize_bounds(bounds) -> Optional[Levels]:
    if bounds is None:
        return None
    try:
        low, high = bounds
        low = float(low)
        high = float(high)
    except Exception:
        return None
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if high > low:
        return (low, high)
    center = low
    radius = max(abs(center) * 0.01, 0.5)
    return (center - radius, center + radius)


def union_bounds(a, b) -> Optional[Levels]:
    a = normalize_bounds(a)
    b = normalize_bounds(b)
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]))


def relative_levels(previous_levels, previous_bounds, current_bounds) -> Optional[Levels]:
    previous_levels = normalize_bounds(previous_levels)
    previous_bounds = normalize_bounds(previous_bounds)
    current_bounds = normalize_bounds(current_bounds)
    if previous_levels is None or previous_bounds is None or current_bounds is None:
        return None

    previous_low, previous_high = previous_bounds
    current_low, current_high = current_bounds
    previous_span = previous_high - previous_low
    current_span = current_high - current_low
    if previous_span <= 0 or current_span <= 0:
        return None

    low_fraction = (previous_levels[0] - previous_low) / previous_span
    high_fraction = (previous_levels[1] - previous_low) / previous_span
    return (
        current_low + low_fraction * current_span,
        current_low + high_fraction * current_span,
    )


def choose_window_levels(
    mode: str,
    previous_levels=None,
    previous_bounds=None,
    current_bounds=None,
    default_levels=None,
    force_auto: bool = False,
) -> WindowLevelDecision:
    """Backward-compatible helper for simple non-montage image decisions."""
    if force_auto:
        return WindowLevelDecision(auto_levels=True, levels=normalize_bounds(default_levels))

    if mode == "absolute" and previous_levels is not None:
        return WindowLevelDecision(auto_levels=False, levels=normalize_bounds(previous_levels))

    if mode == "relative":
        levels = relative_levels(previous_levels, previous_bounds, current_bounds)
        if levels is not None:
            return WindowLevelDecision(auto_levels=False, levels=levels)

    return WindowLevelDecision(auto_levels=True, levels=normalize_bounds(default_levels))


def state_from_source(source: LevelSource | None, *, mode: str | LevelMode = LevelMode.RELATIVE) -> WindowLevelState | None:
    if source is None:
        return None
    levels = normalize_bounds(source.levels)
    histogram = normalize_bounds(source.histogram_range) or levels
    if levels is None or histogram is None:
        return None
    rank = source.rank if isinstance(source.rank, LevelSourceRank) else LevelSourceRank(int(source.rank))
    source_mode = getattr(source, "mode", mode) if rank == LevelSourceRank.EXPLICIT_USER else mode
    try:
        coerced_mode = source_mode if isinstance(source_mode, LevelMode) else LevelMode(str(source_mode))
    except ValueError:
        try:
            coerced_mode = mode if isinstance(mode, LevelMode) else LevelMode(str(mode))
        except ValueError:
            coerced_mode = LevelMode.RELATIVE
    return WindowLevelState(
        semantic_key=source.semantic_key,
        display_levels=levels,
        histogram_range=histogram,
        source_rank=rank,
        source_count=max(0, int(source.source_count)),
        expected_count=max(0, int(source.expected_count)),
        user_locked=rank == LevelSourceRank.EXPLICIT_USER and coerced_mode != LevelMode.RELATIVE,
        mode=coerced_mode,
    )


def source_from_state(state: WindowLevelState | None) -> LevelSource | None:
    return None if state is None else state.as_level_source()


class WindowLevelController:
    """Pure automatic/user window-level policy."""

    def decide(
        self,
        *,
        previous: WindowLevelState | LevelSource | None,
        candidate: LevelSource | None,
        explicit_auto: bool = False,
        user_levels: Levels | None = None,
        mode: str | LevelMode = LevelMode.RELATIVE,
    ) -> WindowLevelState:
        mode = self._coerce_mode(mode)
        previous_state = previous if isinstance(previous, WindowLevelState) else state_from_source(previous, mode=mode)
        candidate_state = state_from_source(candidate, mode=mode)

        if user_levels is not None and not explicit_auto:
            levels = normalize_bounds(user_levels)
            if levels is not None:
                histogram = None if previous_state is None else previous_state.histogram_range
                if candidate_state is not None:
                    histogram = union_bounds(histogram, candidate_state.histogram_range)
                return WindowLevelState(
                    semantic_key=(candidate_state.semantic_key if candidate_state is not None else (previous_state.semantic_key if previous_state else None)),
                    display_levels=levels,
                    histogram_range=histogram or levels,
                    source_rank=LevelSourceRank.EXPLICIT_USER,
                    source_count=0,
                    expected_count=0,
                    user_locked=mode == LevelMode.ABSOLUTE,
                    mode=LevelMode.USER_LOCKED if mode == LevelMode.ABSOLUTE else LevelMode.RELATIVE,
                )

        if explicit_auto:
            if candidate_state is not None:
                return replace(candidate_state, user_locked=False, mode=mode)
            if previous_state is not None:
                return replace(previous_state, user_locked=False, source_rank=max(previous_state.source_rank, LevelSourceRank.PREVIOUS_COMMITTED), mode=mode)
            return self._fallback_state(mode=mode)

        if candidate_state is None:
            if previous_state is not None:
                return previous_state
            return self._fallback_state(mode=mode)

        if previous_state is None:
            return candidate_state

        same_semantic = previous_state.semantic_key == candidate_state.semantic_key

        if previous_state.user_locked and same_semantic:
            histogram = union_bounds(previous_state.histogram_range, candidate_state.histogram_range) or previous_state.histogram_range
            return replace(
                previous_state,
                histogram_range=histogram,
                source_rank=max(previous_state.source_rank, candidate_state.source_rank),
                source_count=max(previous_state.source_count, candidate_state.source_count),
                expected_count=max(previous_state.expected_count, candidate_state.expected_count),
                mode=previous_state.mode,
            )

        if not same_semantic:
            if mode == LevelMode.ABSOLUTE:
                histogram = candidate_state.histogram_range
                return WindowLevelState(
                    semantic_key=candidate_state.semantic_key,
                    display_levels=previous_state.display_levels,
                    histogram_range=histogram,
                    source_rank=candidate_state.source_rank,
                    source_count=candidate_state.source_count,
                    expected_count=candidate_state.expected_count,
                    user_locked=previous_state.user_locked,
                    mode=previous_state.mode if previous_state.user_locked else mode,
                )
            if mode == LevelMode.RELATIVE:
                mapped = relative_levels(previous_state.display_levels, previous_state.histogram_range, candidate_state.histogram_range)
                return WindowLevelState(
                    semantic_key=candidate_state.semantic_key,
                    display_levels=normalize_bounds(mapped) or candidate_state.display_levels,
                    histogram_range=candidate_state.histogram_range,
                    source_rank=candidate_state.source_rank,
                    source_count=candidate_state.source_count,
                    expected_count=candidate_state.expected_count,
                    user_locked=False,
                    mode=mode,
                )
            return candidate_state

        if mode == LevelMode.ABSOLUTE and not previous_state.user_locked:
            histogram = union_bounds(previous_state.histogram_range, candidate_state.histogram_range) or candidate_state.histogram_range
            return replace(
                previous_state,
                histogram_range=histogram,
                source_rank=max(previous_state.source_rank, candidate_state.source_rank),
                source_count=max(previous_state.source_count, candidate_state.source_count),
                expected_count=max(previous_state.expected_count, candidate_state.expected_count),
                mode=mode,
            )

        # Progressive statistics for an already-visible semantic frame are
        # metadata improvements, not a new windowing request.  Re-expressing
        # the current levels against every newly discovered min/max causes the
        # image to visibly flash while montage tiles arrive or while scrolling
        # changes the sampled subset.  Relative mapping is therefore performed
        # only when the semantic source changes (the branch above).  For the
        # same source, keep the numeric display window stable and expand the
        # histogram domain monotonically as better statistics become available.
        histogram = union_bounds(previous_state.histogram_range, candidate_state.histogram_range)
        return replace(
            previous_state,
            histogram_range=histogram or previous_state.histogram_range,
            source_rank=max(previous_state.source_rank, candidate_state.source_rank),
            source_count=max(previous_state.source_count, candidate_state.source_count),
            expected_count=max(previous_state.expected_count, candidate_state.expected_count),
            user_locked=False,
            mode=mode,
        )

    def _fallback_state(self, *, mode: LevelMode) -> WindowLevelState:
        return WindowLevelState(
            semantic_key=None,
            display_levels=(0.0, 1.0),
            histogram_range=(0.0, 1.0),
            source_rank=LevelSourceRank.FALLBACK,
            mode=mode,
        )

    def _coerce_mode(self, mode: str | LevelMode) -> LevelMode:
        if isinstance(mode, LevelMode):
            return mode
        try:
            return LevelMode(str(mode))
        except ValueError:
            return LevelMode.RELATIVE
