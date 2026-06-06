"""Pure helpers for image window/level state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


Levels = Tuple[float, float]


@dataclass(frozen=True)
class WindowLevelDecision:
    auto_levels: bool
    levels: Optional[Levels]


def normalize_levels(levels) -> Optional[Levels]:
    if levels is None:
        return None
    low, high = levels
    return (float(low), float(high))


def relative_levels(previous_levels, previous_bounds, current_bounds) -> Optional[Levels]:
    previous_levels = normalize_levels(previous_levels)
    previous_bounds = normalize_levels(previous_bounds)
    current_bounds = normalize_levels(current_bounds)
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
    """Choose whether the next image update should auto-window or reuse levels."""
    if force_auto:
        return WindowLevelDecision(auto_levels=True, levels=normalize_levels(default_levels))

    if mode == "absolute" and previous_levels is not None:
        return WindowLevelDecision(auto_levels=False, levels=normalize_levels(previous_levels))

    if mode == "relative":
        levels = relative_levels(previous_levels, previous_bounds, current_bounds)
        if levels is not None:
            return WindowLevelDecision(auto_levels=False, levels=levels)

    return WindowLevelDecision(auto_levels=True, levels=normalize_levels(default_levels))
