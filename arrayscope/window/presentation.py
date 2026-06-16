"""Qt-free display presentation and window/level policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from arrayscope.display.levels import finite_bounds
from arrayscope.core.window_levels import choose_window_levels
from arrayscope.window.display_frame import CommittedDisplayFrame
from arrayscope.window.render_model import CommitKind, DisplayPresentation, PresentationDecision, PresentationInput


class LevelSourceRank(IntEnum):
    NONE = 0
    FALLBACK = 1
    PREVIOUS_COMMITTED = 2
    MONTAGE_VISIBLE_SUBSET = 3
    MONTAGE_COMPLETE = 4
    MONTAGE_SAMPLED_FULL = 5
    EXPLICIT_USER = 6


@dataclass(frozen=True)
class LevelSource:
    levels: tuple[float, float]
    histogram_range: tuple[float, float]
    rank: LevelSourceRank
    source_count: int = 0
    expected_count: int = 0
    semantic_key: object | None = None


def normalize_bounds(bounds) -> tuple[float, float] | None:
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


def display_data_bounds(data, histogram_data=None) -> tuple[float, float] | None:
    source = histogram_data if histogram_data is not None else data
    try:
        return normalize_bounds(finite_bounds(source))
    except Exception:
        return None


def fallback_level_source(previous_frame: CommittedDisplayFrame | None, *, fallback=(0.0, 1.0)) -> LevelSource:
    if previous_frame is not None:
        levels = normalize_bounds(previous_frame.levels)
        histogram_range = normalize_bounds(previous_frame.histogram_range)
        if levels is not None:
            return LevelSource(
                levels=levels,
                histogram_range=histogram_range or levels,
                rank=LevelSourceRank.PREVIOUS_COMMITTED,
                semantic_key=previous_frame.key.semantic_key,
            )
        if histogram_range is not None:
            return LevelSource(
                levels=histogram_range,
                histogram_range=histogram_range,
                rank=LevelSourceRank.PREVIOUS_COMMITTED,
                semantic_key=previous_frame.key.semantic_key,
            )
    fallback_bounds = normalize_bounds(fallback) or (0.0, 1.0)
    return LevelSource(levels=fallback_bounds, histogram_range=fallback_bounds, rank=LevelSourceRank.FALLBACK)


def decide_presentation(input: PresentationInput) -> PresentationDecision:
    """Decide display levels, histogram range, and commit eligibility.

    This is the semantic boundary before Qt display mutation. It deliberately
    ignores widget state: previous display policy comes from the committed
    display frame, and montage auto-level policy comes from ranked semantic
    tile coverage instead of partial viewport canvas pixels.
    """
    kind = CommitKind(input.commit_kind)
    payload = input.payload
    if kind in {CommitKind.FULL_NORMAL, CommitKind.DEGRADED_PREVIEW}:
        return _decide_normal_presentation(input)
    if kind in {
        CommitKind.FULL_MONTAGE_INITIAL,
        CommitKind.PROGRESSIVE_MONTAGE_PATCH,
        CommitKind.EXPLICIT_AUTO_WINDOW,
    }:
        return _decide_montage_presentation(input)
    raise ValueError(f"unsupported commit kind: {kind!r}")


def _decide_normal_presentation(input: PresentationInput) -> PresentationDecision:
    payload = input.payload
    default_levels = normalize_bounds(getattr(payload.image, "default_levels", None))
    current_bounds = (
        normalize_bounds(input.level_bounds)
        or display_data_bounds(payload.data, payload.histogram_data)
        or default_levels
        or fallback_level_source(input.previous_frame).histogram_range
    )
    previous_levels = None if input.previous_frame is None else input.previous_frame.levels
    previous_bounds = None if input.previous_frame is None else input.previous_frame.histogram_range
    level_decision = choose_window_levels(
        mode=input.window_mode,
        previous_levels=previous_levels,
        previous_bounds=previous_bounds,
        current_bounds=current_bounds,
        default_levels=default_levels or current_bounds,
        force_auto=bool(input.force_auto),
    )
    levels = normalize_bounds(level_decision.levels) or current_bounds or fallback_level_source(input.previous_frame).levels
    histogram_range = current_bounds or levels
    source = LevelSource(
        levels=levels,
        histogram_range=histogram_range,
        rank=LevelSourceRank.EXPLICIT_USER if input.force_auto else LevelSourceRank.PREVIOUS_COMMITTED,
        source_count=1,
        expected_count=1,
        semantic_key=input.context.semantic_key,
    )
    presentation = DisplayPresentation(
        data=payload.data,
        histogram_data=payload.histogram_data,
        geometry=payload.geometry,
        levels=levels,
        histogram_range=histogram_range,
        viewport_policy=payload.viewport_policy,
        rgb_already_windowed=payload.rgb_already_windowed,
    )
    return PresentationDecision(
        display_presentation=presentation,
        levels=levels,
        histogram_range=histogram_range,
        level_source_rank=int(source.rank),
        level_source_key=source.semantic_key,
        level_source_count=source.source_count,
        expected_source_count=source.expected_count,
        allow_fast_commit=False,
        applied_level_source=source,
    )


def _decide_montage_presentation(input: PresentationInput) -> PresentationDecision:
    payload = input.payload
    semantic = input.semantic_source
    previous = fallback_level_source(input.previous_frame)
    applied = input.applied_level_source
    kind = CommitKind(input.commit_kind)
    explicit_auto = bool(input.force_auto or kind == CommitKind.EXPLICIT_AUTO_WINDOW)
    source = _select_montage_level_source(
        semantic,
        previous,
        applied,
        kind=kind,
        explicit_auto=explicit_auto,
    )
    levels = normalize_bounds(source.levels) or previous.levels
    histogram_range = normalize_bounds(source.histogram_range) or previous.histogram_range or levels
    presentation = DisplayPresentation(
        data=payload.data,
        histogram_data=payload.histogram_data,
        geometry=payload.geometry,
        levels=levels,
        histogram_range=histogram_range,
        viewport_policy=payload.viewport_policy,
        rgb_already_windowed=payload.rgb_already_windowed,
    )
    return PresentationDecision(
        display_presentation=presentation,
        levels=levels,
        histogram_range=histogram_range,
        level_source_rank=int(source.rank),
        level_source_key=source.semantic_key,
        level_source_count=int(source.source_count),
        expected_source_count=int(source.expected_count),
        allow_fast_commit=kind == CommitKind.PROGRESSIVE_MONTAGE_PATCH,
        applied_level_source=source if _source_is_semantic(source) else applied,
    )


def _select_montage_level_source(
    semantic,
    previous: LevelSource,
    applied,
    *,
    kind: CommitKind,
    explicit_auto: bool,
) -> LevelSource:
    semantic = semantic if isinstance(semantic, LevelSource) and normalize_bounds(semantic.levels) is not None else None
    applied = applied if isinstance(applied, LevelSource) and normalize_bounds(applied.levels) is not None else None
    if semantic is None:
        return applied or previous
    if explicit_auto:
        return semantic
    if semantic.rank == LevelSourceRank.MONTAGE_VISIBLE_SUBSET:
        return applied or previous
    if kind == CommitKind.PROGRESSIVE_MONTAGE_PATCH and applied is not None:
        if int(semantic.rank) < int(applied.rank):
            return applied
        if int(semantic.rank) == int(applied.rank) and int(semantic.source_count) <= int(applied.source_count):
            return applied
    return semantic


def _source_is_semantic(source: LevelSource) -> bool:
    return source.rank in {
        LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
        LevelSourceRank.MONTAGE_COMPLETE,
        LevelSourceRank.MONTAGE_SAMPLED_FULL,
        LevelSourceRank.EXPLICIT_USER,
    }
