"""Qt-free display presentation and window/level policy.

Render code supplies facts: pixels, semantic keys, candidate data bounds, and
coverage.  This module is the only place that decides display levels and
histogram ranges before they are applied to Qt widgets.
"""

from __future__ import annotations

from arrayscope.core.window_levels import (
    LevelSource,
    LevelSourceRank,
    WindowLevelController,
    WindowLevelState,
    choose_window_levels,
    normalize_bounds,
)
from arrayscope.display.levels import finite_bounds
from arrayscope.window.display_frame import CommittedDisplayFrame
from arrayscope.window.render_model import CommitKind, DisplayPresentation, PresentationDecision, PresentationInput


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
    kind = CommitKind(input.commit_kind)
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
        rank=LevelSourceRank.PREVIOUS_COMMITTED,
        source_count=1,
        expected_count=1,
        semantic_key=input.context.semantic_key,
    )
    presentation = DisplayPresentation(
        data=payload.data,
        histogram_data=payload.histogram_data,
        histogram_plot_data=payload.histogram_plot_data,
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
    kind = CommitKind(input.commit_kind)
    explicit_auto = bool(input.force_auto or kind == CommitKind.EXPLICIT_AUTO_WINDOW)
    semantic = _valid_source(input.semantic_source)
    previous_source = _effective_previous_source(input)
    state = WindowLevelController().decide(
        previous=previous_source,
        candidate=semantic,
        explicit_auto=explicit_auto,
        mode=input.window_mode,
    )
    presentation = DisplayPresentation(
        data=payload.data,
        histogram_data=payload.histogram_data,
        histogram_plot_data=payload.histogram_plot_data,
        geometry=payload.geometry,
        levels=state.display_levels,
        histogram_range=state.histogram_range,
        viewport_policy=payload.viewport_policy,
        rgb_already_windowed=payload.rgb_already_windowed,
    )
    source = state.as_level_source()
    return PresentationDecision(
        display_presentation=presentation,
        levels=state.display_levels,
        histogram_range=state.histogram_range,
        level_source_rank=int(source.rank),
        level_source_key=source.semantic_key,
        level_source_count=int(source.source_count),
        expected_source_count=int(source.expected_count),
        allow_fast_commit=kind == CommitKind.PROGRESSIVE_MONTAGE_PATCH,
        applied_level_source=source,
    )


def _effective_previous_source(input: PresentationInput) -> LevelSource | None:
    applied = _valid_source(input.applied_level_source)
    if applied is not None:
        return applied
    return fallback_level_source(input.previous_frame)


def _valid_source(source) -> LevelSource | None:
    if not isinstance(source, LevelSource):
        return None
    levels = normalize_bounds(source.levels)
    histogram = normalize_bounds(source.histogram_range) or levels
    if levels is None or histogram is None:
        return None
    try:
        rank = source.rank if isinstance(source.rank, LevelSourceRank) else LevelSourceRank(int(source.rank))
    except Exception:
        return None
    return LevelSource(
        levels=levels,
        histogram_range=histogram,
        rank=rank,
        source_count=max(0, int(source.source_count)),
        expected_count=max(0, int(source.expected_count)),
        semantic_key=source.semantic_key,
        mode=getattr(source, "mode", "relative"),
    )


__all__ = [
    "LevelSource",
    "LevelSourceRank",
    "WindowLevelController",
    "WindowLevelState",
    "normalize_bounds",
    "display_data_bounds",
    "fallback_level_source",
    "decide_presentation",
]
