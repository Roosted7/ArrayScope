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
    normalize_bounds,
)
from arrayscope.display.model.commit import (
    CommitKind,
    DisplayTiledPresentation,
    PresentationDecision,
    PresentationInput,
)
from arrayscope.display.model.frame import CommittedDisplayFrame
from arrayscope.display.shader_mapping import common_shader_mapping


def fallback_level_source(
    previous_frame: CommittedDisplayFrame | None, *, fallback=(0.0, 1.0)
) -> LevelSource:
    if previous_frame is not None:
        levels = normalize_bounds(previous_frame.levels)
        histogram_range = normalize_bounds(previous_frame.histogram_range)
        if levels is not None:
            return LevelSource(
                levels=levels,
                histogram_range=histogram_range or levels,
                rank=LevelSourceRank.PREVIOUS_COMMITTED,
                semantic_key=previous_frame.key.semantic_key,
                evidence_quality=0,
            )
        if histogram_range is not None:
            return LevelSource(
                levels=histogram_range,
                histogram_range=histogram_range,
                rank=LevelSourceRank.PREVIOUS_COMMITTED,
                semantic_key=previous_frame.key.semantic_key,
                evidence_quality=0,
            )
    fallback_bounds = normalize_bounds(fallback) or (0.0, 1.0)
    return LevelSource(
        levels=fallback_bounds, histogram_range=fallback_bounds, rank=LevelSourceRank.FALLBACK
    )


def decide_presentation(input: PresentationInput) -> PresentationDecision:
    kind = CommitKind(input.commit_kind)
    if kind in {
        CommitKind.FULL_FRAME_INITIAL,
        CommitKind.PROGRESSIVE_FRAME_PATCH,
        CommitKind.EXPLICIT_AUTO_WINDOW,
        CommitKind.DEGRADED_PREVIEW,
    }:
        return _decide_montage_presentation(input)
    raise ValueError(f"unsupported commit kind: {kind!r}")


def _decide_montage_presentation(input: PresentationInput) -> PresentationDecision:
    payload = input.payload
    kind = CommitKind(input.commit_kind)
    explicit_auto = bool(input.force_auto or kind == CommitKind.EXPLICIT_AUTO_WINDOW)
    semantic = _valid_source(input.semantic_source)
    requested_levels = normalize_bounds(input.user_levels)
    # A synthetic (0, 1) fallback is useful for an empty automatic frame, but
    # it is not real histogram evidence.  Do not union it into a restored
    # explicit window before the first montage statistics arrive.
    if (
        requested_levels is not None
        and input.previous_frame is None
        and _valid_source(input.applied_level_source) is None
    ):
        previous_source = None
    else:
        previous_source = _effective_previous_source(input)
    state = WindowLevelController().decide(
        previous=previous_source,
        candidate=semantic,
        explicit_auto=explicit_auto,
        user_levels=requested_levels,
        mode=input.window_mode,
    )
    presentation = _presentation_for_payload(
        payload,
        levels=state.display_levels,
        histogram_range=state.histogram_range,
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
        allow_fast_commit=kind == CommitKind.PROGRESSIVE_FRAME_PATCH,
        applied_level_source=source,
    )


def _effective_previous_source(input: PresentationInput) -> LevelSource | None:
    applied = _valid_source(input.applied_level_source)
    if applied is not None:
        return applied
    return fallback_level_source(input.previous_frame)


def _presentation_for_payload(payload, *, levels, histogram_range):
    if payload.tile_state is None or payload.tile_delta is None:
        raise ValueError("display presentations require tile_state and tile_delta")
    base_tile_state = payload.base_tile_state or payload.tile_state
    return DisplayTiledPresentation(
        geometry=payload.geometry,
        levels=levels,
        histogram_range=histogram_range,
        viewport_policy=payload.viewport_policy,
        frame_plan=payload.frame_plan,
        rgb_already_windowed=payload.rgb_already_windowed,
        tile_state=payload.tile_state,
        base_tile_state=base_tile_state,
        tile_delta=payload.tile_delta,
        tile_residency_budget_bytes=int(payload.tile_residency_budget_bytes),
        histogram_plot_data=payload.histogram_plot_data,
        shader_mapping=common_shader_mapping(
            getattr(tile, "shader_mapping", None) for tile in payload.tile_state.payloads.values()
        ),
    )


def _valid_source(source) -> LevelSource | None:
    if not isinstance(source, LevelSource):
        return None
    levels = normalize_bounds(source.levels)
    histogram = normalize_bounds(source.histogram_range) or levels
    if levels is None or histogram is None:
        return None
    try:
        rank = (
            source.rank
            if isinstance(source.rank, LevelSourceRank)
            else LevelSourceRank(int(source.rank))
        )
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
        evidence_quality=max(0, int(getattr(source, "evidence_quality", 0) or 0)),
    )


__all__ = [
    "LevelSource",
    "LevelSourceRank",
    "WindowLevelController",
    "WindowLevelState",
    "decide_presentation",
    "fallback_level_source",
    "normalize_bounds",
]
