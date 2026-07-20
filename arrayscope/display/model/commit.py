"""Qt-free render presentation request and commit models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import numpy as np

from arrayscope.display.geometry import DisplayGeometry
from arrayscope.display.model.frame import (
    CommittedDisplayFrame,
    DisplayFrameKey,
    TilePresentationDelta,
    TilePresentationState,
)
from arrayscope.display.model.tile_identity import TilePresentationIdentity
from arrayscope.display.shader_mapping import ShaderMapping
from arrayscope.display.slice_engine import DisplayImage
from arrayscope.display.viewport import ViewportPolicy


class CommitKind(Enum):
    FULL_FRAME_INITIAL = "full_frame_initial"
    PROGRESSIVE_FRAME_PATCH = "progressive_frame_patch"
    EXPLICIT_AUTO_WINDOW = "explicit_auto_window"
    DEGRADED_PREVIEW = "degraded_preview"


@dataclass(frozen=True)
class RenderRequestContext:
    document_key: object
    request_key: object
    render_generation: int
    semantic_key: object | None = None

    @property
    def frame_key(self) -> DisplayFrameKey:
        return DisplayFrameKey(
            document_key=self.document_key,
            request_key=self.request_key,
            render_generation=int(self.render_generation),
            semantic_key=self.semantic_key,
        )


@dataclass(frozen=True)
class DisplayPayload:
    image: DisplayImage
    geometry: DisplayGeometry
    viewport_policy: ViewportPolicy
    frame_plan: Any = None
    rgb_already_windowed: bool = False
    histogram_plot_data: np.ndarray | None = None
    montage_dirty_tiles: tuple[int, ...] | None = None
    montage_tile_source_ids: dict[int, object] | None = None
    tile_state: TilePresentationState | None = None
    base_tile_state: TilePresentationState | None = None
    tile_delta: TilePresentationDelta | None = None
    tile_residency_budget_bytes: int = 0

    @property
    def data(self) -> np.ndarray:
        return self.image.data

    @property
    def histogram_data(self) -> np.ndarray | None:
        return self.image.histogram_data


@dataclass(frozen=True)
class DisplayTiledPresentation:
    geometry: DisplayGeometry
    levels: tuple[float, float]
    histogram_range: tuple[float, float]
    viewport_policy: ViewportPolicy
    tile_state: TilePresentationState
    base_tile_state: TilePresentationState
    tile_delta: TilePresentationDelta
    tile_residency_budget_bytes: int
    frame_plan: Any = None
    histogram_plot_data: np.ndarray | None = None
    rgb_already_windowed: bool = False
    shader_mapping: ShaderMapping | None = None

    def __post_init__(self) -> None:
        """Bind emitted wrappers to this transaction's accepted levels."""

        levels = tuple(float(value) for value in self.levels)
        transaction_payloads = self.tile_state.active_payloads(self.tile_delta)
        transaction_payloads.update(self.tile_delta.upserts)
        rebound_payloads = {}
        for tile_number, payload in transaction_payloads.items():
            mapping = getattr(payload, "shader_mapping", None)
            prior = getattr(payload, "presentation_identity", None)
            identity = TilePresentationIdentity(
                levels_generation=int(self.tile_delta.level_revision),
                levels=levels,
                scale=getattr(mapping, "scale", getattr(prior, "scale", None)),
                lut_identity=getattr(
                    mapping,
                    "lut_identity",
                    getattr(prior, "lut_identity", None),
                ),
            )
            rebound_payloads[int(tile_number)] = replace(
                payload,
                presentation_identity=identity,
            )
        if not rebound_payloads:
            return
        delta = replace(
            self.tile_delta,
            upserts={
                int(tile_number): rebound_payloads[int(tile_number)]
                for tile_number in self.tile_delta.upserts
            },
        )
        state_payloads = dict(self.tile_state.payloads)
        for tile_number, payload in rebound_payloads.items():
            if int(tile_number) in state_payloads:
                state_payloads[int(tile_number)] = payload
        state = TilePresentationState(
            state_payloads,
            revision=int(self.tile_state.revision),
        )
        object.__setattr__(self, "tile_delta", delta)
        object.__setattr__(self, "tile_state", state)


DisplayPresentation = DisplayTiledPresentation


@dataclass(frozen=True)
class PresentationInput:
    payload: DisplayPayload
    context: RenderRequestContext
    previous_frame: CommittedDisplayFrame | None
    window_mode: str
    force_auto: bool
    commit_kind: CommitKind
    semantic_source: Any = None
    applied_level_source: Any = None
    level_bounds: tuple[float, float] | None = None
    user_levels: tuple[float, float] | None = None


@dataclass(frozen=True)
class PresentationDecision:
    display_presentation: DisplayPresentation
    levels: tuple[float, float]
    histogram_range: tuple[float, float]
    level_source_rank: int
    level_source_key: object | None
    level_source_count: int = 0
    expected_source_count: int = 0
    allow_fast_commit: bool = False
    applied_level_source: Any = None


@dataclass(frozen=True)
class CommitPlan:
    decision: PresentationDecision
    frame_key: DisplayFrameKey
    fast: bool = False
