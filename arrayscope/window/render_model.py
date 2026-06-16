"""Qt-free render presentation request and commit models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from arrayscope.display.geometry import DisplayGeometry
from arrayscope.display.slice_engine import DisplayImage
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.window.display_frame import CommittedDisplayFrame, DisplayFrameKey


class CommitKind(Enum):
    FULL_NORMAL = "full_normal"
    FULL_MONTAGE_INITIAL = "full_montage_initial"
    PROGRESSIVE_MONTAGE_PATCH = "progressive_montage_patch"
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
    rgb_already_windowed: bool = False
    histogram_plot_data: np.ndarray | None = None

    @property
    def data(self) -> np.ndarray:
        return self.image.data

    @property
    def histogram_data(self) -> np.ndarray | None:
        return self.image.histogram_data


@dataclass(frozen=True)
class DisplayPresentation:
    data: np.ndarray
    histogram_data: np.ndarray | None
    histogram_plot_data: np.ndarray | None
    geometry: DisplayGeometry
    levels: tuple[float, float]
    histogram_range: tuple[float, float]
    viewport_policy: ViewportPolicy
    rgb_already_windowed: bool = False


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
