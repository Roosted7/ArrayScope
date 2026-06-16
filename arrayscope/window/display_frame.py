"""Committed visible display frame state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arrayscope.display.geometry import DisplayGeometry


@dataclass(frozen=True)
class DisplayFrameKey:
    document_key: object
    request_key: object
    render_generation: int
    semantic_key: object | None = None


@dataclass(frozen=True)
class CommittedDisplayFrame:
    data: np.ndarray
    histogram_data: np.ndarray | None
    geometry: DisplayGeometry
    levels: tuple[float, float]
    histogram_range: tuple[float, float]
    key: DisplayFrameKey

