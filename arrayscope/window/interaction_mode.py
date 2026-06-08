"""Single owner enum for canvas interaction modes."""

from __future__ import annotations

from enum import Enum


class InteractionMode(Enum):
    CURSOR = "cursor"
    LIVE_PROFILE = "live_profile"
    ROI_LINE = "roi_line"
    ROI_RECTANGLE = "roi_rectangle"
    ROI_POLYLINE = "roi_polyline"
    ROI_FREEHAND = "roi_freehand"
