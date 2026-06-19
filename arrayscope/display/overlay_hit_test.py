"""Qt-free hit testing for interactive display overlays.

The interaction owner and the rendering backend must agree on what is
interactive.  Keeping the geometry tests here avoids separate, slowly
drifting PyQtGraph and VisPy interpretations of ROI handles and outlines.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import numpy as np

from arrayscope.core.roi import RoiGeometry, RoiKind, close_polygon


@dataclass(frozen=True)
class RoiHit:
    """Result of testing one ROI against a display-space point."""

    part: str
    distance: float
    handle_index: int | None = None
    segment_index: int | None = None


def roi_handle_points(geometry: RoiGeometry) -> tuple[tuple[float, float], ...]:
    """Return resize/edit handles in display coordinates.

    The rectangle contract mirrors ``pyqtgraph.RectROI``'s default scale
    handle.  Lines expose both endpoints; polylines expose their vertices.
    A repeated closing freehand point is omitted because it is the same handle
    as the first vertex.
    """

    geometry = _coerce_geometry(geometry)
    if geometry.kind == RoiKind.RECTANGLE:
        if geometry.rect is None:
            return ()
        x, y, width, height = geometry.rect
        return ((float(x + width), float(y + height)),)
    points = tuple((float(x), float(y)) for x, y in geometry.points)
    if geometry.kind == RoiKind.LINE:
        return points[:2]
    if geometry.kind in (RoiKind.POLYLINE, RoiKind.FREEHAND_POLYGON):
        if len(points) > 1 and points[0] == points[-1]:
            points = points[:-1]
        return points
    return ()


def hit_test_roi(
    geometry: RoiGeometry,
    point: tuple[float, float],
    *,
    tolerance: float,
) -> RoiHit | None:
    """Hit-test ROI handles, outline, and movable rectangle body.

    Handles have priority over outlines, and outlines have priority over the
    rectangle interior.  ``tolerance`` is expressed in display/world units and
    should normally be derived from a fixed number of screen pixels.
    """

    geometry = _coerce_geometry(geometry)
    x, y = (float(point[0]), float(point[1]))
    tolerance = max(0.0, float(tolerance))

    best_handle = _nearest_point((x, y), roi_handle_points(geometry))
    if best_handle is not None and best_handle[1] <= tolerance:
        index, distance = best_handle
        return RoiHit("handle", float(distance), handle_index=int(index))

    segments = roi_segments(geometry)
    best_segment = _nearest_segment((x, y), segments)
    if best_segment is not None and best_segment[1] <= tolerance:
        index, distance = best_segment
        return RoiHit("outline", float(distance), segment_index=int(index))

    if geometry.kind == RoiKind.RECTANGLE and geometry.rect is not None:
        rx, ry, width, height = geometry.rect
        x0, x1 = sorted((float(rx), float(rx + width)))
        y0, y1 = sorted((float(ry), float(ry + height)))
        if x0 <= x <= x1 and y0 <= y <= y1:
            return RoiHit("body", 0.0)
    return None


def roi_segments(geometry: RoiGeometry) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """Return line segments forming the interactive ROI outline."""

    geometry = _coerce_geometry(geometry)
    if geometry.kind == RoiKind.RECTANGLE:
        if geometry.rect is None:
            return ()
        x, y, width, height = geometry.rect
        points = (
            (float(x), float(y)),
            (float(x + width), float(y)),
            (float(x + width), float(y + height)),
            (float(x), float(y + height)),
            (float(x), float(y)),
        )
    else:
        points = tuple((float(x), float(y)) for x, y in geometry.points)
        if geometry.kind == RoiKind.LINE:
            points = points[:2]
        elif geometry.kind == RoiKind.FREEHAND_POLYGON:
            points = close_polygon(points)
    if len(points) < 2:
        return ()
    return tuple((points[index], points[index + 1]) for index in range(len(points) - 1))


def point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """Shortest Euclidean distance from ``point`` to a finite segment."""

    px, py = (float(point[0]), float(point[1]))
    x0, y0 = (float(start[0]), float(start[1]))
    x1, y1 = (float(end[0]), float(end[1]))
    dx = x1 - x0
    dy = y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return float(hypot(px - x0, py - y0))
    t = ((px - x0) * dx + (py - y0) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return float(hypot(px - (x0 + t * dx), py - (y0 + t * dy)))


def _nearest_point(point, points):
    if not points:
        return None
    distances = [hypot(float(point[0]) - x, float(point[1]) - y) for x, y in points]
    index = int(np.argmin(distances))
    return index, float(distances[index])


def _nearest_segment(point, segments):
    if not segments:
        return None
    distances = [point_segment_distance(point, start, end) for start, end in segments]
    index = int(np.argmin(distances))
    return index, float(distances[index])


def _coerce_geometry(geometry) -> RoiGeometry:
    if isinstance(geometry, RoiGeometry):
        return geometry
    if isinstance(geometry, dict):
        return RoiGeometry(**geometry)
    raise TypeError("ROI hit testing requires RoiGeometry")
