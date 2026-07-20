"""Qt-free hit testing for interactive display overlays.

The interaction owner and the rendering backend must agree on what is
interactive.  Keeping the geometry tests here avoids separate, slowly
drifting PyQtGraph and VisPy interpretations of ROI handles and outlines.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, hypot

from arrayscope.core.roi import RoiGeometry, RoiKind, close_polygon, roi_bounding_rect


@dataclass(frozen=True)
class RoiHit:
    """Result of testing one ROI against a display-space point."""

    part: str
    distance: float
    handle_index: int | None = None
    segment_index: int | None = None


@dataclass(frozen=True)
class _IndexedRoi:
    selection: object
    order: int
    bounds: tuple[float, float, float, float]
    cells: frozenset[tuple[int, int]]
    global_entry: bool = False


class RoiHitIndex:
    """Small spatial index for interactive ROI hit candidates.

    The index is deliberately approximate: it finds a small topmost-ordered
    candidate set, while exact handle/outline/body checks remain in
    ``hit_test_roi``. Very large ROIs are tracked as global candidates to avoid
    filling thousands of grid cells during interactive drags.
    """

    def __init__(self, *, cell_size: float = 64.0, max_cells_per_roi: int = 512):
        self._cell_size = max(1.0, float(cell_size))
        self._max_cells_per_roi = max(1, int(max_cells_per_roi))
        self._entries: dict[str, _IndexedRoi] = {}
        self._cells: dict[tuple[int, int], set[str]] = {}
        self._global_ids: set[str] = set()
        self._next_order = 0

    def upsert(self, selection: object) -> None:
        roi_id = _selection_id(selection)
        if roi_id is None:
            return
        previous = self._entries.get(roi_id)
        order = self._next_order if previous is None else previous.order
        if previous is None:
            self._next_order += 1
        else:
            self.remove(roi_id)
        geometry = getattr(selection, "geometry", None)
        bounds = None if geometry is None else roi_bounding_rect(geometry)
        if bounds is None:
            return
        cells, global_entry = self._cells_for_bounds(bounds)
        entry = _IndexedRoi(
            selection=selection,
            order=order,
            bounds=bounds,
            cells=frozenset(cells),
            global_entry=global_entry,
        )
        self._entries[roi_id] = entry
        if global_entry:
            self._global_ids.add(roi_id)
            return
        for cell in cells:
            self._cells.setdefault(cell, set()).add(roi_id)

    def remove(self, roi_id: object) -> None:
        roi_id = str(roi_id)
        entry = self._entries.pop(roi_id, None)
        if entry is None:
            return
        self._global_ids.discard(roi_id)
        for cell in entry.cells:
            ids = self._cells.get(cell)
            if ids is None:
                continue
            ids.discard(roi_id)
            if not ids:
                self._cells.pop(cell, None)

    def clear(self) -> None:
        self._entries.clear()
        self._cells.clear()
        self._global_ids.clear()

    def candidates(self, point: tuple[float, float], *, tolerance: float) -> tuple[object, ...]:
        x, y = (float(point[0]), float(point[1]))
        tolerance = max(0.0, float(tolerance))
        ids: set[str] = set(self._global_ids)
        x0, y0, x1, y1 = (x - tolerance, y - tolerance, x + tolerance, y + tolerance)
        for cell in self._cell_range((x0, y0, x1, y1)):
            ids.update(self._cells.get(cell, ()))
        entries = []
        for roi_id in ids:
            entry = self._entries.get(roi_id)
            if entry is not None and _bounds_intersect(entry.bounds, (x0, y0, x1, y1)):
                entries.append(entry)
        entries.sort(key=lambda entry: entry.order, reverse=True)
        return tuple(entry.selection for entry in entries)

    def _cells_for_bounds(
        self,
        bounds: tuple[float, float, float, float],
    ) -> tuple[tuple[tuple[int, int], ...], bool]:
        cells = []
        for cell in self._cell_range(bounds):
            cells.append(cell)
            if len(cells) > self._max_cells_per_roi:
                return (), True
        return tuple(cells), False

    def _cell_range(self, bounds: tuple[float, float, float, float]):
        x0, y0, x1, y1 = (float(value) for value in bounds)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        ix0 = floor(x0 / self._cell_size)
        ix1 = floor(x1 / self._cell_size)
        iy0 = floor(y0 / self._cell_size)
        iy1 = floor(y1 / self._cell_size)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                yield (ix, iy)


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


def roi_segments(
    geometry: RoiGeometry,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
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
    best_index = 0
    best_distance = float("inf")
    px, py = (float(point[0]), float(point[1]))
    for index, (x, y) in enumerate(points):
        distance = hypot(px - float(x), py - float(y))
        if distance < best_distance:
            best_index = int(index)
            best_distance = float(distance)
    return best_index, best_distance


def _nearest_segment(point, segments):
    if not segments:
        return None
    best_index = 0
    best_distance = float("inf")
    for index, (start, end) in enumerate(segments):
        distance = point_segment_distance(point, start, end)
        if distance < best_distance:
            best_index = int(index)
            best_distance = float(distance)
    return best_index, best_distance


def _coerce_geometry(geometry) -> RoiGeometry:
    if isinstance(geometry, RoiGeometry):
        return geometry
    if isinstance(geometry, dict):
        return RoiGeometry(**geometry)
    if hasattr(geometry, "kind") and (hasattr(geometry, "points") or hasattr(geometry, "rect")):
        kind = geometry.kind
        return RoiGeometry(
            kind=getattr(kind, "value", kind),
            points=tuple(getattr(geometry, "points", ()) or ()),
            rect=getattr(geometry, "rect", None),
            line_width=float(getattr(geometry, "line_width", 1.0)),
            closed=bool(getattr(geometry, "closed", False)),
            image_axes=tuple(getattr(geometry, "image_axes", (0, 1)) or (0, 1)),
        )
    raise TypeError("ROI hit testing requires RoiGeometry")


def _selection_id(selection: object) -> str | None:
    roi_id = str(getattr(selection, "id", "") or "")
    return roi_id or None


def _bounds_intersect(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    ax0, ay0, ax1, ay1 = (float(value) for value in a)
    bx0, by0, bx1, by1 = (float(value) for value in b)
    return max(ax0, bx0) <= min(ax1, bx1) and max(ay0, by0) <= min(ay1, by1)
