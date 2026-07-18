"""Qt-free geometry shared by passive overlay renderer backends."""

from __future__ import annotations


def roi_outline_points(geometry) -> tuple[tuple[float, float], ...]:
    """Return the ordered world-space outline for one semantic ROI."""

    kind = str(
        getattr(getattr(geometry, "kind", ""), "value", getattr(geometry, "kind", ""))
    )
    if kind == "rectangle":
        rect = getattr(geometry, "rect", None)
        if rect is None:
            return ()
        x, y, width, height = (float(value) for value in rect)
        return (
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
            (x, y),
        )
    points = tuple(
        (float(point[0]), float(point[1]))
        for point in tuple(getattr(geometry, "points", ()) or ())
    )
    if kind == "line" and len(points) >= 2:
        return points[:2]
    if kind in {"polyline", "freehand_polygon"} and len(points) >= 2:
        return points
    return ()


def montage_overlay_rgba(overlay):
    """Return fill, border, and status-mark RGBA for one tile state."""

    if str(getattr(overlay, "state", "")) == "skipped":
        return (
            _rgba255(130, 70, 20, 95),
            _rgba255(210, 130, 60, 180),
            _rgba255(245, 245, 245, 230),
        )
    return (
        _rgba255(35, 35, 35, 95),
        _rgba255(170, 170, 170, 140),
        _rgba255(245, 245, 245, 230),
    )


def montage_overlay_status_segments(overlay):
    """Return the line-segment symbol for loading or skipped state."""

    x = float(getattr(overlay, "x", 0.0))
    y = float(getattr(overlay, "y", 0.0))
    width = float(max(1.0, getattr(overlay, "width", 1.0)))
    height = float(max(1.0, getattr(overlay, "height", 1.0)))
    tile_extent = min(width, height)
    size = max(tile_extent * 0.08, min(tile_extent * 0.18, 0.35))
    cx = x + width * 0.5
    cy = y + height * 0.5
    if str(getattr(overlay, "state", "")) == "skipped":
        return (
            ((cx - size * 0.5, cy - size * 0.5), (cx + size * 0.5, cy + size * 0.5)),
            ((cx - size * 0.5, cy + size * 0.5), (cx + size * 0.5, cy - size * 0.5)),
        )
    return (
        ((cx, cy), (cx, cy - size * 0.32)),
        ((cx, cy), (cx + size * 0.28, cy)),
    )


def _rgba255(r, g, b, a):
    return (
        float(r) / 255.0,
        float(g) / 255.0,
        float(b) / 255.0,
        float(a) / 255.0,
    )
