"""Pure ROI geometry, sampling, and statistics helpers."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from scipy import ndimage

DEFAULT_LINE_WIDTH = 1.0
DEFAULT_FREEHAND_SIMPLIFY_TOLERANCE = 1.5
MIN_POLYLINE_POINTS = 2
MIN_FREEHAND_POINTS = 3


class RoiKind(Enum):
    LINE = "line"
    RECTANGLE = "rectangle"
    POLYLINE = "polyline"
    FREEHAND_POLYGON = "freehand_polygon"


@dataclass(frozen=True)
class RoiGeometry:
    kind: RoiKind | str
    points: tuple[tuple[float, float], ...] = ()
    rect: tuple[float, float, float, float] | None = None
    line_width: float = DEFAULT_LINE_WIDTH
    closed: bool = False
    image_axes: tuple[int, int] = (0, 1)

    def __post_init__(self):
        kind = self.kind if isinstance(self.kind, RoiKind) else RoiKind(str(self.kind))
        points = tuple((float(x), float(y)) for x, y in self.points)
        rect = None if self.rect is None else tuple(float(value) for value in self.rect)
        if rect is not None and len(rect) != 4:
            raise ValueError("rect must be a 4-tuple")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "rect", rect)
        object.__setattr__(self, "line_width", max(1.0, float(self.line_width)))
        object.__setattr__(self, "closed", bool(self.closed or kind == RoiKind.FREEHAND_POLYGON))
        object.__setattr__(self, "image_axes", tuple(int(axis) for axis in self.image_axes))


@dataclass(frozen=True)
class RoiSelection:
    id: str
    label: str
    geometry: RoiGeometry
    enabled: bool = True
    color: tuple[int, int, int] = (230, 60, 30)


@dataclass(frozen=True)
class RoiStatistics:
    count: int
    finite_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    median: float | None
    std: float | None
    sum: float | None
    rss: float | None
    p05: float | None
    p95: float | None


class RoiStatsAccumulator:
    def __init__(self, *, exact_limit: int = 250_000):
        self.exact_limit = int(exact_limit)
        self.count = 0
        self.finite_count = 0
        self.minimum = None
        self.maximum = None
        self.sum = 0.0
        self.sumsq = 0.0
        self._exact_values: list[np.ndarray] | None = []

    def add_values(self, values) -> None:
        flat = np.asarray(values).ravel()
        self.count += int(flat.size)
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return
        if np.iscomplexobj(finite):
            finite = np.abs(finite)
        finite = finite.astype(float, copy=False)
        self.finite_count += int(finite.size)
        chunk_min = float(np.min(finite))
        chunk_max = float(np.max(finite))
        self.minimum = chunk_min if self.minimum is None else min(float(self.minimum), chunk_min)
        self.maximum = chunk_max if self.maximum is None else max(float(self.maximum), chunk_max)
        self.sum += float(np.sum(finite))
        self.sumsq += float(np.sum(np.abs(finite) ** 2))
        if self._exact_values is not None:
            retained = sum(int(chunk.size) for chunk in self._exact_values)
            if retained + int(finite.size) <= self.exact_limit:
                self._exact_values.append(finite.copy())
            else:
                self._exact_values = None

    def result(self) -> RoiStatistics:
        if self.finite_count == 0:
            return RoiStatistics(
                count=int(self.count),
                finite_count=0,
                minimum=None,
                maximum=None,
                mean=None,
                median=None,
                std=None,
                sum=None,
                rss=None,
                p05=None,
                p95=None,
            )
        mean = float(self.sum / self.finite_count)
        variance = max(0.0, float(self.sumsq / self.finite_count) - mean**2)
        exact = None
        if self._exact_values is not None and self._exact_values:
            exact = np.concatenate(self._exact_values)
        return RoiStatistics(
            count=int(self.count),
            finite_count=int(self.finite_count),
            minimum=None if self.minimum is None else float(self.minimum),
            maximum=None if self.maximum is None else float(self.maximum),
            mean=mean,
            median=None if exact is None else float(np.median(exact)),
            std=float(np.sqrt(variance)),
            sum=float(self.sum),
            rss=float(np.sqrt(self.sumsq)),
            p05=None if exact is None else float(np.percentile(exact, 5)),
            p95=None if exact is None else float(np.percentile(exact, 95)),
        )


def close_polygon(points):
    points = tuple((float(x), float(y)) for x, y in points)
    if not points:
        return ()
    if points[0] == points[-1]:
        return points
    return (*points, points[0])


def simplify_polyline(points, tolerance=DEFAULT_FREEHAND_SIMPLIFY_TOLERANCE):
    points = tuple((float(x), float(y)) for x, y in points)
    if len(points) <= 2:
        return points
    tolerance = float(tolerance)
    if tolerance <= 0:
        return points
    return tuple(_rdp(np.asarray(points, dtype=float), tolerance))


def polygon_roi_mask(shape_2d, points):
    height, width = (int(shape_2d[0]), int(shape_2d[1]))
    polygon = close_polygon(points)
    if height <= 0 or width <= 0 or len(polygon) < 4:
        return np.zeros((max(0, height), max(0, width)), dtype=bool)

    yy, xx = np.mgrid[0:height, 0:width]
    x = xx.astype(float)
    y = yy.astype(float)
    mask = np.zeros((height, width), dtype=bool)
    poly = np.asarray(polygon, dtype=float)
    px = poly[:, 0]
    py = poly[:, 1]
    for i in range(len(poly) - 1):
        x0, y0 = px[i], py[i]
        x1, y1 = px[i + 1], py[i + 1]
        crosses = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-300) + x0)
        mask ^= crosses
    return mask


def polyline_roi_samples(image_2d, points, width=DEFAULT_LINE_WIDTH):
    image = _as_scalar_image(image_2d)
    points = tuple((float(x), float(y)) for x, y in points)
    if len(points) < MIN_POLYLINE_POINTS:
        return np.asarray([], dtype=float)

    segments = []
    for p0, p1 in itertools.pairwise(points):
        segment = _segment_samples(image, p0, p1, width=width)
        if segment.size:
            if segments:
                segment = segment[1:]
            segments.append(segment)
    if not segments:
        return np.asarray([], dtype=float)
    return np.concatenate(segments)


def freehand_roi_values(image_2d, points):
    image = _as_scalar_image(image_2d)
    mask = polygon_roi_mask(image.shape[:2], points)
    return image[mask]


def rectangle_roi_values(image_2d, rect):
    image = _as_scalar_image(image_2d)
    x, y, width, height = (float(value) for value in rect)
    x0 = max(0, int(np.floor(min(x, x + width))))
    x1 = min(image.shape[1], int(np.ceil(max(x, x + width))))
    y0 = max(0, int(np.floor(min(y, y + height))))
    y1 = min(image.shape[0], int(np.ceil(max(y, y + height))))
    if x1 <= x0 or y1 <= y0:
        return np.asarray([], dtype=image.dtype)
    return image[y0:y1, x0:x1].ravel()


def roi_values(image_2d, geometry: RoiGeometry):
    geometry = geometry if isinstance(geometry, RoiGeometry) else RoiGeometry(**geometry)
    if geometry.kind == RoiKind.RECTANGLE:
        if geometry.rect is None:
            return np.asarray([], dtype=float)
        return rectangle_roi_values(image_2d, geometry.rect)
    if geometry.kind == RoiKind.LINE:
        return polyline_roi_samples(image_2d, geometry.points[:2], width=geometry.line_width)
    if geometry.kind == RoiKind.POLYLINE:
        return polyline_roi_samples(image_2d, geometry.points, width=geometry.line_width)
    if geometry.kind == RoiKind.FREEHAND_POLYGON:
        return freehand_roi_values(image_2d, close_polygon(geometry.points))
    raise ValueError(f"unsupported ROI kind: {geometry.kind}")


def roi_mask(shape_2d, geometry: RoiGeometry) -> np.ndarray:
    """Rasterize one ROI as a 2-D boolean mask in its image-plane coordinates."""

    height, width = (int(shape_2d[0]), int(shape_2d[1]))
    shape = (max(0, height), max(0, width))
    geometry = geometry if isinstance(geometry, RoiGeometry) else RoiGeometry(**geometry)
    if geometry.kind == RoiKind.RECTANGLE:
        mask = np.zeros(shape, dtype=bool)
        if geometry.rect is None:
            return mask
        x, y, rect_width, rect_height = geometry.rect
        x0 = max(0, int(np.floor(min(x, x + rect_width))))
        x1 = min(width, int(np.ceil(max(x, x + rect_width))))
        y0 = max(0, int(np.floor(min(y, y + rect_height))))
        y1 = min(height, int(np.ceil(max(y, y + rect_height))))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
        return mask
    if geometry.kind == RoiKind.FREEHAND_POLYGON:
        return polygon_roi_mask(shape, geometry.points)

    mask = np.zeros(shape, dtype=bool)
    points = geometry.points[:2] if geometry.kind == RoiKind.LINE else geometry.points
    for p0, p1 in itertools.pairwise(points):
        distance = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        count = max(2, int(np.ceil(distance)) + 1)
        xs = np.rint(np.linspace(p0[0], p1[0], count)).astype(int)
        ys = np.rint(np.linspace(p0[1], p1[1], count)).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        mask[ys[valid], xs[valid]] = True
    radius = max(0, int(np.ceil(float(geometry.line_width) * 0.5)) - 1)
    if radius:
        mask = ndimage.binary_dilation(mask, iterations=radius)
    return mask


def roi_coordinates(geometry: RoiGeometry) -> np.ndarray:
    """Represent one ROI as an ``N x 2`` float64 coordinate array."""

    geometry = geometry if isinstance(geometry, RoiGeometry) else RoiGeometry(**geometry)
    if geometry.kind == RoiKind.RECTANGLE:
        if geometry.rect is None:
            return np.empty((0, 2), dtype=np.float64)
        x, y, width, height = geometry.rect
        points = (
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        )
    else:
        points = geometry.points
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def roi_values_for_region(
    image_2d, geometry: RoiGeometry, *, offset: tuple[float, float] = (0.0, 0.0)
):
    return roi_values(
        image_2d, translate_roi_geometry(geometry, -float(offset[0]), -float(offset[1]))
    )


def translate_roi_geometry(geometry: RoiGeometry, dx: float, dy: float) -> RoiGeometry:
    geometry = geometry if isinstance(geometry, RoiGeometry) else RoiGeometry(**geometry)
    dx = float(dx)
    dy = float(dy)
    points = tuple((x + dx, y + dy) for x, y in geometry.points)
    rect = None
    if geometry.rect is not None:
        x, y, width, height = geometry.rect
        rect = (x + dx, y + dy, width, height)
    return replace(geometry, points=points, rect=rect)


def clamp_roi_geometry_to_rect(
    geometry: RoiGeometry,
    rect: tuple[float, float, float, float],
) -> RoiGeometry:
    """Translate ROI geometry just enough for its bounds to touch or fit a rect."""

    geometry = geometry if isinstance(geometry, RoiGeometry) else RoiGeometry(**geometry)
    bounds = roi_bounding_rect(geometry)
    if bounds is None:
        return geometry
    x0, y0, x1, y1 = (float(value) for value in rect)
    rx0, rx1 = (min(x0, x1), max(x0, x1))
    ry0, ry1 = (min(y0, y1), max(y0, y1))
    bx0, by0, bx1, by1 = bounds
    dx = _clamp_span_translation(float(bx0), float(bx1), rx0, rx1)
    dy = _clamp_span_translation(float(by0), float(by1), ry0, ry1)
    if dx == 0.0 and dy == 0.0:
        return geometry
    return translate_roi_geometry(geometry, dx, dy)


def _clamp_span_translation(
    inner_min: float, inner_max: float, outer_min: float, outer_max: float
) -> float:
    if inner_max < outer_min:
        return outer_min - inner_min
    if inner_min > outer_max:
        return outer_max - inner_max
    if inner_min < outer_min and inner_max <= outer_max:
        return outer_min - inner_min
    if inner_max > outer_max and inner_min >= outer_min:
        return outer_max - inner_max
    return 0.0


def roi_bounding_rect(geometry: RoiGeometry) -> tuple[float, float, float, float] | None:
    geometry = geometry if isinstance(geometry, RoiGeometry) else RoiGeometry(**geometry)
    if geometry.kind == RoiKind.RECTANGLE and geometry.rect is not None:
        x, y, width, height = geometry.rect
        return (min(x, x + width), min(y, y + height), max(x, x + width), max(y, y + height))
    points = tuple(geometry.points)
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    pad = (
        max(0.0, float(geometry.line_width) * 0.5)
        if geometry.kind in (RoiKind.LINE, RoiKind.POLYLINE)
        else 0.0
    )
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def roi_statistics(values):
    values = np.asarray(values)
    flat = values.ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return RoiStatistics(
            count=int(flat.size),
            finite_count=0,
            minimum=None,
            maximum=None,
            mean=None,
            median=None,
            std=None,
            sum=None,
            rss=None,
            p05=None,
            p95=None,
        )
    finite = finite.astype(float, copy=False)
    return RoiStatistics(
        count=int(flat.size),
        finite_count=int(finite.size),
        minimum=float(np.min(finite)),
        maximum=float(np.max(finite)),
        mean=float(np.mean(finite)),
        median=float(np.median(finite)),
        std=float(np.std(finite)),
        sum=float(np.sum(finite)),
        rss=float(np.sqrt(np.sum(np.abs(finite) ** 2))),
        p05=float(np.percentile(finite, 5)),
        p95=float(np.percentile(finite, 95)),
    )


def _segment_samples(image, p0, p1, width=DEFAULT_LINE_WIDTH):
    x0, y0 = p0
    x1, y1 = p1
    distance = float(np.hypot(x1 - x0, y1 - y0))
    if distance == 0:
        return _sample_coordinates(image, np.asarray([x0]), np.asarray([y0]))
    count = max(2, int(np.ceil(distance)) + 1)
    xs = np.linspace(x0, x1, count)
    ys = np.linspace(y0, y1, count)
    width = max(1.0, float(width))
    if width <= 1.0:
        return _sample_coordinates(image, xs, ys)

    nx = -(y1 - y0) / distance
    ny = (x1 - x0) / distance
    offsets = np.linspace(-(width - 1.0) * 0.5, (width - 1.0) * 0.5, max(1, int(np.ceil(width))))
    samples = [_sample_coordinates(image, xs + offset * nx, ys + offset * ny) for offset in offsets]
    return np.nanmean(np.vstack(samples), axis=0)


def _sample_coordinates(image, xs, ys):
    coords = np.vstack([ys, xs])
    return ndimage.map_coordinates(image, coords, order=1, mode="nearest")


def _as_scalar_image(image_2d):
    image = np.asarray(image_2d)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        rgb = image[..., :3].astype(float)
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    if image.ndim != 2:
        raise ValueError(f"ROI sampling expects a 2D image, got shape {image.shape}")
    return image


def _rdp(points, epsilon):
    if len(points) <= 2:
        return [tuple(point) for point in points]
    start = points[0]
    end = points[-1]
    line = end - start
    norm = np.linalg.norm(line)
    if norm == 0:
        distances = np.linalg.norm(points - start, axis=1)
    else:
        offsets = start - points
        distances = np.abs((line[0] * offsets[:, 1] - line[1] * offsets[:, 0]) / norm)
    index = int(np.argmax(distances))
    if distances[index] > epsilon:
        left = _rdp(points[: index + 1], epsilon)
        right = _rdp(points[index:], epsilon)
        return left[:-1] + right
    return [tuple(start), tuple(end)]
