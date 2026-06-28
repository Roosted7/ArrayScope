"""Qt-free identity and coverage policy for montage viewport updates."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from arrayscope.core.roi import RoiGeometry, RoiKind, RoiSelection
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.montage import montage_rect_for_viewport, optimal_montage_columns
from arrayscope.display.viewport import view_ranges_near


MIN_VIEW_SPAN = 1e-9


@dataclass(frozen=True)
class MontageViewportPlan:
    """Stable montage layout plus transient viewport scheduling state."""

    axis: int | None
    all_indices: tuple[int, ...]
    viewport_shape: tuple[int, int]
    tile_shape: tuple[int, int]
    plan: object
    view_range: object
    shader_display: bool
    persistent_tile_residency: bool
    priority_focus: tuple[float, float] | None = None

    def candidate_tiles(self, *, margin_tiles: int = 0, prioritize: bool = False):
        rect = montage_rect_for_viewport(
            self.plan,
            view_range=self.view_range,
            viewport_shape=self.viewport_shape,
        )
        tiles = self.plan.tiles_intersecting(
            ((rect[0], rect[2]), (rect[1], rect[3])),
            margin_tiles=max(0, int(margin_tiles)),
        )
        if prioritize:
            return prioritize_montage_tiles(
                tiles,
                view_range=((rect[0], rect[2]), (rect[1], rect[3])),
                focus=self.priority_focus,
            )
        return tiles

    def prioritize_tiles(self, tiles):
        rect = montage_rect_for_viewport(
            self.plan,
            view_range=self.view_range,
            viewport_shape=self.viewport_shape,
        )
        return prioritize_montage_tiles(
            tiles,
            view_range=((rect[0], rect[2]), (rect[1], rect[3])),
            focus=self.priority_focus,
        )


@dataclass(frozen=True)
class MontageViewportRetargetPolicy:
    """How a tiled backend handles viewport-only montage changes."""

    enabled: bool
    coverage_margin_tiles: int = 0
    near_margin_tiles: int = 0
    update_delay_ms: int = 120


@dataclass(frozen=True)
class MontageViewportReflow:
    """Pure viewport/layout reflow decision for a montage session update."""

    viewport_plan: MontageViewportPlan
    view_range_to_apply: tuple[tuple[float, float], tuple[float, float]] | None = None
    last_auto_view_range: tuple[tuple[float, float], tuple[float, float]] | None = None


@dataclass(frozen=True)
class MontageViewportIntent:
    """Canonical auto/manual facts consumed by montage viewport planning."""

    fit_locked: bool = False
    auto_active: bool = False

    @property
    def auto_like(self) -> bool:
        return bool(self.fit_locked or self.auto_active)


def montage_viewport_intent(viewport_controller, view_range) -> MontageViewportIntent:
    """Return semantic viewport intent without coupling callers to controller internals."""

    if viewport_controller is None:
        return MontageViewportIntent(auto_active=True)
    fit_locked = bool(viewport_controller.is_fit_locked())
    auto_active_fn = getattr(viewport_controller, "is_auto_active", None)
    auto_active = bool(auto_active_fn() if callable(auto_active_fn) else False)
    return MontageViewportIntent(
        fit_locked=fit_locked,
        auto_active=auto_active,
    )


def prioritize_montage_tiles(tiles, *, view_range, focus=None):
    """Return tiles ordered from normalized viewport-focus distance outward."""

    if tiles is None:
        return ()
    tiles = tuple(tiles)
    if not tiles:
        return ()
    try:
        x_range, y_range = view_range
        x0, x1 = float(x_range[0]), float(x_range[1])
        y0, y1 = float(y_range[0]), float(y_range[1])
    except Exception:
        try:
            x0 = min(float(tile.x0) for tile in tiles)
            y0 = min(float(tile.y0) for tile in tiles)
            x1 = max(float(tile.x0 + tile.width) for tile in tiles)
            y1 = max(float(tile.y0 + tile.height) for tile in tiles)
        except Exception:
            return tiles
    span_x = max(1.0, abs(float(x1) - float(x0)))
    span_y = max(1.0, abs(float(y1) - float(y0)))
    if focus is None:
        focus_x = (float(x0) + float(x1)) * 0.5
        focus_y = (float(y0) + float(y1)) * 0.5
    else:
        try:
            focus_x = float(focus[0])
            focus_y = float(focus[1])
        except (IndexError, KeyError, TypeError, ValueError):
            focus_x = (float(x0) + float(x1)) * 0.5
            focus_y = (float(y0) + float(y1)) * 0.5

    def score(tile):
        center_x = float(tile.x0) + float(tile.width) * 0.5
        center_y = float(tile.y0) + float(tile.height) * 0.5
        dx = (center_x - focus_x) / span_x
        dy = (center_y - focus_y) / span_y
        return (dx * dx + dy * dy, int(tile.montage_index))

    return tuple(sorted(tiles, key=score))


def effective_montage_columns(
    count: int,
    tile_shape: tuple[int, int],
    viewport_shape: tuple[int, int],
    *,
    requested_columns: int | None,
    fit_locked: bool = False,
    auto_active: bool = False,
) -> int | None:
    """Choose applied montage columns without rewriting semantic view state.

    Automatic layout is used when no column preference exists, when Fit owns
    the camera, when the viewport is explicitly auto-owned, or when the current
    view is still near the automatic pose.  Manual pan/zoom keeps requested
    columns.
    """

    count = max(0, int(count))
    if count < 1:
        return None
    tile_shape = (max(1, int(tile_shape[0])), max(1, int(tile_shape[1])))
    viewport_shape = (max(1, int(viewport_shape[0])), max(1, int(viewport_shape[1])))
    automatic = optimal_montage_columns(count, tile_shape, viewport_shape)
    if requested_columns is None:
        return automatic
    if bool(fit_locked) or bool(auto_active):
        return automatic
    return max(1, min(int(requested_columns), count))


def square_montage_fit_view_range(plan, viewport_shape) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return a square-pixel fit range for the full applied montage layout."""

    display_height, display_width = tuple(int(value) for value in plan.display_shape[:2])
    viewport_height, viewport_width = _normalized_viewport_shape(viewport_shape)
    content_width = max(MIN_VIEW_SPAN, float(display_width))
    content_height = max(MIN_VIEW_SPAN, float(display_height))
    viewport_aspect = float(viewport_width) / float(viewport_height)
    content_aspect = content_width / content_height
    fitted_width = content_width
    fitted_height = content_height
    if viewport_aspect > content_aspect:
        fitted_width = fitted_height * viewport_aspect
    elif viewport_aspect < content_aspect:
        fitted_height = fitted_width / viewport_aspect
    center_x = content_width * 0.5
    center_y = content_height * 0.5
    return (
        (center_x - fitted_width * 0.5, center_x + fitted_width * 0.5),
        (center_y - fitted_height * 0.5, center_y + fitted_height * 0.5),
    )


def plan_full_view_range(plan) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the exact world bounds of an applied montage plan."""

    height, width = tuple(int(value) for value in plan.display_shape[:2])
    return ((0.0, float(max(1, width))), (0.0, float(max(1, height))))


def retarget_montage_viewport_plan(
    previous_plan,
    viewport_plan: MontageViewportPlan,
    previous_viewport_shape,
    *,
    fit_locked: bool = False,
    auto_active: bool = False,
    skip_remap: bool = False,
    focus: tuple[float, float] | None = None,
) -> MontageViewportReflow:
    """Retarget a montage viewport after resize/layout reflow.

    Manual views are translated only when a same-source layout reflow moves the
    source tile underneath the focus point.  They are never refit by resize or
    column changes.  Fit and genuinely near-auto views take the fit paths.
    """

    next_plan = viewport_plan.plan
    if previous_plan is None or next_plan is None:
        return MontageViewportReflow(viewport_plan)
    layout_changed = getattr(previous_plan, "geometry", None) != getattr(next_plan, "geometry", None)
    viewport_shape_changed = tuple(previous_viewport_shape or ()) != tuple(viewport_plan.viewport_shape)
    if not layout_changed and not viewport_shape_changed:
        return MontageViewportReflow(viewport_plan)
    if viewport_plan.view_range is None or bool(skip_remap):
        return MontageViewportReflow(viewport_plan)

    current_range = _normalized_view_range(viewport_plan.view_range)
    last_auto_view_range = None
    if bool(fit_locked):
        next_range = plan_full_view_range(next_plan)
    elif bool(auto_active):
        auto_range = square_montage_fit_view_range(next_plan, viewport_plan.viewport_shape)
        next_range = auto_range
        last_auto_view_range = next_range
    else:
        next_range = _manual_montage_reflow_range(
            previous_plan,
            next_plan,
            current_range,
            previous_viewport_shape,
            viewport_plan.viewport_shape,
            focus=focus,
        )
        if next_range is None:
            return MontageViewportReflow(viewport_plan)

    retargeted = replace(viewport_plan, view_range=next_range)
    apply_range = None if view_ranges_near(current_range, next_range, tolerance_fraction=1e-9) else next_range
    return MontageViewportReflow(
        retargeted,
        view_range_to_apply=apply_range,
        last_auto_view_range=last_auto_view_range,
    )


def _manual_montage_reflow_range(
    previous_plan,
    next_plan,
    current_range,
    previous_viewport_shape,
    next_viewport_shape,
    *,
    focus: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    return remap_montage_view_range(
        previous_plan,
        next_plan,
        current_range,
        previous_viewport_shape,
        next_viewport_shape,
        focus=focus,
    )


def remap_montage_view_range(
    previous_plan,
    next_plan,
    view_range,
    previous_viewport_shape,
    next_viewport_shape,
    *,
    focus: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Transfer a manual view through a montage layout change.

    The same source-local point remains at the same screen fraction.  World
    spans scale with viewport size so the screen zoom, measured as world units
    per viewport pixel, stays constant while resize shows more or less content.
    """

    try:
        view_range = _normalized_view_range(view_range)
        x0, x1 = float(view_range[0][0]), float(view_range[0][1])
        y0, y1 = float(view_range[1][0]), float(view_range[1][1])
    except Exception:
        return None

    span_x = abs(x1 - x0)
    span_y = abs(y1 - y0)
    if span_x <= MIN_VIEW_SPAN or span_y <= MIN_VIEW_SPAN:
        return None
    next_span_x, next_span_y = _resize_preserving_screen_zoom_spans(
        span_x,
        span_y,
        previous_viewport_shape,
        next_viewport_shape,
    )

    focus_x, focus_y = _focus_or_center(view_range, focus)
    if getattr(previous_plan, "geometry", None) == getattr(next_plan, "geometry", None):
        next_focus_x, next_focus_y = focus_x, focus_y
    else:
        anchor = _tile_local_anchor(previous_plan, view_range, (focus_x, focus_y), allow_nearest=True)
        if anchor is None:
            return None
        source_index, local_x, local_y = anchor
        next_tile = _tile_for_source(next_plan, source_index)
        if next_tile is None:
            return None
        next_focus_x = float(next_tile.x0) + min(max(0.0, local_x), float(next_tile.width))
        next_focus_y = float(next_tile.y0) + min(max(0.0, local_y), float(next_tile.height))
    fraction_x = min(1.0, max(0.0, (focus_x - min(x0, x1)) / span_x))
    fraction_y = min(1.0, max(0.0, (focus_y - min(y0, y1)) / span_y))
    return (
        (next_focus_x - fraction_x * next_span_x, next_focus_x + (1.0 - fraction_x) * next_span_x),
        (next_focus_y - fraction_y * next_span_y, next_focus_y + (1.0 - fraction_y) * next_span_y),
    )


def remap_montage_roi_geometry(previous_plan, next_plan, geometry: RoiGeometry) -> RoiGeometry | None:
    """Move ROI geometry through a same-source montage layout reflow.

    Returns ``None`` when the source population changed.  Rectangle ROIs keep
    their shape and move by a source-local anchor, while point-based ROIs move
    each point by the source tile underneath that point.
    """

    previous_layout = _source_layout(previous_plan)
    next_layout = _source_layout(next_plan)
    return _remap_montage_roi_geometry_with_layout(previous_layout, next_layout, geometry)


def remap_montage_roi_selections(previous_plan, next_plan, selections) -> tuple[RoiSelection, ...]:
    """Return ROI selections remapped through a same-source montage layout reflow.

    The old/new source layout maps are built once per transition.  Geometry
    that cannot be represented cleanly in the existing ROI kind is left
    unchanged.
    """

    previous_layout = _source_layout(previous_plan)
    next_layout = _source_layout(next_plan)
    remapped = []
    for selection in tuple(selections or ()):
        if not isinstance(selection, RoiSelection):
            remapped.append(selection)
            continue
        geometry = _remap_montage_roi_geometry_with_layout(previous_layout, next_layout, selection.geometry)
        if geometry is not None and geometry != selection.geometry:
            remapped.append(replace(selection, geometry=geometry))
        else:
            remapped.append(selection)
    return tuple(remapped)


def _remap_montage_roi_geometry_with_layout(previous_layout, next_layout, geometry: RoiGeometry) -> RoiGeometry | None:
    geometry = geometry if isinstance(geometry, RoiGeometry) else RoiGeometry(**geometry)
    if previous_layout.source_indices != next_layout.source_indices:
        return None
    if geometry.kind == RoiKind.RECTANGLE:
        return _remap_rectangle_roi(previous_layout, next_layout, geometry)
    if geometry.kind in (RoiKind.LINE, RoiKind.POLYLINE, RoiKind.FREEHAND_POLYGON):
        remapped = _remap_points(previous_layout, next_layout, geometry.points)
        if remapped is None:
            return None
        return replace(geometry, points=remapped)
    return None


def montage_session_key(document_key, view_state, viewport_plan: MontageViewportPlan, colormap_lut) -> tuple[object, ...]:
    """Stable render-session identity, excluding transient viewport coverage."""

    if viewport_plan.axis is None:
        scope_state = view_state.with_montage_axis(None, columns=None, indices=None, text=None)
        axis_key = None
    else:
        scope_state = view_state.with_montage_axis(
            viewport_plan.axis,
            columns=None,
            indices=None,
            text=None,
        )
        axis_key = int(viewport_plan.axis)
    lut_key = None if colormap_lut is None else np.asarray(colormap_lut).tobytes()
    return (
        "montage_tiles",
        document_key,
        scope_state,
        axis_key,
        tuple(int(index) for index in viewport_plan.all_indices),
        tuple(int(value) for value in viewport_plan.tile_shape),
        int(viewport_plan.plan.gap),
        lut_key,
        bool(viewport_plan.shader_display),
    )


def montage_viewport_retarget_policy(capabilities, display_mode: str) -> MontageViewportRetargetPolicy:
    """Return viewport-retarget behaviour for the current presentation surface.

    Montage retargeting updates the tiled presentation when only camera/view
    coverage changes. Persistent residency backends additionally keep
    near-viewport warm coverage.
    """

    if bool(getattr(capabilities, "persistent_tile_residency", False)):
        return MontageViewportRetargetPolicy(
            enabled=True,
            coverage_margin_tiles=1,
            near_margin_tiles=2,
            update_delay_ms=90,
        )
    return MontageViewportRetargetPolicy(enabled=True)


def _normalized_view_range(view_range) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(view_range[0][0]), float(view_range[0][1])),
        (float(view_range[1][0]), float(view_range[1][1])),
    )


def _normalized_viewport_shape(viewport_shape) -> tuple[int, int]:
    return (max(1, int(viewport_shape[0])), max(1, int(viewport_shape[1])))


def _resize_preserving_screen_zoom_spans(
    span_x: float,
    span_y: float,
    previous_viewport_shape,
    next_viewport_shape,
) -> tuple[float, float]:
    try:
        previous_height, previous_width = _normalized_viewport_shape(previous_viewport_shape)
        next_height, next_width = _normalized_viewport_shape(next_viewport_shape)
    except Exception:
        return (float(span_x), float(span_y))
    return (
        max(MIN_VIEW_SPAN, float(span_x) * float(next_width) / float(previous_width)),
        max(MIN_VIEW_SPAN, float(span_y) * float(next_height) / float(previous_height)),
    )


def _focus_or_center(view_range, focus) -> tuple[float, float]:
    if focus is not None:
        try:
            return (float(focus[0]), float(focus[1]))
        except (IndexError, KeyError, TypeError, ValueError):
            pass
    return (
        (float(view_range[0][0]) + float(view_range[0][1])) * 0.5,
        (float(view_range[1][0]) + float(view_range[1][1])) * 0.5,
    )


def _tile_local_anchor(plan, view_range, focus, *, allow_nearest: bool = True) -> tuple[int, float, float] | None:
    tile = _tile_at_point(plan, focus)
    if tile is None and allow_nearest:
        tile = _nearest_visible_tile(plan, view_range, focus)
    if tile is None:
        return None
    local_x = float(focus[0]) - float(tile.x0)
    local_y = float(focus[1]) - float(tile.y0)
    return (int(tile.source_index), local_x, local_y)


def _tile_at_point(plan, point):
    tile_at = getattr(plan, "tile_at", None)
    if callable(tile_at):
        try:
            tile = tile_at(int(point[0]), int(point[1]))
            if tile is not None:
                return tile
        except Exception:
            pass
    try:
        x = float(point[0])
        y = float(point[1])
    except Exception:
        return None
    for tile in tuple(getattr(plan, "tiles", ()) or ()):
        x0 = float(tile.x0)
        y0 = float(tile.y0)
        if x0 <= x < x0 + float(tile.width) and y0 <= y < y0 + float(tile.height):
            return tile
    return None


def _tile_for_source(plan, source_index: int):
    source_index = int(source_index)
    for tile in tuple(getattr(plan, "tiles", ()) or ()):
        if int(tile.source_index) == source_index:
            return tile
    return None


@dataclass(frozen=True)
class _SourceLayout:
    plan: object
    source_indices: tuple[int, ...]
    tiles_by_source: dict[int, object]


def _source_layout(plan) -> _SourceLayout:
    tiles = tuple(getattr(plan, "tiles", ()) or ())
    return _SourceLayout(
        plan=plan,
        source_indices=tuple(int(tile.source_index) for tile in tiles),
        tiles_by_source={int(tile.source_index): tile for tile in tiles},
    )


def _remap_points(previous_layout, next_layout, points):
    remapped = []
    for point in tuple(points or ()):
        mapped = _remap_point(previous_layout, next_layout, point)
        if mapped is None:
            return None
        remapped.append(mapped)
    return tuple(remapped)


def _remap_point(previous_layout, next_layout, point) -> tuple[float, float] | None:
    anchor = _tile_local_anchor(previous_layout.plan, _point_view_range(point), point)
    if anchor is None:
        return None
    source_index, local_x, local_y = anchor
    next_tile = next_layout.tiles_by_source.get(int(source_index))
    if next_tile is None:
        return None
    return (
        float(next_tile.x0) + float(local_x),
        float(next_tile.y0) + float(local_y),
    )


def _remap_rectangle_roi(previous_layout, next_layout, geometry: RoiGeometry) -> RoiGeometry | None:
    if geometry.rect is None:
        return None
    x, y, width, height = (float(value) for value in geometry.rect)
    remapped_anchor = _remap_point(previous_layout, next_layout, (x, y))
    if remapped_anchor is None:
        return None
    return replace(geometry, rect=(remapped_anchor[0], remapped_anchor[1], width, height))


def _point_view_range(point) -> tuple[tuple[float, float], tuple[float, float]]:
    x = float(point[0])
    y = float(point[1])
    return ((x - 0.5, x + 0.5), (y - 0.5, y + 0.5))


def _nearest_visible_tile(plan, view_range, focus):
    tile = _nearest_tile_by_grid(plan, focus)
    if tile is not None:
        return tile
    tiles_intersecting = getattr(plan, "tiles_intersecting", None)
    if callable(tiles_intersecting):
        try:
            candidates = tuple(tiles_intersecting(view_range, margin_tiles=0))
        except Exception:
            candidates = ()
    else:
        candidates = ()
    if not candidates:
        candidates = tuple(getattr(plan, "tiles", ()) or ())
    if not candidates:
        return None
    focus_x, focus_y = float(focus[0]), float(focus[1])

    def distance(tile) -> tuple[float, int]:
        closest_x = min(max(focus_x, float(tile.x0)), float(tile.x0 + tile.width))
        closest_y = min(max(focus_y, float(tile.y0)), float(tile.y0 + tile.height))
        dx = focus_x - closest_x
        dy = focus_y - closest_y
        return (dx * dx + dy * dy, int(tile.montage_index))

    return min(candidates, key=distance)


def _nearest_tile_by_grid(plan, focus):
    try:
        x = float(focus[0])
        y = float(focus[1])
        tile_height, tile_width = tuple(int(value) for value in getattr(plan, "tile_shape"))
        gap = max(0, int(getattr(plan, "gap")))
        columns = max(1, int(getattr(plan, "columns")))
        rows = max(1, int(getattr(plan, "rows")))
        tiles = tuple(getattr(plan, "tiles", ()) or ())
    except Exception:
        return None
    if tile_width <= 0 or tile_height <= 0 or not tiles:
        return None
    stride_x = tile_width + gap
    stride_y = tile_height + gap
    if stride_x <= 0 or stride_y <= 0:
        return None
    col = _nearest_grid_index(x, tile_width, stride_x, columns)
    row = _nearest_grid_index(y, tile_height, stride_y, rows)
    tile_number = row * columns + col
    if tile_number < 0 or tile_number >= len(tiles):
        return None
    return tiles[tile_number]


def _nearest_grid_index(value: float, tile_size: int, stride: int, count: int) -> int:
    if value <= 0.0:
        return 0
    raw = int(value // float(stride))
    local = value - float(raw * stride)
    if local >= float(tile_size) + max(0.0, float(stride - tile_size)) * 0.5:
        raw += 1
    return max(0, min(int(count) - 1, int(raw)))


def montage_viewport_update_delay_ms(window) -> int:
    """Delay expensive tile discovery while camera motion stays immediate."""

    try:
        capabilities = image_view_backend_capabilities(window.img_view)
    except Exception:
        capabilities = None
    try:
        mode = str(window.img_view.montageDisplayMode())
    except Exception:
        mode = ""
    return montage_viewport_retarget_policy(capabilities, mode).update_delay_ms
