"""Qt-free identity and coverage policy for montage viewport updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.montage import montage_rect_for_viewport, optimal_montage_columns


MIN_VIEW_SPAN = 1e-9


@dataclass(frozen=True)
class MontageViewportPlan:
    """Stable montage layout plus transient viewport scheduling state."""

    axis: int
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
    near_auto: bool,
    fit_locked: bool = False,
) -> int | None:
    """Choose applied montage columns without rewriting semantic view state.

    Explicit Fit keeps the user's requested column count.  Otherwise, automatic
    layout is used when no column preference exists or when the current view is
    still near the automatic pose; manual pan/zoom keeps the requested columns.
    """

    count = max(0, int(count))
    if count < 1:
        return None
    tile_shape = (max(1, int(tile_shape[0])), max(1, int(tile_shape[1])))
    viewport_shape = (max(1, int(viewport_shape[0])), max(1, int(viewport_shape[1])))
    automatic = optimal_montage_columns(count, tile_shape, viewport_shape)
    if requested_columns is None:
        return automatic
    if bool(fit_locked):
        return max(1, min(int(requested_columns), count))
    if bool(near_auto):
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

    The same tile-local point remains at the same screen fraction, while the
    world span follows the viewport size so a window resize does not silently
    change the user's zoom density.
    """

    try:
        view_range = _normalized_view_range(view_range)
        previous_viewport_height, previous_viewport_width = _normalized_viewport_shape(previous_viewport_shape)
        next_viewport_height, next_viewport_width = _normalized_viewport_shape(next_viewport_shape)
        x0, x1 = float(view_range[0][0]), float(view_range[0][1])
        y0, y1 = float(view_range[1][0]), float(view_range[1][1])
    except Exception:
        return None

    span_x = abs(x1 - x0)
    span_y = abs(y1 - y0)
    if span_x <= MIN_VIEW_SPAN or span_y <= MIN_VIEW_SPAN:
        return None

    focus_x, focus_y = _focus_or_center(view_range, focus)
    anchor = _tile_local_anchor(previous_plan, view_range, (focus_x, focus_y))
    if anchor is None:
        return None
    tile_number, local_x, local_y = anchor
    next_tiles = tuple(getattr(next_plan, "tiles", ()) or ())
    if tile_number < 0 or tile_number >= len(next_tiles):
        return None

    next_tile = next_tiles[tile_number]
    next_focus_x = float(next_tile.x0) + min(max(0.0, local_x), float(next_tile.width))
    next_focus_y = float(next_tile.y0) + min(max(0.0, local_y), float(next_tile.height))
    fraction_x = min(1.0, max(0.0, (focus_x - min(x0, x1)) / span_x))
    fraction_y = min(1.0, max(0.0, (focus_y - min(y0, y1)) / span_y))
    world_per_pixel = max(
        span_x / float(previous_viewport_width),
        span_y / float(previous_viewport_height),
        MIN_VIEW_SPAN,
    )
    next_span_x = world_per_pixel * float(next_viewport_width)
    next_span_y = world_per_pixel * float(next_viewport_height)
    return (
        (next_focus_x - fraction_x * next_span_x, next_focus_x + (1.0 - fraction_x) * next_span_x),
        (next_focus_y - fraction_y * next_span_y, next_focus_y + (1.0 - fraction_y) * next_span_y),
    )


def montage_session_key(document_key, view_state, viewport_plan: MontageViewportPlan, colormap_lut) -> tuple[object, ...]:
    """Stable render-session identity, excluding transient viewport coverage."""

    scope_state = view_state.with_montage_axis(
        viewport_plan.axis,
        columns=None,
        indices=None,
        text=None,
    )
    lut_key = None if colormap_lut is None else np.asarray(colormap_lut).tobytes()
    return (
        "montage_tiles",
        document_key,
        scope_state,
        int(viewport_plan.axis),
        tuple(int(index) for index in viewport_plan.all_indices),
        tuple(int(value) for value in viewport_plan.tile_shape),
        int(viewport_plan.plan.gap),
        lut_key,
        bool(viewport_plan.shader_display),
    )


def montage_viewport_retarget_policy(capabilities, display_mode: str) -> MontageViewportRetargetPolicy:
    """Return viewport-retarget behaviour for the current presentation surface.

    Direct tiled backends can retarget the current montage session when only
    camera/view coverage changes.  Persistent residency backends additionally
    keep near-viewport warm coverage; PyQtGraph's direct tile layer does not,
    but it still must not restart the semantic render session for fit/pan.
    """

    mode = str(display_mode or "")
    if "tile_layer" not in mode or not bool(getattr(capabilities, "direct_montage_tile_payloads", False)):
        return MontageViewportRetargetPolicy(enabled=False)
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


def _tile_local_anchor(plan, view_range, focus) -> tuple[int, float, float] | None:
    tile = None
    tile_at = getattr(plan, "tile_at", None)
    if callable(tile_at):
        try:
            tile = tile_at(int(focus[0]), int(focus[1]))
        except Exception:
            tile = None
    if tile is None:
        tile = _nearest_visible_tile(plan, view_range, focus)
    if tile is None:
        return None
    local_x = float(focus[0]) - float(tile.x0)
    local_y = float(focus[1]) - float(tile.y0)
    return (int(tile.montage_index), local_x, local_y)


def _nearest_visible_tile(plan, view_range, focus):
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
