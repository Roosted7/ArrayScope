"""Qt-free identity and coverage policy for montage viewport updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.montage import montage_rect_for_viewport


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
        int(viewport_plan.plan.columns),
        int(viewport_plan.plan.gap),
        lut_key,
        bool(viewport_plan.shader_display),
    )


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
    if bool(getattr(capabilities, "persistent_tile_residency", False)) and "tile_layer" in mode:
        return 90
    return 120
