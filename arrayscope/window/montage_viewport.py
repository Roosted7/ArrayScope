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

    def candidate_tiles(self, *, margin_tiles: int = 0):
        rect = montage_rect_for_viewport(
            self.plan,
            view_range=self.view_range,
            viewport_shape=self.viewport_shape,
        )
        return self.plan.tiles_intersecting(
            ((rect[0], rect[2]), (rect[1], rect[3])),
            margin_tiles=max(0, int(margin_tiles)),
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
