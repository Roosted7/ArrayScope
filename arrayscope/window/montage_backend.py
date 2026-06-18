"""Montage display backend policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arrayscope.app.settings_state import MontageDisplayBackendChoice, normalize_montage_display_backend_choice


LARGE_MONTAGE_CANVAS_PIXELS = 2_000_000


@dataclass(frozen=True)
class MontageBackendDecision:
    backend: str
    reason: str
    warning: str | None = None
    expected_tile_layer: bool = False


def choose_montage_backend(
    geometry,
    data,
    *,
    setting=MontageDisplayBackendChoice.AUTO,
    previous_upload_ms: float = 0.0,
    patched_tiles: int = 0,
    current_mode: str = "canvas",
    renderer_backend: str = "pyqtgraph",
    very_slow_upload_ms: float = 100.0,
) -> MontageBackendDecision:
    if getattr(geometry, "montage", None) is None:
        return MontageBackendDecision("canvas", "not a montage display")

    setting = normalize_montage_display_backend_choice(setting)
    pixels = _canvas_pixels(data)
    rgb_like = _is_rgb_like(data)
    large = pixels > LARGE_MONTAGE_CANVAS_PIXELS
    large_rgb = large and rgb_like
    renderer_backend = str(getattr(renderer_backend, "value", renderer_backend) or "pyqtgraph").lower()
    current_is_tile_layer = str(current_mode) in {"tile_layer", "vispy_tile_layer"}

    if setting == MontageDisplayBackendChoice.TILE_LAYER:
        return MontageBackendDecision("tile_layer", "user forced tile layer", expected_tile_layer=True)

    if setting == MontageDisplayBackendChoice.CANVAS:
        warning = None
        if large_rgb:
            warning = "canvas fallback is manual and may be slow for large RGB/complex montage"
        return MontageBackendDecision("canvas", "user forced canvas fallback", warning=warning, expected_tile_layer=False)

    if large_rgb:
        return MontageBackendDecision(
            "tile_layer",
            f"RGB/complex montage canvas pixels {pixels} > {LARGE_MONTAGE_CANVAS_PIXELS}",
            expected_tile_layer=True,
        )
    if large and renderer_backend == "vispy":
        return MontageBackendDecision(
            "tile_layer",
            f"VisPy montage canvas pixels {pixels} > {LARGE_MONTAGE_CANVAS_PIXELS}; avoid full texture uploads",
            expected_tile_layer=True,
        )
    if float(previous_upload_ms or 0.0) > float(very_slow_upload_ms):
        return MontageBackendDecision(
            "tile_layer",
            f"previous montage upload {float(previous_upload_ms):.1f} ms > {float(very_slow_upload_ms):.1f} ms",
            expected_tile_layer=True,
        )
    if int(patched_tiles or 0) > 8:
        return MontageBackendDecision(
            "tile_layer",
            f"patched tiles last flush {int(patched_tiles)} > 8",
            expected_tile_layer=True,
        )
    if current_is_tile_layer:
        return MontageBackendDecision("tile_layer", "preserving active tile-layer backend", expected_tile_layer=True)
    if large:
        return MontageBackendDecision("canvas", f"large scalar montage canvas pixels {pixels}; scalar levels are cheap")
    return MontageBackendDecision("canvas", f"small montage canvas pixels {pixels}")


def backend_warning_for_actual_commit(decision: MontageBackendDecision, actual_backend: str) -> str | None:
    if decision.warning:
        return decision.warning
    if decision.expected_tile_layer and str(actual_backend) != "tile_layer":
        return "large montage committed through canvas mode; expected tile_layer"
    return None


def _canvas_pixels(data) -> int:
    shape = tuple(np.shape(data)[:2])
    if len(shape) != 2:
        return 0
    return int(shape[0]) * int(shape[1])


def _is_rgb_like(data) -> bool:
    shape = tuple(np.shape(data))
    return len(shape) == 3 and int(shape[-1]) in (3, 4)
