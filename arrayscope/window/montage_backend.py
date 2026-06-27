"""Montage display backend policy."""

from __future__ import annotations

from dataclasses import dataclass

from arrayscope.display.backend_contract import ImageViewBackendCapabilities


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
    previous_upload_ms: float = 0.0,
    patched_tiles: int = 0,
    current_mode: str = "canvas",
    renderer_backend: str = "pyqtgraph",
    renderer_capabilities: ImageViewBackendCapabilities | None = None,
    very_slow_upload_ms: float = 100.0,
) -> MontageBackendDecision:
    if getattr(geometry, "montage", None) is None:
        return MontageBackendDecision("canvas", "not a montage display")

    direct_tile_payloads = (
        bool(renderer_capabilities.direct_montage_tile_payloads)
        if isinstance(renderer_capabilities, ImageViewBackendCapabilities)
        else str(getattr(renderer_backend, "value", renderer_backend) or "pyqtgraph").lower() in {"pyqtgraph", "vispy"}
    )
    renderer_name = (
        str(renderer_capabilities.name)
        if isinstance(renderer_capabilities, ImageViewBackendCapabilities)
        else renderer_backend
    )
    if not direct_tile_payloads:
        return MontageBackendDecision(
            "tile_layer",
            f"{renderer_name} montage requires direct tiled payloads",
            warning="renderer capabilities do not provide the montage presentation contract",
            expected_tile_layer=True,
        )
    return MontageBackendDecision(
        "tile_layer",
        f"{renderer_name} supports direct tiled montage payloads",
        expected_tile_layer=True,
    )


def backend_warning_for_actual_commit(decision: MontageBackendDecision, actual_backend: str) -> str | None:
    if decision.warning:
        return decision.warning
    if decision.expected_tile_layer and str(actual_backend) != "tile_layer":
        return "montage committed outside the tiled presentation path"
    return None
