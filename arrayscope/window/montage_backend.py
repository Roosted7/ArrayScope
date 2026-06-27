"""Montage display backend policy."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class MontageBackendDecision:
    backend: str
    reason: str
    warning: str | None = None


def choose_montage_backend(
    geometry,
    data,
    *,
    renderer_backend: str = "pyqtgraph",
) -> MontageBackendDecision:
    if getattr(geometry, "montage", None) is None:
        return MontageBackendDecision("none", "not a montage display")
    renderer_name = str(getattr(renderer_backend, "value", renderer_backend) or "pyqtgraph")
    return MontageBackendDecision("tile_layer", f"{renderer_name} tiled montage presentation")


def backend_warning_for_actual_commit(decision: MontageBackendDecision, actual_backend: str) -> str | None:
    if decision.warning:
        return decision.warning
    return None
