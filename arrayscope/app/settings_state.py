"""Pure settings serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from arrayscope.app.theme import ThemeChoice, normalize_theme_choice


class PanelResizeBehavior(Enum):
    BEST_EFFORT = "best_effort"
    OFF = "off"


@dataclass(frozen=True)
class AppSettingsState:
    theme: ThemeChoice = ThemeChoice.SYSTEM
    prefetch_nearby_slices: bool = False
    panel_resize_behavior: PanelResizeBehavior = PanelResizeBehavior.BEST_EFFORT


def settings_from_mapping(values) -> AppSettingsState:
    values = dict(values or {})
    return AppSettingsState(
        theme=normalize_theme_choice(values.get("theme")),
        prefetch_nearby_slices=_to_bool(values.get("prefetch_nearby_slices", False)),
        panel_resize_behavior=normalize_panel_resize_behavior(values.get("panel_resize_behavior")),
    )


def settings_to_mapping(settings: AppSettingsState):
    return {
        "theme": settings.theme.value,
        "prefetch_nearby_slices": bool(settings.prefetch_nearby_slices),
        "panel_resize_behavior": settings.panel_resize_behavior.value,
    }


def normalize_panel_resize_behavior(value) -> PanelResizeBehavior:
    if isinstance(value, PanelResizeBehavior):
        return value
    try:
        return PanelResizeBehavior(str(value))
    except Exception:
        return PanelResizeBehavior.BEST_EFFORT


def _to_bool(value):
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)
