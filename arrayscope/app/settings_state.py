"""Pure settings serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from arrayscope.app.theme import ThemeChoice, normalize_theme_choice
from arrayscope.core.memory_policy import MemoryProfileChoice, normalize_memory_profile_choice


class PanelResizeBehavior(Enum):
    BEST_EFFORT = "best_effort"
    STRONG_WAYLAND = "strong_wayland"
    OFF = "off"


class FFTBackendChoice(Enum):
    AUTO = "auto"
    SCIPY = "scipy"
    PYFFTW = "pyfftw"
    NUMPY = "numpy"


class FFTWorkersChoice(Enum):
    AUTO = "auto"
    ONE = "1"
    TWO = "2"
    FOUR = "4"
    ALL_MINUS_ONE = "all_minus_one"


@dataclass(frozen=True)
class AppSettingsState:
    theme: ThemeChoice = ThemeChoice.SYSTEM
    prefetch_nearby_slices: bool = False
    panel_resize_behavior: PanelResizeBehavior = PanelResizeBehavior.BEST_EFFORT
    fft_backend: FFTBackendChoice = FFTBackendChoice.AUTO
    fft_workers: FFTWorkersChoice = FFTWorkersChoice.AUTO
    memory_profile: MemoryProfileChoice = MemoryProfileChoice.BALANCED
    render_memory_budget_mb: int = 512


def settings_from_mapping(values) -> AppSettingsState:
    values = dict(values or {})
    return AppSettingsState(
        theme=normalize_theme_choice(values.get("theme")),
        prefetch_nearby_slices=_to_bool(values.get("prefetch_nearby_slices", False)),
        panel_resize_behavior=normalize_panel_resize_behavior(values.get("panel_resize_behavior")),
        fft_backend=normalize_fft_backend_choice(values.get("fft_backend")),
        fft_workers=normalize_fft_workers_choice(values.get("fft_workers")),
        memory_profile=normalize_memory_profile_choice(values.get("memory_profile")),
        render_memory_budget_mb=normalize_render_memory_budget_mb(values.get("render_memory_budget_mb", 512)),
    )


def settings_to_mapping(settings: AppSettingsState):
    return {
        "theme": settings.theme.value,
        "prefetch_nearby_slices": bool(settings.prefetch_nearby_slices),
        "panel_resize_behavior": settings.panel_resize_behavior.value,
        "fft_backend": settings.fft_backend.value,
        "fft_workers": settings.fft_workers.value,
        "memory_profile": settings.memory_profile.value,
        "render_memory_budget_mb": int(settings.render_memory_budget_mb),
    }


def normalize_panel_resize_behavior(value) -> PanelResizeBehavior:
    if isinstance(value, PanelResizeBehavior):
        return value
    try:
        return PanelResizeBehavior(str(value))
    except Exception:
        return PanelResizeBehavior.BEST_EFFORT


def normalize_fft_backend_choice(value) -> FFTBackendChoice:
    if isinstance(value, FFTBackendChoice):
        return value
    try:
        return FFTBackendChoice(str(value))
    except Exception:
        return FFTBackendChoice.AUTO


def normalize_fft_workers_choice(value) -> FFTWorkersChoice:
    if isinstance(value, FFTWorkersChoice):
        return value
    try:
        return FFTWorkersChoice(str(value))
    except Exception:
        return FFTWorkersChoice.AUTO


def normalize_render_memory_budget_mb(value) -> int:
    try:
        mb = int(value)
    except Exception:
        return 512
    return max(128, min(8192, mb))


def _to_bool(value):
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)
