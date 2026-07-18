"""Pure settings serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from arrayscope.app.qt_platform import QtPlatformChoice, normalize_qt_platform_choice
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


class ImageRenderingBackendChoice(Enum):
    AUTO = "auto"
    PYQTGRAPH = "pyqtgraph"
    VISPY = "vispy"
    # Experimental explicit pin only: AUTO never resolves to wgpu.
    WGPU = "wgpu"


class MontageQualityPolicyChoice(Enum):
    NATIVE_ONLY = "native-only"
    RESIDENT = "resident"


@dataclass(frozen=True)
class AppSettingsState:
    theme: ThemeChoice = ThemeChoice.SYSTEM
    prefetch_nearby_slices: bool = False
    panel_resize_behavior: PanelResizeBehavior = PanelResizeBehavior.BEST_EFFORT
    fft_backend: FFTBackendChoice = FFTBackendChoice.AUTO
    fft_workers: FFTWorkersChoice = FFTWorkersChoice.AUTO
    image_rendering_backend: ImageRenderingBackendChoice = ImageRenderingBackendChoice.AUTO
    montage_quality_policy: MontageQualityPolicyChoice = MontageQualityPolicyChoice.RESIDENT
    memory_profile: MemoryProfileChoice = MemoryProfileChoice.BALANCED
    render_memory_budget_mb: int = 512
    # Linux/Wayland only; applied pre-QApplication (arrayscope.app.qt_platform).
    qt_platform: QtPlatformChoice = QtPlatformChoice.AUTO


def settings_from_mapping(values) -> AppSettingsState:
    values = dict(values or {})
    return AppSettingsState(
        theme=normalize_theme_choice(values.get("theme")),
        prefetch_nearby_slices=_to_bool(values.get("prefetch_nearby_slices", False)),
        panel_resize_behavior=normalize_panel_resize_behavior(values.get("panel_resize_behavior")),
        fft_backend=normalize_fft_backend_choice(values.get("fft_backend")),
        fft_workers=normalize_fft_workers_choice(values.get("fft_workers")),
        image_rendering_backend=normalize_image_rendering_backend_choice(values.get("image_rendering_backend")),
        montage_quality_policy=normalize_montage_quality_policy_choice(values.get("montage_quality_policy")),
        memory_profile=normalize_memory_profile_choice(values.get("memory_profile")),
        render_memory_budget_mb=normalize_render_memory_budget_mb(values.get("render_memory_budget_mb", 512)),
        qt_platform=normalize_qt_platform_choice(values.get("qt_platform")),
    )


def settings_to_mapping(settings: AppSettingsState):
    return {
        "theme": settings.theme.value,
        "prefetch_nearby_slices": bool(settings.prefetch_nearby_slices),
        "panel_resize_behavior": settings.panel_resize_behavior.value,
        "fft_backend": settings.fft_backend.value,
        "fft_workers": settings.fft_workers.value,
        "image_rendering_backend": settings.image_rendering_backend.value,
        "montage_quality_policy": settings.montage_quality_policy.value,
        "memory_profile": settings.memory_profile.value,
        "render_memory_budget_mb": int(settings.render_memory_budget_mb),
        "qt_platform": settings.qt_platform.value,
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
    value = getattr(value, "value", value)
    try:
        return FFTBackendChoice(str(value))
    except Exception:
        return FFTBackendChoice.AUTO


def normalize_fft_workers_choice(value) -> FFTWorkersChoice:
    if isinstance(value, FFTWorkersChoice):
        return value
    value = getattr(value, "value", value)
    try:
        return FFTWorkersChoice(str(value))
    except Exception:
        return FFTWorkersChoice.AUTO


def normalize_image_rendering_backend_choice(value) -> ImageRenderingBackendChoice:
    if isinstance(value, ImageRenderingBackendChoice):
        return value
    value = getattr(value, "value", value)
    try:
        return ImageRenderingBackendChoice(str(value))
    except Exception:
        return ImageRenderingBackendChoice.AUTO

def normalize_montage_quality_policy_choice(value) -> MontageQualityPolicyChoice:
    if isinstance(value, MontageQualityPolicyChoice):
        return value
    value = getattr(value, "value", value)
    try:
        return MontageQualityPolicyChoice(str(value))
    except Exception:
        # ADR 0050: resident is the montage default; native-only remains the
        # explicit fallback policy (and the effective one on non-VisPy
        # backends via the frame renderer capability gate).
        return MontageQualityPolicyChoice.RESIDENT


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
