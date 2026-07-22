"""Pure settings serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from arrayscope.app.free_threading import FreeThreadingChoice, normalize_free_threading_choice
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


class WgpuPresentMethodChoice(Enum):
    # AUTO resolves per session: the screen swapchain exactly where the
    # measured gate-B recipe applies (native Wayland), bitmap everywhere
    # else.  Screen on other platforms (e.g. xcb) would be a different,
    # unmeasured path — AUTO widens only behind new evidence.
    AUTO = "auto"
    BITMAP = "bitmap"
    # SCREEN is the explicit pin (native-Wayland swapchain; the view falls
    # back to bitmap loudly anywhere the screen path cannot exist).
    SCREEN = "screen"


class MontageQualityPolicyChoice(Enum):
    NATIVE_ONLY = "native-only"
    RESIDENT = "resident"


class ChunkTransportCodecChoice(Enum):
    # G7 host-cache experiment. RAW is the byte-identical production default.
    # AUTO/ZFP/BLOSC2 opt into a lossless compressed backing tier under one
    # total byte budget; they remain explicit until an off-GUI tier at the
    # actual expensive miss owner proves a live benefit.
    AUTO = "auto"
    RAW = "raw"
    ZFP = "zfp"
    BLOSC2 = "blosc2"


class TextureCodecChoice(Enum):
    # G7 lossy display-cache experiment. OFF is the exact production default.
    # AUTO/BC opt into native block-compressed texture pools where supported;
    # exact settled semantic evidence remains separate from those pixels.
    AUTO = "auto"
    OFF = "off"
    BC = "bc"


@dataclass(frozen=True)
class AppSettingsState:
    theme: ThemeChoice = ThemeChoice.SYSTEM
    prefetch_nearby_slices: bool = False
    panel_resize_behavior: PanelResizeBehavior = PanelResizeBehavior.BEST_EFFORT
    fft_backend: FFTBackendChoice = FFTBackendChoice.AUTO
    fft_workers: FFTWorkersChoice = FFTWorkersChoice.AUTO
    image_rendering_backend: ImageRenderingBackendChoice = ImageRenderingBackendChoice.AUTO
    # wgpu backend only; screen is an explicit experimental pin (queue row 3).
    wgpu_present_method: WgpuPresentMethodChoice = WgpuPresentMethodChoice.BITMAP
    montage_quality_policy: MontageQualityPolicyChoice = MontageQualityPolicyChoice.RESIDENT
    # G7: explicit lossless host-cache experiment; RAW is the production path.
    chunk_transport_codec: ChunkTransportCodecChoice = ChunkTransportCodecChoice.RAW
    # G7: explicit lossy display-cache experiment; OFF is the exact path.
    texture_codec: TextureCodecChoice = TextureCodecChoice.OFF
    # wgpu display legibility aids (shader Stage A). Both default off so the
    # default render is byte-identical; the pixel grid is zoom-gated even on.
    wgpu_pixel_grid: bool = False
    wgpu_clip_indicator: bool = False
    memory_profile: MemoryProfileChoice = MemoryProfileChoice.BALANCED
    render_memory_budget_mb: int = 512
    # Linux/Wayland only; applied pre-QApplication (arrayscope.app.qt_platform).
    qt_platform: QtPlatformChoice = QtPlatformChoice.AUTO
    # Free-threaded builds (3.14t) only; applied at CLI launch
    # (arrayscope.app.free_threading; auto_disabled is crash-supervisor-written).
    python_free_threading: FreeThreadingChoice = FreeThreadingChoice.ENABLED


def settings_from_mapping(values) -> AppSettingsState:
    values = dict(values or {})
    return AppSettingsState(
        theme=normalize_theme_choice(values.get("theme")),
        prefetch_nearby_slices=_to_bool(values.get("prefetch_nearby_slices", False)),
        panel_resize_behavior=normalize_panel_resize_behavior(values.get("panel_resize_behavior")),
        fft_backend=normalize_fft_backend_choice(values.get("fft_backend")),
        fft_workers=normalize_fft_workers_choice(values.get("fft_workers")),
        image_rendering_backend=normalize_image_rendering_backend_choice(
            values.get("image_rendering_backend")
        ),
        wgpu_present_method=normalize_wgpu_present_method_choice(values.get("wgpu_present_method")),
        montage_quality_policy=normalize_montage_quality_policy_choice(
            values.get("montage_quality_policy")
        ),
        chunk_transport_codec=normalize_chunk_transport_codec_choice(
            values.get("chunk_transport_codec")
        ),
        texture_codec=normalize_texture_codec_choice(values.get("texture_codec")),
        wgpu_pixel_grid=_to_bool(values.get("wgpu_pixel_grid", False)),
        wgpu_clip_indicator=_to_bool(values.get("wgpu_clip_indicator", False)),
        memory_profile=normalize_memory_profile_choice(values.get("memory_profile")),
        render_memory_budget_mb=normalize_render_memory_budget_mb(
            values.get("render_memory_budget_mb", 512)
        ),
        qt_platform=normalize_qt_platform_choice(values.get("qt_platform")),
        python_free_threading=normalize_free_threading_choice(values.get("python_free_threading")),
    )


def settings_to_mapping(settings: AppSettingsState):
    return {
        "theme": settings.theme.value,
        "prefetch_nearby_slices": bool(settings.prefetch_nearby_slices),
        "panel_resize_behavior": settings.panel_resize_behavior.value,
        "fft_backend": settings.fft_backend.value,
        "fft_workers": settings.fft_workers.value,
        "image_rendering_backend": settings.image_rendering_backend.value,
        "wgpu_present_method": settings.wgpu_present_method.value,
        "montage_quality_policy": settings.montage_quality_policy.value,
        "chunk_transport_codec": settings.chunk_transport_codec.value,
        "texture_codec": settings.texture_codec.value,
        "wgpu_pixel_grid": bool(settings.wgpu_pixel_grid),
        "wgpu_clip_indicator": bool(settings.wgpu_clip_indicator),
        "memory_profile": settings.memory_profile.value,
        "render_memory_budget_mb": int(settings.render_memory_budget_mb),
        "qt_platform": settings.qt_platform.value,
        "python_free_threading": settings.python_free_threading.value,
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


def normalize_wgpu_present_method_choice(value) -> WgpuPresentMethodChoice:
    if isinstance(value, WgpuPresentMethodChoice):
        return value
    value = getattr(value, "value", value)
    try:
        return WgpuPresentMethodChoice(str(value))
    except Exception:
        return WgpuPresentMethodChoice.BITMAP


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


def normalize_chunk_transport_codec_choice(value) -> ChunkTransportCodecChoice:
    if isinstance(value, ChunkTransportCodecChoice):
        return value
    value = getattr(value, "value", value)
    try:
        return ChunkTransportCodecChoice(str(value))
    except Exception:
        # Unknown/absent -> RAW: live benefit is evidence-gated; explicit codec
        # choices remain available for experiments.
        return ChunkTransportCodecChoice.RAW


def normalize_texture_codec_choice(value) -> TextureCodecChoice:
    if isinstance(value, TextureCodecChoice):
        return value
    value = getattr(value, "value", value)
    try:
        return TextureCodecChoice(str(value))
    except Exception:
        # Unknown/absent -> OFF: compression remains an explicit experiment until
        # matched live evidence proves a product benefit on this device/workload.
        return TextureCodecChoice.OFF


def texture_codec_executor_mode(choice: TextureCodecChoice, *, bc_available: bool) -> str:
    """Resolve the display-codec setting to a WgpuPlaneExecutor mode string.

    Returns ``"off"``/``"on"``/``"auto"`` for the ``compressed_textures``
    constructor argument. AUTO engages only when explicitly selected and the
    device has BC/ASTC; BC is the explicit force-on; OFF stays raw.
    """

    if choice == TextureCodecChoice.OFF:
        return "off"
    if choice == TextureCodecChoice.BC:
        return "on"
    # Explicit AUTO: use a native codec when one exists, raw otherwise.
    return "auto" if bc_available else "off"


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
