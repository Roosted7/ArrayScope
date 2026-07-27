"""Pure settings serialization helpers."""

from __future__ import annotations

import os
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
    # AUTO prefers wgpu on Linux with a real GPU device
    # (see resolve_auto_backend_choice). This value is the explicit pin.
    WGPU = "wgpu"


class RenderResponsivenessChoice(Enum):
    RESPONSIVE = "responsive"
    BALANCED = "balanced"
    THROUGHPUT = "throughput"

    @property
    def weight(self) -> float:
        return {
            RenderResponsivenessChoice.RESPONSIVE: 2.0,
            RenderResponsivenessChoice.BALANCED: 1.0,
            RenderResponsivenessChoice.THROUGHPUT: 0.3,
        }[self]


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
    render_responsiveness: RenderResponsivenessChoice = RenderResponsivenessChoice.BALANCED
    # Provenance only, not another user setting. A detected default must not
    # become a persisted override when an unrelated preference is saved.
    render_responsiveness_explicit: bool = False
    # wgpu backend only; screen is an explicit experimental pin (queue row 3).
    wgpu_present_method: WgpuPresentMethodChoice = WgpuPresentMethodChoice.BITMAP
    montage_quality_policy: MontageQualityPolicyChoice = MontageQualityPolicyChoice.RESIDENT
    # G7: explicit lossless host-cache experiment; RAW is the production path.
    chunk_transport_codec: ChunkTransportCodecChoice = ChunkTransportCodecChoice.RAW
    # G7: explicit lossy display-cache experiment; OFF is the exact path.
    texture_codec: TextureCodecChoice = TextureCodecChoice.OFF
    # wgpu display aids (shader Stage A + C1). The two Stage A aids default
    # off: they draw marks that are not data. The C1 minification filter
    # defaults ON — it is the honest answer for a draw where a screen pixel
    # covers several source texels, and on a zoomed-out montage the
    # point-sampled alternative shows one texel in ~35 and shimmers under pan.
    # It costs +1.7 ms on a full-window minified draw and is inert on any draw
    # at or below 1:1, so magnification stays exactly nearest.
    wgpu_pixel_grid: bool = False
    wgpu_clip_indicator: bool = False
    wgpu_minification_filter: bool = True
    # Montage fast path: a displayed-axis crop-window scrub whose new source
    # window is already physically resident short-circuits to a pure page rebind
    # (no re-evaluation). A rebound window re-anchors its auto levels from the
    # semantic evidence owner, on raw AND operation-pipeline montages, and both
    # paths settle identical levels; only the schedule differs (see
    # docs/redesign/histogram-evidence-pipeline-2026-07-23.md).
    resident_crop_rebind: bool = True
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
        render_responsiveness=(
            normalize_render_responsiveness_choice(values.get("render_responsiveness"))
            if "render_responsiveness" in values
            else default_render_responsiveness_choice()
        ),
        render_responsiveness_explicit="render_responsiveness" in values,
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
        wgpu_minification_filter=_to_bool(values.get("wgpu_minification_filter", True)),
        resident_crop_rebind=_to_bool(values.get("resident_crop_rebind", True)),
        memory_profile=normalize_memory_profile_choice(values.get("memory_profile")),
        render_memory_budget_mb=normalize_render_memory_budget_mb(
            values.get("render_memory_budget_mb", 512)
        ),
        qt_platform=normalize_qt_platform_choice(values.get("qt_platform")),
        python_free_threading=normalize_free_threading_choice(values.get("python_free_threading")),
    )


def settings_to_mapping(settings: AppSettingsState):
    values = {
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
        "wgpu_minification_filter": bool(settings.wgpu_minification_filter),
        "resident_crop_rebind": bool(settings.resident_crop_rebind),
        "memory_profile": settings.memory_profile.value,
        "render_memory_budget_mb": int(settings.render_memory_budget_mb),
        "qt_platform": settings.qt_platform.value,
        "python_free_threading": settings.python_free_threading.value,
    }
    if settings.render_responsiveness_explicit:
        values["render_responsiveness"] = settings.render_responsiveness.value
    return values


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


def normalize_render_responsiveness_choice(value) -> RenderResponsivenessChoice:
    if isinstance(value, RenderResponsivenessChoice):
        return value
    value = getattr(value, "value", value)
    try:
        return RenderResponsivenessChoice(str(value))
    except Exception:
        return RenderResponsivenessChoice.BALANCED


def default_render_responsiveness_choice(
    *,
    environ=None,
    topology=None,
) -> RenderResponsivenessChoice:
    """Seed the preset from session facts; a persisted choice always wins."""

    environment = os.environ if environ is None else environ
    remote_keys = (
        "SSH_CONNECTION",
        "SSH_CLIENT",
        "VNCSESSION",
        "VNCDESKTOP",
        "XRDP_SESSION",
        "X2GO_SESSION",
        "NXSESSIONID",
        "WAYPIPE_SOCKET",
    )
    if any(str(environment.get(key, "") or "").strip() for key in remote_keys):
        return RenderResponsivenessChoice.THROUGHPUT
    if str(environment.get("QT_QPA_PLATFORM", "") or "").lower() in {
        "offscreen",
        "minimal",
    }:
        return RenderResponsivenessChoice.THROUGHPUT
    software_values = " ".join(
        str(environment.get(key, "") or "").lower()
        for key in (
            "LIBGL_ALWAYS_SOFTWARE",
            "GALLIUM_DRIVER",
            "MESA_LOADER_DRIVER_OVERRIDE",
            "QT_QUICK_BACKEND",
        )
    )
    if any(
        marker in software_values
        for marker in ("llvmpipe", "softpipe", "swrast", "swiftshader", "software")
    ) or str(environment.get("LIBGL_ALWAYS_SOFTWARE", "")).lower() in {
        "1",
        "true",
        "yes",
    }:
        return RenderResponsivenessChoice.THROUGHPUT
    if topology is None:
        from arrayscope.gpu.device_topology import detect_topology

        topology = detect_topology()
    device_name = str(getattr(topology, "device_name", "") or "").lower()
    if str(getattr(topology, "kind", "unknown")) == "unknown" or any(
        marker in device_name
        for marker in ("llvmpipe", "softpipe", "swrast", "swiftshader", "software")
    ):
        return RenderResponsivenessChoice.THROUGHPUT
    return RenderResponsivenessChoice.BALANCED


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
        # explicit fallback policy on backends without resident LOD.
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
