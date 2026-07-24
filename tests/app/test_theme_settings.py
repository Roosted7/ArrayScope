import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).parents[2]
PACKAGE = types.ModuleType("arrayscope")
PACKAGE.__path__ = [str(ROOT / "arrayscope")]
sys.modules.setdefault("arrayscope", PACKAGE)


MODULE_PATHS = {
    "axis_utils": ("arrayscope.core.axis_utils", ROOT / "arrayscope" / "core" / "axis_utils.py"),
    "cache_status": (
        "arrayscope.core.cache_status",
        ROOT / "arrayscope" / "core" / "cache_status.py",
    ),
    "dimension_roles": (
        "arrayscope.core.dimension_roles",
        ROOT / "arrayscope" / "core" / "dimension_roles.py",
    ),
    "view_state": ("arrayscope.core.view_state", ROOT / "arrayscope" / "core" / "view_state.py"),
    "window_levels": (
        "arrayscope.core.window_levels",
        ROOT / "arrayscope" / "core" / "window_levels.py",
    ),
    "memory_budget": (
        "arrayscope.core.memory_budget",
        ROOT / "arrayscope" / "core" / "memory_budget.py",
    ),
    "memory_policy": (
        "arrayscope.core.memory_policy",
        ROOT / "arrayscope" / "core" / "memory_policy.py",
    ),
    "dim_ops": ("arrayscope.operations.dim_ops", ROOT / "arrayscope" / "operations" / "dim_ops.py"),
    "operation_pipeline": (
        "arrayscope.operations.pipeline",
        ROOT / "arrayscope" / "operations" / "pipeline.py",
    ),
    "operation_stack": (
        "arrayscope.operations.stack",
        ROOT / "arrayscope" / "operations" / "stack.py",
    ),
    "operation_evaluator": (
        "arrayscope.operations.evaluator",
        ROOT / "arrayscope" / "operations" / "evaluator.py",
    ),
    "operation_registry": (
        "arrayscope.operations.registry",
        ROOT / "arrayscope" / "operations" / "registry.py",
    ),
    "operation_recipes": (
        "arrayscope.operations.recipes",
        ROOT / "arrayscope" / "operations" / "recipes.py",
    ),
    "operation_coordinator": (
        "arrayscope.operations.coordinator",
        ROOT / "arrayscope" / "operations" / "coordinator.py",
    ),
    "slice_engine": (
        "arrayscope.display.slice_engine",
        ROOT / "arrayscope" / "display" / "slice_engine.py",
    ),
    "profile": ("arrayscope.profiles.model", ROOT / "arrayscope" / "profiles" / "model.py"),
    "profile_coordinator": (
        "arrayscope.profiles.coordinator",
        ROOT / "arrayscope" / "profiles" / "coordinator.py",
    ),
    "theme": ("arrayscope.app.theme", ROOT / "arrayscope" / "app" / "theme.py"),
    "settings_state": (
        "arrayscope.app.settings_state",
        ROOT / "arrayscope" / "app" / "settings_state.py",
    ),
}


def load_module(name):
    module_name, path = MODULE_PATHS[name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


theme = load_module("theme")
load_module("memory_budget")
load_module("memory_policy")
settings_state = load_module("settings_state")


def test_theme_backend_uses_builtin_palette_when_optional_backend_missing():
    result = theme.choose_theme_backend("dark", available_backends=())

    assert result.requested == theme.ThemeChoice.DARK
    assert result.applied == theme.ThemeChoice.DARK
    assert result.backend == "builtin"
    assert result.warning is None


def test_builtin_light_palette_path_is_selectable():
    result = theme.choose_theme_backend("light", available_backends=())

    assert result.applied == theme.ThemeChoice.LIGHT
    assert result.backend == "builtin"


def test_normalize_theme_choice_accepts_enum_values():
    assert theme.normalize_theme_choice(theme.ThemeChoice.DARK) == theme.ThemeChoice.DARK


def test_hover_token_is_distinct_from_alt_and_base_in_builtin_themes():
    # Regression: hover used to reuse surface_alt, so hovered list rows were
    # invisible on alternate (even) rows. Hover must be its own shade, and the
    # alternate-row shade must stay distinct from the base background.
    for tokens in (theme.DARK_TOKENS, theme.LIGHT_TOKENS):
        assert tokens.surface_hover != tokens.surface_alt
        assert tokens.surface_hover != tokens.base
        assert tokens.surface_alt != tokens.base


def test_native_palette_hover_token_is_distinct_from_alt_and_base():
    # The system-derived path must also compute a hover shade that never
    # collapses into the alternate-row or base backgrounds.
    tokens = theme._tokens_from_native_palette(None)
    assert tokens.surface_hover != tokens.surface_alt
    assert tokens.surface_hover != tokens.base
    assert tokens.surface_alt != tokens.base


def test_theme_backend_keeps_builtin_palette_even_when_optional_backend_available():
    result = theme.choose_theme_backend("light", available_backends=("qdarktheme",))

    assert result.applied == theme.ThemeChoice.LIGHT
    assert result.backend == "builtin"


def test_settings_round_trip_defaults_and_values():
    settings = settings_state.settings_from_mapping(
        {
            "theme": "dark",
            "prefetch_nearby_slices": "true",
            "panel_resize_behavior": "off",
            "fft_backend": "pyfftw",
            "fft_workers": "2",
            "image_rendering_backend": "vispy",
            "wgpu_present_method": "screen",
            "montage_quality_policy": "resident",
            "chunk_transport_codec": "zfp",
            "texture_codec": "bc",
            "wgpu_pixel_grid": "true",
            "wgpu_clip_indicator": "true",
            "resident_crop_rebind": "true",
            "memory_profile": "aggressive",
            "render_memory_budget_mb": "1024",
            "qt_platform": "xcb",
            "python_free_threading": "force_disabled",
        }
    )
    values = settings_state.settings_to_mapping(settings)

    assert values == {
        "theme": "dark",
        "prefetch_nearby_slices": True,
        "panel_resize_behavior": "off",
        "fft_backend": "pyfftw",
        "fft_workers": "2",
        "image_rendering_backend": "vispy",
        "wgpu_present_method": "screen",
        "montage_quality_policy": "resident",
        "chunk_transport_codec": "zfp",
        "texture_codec": "bc",
        "wgpu_pixel_grid": True,
        "wgpu_clip_indicator": True,
        "resident_crop_rebind": True,
        "memory_profile": "aggressive",
        "render_memory_budget_mb": 1024,
        "qt_platform": "xcb",
        "python_free_threading": "force_disabled",
    }
    defaults = settings_state.settings_from_mapping({})
    assert defaults.wgpu_pixel_grid is False
    assert defaults.wgpu_clip_indicator is False
    assert defaults.resident_crop_rebind is False
    assert defaults.theme == theme.ThemeChoice.SYSTEM
    assert defaults.panel_resize_behavior == settings_state.PanelResizeBehavior.BEST_EFFORT
    assert defaults.fft_backend == settings_state.FFTBackendChoice.AUTO
    assert defaults.fft_workers == settings_state.FFTWorkersChoice.AUTO
    assert defaults.image_rendering_backend == settings_state.ImageRenderingBackendChoice.AUTO
    # Screen presentation is opt-in (queue row 3): bitmap default, unknown
    # values normalize back to bitmap, and auto (screen where the measured
    # native-Wayland path exists) round-trips.
    assert defaults.wgpu_present_method == settings_state.WgpuPresentMethodChoice.BITMAP
    assert (
        settings_state.settings_from_mapping({"wgpu_present_method": "unknown"}).wgpu_present_method
        == settings_state.WgpuPresentMethodChoice.BITMAP
    )
    auto = settings_state.settings_from_mapping({"wgpu_present_method": "auto"})
    assert auto.wgpu_present_method == settings_state.WgpuPresentMethodChoice.AUTO
    assert settings_state.settings_to_mapping(auto)["wgpu_present_method"] == "auto"
    assert defaults.memory_profile == settings_state.MemoryProfileChoice.BALANCED
    assert defaults.render_memory_budget_mb == 512
    assert defaults.qt_platform == settings_state.QtPlatformChoice.AUTO
    # ADR 0050: resident LOD is the montage default once validated on hardware.
    assert defaults.montage_quality_policy == settings_state.MontageQualityPolicyChoice.RESIDENT
    # Component codecs remain available, but the product path stays raw until a
    # matched end-to-end benchmark proves a user-visible win.
    assert defaults.chunk_transport_codec == settings_state.ChunkTransportCodecChoice.RAW
    assert (
        settings_state.settings_from_mapping(
            {"chunk_transport_codec": "unknown"}
        ).chunk_transport_codec
        == settings_state.ChunkTransportCodecChoice.RAW
    )
    # Native block compression is also evidence-gated: OFF is the safe default
    # because standalone codec ratio does not include the live CPU/pool costs.
    assert defaults.texture_codec == settings_state.TextureCodecChoice.OFF
    assert (
        settings_state.settings_from_mapping({"texture_codec": "unknown"}).texture_codec
        == settings_state.TextureCodecChoice.OFF
    )
    assert (
        settings_state.texture_codec_executor_mode(
            settings_state.TextureCodecChoice.AUTO, bc_available=True
        )
        == "auto"
    )
    assert (
        settings_state.texture_codec_executor_mode(
            settings_state.TextureCodecChoice.AUTO, bc_available=False
        )
        == "off"
    )
    assert (
        settings_state.texture_codec_executor_mode(
            settings_state.TextureCodecChoice.BC, bc_available=False
        )
        == "on"
    )
    assert (
        settings_state.texture_codec_executor_mode(
            settings_state.TextureCodecChoice.OFF, bc_available=True
        )
        == "off"
    )
    unknown = settings_state.settings_from_mapping({"panel_resize_behavior": "unknown"})
    assert unknown.panel_resize_behavior == settings_state.PanelResizeBehavior.BEST_EFFORT
    unknown_quality = settings_state.settings_from_mapping({"montage_quality_policy": "unknown"})
    assert (
        unknown_quality.montage_quality_policy == settings_state.MontageQualityPolicyChoice.RESIDENT
    )


def test_performance_settings_normalize_unknowns_and_clamp_budget():
    unknown = settings_state.settings_from_mapping(
        {"fft_backend": "unknown", "fft_workers": "many", "image_rendering_backend": "nope"}
    )
    assert unknown.fft_backend == settings_state.FFTBackendChoice.AUTO
    assert unknown.fft_workers == settings_state.FFTWorkersChoice.AUTO
    assert unknown.image_rendering_backend == settings_state.ImageRenderingBackendChoice.AUTO
    assert (
        settings_state.settings_from_mapping({"memory_profile": "bad"}).memory_profile
        == settings_state.MemoryProfileChoice.BALANCED
    )

    assert (
        settings_state.settings_from_mapping(
            {"render_memory_budget_mb": "bad"}
        ).render_memory_budget_mb
        == 512
    )
    assert (
        settings_state.settings_from_mapping(
            {"render_memory_budget_mb": 64}
        ).render_memory_budget_mb
        == 128
    )
    assert (
        settings_state.settings_from_mapping(
            {"render_memory_budget_mb": 9000}
        ).render_memory_budget_mb
        == 8192
    )


def test_panel_resize_behavior_accepts_strong_wayland():
    settings = settings_state.settings_from_mapping({"panel_resize_behavior": "strong_wayland"})

    assert settings.panel_resize_behavior == settings_state.PanelResizeBehavior.STRONG_WAYLAND
    assert settings_state.settings_to_mapping(settings)["panel_resize_behavior"] == "strong_wayland"
