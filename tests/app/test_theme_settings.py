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


def test_stylesheet_leaves_the_combobox_arrow_to_the_native_style():
    # Regression: styling ``QComboBox::drop-down`` (or ``::down-arrow``) makes Qt
    # drop the native arrow primitive and QSS can only replace it with an image
    # asset we do not ship -- every combobox then looked like a read-only line
    # edit, which is how the custom-operation import dialog hid its function
    # picker. Keep both subcontrols unstyled.
    for tokens in (theme.DARK_TOKENS, theme.LIGHT_TOKENS):
        stylesheet = theme.build_stylesheet(tokens)
        assert "QComboBox::drop-down" not in stylesheet
        assert "QComboBox::down-arrow" not in stylesheet


def test_theme_backend_keeps_builtin_palette_even_when_optional_backend_available():
    result = theme.choose_theme_backend("light", available_backends=("qdarktheme",))

    assert result.applied == theme.ThemeChoice.LIGHT
    assert result.backend == "builtin"


def test_settings_round_trip_defaults_and_values():
    """A persisted retired backend value migrates through the AUTO policy."""

    settings = settings_state.settings_from_mapping(
        {
            "theme": "dark",
            "prefetch_nearby_slices": "true",
            "panel_resize_behavior": "off",
            "fft_backend": "pyfftw",
            "fft_workers": "2",
            "image_rendering_backend": "vispy",
            "render_responsiveness": "responsive",
            "wgpu_present_method": "screen",
            "montage_quality_policy": "resident",
            "chunk_transport_codec": "zfp",
            "texture_codec": "bc",
            "wgpu_pixel_grid": "true",
            "wgpu_clip_indicator": "true",
            "wgpu_minification_filter": "true",
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
        "image_rendering_backend": "auto",
        "render_responsiveness": "responsive",
        "wgpu_present_method": "screen",
        "montage_quality_policy": "resident",
        "chunk_transport_codec": "zfp",
        "texture_codec": "bc",
        "wgpu_pixel_grid": True,
        "wgpu_clip_indicator": True,
        "wgpu_minification_filter": True,
        "resident_crop_rebind": True,
        "memory_profile": "aggressive",
        "render_memory_budget_mb": 1024,
        "qt_platform": "xcb",
        "python_free_threading": "force_disabled",
    }
    assert settings.image_rendering_backend == settings_state.ImageRenderingBackendChoice.AUTO
    defaults = settings_state.settings_from_mapping({})
    assert defaults.wgpu_pixel_grid is False
    assert defaults.wgpu_clip_indicator is False
    assert defaults.wgpu_minification_filter is True
    assert defaults.resident_crop_rebind is True
    assert defaults.theme == theme.ThemeChoice.SYSTEM
    assert defaults.panel_resize_behavior == settings_state.PanelResizeBehavior.BEST_EFFORT
    assert defaults.fft_backend == settings_state.FFTBackendChoice.AUTO
    assert defaults.fft_workers == settings_state.FFTWorkersChoice.AUTO
    assert defaults.image_rendering_backend == settings_state.ImageRenderingBackendChoice.AUTO
    assert defaults.render_responsiveness in {
        settings_state.RenderResponsivenessChoice.BALANCED,
        settings_state.RenderResponsivenessChoice.THROUGHPUT,
    }
    assert "render_responsiveness" not in settings_state.settings_to_mapping(defaults)
    # Presentation resolves per session: AUTO is the default (screen exactly
    # where the measured native-Wayland path exists, bitmap everywhere else),
    # unknown values normalize back to that same default, and an explicit
    # bitmap pin round-trips.
    assert defaults.wgpu_present_method == settings_state.WgpuPresentMethodChoice.AUTO
    assert (
        settings_state.settings_from_mapping({"wgpu_present_method": "unknown"}).wgpu_present_method
        == settings_state.WgpuPresentMethodChoice.AUTO
    )
    bitmap = settings_state.settings_from_mapping({"wgpu_present_method": "bitmap"})
    assert bitmap.wgpu_present_method == settings_state.WgpuPresentMethodChoice.BITMAP
    assert settings_state.settings_to_mapping(bitmap)["wgpu_present_method"] == "bitmap"
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


def test_every_present_method_default_owner_names_the_same_choice():
    """One default, stated in four places, pinned so they cannot drift apart.

    This setting has more default owners than it looks: the dataclass field,
    the normalizer's fallback (which is what a mapping *without* the key
    reaches, since ``settings_from_mapping`` feeds ``None`` through it), the
    menus' QSettings read fallback, and the factory's ``getattr`` fallback.
    A disagreement between any two of them is invisible in normal use — it
    only shows up as "the app behaves differently depending on whether the
    setting was ever written", which is exactly the bug shape this repo has
    hit before with other settings.
    """

    import inspect

    from arrayscope.display import image_view_factory
    from arrayscope.ui import menus

    expected = settings_state.WgpuPresentMethodChoice.AUTO

    # 1. the dataclass field, and 2. the normalizer's fallback (via a mapping
    # that does not carry the key at all).
    assert settings_state.AppSettingsState().wgpu_present_method == expected
    assert settings_state.settings_from_mapping({}).wgpu_present_method == expected
    assert settings_state.normalize_wgpu_present_method_choice(None) == expected

    # 3. and 4. are literal fallback arguments rather than call results, so
    # they are read from the source that states them.
    menus_source = inspect.getsource(menus.WindowMenuMixin._load_app_settings)
    assert f'"wgpu_present_method", WgpuPresentMethodChoice.{expected.name}.value' in menus_source
    factory_source = inspect.getsource(image_view_factory.create_image_view)
    assert f'"wgpu_present_method", WgpuPresentMethodChoice.{expected.name}' in factory_source


def test_performance_settings_normalize_unknowns_and_clamp_budget():
    unknown = settings_state.settings_from_mapping(
        {"fft_backend": "unknown", "fft_workers": "many", "image_rendering_backend": "nope"}
    )
    assert unknown.fft_backend == settings_state.FFTBackendChoice.AUTO
    assert unknown.fft_workers == settings_state.FFTWorkersChoice.AUTO
    assert unknown.image_rendering_backend == settings_state.ImageRenderingBackendChoice.AUTO
    assert (
        settings_state.settings_from_mapping({"render_responsiveness": "bad"}).render_responsiveness
        == settings_state.RenderResponsivenessChoice.BALANCED
    )


def test_pyqtgraph_pin_skips_the_gpu_topology_probe(monkeypatch):
    """An explicit PyQtGraph pin must not reach the wgpu adapter probe.

    The probe imports wgpu and requests an adapter; on a PyQtGraph session it
    was the only wgpu entry point in the whole process, so it initialized the
    Vulkan backend (and printed its adapter warnings) for a session that never
    issues a GPU draw.
    """

    from types import SimpleNamespace

    from arrayscope.gpu import device_topology

    probes = []

    def _record_probe(**kwargs):
        probes.append(kwargs)
        return SimpleNamespace(kind="integrated", device_name="gpu")

    monkeypatch.setattr(device_topology, "detect_topology", _record_probe)

    pinned = settings_state.default_render_responsiveness_choice(
        environ={}, image_backend="pyqtgraph"
    )

    assert probes == []
    assert pinned == settings_state.RenderResponsivenessChoice.BALANCED

    # Control: AUTO must still probe -- it cannot resolve a backend without
    # asking whether a device exists, and without this arm the assertion above
    # would also pass for a probe that never runs for any backend.
    auto = settings_state.default_render_responsiveness_choice(environ={}, image_backend="auto")

    assert len(probes) == 1
    assert auto == settings_state.RenderResponsivenessChoice.BALANCED


def test_settings_mapping_forwards_the_backend_pin_to_the_responsiveness_default(monkeypatch):
    """The pin only suppresses the probe if the mapping actually passes it on."""

    seen = {}

    def _record(**kwargs):
        seen.update(kwargs)
        return settings_state.RenderResponsivenessChoice.BALANCED

    monkeypatch.setattr(settings_state, "default_render_responsiveness_choice", _record)

    settings_state.settings_from_mapping({"image_rendering_backend": "pyqtgraph"})

    assert seen["image_backend"] == settings_state.ImageRenderingBackendChoice.PYQTGRAPH


def test_render_responsiveness_default_detects_remote_but_explicit_choice_wins():
    from types import SimpleNamespace

    detected = settings_state.default_render_responsiveness_choice(
        environ={"SSH_CONNECTION": "client server"},
        topology=SimpleNamespace(kind="discrete", device_name="GPU"),
    )
    local = settings_state.default_render_responsiveness_choice(
        environ={},
        topology=SimpleNamespace(kind="discrete", device_name="GPU"),
    )

    assert detected == settings_state.RenderResponsivenessChoice.THROUGHPUT
    assert local == settings_state.RenderResponsivenessChoice.BALANCED
    assert (
        settings_state.settings_from_mapping(
            {"render_responsiveness": "responsive"}
        ).render_responsiveness
        == settings_state.RenderResponsivenessChoice.RESPONSIVE
    )
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


class _RecordedQSettings:
    """The two QSettings methods ``_load_app_settings`` actually calls."""

    def __init__(self, stored=None):
        self._stored = dict(stored or {})

    def contains(self, key):
        return key in self._stored

    def value(self, key, default=None):
        return self._stored.get(key, default)


def test_explicit_render_responsiveness_is_restored_across_a_restart():
    """A preset chosen from the menu must still be that preset next launch.

    This setting is carried by key *presence*, not by a value: both
    ``render_responsiveness`` and the ``render_responsiveness_explicit``
    provenance flag come out of the same ``"render_responsiveness" in values``
    test.  So the QSettings read has to preserve presence in both directions.
    Reading it with a fallback like every other key would make every session
    look explicit — and then persist a merely *detected* seed as a user
    override.  Omitting it unconditionally (the original bug) sent the saved
    choice through detection again on every launch and dropped it from the
    next save, so it never survived a restart at all.
    """

    from types import SimpleNamespace

    from arrayscope.ui import menus

    chosen = menus.WindowMenuMixin._load_app_settings(
        SimpleNamespace(_settings=_RecordedQSettings({"render_responsiveness": "responsive"}))
    )

    assert chosen.render_responsiveness == settings_state.RenderResponsivenessChoice.RESPONSIVE
    assert chosen.render_responsiveness_explicit is True
    # ...and it is still written back, so the choice does not decay after one
    # unrelated settings save.
    assert settings_state.settings_to_mapping(chosen)["render_responsiveness"] == "responsive"

    # The backend pin only keeps this arm off the GPU: with no preset key at
    # all the choice is detected, and AUTO detection requests a wgpu adapter on
    # a run that has a real display (ring 3/4).  Presence is what is under test.
    never_set = menus.WindowMenuMixin._load_app_settings(
        SimpleNamespace(_settings=_RecordedQSettings({"image_rendering_backend": "pyqtgraph"}))
    )

    assert never_set.render_responsiveness_explicit is False
    assert "render_responsiveness" not in settings_state.settings_to_mapping(never_set)


def test_panel_resize_behavior_accepts_strong_wayland():
    settings = settings_state.settings_from_mapping({"panel_resize_behavior": "strong_wayland"})

    assert settings.panel_resize_behavior == settings_state.PanelResizeBehavior.STRONG_WAYLAND
    assert settings_state.settings_to_mapping(settings)["panel_resize_behavior"] == "strong_wayland"
