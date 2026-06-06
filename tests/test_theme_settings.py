import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).parents[1]
PACKAGE = types.ModuleType("arrayscope")
PACKAGE.__path__ = [str(ROOT / "arrayscope")]
sys.modules.setdefault("arrayscope", PACKAGE)


def load_module(name):
    path = ROOT / "arrayscope" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"arrayscope.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


theme = load_module("theme")
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


def test_theme_backend_keeps_builtin_palette_even_when_optional_backend_available():
    result = theme.choose_theme_backend("light", available_backends=("qdarktheme",))

    assert result.applied == theme.ThemeChoice.LIGHT
    assert result.backend == "builtin"


def test_settings_round_trip_defaults_and_values():
    settings = settings_state.settings_from_mapping({"theme": "dark", "prefetch_nearby_slices": "true"})
    values = settings_state.settings_to_mapping(settings)

    assert values == {"theme": "dark", "prefetch_nearby_slices": True}
    assert settings_state.settings_from_mapping({}).theme == theme.ThemeChoice.SYSTEM
