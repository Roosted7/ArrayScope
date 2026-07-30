import importlib.util
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pyqtgraph") is None,
    reason="pyqtgraph is not installed",
)


def test_arrayscope_defaults_pyqtgraph_to_pyside6():
    env = os.environ.copy()
    env.pop("PYQTGRAPH_QT_LIB", None)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import arrayscope, pyqtgraph.Qt as Qt; print(Qt.QT_LIB)",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.splitlines()[-1] == "PySide6"


def test_arrayscope_respects_explicit_pyqtgraph_binding():
    if importlib.util.find_spec("PyQt5") is None:
        pytest.skip("PyQt5 is not installed")

    env = os.environ.copy()
    env["PYQTGRAPH_QT_LIB"] = "PyQt5"
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import arrayscope, pyqtgraph.Qt as Qt; print(Qt.QT_LIB)",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.splitlines()[-1] == "PyQt5"


def _pinned(monkeypatch, environment):
    """Run the appearance pin against a controlled environment, return it."""

    from arrayscope.app.qt_binding import pin_wayland_appearance

    for name in (
        "WAYLAND_DISPLAY",
        "QT_QPA_PLATFORM",
        "QT_WAYLAND_DECORATION",
        "QT_QPA_PLATFORMTHEME",
        "XDG_CURRENT_DESKTOP",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    pin_wayland_appearance()
    return {
        name: os.environ.get(name) for name in ("QT_WAYLAND_DECORATION", "QT_QPA_PLATFORMTHEME")
    }


def test_gnome_wayland_gets_the_chrome_arrayscope_has_always_had(monkeypatch):
    """The PyPI wheel bundles adwaita decorations and a gtk3 theme that Qt
    auto-prefers on GNOME; without the pin, the app's face changes with the
    package manager (observed 2026-07-30, conda -> uv)."""

    pinned = _pinned(monkeypatch, {"WAYLAND_DISPLAY": "wayland-0", "XDG_CURRENT_DESKTOP": "GNOME"})
    assert pinned == {
        "QT_WAYLAND_DECORATION": "bradient",
        "QT_QPA_PLATFORMTHEME": "xdgdesktopportal",
    }


def test_an_explicit_appearance_choice_always_wins(monkeypatch):
    pinned = _pinned(
        monkeypatch,
        {
            "WAYLAND_DISPLAY": "wayland-0",
            "XDG_CURRENT_DESKTOP": "GNOME",
            "QT_WAYLAND_DECORATION": "adwaita",
            "QT_QPA_PLATFORMTHEME": "gtk3",
        },
    )
    assert pinned == {"QT_WAYLAND_DECORATION": "adwaita", "QT_QPA_PLATFORMTHEME": "gtk3"}


def test_kde_keeps_its_native_theme_and_x11_is_left_alone(monkeypatch):
    """On KDE the compositor draws server-side decorations and Qt's own kde
    theme is the native look; on X11 the titlebar is the window manager's.
    Pinning either would make the app less native, not more."""

    kde = _pinned(monkeypatch, {"WAYLAND_DISPLAY": "wayland-0", "XDG_CURRENT_DESKTOP": "KDE"})
    assert kde["QT_QPA_PLATFORMTHEME"] is None
    assert kde["QT_WAYLAND_DECORATION"] == "bradient"

    x11 = _pinned(monkeypatch, {"XDG_CURRENT_DESKTOP": "GNOME"})
    assert x11 == {"QT_WAYLAND_DECORATION": None, "QT_QPA_PLATFORMTHEME": None}


def test_headless_runs_are_never_pinned(monkeypatch):
    """The offscreen suite must not inherit desktop-appearance state: the map
    is keyed by QT_QPA_PLATFORM, and a pin that leaked into test processes
    would vary their environment with the desktop they ran under."""

    pinned = _pinned(
        monkeypatch,
        {
            "WAYLAND_DISPLAY": "wayland-0",
            "XDG_CURRENT_DESKTOP": "GNOME",
            "QT_QPA_PLATFORM": "offscreen",
        },
    )
    assert pinned == {"QT_WAYLAND_DECORATION": None, "QT_QPA_PLATFORMTHEME": None}


def test_importing_launch_never_pins_appearance(monkeypatch):
    """The pin must wait for _prepare_qt_environment, not fire on import.

    An import-time pin runs before a test harness can say "offscreen", so a
    desktop session's WAYLAND_DISPLAY leaked the portal theme into headless
    processes — whose settle loops then paid real DBus round-trips (measured
    11.6 s -> 30.5 s on one settle-budget test, deterministically, serial)."""

    import importlib
    import sys

    for name in ("QT_WAYLAND_DECORATION", "QT_QPA_PLATFORMTHEME", "QT_QPA_PLATFORM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")

    sys.modules.pop("arrayscope.app.launch", None)
    importlib.import_module("arrayscope.app.launch")
    assert os.environ.get("QT_WAYLAND_DECORATION") is None
    assert os.environ.get("QT_QPA_PLATFORMTHEME") is None
