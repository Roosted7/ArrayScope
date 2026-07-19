"""Locate packaged icon assets (arrayscope/resources/icons)."""

from __future__ import annotations

from pathlib import Path

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)


def icons_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "resources" / "icons"


def _existing(path: Path):
    return path if path.exists() else None


def icon_png(size: int):
    return _existing(icons_dir() / f"arrayscope-{size}.png")


def icon_svg():
    return _existing(icons_dir() / "arrayscope.svg")


def icon_ico():
    return _existing(icons_dir() / "arrayscope.ico")


def application_icon_path():
    """Best icon for QWindow/QApplication use (PNG preferred over SVG)."""
    return icon_png(256) or icon_png(128) or icon_svg()
