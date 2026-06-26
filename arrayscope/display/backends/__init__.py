"""Rendering surface protocol and resolver."""

from __future__ import annotations

from arrayscope.display.backends.base import ImageSurface, RasterCommitMode, surface_for_view


__all__ = [
    "ImageSurface",
    "RasterCommitMode",
    "surface_for_view",
]
