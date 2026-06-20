"""Backend capabilities shared by the display orchestration layer.

The viewer should ask what a rendering surface can do, not branch on a library
name.  This deliberately describes semantic behaviour rather than concrete Qt,
PyQtGraph, VisPy, OpenGL, or future Qt Quick implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageViewBackendCapabilities:
    """Capabilities that affect display planning and commit policy."""

    name: str
    direct_montage_tile_payloads: bool = False
    prefers_tiled_montages: bool = False
    supports_montage_canvas: bool = True
    persistent_tile_residency: bool = False
    shader_windowing: bool = False
    native_pointer_interaction: bool = True


PYQTGRAPH_CAPABILITIES = ImageViewBackendCapabilities(
    name="pyqtgraph",
    direct_montage_tile_payloads=True,
    prefers_tiled_montages=False,
    persistent_tile_residency=False,
    shader_windowing=False,
    native_pointer_interaction=True,
)

VISPY_CAPABILITIES = ImageViewBackendCapabilities(
    name="vispy",
    direct_montage_tile_payloads=True,
    prefers_tiled_montages=True,
    supports_montage_canvas=False,
    persistent_tile_residency=True,
    shader_windowing=True,
    # The current backend intentionally uses the shared PyQtGraph interaction
    # surface.  Marking this accurately prevents the hybrid experiment from
    # being mistaken for a fully native VisPy viewer.
    native_pointer_interaction=False,
)


def image_view_backend_capabilities(view) -> ImageViewBackendCapabilities:
    """Return capabilities for a view, with compatibility for older plugins."""

    capabilities = getattr(view, "rendering_capabilities", None)
    if isinstance(capabilities, ImageViewBackendCapabilities):
        return capabilities

    name = str(getattr(view, "rendering_backend_name", "pyqtgraph") or "pyqtgraph").lower()
    direct = bool(getattr(view, "supports_direct_montage_tile_payloads", False))
    if name == "vispy":
        return ImageViewBackendCapabilities(
            name=name,
            direct_montage_tile_payloads=direct,
            prefers_tiled_montages=True,
            supports_montage_canvas=False,
            persistent_tile_residency=True,
            shader_windowing=True,
            native_pointer_interaction=False,
        )
    return ImageViewBackendCapabilities(name=name, direct_montage_tile_payloads=direct)
