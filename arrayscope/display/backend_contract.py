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
    persistent_tile_residency: bool = False
    tile_residency_kind: str = "none"
    shader_windowing: bool = False
    native_pointer_interaction: bool = True


PYQTGRAPH_CAPABILITIES = ImageViewBackendCapabilities(
    name="pyqtgraph",
    persistent_tile_residency=True,
    tile_residency_kind="cpu_item",
    shader_windowing=False,
    native_pointer_interaction=True,
)

VISPY_CAPABILITIES = ImageViewBackendCapabilities(
    name="vispy",
    persistent_tile_residency=True,
    tile_residency_kind="gpu_atlas",
    shader_windowing=True,
    # The current backend intentionally uses the shared PyQtGraph interaction
    # surface.  Marking this accurately prevents the hybrid experiment from
    # being mistaken for a fully native VisPy viewer.
    native_pointer_interaction=False,
)


def image_view_backend_capabilities(view) -> ImageViewBackendCapabilities:
    """Return capabilities for a view or its composed image surface."""

    if view is None:
        return ImageViewBackendCapabilities(name="pyqtgraph")
    surface = getattr(view, "surface", None)
    surface_capabilities = getattr(surface, "capabilities", None)
    if isinstance(surface_capabilities, ImageViewBackendCapabilities):
        return surface_capabilities
    capabilities = getattr(view, "rendering_capabilities", None)
    if isinstance(capabilities, ImageViewBackendCapabilities):
        return capabilities
    if capabilities is not None and hasattr(capabilities, "name"):
        return ImageViewBackendCapabilities(
            name=str(getattr(capabilities, "name", "pyqtgraph") or "pyqtgraph"),
            persistent_tile_residency=bool(getattr(capabilities, "persistent_tile_residency", False)),
            tile_residency_kind=str(getattr(capabilities, "tile_residency_kind", "none") or "none"),
            shader_windowing=bool(getattr(capabilities, "shader_windowing", False)),
            native_pointer_interaction=bool(getattr(capabilities, "native_pointer_interaction", True)),
        )

    return ImageViewBackendCapabilities(name="pyqtgraph")
