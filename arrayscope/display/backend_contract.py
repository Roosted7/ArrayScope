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
    # Whether the backend can bind a page whose per-axis reduction differs
    # (for example y/64 with x/128 from an anisotropic viewport aspect).
    # A backend that keys its ladder as one isotropic mip chain cannot, so
    # the LOD demand is squared off before any such page key is minted.
    anisotropic_lod_pages: bool = True
    # Whether the backend applies an X/Y axis-order swap (transpose) as a pure
    # DISPLAY transform -- sampling canonically materialized tiles with a
    # swapped UV/index mapping -- instead of baking the display order into tile
    # pixels.  When True the engine keeps every payload/identity in canonical
    # (sorted-image-axes) orientation so a transpose costs the same as a flip;
    # when False the legacy path reorders pixels at materialization.
    display_axis_transpose: bool = False


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

WGPU_CAPABILITIES = ImageViewBackendCapabilities(
    name="wgpu",
    # Queue row 3(b) now binds content-keyed montage planes through the wgpu
    # page table and implements both tiled and single-plane warming.  This
    # declaration also selects the shared bounded GPU commit policy; leaving
    # the obsolete MVP value false made journey commits report no item cap.
    persistent_tile_residency=True,
    tile_residency_kind="gpu_atlas",
    shader_windowing=True,
    native_pointer_interaction=False,
    # ``plane_chunk_key`` keys every page as ``reduction=(level, level)`` and
    # the executor addresses one isotropic mip span per plane, so an
    # anisotropic page has no representable identity on this backend.
    anisotropic_lod_pages=False,
    # The vertex shader samples canonical tiles with a swapped UV walk
    # (``Tile.transposed``), so an X/Y axis-order swap rebinds existing
    # residency as a display transform instead of re-uploading reoriented tiles.
    display_axis_transpose=True,
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
            persistent_tile_residency=bool(
                getattr(capabilities, "persistent_tile_residency", False)
            ),
            tile_residency_kind=str(getattr(capabilities, "tile_residency_kind", "none") or "none"),
            shader_windowing=bool(getattr(capabilities, "shader_windowing", False)),
            native_pointer_interaction=bool(
                getattr(capabilities, "native_pointer_interaction", True)
            ),
            anisotropic_lod_pages=bool(getattr(capabilities, "anisotropic_lod_pages", True)),
            display_axis_transpose=bool(getattr(capabilities, "display_axis_transpose", False)),
        )

    return ImageViewBackendCapabilities(name="pyqtgraph")
