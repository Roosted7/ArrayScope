"""Rendering backend adapters and factory."""

from __future__ import annotations

from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.backends.base import ImageRenderBackend, ImageViewMethodBackendAdapter, RasterCommitMode
from arrayscope.display.backends.pyqtgraph import PyQtGraphBackendAdapter
from arrayscope.display.backends.vispy import VisPyBackendAdapter


def backend_adapter_for_view(view) -> ImageRenderBackend:
    """Return the semantic backend adapter attached to ``view``.

    Custom/plugin views may provide ``render_backend_adapter`` directly. The
    built-in views are wrapped according to declared capabilities rather than
    by importing or type-checking concrete widget classes.
    """

    attached = getattr(view, "render_backend_adapter", None)
    if isinstance(attached, ImageRenderBackend):
        return attached

    capabilities = image_view_backend_capabilities(view)
    if capabilities.name == "vispy":
        adapter: ImageRenderBackend = VisPyBackendAdapter(view)
    elif capabilities.name == "pyqtgraph":
        adapter = PyQtGraphBackendAdapter(view)
    else:
        # Third-party backends can keep the legacy ImageView2D-compatible
        # boundary during migration without pretending to be a built-in one.
        adapter = ImageViewMethodBackendAdapter(view)

    try:
        setattr(view, "render_backend_adapter", adapter)
    except Exception:
        pass
    return adapter


__all__ = [
    "ImageRenderBackend",
    "ImageViewMethodBackendAdapter",
    "PyQtGraphBackendAdapter",
    "RasterCommitMode",
    "VisPyBackendAdapter",
    "backend_adapter_for_view",
]
