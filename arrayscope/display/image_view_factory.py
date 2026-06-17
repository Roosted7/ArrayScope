"""Image-view backend factory."""

from __future__ import annotations

from arrayscope.app.settings_state import ImageRenderingBackendChoice
from arrayscope.display.imageview2d import ImageView2D


def create_image_view(settings=None, *, notify=None):
    """Create the selected image view implementation.

    PyQtGraph remains the stable default.  The VisPy backend is intentionally
    optional and selected only when the user asks for it; this keeps CI and
    non-OpenGL environments usable while giving us a clean experiment path.
    """

    choice = getattr(settings, "image_rendering_backend", ImageRenderingBackendChoice.PYQTGRAPH)
    if getattr(choice, "value", choice) == ImageRenderingBackendChoice.VISPY.value:
        try:
            from arrayscope.display.vispy_imageview2d import VisPyImageView2D

            return VisPyImageView2D()
        except Exception as exc:
            if callable(notify):
                notify(f"VisPy renderer unavailable; using PyQtGraph ({exc})")
    return ImageView2D()
