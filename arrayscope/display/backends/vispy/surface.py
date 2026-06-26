"""VisPy image surface."""

from arrayscope.display.vispy_imageview2d import VisPyImageView2D


class VisPySurface(VisPyImageView2D):
    """Concrete VisPy image surface used by the shared display shell."""

    surface_kind = "vispy"

__all__ = ["VisPySurface"]
