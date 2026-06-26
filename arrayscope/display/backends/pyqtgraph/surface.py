"""PyQtGraph image surface."""

from arrayscope.display.imageview2d import ImageView2D


class PyQtGraphSurface(ImageView2D):
    """Concrete PyQtGraph image surface used by the shared display shell."""

    surface_kind = "pyqtgraph"

    def interaction_event_owner(self) -> str:
        return "pyqtgraph"

__all__ = ["PyQtGraphSurface"]
