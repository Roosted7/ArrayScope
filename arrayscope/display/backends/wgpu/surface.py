"""wgpu image surface."""

from arrayscope.display.wgpu_imageview2d import WgpuImageView2D


class WgpuSurface(WgpuImageView2D):
    """Concrete wgpu image surface used by the shared display shell."""

    surface_kind = "wgpu"

__all__ = ["WgpuSurface"]
