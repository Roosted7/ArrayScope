"""
arrayscope - Interactive N-dimensional array viewer with FFT support
"""

from .qt_binding import prefer_pyside6

prefer_pyside6()

from .imageview2d import ImageView2D
from .launch import arrayscope
from .window import ArrayScopeWindow, Domain

__version__ = "0.0.1"
__all__ = ["arrayscope", "ArrayScopeWindow", "ImageView2D", "Domain"]
