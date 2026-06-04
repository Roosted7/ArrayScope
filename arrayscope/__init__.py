"""
arrayscope - Interactive N-dimensional array viewer with FFT support
"""

from .qt_binding import prefer_pyside6

prefer_pyside6()

from .arrayscope import arrayscope, ArrayScopeWindow, Domain
from .imageview2d import ImageView2D

__version__ = "0.0.1"
__all__ = ["arrayscope", "ArrayScopeWindow", "ImageView2D", "Domain"]
