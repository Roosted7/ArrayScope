"""PyQtGraph display-backend adapter."""

from arrayscope.display.backends.base import ImageViewMethodBackendAdapter


class PyQtGraphBackendAdapter(ImageViewMethodBackendAdapter):
    expected_backend_name = "pyqtgraph"
