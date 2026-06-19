"""VisPy display-backend adapter."""

from arrayscope.display.backends.base import ImageViewMethodBackendAdapter


class VisPyBackendAdapter(ImageViewMethodBackendAdapter):
    expected_backend_name = "vispy"
