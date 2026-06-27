from types import SimpleNamespace

from arrayscope.display.backend_contract import ImageViewBackendCapabilities, image_view_backend_capabilities


def test_explicit_backend_capabilities_are_preserved():
    capabilities = ImageViewBackendCapabilities(name="custom", persistent_tile_residency=True)

    assert image_view_backend_capabilities(SimpleNamespace(rendering_capabilities=capabilities)) is capabilities


def test_missing_backend_capabilities_default_to_pyqtgraph_baseline():
    capabilities = image_view_backend_capabilities(SimpleNamespace())

    assert capabilities.name == "pyqtgraph"
    assert capabilities.persistent_tile_residency is False
