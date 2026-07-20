from types import SimpleNamespace

from arrayscope.display.backend_contract import (
    WGPU_CAPABILITIES,
    ImageViewBackendCapabilities,
    image_view_backend_capabilities,
)


def test_explicit_backend_capabilities_are_preserved():
    capabilities = ImageViewBackendCapabilities(name="custom", persistent_tile_residency=True)

    assert (
        image_view_backend_capabilities(SimpleNamespace(rendering_capabilities=capabilities))
        is capabilities
    )


def test_missing_backend_capabilities_default_to_pyqtgraph_baseline():
    capabilities = image_view_backend_capabilities(SimpleNamespace())

    assert capabilities.name == "pyqtgraph"
    assert capabilities.persistent_tile_residency is False


def test_wgpu_declares_its_live_persistent_tiled_residency_contract():
    assert WGPU_CAPABILITIES.persistent_tile_residency is True
    assert WGPU_CAPABILITIES.tile_residency_kind == "gpu_atlas"
    assert WGPU_CAPABILITIES.shader_windowing is True
