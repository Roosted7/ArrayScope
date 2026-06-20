from types import SimpleNamespace

from arrayscope.display.backend_contract import (
    ImageViewBackendCapabilities,
    VISPY_CAPABILITIES,
    image_view_backend_capabilities,
)


def test_explicit_backend_capabilities_are_preserved():
    capabilities = ImageViewBackendCapabilities(name="custom", direct_montage_tile_payloads=True)

    assert image_view_backend_capabilities(SimpleNamespace(rendering_capabilities=capabilities)) is capabilities


def test_legacy_vispy_plugin_gets_conservative_hybrid_capabilities():
    capabilities = image_view_backend_capabilities(
        SimpleNamespace(rendering_backend_name="vispy", supports_direct_montage_tile_payloads=True)
    )

    assert capabilities.name == VISPY_CAPABILITIES.name
    assert capabilities.prefers_tiled_montages is True
    assert capabilities.supports_montage_canvas is False
    assert capabilities.persistent_tile_residency is True
    assert capabilities.native_pointer_interaction is False
