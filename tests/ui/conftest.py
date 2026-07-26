import pytest


@pytest.fixture(autouse=True)
def _require_parametrized_wgpu_adapter(request):
    """Keep backend-matrix cases honest when this host has no Vulkan device.

    A test parametrized as ``backend="wgpu"`` must exercise WGPU, not continue
    through a late device failure or a PyQtGraph fallback. Direct WGPU tests use
    ``require_wgpu_adapter`` themselves.
    """

    callspec = getattr(request.node, "callspec", None)
    if callspec is not None and callspec.params.get("backend") == "wgpu":
        from tests.ui.helpers import require_wgpu_adapter

        require_wgpu_adapter()
