"""One adapter per process: both callers take the shared owner's."""

from __future__ import annotations

from typing import ClassVar

import pytest

from arrayscope.gpu import device_topology as topo
from arrayscope.gpu import wgpu_adapter


class _FakeAdapter:
    features: ClassVar[tuple] = ()
    info: ClassVar[dict] = {
        "adapter_type": "IntegratedGPU",
        "device": "Fake iGPU",
        "backend_type": "Vulkan",
    }


@pytest.fixture
def counted_adapter_requests(monkeypatch):
    """Count real adapter requests, with no GPU touched and no cache leaked."""

    # Load the native backend BEFORE patching: importing it rebinds
    # ``wgpu.gpu`` to the backend's own object, so a patch applied to the
    # placeholder would be silently discarded when the owner imports
    # ``set_instance_extras`` -- and the count would read zero for the wrong
    # reason.  Loading the library creates no instance and no adapter.
    pytest.importorskip("wgpu.backends.wgpu_native")
    wgpu = pytest.importorskip("wgpu")
    requests = []

    def _request_adapter_sync(**kwargs):
        requests.append(kwargs)
        return _FakeAdapter()

    monkeypatch.setattr(wgpu.gpu, "request_adapter_sync", _request_adapter_sync)
    # A process that already built the adapter (another GPU test) would report
    # zero requests and pass vacuously; start from "nothing asked yet".
    monkeypatch.setattr(wgpu_adapter, "_ADAPTER", None)
    monkeypatch.setattr(topo, "_CACHED", None)
    return requests


def test_display_and_topology_share_one_adapter_request(counted_adapter_requests, monkeypatch):
    """The topology probe and the display device must not enumerate separately.

    Two requests print the driver's adapter warnings twice, and the loser of
    the race gets an instance it did not configure -- the display pins
    Vulkan-only on the instance, which only takes before the first request.
    """

    view = pytest.importorskip("arrayscope.display.wgpu_imageview2d")
    monkeypatch.setattr(view, "_SHARED_WGPU_DEVICE", None)
    monkeypatch.setattr(
        _FakeAdapter, "request_device_sync", lambda self, **kwargs: "device", raising=False
    )

    probed = topo.detect_topology(force=True)
    device = view._shared_wgpu_device()

    assert len(counted_adapter_requests) == 1
    assert probed.kind == "integrated"
    assert probed.device_name == "Fake iGPU"
    assert device == "device"


def test_shared_adapter_is_requested_once_and_reused(counted_adapter_requests):
    first = wgpu_adapter.shared_wgpu_adapter()
    second = wgpu_adapter.shared_wgpu_adapter()

    assert first is second
    assert len(counted_adapter_requests) == 1
    assert counted_adapter_requests[0] == {"power_preference": "low-power"}


def test_power_preference_cannot_change_under_an_existing_adapter(counted_adapter_requests):
    """The adapter carries the profile, so a late change would misdescribe it."""

    # Same value stays legal -- it says nothing new about the adapter.
    wgpu_adapter.configure_shared_wgpu_power_preference("low-power")
    wgpu_adapter.shared_wgpu_adapter()

    with pytest.raises(RuntimeError, match="fresh benchmark process"):
        wgpu_adapter.configure_shared_wgpu_power_preference("high-performance")

    with pytest.raises(ValueError, match="unknown wgpu power preference"):
        wgpu_adapter.configure_shared_wgpu_power_preference("fastest")


def test_configured_preference_reaches_the_request(counted_adapter_requests, monkeypatch):
    monkeypatch.setattr(wgpu_adapter, "_POWER_PREFERENCE", "low-power")
    wgpu_adapter.configure_shared_wgpu_power_preference("high-performance")

    wgpu_adapter.shared_wgpu_adapter()

    assert counted_adapter_requests[0] == {"power_preference": "high-performance"}
    assert wgpu_adapter.shared_wgpu_power_preference() == "high-performance"
