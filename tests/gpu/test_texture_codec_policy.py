"""G7 Phase B: topology-aware texture-codec policy (offscreen, no GPU).

Pins the ``discrete_transfer_candidate`` seam wiring: discrete -> BC (NVIDIA has
no ASTC), integrated -> ASTC when supported else BC, and default OFF (the render
path stays byte-identical unless a caller opts in).
"""

from __future__ import annotations

from arrayscope.gpu.cache_policy import decide_texture_codec
from arrayscope.gpu.device_topology import DeviceTopology

_DISCRETE = DeviceTopology(kind="discrete", unified_memory=False, device_name="NVIDIA", backend="Vulkan")
_INTEGRATED = DeviceTopology(kind="integrated", unified_memory=True, device_name="Intel", backend="Vulkan")
_UNKNOWN = DeviceTopology(kind="unknown", unified_memory=True)


def test_default_is_off_and_render_path_is_byte_identical():
    for topo in (_DISCRETE, _INTEGRATED, _UNKNOWN):
        d = decide_texture_codec(topology=topo)  # enable defaults False
        assert d.engage is False
        assert d.family == "none"
        assert d.scalar_format == "r32float"
        assert d.complex_format == "rg32float"


def test_discrete_prefers_bc_and_sets_transfer_candidate():
    d = decide_texture_codec(topology=_DISCRETE, enable=True, astc_supported=True)
    # even with astc "supported", a discrete NVIDIA device uses BC (no ASTC there)
    assert d.engage is True
    assert d.family == "bc"
    assert d.scalar_format == "bc4-r-unorm"
    assert d.complex_format == "bc5-rg-unorm"
    assert d.discrete_transfer_candidate is True


def test_integrated_prefers_astc_when_supported():
    d = decide_texture_codec(topology=_INTEGRATED, enable=True, astc_supported=True, astc_block=(6, 6))
    assert d.engage is True
    assert d.family == "astc"
    assert d.scalar_format == "astc-6x6-unorm"
    assert d.astc_block == (6, 6)
    assert d.discrete_transfer_candidate is False  # unified memory: no PCIe to cut


def test_integrated_falls_back_to_bc_without_astc():
    d = decide_texture_codec(topology=_INTEGRATED, enable=True, astc_supported=False)
    assert d.engage is True
    assert d.family == "bc"
    assert d.scalar_format == "bc4-r-unorm"
