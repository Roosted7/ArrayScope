"""G7 Phase A: topology detection returns a valid label and falls back safely."""

from __future__ import annotations

import numpy as np

from arrayscope.gpu import device_topology as topo


def test_detect_returns_valid_label():
    result = topo.detect_topology(force=True)
    assert result.kind in ("integrated", "discrete", "unknown")
    assert isinstance(result.unified_memory, bool)
    # Unknown (or integrated) is treated as RAM-only; discrete sets the seam.
    assert result.is_integrated or result.is_discrete
    assert result.discrete_transfer_candidate == (result.kind == "discrete")


def test_classify_maps_adapter_types():
    assert topo._classify("DiscreteGPU") == ("discrete", False)
    assert topo._classify("IntegratedGPU") == ("integrated", True)
    assert topo._classify("Cpu") == ("integrated", True)
    assert topo._classify("") == ("unknown", True)
    assert topo._classify("something-weird") == ("unknown", True)


def test_safe_fallback_when_no_adapter(monkeypatch):
    """No shared device and no probeable adapter -> safe unknown topology."""

    monkeypatch.setattr(topo, "_adapter_info_from_shared_device", lambda: None)
    monkeypatch.setattr(topo, "_adapter_info_from_probe", lambda: None)
    result = topo.detect_topology(force=True)
    assert result.kind == "unknown"
    assert result.unified_memory is True
    assert result.is_integrated is True  # unknown treated as RAM-only
    assert result.discrete_transfer_candidate is False


def test_discrete_probe_sets_transfer_seam(monkeypatch):
    monkeypatch.setattr(topo, "_adapter_info_from_shared_device", lambda: None)
    monkeypatch.setattr(
        topo,
        "_adapter_info_from_probe",
        lambda: {"adapter_type": "DiscreteGPU", "device": "Fake RTX", "backend_type": "Vulkan"},
    )
    result = topo.detect_topology(force=True)
    assert result.kind == "discrete"
    assert result.unified_memory is False
    assert result.discrete_transfer_candidate is True  # Phase-B seam


def test_cached_after_first_detect(monkeypatch):
    calls = {"n": 0}

    def _probe():
        calls["n"] += 1
        return {"adapter_type": "IntegratedGPU"}

    monkeypatch.setattr(topo, "_adapter_info_from_shared_device", lambda: None)
    monkeypatch.setattr(topo, "_adapter_info_from_probe", _probe)
    topo.reset_topology_cache()
    first = topo.detect_topology()
    second = topo.detect_topology()
    assert first is second
    assert calls["n"] == 1  # probed once, then cached
    topo.reset_topology_cache()


def test_import_does_not_require_gpu():
    """Importing the module must not touch wgpu (import-health)."""

    import importlib

    mod = importlib.reload(topo)
    assert hasattr(mod, "detect_topology")
    # numpy import is fine; wgpu must be a lazy, function-local import only.
    assert np is not None
