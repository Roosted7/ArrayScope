"""GPU topology detection for the G7 host-cache policy (Phase A).

The RAM axis of the compressed host cache (Phase A) helps on *every* device --
fitting more of the working set in a fixed RAM budget avoids recomputes/re-reads
regardless of where the pixels ultimately land.  Topology still matters for the
*policy seam*: a discrete GPU across a PCIe link additionally cares about the
host->VRAM transfer bytes (Phase B -- GPU-side decode), while an integrated GPU
with unified memory does not.  This module reports just enough topology for the
policy to (a) engage the RAM win everywhere and (b) leave a labelled seam for
Phase B to add the transfer decision.

Import health: importing this module never touches wgpu.  Detection is lazy and
wrapped so a headless/software host (no adapter, no Vulkan) returns a safe
``unknown`` topology treated as integrated / RAM-only -- the RAM win still
applies, and no code path requires a GPU to import ``arrayscope``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

__all__ = [
    "DeviceTopology",
    "detect_topology",
    "reset_topology_cache",
]


@dataclass(frozen=True)
class DeviceTopology:
    """The render device's memory topology, as the RAM/transfer policy sees it.

    ``kind`` is ``"integrated"``, ``"discrete"`` or ``"unknown"``.  ``unified_memory``
    is True when host and device share RAM (integrated) -- there is no PCIe
    transfer to save, so Phase B's transfer decision is a no-op there.
    ``discrete_transfer_candidate`` is the Phase-B seam: True only for a discrete
    device, where compressing host->VRAM bytes could pay off once a GPU-side
    decoder exists.  Phase A never reads it to decide the RAM win.
    """

    kind: str = "unknown"
    unified_memory: bool = True
    device_name: str = ""
    backend: str = ""

    @property
    def is_discrete(self) -> bool:
        return self.kind == "discrete"

    @property
    def is_integrated(self) -> bool:
        # Unknown is treated as integrated / RAM-only: the safe fallback.
        return self.kind in ("integrated", "unknown")

    @property
    def discrete_transfer_candidate(self) -> bool:
        """Phase-B seam: a discrete PCIe device whose transfer bytes could be cut.

        Phase A (RAM win) ignores this; it is here so the transfer decision has a
        typed place to live without a later signature change.
        """

        return self.kind == "discrete"


# Unknown-but-safe default: no GPU touched, RAM-only behaviour.
_UNKNOWN = DeviceTopology(kind="unknown", unified_memory=True, device_name="", backend="")

_CACHED: DeviceTopology | None = None


def _classify(adapter_type: str) -> tuple[str, bool]:
    """Map a wgpu ``adapter_type`` string to (kind, unified_memory)."""

    lowered = str(adapter_type or "").lower()
    if "discrete" in lowered:
        return "discrete", False
    if "integrated" in lowered or "cpu" in lowered or "software" in lowered:
        # CPU/software adapters have no distinct device RAM -> treat as unified.
        return "integrated", True
    if "virtual" in lowered:
        return "integrated", True
    return "unknown", True


def _adapter_info_from_shared_device() -> dict | None:
    """Reuse the process-wide wgpu device's adapter info if one already exists.

    Reads the view module's shared-device global *only if that module is already
    imported* -- if no view has been created there is no shared device to reuse,
    so we never import (and never construct a device or adapter).  This keeps
    topology detection free of side effects and avoids importing the heavy view
    module just to probe topology.
    """

    module = sys.modules.get("arrayscope.display.wgpu_imageview2d")
    if module is None:
        return None
    device = getattr(module, "_SHARED_WGPU_DEVICE", None)
    if device is None:
        return None
    try:
        return dict(device.adapter.info)
    except Exception:
        return None


def _adapter_info_from_probe() -> dict | None:
    """Cheap one-shot adapter probe -- one adapter, no device created."""

    try:
        import wgpu
    except Exception:
        return None
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="low-power")
    except Exception:
        return None
    if adapter is None:
        return None
    try:
        return dict(adapter.info)
    except Exception:
        return None


def detect_topology(*, force: bool = False) -> DeviceTopology:
    """Report the render device topology, cached, with a safe fallback.

    Prefers the already-created shared wgpu device's adapter (no new adapter);
    otherwise does a single cheap adapter probe.  Any failure (no GPU, no
    Vulkan, software host) yields the ``unknown`` topology, which the policy
    treats as integrated / RAM-only.
    """

    global _CACHED
    if _CACHED is not None and not force:
        return _CACHED

    info = _adapter_info_from_shared_device()
    if info is None:
        info = _adapter_info_from_probe()

    if not info:
        _CACHED = _UNKNOWN
        return _CACHED

    kind, unified = _classify(info.get("adapter_type", ""))
    _CACHED = DeviceTopology(
        kind=kind,
        unified_memory=unified,
        device_name=str(info.get("device", "")),
        backend=str(info.get("backend_type", "")),
    )
    return _CACHED


def reset_topology_cache() -> None:
    """Drop the cached topology (tests; adapter/device lifecycle changes)."""

    global _CACHED
    _CACHED = None
