"""The one wgpu adapter of the process.

Two callers used to request their own: the display's shared device
(``arrayscope.display.wgpu_imageview2d._shared_wgpu_device``) and the G7
topology probe (``arrayscope.gpu.device_topology``).  Two requests are not
merely wasteful -- whichever ran first created the global wgpu *instance*, and
the display's ``set_instance_extras(backends=["Vulkan"])`` pin only takes on an
instance that does not exist yet.  With the topology probe (settings load)
running before the first view, the pin quietly became a no-op and the GL
backend was back in the instance, which gate-B Tier 0 records as fatal under
Wayland.  Each request also enumerates adapters, so the driver's warnings
printed once per requester.

So the adapter is created here, once, with the Vulkan pin applied first, and
both callers take it.  This module lives in ``arrayscope.gpu`` because the
dependency may only point display -> gpu (see the package docstring); it holds
no Qt and no display state -- an adapter, not a device.

Import health: importing this module never touches wgpu.
"""

from __future__ import annotations

import contextlib
import threading

__all__ = [
    "configure_shared_wgpu_power_preference",
    "shared_wgpu_adapter",
    "shared_wgpu_power_preference",
]

_ADAPTER = None
_ADAPTER_LOCK = threading.Lock()
_POWER_PREFERENCE = "low-power"


def shared_wgpu_power_preference() -> str:
    """The power preference the shared adapter was (or will be) requested with."""

    return _POWER_PREFERENCE


def configure_shared_wgpu_power_preference(power_preference: str) -> None:
    """Select the adapter profile before it exists (benchmark processes only).

    Refuses to change the preference once the adapter has been created: the
    adapter carries it, so a later change would describe hardware that is not
    the one in use.  A benchmark that wants the other profile needs a fresh
    process.
    """

    global _POWER_PREFERENCE
    value = str(power_preference)
    if value not in {"low-power", "high-performance"}:
        raise ValueError(f"unknown wgpu power preference {power_preference!r}")
    if _ADAPTER is not None and value != _POWER_PREFERENCE:
        raise RuntimeError("the shared wgpu adapter already exists; use a fresh benchmark process")
    _POWER_PREFERENCE = value


def shared_wgpu_adapter():
    """Request (once) and return the process's wgpu adapter.

    Raises whatever wgpu raises on a host with no usable adapter -- callers
    decide what a missing GPU means for them (the AUTO backend probe reports
    it; the topology probe falls back to ``unknown``).
    """

    global _ADAPTER
    # Fast path: one adapter per process, reused everywhere.
    if _ADAPTER is not None:
        return _ADAPTER
    with _ADAPTER_LOCK:
        # Re-check under the lock: the startup warm-up thread may have finished
        # creating it while we waited.
        if _ADAPTER is not None:
            return _ADAPTER
        import wgpu
        from wgpu.backends.wgpu_native.extras import set_instance_extras

        # Now that wgpu owns its logger, the launch-time noise hook has
        # something to attach to.  It deliberately did nothing earlier:
        # creating the "wgpu" logger before this import breaks wgpu's own
        # logger-class assertion.  Here is still before the adapter request
        # below, which is what emits the messages being filtered.
        from arrayscope.app.log_noise import install_wgpu_log_noise_filter

        install_wgpu_log_noise_filter()

        with contextlib.suppress(RuntimeError):
            # Vulkan-only instance: the GL backend's EGL re-init is fatal under
            # Wayland (gate-B Tier 0).  Must precede the process's first adapter
            # request -- once the instance exists this raises, and the pin is
            # lost.  Being the single requester is what makes that ordering hold.
            set_instance_extras(backends=["Vulkan"])
        _ADAPTER = wgpu.gpu.request_adapter_sync(power_preference=_POWER_PREFERENCE)
    return _ADAPTER
