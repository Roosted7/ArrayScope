"""Optional in-process sigpy operation pack (centered FFT / IFFT).

This pack is **optional**.  It self-guards on ``import sigpy``: when sigpy is not
installed the pack registers nothing, so importing ArrayScope never imports
sigpy and import-health stays green.  Availability is decided with
:func:`importlib.util.find_spec` (a cheap metadata check that does *not* import
sigpy); the heavy ``import sigpy`` happens lazily, inside each op's bound
callable, only when the op is actually applied.  Building an op or enumerating
the registry therefore costs nothing beyond a find_spec lookup.

Capability decisions (honest by design -- the Tier-2 conformance harness in
``arrayscope.operations.plugin_conformance`` exists precisely to catch a false
claim, so we do not make one):

- **FFT / IFFT** are **OPAQUE / Tier-1** (``region_capable=False``).  A centered
  FFT along an axis is a GLOBAL transform on that axis: every output sample
  depends on every input sample along the transformed axis, so
  ``fft(whole)[region] != fft(whole[region])`` across that axis -- the op is
  *not* windowable.  Declaring ``region_capable=True`` would be a false claim
  that the harness would reject and downgrade; we declare OPAQUE up front, which
  is both honest and correct.  (The op is still shape-preserving, so the whole
  built-in OPAQUE plugin path -- materialize whole, take the requested slab --
  serves it faithfully.)

Deferred sigpy operations (they do not fit the ``fn(ndarray) -> ndarray`` +
scalar-parameter plugin contract without engine changes, so they are out of
scope for v1 rather than shipped fragile):

- ``sigpy.nufft_adjoint(input, coord, ...)`` -- requires a k-space *coordinate
  array* as a second argument.  The plugin parameter model carries scalars
  (int/float), not a companion ndarray, so there is no honest way to bind
  ``coord`` through the recipe/dock contract.
- ``sigpy.mri.app.EspiritCalib(ksp, ...)`` -- needs coil-axis + calibration
  semantics, a compute device, and it *changes dimensionality* (produces
  sensitivity maps) in a way the scalar-param shape adapter cannot predict.  It
  is an iterative app object, not a pure ``fn(ndarray) -> ndarray``.

Both are documented in ``docs/plugin-operations.md`` under "sigpy pack".
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping

import numpy as np

from arrayscope.operations.plugins import PluginOperationSpec
from arrayscope.operations.registry import register_pack_operation

# Namespaced, collision-safe ids (must contain the plugin namespace separator).
FFT_ID = "sigpy:fft"
IFFT_ID = "sigpy:ifft"


def sigpy_available() -> bool:
    """Whether sigpy is importable, without importing it.

    Uses ``find_spec`` so a mere availability check (registration, enumeration)
    never pays the cost of importing sigpy and never trips import-health.
    """

    return importlib.util.find_spec("sigpy") is not None


def _fft_output_dtype(input_dtype):
    """Dtype sigpy's centered FFT/IFFT produces for a given input dtype.

    Mirrors sigpy 0.1.27 observed behaviour: complex128 input stays complex128;
    every other input dtype (real or complex64) yields complex64.
    """

    if input_dtype is None:
        return None
    dt = np.dtype(input_dtype)
    return np.dtype(np.complex128) if dt == np.dtype(np.complex128) else np.dtype(np.complex64)


def _build_fft(axis: int | None, params: Mapping[str, object]) -> Callable[[object], object]:
    del params
    resolved_axis = int(axis)  # requires_axis=True guarantees a bound axis

    def fn(data):
        import sigpy  # lazy: sigpy is imported only when the op is applied

        return sigpy.fft(np.asarray(data), axes=(resolved_axis,))

    return fn


def _build_ifft(axis: int | None, params: Mapping[str, object]) -> Callable[[object], object]:
    del params
    resolved_axis = int(axis)

    def fn(data):
        import sigpy  # lazy

        return sigpy.ifft(np.asarray(data), axes=(resolved_axis,))

    return fn


def fft_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=FFT_ID,
        label="Centered FFT (sigpy)",
        build=_build_fft,
        output_dtype=_fft_output_dtype,
        requires_axis=True,
        changes_shape=False,
        # Global transform along the axis -> NOT windowable -> OPAQUE / Tier-1.
        region_capable=False,
    )


def ifft_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=IFFT_ID,
        label="Centered iFFT (sigpy)",
        build=_build_ifft,
        output_dtype=_fft_output_dtype,
        requires_axis=True,
        changes_shape=False,
        region_capable=False,
    )


def pack_specs() -> tuple[PluginOperationSpec, ...]:
    """The specs this pack contributes (independent of sigpy being installed)."""

    return (fft_spec(), ifft_spec())


def register(register_fn=register_pack_operation) -> bool:
    """Register the sigpy ops if (and only if) sigpy is importable.

    Returns ``True`` when the pack contributed its ops, ``False`` when sigpy is
    absent (the pack silently contributes nothing).  Idempotent: re-registering
    the same namespaced ids is a no-op overwrite.  Called by
    :func:`arrayscope.operations.registry.load_operation_packs`.
    """

    if not sigpy_available():
        return False
    for spec in pack_specs():
        register_fn(spec)
    return True
