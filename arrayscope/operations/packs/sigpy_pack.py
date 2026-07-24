"""Optional in-process sigpy operation pack (thresholding + centered resize).

This pack ships a *small, honest* set of `sigpy <https://sigpy.readthedocs.io>`_
operations that add value ArrayScope does **not** already have, wrapped as
in-process pack ops (the same machinery as the BART pack and entry-point
plugins).  It is **optional**: it self-guards on ``import sigpy`` -- when sigpy is
absent the pack registers nothing, so importing ArrayScope never imports sigpy
and import-health stays green.  Availability is decided with
:func:`importlib.util.find_spec` (a metadata check that does *not* import sigpy);
the heavy ``import sigpy`` happens lazily inside each op's bound callable, only
when the op is actually applied.  Building an op or enumerating the registry
therefore costs nothing beyond a ``find_spec`` lookup.

Why *these* ops (and not an FFT).  An earlier sigpy pack shipped ``sigpy:fft`` /
``sigpy:ifft`` and was removed as redundant: ArrayScope already has a built-in
centered FFT op (``centered_fft`` / ``centered_ifft``) *and* a pluggable FFT
backend setting, and ``sigpy.fft`` is ``numpy.fft`` underneath -- it duplicated
capability without adding any (see docs/graveyard.md).  So this pack deliberately
ships **no FFT**.  It ships the genuinely-additive, unary-contract-fitting
sigpy operations instead:

- ``sigpy:soft_thresh`` -- pointwise complex **soft** thresholding (magnitude
  shrinkage): ``sign(x) * max(|x| - lamda, 0)``.  The workhorse sparsity /
  denoising primitive for MRI (the proximal operator of the L1 norm).  As a view
  stage it lets you *see* the effect of an L1 shrink on image- or wavelet-domain
  data.  **Tier-2 windowable**: it is strictly pointwise, so
  ``fn(whole)[region] == fn(whole[region])`` on every axis -- and we let the
  conformance harness *prove* that rather than assert it.
- ``sigpy:hard_thresh`` -- pointwise complex **hard** thresholding: keep samples
  whose magnitude exceeds ``lamda``, zero the rest.  A sparsifying / support-view
  companion to soft_thresh.  Also strictly pointwise -> **Tier-2 windowable**.
- ``sigpy:resize`` -- **centered** zero-pad / center-crop of one axis to a target
  length (``sigpy.resize``).  This is the canonical k-space *zero-fill*
  interpolation (pad k-space -> sinc-interpolate the image) and its inverse
  center-crop.  It is genuinely additive: the built-in ``crop`` op only *shrinks*
  an axis by an explicit ``[start:stop]`` window and does not center; ``resize``
  additionally *grows* an axis (zero-fill) and always centers the content.
  Shape-changing -> **Tier-1 OPAQUE** (a centered resize re-indexes the whole
  axis; it is not windowable, and the conformance harness rejects any
  shape-changer anyway).

Capability decisions are honest by design -- the Tier-2 conformance harness in
``arrayscope.operations.plugin_conformance`` exists precisely to catch a false
windowable claim, so the two threshold ops make the claim *and stand behind it*
(the gate property-tests ``fn(whole)[region] == fn(whole[region])`` before it is
honored; a failure would downgrade the op to OPAQUE, never show wrong pixels).

**Numeric precision (float32 discipline).**  sigpy's ``soft_thresh`` /
``hard_thresh`` internally promote to ``complex128`` and always return
``complex128`` regardless of input dtype.  Blindly returning that would silently
double every float32/complex64 array's footprint -- against the repo's
numeric-precision narrowing effort.  So the threshold ops **narrow the result
back** to match the input: complex stays complex (``complex64`` <-> ``complex128``
by width), real floats stay real floats (``float32``/``float64``), and other real
inputs (integers/bool) narrow to ``float32``.  The narrowing cast is itself
pointwise, so it does not disturb the windowable property the harness checks.
``sigpy:resize`` preserves the input dtype (``sigpy.resize`` allocates in the
input dtype), so it needs no dtype adapter.

Deferred sigpy operations (unchanged from the original deferral -- they do not
fit the ``fn(ndarray) -> ndarray`` + scalar-parameter unary contract without
engine changes, so they are out of scope rather than shipped fragile):

- ``sigpy.nufft`` / ``sigpy.nufft_adjoint(input, coord, ...)`` -- needs a k-space
  *coordinate array* as a second argument.  The plugin parameter model carries
  scalars (int/float), not a companion ndarray, so there is no honest way to bind
  ``coord`` through the recipe/dock contract.
- ``sigpy.mri.app.EspiritCalib(ksp, ...)`` -- needs coil-axis + calibration
  semantics and a compute device, and it *changes dimensionality* (produces
  sensitivity maps) in a way the scalar-param shape adapter cannot predict.  It
  is an iterative app object, not a pure ``fn(ndarray) -> ndarray``.
- ``sigpy.fwt`` / ``sigpy.iwt`` (wavelet transform pair) -- a natural "reversible
  view stage" candidate, but ``iwt`` requires the *original* ``oshape`` **and**
  the ``coeff_slices`` structure that ``fwt`` produced in order to invert.  The
  scalar-parameter model cannot carry that structural metadata between two
  independent unary steps, so the forward/inverse pair cannot be expressed
  honestly (the inverse would have to guess the forward's packing).  Deferred for
  the same reason nufft/espirit are: correctness over coverage.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping

import numpy as np

from arrayscope.operations.plugins import PluginOperationSpec
from arrayscope.operations.registry import OperationParameter, register_pack_operation

# Namespaced, collision-safe ids (must contain the plugin namespace separator).
SOFT_THRESH_ID = "sigpy:soft_thresh"
HARD_THRESH_ID = "sigpy:hard_thresh"
RESIZE_ID = "sigpy:resize"


# --- availability (cheap, lazy: never imports sigpy) -------------------------


def sigpy_available() -> bool:
    """Whether sigpy is importable, without importing it.

    Uses :func:`importlib.util.find_spec` so a mere availability check
    (registration, enumeration) never pays the cost of importing sigpy and never
    trips import-health.  The heavy ``import sigpy`` is deferred into each op's
    bound callable (below), so it runs only when an op is actually applied.
    """

    return importlib.util.find_spec("sigpy") is not None


# --- dtype narrowing (float32 discipline) ------------------------------------


def _thresh_output_dtype(input_dtype) -> np.dtype:
    """Narrow sigpy's always-complex128 threshold output back to the input.

    sigpy's ``soft_thresh`` / ``hard_thresh`` return ``complex128`` for every
    input.  Thresholding is magnitude-based, so the honest output type mirrors the
    input: complex stays complex (by width), real floats stay real floats, and
    other real inputs narrow to ``float32`` (the repo's narrowing default -- these
    are display/denoise views, not exact-integer arithmetic).
    """

    dt = np.dtype(input_dtype)
    if dt.kind == "c":
        return np.dtype("complex64") if dt.itemsize <= 8 else np.dtype("complex128")
    if dt == np.dtype("float64"):
        return np.dtype("float64")
    if dt == np.dtype("float32"):
        return np.dtype("float32")
    # integer / bool / other real -> narrow to float32.
    return np.dtype("float32")


# --- op fn builders (heavy ``import sigpy`` deferred to here) -----------------


def _build_thresh(kind: str) -> Callable[[int | None, Mapping[str, object]], Callable]:
    """Return a ``build(axis, params)`` for the soft/hard threshold op.

    ``kind`` selects ``sigpy.soft_thresh`` or ``sigpy.hard_thresh``.  The bound fn
    is strictly pointwise, and narrows the always-complex128 sigpy result back to
    the input-appropriate dtype (see :func:`_thresh_output_dtype`).  The narrowing
    cast is itself pointwise, so it preserves the windowable property.
    """

    def build(axis: int | None, params: Mapping[str, object]) -> Callable[[object], object]:
        del axis  # thresholds are global/pointwise -> no axis
        lamda = float(params["lamda"])  # defensive coercion (recipe/CLI robustness)

        def fn(data):
            import sigpy as sp

            data = np.asarray(data)
            out_dtype = _thresh_output_dtype(data.dtype)
            thresh_fn = sp.soft_thresh if kind == "soft" else sp.hard_thresh
            result = np.asarray(thresh_fn(lamda, data))  # complex128
            if out_dtype.kind != "c":
                result = result.real
            return result.astype(out_dtype, copy=False)

        return fn

    return build


def _build_resize(axis: int | None, params: Mapping[str, object]) -> Callable[[object], object]:
    """Centered zero-pad / center-crop of ``axis`` to a target ``size``."""

    resolved_axis = int(axis)  # requires_axis=True guarantees a bound axis
    size = int(params["size"])

    def fn(data):
        import sigpy as sp

        data = np.asarray(data)
        ax = resolved_axis % data.ndim
        oshape = list(data.shape)
        oshape[ax] = size
        return sp.resize(data, tuple(oshape))

    return fn


def _resize_output_shape(shape, axis, params):
    """Output shape: replace ``shape[axis]`` with the target ``size``."""

    size = int(params["size"])
    ax = int(axis) % len(shape)
    out = [int(s) for s in shape]
    out[ax] = size
    return tuple(out)


# --- specs -------------------------------------------------------------------


def soft_thresh_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=SOFT_THRESH_ID,
        label="Soft threshold (sigpy)",
        build=_build_thresh("soft"),
        output_dtype=_thresh_output_dtype,
        parameters=(OperationParameter("lamda", "Threshold λ", kind="float"),),
        requires_axis=False,
        changes_shape=False,
        # Strictly pointwise (magnitude shrinkage) -> a genuine Tier-2 windowable
        # claim, honored only after the conformance harness proves it.
        region_capable=True,
    )


def hard_thresh_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=HARD_THRESH_ID,
        label="Hard threshold (sigpy)",
        build=_build_thresh("hard"),
        output_dtype=_thresh_output_dtype,
        parameters=(OperationParameter("lamda", "Threshold λ", kind="float"),),
        requires_axis=False,
        changes_shape=False,
        # Strictly pointwise (keep-or-kill by magnitude) -> Tier-2 windowable.
        region_capable=True,
    )


def resize_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=RESIZE_ID,
        label="Resize / zero-pad axis (sigpy)",
        build=_build_resize,
        output_shape=_resize_output_shape,
        # sigpy.resize allocates in the input dtype -> identity dtype adapter.
        parameters=(OperationParameter("size", "Target size", kind="int"),),
        requires_axis=True,
        changes_shape=True,
        # Centered resize re-indexes the whole axis and changes shape -> OPAQUE.
        region_capable=False,
    )


def pack_specs() -> tuple[PluginOperationSpec, ...]:
    """The specs this pack contributes (independent of sigpy being installed)."""

    return (soft_thresh_spec(), hard_thresh_spec(), resize_spec())


def register(register_fn=register_pack_operation) -> bool:
    """Register the sigpy ops iff sigpy is importable.

    Returns ``True`` when the pack contributed its ops, ``False`` when sigpy is
    absent (the pack silently contributes nothing).  Idempotent.  Called by
    :func:`arrayscope.operations.registry.load_operation_packs`.
    """

    if not sigpy_available():
        return False
    for spec in pack_specs():
        register_fn(spec)
    return True
