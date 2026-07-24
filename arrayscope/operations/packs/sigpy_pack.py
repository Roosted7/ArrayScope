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
- ``sigpy:circshift`` -- **circular shift** of one axis by an integer amount
  (``sigpy.circshift``, equivalent to ``numpy.roll`` along the axis; the shift may
  be negative).  Additive over the built-ins: ``reverse`` mirrors an axis and
  ``fftshift`` rolls by exactly half, but there is no general roll-by-k.  It is
  shape- and dtype-preserving, but **not** windowable -- a circular shift wraps
  samples around the axis boundary, so ``fn(whole)[region] != fn(whole[region])``.
  Tier-1 OPAQUE.
- ``sigpy:downsample`` -- **strided decimation** of one axis by an integer factor
  (``sigpy.downsample``; ``input[..., ::factor, ...]``, *no* anti-alias filter --
  honest naming: it decimates, it does not low-pass first).  Shape-changing
  (``ceil(n / factor)``), dtype-preserving.  Additive: ``crop`` takes a contiguous
  window, never a strided subsample.  Tier-1 OPAQUE.
- ``sigpy:upsample`` -- **zero-insertion upsample** of one axis by an integer
  factor (``sigpy.upsample``; the exact adjoint of ``downsample`` -- it scatters
  the samples to every ``factor``-th position of a zero array).  Shape-changing
  (``n * factor``), dtype-preserving.  Tier-1 OPAQUE.  ``downsample`` +
  ``upsample`` form a natural strided pair.

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
CIRCSHIFT_ID = "sigpy:circshift"
DOWNSAMPLE_ID = "sigpy:downsample"
UPSAMPLE_ID = "sigpy:upsample"


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


def _build_circshift(axis: int | None, params: Mapping[str, object]) -> Callable[[object], object]:
    """Circular shift of ``axis`` by an integer ``shift`` (numpy.roll semantics)."""

    resolved_axis = int(axis)  # requires_axis=True guarantees a bound axis
    shift = int(params["shift"])

    def fn(data):
        import sigpy as sp

        data = np.asarray(data)
        ax = resolved_axis % data.ndim
        # sigpy.circshift(input, shifts, axes): both are per-listed-axis lists.
        return sp.circshift(data, [shift], axes=[ax])

    return fn


def _axis_factors(factor: int, axis: int, ndim: int) -> list[int]:
    """Per-axis factor list selecting a single axis (``1`` everywhere else)."""

    factors = [1] * ndim
    factors[axis % ndim] = factor
    return factors


def _build_downsample(axis: int | None, params: Mapping[str, object]) -> Callable[[object], object]:
    """Strided decimation of ``axis`` by ``factor`` (``input[..., ::factor, ...]``)."""

    resolved_axis = int(axis)
    factor = int(params["factor"])

    def fn(data):
        import sigpy as sp

        data = np.asarray(data)
        return sp.downsample(data, _axis_factors(factor, resolved_axis, data.ndim))

    return fn


def _build_upsample(axis: int | None, params: Mapping[str, object]) -> Callable[[object], object]:
    """Zero-insertion upsample of ``axis`` by ``factor`` (adjoint of downsample)."""

    resolved_axis = int(axis)
    factor = int(params["factor"])

    def fn(data):
        import sigpy as sp

        data = np.asarray(data)
        ax = resolved_axis % data.ndim
        oshape = list(data.shape)
        oshape[ax] = int(data.shape[ax]) * factor
        return sp.upsample(data, tuple(oshape), _axis_factors(factor, ax, data.ndim))

    return fn


def _downsample_output_shape(shape, axis, params):
    """Output shape: ``shape[axis]`` -> ``ceil(shape[axis] / factor)``.

    Exactly the length of ``range(0, n, factor)`` -- the number of samples the
    strided slice ``input[..., ::factor, ...]`` keeps.
    """

    factor = int(params["factor"])
    ax = int(axis) % len(shape)
    out = [int(s) for s in shape]
    out[ax] = (out[ax] + factor - 1) // factor
    return tuple(out)


def _upsample_output_shape(shape, axis, params):
    """Output shape: ``shape[axis]`` -> ``shape[axis] * factor``."""

    factor = int(params["factor"])
    ax = int(axis) % len(shape)
    out = [int(s) for s in shape]
    out[ax] = out[ax] * factor
    return tuple(out)


# --- specs -------------------------------------------------------------------


def soft_thresh_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=SOFT_THRESH_ID,
        label="Soft threshold (sigpy)",
        build=_build_thresh("soft"),
        output_dtype=_thresh_output_dtype,
        parameters=(
            OperationParameter(
                "lamda",
                "Threshold λ",
                kind="float",
                default=0.0,
                minimum=0.0,
                step=0.01,
                description="Magnitude shrinkage: sign(x)·max(|x|-λ, 0).",
            ),
        ),
        requires_axis=False,
        changes_shape=False,
        # Strictly pointwise (magnitude shrinkage) -> a genuine Tier-2 windowable
        # claim, honored only after the conformance harness proves it.
        region_capable=True,
        group="SigPy",
        description="L1 soft-threshold (magnitude shrinkage) every sample.",
        icon="filter_alt",
    )


def hard_thresh_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=HARD_THRESH_ID,
        label="Hard threshold (sigpy)",
        build=_build_thresh("hard"),
        output_dtype=_thresh_output_dtype,
        parameters=(
            OperationParameter(
                "lamda",
                "Threshold λ",
                kind="float",
                default=0.0,
                minimum=0.0,
                step=0.01,
                description="Keep samples with |x| > λ, zero the rest.",
            ),
        ),
        requires_axis=False,
        changes_shape=False,
        # Strictly pointwise (keep-or-kill by magnitude) -> Tier-2 windowable.
        region_capable=True,
        group="SigPy",
        description="Keep samples above magnitude λ, zero the rest.",
        icon="filter_alt",
    )


def resize_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=RESIZE_ID,
        label="Resize / zero-pad axis (sigpy)",
        build=_build_resize,
        output_shape=_resize_output_shape,
        # sigpy.resize allocates in the input dtype -> identity dtype adapter.
        parameters=(
            OperationParameter(
                "size",
                "Target size",
                kind="int",
                minimum=1,
                description="Centered zero-pad (grow) or center-crop (shrink) to this length.",
            ),
        ),
        requires_axis=True,
        changes_shape=True,
        # Centered resize re-indexes the whole axis and changes shape -> OPAQUE.
        region_capable=False,
        group="SigPy",
        description="Centered zero-pad / center-crop one axis to a target length.",
        icon="aspect_ratio",
    )


def circshift_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=CIRCSHIFT_ID,
        label="Circular shift axis (sigpy)",
        build=_build_circshift,
        # circshift is a pure reindex -> dtype-preserving (identity adapter).
        parameters=(
            OperationParameter(
                "shift",
                "Shift",
                kind="int",
                default=0,
                step=1,
                description="Roll the axis by this many samples (negative rolls the other way).",
            ),
        ),
        requires_axis=True,
        changes_shape=False,
        # A circular shift wraps samples across the axis boundary, so it does not
        # commute with sub-region reads -> never a Tier-2 windowable claim.
        region_capable=False,
        group="SigPy",
        description="Circularly shift (roll) one axis by an integer number of samples.",
        icon="sync",
    )


def downsample_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=DOWNSAMPLE_ID,
        label="Downsample axis (sigpy)",
        build=_build_downsample,
        output_shape=_downsample_output_shape,
        # Strided slice -> dtype-preserving (identity adapter).
        parameters=(
            OperationParameter(
                "factor",
                "Factor",
                kind="int",
                default=2,
                minimum=1,
                step=1,
                description="Keep every factor-th sample along the axis (no anti-alias filter).",
            ),
        ),
        requires_axis=True,
        changes_shape=True,
        # Strided decimation re-indexes the whole axis and changes shape -> OPAQUE.
        region_capable=False,
        group="SigPy",
        description="Strided decimation of one axis by an integer factor (no filtering).",
        icon="compress",
    )


def upsample_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=UPSAMPLE_ID,
        label="Upsample axis (sigpy)",
        build=_build_upsample,
        output_shape=_upsample_output_shape,
        # Zero-insertion scatter -> dtype-preserving (identity adapter).
        parameters=(
            OperationParameter(
                "factor",
                "Factor",
                kind="int",
                default=2,
                minimum=1,
                step=1,
                description="Scatter each sample to every factor-th slot; fill the rest with zeros.",
            ),
        ),
        requires_axis=True,
        changes_shape=True,
        # Zero-insertion re-indexes the whole axis and changes shape -> OPAQUE.
        region_capable=False,
        group="SigPy",
        description="Zero-insertion upsample of one axis by an integer factor (downsample adjoint).",
        icon="expand",
    )


def pack_specs() -> tuple[PluginOperationSpec, ...]:
    """The specs this pack contributes (independent of sigpy being installed)."""

    return (
        soft_thresh_spec(),
        hard_thresh_spec(),
        resize_spec(),
        circshift_spec(),
        downsample_spec(),
        upsample_spec(),
    )


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
