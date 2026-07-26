"""Pure operation capability declarations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OperationKind(Enum):
    VIEW = "view"
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    TRANSFORM = "transform"
    RESHAPE = "reshape"


class OperationClass(Enum):
    """Execution class of an operation (tensor-engine endpoint proposal).

    The class states what an operation *is*, so backends can decide where it
    runs: coordinate metadata never materializes an array (a flip is an index
    map), shader-on-read work happens at sampling time, derived-chunk compute
    produces content-keyed chunks, reductions return small results, global
    transforms are cost-model territory (CPU vs GPU), and opaque operations
    always CPU-materialize. Nothing maps to DERIVED_CHUNK_COMPUTE yet — it
    exists so LOD/smoothing kernels land in a named class, not a special
    case.
    """

    COORDINATE_METADATA = "coordinate_metadata"
    SHADER_ON_READ = "shader_on_read"
    DERIVED_CHUNK_COMPUTE = "derived_chunk_compute"
    REDUCTION = "reduction"
    GLOBAL_TRANSFORM = "global_transform"
    OPAQUE = "opaque"


@dataclass(frozen=True)
class OperationCapabilities:
    kind: OperationKind
    blocking_axes: tuple[int, ...] = ()
    chunkable_axes: tuple[int, ...] = ()
    expands_request_axes: tuple[int, ...] = ()
    cache_stage: bool = False
    temp_multiplier: float = 1.0
    can_fuse: bool = False
    notes: tuple[str, ...] = ()
    # ADR 0050 display-LOD contract: True only when box-mean downsampling of
    # the display axes commutes acceptably with this operation FOR DISPLAY
    # (pointwise value maps such as conjugate/abs/phase/component select and
    # scalar arithmetic).  Domain transforms (FFT) and anything that moves or
    # mixes samples across the display axes must stay False.  Only display
    # payload derivations may consult this flag; exact consumers always use
    # the native pipeline.
    lod_commuting: bool = False
    # True when the operation is a linear map of its input samples over REAL
    # scalars: f(a*x + b*y) == a*f(x) + b*f(y) for real a and b.  Box-mean
    # reduction is an average with real, non-negative weights, so a real-linear
    # operation commutes with it EXACTLY (to float error) on every axis the
    # operation does not touch.  Real rather than complex linearity is the
    # property that matters here: conjugation is antilinear over C yet still
    # commutes with a real-weighted average.
    #
    # Unlike `lod_commuting` this flag says nothing on its own about display
    # axes -- it is half of a per-axis question.  `pipeline_commutes_for_
    # display_lod` combines it with the axes the operation declares.
    real_linear: bool = False


def normalize_capabilities(
    capabilities: OperationCapabilities, *, ndim: int
) -> OperationCapabilities:
    ndim = int(ndim)
    return OperationCapabilities(
        kind=_normalize_kind(capabilities.kind),
        blocking_axes=_normalize_axes(capabilities.blocking_axes, ndim=ndim),
        chunkable_axes=_normalize_axes(capabilities.chunkable_axes, ndim=ndim),
        expands_request_axes=_normalize_axes(capabilities.expands_request_axes, ndim=ndim),
        cache_stage=bool(capabilities.cache_stage),
        temp_multiplier=float(capabilities.temp_multiplier),
        can_fuse=bool(capabilities.can_fuse),
        notes=tuple(str(note) for note in capabilities.notes),
        lod_commuting=bool(capabilities.lod_commuting),
        real_linear=bool(capabilities.real_linear),
    )


def default_chunkable_axes(kind: OperationKind, *, ndim: int, blocking_axes=()) -> tuple[int, ...]:
    kind = _normalize_kind(kind)
    ndim = int(ndim)
    blocked = set(_normalize_axes(blocking_axes, ndim=ndim))
    if kind == OperationKind.TRANSFORM:
        return tuple(axis for axis in range(ndim) if axis not in blocked)
    if kind in {
        OperationKind.VIEW,
        OperationKind.ELEMENTWISE,
        OperationKind.RESHAPE,
        OperationKind.REDUCTION,
    }:
        return tuple(axis for axis in range(ndim) if axis not in blocked)
    return ()


def pipeline_commutes_for_display_lod(
    operations, base_shape, base_dtype=None, *, display_axes=()
) -> bool:
    """True when every operation may take box-mean-reduced display input.

    The reduce-before-ops path (ADR 0050) is valid only when the ENTIRE
    pipeline commutes: one non-commuting stage makes reduced input change
    the result.  Shape-changing steps are additionally rejected because the
    display-axis identification below the reduction would no longer match
    the native region plan.  Conservative by construction: unknown or
    capability-less operations return False.

    Commuting is a question about the operation *and* the display axes, so a
    stage may earn its licence two independent ways:

    - ``lod_commuting`` -- the axis-blind licence for pointwise value maps
      (conjugate and friends).  A pointwise map cannot care which axes are
      displayed, so this holds whatever ``display_axes`` says.
    - real linearity off the display axes -- a stage that declares
      ``real_linear`` and touches no display axis (through ``blocking_axes``,
      ``expands_request_axes``, or its own ``axis``/``axes``) acts
      independently within each display-axis position.  Box-mean reduction
      averages *across* those positions with real, non-negative weights, so
      reduce-then-apply and apply-then-reduce are the same array to float
      error.  ``CenteredFFT(axis=2)`` under display axes (0, 1) -- an FFT
      along the montage axis -- is the case this exists for, and it is exact,
      not a quality compromise.

    Linearity is not optional and axis-disjointness alone is not enough: a
    nonlinear per-line map (a magnitude along the line, say) does not survive
    an average taken across lines, however disjoint its axis is.  Nor is
    ``OperationKind.TRANSFORM`` evidence of it -- the kind says the stage is a
    global transform, not that it is linear.  Only the declared flag counts.

    ``display_axes=()`` means the display axes are not known at the call site;
    only the axis-blind licence then applies, which is the pre-2026-07 answer.
    A display axis out of range for ``base_shape`` is a caller mismatch and
    disqualifies the pipeline.

    A REDUCTION stage disqualifies the pipeline outright.  Reductions are
    shape-changing here and so already rejected, but the check is explicit
    because a reduction is the one kind whose result depends on how many
    samples went into it -- exactly what a display reduction changes.

    The guarantee is only as strong as the weakest licence in the chain: an
    all-``real_linear`` chain is exact, while mixing in a ``lod_commuting``
    stage inherits that flag's weaker "acceptable for display" contract.
    """

    shape = tuple(int(size) for size in base_shape)
    display = frozenset(int(axis) for axis in tuple(display_axes or ()))
    if any(axis < 0 or axis >= len(shape) for axis in display):
        return False
    dtype = base_dtype
    for operation in tuple(operations or ()):
        capabilities = getattr(operation, "capabilities", None)
        output_shape = getattr(operation, "output_shape", None)
        if not callable(capabilities) or not callable(output_shape):
            return False
        if tuple(int(size) for size in output_shape(shape)) != shape:
            return False
        caps = normalize_capabilities(capabilities(shape, dtype), ndim=len(shape))
        if caps.kind is OperationKind.REDUCTION:
            return False
        commutes_off_display_axes = bool(
            display
            and caps.real_linear
            and display.isdisjoint(_operation_touched_axes(operation, caps))
        )
        if not caps.lod_commuting and not commutes_off_display_axes:
            return False
        output_dtype = getattr(operation, "output_dtype", None)
        if callable(output_dtype):
            dtype = output_dtype(dtype)
    return True


def pipeline_supports_reduced_display_lod(operations, base_shape, base_dtype=None) -> bool:
    """True when display axes may be reduced before evaluating this pipeline.

    This is broader than ``pipeline_commutes_for_display_lod``.  It answers a
    display-quality question: may the operation run on reduced input to produce
    a lower-LOD presentation?  Shape-preserving transforms such as FFT are
    allowed even though they do not mathematically commute with box reduction;
    exact/native consumers still use the native pipeline.

    It takes no ``display_axes``, deliberately.  Its permissiveness is
    axis-independent by construction -- a transform along a *display* axis is
    still a legitimate ``quality="preview"`` presentation, which is the whole
    point of being broader.  It did accept a ``display_axes`` argument and
    never read it; that left two apparent owners of the axis question, only
    one of which answered it.  ``pipeline_commutes_for_display_lod`` is now
    that one owner.
    """

    shape = tuple(int(size) for size in base_shape)
    dtype = base_dtype
    for operation in tuple(operations or ()):
        capabilities = getattr(operation, "capabilities", None)
        output_shape = getattr(operation, "output_shape", None)
        if not callable(capabilities) or not callable(output_shape):
            return False
        next_shape = tuple(int(size) for size in output_shape(shape))
        if next_shape != shape:
            return False
        caps = normalize_capabilities(capabilities(shape, dtype), ndim=len(shape))
        if caps.kind is OperationKind.REDUCTION:
            return False
        output_dtype = getattr(operation, "output_dtype", None)
        if callable(output_dtype):
            dtype = output_dtype(dtype)
        shape = next_shape
    return True


def pipeline_windowable_display_axes(
    operations, base_shape, base_dtype=None, *, display_axes=()
) -> tuple[int, ...]:
    """Display axes whose subset window commutes with the whole pipeline.

    ADR 0055 G3 window-shift fast path: resident chunks may be reused across
    a display-axis window shift (``X=100:200 → 101:201``) only when
    ``chain(data)[window] == chain(data[window])`` on that axis — otherwise a
    shift is genuinely new content (FFT along a displayed axis being the
    canonical case) and must re-evaluate.

    Conservative by construction (a wrong inclusion shows plausible-but-wrong
    pixels at interactive speed):

    - unknown or capability-less operations disqualify every axis;
    - any shape-changing step disqualifies every axis (axis identity below
      the chain would no longer match the source grid);
    - an axis in any step's ``expands_request_axes`` is out (the step needs
      the full axis to produce a sub-window);
    - a step that declares the axis is accepted only when ELEMENTWISE:
      coordinate-remapping view steps (reverse, fftshift) are windowable in
      principle but excluded until source-anchored chunk keys apply their
      coordinate maps.

    An empty chain returns every display axis — the raw-view case that the
    v1 fast path targets.
    """

    shape = tuple(int(size) for size in base_shape)
    candidates = list(dict.fromkeys(int(axis) for axis in tuple(display_axes or ())))
    dtype = base_dtype
    for operation in tuple(operations or ()):
        if not candidates:
            return ()
        capabilities = getattr(operation, "capabilities", None)
        output_shape = getattr(operation, "output_shape", None)
        if not callable(capabilities) or not callable(output_shape):
            return ()
        if tuple(int(size) for size in output_shape(shape)) != shape:
            return ()
        caps = normalize_capabilities(capabilities(shape, dtype), ndim=len(shape))
        declared = set(_operation_declared_axes(operation))
        expanded = set(caps.expands_request_axes)
        candidates = [
            axis
            for axis in candidates
            if axis not in expanded
            and (axis not in declared or caps.kind is OperationKind.ELEMENTWISE)
        ]
        output_dtype = getattr(operation, "output_dtype", None)
        if callable(output_dtype):
            dtype = output_dtype(dtype)
    return tuple(candidates)


def operation_execution_class(operation, base_shape, base_dtype=None) -> OperationClass:
    """Execution class of one operation, derived from its capability kind.

    Conservative by construction: anything without a capability declaration
    is OPAQUE (CPU materialization). VIEW and RESHAPE steps are coordinate
    metadata — they move or relabel samples without arithmetic, so a backend
    may express them as index transforms instead of copies. ELEMENTWISE steps
    are shader-on-read candidates; whether a given backend actually fuses
    them stays a backend decision — the class only rules out *needing*
    materialization. Only the declared kind is consulted (reshape steps
    legitimately declare output-ndim axes, so full normalization against the
    input shape is not applicable here).
    """

    capabilities = getattr(operation, "capabilities", None)
    if not callable(capabilities):
        return OperationClass.OPAQUE
    try:
        declared = capabilities(tuple(int(size) for size in base_shape), base_dtype)
        kind = _normalize_kind(declared.kind)
    except Exception:
        return OperationClass.OPAQUE
    if kind in (OperationKind.VIEW, OperationKind.RESHAPE):
        return OperationClass.COORDINATE_METADATA
    if kind is OperationKind.ELEMENTWISE:
        return OperationClass.SHADER_ON_READ
    if kind is OperationKind.REDUCTION:
        return OperationClass.REDUCTION
    if kind is OperationKind.TRANSFORM:
        return OperationClass.GLOBAL_TRANSFORM
    return OperationClass.OPAQUE


def _normalize_kind(kind) -> OperationKind:
    if isinstance(kind, OperationKind):
        return kind
    value = getattr(kind, "value", kind)
    return OperationKind(value)


def _normalize_axes(axes, *, ndim: int) -> tuple[int, ...]:
    result = []
    for axis in tuple(axes or ()):
        axis = int(axis)
        if axis < 0 or axis >= int(ndim):
            raise ValueError(f"axis {axis} is out of range for ndim {ndim}")
        if axis not in result:
            result.append(axis)
    return tuple(result)


def _operation_touched_axes(operation, capabilities: OperationCapabilities) -> frozenset[int]:
    """Axes an operation may read or write beyond the sample's own position.

    The union of what the capability declares (``blocking_axes``,
    ``expands_request_axes``) and what the operation names itself
    (``axis``/``axes``).  The self-named axes matter on their own: a VIEW step
    such as ``FFTShift`` or ``ReverseAxis`` blocks and expands nothing, yet it
    permutes coordinates along its axis, and a permutation only commutes with
    box-mean blocking when the block boundaries line up.
    """

    axes = set(capabilities.blocking_axes)
    axes.update(capabilities.expands_request_axes)
    axes.update(_operation_declared_axes(operation))
    return frozenset(axes)


def _operation_declared_axes(operation) -> tuple[int, ...]:
    axes = []
    for name in ("axis", "axes"):
        if not hasattr(operation, name):
            continue
        value = getattr(operation, name)
        if value is None:
            continue
        if isinstance(value, (tuple, list)):
            axes.extend(int(axis) for axis in value)
        else:
            axes.append(int(value))
    return tuple(dict.fromkeys(axes))
