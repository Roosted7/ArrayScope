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


def normalize_capabilities(capabilities: OperationCapabilities, *, ndim: int) -> OperationCapabilities:
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
    )


def default_chunkable_axes(kind: OperationKind, *, ndim: int, blocking_axes=()) -> tuple[int, ...]:
    kind = _normalize_kind(kind)
    ndim = int(ndim)
    blocked = set(_normalize_axes(blocking_axes, ndim=ndim))
    if kind == OperationKind.TRANSFORM:
        return tuple(axis for axis in range(ndim) if axis not in blocked)
    if kind in {OperationKind.VIEW, OperationKind.ELEMENTWISE, OperationKind.RESHAPE, OperationKind.REDUCTION}:
        return tuple(axis for axis in range(ndim) if axis not in blocked)
    return ()


def pipeline_commutes_for_display_lod(operations, base_shape, base_dtype=None) -> bool:
    """True when every operation may take box-mean-reduced display input.

    The reduce-before-ops path (ADR 0050) is valid only when the ENTIRE
    pipeline commutes: one non-commuting stage makes reduced input change
    the result.  Shape-changing steps are additionally rejected because the
    display-axis identification below the reduction would no longer match
    the native region plan.  Conservative by construction: unknown or
    capability-less operations return False.
    """

    shape = tuple(int(size) for size in base_shape)
    dtype = base_dtype
    for operation in tuple(operations or ()):
        capabilities = getattr(operation, "capabilities", None)
        if not callable(capabilities):
            return False
        if tuple(operation.output_shape(shape)) != shape:
            return False
        if not bool(getattr(capabilities(shape, dtype), "lod_commuting", False)):
            return False
        output_dtype = getattr(operation, "output_dtype", None)
        if callable(output_dtype):
            dtype = output_dtype(dtype)
    return True


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
