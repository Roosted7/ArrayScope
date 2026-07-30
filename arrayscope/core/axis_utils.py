"""Shared axis and index validation helpers."""

from __future__ import annotations


def _ndim(shape_or_ndim) -> int:
    if isinstance(shape_or_ndim, int):
        return int(shape_or_ndim)
    return len(tuple(shape_or_ndim))


def validate_axis(shape_or_ndim, axis, *, label: str = "axis") -> int:
    axis = int(axis)
    ndim = _ndim(shape_or_ndim)
    if axis < 0 or axis >= ndim:
        raise ValueError(f"{label} {axis} is out of bounds for {ndim}D data")
    return axis


def validate_distinct_axes(shape_or_ndim, axes, *, count: int | None = None) -> tuple[int, ...]:
    axes = tuple(validate_axis(shape_or_ndim, axis) for axis in axes)
    if count is not None and len(axes) != int(count):
        raise ValueError(f"axes must contain exactly {count} axes")
    if len(set(axes)) != len(axes):
        raise ValueError("axes must be distinct")
    return axes


#: The arity every axis-set declaration gets unless it narrows it: at least one
#: axis, no upper bound beyond the array's own rank.
DEFAULT_AXIS_ARITY: tuple[int, int | None] = (1, None)


def normalize_axis_arity(arity) -> tuple[int, int | None]:
    """Validate and canonicalize a ``(minimum, maximum)`` axis-count pair.

    ``maximum`` may be ``None`` for "as many as the array has". A declaration
    that cannot be satisfied by any array (``minimum`` below one, or a maximum
    under the minimum) is an authoring error, not something to clamp.
    """

    minimum, maximum = tuple(arity) if arity is not None else DEFAULT_AXIS_ARITY
    minimum = int(minimum)
    maximum = None if maximum is None else int(maximum)
    if minimum < 1:
        raise ValueError(f"axis arity minimum must be at least 1, got {minimum}")
    if maximum is not None and maximum < minimum:
        raise ValueError(f"axis arity maximum {maximum} is below its minimum {minimum}")
    return (minimum, maximum)


def normalize_axis_selection(
    axes,
    *,
    ndim=None,
    arity: tuple[int, int | None] = DEFAULT_AXIS_ARITY,
    label: str = "axes",
) -> tuple[int, ...]:
    """The canonical form of an axis SET: sorted, unique, in range, in arity.

    This is the one place an axis-set selection is admitted, so a recipe, a
    palette invocation, a popup edit and a direct ``create_operation`` call
    cannot disagree about what "axes 2 and 0" means. Ordering is normalized
    because an axis set has no meaningful order -- ``(2, 0)`` and ``(0, 2)``
    name the same set, and letting both through would make two identical
    operations compare unequal and miss each other's cache entries.

    ``ndim`` is optional because a selection is often admitted before the array
    it applies to is known (``create_operation`` from a palette, say); range
    checking then falls to the operation's own ``output_shape``/``apply``,
    exactly as it does for a single ``axis``. When ``ndim`` *is* known the check
    happens here, at the earliest point that can name the mistake.
    """

    minimum, maximum = normalize_axis_arity(arity)
    if axes is None:
        raise ValueError(f"{label} must be given")
    if isinstance(axes, (int, bool)):
        raise TypeError(f"{label} must be a sequence of axes, not a single {type(axes).__name__}")
    resolved: list[int] = []
    for axis in tuple(axes):
        axis = int(axis)
        if axis < 0:
            # A negative axis is only meaningful against a known rank, and every
            # other axis surface in ArrayScope refuses one; refusing it here
            # keeps one convention rather than two.
            raise ValueError(f"{label} must be non-negative, got {axis}")
        if ndim is not None and axis >= int(ndim):
            raise ValueError(f"{label} entry {axis} is out of bounds for {int(ndim)}D data")
        if axis in resolved:
            raise ValueError(f"{label} contains axis {axis} more than once")
        resolved.append(axis)
    if len(resolved) < minimum:
        raise ValueError(f"{label} needs at least {minimum} axis/axes, got {len(resolved)}")
    if maximum is not None and len(resolved) > maximum:
        raise ValueError(f"{label} accepts at most {maximum} axis/axes, got {len(resolved)}")
    return tuple(sorted(resolved))


def clamp_index(shape, axis, index) -> int:
    shape = tuple(int(size) for size in shape)
    axis = validate_axis(shape, axis)
    size = shape[axis]
    if size < 1:
        raise ValueError(f"axis {axis} has invalid size {size}")
    return max(0, min(int(index), size - 1))


def non_singleton_axes(shape) -> tuple[int, ...]:
    return tuple(axis for axis, size in enumerate(tuple(shape)) if int(size) != 1)
