"""Qt-free axis identity metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class AxisInfo:
    id: str
    label: str
    size: int
    unit: str | None = None
    coordinate: str | None = None
    source_index: int | None = None
    spacing: float | None = None
    origin: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "size", int(self.size))
        if self.size < 1:
            raise ValueError("axis size must be at least 1")
        if self.unit is not None:
            object.__setattr__(self, "unit", str(self.unit))
        if self.coordinate is not None:
            object.__setattr__(self, "coordinate", str(self.coordinate))
        if self.source_index is not None:
            object.__setattr__(self, "source_index", int(self.source_index))
        if self.spacing is not None:
            spacing = float(self.spacing)
            if spacing == 0.0:
                raise ValueError("axis spacing must be nonzero")
            object.__setattr__(self, "spacing", spacing)
        if self.origin is not None:
            object.__setattr__(self, "origin", float(self.origin))


AxisInfoTuple = tuple[AxisInfo, ...]


def default_axes(shape) -> AxisInfoTuple:
    return tuple(
        AxisInfo(id=f"axis-{axis}", label=f"Dim {axis}", size=int(size), source_index=axis)
        for axis, size in enumerate(shape)
    )


def default_label(index) -> str:
    return f"Dim {int(index)}"


def has_custom_label(axis_info, index) -> bool:
    return bool(axis_info.label) and axis_info.label != default_label(index)


def axis_display_name(axis_info, index) -> str:
    """Short name for compact UI surfaces: custom label if set, else the position."""
    if axis_info is not None and has_custom_label(axis_info, index):
        return axis_info.label
    return str(int(index))


def axis_metadata_summary(axis_info) -> str:
    """Human-readable one-per-line metadata summary for tooltips/status surfaces."""
    parts = [f"{axis_info.label} [{axis_info.size}]"]
    if axis_info.unit is not None:
        parts.append(f"unit: {axis_info.unit}")
    if axis_info.spacing is not None:
        spacing_text = f"spacing: {axis_info.spacing:g}"
        if axis_info.unit is not None:
            spacing_text += f" {axis_info.unit}"
        parts.append(spacing_text)
    if axis_info.origin is not None:
        origin_text = f"origin: {axis_info.origin:g}"
        if axis_info.unit is not None:
            origin_text += f" {axis_info.unit}"
        parts.append(origin_text)
    if axis_info.coordinate is not None:
        parts.append(f"coordinate: {axis_info.coordinate}")
    return "\n".join(parts)


def axes_for_shape(axes, shape) -> AxisInfoTuple:
    shape = tuple(int(size) for size in shape)
    if axes is None:
        return default_axes(shape)
    axes = tuple(_coerce_axis_info(axis) for axis in axes)
    if len(axes) != len(shape):
        raise ValueError("axis metadata length must match shape length")
    return tuple(replace(axis, size=size) for axis, size in zip(axes, shape, strict=False))


def output_axes_for_operations(axes, operations) -> AxisInfoTuple:
    result = tuple(axes)
    for operation in operations:
        result = output_axes_for_operation(result, operation)
    return result


def output_axes_for_operation(axes, operation) -> AxisInfoTuple:
    name = type(operation).__name__
    if name in {"Crop"}:
        axis = _axis(operation, axes)
        return _replace_axis(
            axes, axis, _cropped_axis(axes[axis], int(operation.start), int(operation.stop))
        )
    if name in {"ReverseAxis"}:
        axis = _axis(operation, axes)
        return _replace_axis(axes, axis, _reversed_axis(axes[axis]))
    if name in {"FFTShift", "Roll"}:
        # Samples are rotated, so index->coordinate mapping is no longer affine.
        axis = _axis(operation, axes)
        return _replace_axis(axes, axis, replace(axes[axis], spacing=None, origin=None))
    if name in {"CenteredFFT", "CenteredIFFT"}:
        # The axis moves to a reciprocal domain; physical unit/spacing no longer apply.
        axis = _axis(operation, axes)
        return _replace_axis(axes, axis, replace(axes[axis], unit=None, spacing=None, origin=None))
    if name in {
        "Clip",
        "Conjugate",
        "CumulativeSum",
        "Gradient",
        "HardThreshold",
        "ImaginaryPart",
        "LogMagnitude",
        "Magnitude",
        "Normalize",
        "Offset",
        "Phase",
        "Power",
        "RealPart",
        "Scale",
        "SoftThreshold",
    }:
        return tuple(axes)
    if name in {
        "Maximum",
        "Mean",
        "Median",
        "Minimum",
        "Percentile",
        "RootSumSquares",
        "StandardDeviation",
        "Sum",
        "Variance",
    }:
        axis = _axis(operation, axes)
        return tuple(axis_info for index, axis_info in enumerate(axes) if index != axis)
    if name in {"Difference"}:
        axis = _axis(operation, axes)
        source = axes[axis]
        origin = source.origin
        if origin is not None and source.spacing is not None:
            origin += source.spacing / 2
        return _replace_axis(axes, axis, replace(source, size=source.size - 1, origin=origin))
    if name in {"Pad"}:
        axis = _axis(operation, axes)
        source = axes[axis]
        origin = source.origin
        if origin is not None and source.spacing is not None:
            origin -= int(operation.before) * source.spacing
        return _replace_axis(
            axes,
            axis,
            replace(
                source,
                size=source.size + int(operation.before) + int(operation.after),
                origin=origin,
            ),
        )
    if name in {"Resample"}:
        axis = _axis(operation, axes)
        source = axes[axis]
        size = operation.output_shape(tuple(item.size for item in axes))[axis]
        spacing = source.spacing
        if spacing is not None and size > 1 and source.size > 1:
            spacing *= (source.size - 1) / (size - 1)
        return _replace_axis(axes, axis, replace(source, size=size, spacing=spacing))
    if name in {"Squeeze"}:
        axis = _axis(operation, axes)
        return tuple(axis_info for index, axis_info in enumerate(axes) if index != axis)
    if name in {"Transpose"}:
        axis = _axis(operation, axes)
        other_axis = int(operation.other_axis) % len(axes)
        result = list(axes)
        result[axis], result[other_axis] = result[other_axis], result[axis]
        return tuple(result)
    if name in {"CombineRealImagAxis"}:
        axis = _axis(operation, axes)
        return _replace_axis(
            axes,
            axis,
            replace(axes[axis], size=1, coordinate="complex", unit=None, spacing=None, origin=None),
        )
    if name in {"SplitComplexAxis"}:
        axis = _axis(operation, axes)
        return _replace_axis(
            axes,
            axis,
            replace(
                axes[axis], size=2, coordinate="real-imag", unit=None, spacing=None, origin=None
            ),
        )
    return tuple(axes)


def _cropped_axis(axis_info, start, stop) -> AxisInfo:
    origin = axis_info.origin
    if origin is not None and axis_info.spacing is not None:
        origin = origin + start * axis_info.spacing
    return replace(axis_info, size=stop - start, origin=origin)


def _reversed_axis(axis_info) -> AxisInfo:
    spacing = axis_info.spacing
    origin = axis_info.origin
    if origin is not None and spacing is not None:
        origin = origin + (axis_info.size - 1) * spacing
    if spacing is not None:
        spacing = -spacing
    return replace(axis_info, spacing=spacing, origin=origin)


def _coerce_axis_info(axis) -> AxisInfo:
    if isinstance(axis, AxisInfo):
        return axis
    if isinstance(axis, dict):
        return AxisInfo(**axis)
    raise TypeError(f"unsupported axis metadata: {axis!r}")


def _axis(operation, axes) -> int:
    axis = int(operation.axis)
    if axis < 0 or axis >= len(axes):
        raise ValueError(f"axis {axis} is out of bounds for {len(axes)}D metadata")
    return axis


def _replace_axis(axes, axis, axis_info):
    result = list(axes)
    result[int(axis)] = axis_info
    return tuple(result)
