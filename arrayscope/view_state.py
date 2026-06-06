"""Qt-free state describing the current view of an n-dimensional array."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Tuple

from .axis_utils import clamp_index, non_singleton_axes, validate_axis, validate_distinct_axes


class ChannelMode(Enum):
    COMPLEX = "complex"
    REAL = "real"
    IMAG = "imag"
    ABS = "abs"
    ANGLE = "angle"


class ScaleMode(Enum):
    LINEAR = "linear"
    SYMLOG = "symlog"


def _coerce_enum(enum_type, value):
    if isinstance(value, enum_type):
        return value
    if hasattr(value, "value"):
        value = value.value
    return enum_type(value)


@dataclass(frozen=True)
class ViewState:
    ndim: int
    shape: Tuple[int, ...]
    image_axes: Optional[Tuple[int, int]]
    line_axis: Optional[int]
    slice_indices: Tuple[int, ...]
    channel: ChannelMode = ChannelMode.REAL
    scale: ScaleMode = ScaleMode.LINEAR
    axis_flipped: Tuple[bool, ...] = ()
    axis_fftshifted: Tuple[bool, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "ndim", int(self.ndim))
        object.__setattr__(self, "shape", tuple(int(size) for size in self.shape))
        object.__setattr__(self, "slice_indices", tuple(int(index) for index in self.slice_indices))
        object.__setattr__(self, "axis_flipped", tuple(bool(value) for value in self.axis_flipped))
        object.__setattr__(self, "axis_fftshifted", tuple(bool(value) for value in self.axis_fftshifted))
        object.__setattr__(self, "channel", _coerce_enum(ChannelMode, self.channel))
        object.__setattr__(self, "scale", _coerce_enum(ScaleMode, self.scale))

        if self.image_axes is not None:
            object.__setattr__(self, "image_axes", tuple(int(axis) for axis in self.image_axes))
        if self.line_axis is not None:
            object.__setattr__(self, "line_axis", int(self.line_axis))

        self.validate()

    @classmethod
    def from_shape(cls, shape):
        shape = tuple(int(size) for size in shape)
        ndim = len(shape)
        selectable_axes = list(non_singleton_axes(shape))

        if ndim >= 2:
            selected_axes = list(selectable_axes[:2])
            if len(selected_axes) < 2:
                selected_axes = [0, 1]
            image_axes = tuple(selected_axes)
        else:
            image_axes = None

        line_axis = selectable_axes[0] if selectable_axes else (0 if ndim else None)

        return cls(
            ndim=ndim,
            shape=shape,
            image_axes=image_axes,
            line_axis=line_axis,
            slice_indices=(0,) * ndim,
            axis_flipped=(False,) * ndim,
            axis_fftshifted=(False,) * ndim,
        )

    def with_slice(self, axis, index):
        axis = self._validate_axis(axis)
        index = int(index)
        self._validate_slice_index(axis, index)
        slice_indices = list(self.slice_indices)
        slice_indices[axis] = index
        return replace(self, slice_indices=tuple(slice_indices))

    def with_slice_indices(self, slice_indices):
        slice_indices = tuple(slice_indices)
        if len(slice_indices) != self.ndim:
            raise ValueError("slice_indices length must match ndim")
        slice_indices = tuple(clamp_index(self.shape, axis, index) for axis, index in enumerate(slice_indices))
        return replace(self, slice_indices=slice_indices)

    def with_image_axes(self, axis0, axis1):
        axis0, axis1 = validate_distinct_axes(self.ndim, (axis0, axis1), count=2)
        return replace(self, image_axes=(axis0, axis1))

    def with_image_axis(self, role, axis):
        if self.image_axes is None:
            raise ValueError("image_axes must be set before assigning an image axis")
        axis = self._validate_axis(axis)
        primary_axis, secondary_axis = self.image_axes
        if role == "y":
            primary_axis = axis
            if primary_axis == secondary_axis:
                secondary_axis = self._fallback_distinct_axis(primary_axis)
        elif role == "x":
            secondary_axis = axis
            if primary_axis == secondary_axis:
                primary_axis = self._fallback_distinct_axis(secondary_axis)
        else:
            raise ValueError(f"unknown image axis role: {role}")
        return self.with_image_axes(primary_axis, secondary_axis)

    def transposed_image_axes(self):
        if self.image_axes is None:
            return self
        axis0, axis1 = self.image_axes
        return self.with_image_axes(axis1, axis0)

    def with_line_axis(self, axis):
        if axis is None:
            return replace(self, line_axis=None)
        return replace(self, line_axis=self._validate_axis(axis))

    def with_channel(self, channel):
        return replace(self, channel=ChannelMode(channel))

    def with_scale(self, scale):
        return replace(self, scale=ScaleMode(scale))

    def with_axis_flipped(self, axis, flipped):
        axis = self._validate_axis(axis)
        axis_flipped = list(self.axis_flipped)
        axis_flipped[axis] = bool(flipped)
        return replace(self, axis_flipped=tuple(axis_flipped))

    def with_axis_fftshifted(self, axis, fftshifted):
        axis = self._validate_axis(axis)
        axis_fftshifted = list(self.axis_fftshifted)
        axis_fftshifted[axis] = bool(fftshifted)
        return replace(self, axis_fftshifted=tuple(axis_fftshifted))

    def for_shape(self, shape, *, preserve_flags: bool = True):
        shape = tuple(int(size) for size in shape)
        ndim = len(shape)
        migrated = ViewState.from_shape(shape)

        slice_indices = tuple(
            clamp_index(shape, axis, self.slice_indices[axis] if axis < self.ndim else 0)
            for axis in range(ndim)
        )

        if ndim >= 2 and self.image_axes is not None:
            retained = [axis for axis in self.image_axes if axis < ndim and shape[axis] != 1]
            candidates = list(non_singleton_axes(shape)) + list(range(ndim))
            for axis in candidates:
                if len(retained) >= 2:
                    break
                if axis not in retained:
                    retained.append(axis)
            image_axes = tuple(retained[:2]) if len(retained) >= 2 else migrated.image_axes
        else:
            image_axes = migrated.image_axes

        if self.line_axis is not None and self.line_axis < ndim and shape[self.line_axis] != 1:
            line_axis = self.line_axis
        else:
            line_axis = migrated.line_axis

        axis_flipped = migrated.axis_flipped
        axis_fftshifted = migrated.axis_fftshifted
        if preserve_flags:
            axis_flipped = tuple(
                bool(self.axis_flipped[axis]) if axis < len(self.axis_flipped) else False
                for axis in range(ndim)
            )
            axis_fftshifted = tuple(
                bool(self.axis_fftshifted[axis]) if axis < len(self.axis_fftshifted) else False
                for axis in range(ndim)
            )

        return ViewState(
            ndim=ndim,
            shape=shape,
            image_axes=image_axes,
            line_axis=line_axis,
            slice_indices=slice_indices,
            channel=self.channel,
            scale=self.scale,
            axis_flipped=axis_flipped,
            axis_fftshifted=axis_fftshifted,
        )

    def display_axes(self):
        if self.image_axes is not None:
            return self.image_axes
        if self.line_axis is not None:
            return (self.line_axis,)
        return ()

    def non_display_axes(self):
        display_axes = set(self.display_axes())
        return tuple(axis for axis in range(self.ndim) if axis not in display_axes)

    def validate(self):
        if self.ndim < 1:
            raise ValueError("ndim must be at least 1")
        if len(self.shape) != self.ndim:
            raise ValueError("shape length must match ndim")
        if any(size < 1 for size in self.shape):
            raise ValueError("shape sizes must be at least 1")

        if self.image_axes is not None:
            if len(self.image_axes) != 2:
                raise ValueError("image_axes must contain exactly two axes")
            axis0, axis1 = self.image_axes
            validate_distinct_axes(self.ndim, (axis0, axis1), count=2)

        if self.line_axis is not None:
            self._validate_axis(self.line_axis)

        if len(self.slice_indices) != self.ndim:
            raise ValueError("slice_indices length must match ndim")
        for axis, index in enumerate(self.slice_indices):
            self._validate_slice_index(axis, index)

        if len(self.axis_flipped) != self.ndim:
            raise ValueError("axis_flipped length must match ndim")
        if len(self.axis_fftshifted) != self.ndim:
            raise ValueError("axis_fftshifted length must match ndim")

        return self

    def _validate_axis(self, axis):
        return validate_axis(self.ndim, axis)

    def _validate_slice_index(self, axis, index):
        index = int(index)
        size = self.shape[int(axis)]
        if index < 0 or index >= size:
            raise ValueError(f"slice index {index} is out of bounds for axis {axis} with size {size}")
        return index

    def _fallback_distinct_axis(self, axis):
        candidates = list(non_singleton_axes(self.shape)) + list(range(self.ndim))
        for candidate in candidates:
            if candidate != axis:
                return candidate
        raise ValueError("image axes must be distinct")
