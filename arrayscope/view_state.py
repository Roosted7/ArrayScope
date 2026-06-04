"""Qt-free state describing the current view of an n-dimensional array."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Tuple


class ChannelMode(Enum):
    COMPLEX = "complex"
    REAL = "real"
    IMAG = "imag"
    ABS = "abs"
    ANGLE = "angle"


class ScaleMode(Enum):
    LINEAR = "linear"
    SYMLOG = "symlog"


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
        object.__setattr__(self, "channel", ChannelMode(self.channel))
        object.__setattr__(self, "scale", ScaleMode(self.scale))

        if self.image_axes is not None:
            object.__setattr__(self, "image_axes", tuple(int(axis) for axis in self.image_axes))
        if self.line_axis is not None:
            object.__setattr__(self, "line_axis", int(self.line_axis))

        self.validate()

    @classmethod
    def from_shape(cls, shape):
        shape = tuple(int(size) for size in shape)
        ndim = len(shape)
        non_singleton_axes = [axis for axis, size in enumerate(shape) if size != 1]

        if ndim >= 2:
            selected_axes = list(non_singleton_axes[:2])
            if len(selected_axes) < 2:
                selected_axes = [0, 1]
            image_axes = tuple(selected_axes)
        else:
            image_axes = None

        line_axis = non_singleton_axes[0] if non_singleton_axes else (0 if ndim else None)

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

    def with_image_axes(self, axis0, axis1):
        axis0 = self._validate_axis(axis0)
        axis1 = self._validate_axis(axis1)
        if axis0 == axis1:
            raise ValueError("image axes must be distinct")
        return replace(self, image_axes=(axis0, axis1))

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
            self._validate_axis(axis0)
            self._validate_axis(axis1)
            if axis0 == axis1:
                raise ValueError("image axes must be distinct")

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
        axis = int(axis)
        if axis < 0 or axis >= self.ndim:
            raise ValueError(f"axis {axis} is out of bounds for {self.ndim}D shape")
        return axis

    def _validate_slice_index(self, axis, index):
        index = int(index)
        size = self.shape[int(axis)]
        if index < 0 or index >= size:
            raise ValueError(f"slice index {index} is out of bounds for axis {axis} with size {size}")
        return index
