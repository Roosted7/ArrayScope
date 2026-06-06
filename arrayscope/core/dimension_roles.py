"""Pure dimension-role state helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DimensionRoles:
    image_axes: Tuple[int, int]
    profile_axes: Tuple[int, ...]

    @classmethod
    def from_axes(cls, image_axes, profile_axes=()):
        image_axes = tuple(int(axis) for axis in image_axes)
        if len(image_axes) != 2 or image_axes[0] == image_axes[1]:
            raise ValueError("image axes must contain two distinct axes")
        profile_axes = tuple(dict.fromkeys(int(axis) for axis in profile_axes))
        return cls(image_axes=image_axes, profile_axes=profile_axes)

    def with_image_axis(self, role, axis):
        axis = int(axis)
        y_axis, x_axis = self.image_axes
        if role == "y":
            y_axis = axis
            if y_axis == x_axis:
                x_axis = self.image_axes[0]
        elif role == "x":
            x_axis = axis
            if x_axis == y_axis:
                y_axis = self.image_axes[1]
        else:
            raise ValueError(f"unknown image role: {role}")
        if y_axis == x_axis:
            raise ValueError("image axes must stay distinct")
        return DimensionRoles.from_axes((y_axis, x_axis), self.profile_axes)

    def with_single_profile_axis(self, axis):
        return DimensionRoles.from_axes(self.image_axes, (int(axis),))

    def with_toggled_profile_axis(self, axis):
        axis = int(axis)
        profile_axes = list(self.profile_axes)
        if axis in profile_axes:
            profile_axes.remove(axis)
        else:
            profile_axes.append(axis)
        return DimensionRoles.from_axes(self.image_axes, profile_axes)
