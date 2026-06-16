"""Pure display-coordinate mapping for image and montage views."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

from arrayscope.core.view_state import ViewState


@dataclass(frozen=True)
class MontageGeometry:
    indices: tuple[int, ...]
    tile_shape: tuple[int, int]
    columns: int
    rows: int
    gap: int = 1

    def __post_init__(self):
        indices = tuple(int(index) for index in self.indices)
        tile_shape = tuple(int(size) for size in self.tile_shape)
        if len(tile_shape) != 2:
            raise ValueError("tile_shape must be (height, width)")
        if tile_shape[0] < 1 or tile_shape[1] < 1:
            raise ValueError("tile dimensions must be at least 1")
        columns = max(1, int(self.columns))
        rows = max(0, int(self.rows))
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "tile_shape", tile_shape)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "gap", max(0, int(self.gap)))

    @property
    def tile_height(self) -> int:
        return self.tile_shape[0]

    @property
    def tile_width(self) -> int:
        return self.tile_shape[1]

    def __getitem__(self, key):
        # Compatibility for older smoke tests and callers while render.py is
        # migrated away from dict-shaped montage geometry.
        aliases = {
            "indices": self.indices,
            "tile_height": self.tile_height,
            "tile_width": self.tile_width,
            "columns": self.columns,
            "rows": self.rows,
            "gap": self.gap,
        }
        return aliases[key]


@dataclass(frozen=True)
class ViewPointMapping:
    view_x: int
    view_y: int
    canvas_x: int
    canvas_y: int
    local_x: int
    local_y: int
    array_index: tuple[int, ...]
    tile_number: int | None = None
    montage_axis: int | None = None
    montage_index: int | None = None


@dataclass(frozen=True)
class DisplayPointContext:
    mapping: ViewPointMapping
    value_prefix: str
    context_text: str


@dataclass(frozen=True)
class MontagePointStatus:
    kind: Literal["loaded", "loading", "skipped", "unloaded", "gap", "outside"]
    tile_number: int | None = None
    montage_index: int | None = None
    source_index: int | None = None
    local_x: int | None = None
    local_y: int | None = None
    message: str = ""


@dataclass(frozen=True)
class DisplayGeometry:
    view_state: ViewState
    display_shape: tuple[int, int]
    montage: MontageGeometry | None = None
    montage_origin_x: int = 0
    montage_origin_y: int = 0
    montage_tile_states: tuple[object, ...] = ()

    def __post_init__(self):
        shape = tuple(int(size) for size in self.display_shape)
        if len(shape) != 2:
            raise ValueError("display_shape must be (height, width)")
        if shape[0] < 1 or shape[1] < 1:
            raise ValueError("display dimensions must be at least 1")
        object.__setattr__(self, "display_shape", shape)
        object.__setattr__(self, "montage_origin_x", int(self.montage_origin_x))
        object.__setattr__(self, "montage_origin_y", int(self.montage_origin_y))
        object.__setattr__(self, "montage_tile_states", tuple(self.montage_tile_states or ()))

    def view_point_to_canvas_point(self, x: float, y: float) -> tuple[int, int] | None:
        view_x = int(math.floor(float(x)))
        view_y = int(math.floor(float(y)))
        canvas_x = view_x - (self.montage_origin_x if self.montage is not None else 0)
        canvas_y = view_y - (self.montage_origin_y if self.montage is not None else 0)
        if canvas_x < 0 or canvas_y < 0 or canvas_x >= self.display_shape[1] or canvas_y >= self.display_shape[0]:
            return None
        return canvas_x, canvas_y

    def view_point_to_tile_point(self, x: float, y: float, *, require_loaded: bool = True) -> MontagePointStatus | None:
        if self.montage is None:
            return None
        view_x = int(math.floor(float(x)))
        view_y = int(math.floor(float(y)))
        mapped = self._map_montage_point(view_x, view_y)
        if mapped is None:
            if self._point_inside_montage_bounds(view_x, view_y):
                return MontagePointStatus("gap", message="empty montage gap")
            return MontagePointStatus("outside", message="outside montage")
        tile_number, montage_index, local_x, local_y = mapped
        kind = "unloaded"
        if not self.montage_tile_states:
            kind = "loaded"
        elif tile_number < len(self.montage_tile_states):
            kind = _state_value(self.montage_tile_states[tile_number])
        if kind not in {"loaded", "loading", "skipped", "unloaded"}:
            kind = "unloaded"
        if require_loaded and kind != "loaded":
            messages = {
                "loading": "tile loading...",
                "unloaded": "tile loading...",
                "skipped": "tile skipped by memory budget",
            }
            return MontagePointStatus(
                kind=kind,
                tile_number=tile_number,
                montage_index=tile_number,
                source_index=montage_index,
                local_x=local_x,
                local_y=local_y,
                message=messages.get(kind, ""),
            )
        return MontagePointStatus(
            kind=kind,
            tile_number=tile_number,
            montage_index=tile_number,
            source_index=montage_index,
            local_x=local_x,
            local_y=local_y,
            message="" if kind == "loaded" else "tile loading..." if kind in {"loading", "unloaded"} else "tile skipped by memory budget",
        )

    def view_point_to_array_index(self, x: float, y: float, *, require_loaded: bool = True) -> ViewPointMapping | None:
        if self.view_state.image_axes is None:
            return None
        view_x = int(math.floor(float(x)))
        view_y = int(math.floor(float(y)))
        canvas_x = view_x - (self.montage_origin_x if self.montage is not None else 0)
        canvas_y = view_y - (self.montage_origin_y if self.montage is not None else 0)
        local_x = view_x
        local_y = view_y
        tile_number = None
        montage_index = None
        view_state = self.view_state

        if self.montage is not None:
            status = self.view_point_to_tile_point(view_x, view_y, require_loaded=require_loaded)
            if status is None or status.kind in {"gap", "outside"} or (require_loaded and status.kind != "loaded"):
                return None
            tile_number = status.tile_number
            montage_index = status.source_index
            local_x = status.local_x
            local_y = status.local_y
            if view_state.montage_axis is None:
                return None
            view_state = view_state.with_slice(view_state.montage_axis, montage_index)

        axis_index = self._local_point_to_axis_indices(view_state, local_x, local_y)
        if axis_index is None:
            return None
        primary_value, secondary_value = axis_index
        primary_axis, secondary_axis = view_state.image_axes
        index = list(view_state.slice_indices)
        index[primary_axis] = primary_value
        index[secondary_axis] = secondary_value
        return ViewPointMapping(
            view_x=view_x,
            view_y=view_y,
            canvas_x=canvas_x,
            canvas_y=canvas_y,
            local_x=local_x,
            local_y=local_y,
            array_index=tuple(index),
            tile_number=tile_number,
            montage_axis=self.view_state.montage_axis if self.montage is not None else None,
            montage_index=montage_index,
        )

    def view_point_to_profile_states(self, x: float, y: float, profile_axes: Iterable[int], *, require_loaded: bool = False) -> tuple[ViewState, ...]:
        mapping = self.view_point_to_array_index(x, y, require_loaded=require_loaded)
        if mapping is None or self.view_state.image_axes is None:
            return ()
        view_state = self.view_state
        if mapping.montage_axis is not None and mapping.montage_index is not None:
            view_state = view_state.with_slice(mapping.montage_axis, mapping.montage_index).with_montage_axis(None)
        primary_axis, secondary_axis = view_state.image_axes
        states = []
        for axis in profile_axes:
            axis = int(axis)
            if axis < 0 or axis >= view_state.ndim:
                continue
            profile_state = view_state.with_line_axis(axis)
            if axis != primary_axis:
                profile_state = profile_state.with_slice(primary_axis, mapping.array_index[primary_axis]).with_axis_range(primary_axis, None)
            if axis != secondary_axis:
                profile_state = profile_state.with_slice(secondary_axis, mapping.array_index[secondary_axis]).with_axis_range(secondary_axis, None)
            states.append(profile_state)
        return tuple(states)

    def clamp_view_point(self, x: float, y: float) -> tuple[int, int] | None:
        if self.view_state.image_axes is None:
            return None
        point_x = int(math.floor(float(x)))
        point_y = int(math.floor(float(y)))
        if self.montage is not None:
            return self._clamp_to_montage_tile(point_x, point_y)
        primary_axis, secondary_axis = self.view_state.image_axes
        width = self._display_axis_size(self.view_state, secondary_axis)
        height = self._display_axis_size(self.view_state, primary_axis)
        return (max(0, min(point_x, width - 1)), max(0, min(point_y, height - 1)))

    def context_for_view_point(self, x: float, y: float, *, require_loaded: bool = True) -> DisplayPointContext | None:
        mapping = self.view_point_to_array_index(x, y, require_loaded=require_loaded)
        if mapping is None:
            return None
        context_axes = self.view_state.non_display_axes()
        parts = []
        for axis in context_axes:
            if axis == mapping.montage_axis and mapping.montage_index is not None:
                parts.append(f"d{axis}={mapping.montage_index}")
            else:
                parts.append(f"d{axis}={mapping.array_index[axis]}")
        return DisplayPointContext(
            mapping=mapping,
            value_prefix=f"({mapping.local_x}, {mapping.local_y})",
            context_text=" ".join(parts),
        )

    def _map_montage_point(self, x: int, y: int):
        geometry = self.montage
        if geometry is None:
            return None
        stride_x = geometry.tile_width + geometry.gap
        stride_y = geometry.tile_height + geometry.gap
        column = x // stride_x
        row = y // stride_y
        if column < 0 or row < 0 or column >= geometry.columns or row >= geometry.rows:
            return None
        local_x = x - column * stride_x
        local_y = y - row * stride_y
        if local_x < 0 or local_x >= geometry.tile_width or local_y < 0 or local_y >= geometry.tile_height:
            return None
        tile_number = row * geometry.columns + column
        if tile_number >= len(geometry.indices):
            return None
        return tile_number, geometry.indices[tile_number], local_x, local_y

    def _point_inside_montage_bounds(self, x: int, y: int) -> bool:
        geometry = self.montage
        if geometry is None:
            return False
        full_width = geometry.columns * geometry.tile_width + max(0, geometry.columns - 1) * geometry.gap
        full_height = geometry.rows * geometry.tile_height + max(0, geometry.rows - 1) * geometry.gap
        return 0 <= int(x) < full_width and 0 <= int(y) < full_height

    def _clamp_to_montage_tile(self, x: int, y: int) -> tuple[int, int] | None:
        geometry = self.montage
        if geometry is None or not geometry.indices:
            return None
        best = None
        best_distance = None
        for tile_number, _index in enumerate(geometry.indices):
            row = tile_number // geometry.columns
            column = tile_number % geometry.columns
            x0 = column * (geometry.tile_width + geometry.gap)
            y0 = row * (geometry.tile_height + geometry.gap)
            cx = max(x0, min(x, x0 + geometry.tile_width - 1))
            cy = max(y0, min(y, y0 + geometry.tile_height - 1))
            distance = (cx - x) ** 2 + (cy - y) ** 2
            if best_distance is None or distance < best_distance:
                best = (cx, cy)
                best_distance = distance
        return best

    def _local_point_to_axis_indices(self, view_state: ViewState, local_x: int, local_y: int) -> tuple[int, int] | None:
        primary_axis, secondary_axis = view_state.image_axes
        actual_y = self._display_index_to_axis_index(view_state, primary_axis, local_y)
        actual_x = self._display_index_to_axis_index(view_state, secondary_axis, local_x)
        if actual_y is None or actual_x is None:
            return None
        return actual_y, actual_x

    @staticmethod
    def _display_axis_size(view_state: ViewState, axis: int) -> int:
        indices = view_state.axis_range_indices[int(axis)]
        if indices is not None:
            return len(indices)
        return int(view_state.shape[int(axis)])

    @staticmethod
    def _display_index_to_axis_index(view_state: ViewState, axis: int, display_index: int) -> int | None:
        axis = int(axis)
        display_index = int(display_index)
        indices = view_state.axis_range_indices[axis]
        if indices is None:
            if display_index < 0 or display_index >= view_state.shape[axis]:
                return None
            return display_index
        if display_index < 0 or display_index >= len(indices):
            return None
        return int(indices[display_index])


def _state_value(state: object) -> str:
    value = getattr(state, "value", state)
    return str(value)
