"""Lazy slab planning and evaluation for operation-backed views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.operations.planner import apply_operation_to_region, plan_region_request
from arrayscope.operations.regions import apply_region, index_spec_from_region, region_shape


@dataclass(frozen=True)
class SlabRequest:
    kind: Literal["image", "line", "scalar", "export_frame"]
    view_state: object
    keep_axes: tuple[int, ...]
    slice_indices: tuple[int, ...]
    ranged_axes: tuple[int, ...] = ()
    frame_axis: int | None = None
    frame_index: int | None = None


@dataclass(frozen=True)
class SlabPlan:
    base_index: tuple
    base_shape: tuple[int, ...]
    slab_ops: tuple
    output_shape: tuple[int, ...]
    estimated_nbytes: int | None = None
    requires_full_materialization: bool = False
    region_plan: object | None = None


def request_for_image(view_state) -> SlabRequest:
    keep_axes = tuple(view_state.image_axes or ())
    return SlabRequest("image", view_state, keep_axes, tuple(view_state.slice_indices), _ranged_keep_axes(view_state, keep_axes))


def request_for_line(view_state) -> SlabRequest:
    keep_axes = () if view_state.line_axis is None else (int(view_state.line_axis),)
    return SlabRequest("line", view_state, keep_axes, tuple(view_state.slice_indices), _ranged_keep_axes(view_state, keep_axes))


def request_for_scalar(view_state, index) -> SlabRequest:
    slices = list(view_state.slice_indices)
    for axis, value in enumerate(index):
        slices[axis] = int(value)
    return SlabRequest("scalar", view_state, (), tuple(slices))


def request_for_export_frame(view_state, frame_axis, frame_index) -> SlabRequest:
    slices = list(view_state.slice_indices)
    slices[int(frame_axis)] = int(frame_index)
    keep_axes = tuple(view_state.image_axes or ())
    return SlabRequest(
        "export_frame",
        view_state,
        keep_axes,
        tuple(slices),
        _ranged_keep_axes(view_state, keep_axes),
        frame_axis=int(frame_axis),
        frame_index=int(frame_index),
    )


def plan_slab(document: ArrayDocument, request: SlabRequest) -> SlabPlan:
    region_plan = plan_region_request(document, request)
    base_shape = region_shape(np.shape(document.base_data), region_plan.required_input_region)
    dtype = getattr(document.base_data, "dtype", None)
    estimated = None if dtype is None else int(np.prod(base_shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    output_shape = region_shape(document.current_shape, region_plan.final_region)
    return SlabPlan(
        base_index=index_spec_from_region(region_plan.required_input_region),
        base_shape=tuple(int(v) for v in base_shape),
        slab_ops=tuple(document.enabled_operations),
        output_shape=tuple(int(v) for v in output_shape),
        estimated_nbytes=estimated,
        requires_full_materialization=False,
        region_plan=region_plan,
    )


def evaluate_slab(document: ArrayDocument, request: SlabRequest):
    plan = plan_slab(document, request)
    region_plan = plan.region_plan
    data = apply_region(document.base_data, region_plan.required_input_region)
    for transition in region_plan.transitions:
        data = apply_operation_to_region(
            transition.operation,
            data,
            input_region=transition.required_input_region,
            output_region=transition.output_region,
        )
    return data


def _ranged_keep_axes(view_state, keep_axes):
    axis_ranges = getattr(view_state, "axis_range_indices", ())
    ranged = []
    for axis in keep_axes:
        if axis < len(axis_ranges) and axis_ranges[int(axis)] is not None:
            ranged.append(int(axis))
    return tuple(ranged)
