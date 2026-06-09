"""Internal runtime optimizer for operation stacks.

The optimizer never rewrites user-authored operation steps or recipes. It only
builds a smaller equivalent operation tuple for runtime evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arrayscope.operations.capabilities import OperationCapabilities, OperationKind, default_chunkable_axes
from arrayscope.operations.regions import RegionSpec


@dataclass(frozen=True)
class OptimizationStep:
    kind: str
    message: str
    original_start: int
    original_stop: int
    replacement_count: int


@dataclass(frozen=True)
class OptimizedOperationPlan:
    original_operations: tuple
    operations: tuple
    output_shape: tuple[int, ...]
    output_dtype: object | None
    steps: tuple[OptimizationStep, ...]


@dataclass(frozen=True)
class CastDType:
    dtype: str

    def apply(self, data):
        return np.asarray(data).astype(np.dtype(self.dtype), copy=False)

    def output_shape(self, shape):
        return tuple(int(size) for size in shape)

    def output_dtype(self, input_dtype):
        del input_dtype
        return np.dtype(self.dtype)

    def capabilities(self, input_shape, input_dtype=None) -> OperationCapabilities:
        del input_dtype
        ndim = len(tuple(input_shape))
        return OperationCapabilities(
            kind=OperationKind.ELEMENTWISE,
            chunkable_axes=default_chunkable_axes(OperationKind.ELEMENTWISE, ndim=ndim),
            cache_stage=False,
            can_fuse=True,
        )

    def required_input_region(self, input_shape, output_region: RegionSpec) -> RegionSpec:
        del input_shape
        return output_region

    def apply_to_region(self, data, *, input_region: RegionSpec, output_region: RegionSpec):
        del input_region, output_region
        return self.apply(data)


def optimize_operations(base_shape, base_dtype, operations) -> OptimizedOperationPlan:
    from arrayscope.operations.pipeline import evaluate_shape

    original_operations = tuple(operations)
    original_shape, original_dtype = _evaluate_shape_dtype(base_shape, base_dtype, original_operations)
    optimized = original_operations
    steps = []

    while True:
        result = _simplify_once(base_shape, base_dtype, optimized)
        if result is None:
            break
        optimized, step = result
        steps.append(step)

    output_shape, output_dtype = _evaluate_shape_dtype(base_shape, base_dtype, optimized)
    if tuple(output_shape) != tuple(original_shape):
        raise ValueError(f"optimized operation shape {output_shape} does not match original shape {original_shape}")
    if _dtype_key(output_dtype) != _dtype_key(original_dtype):
        raise ValueError(f"optimized operation dtype {output_dtype} does not match original dtype {original_dtype}")
    if tuple(evaluate_shape(base_shape, optimized)) != tuple(original_shape):
        raise ValueError("optimized operation shape contract does not match original operations")

    return OptimizedOperationPlan(
        original_operations=original_operations,
        operations=tuple(optimized),
        output_shape=tuple(int(size) for size in original_shape),
        output_dtype=original_dtype,
        steps=tuple(steps),
    )


def format_optimization_steps(steps: tuple[OptimizationStep, ...]) -> tuple[str, ...]:
    return tuple(step.message for step in tuple(steps))


def _simplify_once(base_shape, base_dtype, operations):
    from arrayscope.operations.pipeline import CenteredFFT, CenteredIFFT, Conjugate, Crop, ReverseAxis

    operations = tuple(operations)
    shapes, dtypes = _prefix_shape_dtype(base_shape, base_dtype, operations)

    for index in range(len(operations) - 1):
        left = operations[index]
        right = operations[index + 1]
        pair = (left, right)
        if (
            _op_name(left) in {"CenteredFFT", "CenteredIFFT"}
            and _op_name(right) in {"CenteredFFT", "CenteredIFFT"}
            and _op_name(left) != _op_name(right)
            and int(left.axis) == int(right.axis)
        ):
            input_dtype = dtypes[index]
            output_dtype = dtypes[index + 2]
            replacement = _dtype_preserving_replacement(input_dtype, output_dtype)
            axis = int(left.axis)
            suffix = "" if not replacement else f", inserted dtype cast {replacement[0].dtype}"
            return _replace(
                operations,
                index,
                index + 2,
                replacement,
                OptimizationStep(
                    "fft_ifft_cancel",
                    f"removed {_op_name(left)}/{_op_name(right)} axis {axis}{suffix}",
                    index,
                    index + 2,
                    len(replacement),
                ),
            )
        if _op_name(left) == "ReverseAxis" and _op_name(right) == "ReverseAxis" and int(left.axis) == int(right.axis):
            return _replace(
                operations,
                index,
                index + 2,
                (),
                OptimizationStep("reverse_pair", f"removed ReverseAxis pair axis {int(left.axis)}", index, index + 2, 0),
            )
        if _op_name(left) == "Conjugate" and _op_name(right) == "Conjugate":
            return _replace(
                operations,
                index,
                index + 2,
                (),
                OptimizationStep("conjugate_pair", "removed Conjugate pair", index, index + 2, 0),
            )
        if _op_name(left) == "Crop" and _op_name(right) == "Crop" and int(left.axis) == int(right.axis):
            composed = Crop(axis=int(left.axis), start=int(left.start) + int(right.start), stop=int(left.start) + int(right.stop))
            pair_shape, pair_dtype = _evaluate_shape_dtype(shapes[index], dtypes[index], pair)
            new_shape, new_dtype = _evaluate_shape_dtype(shapes[index], dtypes[index], (composed,))
            if tuple(pair_shape) == tuple(new_shape) and _dtype_key(pair_dtype) == _dtype_key(new_dtype):
                return _replace(
                    operations,
                    index,
                    index + 2,
                    (composed,),
                    OptimizationStep(
                        "crop_compose",
                        f"composed Crop axis {int(left.axis)} [{int(left.start)}, {int(left.stop)}) + [{int(right.start)}, {int(right.stop)})",
                        index,
                        index + 2,
                        1,
                    ),
                )
        if _op_name(left) == "CastDType" and _op_name(right) == "CastDType":
            return _replace(
                operations,
                index,
                index + 2,
                (right,),
                OptimizationStep(
                    "cast_coalesce",
                    f"coalesced dtype casts {left.dtype} -> {right.dtype}",
                    index,
                    index + 2,
                    1,
                ),
            )
    return None


def _op_name(operation) -> str:
    return type(operation).__name__


def _replace(operations, start, stop, replacement, step):
    return tuple(operations[:start]) + tuple(replacement) + tuple(operations[stop:]), step


def _dtype_preserving_replacement(input_dtype, output_dtype) -> tuple:
    if _dtype_key(input_dtype) == _dtype_key(output_dtype) or output_dtype is None:
        return ()
    return (CastDType(str(np.dtype(output_dtype))),)


def _prefix_shape_dtype(base_shape, base_dtype, operations):
    shapes = [tuple(int(size) for size in base_shape)]
    dtypes = [None if base_dtype is None else np.dtype(base_dtype)]
    shape = shapes[0]
    dtype = dtypes[0]
    for operation in tuple(operations):
        shape = tuple(int(size) for size in operation.output_shape(shape))
        if dtype is not None and hasattr(operation, "output_dtype"):
            dtype = operation.output_dtype(dtype)
            dtype = None if dtype is None else np.dtype(dtype)
        shapes.append(shape)
        dtypes.append(dtype)
    return tuple(shapes), tuple(dtypes)


def _evaluate_shape_dtype(base_shape, base_dtype, operations):
    shapes, dtypes = _prefix_shape_dtype(base_shape, base_dtype, operations)
    return shapes[-1], dtypes[-1]


def _dtype_key(dtype):
    return None if dtype is None else str(np.dtype(dtype))
