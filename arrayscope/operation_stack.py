"""Pure operation-stack editing helpers."""

from __future__ import annotations

from .operation_pipeline import evaluate_shape


def delete_operation(operations, index, base_shape):
    operations = tuple(operations)
    index = _validate_index(index, operations)
    return validate_operation_stack(operations[:index] + operations[index + 1 :], base_shape)


def move_operation(operations, index, direction, base_shape):
    operations = list(operations)
    index = _validate_index(index, operations)
    new_index = index + int(direction)
    if new_index < 0 or new_index >= len(operations):
        return tuple(operations)
    operations[index], operations[new_index] = operations[new_index], operations[index]
    return validate_operation_stack(tuple(operations), base_shape)


def reorder_operations(operations, order, base_shape):
    operations = tuple(operations)
    order = tuple(int(index) for index in order)
    if len(order) != len(operations) or set(order) != set(range(len(operations))):
        raise ValueError("operation reorder must contain each operation index exactly once")
    return validate_operation_stack(tuple(operations[index] for index in order), base_shape)


def validate_operation_stack(operations, base_shape):
    operations = tuple(operations)
    evaluate_shape(base_shape, operations)
    return operations


def _validate_index(index, operations):
    index = int(index)
    if index < 0 or index >= len(operations):
        raise IndexError("operation index is out of range")
    return index
