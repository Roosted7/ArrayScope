"""Pure operation-stack editing helpers."""

from __future__ import annotations

from dataclasses import replace

from arrayscope.operations.pipeline import OperationStep, evaluate_shape


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


def delete_step(steps, index, base_shape):
    steps = tuple(steps)
    index = _validate_index(index, steps)
    return validate_operation_steps(steps[:index] + steps[index + 1 :], base_shape)


def move_step(steps, index, direction, base_shape):
    steps = list(steps)
    index = _validate_index(index, steps)
    new_index = index + int(direction)
    if new_index < 0 or new_index >= len(steps):
        return tuple(steps)
    steps[index], steps[new_index] = steps[new_index], steps[index]
    return validate_operation_steps(tuple(steps), base_shape)


def reorder_steps(steps, order, base_shape):
    steps = tuple(steps)
    order = tuple(int(index) for index in order)
    if len(order) != len(steps) or set(order) != set(range(len(steps))):
        raise ValueError("operation reorder must contain each operation index exactly once")
    return validate_operation_steps(tuple(steps[index] for index in order), base_shape)


def set_step_enabled(steps, index, enabled, base_shape):
    steps = list(steps)
    index = _validate_index(index, steps)
    steps[index] = replace(steps[index], enabled=bool(enabled))
    return validate_operation_steps(tuple(steps), base_shape)


def replace_step_operation(steps, index, operation, base_shape):
    steps = list(steps)
    index = _validate_index(index, steps)
    step = steps[index]
    if not isinstance(step, OperationStep):
        step = OperationStep(step)
    steps[index] = replace(step, operation=operation)
    return validate_operation_steps(tuple(steps), base_shape)


def validate_operation_steps(steps, base_shape):
    steps = tuple(OperationStep(step) if not isinstance(step, OperationStep) else step for step in steps)
    evaluate_shape(base_shape, tuple(step.operation for step in steps if step.enabled))
    return steps


def _validate_index(index, operations):
    index = int(index)
    if index < 0 or index >= len(operations):
        raise IndexError("operation index is out of range")
    return index
