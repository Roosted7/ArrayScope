"""Qt-free orchestration for operation-backed array documents."""

from __future__ import annotations

import numpy as np

from arrayscope.operations.evaluator import OperationEvaluator
from arrayscope.operations.pipeline import ArrayDocument, evaluate_shape
from arrayscope.operations.registry import create_operation
from arrayscope.operations.stack import delete_step, move_step, reorder_steps, replace_step_operation, set_step_enabled


class OperationCoordinator:
    def __init__(self, base_data, operations=()):
        self.base_data = base_data
        self.document = ArrayDocument(base_data, operations=tuple(operations))
        self.evaluator = OperationEvaluator(self.document)
        self._reject_scalar(self.document)

    @property
    def shape(self):
        return self.document.current_shape

    def set_document(self, document):
        self._reject_scalar(document)
        self.base_data = document.base_data
        self.document = document
        self.evaluator.set_document(document)
        return self.document

    def append_operation(self, operation_id, axis=None, parameters=None):
        operation = create_operation(operation_id, axis=axis, parameters=parameters or {})
        return self.set_document(self.document.with_operation(operation))

    def undo(self):
        return self.set_document(self.document.without_last_operation())

    def clear(self):
        return self.set_document(ArrayDocument(self.base_data))

    def delete(self, index):
        steps = delete_step(self.document.steps, index, self.base_data.shape)
        return self.set_document(ArrayDocument(self.base_data, steps=steps))

    def move(self, index, direction):
        steps = move_step(self.document.steps, index, direction, self.base_data.shape)
        return self.set_document(ArrayDocument(self.base_data, steps=steps))

    def reorder(self, order):
        steps = reorder_steps(self.document.steps, order, self.base_data.shape)
        return self.set_document(ArrayDocument(self.base_data, steps=steps))

    def set_enabled(self, index, enabled):
        steps = set_step_enabled(self.document.steps, index, enabled, self.base_data.shape)
        return self.set_document(ArrayDocument(self.base_data, steps=steps))

    def replace_operation(self, index, operation_id, axis=None, parameters=None):
        operation = create_operation(operation_id, axis=axis, parameters=parameters or {})
        steps = replace_step_operation(self.document.steps, index, operation, self.base_data.shape)
        return self.set_document(ArrayDocument(self.base_data, steps=steps))

    def load_operations(self, operations):
        return self.set_document(ArrayDocument(self.base_data, operations=tuple(operations)))

    def load_steps(self, steps):
        return self.set_document(ArrayDocument(self.base_data, steps=tuple(steps)))

    def replace_base_data(self, data):
        self.base_data = data
        return self.set_document(ArrayDocument(self.base_data))

    def materialize(self):
        self.base_data = np.array(self.evaluator.current_data(), copy=True)
        return self.set_document(ArrayDocument(self.base_data))

    def operation_shapes(self):
        shapes = []
        shape = tuple(self.base_data.shape)
        for step in self.document.steps:
            if step.enabled:
                shape = evaluate_shape(shape, (step.operation,))
            shapes.append(shape)
        return tuple(shapes)

    def operation_dtypes(self):
        dtypes = []
        data = self.base_data
        for step in self.document.steps:
            if step.enabled:
                data = step.operation.apply(data)
            dtypes.append(getattr(data, "dtype", None))
        return tuple(dtypes)

    def _reject_scalar(self, document):
        if len(document.current_shape) < 1:
            raise ValueError("operation would produce a scalar, which this viewer cannot display yet")
