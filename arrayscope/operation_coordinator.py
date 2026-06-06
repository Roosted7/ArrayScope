"""Qt-free orchestration for operation-backed array documents."""

from __future__ import annotations

import numpy as np

from .operation_evaluator import OperationEvaluator
from .operation_pipeline import ArrayDocument, evaluate_shape
from .operation_registry import create_operation
from .operation_stack import delete_operation, move_operation, reorder_operations


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
        operations = delete_operation(self.document.operations, index, self.base_data.shape)
        return self.set_document(ArrayDocument(self.base_data, operations=operations))

    def move(self, index, direction):
        operations = move_operation(self.document.operations, index, direction, self.base_data.shape)
        return self.set_document(ArrayDocument(self.base_data, operations=operations))

    def reorder(self, order):
        operations = reorder_operations(self.document.operations, order, self.base_data.shape)
        return self.set_document(ArrayDocument(self.base_data, operations=operations))

    def load_operations(self, operations):
        return self.set_document(ArrayDocument(self.base_data, operations=tuple(operations)))

    def replace_base_data(self, data):
        self.base_data = data
        return self.set_document(ArrayDocument(self.base_data))

    def materialize(self):
        self.base_data = np.array(self.evaluator.current_data(), copy=True)
        return self.set_document(ArrayDocument(self.base_data))

    def operation_shapes(self):
        shapes = []
        shape = tuple(self.base_data.shape)
        for operation in self.document.operations:
            shape = evaluate_shape(shape, (operation,))
            shapes.append(shape)
        return tuple(shapes)

    def _reject_scalar(self, document):
        if len(document.current_shape) < 1:
            raise ValueError("operation would produce a scalar, which this viewer cannot display yet")

