"""Qt-free orchestration for operation-backed array documents."""

from __future__ import annotations

import numpy as np

from arrayscope.operations.cost import (
    operation_output_dtype,
)
from arrayscope.operations.evaluator import OperationEvaluator
from arrayscope.operations.pipeline import ArrayDocument, evaluate_shape
from arrayscope.operations.registry import create_operation
from arrayscope.operations.stack import (
    delete_step,
    move_step,
    reorder_steps,
    replace_step_operation,
    set_step_enabled,
)


class OperationCoordinator:
    def __init__(
        self, base_data, operations=(), *, steps=None, axes=None, chunk_transport_codec="raw"
    ):
        self.base_data = base_data
        self.document = ArrayDocument(
            base_data, operations=tuple(operations), steps=steps, axes=axes
        )
        self.evaluator = OperationEvaluator(
            self.document, chunk_transport_codec=str(chunk_transport_codec)
        )
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

    def append_created_operation(self, operation):
        """Append an operation object that the caller already built.

        Splitting this out of :meth:`append_operation` lets a caller that has to
        *inspect* the operation first -- the UI predicts its output rank before
        deciding which window the result belongs in -- build it exactly once. A
        second ``create_operation`` would re-run every bound input slot's
        resolver.
        """

        return self.set_document(self.document.with_operation(operation))

    def undo(self):
        return self.set_document(self.document.without_last_operation())

    def clear(self):
        return self.set_document(self._document(steps=()))

    def delete(self, index):
        steps = delete_step(self.document.steps, index, self.base_data.shape)
        return self.set_document(self._document(steps=steps))

    def move(self, index, direction):
        steps = move_step(self.document.steps, index, direction, self.base_data.shape)
        return self.set_document(self._document(steps=steps))

    def reorder(self, order):
        steps = reorder_steps(self.document.steps, order, self.base_data.shape)
        return self.set_document(self._document(steps=steps))

    def set_enabled(self, index, enabled):
        steps = set_step_enabled(self.document.steps, index, enabled, self.base_data.shape)
        return self.set_document(self._document(steps=steps))

    def replace_operation(
        self,
        index,
        operation_id,
        axis=None,
        parameters=None,
        *,
        axes=None,
        slot_bindings=None,
        slot_resolver=None,
    ):
        operation = create_operation(
            operation_id,
            axis=axis,
            parameters=parameters or {},
            axes=axes,
            slot_bindings=slot_bindings,
            slot_resolver=slot_resolver,
        )
        steps = replace_step_operation(self.document.steps, index, operation, self.base_data.shape)
        return self.set_document(self._document(steps=steps))

    def load_operations(self, operations):
        return self.set_document(
            ArrayDocument(
                self.base_data,
                operations=tuple(operations),
                revision=self.document.revision,
                axes=self.document.base_axes,
            )
        )

    def load_steps(self, steps):
        return self.set_document(self._document(steps=tuple(steps)))

    def replace_base_data(self, data):
        return self.replace_base_and_clear_steps(data)

    def reload_base_data(self, data, *, preserve_steps=True):
        return self.set_document(
            self.document.reload_base_data(data, preserve_steps=preserve_steps)
        )

    def replace_base_and_clear_steps(self, data):
        return self.set_document(self.document.replace_base_and_clear_steps(data))

    def mark_base_data_changed(self):
        return self.set_document(self.document.mark_base_data_changed())

    def materialize(self):
        self.base_data = np.array(self.document.materialize(), copy=True)
        return self.replace_base_and_clear_steps(self.base_data)

    def operation_shapes(self):
        shapes = []
        shape = tuple(self.base_data.shape)
        for step in self.document.steps:
            if step.enabled:
                shape = evaluate_shape(shape, (step.operation,))
            shapes.append(shape)
        return tuple(shapes)

    def operation_dtypes(self):
        return self.operation_dtype_estimates()

    def operation_dtype_estimates(self):
        dtypes = []
        dtype = getattr(self.base_data, "dtype", None)
        for step in self.document.steps:
            if step.enabled:
                dtype = _operation_output_dtype(dtype, step.operation)
            dtypes.append(dtype)
        return tuple(dtypes)

    def _reject_scalar(self, document):
        """Refuse a 0-D document, and only a 0-D one.

        The floor is *rank zero*, not rank one: a 1-D result is displayable --
        it is the line plot -- and the UI's own pre-guard agrees (a reduction is
        offered only while ``ndim > 1``, which is what stops a 1-D array being
        reduced to a scalar, not what stops a 2-D one becoming a line).
        """

        if len(document.current_shape) < 1:
            raise ValueError(
                "operation would reduce the array to a single value; the viewer "
                "needs at least one axis (a 1-D result renders as a line plot)"
            )

    def _document(self, *, steps):
        return ArrayDocument(
            self.base_data,
            steps=tuple(steps),
            revision=self.document.revision,
            axes=self.document.base_axes,
        )


def _operation_output_dtype(dtype, operation):
    return operation_output_dtype(dtype, operation)
