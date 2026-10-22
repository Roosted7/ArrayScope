"""Qt-free orchestration for operation-backed array documents."""

from __future__ import annotations

from dataclasses import dataclass
from mmap import mmap

import numpy as np

from arrayscope.core.memory_policy import input_nbytes_for
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

# History stores operation definitions, never copies of the primary pixels.
# Auxiliary snapshots retained solely by history have their own logical-byte
# ceiling; this is independent of the render/cache budgets and is not an RSS
# estimate. Unknown-size retained inputs are conservatively over budget.
HISTORY_EXTRA_INPUT_BUDGET_BYTES = 512 * 1024 * 1024
HISTORY_MAX_CHANGES = 64


@dataclass(frozen=True)
class _HistoryEntry:
    steps: tuple
    description: str


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
        self._undo: list[_HistoryEntry] = []
        self._redo: list[_HistoryEntry] = []
        self.history_reset_reason = ""

    @property
    def shape(self):
        return self.document.current_shape

    def set_document(self, document):
        """Adopt externally owned state, retiring history across state boundaries.

        Windows also mirror the coordinator's own committed document through
        this method. That redundant adoption must preserve operation history.
        ``revision`` is the source-content epoch, not an operation edit counter.
        """
        self._reject_scalar(document)
        if (
            document.base_data is not self.base_data
            or document.revision != self.document.revision
            or document.base_axes != self.document.base_axes
            or not _same_steps(document.steps, self.document.steps)
        ):
            self._undo.clear()
            self._redo.clear()
            self.history_reset_reason = (
                "Operation history reset because the source or document changed."
            )
        return self._adopt(document)

    def _adopt(self, document):
        self.base_data = document.base_data
        self.document = document
        self.evaluator.set_document(document)
        return self.document

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)

    @property
    def undo_description(self):
        return self._undo[-1].description if self._undo else ""

    @property
    def redo_description(self):
        return self._redo[-1].description if self._redo else ""

    def _edit(self, document, description):
        self._reject_scalar(document)
        if _same_steps(document.steps, self.document.steps):
            return self.document
        self.history_reset_reason = ""
        self._undo.append(_HistoryEntry(self.document.steps, description))
        del self._undo[:-HISTORY_MAX_CHANGES]
        self._redo.clear()
        self._adopt(document)
        self._prune_history_inputs()
        return self.document

    def inherit_history_from(self, other, *, description=None):
        """Copy history into a same-source rank successor without sharing lists.

        A description records a new edit performed only in a sibling window.
        Without one, the successor must represent the already committed stack.
        """
        if (
            self.base_data is not other.base_data
            or self.document.base_axes != other.document.base_axes
        ):
            raise ValueError("operation history can only follow the same source")
        same = _same_steps(self.document.steps, other.document.steps)
        if description is None and not same:
            raise ValueError("operation history requires the same operation stack")
        self._undo = list(other._undo)
        self._redo = list(other._redo)
        self.history_reset_reason = other.history_reset_reason
        self._adopt(
            ArrayDocument(
                self.base_data,
                steps=self.document.steps,
                revision=other.document.revision,
                axes=other.document.base_axes,
            )
        )
        if description is not None and not same:
            self.history_reset_reason = ""
            self._undo.append(_HistoryEntry(other.document.steps, str(description)))
            del self._undo[:-HISTORY_MAX_CHANGES]
            self._redo.clear()
        self._prune_history_inputs()
        return self.document

    def _prune_history_inputs(self):
        current = _input_owners(self.document.steps, primary=self.base_data)
        while self._undo or self._redo:
            retained = {}
            for entry in (*self._undo, *self._redo):
                retained.update(_input_owners(entry.steps))
            extra = [size for identity, size in retained.items() if identity not in current]
            if (
                all(size is not None for size in extra)
                and sum(extra) <= HISTORY_EXTRA_INPUT_BUDGET_BYTES
            ):
                break
            # Preserve the nearest reachable transitions. Redo[0] is the most
            # distant future; undo[0] is the oldest past. Dropping a prefix
            # cannot jump over an intermediate edit.
            if self._redo:
                self._redo.pop(0)
            else:
                self._undo.pop(0)
            self.history_reset_reason = (
                "Operation history was shortened to stay within the memory "
                "budget for retained extra inputs (512 MiB)."
            )

    def append_created_operation(self, operation):
        """Append an operation object that the caller already built.

        Splitting this out of :meth:`append_operation` lets a caller that has to
        *inspect* the operation first -- the UI predicts its output rank before
        deciding which window the result belongs in -- build it exactly once. A
        second ``create_operation`` would re-run every bound input slot's
        resolver.
        """

        return self._edit(self.document.with_operation(operation), "Add operation")

    def undo(self):
        if not self._undo:
            return self.document
        entry = self._undo[-1]
        document = self._document(steps=entry.steps)
        self._reject_scalar(document)
        self._undo.pop()
        self._redo.append(_HistoryEntry(self.document.steps, entry.description))
        self._adopt(document)
        self._prune_history_inputs()
        return self.document

    def redo(self):
        if not self._redo:
            return self.document
        entry = self._redo[-1]
        document = self._document(steps=entry.steps)
        self._reject_scalar(document)
        self._redo.pop()
        self._undo.append(_HistoryEntry(self.document.steps, entry.description))
        self._adopt(document)
        self._prune_history_inputs()
        return self.document

    def clear(self):
        return self._edit(self._document(steps=()), "Clear operations")

    def delete(self, index):
        steps = delete_step(self.document.steps, index, self.base_data.shape)
        return self._edit(self._document(steps=steps), "Delete operation")

    def move(self, index, direction):
        steps = move_step(self.document.steps, index, direction, self.base_data.shape)
        return self._edit(self._document(steps=steps), "Move operation")

    def reorder(self, order):
        steps = reorder_steps(self.document.steps, order, self.base_data.shape)
        return self._edit(self._document(steps=steps), "Reorder operations")

    def set_enabled(self, index, enabled):
        steps = set_step_enabled(self.document.steps, index, enabled, self.base_data.shape)
        return self._edit(
            self._document(steps=steps), "Enable operation" if enabled else "Disable operation"
        )

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
        return self._edit(self._document(steps=steps), "Edit operation parameters")

    def load_operations(self, operations):
        return self._edit(
            ArrayDocument(
                self.base_data,
                operations=tuple(operations),
                revision=self.document.revision,
                axes=self.document.base_axes,
            ),
            "Replace operation stack",
        )

    def load_steps(self, steps):
        return self._edit(self._document(steps=tuple(steps)), "Replace operation stack")

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
        data = np.array(self.document.materialize(), copy=True)
        return self.replace_base_and_clear_steps(data)

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


def _same_steps(first, second):
    """Compare editable meaning while preserving existing runtime step IDs."""
    if len(first) != len(second):
        return False
    for left, right in zip(first, second, strict=True):
        if left is right:
            continue
        if left.enabled != right.enabled or left.unavailable_reason != right.unavailable_reason:
            return False
        try:
            if not bool(left.operation == right.operation):
                return False
        except (TypeError, ValueError):
            return False
        left_slots = getattr(left.operation, "resolved_slots", ())
        right_slots = getattr(right.operation, "resolved_slots", ())
        if tuple((name, slot.signature) for name, slot in left_slots) != tuple(
            (name, slot.signature) for name, slot in right_slots
        ):
            return False
    return True


def _input_owners(steps, *, primary=None):
    """Metadata-only logical bytes of source owners referenced by input slots.

    Follow ndarray bases so views count once and include nested document inputs.
    File paths and ROI geometry do not retain eager array payloads. Never read a
    lazy source or invoke ``asarray`` to discover a size.
    """
    owners = {}
    visited = set()

    def source_owner(source):
        if source is None:
            return
        if id(source) in visited:
            return
        visited.add(id(source))
        if isinstance(source, ArrayDocument):
            source_owner(source.base_data)
            step_owners(source.steps)
            return
        while True:
            if isinstance(source, np.ndarray) and source.base is not None:
                source = source.base
            elif isinstance(source, memoryview):
                source = source.obj
            else:
                break
        try:
            size = input_nbytes_for(source)
            if size is None and isinstance(source, (bytes, bytearray, mmap)):
                size = len(source)
        except Exception:
            # A source may already be closed or expose failing metadata. An
            # edit must not fail after commit merely while pricing its past.
            size = None
        owners[id(source)] = size

    def step_owners(source_steps):
        for step in source_steps:
            for _, slot in getattr(step.operation, "resolved_slots", ()):
                if slot.loader in {"array", "document", "dimension-set"}:
                    source_owner(slot.source)

    if primary is not None:
        source_owner(primary)
    step_owners(steps)
    return owners
