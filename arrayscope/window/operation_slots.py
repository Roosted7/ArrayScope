"""Window-owned resolution and invalidation for operation input slots."""

from __future__ import annotations

import os
from dataclasses import replace
from uuid import uuid4

import numpy as np
from pyqtgraph.Qt import QtCore

from arrayscope.core.roi import roi_coordinates
from arrayscope.operations.input_slots import (
    SLOT_DIMENSION_SET,
    SLOT_OPEN_DOCUMENT,
    SLOT_ROI_COORDINATES,
    SLOT_ROI_MASK,
    SLOT_SAVED_ARRAY,
    ResolvedSlot,
    SlotBinding,
    SlotSourceOption,
    inspect_saved_array,
)
from arrayscope.operations.plugins import PluginOperation
from arrayscope.operations.registry import create_operation


class OperationSlotSourcesMixin:
    """Resolve slot recipe data against window-local source catalogs.

    Compare-group membership is the existing owner for open sibling documents.
    ROI identity and geometry come from ``roi_store``.  A resolved operation
    carries only an immutable document/geometry/file snapshot into workers.
    """

    def _init_operation_slot_sources(self) -> None:
        self._operation_source_id = uuid4().hex
        self._slot_file_watcher = QtCore.QFileSystemWatcher(self)
        self._slot_file_watcher.fileChanged.connect(self._on_slot_file_changed)
        self._refreshing_operation_slots = False

    def _slot_source_options(self, entry) -> dict[str, tuple[SlotSourceOption, ...]]:
        options: dict[str, tuple[SlotSourceOption, ...]] = {}
        for slot in entry.input_slots:
            choices: list[SlotSourceOption] = []
            if SLOT_DIMENSION_SET in slot.accepts:
                binding = self._current_dimension_set_binding()
                choices.append(
                    SlotSourceOption(
                        binding,
                        binding.label,
                        "The current fixed indices of this array; displayed axes remain whole.",
                    )
                )
            if SLOT_OPEN_DOCUMENT in slot.accepts:
                choices.extend(self._open_document_slot_options())
            for selection in tuple(getattr(self.roi_store, "selections", ()) or ()):
                for kind, representation in (
                    (SLOT_ROI_MASK, "mask"),
                    (SLOT_ROI_COORDINATES, "coordinates"),
                ):
                    if kind not in slot.accepts:
                        continue
                    binding = SlotBinding(
                        kind,
                        source_id=str(selection.id),
                        label=f"{selection.label} ({representation})",
                    )
                    choices.append(
                        SlotSourceOption(
                            binding,
                            binding.label,
                            f"One ROI from this document as {representation}.",
                        )
                    )
            options[slot.name] = tuple(choices)
        return options

    def _current_dimension_set_binding(self) -> SlotBinding:
        shape = tuple(int(size) for size in np.shape(self.base_data))
        image_axes = set(getattr(self.view_state, "image_axes", ()) or ())
        slice_indices = tuple(getattr(self.view_state, "slice_indices", ()) or ())
        indices = tuple(
            None
            if axis in image_axes
            else min(
                max(0, int(slice_indices[axis]) if axis < len(slice_indices) else 0),
                max(0, size - 1),
            )
            for axis, size in enumerate(shape)
        )
        fixed = ", ".join(
            f"d{axis}={value}" for axis, value in enumerate(indices) if value is not None
        )
        return SlotBinding(
            SLOT_DIMENSION_SET,
            source_id=self._operation_source_id,
            indices=indices,
            label=f"Current dimension set{f' ({fixed})' if fixed else ''}",
        )

    def _open_document_slot_options(self) -> tuple[SlotSourceOption, ...]:
        group = getattr(self, "_compare_group", None)
        members = () if group is None else group.members()
        options = []
        for member in members:
            if member is self:
                continue
            source_id = str(getattr(member, "_operation_source_id", "") or "")
            if not source_id:
                continue
            label = (
                getattr(member, "compare_label", "")
                or member.windowTitle()
                or f"Document {len(options) + 2}"
            )
            binding = SlotBinding(
                SLOT_OPEN_DOCUMENT,
                source_id=source_id,
                label=f"Open document {label}",
            )
            options.append(
                SlotSourceOption(
                    binding,
                    binding.label,
                    "The sibling document's current derived array.",
                )
            )
        return tuple(options)

    def _resolve_operation_slot(self, slot, binding: SlotBinding) -> ResolvedSlot:
        binding = SlotBinding.from_payload(binding)
        if binding.kind == SLOT_DIMENSION_SET:
            if binding.source_id != self._operation_source_id:
                raise ValueError("the referenced array is not this document")
            shape = tuple(int(size) for size in np.shape(self.base_data))
            if len(binding.indices) != len(shape):
                raise ValueError("the saved dimension set has the wrong rank")
            output_shape = tuple(
                size for size, index in zip(shape, binding.indices, strict=True) if index is None
            )
            return ResolvedSlot(
                binding=binding,
                shape=output_shape,
                dtype=np.dtype(
                    getattr(self.base_data, "dtype", np.asarray(self.base_data).dtype)
                ).str,
                source_identity=(
                    self._operation_source_id,
                    id(self.base_data),
                    int(self.document.revision),
                    binding.indices,
                ),
                loader="dimension-set",
                source=self.base_data,
            )
        if binding.kind == SLOT_OPEN_DOCUMENT:
            member = self._find_open_document(binding.source_id)
            if member is None:
                raise ValueError("the referenced open document is closed")
            document = member.document
            return ResolvedSlot(
                binding=binding,
                shape=tuple(int(size) for size in document.current_shape),
                dtype=np.dtype(member.data.dtype).str,
                source_identity=(
                    binding.source_id,
                    id(document.base_data),
                    int(document.revision),
                    tuple(document.steps),
                ),
                loader="document",
                source=document,
            )
        if binding.kind in {SLOT_ROI_MASK, SLOT_ROI_COORDINATES}:
            selection = self.roi_store.get(binding.source_id)
            if selection is None:
                raise ValueError(f"ROI {binding.source_id!r} no longer exists")
            if binding.kind == SLOT_ROI_COORDINATES:
                coordinates = roi_coordinates(selection.geometry)
                shape = tuple(int(size) for size in coordinates.shape)
                dtype = coordinates.dtype.str
                loader = "roi-coordinates"
            else:
                shape = self._roi_slot_shape()
                dtype = np.dtype(bool).str
                loader = "roi-mask"
            return ResolvedSlot(
                binding=binding,
                shape=shape,
                dtype=dtype,
                source_identity=(
                    self._operation_source_id,
                    binding.source_id,
                    binding.kind,
                    selection.geometry,
                    shape,
                ),
                loader=loader,
                source=selection.geometry,
            )
        if binding.kind == SLOT_SAVED_ARRAY:
            path = os.path.abspath(os.path.expanduser(binding.path))
            shape, dtype = inspect_saved_array(path)
            stat = os.stat(path)
            return ResolvedSlot(
                binding=replace(binding, path=path),
                shape=shape,
                dtype=dtype.str,
                source_identity=(path, int(stat.st_mtime_ns), int(stat.st_size)),
                loader="saved-array",
                source=path,
            )
        raise ValueError(f"unsupported input binding kind: {binding.kind!r}")

    def _find_open_document(self, source_id: str):
        group = getattr(self, "_compare_group", None)
        for member in () if group is None else group.members():
            if str(getattr(member, "_operation_source_id", "")) == str(source_id):
                return member
        return None

    def _roi_slot_shape(self) -> tuple[int, int]:
        image = getattr(getattr(self, "img_view", None), "image", None)
        if image is not None and np.ndim(image) >= 2:
            return tuple(int(size) for size in np.shape(image)[:2])
        image_axes = tuple(getattr(self.view_state, "image_axes", ()) or ())
        shape = tuple(int(size) for size in self.data.shape)
        if len(image_axes) >= 2:
            return (shape[image_axes[1]], shape[image_axes[0]])
        return tuple(shape[:2])  # type: ignore[return-value]

    def _refresh_operation_slot_bindings(
        self,
        *,
        roi_id: str | None = None,
        document_id: str | None = None,
        file_path: str | None = None,
        render: bool = True,
    ) -> bool:
        if self._refreshing_operation_slots or not hasattr(self, "document"):
            return False
        self._refreshing_operation_slots = True
        try:
            changed = False
            steps = []
            for step in self.document.steps:
                operation = step.operation
                if not isinstance(operation, PluginOperation) or not operation.slot_bindings:
                    steps.append(step)
                    continue
                bindings = dict(operation.slot_bindings)
                if roi_id is not None and not any(
                    binding.kind in {SLOT_ROI_MASK, SLOT_ROI_COORDINATES}
                    and binding.source_id == str(roi_id)
                    for binding in bindings.values()
                ):
                    steps.append(step)
                    continue
                if document_id is not None and not any(
                    binding.kind == SLOT_OPEN_DOCUMENT and binding.source_id == str(document_id)
                    for binding in bindings.values()
                ):
                    steps.append(step)
                    continue
                if file_path is not None and not any(
                    binding.kind == SLOT_SAVED_ARRAY
                    and os.path.abspath(binding.path) == os.path.abspath(file_path)
                    for binding in bindings.values()
                ):
                    steps.append(step)
                    continue
                rebuilt = create_operation(
                    operation.plugin_id,
                    axis=operation.axis,
                    parameters=dict(operation.params),
                    slot_bindings=bindings,
                    slot_resolver=self._resolve_operation_slot,
                )
                reason = rebuilt.current_unavailable_reason()
                next_enabled = bool(step.enabled and not reason)
                if step.unavailable_reason and not reason:
                    next_enabled = True
                next_step = replace(
                    step,
                    operation=rebuilt,
                    enabled=next_enabled,
                    unavailable_reason=reason,
                )
                changed = changed or next_step != step
                steps.append(next_step)
            if not changed:
                return False
            document = self.operation_coordinator._document(steps=tuple(steps))
            self._set_document(document)
            self._watch_operation_slot_files()
            if render:
                self.render(reason="operation-input", force_autolevel=True)
            return True
        finally:
            self._refreshing_operation_slots = False

    def _watch_operation_slot_files(self) -> None:
        watcher = getattr(self, "_slot_file_watcher", None)
        if watcher is None:
            return
        existing = tuple(watcher.files())
        if existing:
            watcher.removePaths(existing)
        paths = {
            os.path.abspath(binding.path)
            for step in self.document.steps
            if isinstance(step.operation, PluginOperation)
            for _name, binding in step.operation.slot_bindings
            if binding.kind == SLOT_SAVED_ARRAY and os.path.isfile(binding.path)
        }
        if paths:
            watcher.addPaths(sorted(paths))

    def _on_slot_file_changed(self, path: str) -> None:
        self._refresh_operation_slot_bindings(file_path=path)

    def _notify_operation_source_changed(self) -> None:
        if self._refreshing_operation_slots:
            return
        group = getattr(self, "_compare_group", None)
        if group is None:
            return
        source_id = self._operation_source_id
        for member in group.members():
            if member is not self:
                member._refresh_operation_slot_bindings(document_id=source_id)
