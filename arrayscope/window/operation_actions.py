from __future__ import annotations

import contextlib

import numpy as np
import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui, QtWidgets

from arrayscope.core.memory_budget import DEFAULT_VISIBLE_RENDER_BUDGET_BYTES, format_bytes
from arrayscope.core.view_recipe import DisplaySettings, ViewRecipe
from arrayscope.core.view_recipe import load_view_recipe as load_view_recipe_file
from arrayscope.core.view_recipe import save_view_recipe as save_view_recipe_file
from arrayscope.display.colormap_policy import default_colormap_name
from arrayscope.io.numpy_save import save_derived_array
from arrayscope.operations import fft_backend
from arrayscope.operations.cost import estimate_pipeline_cost
from arrayscope.operations.evaluator import LARGE_MATERIALIZE_BYTES
from arrayscope.operations.parameter_forms import build_parameter_form
from arrayscope.operations.recipes import dumps_recipe, load_recipe_steps, save_recipe
from arrayscope.operations.registry import (
    get_operation_entry,
    operation_id_for,
    operation_parameter_value,
)
from arrayscope.ui.command_palette import CommandPaletteDialog, PaletteCommand
from arrayscope.ui.file_dialogs import get_open_file_name, get_save_file_name
from arrayscope.ui.icons import material_icon, set_action_icon
from arrayscope.ui.operation_add_popup import OperationAddPopup
from arrayscope.ui.operation_listing import build_operation_listing
from arrayscope.ui.operation_params_popup import OperationParamsPopup
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.domain import Domain


class OperationActionsMixin:
    def dimClicked(self, event, label, dim):
        if dim >= self.data.ndim or self.singleton[dim]:
            return
        if event.button() == Qt.QtCore.Qt.MouseButton.RightButton:
            return

        p = QtGui.QPalette()

        # If already transformed, any click returns to native
        if self.domain[dim] == Domain.FOURIER:
            # From FFT domain, go back to native (undo)
            self.domain[dim] = Domain.NATIVE
            p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("black"))
            label.setStyleSheet("font-weight: normal;")
            self._apply_ifft(dim)  # Undo the FFT by applying IFFT
        elif self.domain[dim] == Domain.INV_FOURIER:
            # From IFFT domain, go back to native (undo)
            self.domain[dim] = Domain.NATIVE
            p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("black"))
            label.setStyleSheet("font-weight: normal;")
            self._apply_fft(dim)  # Undo the IFFT by applying FFT
        elif event.button() == Qt.QtCore.Qt.MouseButton.RightButton:
            # Right click from native: apply IFFT
            self.domain[dim] = Domain.INV_FOURIER
            p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("green"))
            label.setStyleSheet("font-weight: bold; color: green;")
            self._apply_ifft(dim)
        else:
            # Left click from native: apply FFT
            self.domain[dim] = Domain.FOURIER
            p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("blue"))
            label.setStyleSheet("font-weight: bold; color: blue;")
            self._apply_fft(dim)

        label.setPalette(p)
        self.update_image_view()
        self.update_line_plot()

    def _apply_fft(self, dim):
        """Apply forward FFT along specified dimension."""
        self._append_operation("centered_fft", dim)

    def _apply_ifft(self, dim):
        """Apply inverse FFT along specified dimension."""
        self._append_operation("centered_ifft", dim)

    def _build_operation_context_menu(self, dim, anchor) -> QtWidgets.QMenu:
        """Construct (but do not show) the sectioned chip "+" menu.

        Split out from :meth:`_show_operation_context_menu` so the same menu can
        be grabbed for screenshots (or otherwise driven) without the blocking
        ``exec``. ``anchor`` is the global point a parameterized op's stage-2
        popup opens at.
        """

        menu = QtWidgets.QMenu(self)
        menu.setToolTipsVisible(True)
        sections = build_operation_listing()
        main_sections = [section for section in sections if not section.is_more]
        more_sections = [section for section in sections if section.is_more]
        for index, section in enumerate(main_sections):
            self._add_menu_section_header(menu, section.title, first=index == 0)
            self._add_operation_menu_actions(menu, section.entries, dim, anchor)
        if more_sections:
            more_menu = menu.addMenu(material_icon("more_horiz"), "More…")
            more_menu.setToolTipsVisible(True)
            for index, section in enumerate(more_sections):
                self._add_menu_section_header(more_menu, section.title, first=index == 0)
                self._add_operation_menu_actions(more_menu, section.entries, dim, anchor)
        return menu

    @staticmethod
    def _add_menu_section_header(menu, title, *, first):
        """Add a legible, disabled group header to a chip "+" menu.

        Fusion renders ``QMenu.addSection`` as an unlabeled separator, so the
        group title is invisible. Instead use a disabled ``QAction`` carrying
        the uppercased title (styled muted by the app stylesheet), preceded by a
        separator for every group after the first.
        """

        if not first:
            menu.addSeparator()
        header = menu.addAction(title.upper())
        header.setEnabled(False)
        font = header.font()
        font.setBold(True)
        header.setFont(font)

    def _show_operation_context_menu(self, pos, widget, dim):
        if dim >= self.data.ndim:
            return

        # The axis is fixed by the chip, so a parameterized op anchors its
        # stage-2 popup just above the chip's "+" button (this global point).
        anchor = widget.mapToGlobal(pos)
        menu = self._build_operation_context_menu(dim, anchor)
        menu.exec(anchor)

    def _add_operation_menu_actions(self, menu, entries, dim, anchor):
        for entry in entries:
            action = menu.addAction(entry.label)
            set_action_icon(action, _operation_icon_name(entry.id))
            action.setData(entry.id)
            action.setEnabled(self._operation_entry_enabled(entry, dim))
            if entry.unavailable_reason:
                action.setToolTip(entry.unavailable_reason)
                action.setStatusTip(entry.unavailable_reason)
            action.triggered.connect(
                lambda checked=False, operation_id=entry.id, dim=dim, anchor=anchor: (
                    self.request_operation(operation_id, dim, anchor=anchor)
                )
            )

    def _show_operation_context_menu_for_axis(self, dim):
        if dim >= self.data.ndim:
            return
        widget = self.dimension_strip.chip(dim) if hasattr(self, "dimension_strip") else self
        self._show_operation_context_menu(widget.rect().bottomLeft(), widget, dim)

    def _operation_entry_enabled(self, entry, dim):
        return _operation_enabled_for(
            entry, self.data.ndim, self._current_is_complex(), self.data.shape, dim
        )

    def _operation_entry_enabled_anywhere(self, entry):
        """Whether ``entry`` could apply to *some* axis (dock add, no fixed dim)."""

        return _operation_enabled_for(
            entry, self.data.ndim, self._current_is_complex(), self.data.shape, None
        )

    def _append_operation(self, operation_id, dim=None):
        return self.request_operation(operation_id, dim)

    def request_operation(self, operation_id, dim=None, *, anchor=None):
        """Add ``operation_id`` on axis ``dim``.

        A parameterless op is committed immediately. A parameterized op opens
        the (non-modal) parameter popup anchored at ``anchor`` (cursor position
        by default); its confirm commits the collected values. No nested event
        loop -- the append happens in the popup's accept callback.
        """

        try:
            entry = get_operation_entry(operation_id)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Operation Error", f"Failed to apply operation:\n{e}"
            )
            return None
        if entry.unavailable_reason:
            show_status_message(self, entry.unavailable_reason, timeout=4500)
            return None
        form = build_parameter_form(
            entry,
            shape=self.data.shape,
            axis=dim,
            slot_options=self._slot_source_options(entry),
        )
        if form is None:
            return self._commit_operation(operation_id, dim)

        popup = OperationParamsPopup(
            entry,
            form,
            lambda values, bindings, operation_id=operation_id, dim=dim: self._commit_operation(
                operation_id,
                dim,
                parameters=values,
                slot_bindings=bindings,
            ),
            parent=self,
        )
        self._store_operation_popup("_operation_params_popup", popup)
        popup.open_at(anchor if anchor is not None else QtGui.QCursor.pos())
        return None

    def _store_operation_popup(self, attr, popup):
        """Keep one live popup per slot, reaping the prior one.

        The op popups opt out of WA_DeleteOnClose (so an auto-close on focus
        loss cannot delete an object a caller still references), so retire the
        previous popup explicitly instead of letting hidden popups pile up.
        """

        previous = getattr(self, attr, None)
        if previous is not None:
            with contextlib.suppress(RuntimeError):
                previous.deleteLater()
        setattr(self, attr, popup)

    def _commit_operation(
        self,
        operation_id,
        dim=None,
        parameters=None,
        *,
        slot_bindings=None,
    ):
        try:
            self.operation_coordinator.append_operation(
                operation_id,
                axis=dim,
                parameters=parameters or {},
                slot_bindings=slot_bindings,
                slot_resolver=self._resolve_operation_slot,
            )
            if dim is not None:
                self._last_operation_axis = int(dim)
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Operation Error", f"Failed to apply operation:\n{e}"
            )
            return None

        self.render(reason="operation", force_autolevel=True)
        return True

    def open_operation_adder(self, search=False, anchor=None):
        if search:
            return self.open_command_palette()
        popup = OperationAddPopup(
            build_operation_listing(),
            axis_choices=self._axis_choices(),
            default_axis=self._default_operation_axis(),
            is_enabled=self._operation_entry_enabled_anywhere,
            on_search=self.open_command_palette,
            on_accept=lambda op_id, axis, anchor=anchor: self.request_operation(
                op_id, axis, anchor=anchor
            ),
            on_needs_parameters=lambda op_id, axis, anchor=anchor: self.request_operation(
                op_id, axis, anchor=anchor
            ),
            parent=self,
        )
        self._store_operation_popup("_operation_add_popup", popup)
        popup.open_at(anchor if anchor is not None else QtGui.QCursor.pos(), place="below")
        return None

    def open_command_palette(self):
        # The palette is the catalogue's single search surface. Flatten the same
        # library-backed listing used by the popup and dimension menu so hidden
        # operations stay hidden and manager ordering remains authoritative.
        operation_entries = [
            entry for section in build_operation_listing() for entry in section.entries
        ]
        commands = [
            PaletteCommand(
                entry.id,
                entry.label,
                kind="operation",
                requires_axis=entry.requires_axis,
                icon=_operation_icon_name(entry.id),
                enabled=(
                    not bool(entry.unavailable_reason)
                    and self._operation_entry_enabled_anywhere(entry)
                ),
                unavailable_reason=entry.unavailable_reason,
            )
            for entry in operation_entries
        ]
        commands.extend(
            [
                PaletteCommand("fit", "Fit image to viewport", icon="fit_screen"),
                PaletteCommand("one_to_one", "Set image zoom to 1:1", icon="aspect_ratio"),
                PaletteCommand("auto_window", "Auto window levels", icon="tonality"),
                PaletteCommand("reset_layout", "Reset layout", icon="reset_wrench"),
                PaletteCommand("toggle_profile", "Toggle profile dock", icon="show_chart"),
                PaletteCommand("show_inspection", "Show inspection dock", icon="analytics"),
                PaletteCommand("roi_line", "Line ROI tool", icon="show_chart"),
                PaletteCommand("roi_rectangle", "Rectangle ROI tool", icon="crop"),
                PaletteCommand("roi_polyline", "Draw polyline ROI", icon="waves"),
                PaletteCommand("roi_freehand", "Draw freehand ROI", icon="edit"),
                PaletteCommand("export_derived", "Export derived array", icon="download"),
                PaletteCommand("save_recipe", "Save operation recipe", icon="save"),
                PaletteCommand("load_recipe", "Load operation recipe", icon="folder_open"),
                PaletteCommand("save_view_recipe", "Save view recipe", icon="view_quilt"),
                PaletteCommand("load_view_recipe", "Load view recipe", icon="folder_open"),
                PaletteCommand("edit_colormaps", "Edit colormaps", icon="palette"),
                PaletteCommand("manage_operations", "Manage operations", icon="tune"),
            ]
        )
        default_axis = self._default_operation_axis()
        dialog = CommandPaletteDialog(
            commands, axis_choices=self._axis_choices(), default_axis=default_axis, parent=self
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        command, axis = dialog.selected()
        if command is None:
            return None
        if command.kind == "operation":
            return self.request_operation(command.id, axis)
        return self._run_palette_command(command.id)

    def _run_palette_command(self, command_id):
        actions = {
            "edit_colormaps": self.open_colormap_designer,
            "manage_operations": self.open_operation_manager,
            "fit": self.fit_image_to_view,
            "one_to_one": self.one_to_one_image,
            "auto_window": self.auto_window_levels,
            "reset_layout": self.reset_layout,
            "toggle_profile": self.toggle_profile_dock,
            "show_inspection": self._show_inspection_dock,
            "roi_line": lambda: self._select_roi_tool("roi_line"),
            "roi_rectangle": lambda: self._select_roi_tool("roi_rectangle"),
            "roi_polyline": lambda: self._select_roi_tool("roi_polyline"),
            "roi_freehand": lambda: self._select_roi_tool("roi_freehand"),
            "export_derived": self.export_derived_array,
            "save_recipe": self.save_operation_recipe,
            "load_recipe": self.load_operation_recipe,
            "save_view_recipe": self.save_view_recipe,
            "load_view_recipe": self.load_view_recipe,
        }
        action = actions.get(command_id)
        if action is not None:
            return action()
        return None

    def _select_roi_tool(self, tool):
        from arrayscope.window.interaction_mode import InteractionMode

        mode = InteractionMode(tool)
        self.interaction_mode = mode
        if hasattr(self, "widgets") and tool != "profile":
            self.widgets["buttons"]["display"]["live_profile"].setChecked(False)
        if tool in {"roi_polyline", "roi_freehand"}:
            if hasattr(self, "img_view"):
                return self.img_view.beginRoiDrawingOnce(tool)
            return False
        if hasattr(self, "inspection_dock"):
            self.inspection_dock.set_current_tool(tool)
        self._on_inspection_tool_changed(tool)
        self._show_inspection_dock()

    def _axis_choices(self):
        choices = []
        image_axes = self.view_state.image_axes or ()
        for axis, size in enumerate(self.data.shape):
            parts = [f"dim {axis} [{size}]"]
            if len(image_axes) > 0 and image_axes[0] == axis:
                parts.append("Y")
            if len(image_axes) > 1 and image_axes[1] == axis:
                parts.append("X")
            if axis in getattr(self, "profile_axes", ()):
                parts.append("P")
            if axis not in image_axes:
                parts.append(f"slice={self.view_state.slice_indices[axis]}")
            choices.append((" ".join(parts), axis))
        return choices

    def _default_operation_axis(self):
        candidates = []
        focused_axis = getattr(self, "_focused_dimension_axis", None)
        if focused_axis is not None:
            candidates.append(focused_axis)
        last_axis = getattr(self, "_last_operation_axis", None)
        if last_axis is not None:
            candidates.append(last_axis)
        display_axes = set(self.view_state.display_axes())
        candidates.extend(
            axis
            for axis, size in enumerate(self.data.shape)
            if size != 1 and axis not in display_axes
        )
        if self.view_state.line_axis is not None:
            candidates.append(self.view_state.line_axis)
        if self.view_state.image_axes is not None:
            candidates.extend((self.view_state.image_axes[1], self.view_state.image_axes[0]))
        candidates.extend(range(self.data.ndim))
        for axis in candidates:
            if axis is not None and 0 <= int(axis) < self.data.ndim:
                return int(axis)
        return None

    def undo_last_operation(self):
        self.operation_coordinator.undo()
        self._set_document(self.operation_coordinator.document)
        self.render(reason="operation-undo", force_autolevel=True)

    def delete_selected_operation(self, index):
        if index is None:
            return
        try:
            self.operation_coordinator.delete(index)
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Operation Error", f"Cannot delete operation:\n{e}")
            return
        self.render(reason="operation-delete", force_autolevel=True)

    def move_selected_operation(self, index, direction):
        if index is None:
            return
        try:
            self.operation_coordinator.move(index, direction)
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Operation Error", f"Cannot reorder operation:\n{e}"
            )
            return
        self.render(reason="operation-move", force_autolevel=True)

    def reorder_operations(self, order):
        try:
            self.operation_coordinator.reorder(order)
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Operation Error", f"Cannot reorder operation stack:\n{e}"
            )
            self._update_operation_dock()
            return False
        self.render(reason="operation-reorder", force_autolevel=True)
        return True

    def clear_operations(self):
        self.operation_coordinator.clear()
        self._set_document(self.operation_coordinator.document)
        self.render(reason="operation-clear", force_autolevel=True)

    RECENT_RECIPES_KEY = "recent_recipes"
    RECENT_RECIPES_LIMIT = 5

    def recent_recipe_paths(self):
        value = self._settings.value(self.RECENT_RECIPES_KEY)
        if isinstance(value, str):
            value = [value]
        return [str(path) for path in (value or []) if str(path)]

    def _remember_recent_recipe(self, file_path):
        paths = [str(file_path)] + [p for p in self.recent_recipe_paths() if p != str(file_path)]
        self._settings.setValue(self.RECENT_RECIPES_KEY, paths[: self.RECENT_RECIPES_LIMIT])

    def save_operation_recipe(self):
        file_path, _ = get_save_file_name(
            self,
            "Save operation recipe",
            "arrayscope-recipe.json",
            "JSON files (*.json)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".json"):
            file_path += ".json"
        try:
            save_recipe(file_path, self.document.steps)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Recipe Save Error", f"Failed to save recipe:\n{e}")
            return
        self._remember_recent_recipe(file_path)

    def load_operation_recipe(self):
        file_path, _ = get_open_file_name(
            self,
            "Load operation recipe",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not file_path:
            return
        self.load_operation_recipe_from_path(file_path)

    def load_operation_recipe_from_path(self, file_path):
        try:
            steps = load_recipe_steps(
                file_path,
                self.base_data.shape,
                slot_resolver=self._resolve_operation_slot,
            )
            self.operation_coordinator.load_steps(steps)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Recipe Load Error", f"Failed to load recipe:\n{e}")
            return
        try:
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Recipe Load Error", f"Failed to load recipe:\n{e}")
            return
        self._remember_recent_recipe(file_path)
        self.render(reason="recipe-load", force_autolevel=True)

    def populate_recent_recipes_menu(self, menu):
        """Fill `menu` with the most recently saved/loaded recipes."""
        import os

        menu.clear()
        paths = self.recent_recipe_paths()
        if not paths:
            action = menu.addAction("No recent recipes")
            action.setEnabled(False)
            return
        for path in paths:
            action = menu.addAction(os.path.basename(path))
            action.setToolTip(path)
            action.triggered.connect(
                lambda _checked=False, path=path: self.load_operation_recipe_from_path(path)
            )

    def materialize_current_array(self):
        if not self._confirm_expensive_full_array("Materialize", self.data.shape, self.data.dtype):
            return
        document = self.document

        def evaluate():
            return np.array(document.materialize(), copy=True)

        def done(data):
            self.operation_coordinator.replace_base_data(data)
            self._set_document(self.operation_coordinator.document)
            self.render(reason="materialize", force_autolevel=True)
            show_status_message(self, "Materialized current derived array")

        self.evaluation_controller.start(
            evaluate,
            on_done=done,
            on_error=lambda exc: QtWidgets.QMessageBox.warning(
                self, "Materialize Error", f"Failed to materialize:\n{exc}"
            ),
            on_slow=lambda: show_status_message(self, "Materializing derived array..."),
        )

    def set_operation_enabled(self, index, enabled):
        if (
            bool(enabled)
            and 0 <= int(index) < len(self.document.steps)
            and self.document.steps[int(index)].unavailable_reason
        ):
            show_status_message(
                self,
                self.document.steps[int(index)].unavailable_reason,
                timeout=4500,
            )
            return
        try:
            self.operation_coordinator.set_enabled(index, enabled)
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Operation Error", f"Cannot update operation:\n{e}")
            return
        self.render(reason="operation-enabled", force_autolevel=True)

    def edit_operation(self, index, anchor=None):
        """Re-edit any parameterized operation via the shared params popup."""

        if index is None or index < 0 or index >= len(self.document.steps):
            return
        operation = self.document.steps[index].operation
        try:
            operation_id = operation_id_for(operation)
            entry = get_operation_entry(operation_id)
        except Exception:
            return
        if not entry.parameters and not entry.input_slots:
            return
        axis = getattr(operation, "axis", None)
        form = build_parameter_form(
            entry,
            shape=self.base_data.shape,
            axis=axis,
            slot_options=self._slot_source_options(entry),
            slot_bindings=dict(getattr(operation, "slot_bindings", ()) or ()),
        )
        if form is None:
            return
        # Seed the form with the operation's current values so the popup opens
        # showing what is in effect, not the defaults.
        for name, value in operation_parameter_values(operation, entry).items():
            if value is not None:
                form.set_value(name, value)

        def _apply(
            values,
            bindings,
            index=index,
            operation_id=operation_id,
            axis=axis,
        ):
            try:
                self.operation_coordinator.replace_operation(
                    index,
                    operation_id,
                    axis=axis,
                    parameters=values,
                    slot_bindings=bindings,
                    slot_resolver=self._resolve_operation_slot,
                )
                self._set_document(self.operation_coordinator.document)
            except Exception as e:
                show_status_message(self, f"Cannot edit operation: {e}", timeout=4000)
                return
            self.render(reason="operation-edit", force_autolevel=True)

        popup = OperationParamsPopup(entry, form, _apply, parent=self)
        self._store_operation_popup("_operation_params_popup", popup)
        popup.open_at(anchor if anchor is not None else QtGui.QCursor.pos())

    def change_operation_axis(self, index, axis):
        """Re-target an existing operation onto another dimension."""
        if index is None or index < 0 or index >= len(self.document.steps):
            return
        operation = self.document.steps[index].operation
        try:
            operation_id = operation_id_for(operation)
            entry = get_operation_entry(operation_id)
        except Exception:
            return
        if not entry.requires_axis:
            return
        axis = int(axis)
        if axis == int(getattr(operation, "axis", -1)):
            return
        parameters = {
            parameter.name: operation_parameter_values(operation, entry).get(parameter.name)
            for parameter in entry.parameters
        }
        try:
            self.operation_coordinator.replace_operation(
                index,
                operation_id,
                axis=axis,
                parameters=parameters,
                slot_bindings=dict(getattr(operation, "slot_bindings", ()) or ()),
                slot_resolver=self._resolve_operation_slot,
            )
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Operation Error", f"Cannot change dimension:\n{e}")
            return
        self.render(reason="operation-axis", force_autolevel=True)

    def export_derived_array(self):
        file_path, _ = get_save_file_name(
            self,
            "Export derived array",
            "arrayscope-derived.npz",
            "NumPy archive (*.npz);;NumPy array (*.npy)",
        )
        if not file_path:
            return None
        if not self._confirm_expensive_full_array("Export", self.data.shape, self.data.dtype):
            return None
        recipe_json = dumps_recipe(self.document.steps)
        view_recipe_json = self._current_view_recipe_json()
        document = self.document

        def evaluate_and_save():
            data = document.materialize()
            return save_derived_array(
                file_path,
                data,
                recipe_json=recipe_json,
                view_recipe_json=view_recipe_json,
                sidecar=True,
            )

        def done(written):
            show_status_message(self, f"Exported derived array to {written[0]}")

        self.evaluation_controller.start(
            evaluate_and_save,
            on_done=done,
            on_error=lambda exc: QtWidgets.QMessageBox.warning(
                self, "Export Error", f"Failed to export derived array:\n{exc}"
            ),
            on_slow=lambda: show_status_message(self, "Exporting derived array..."),
        )
        return None

    def _confirm_expensive_full_array(self, action, shape, dtype):
        enabled_operations = tuple(self.document.enabled_operations)
        cost = estimate_pipeline_cost(
            self.base_data.shape,
            getattr(self.base_data, "dtype", None),
            enabled_operations,
        )
        output_bytes = cost.estimated_output_bytes or 0
        peak_bytes = cost.estimated_peak_bytes or output_bytes
        budget_bytes = (
            self._visible_render_budget_bytes()
            if hasattr(self, "_visible_render_budget_bytes")
            else DEFAULT_VISIBLE_RENDER_BUDGET_BYTES
        )
        should_warn = (
            output_bytes > LARGE_MATERIALIZE_BYTES
            or peak_bytes > budget_bytes
            or bool(cost.warnings)
        )
        if not should_warn:
            return True
        expensive_text = ""
        if cost.operation_costs and enabled_operations:
            paired = tuple(zip(enabled_operations, cost.operation_costs, strict=False))
            operation, operation_cost = max(
                paired, key=lambda item: item[1].estimated_peak_bytes or 0
            )
            axis_text = (
                ""
                if not operation_cost.requires_full_axis
                else f" axis {operation_cost.requires_full_axis[0]}"
            )
            expensive_text = f" due to {type(operation).__name__}{axis_text}"
        worker_text = ""
        if any(
            type(operation).__name__ in {"CenteredFFT", "CenteredIFFT"}
            for operation in enabled_operations
        ):
            _backend_choice, workers_choice = fft_backend.get_fft_runtime_options()
            worker_text = f" FFT workers: {workers_choice.value}."
        message = (
            f"{action} will evaluate the full derived array "
            f"(shape {tuple(int(size) for size in shape)}, dtype {np.dtype(dtype)}, "
            f"output {format_bytes(output_bytes)}, estimated peak {format_bytes(peak_bytes)}{expensive_text})."
            f"{worker_text} Continue?"
        )
        result = QtWidgets.QMessageBox.warning(
            self,
            f"Large {action}",
            message,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        return result == QtWidgets.QMessageBox.StandardButton.Yes

    def save_view_recipe(self):
        file_path, _ = get_save_file_name(
            self,
            "Save view recipe",
            "arrayscope-view.json",
            "JSON files (*.json)",
        )
        if not file_path:
            return None
        if not file_path.lower().endswith(".json"):
            file_path += ".json"
        save_view_recipe_file(file_path, self._current_view_recipe())
        show_status_message(self, f"Saved view recipe to {file_path}")
        return file_path

    def load_view_recipe(self):
        file_path, _ = get_open_file_name(
            self,
            "Load view recipe",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not file_path:
            return None
        try:
            recipe = load_view_recipe_file(file_path, self.base_data.shape)
            self.operation_coordinator.load_steps(recipe.steps)
            self._set_document(self.operation_coordinator.document)
            self._set_view_state(recipe.view_state.for_shape(self.data.shape, preserve_flags=True))
            self._apply_display_settings(recipe.display)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "View Recipe Error", f"Failed to load view recipe:\n{e}"
            )
            return None
        self.render(
            reason="view-recipe-load",
            force_autolevel=self._pending_display_levels_for_render() is None,
        )
        return file_path

    def _current_view_recipe(self):
        return ViewRecipe(
            view_state=self.view_state,
            display=self._current_display_settings(),
            steps=self.document.steps,
        )

    def _current_view_recipe_json(self):
        from arrayscope.core.view_recipe import dumps_view_recipe

        return dumps_view_recipe(self._current_view_recipe())

    def _current_display_settings(self):
        levels = None
        try:
            levels = tuple(float(value) for value in self.img_view.getLevels())
        except Exception:
            levels = None
        return DisplaySettings(
            channel=self.view_state.channel.value,
            scale=self.view_state.scale.value,
            aspect_mode=getattr(self.img_view, "displayMode", "square_pixels"),
            window_mode=self._current_window_mode(),
            levels=levels,
            colormap=getattr(self, "current_colormap", None)
            if bool(getattr(self, "_colormap_user_selected", False))
            else None,
            profile_visible=hasattr(self, "profile_dock") and self.profile_dock.isVisible(),
            live_profile=self.widgets["buttons"]["display"]["live_profile"].isChecked(),
        )

    def _apply_display_settings(self, settings):
        self._set_view_state(
            self.view_state.with_channel(settings.channel).with_scale(settings.scale)
        )
        self._coerce_channel_for_current_dtype()
        aspect_mode = (
            settings.aspect_mode if settings.aspect_mode in {"square_pixels", "fit"} else "fit"
        )
        self.img_view.setDisplayMode(aspect_mode)
        self.widgets["buttons"]["display"]["window_relative"].setChecked(
            settings.window_mode != "absolute"
        )
        self.widgets["buttons"]["display"]["window_absolute"].setChecked(
            settings.window_mode == "absolute"
        )
        self.widgets["buttons"]["display"]["live_profile"].setChecked(settings.live_profile)
        saved_colormap = None if settings.colormap is None else str(settings.colormap)
        if saved_colormap is None or saved_colormap == default_colormap_name(settings.channel):
            self._colormap_user_selected = False
            self._apply_channel_colormap()
        else:
            self._set_display_colormap(saved_colormap, user_selected=True, request_render=False)
        queued_levels = self._queue_display_levels(
            settings.levels if settings.window_mode == "absolute" else None
        )
        if queued_levels is not None:
            # _set_document and channel coercion intentionally request automatic
            # levels for ordinary state changes. Absolute recipe levels are
            # stronger and must reach the next semantic commit unchanged.
            self._force_autolevel = False
        if settings.profile_visible:
            self._profile_dock_user_visible = True
            if not bool(getattr(self, "_suspend_progressive_dock_sync", False)):
                self.layout_manager.set_managed_dock_visible(
                    self.profile_dock, True, reason="view-recipe"
                )


_REDUCTION_OPERATION_IDS = frozenset({"mean", "rss", "sum", "max", "min"})


def _operation_enabled_for(entry, ndim, is_complex, shape, dim) -> bool:
    """Per-id axis-gating rules shared by the fixed-axis and any-axis checks.

    ``dim`` is the axis the op would target, or ``None`` to ask whether the op
    could apply on *some* axis (the dock add flow, before an axis is chosen).
    """

    if entry.unavailable_reason:
        return False
    if dim is not None and dim >= ndim:
        return False
    if entry.id in _REDUCTION_OPERATION_IDS and ndim <= 1:
        return False
    if entry.id == "combine_real_imag":
        has_pair_axis = shape[dim] == 2 if dim is not None else 2 in tuple(shape)
        return (not is_complex) and has_pair_axis
    if entry.id == "split_complex":
        has_singleton_axis = shape[dim] == 1 if dim is not None else 1 in tuple(shape)
        return is_complex and has_singleton_axis
    return True


def operation_parameter_values(operation, entry) -> dict:
    """Current parameter values of ``operation`` keyed by declared name.

    Built-in operations store each parameter as an attribute (``Crop.start``);
    a :class:`~arrayscope.operations.plugins.PluginOperation` keeps them in its
    opaque ``params`` mapping. Uses the registry's normalizer so the params
    popup can seed from any op's live values.
    """

    return {
        parameter.name: operation_parameter_value(operation, parameter.name)
        for parameter in entry.parameters
    }


def _operation_icon_name(operation_id):
    # Icon knowledge now lives on the registry entry (``entry.icon``); this stays
    # a thin lookup so existing callers are unchanged. Falls back to the generic
    # icon for an unknown/uninstalled id rather than raising.
    try:
        return get_operation_entry(operation_id).icon
    except ValueError:
        return "data_array"
