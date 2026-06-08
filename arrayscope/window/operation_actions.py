from __future__ import annotations

import numpy as np

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui, QtWidgets

from arrayscope.core.view_recipe import DisplaySettings, ViewRecipe, load_view_recipe as load_view_recipe_file, save_view_recipe as save_view_recipe_file
from arrayscope.io.numpy_save import estimate_nbytes, save_derived_array
from arrayscope.operations.evaluator import LARGE_MATERIALIZE_BYTES
from arrayscope.operations.recipes import dumps_recipe, load_recipe_steps, save_recipe
from arrayscope.operations.registry import get_operation_entry, operation_entries
from arrayscope.ui.command_palette import CommandPaletteDialog, PaletteCommand
from arrayscope.ui.file_dialogs import get_open_file_name, get_save_file_name
from arrayscope.ui.icons import set_action_icon
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
            p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor('black'))
            label.setStyleSheet("font-weight: normal;")
            self._apply_ifft(dim)  # Undo the FFT by applying IFFT
        elif self.domain[dim] == Domain.INV_FOURIER:
            # From IFFT domain, go back to native (undo)
            self.domain[dim] = Domain.NATIVE
            p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor('black'))
            label.setStyleSheet("font-weight: normal;")
            self._apply_fft(dim)  # Undo the IFFT by applying FFT
        elif event.button() == Qt.QtCore.Qt.MouseButton.RightButton:
            # Right click from native: apply IFFT
            self.domain[dim] = Domain.INV_FOURIER
            p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor('green'))
            label.setStyleSheet("font-weight: bold; color: green;")
            self._apply_ifft(dim)
        else:
            # Left click from native: apply FFT
            self.domain[dim] = Domain.FOURIER
            p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor('blue'))
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

    def _show_operation_context_menu(self, pos, widget, dim):
        if dim >= self.data.ndim:
            return

        menu = QtWidgets.QMenu(self)
        profile_action = menu.addAction("Use as profile axis")
        set_action_icon(profile_action, "show_chart")
        profile_action.setEnabled(not self.singleton[dim])
        profile_action.triggered.connect(lambda checked=False, dim=dim: self.set_profile_axis_from_menu(dim))
        live_profile_action = menu.addAction("Live profile from this axis")
        set_action_icon(live_profile_action, "monitor_heart")
        live_profile_action.setCheckable(True)
        live_profile_action.setChecked(self.widgets["buttons"]["display"]["live_profile"].isChecked() and dim in getattr(self, "profile_axes", ()))
        live_profile_action.setEnabled(not self.singleton[dim])
        live_profile_action.triggered.connect(lambda checked=False, dim=dim: self._set_live_profile_for_axis_from_menu(dim, bool(checked)))
        menu.addSeparator()
        for entry in operation_entries():
            action = menu.addAction(entry.label)
            set_action_icon(action, _operation_icon_name(entry.id))
            action.setData(entry.id)
            action.setEnabled(self._operation_entry_enabled(entry, dim))
            action.triggered.connect(lambda checked=False, operation_id=entry.id, dim=dim: self.request_operation(operation_id, dim))

        menu.exec(widget.mapToGlobal(pos))

    def _enable_live_profile_for_axis(self, dim):
        self.set_profile_axes_exactly((dim,))
        self.widgets["buttons"]["display"]["live_profile"].setChecked(True)
        if hasattr(self, "display_toolbar"):
            self.display_toolbar.set_current(live_profile=True)

    def _set_live_profile_for_axis_from_menu(self, dim, enabled):
        if enabled:
            self._enable_live_profile_for_axis(dim)
        else:
            self.widgets["buttons"]["display"]["live_profile"].setChecked(False)

    def set_profile_axis_from_menu(self, dim):
        self.set_profile_axes_exactly((dim,))
        self._profile_dock_user_visible = True
        self.layout_manager.set_managed_dock_visible(self.profile_dock, True, reason="profile-axis-menu")
        self._schedule_view_geometry_refresh()

    def _show_operation_context_menu_for_axis(self, dim):
        if dim >= self.data.ndim:
            return
        widget = self.dimension_strip.chip(dim) if hasattr(self, "dimension_strip") else self
        self._show_operation_context_menu(widget.rect().bottomLeft(), widget, dim)

    def _operation_entry_enabled(self, entry, dim):
        if dim >= self.data.ndim:
            return False
        if entry.id in {"mean", "rss", "sum", "max", "min"} and self.data.ndim <= 1:
            return False
        if entry.id == "combine_real_imag":
            return (not self._current_is_complex()) and self.data.shape[dim] == 2
        if entry.id == "split_complex":
            return self._current_is_complex() and self.data.shape[dim] == 1
        return True

    def _append_operation(self, operation_id, dim=None):
        return self.request_operation(operation_id, dim)

    def request_operation(self, operation_id, dim=None):
        try:
            parameters = self._collect_operation_parameters(operation_id, dim)
            if parameters is None:
                return
            self.operation_coordinator.append_operation(operation_id, axis=dim, parameters=parameters)
            if dim is not None:
                self._last_operation_axis = int(dim)
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Operation Error", f"Failed to apply operation:\n{e}")
            return

        self.render(reason="operation", force_autolevel=True)
        return True

    def _collect_operation_parameters(self, operation_id, dim):
        if operation_id != "crop":
            return {}

        axis_size = self.data.shape[dim]
        return self._crop_parameters_dialog(dim, axis_size)

    def _crop_parameters_dialog(self, dim, axis_size, *, start=0, stop=None):
        stop = axis_size if stop is None else stop
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Crop Axis")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel(f"dim {dim} [0, {axis_size})"))
        start_spin = QtWidgets.QSpinBox(minimum=0, maximum=axis_size, value=int(start))
        stop_spin = QtWidgets.QSpinBox(minimum=0, maximum=axis_size, value=int(stop))
        preview = QtWidgets.QLabel()
        form = QtWidgets.QFormLayout()
        form.addRow("Start", start_spin)
        form.addRow("Stop", stop_spin)
        form.addRow("Output length", preview)
        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        def sync():
            valid = start_spin.value() <= stop_spin.value()
            preview.setText(str(max(0, stop_spin.value() - start_spin.value())))
            buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(valid)

        start_spin.valueChanged.connect(lambda _value: sync())
        stop_spin.valueChanged.connect(lambda _value: sync())
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        sync()
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return {"start": start_spin.value(), "stop": stop_spin.value()}

    def open_operation_adder(self, search=False):
        if search:
            return self.open_command_palette()
        entries = tuple(operation_entries())
        labels = [entry.label for entry in entries]
        label, ok = QtWidgets.QInputDialog.getItem(self, "Add Operation", "Operation", labels, 0, False)
        if not ok:
            return None
        entry = entries[labels.index(label)]
        axis = self._choose_operation_axis(entry) if entry.requires_axis else None
        if entry.requires_axis and axis is None:
            return None
        return self.request_operation(entry.id, axis)

    def open_command_palette(self):
        commands = [
            PaletteCommand(entry.id, entry.label, kind="operation", requires_axis=entry.requires_axis, icon=_operation_icon_name(entry.id))
            for entry in operation_entries()
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
                PaletteCommand("roi_polyline", "Polyline ROI tool", icon="waves"),
                PaletteCommand("roi_freehand", "Freehand ROI tool", icon="edit"),
                PaletteCommand("export_derived", "Export derived array", icon="download"),
                PaletteCommand("save_recipe", "Save operation recipe", icon="save"),
                PaletteCommand("load_recipe", "Load operation recipe", icon="folder_open"),
                PaletteCommand("save_view_recipe", "Save view recipe", icon="view_quilt"),
                PaletteCommand("load_view_recipe", "Load view recipe", icon="folder_open"),
            ]
        )
        default_axis = self._default_operation_axis()
        dialog = CommandPaletteDialog(commands, axis_choices=self._axis_choices(), default_axis=default_axis, parent=self)
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
        if hasattr(self, "inspection_dock"):
            self.inspection_dock.set_current_tool(tool)
        self._on_inspection_tool_changed(tool)
        self._show_inspection_dock()

    def _choose_operation_axis(self, entry):
        choices = self._axis_choices()
        if not choices:
            return None
        default = self._default_operation_axis()
        labels = [label for label, _axis in choices]
        default_index = 0
        if default is not None:
            for index, (_label, axis) in enumerate(choices):
                if axis == default:
                    default_index = index
                    break
        label, ok = QtWidgets.QInputDialog.getItem(self, entry.label, "Axis", labels, default_index, False)
        if not ok:
            return None
        return choices[labels.index(label)][1]

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
        candidates.extend(axis for axis, size in enumerate(self.data.shape) if size != 1 and axis not in display_axes)
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
            QtWidgets.QMessageBox.warning(self, "Operation Error", f"Cannot reorder operation:\n{e}")
            return
        self.render(reason="operation-move", force_autolevel=True)

    def reorder_operations(self, order):
        try:
            self.operation_coordinator.reorder(order)
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Operation Error", f"Cannot reorder operation stack:\n{e}")
            self._update_operation_dock()
            return False
        self.render(reason="operation-reorder", force_autolevel=True)
        return True

    def clear_operations(self):
        self.operation_coordinator.clear()
        self._set_document(self.operation_coordinator.document)
        self.render(reason="operation-clear", force_autolevel=True)

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

    def load_operation_recipe(self):
        file_path, _ = get_open_file_name(
            self,
            "Load operation recipe",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not file_path:
            return
        try:
            steps = load_recipe_steps(file_path, self.base_data.shape)
            self.operation_coordinator.load_steps(steps)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Recipe Load Error", f"Failed to load recipe:\n{e}")
            return
        try:
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Recipe Load Error", f"Failed to load recipe:\n{e}")
            return
        self.render(reason="recipe-load", force_autolevel=True)

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
            on_error=lambda exc: QtWidgets.QMessageBox.warning(self, "Materialize Error", f"Failed to materialize:\n{exc}"),
            on_slow=lambda: show_status_message(self, "Materializing derived array..."),
        )

    def set_operation_enabled(self, index, enabled):
        try:
            self.operation_coordinator.set_enabled(index, enabled)
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Operation Error", f"Cannot update operation:\n{e}")
            return
        self.render(reason="operation-enabled", force_autolevel=True)

    def edit_operation(self, index):
        if index is None or index < 0 or index >= len(self.document.steps):
            return
        step = self.document.steps[index]
        operation = step.operation
        if type(operation).__name__ != "Crop":
            return
        axis_size = self.base_data.shape[operation.axis]
        params = self._crop_parameters_dialog(operation.axis, axis_size, start=operation.start, stop=operation.stop)
        if params is None:
            return
        try:
            self.operation_coordinator.replace_operation(index, "crop", axis=operation.axis, parameters=params)
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Operation Error", f"Cannot edit operation:\n{e}")
            return
        self.render(reason="operation-edit", force_autolevel=True)

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
            return save_derived_array(file_path, data, recipe_json=recipe_json, view_recipe_json=view_recipe_json, sidecar=True)

        def done(written):
            show_status_message(self, f"Exported derived array to {written[0]}")

        self.evaluation_controller.start(
            evaluate_and_save,
            on_done=done,
            on_error=lambda exc: QtWidgets.QMessageBox.warning(self, "Export Error", f"Failed to export derived array:\n{exc}"),
            on_slow=lambda: show_status_message(self, "Exporting derived array..."),
        )
        return None

    def _confirm_expensive_full_array(self, action, shape, dtype):
        nbytes = estimate_nbytes(shape, dtype)
        expensive_fft = any(
            type(step.operation).__name__ in {"CenteredFFT", "CenteredIFFT"}
            and getattr(step.operation, "axis", 0) < len(self.base_data.shape)
            and self.base_data.shape[getattr(step.operation, "axis")] > 4096
            for step in self.document.steps
            if step.enabled
        )
        if nbytes <= LARGE_MATERIALIZE_BYTES and not expensive_fft:
            return True
        message = f"{action} will evaluate the full derived array ({_format_nbytes(nbytes)}). Continue?"
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
            QtWidgets.QMessageBox.warning(self, "View Recipe Error", f"Failed to load view recipe:\n{e}")
            return None
        self.render(reason="view-recipe-load", force_autolevel=True)
        return file_path

    def _current_view_recipe(self):
        return ViewRecipe(view_state=self.view_state, display=self._current_display_settings(), steps=self.document.steps)

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
            colormap=getattr(self, "current_colormap", None),
            profile_visible=hasattr(self, "profile_dock") and self.profile_dock.isVisible(),
            live_profile=self.widgets["buttons"]["display"]["live_profile"].isChecked(),
        )

    def _apply_display_settings(self, settings):
        self._set_view_state(self.view_state.with_channel(settings.channel).with_scale(settings.scale))
        self._coerce_channel_for_current_dtype()
        aspect_mode = settings.aspect_mode if settings.aspect_mode in {"square_pixels", "fit"} else "fit"
        self.img_view.setDisplayMode(aspect_mode)
        self.widgets["buttons"]["display"]["window_relative"].setChecked(settings.window_mode != "absolute")
        self.widgets["buttons"]["display"]["window_absolute"].setChecked(settings.window_mode == "absolute")
        self.widgets["buttons"]["display"]["live_profile"].setChecked(settings.live_profile)
        if settings.levels is not None:
            try:
                self.img_view.setLevels(*settings.levels)
            except Exception:
                pass
        if settings.profile_visible:
            self._profile_dock_user_visible = True
            self.layout_manager.set_managed_dock_visible(self.profile_dock, True, reason="view-recipe")


def _operation_icon_name(operation_id):
    return {
        "crop": "crop",
        "mean": "functions",
        "sum": "functions",
        "max": "vertical_align_top",
        "min": "vertical_align_bottom",
        "rss": "analytics",
        "centered_fft": "waves",
        "centered_ifft": "waves",
        "combine_real_imag": "join_inner",
        "split_complex": "call_split",
    }.get(operation_id, "data_array")


def _format_nbytes(nbytes):
    nbytes = int(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
