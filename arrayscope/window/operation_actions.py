from __future__ import annotations

import numpy as np

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui, QtWidgets

from arrayscope.operations.recipes import load_recipe, save_recipe
from arrayscope.operations.registry import operation_entries
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
        for entry in operation_entries():
            action = menu.addAction(entry.label)
            action.setData(entry.id)
            action.setEnabled(self._operation_entry_enabled(entry, dim))
            action.triggered.connect(lambda checked=False, operation_id=entry.id: self._append_operation(operation_id, dim))

        menu.exec(widget.mapToGlobal(pos))

    def _operation_entry_enabled(self, entry, dim):
        if dim >= self.data.ndim:
            return False
        if entry.id in {"mean", "rss", "sum", "max", "min"} and self.data.ndim <= 1:
            return False
        if entry.id == "combine_real_imag":
            return (not np.iscomplexobj(self.data)) and self.data.shape[dim] == 2
        if entry.id == "split_complex":
            return np.iscomplexobj(self.data) and self.data.shape[dim] == 1
        return True

    def _append_operation(self, operation_id, dim=None):
        try:
            parameters = self._collect_operation_parameters(operation_id, dim)
            if parameters is None:
                return
            self.operation_coordinator.append_operation(operation_id, axis=dim, parameters=parameters)
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Operation Error", f"Failed to apply operation:\n{e}")
            return

        self.render(reason="operation", force_autolevel=True)

    def _collect_operation_parameters(self, operation_id, dim):
        if operation_id != "crop":
            return {}

        axis_size = self.data.shape[dim]
        start, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Crop Axis",
            f"Start index for dim {dim}",
            0,
            0,
            axis_size,
            1,
        )
        if not ok:
            return None

        stop, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Crop Axis",
            f"Stop index for dim {dim} (exclusive)",
            axis_size,
            start,
            axis_size,
            1,
        )
        if not ok:
            return None

        return {"start": start, "stop": stop}

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
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
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
            save_recipe(file_path, self.document.operations)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Recipe Save Error", f"Failed to save recipe:\n{e}")

    def load_operation_recipe(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load operation recipe",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not file_path:
            return
        try:
            operations = load_recipe(file_path, self.base_data.shape)
            self.operation_coordinator.load_operations(operations)
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
        self.operation_coordinator.materialize()
        self._set_document(self.operation_coordinator.document)
        self.render(reason="materialize", force_autolevel=True)
