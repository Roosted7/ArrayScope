from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui, QtWidgets

from arrayscope.display.colormaps import gray_colormap, named_colormap, phase_colormap
from arrayscope.ui.dialogs import SaveRangeDialog
from arrayscope.operations.recipes import load_recipe, save_recipe
from arrayscope.operations.registry import operation_entries
from arrayscope.profiles.model import clamp_marker_position, image_hover_indices, profile_y_range
from arrayscope.display.slice_engine import apply_channel
from arrayscope.app.settings_state import AppSettingsState, settings_from_mapping, settings_to_mapping
from arrayscope.app.theme import ThemeChoice, apply_theme_to_qapplication
from arrayscope.export.video import VideoExportWorker, VideoExportDialog, VideoExportSettingsDialog
from arrayscope.core.view_state import ChannelMode, ScaleMode
from arrayscope.core.window_levels import choose_window_levels
from arrayscope.io.numpy_save_qt import save_current_numpy_file
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.domain import Domain


class FileReloadMixin:
    def _save_current_numpy_file(self):
        """Save the currently displayed array state to a NumPy .npy file."""
        try:
            result = save_current_numpy_file(self, self.data, self._filepath)
            if result is not None:
                file_path, output_shape = result
                show_status_message(self, f"Saved array {list(output_shape)} to {file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save Error", f"Failed to save NumPy file:\n{e}")

    def _on_file_changed(self, path):
        """Called by QFileSystemWatcher when the source file changes on disk."""
        self._reload_btn.setText("⚠️")
        self._reload_btn.setToolTip("File changed — click to reload")
        # Re-add the path: handles atomic replacement where the original inode disappears
        if self._file_watcher:
            self._file_watcher.addPath(path)

    def _reload_file(self):
        """Reload data from the source file, preserving slice positions where possible."""
        if self._filepath is None:
            return
        try:
            new_data = None
            new_dataset_path = self._dataset_path

            if self._selector_class_name is None:
                from arrayscope.io.file_interpreters import load_file
                new_data = load_file(self._filepath)
            else:
                from arrayscope.io.selectors import H5DatasetSelector, NpzDatasetSelector, MatDatasetSelector
                selector_map = {
                    'H5DatasetSelector': H5DatasetSelector,
                    'NpzDatasetSelector': NpzDatasetSelector,
                    'MatDatasetSelector': MatDatasetSelector,
                }
                selector_cls = selector_map.get(self._selector_class_name)
                if selector_cls is None:
                    return
                selector = selector_cls(self._filepath)
                compatible_keys = {d[0] for d in selector.compatible_datasets}
                if self._dataset_path is not None and self._dataset_path in compatible_keys:
                    new_data = selector.load_data(self._dataset_path)
                    selector.close()
                elif selector.requires_gui():
                    selected = selector.show()
                    if selected is None:
                        selector.close()
                        return  # User cancelled — keep ⚠️ visible
                    new_data = selector.load_data(selected)
                    new_dataset_path = selected
                    selector.close()
                else:
                    result = selector.get_single_data()
                    selector.close()
                    if result is None:
                        return
                    new_dataset_path, new_data = result

            if new_data is None:
                return

            self._dataset_path = new_dataset_path
            self._reset_data(new_data)
            self._reload_btn.setText("⟳")
            self._reload_btn.setToolTip("Reload file")
            if self._file_watcher and self._filepath:
                self._file_watcher.addPath(str(self._filepath))
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Reload Error", f"Failed to reload:\n{e}")

    def _reset_data(self, new_data):
        """Replace the displayed data, clamping slice positions to the new shape."""
        old_ndim = self.data.ndim
        new_ndim = new_data.ndim

        if new_ndim != old_ndim:
            # Per-dimension widgets were built for old_ndim and cannot be rebuilt in-place.
            # Open a fresh window with the new data and close this one.
            win = type(self)(new_data,
                                filepath=self._filepath,
                                dataset_path=self._dataset_path,
                                selector_class_name=self._selector_class_name)
            win.setWindowTitle(self.windowTitle())
            win.show()
            self.close()
            return

        self._replace_base_data(new_data)
        self.singleton = [e == 1 for e in new_data.shape]

        if np.iscomplexobj(new_data):
            self.can_combine_as_complex = [False] * new_ndim
        else:
            self.can_combine_as_complex = [new_data.shape[i] == 2 for i in range(new_ndim)]
        self.combined_as_complex = [False] * new_ndim

        # Reset FFT domain state and dim label styling
        self.domain = [Domain.NATIVE for _ in range(new_ndim)]
        for i, label in enumerate(self.widgets['labels']['dims']):
            label.setStyleSheet(self.DIMENSION_LABEL_STYLE)
            label.setText(f'[{new_data.shape[i]}]')

        # Update spinbox maximums (auto-clamps current value)
        for i in range(new_ndim):
            self.widgets['spins']['slice_indices'][i].setMaximum(new_data.shape[i] - 1)

        self._set_view_state(self.view_state.for_shape(new_data.shape, preserve_flags=True))
        self._update_channel_controls()
        self.update_complex_indicators()
        self.update_shift_indicators()
        self.update_dimension_controls()
        self._force_autolevel = True
        self.render(reason="reload", force_autolevel=True)
