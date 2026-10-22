from __future__ import annotations

import numpy as np
from pyqtgraph.Qt import QtWidgets

from arrayscope.core.compute_policy import ComputeLane
from arrayscope.io.numpy_export_qt import ArrayExportSession, ExportSourceWatch
from arrayscope.io.numpy_save_qt import collect_numpy_save_request
from arrayscope.io.region_export import validate_region_export_snapshot
from arrayscope.ui.icons import set_button_icon
from arrayscope.window.domain import Domain


class FileReloadMixin:
    def _save_current_numpy_file(self):
        """Choose an exact selection, then export it without blocking the GUI."""
        active = getattr(self, "_numpy_export_session", None)
        if active is not None and active.is_running:
            active.show_progress()
            return active
        from arrayscope.app.open_flow import _ACTIVE_SESSIONS

        if any(session.window is self and not session._terminal for session in _ACTIVE_SESSIONS):
            QtWidgets.QMessageBox.information(
                self,
                "File is still loading",
                "Wait for loading to finish, or stop loading, before exporting this array.",
            )
            return None
        document = self.document
        dtype = self.data.dtype
        try:
            with ExportSourceWatch(document) as source_watch:
                request = collect_numpy_save_request(self, document, self._filepath, dtype=dtype)
            if request is None or getattr(self, "_closing", False):
                return None
            if source_watch.changed:
                raise ValueError(source_watch.reason)
            if (
                self.document.base_data is document.base_data
                and self.document.revision != document.revision
            ):
                raise ValueError(
                    "The source changed while choosing export options. Try exporting again."
                )
            validate_region_export_snapshot(request)
            session = ArrayExportSession(
                self,
                request,
                evaluation_context=self._evaluation_context(ComputeLane.STAGE),
                dtype=dtype,
            )
            self._numpy_export_session = session
            return session.start()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save Error", f"Failed to save NumPy file:\n{e}")
            return None

    def _on_file_changed(self, path):
        """Called by QFileSystemWatcher when the source file changes on disk."""
        set_button_icon(self._reload_btn, "warning", tooltip="File changed - click to reload")
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
                from arrayscope.io.selectors import (
                    H5DatasetSelector,
                    MatDatasetSelector,
                    NpzDatasetSelector,
                )

                selector_map = {
                    "H5DatasetSelector": H5DatasetSelector,
                    "NpzDatasetSelector": NpzDatasetSelector,
                    "MatDatasetSelector": MatDatasetSelector,
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
                        return  # User cancelled; keep the file-changed indicator visible.
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
            set_button_icon(self._reload_btn, "refresh", tooltip="Reload file")
            self._reload_btn.setToolTip("Reload file")
            if self._file_watcher and self._filepath:
                self._file_watcher.addPath(str(self._filepath))
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Reload Error", f"Failed to reload:\n{e}")

    def _reset_data(self, new_data):
        """Replace the displayed data, clamping slice positions to the new shape."""
        old_ndim = self.data.ndim
        reload_viewport = self._begin_reload_viewport_continuity()
        # The stored hover focus predates the reload (typically wherever the
        # pointer last crossed the image before reaching the reload control).
        # The transitional reload viewport is huge, so the stale point passes
        # the in-viewport validation and would anchor the whole montage
        # fill order there.
        clear_hover = getattr(getattr(self, "renderer", None), "_clear_image_hover_state", None)
        if callable(clear_hover):
            clear_hover()

        preserve_steps = bool(getattr(self.document, "steps", ()))
        try:
            self._reload_base_data(new_data, preserve_steps=preserve_steps)
        except Exception as exc:
            if not preserve_steps:
                raise
            choice = self._confirm_reload_clearing_operations(exc)
            if choice == "cancel":
                return
            if choice == "save":
                saved = self.save_view_recipe()
                if not saved:
                    return
            self._reload_base_data(new_data, preserve_steps=False)

        if self.data.ndim != old_ndim:
            # Per-dimension widgets were built for old_ndim and cannot be
            # rebuilt in place, so the reloaded array moves to a fresh window
            # (arrayscope.window.rank_handoff states what that carries and what
            # it drops). The reloaded data replaces the base array, so the
            # steps that survived the reload above travel with it.
            self.open_window_at_new_rank(
                new_data,
                steps=self.document.steps or None,
                axes=self.document.base_axes,
            )
            return

        self.singleton = [e == 1 for e in self.data.shape]

        if np.issubdtype(np.dtype(self.data.dtype), np.complexfloating):
            self.can_combine_as_complex = [False] * self.data.ndim
        else:
            self.can_combine_as_complex = [self.data.shape[i] == 2 for i in range(self.data.ndim)]
        self.combined_as_complex = [False] * self.data.ndim

        # Reset FFT domain state and dim label styling
        self.domain = [Domain.NATIVE for _ in range(self.data.ndim)]
        for i, label in enumerate(self.widgets["labels"]["dims"]):
            label.setStyleSheet(self.DIMENSION_LABEL_STYLE)
            if i < self.data.ndim:
                label.setText(f"[{self.data.shape[i]}]")

        # Update spinbox maximums (auto-clamps current value)
        for i in range(self.data.ndim):
            self.widgets["spins"]["slice_indices"][i].setMaximum(self.data.shape[i] - 1)

        self._set_view_state(self.view_state.for_shape(self.data.shape, preserve_flags=True))
        self._update_channel_controls()
        self.update_complex_indicators()
        self.update_shift_indicators()
        self.update_dimension_controls()
        try:
            self.render(reason="reload", force_autolevel=False)
        finally:
            self._finish_reload_viewport_continuity(reload_viewport)

    def _begin_reload_viewport_continuity(self):
        if getattr(getattr(self, "view_state", None), "montage_axis", None) is None:
            return None
        try:
            view_range = self.img_view.getView().viewRange()
            preserved = (
                (float(view_range[0][0]), float(view_range[0][1])),
                (float(view_range[1][0]), float(view_range[1][1])),
            )
        except Exception:
            return None
        controller = getattr(getattr(self, "img_view", None), "viewport_controller", None)
        mode = getattr(getattr(controller, "mode", None), "value", None)
        return self._begin_viewport_continuity(
            reason="reload",
            view_range=preserved,
            viewport_shape=self._current_image_viewport_shape(),
            mode=mode,
            semantic_key=("reload", getattr(getattr(self, "document", None), "revision", None)),
            message_enabled=False,
        )

    def _finish_reload_viewport_continuity(self, reload_viewport) -> None:
        if reload_viewport is None:
            return
        self._apply_viewport_continuity_when_ready()
        self.retarget_montage_viewport()

    def _confirm_reload_clearing_operations(self, exc):
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Reload Incompatible With Operations")
        box.setText("The reloaded data is not compatible with the current operation stack.")
        box.setInformativeText(
            f"{exc}\n\nClear operations and reload, save a view recipe first, or cancel?"
        )
        clear_button = box.addButton(
            "Reload and clear operations", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
        )
        save_button = box.addButton(
            "Save view recipe first", QtWidgets.QMessageBox.ButtonRole.ActionRole
        )
        cancel_button = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_button)
        box.exec()
        clicked = box.clickedButton()
        if clicked is clear_button:
            return "clear"
        if clicked is save_button:
            return "save"
        return "cancel"
