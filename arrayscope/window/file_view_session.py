from __future__ import annotations

from pathlib import Path

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from arrayscope.core.roi import RoiSelection
from arrayscope.core.roi_store import RoiStore
from arrayscope.core.view_session import (
    FileViewSession,
    ViewportSession,
    canonical_file_exists,
    load_session_file,
    metadata_for_file,
    save_session_file,
    settings_key_for_metadata,
)
from arrayscope.display.viewport import ViewportMode
from arrayscope.ui.toasts import show_revert_action, show_status_action, show_status_message


class FileViewSessionMixin:
    def _current_file_session_metadata(self):
        filepath = getattr(self, "_filepath", None)
        if not canonical_file_exists(filepath):
            return None
        return metadata_for_file(
            filepath,
            dataset_path=getattr(self, "_dataset_path", None),
            selector_class_name=getattr(self, "_selector_class_name", None),
            data=getattr(self, "base_data", None),
        )

    def _current_file_view_session(self):
        metadata = self._current_file_session_metadata()
        if metadata is None or self._file_view_session_persistence_disabled(metadata):
            return None
        return FileViewSession(
            metadata=metadata,
            recipe=self._current_view_recipe(),
            viewport=self._current_viewport_session(),
            rois=tuple(getattr(getattr(self, "roi_store", None), "selections", ()) or ()),
            selected_roi_id=getattr(getattr(self, "roi_store", None), "selected_id", None),
        )

    def _current_viewport_session(self):
        view = getattr(getattr(self, "img_view", None), "getView", lambda: None)()
        if view is None:
            return None
        try:
            view_range = view.viewRange()
            normalized = (
                (float(view_range[0][0]), float(view_range[0][1])),
                (float(view_range[1][0]), float(view_range[1][1])),
            )
        except Exception:
            normalized = None
        controller = getattr(self.img_view, "viewport_controller", None)
        mode = getattr(getattr(controller, "mode", None), "value", None) or ViewportMode.AUTO_UNTOUCHED.value
        return ViewportSession(
            mode=str(mode),
            view_range=normalized,
            viewport_shape=self._current_image_viewport_shape(),
        )

    def _save_file_view_session_on_close(self) -> None:
        session = self._current_file_view_session()
        if session is None:
            return
        path = save_session_file(_file_view_session_config_dir(), session)
        self._settings.setValue(settings_key_for_metadata(session.metadata), path.name)
        self._settings.sync()

    def _restore_file_view_session_if_available(self) -> bool:
        current_metadata = self._current_file_session_metadata()
        if current_metadata is None:
            return False
        settings_key = settings_key_for_metadata(current_metadata)
        if self._file_view_session_persistence_disabled(current_metadata):
            show_status_action(
                self,
                "Saved view disabled for this file.",
                "Enable",
                lambda metadata=current_metadata: self._enable_file_view_session_persistence(metadata),
                timeout=7000,
            )
            return False
        filename = self._settings.value(settings_key)
        if not filename:
            return False
        try:
            session = load_session_file(
                _file_view_session_config_dir(),
                current_metadata,
                self.base_data.shape,
                filename=str(filename),
            )
        except Exception as exc:
            show_status_message(self, f"Skipped saved view: {exc}", timeout=7000)
            return False
        if session is None:
            return False
        defaults = self._snapshot_file_session_defaults()
        try:
            self.operation_coordinator.load_steps(session.recipe.steps)
            self._set_document(self.operation_coordinator.document)
            self._set_view_state(session.recipe.view_state.for_shape(self.data.shape, preserve_flags=True))
            self._apply_display_settings(session.recipe.display)
            self._restore_roi_session(session.rois, selected_id=session.selected_roi_id)
            self._pending_file_session_viewport = session.viewport
            self._file_session_restore_defaults = defaults
            self._file_session_restore_applied = False
            self._file_session_restore_message_enabled = True
            self._suppress_montage_autofit_revert_message = bool(
                session.viewport is not None and session.viewport.view_range is not None
            )
        except Exception as exc:
            self._suppress_montage_autofit_revert_message = False
            QtWidgets.QMessageBox.warning(self, "View Restore Error", f"Failed to restore saved view:\n{exc}")
            return False
        return True

    def _snapshot_file_session_defaults(self):
        return {
            "recipe": self._current_view_recipe(),
            "viewport": self._current_viewport_session(),
            "rois": tuple(getattr(getattr(self, "roi_store", None), "selections", ()) or ()),
            "selected_roi_id": getattr(getattr(self, "roi_store", None), "selected_id", None),
        }

    def _restore_roi_session(self, rois, *, selected_id=None) -> None:
        selections = tuple(selection for selection in tuple(rois or ()) if isinstance(selection, RoiSelection))
        if selected_id is not None and str(selected_id) not in {selection.id for selection in selections}:
            selected_id = None
        setter = getattr(self.img_view, "setRoiSelections", None)
        if callable(setter):
            setter(selections, selected_id=selected_id)
        self.roi_store = RoiStore(selections=selections, selected_id=selected_id).replace_all(selections)
        if selected_id is not None:
            self.roi_store = self.roi_store.select(selected_id)
        if hasattr(self, "inspection_dock"):
            self.inspection_dock.set_rois(self.roi_store.selections)

    def _apply_pending_file_session_viewport(self) -> None:
        viewport = getattr(self, "_pending_file_session_viewport", None)
        if viewport is None or getattr(self, "_file_session_restore_applied", False):
            return
        is_visible = getattr(self, "isVisible", None)
        if callable(is_visible) and not bool(is_visible()):
            return
        if not self._file_session_viewport_restore_ready():
            return
        view = getattr(getattr(self, "img_view", None), "getView", lambda: None)()
        if view is None or viewport.view_range is None:
            return
        previous_applying = bool(getattr(self.img_view, "_viewport_applying", False))
        self.img_view._viewport_applying = True
        try:
            view.setRange(xRange=viewport.view_range[0], yRange=viewport.view_range[1], padding=0)
        finally:
            self.img_view._viewport_applying = previous_applying
        sync_viewport = getattr(self.img_view, "_sync_vispy_camera_to_view", None)
        if callable(sync_viewport):
            sync_viewport()
        if getattr(getattr(self, "view_state", None), "montage_axis", None) is not None:
            self._skip_next_montage_viewport_remap = True
        self._schedule_file_session_viewport_retarget()
        controller = getattr(self.img_view, "viewport_controller", None)
        if controller is not None:
            try:
                controller.mode = ViewportMode(str(viewport.mode))
            except Exception:
                controller.mode = ViewportMode.USER
            if controller.mode == ViewportMode.AUTO_UNTOUCHED:
                controller.last_auto_view_range = viewport.view_range
        self._file_session_restore_applied = True
        self._suppress_montage_autofit_revert_message = False
        if bool(getattr(self, "_file_session_restore_message_enabled", True)):
            show_revert_action(
                self,
                "Restored saved view for this file.",
                self._revert_file_view_session_restore,
                timeout=7000,
            )

    def _current_image_viewport_shape(self) -> tuple[int, int] | None:
        try:
            size = self.img_view.graphicsView.viewport().size()
            height = max(1, int(size.height()))
            width = max(1, int(size.width()))
        except Exception:
            return None
        return (height, width)

    def _apply_pending_file_session_viewport_size(self) -> bool:
        viewport = getattr(self, "_pending_file_session_viewport", None)
        target_shape = None if viewport is None else getattr(viewport, "viewport_shape", None)
        if target_shape is None:
            return False
        try:
            target_height = max(1, int(target_shape[0]))
            target_width = max(1, int(target_shape[1]))
        except Exception:
            return False
        current_shape = self._current_image_viewport_shape()
        if current_shape is None:
            return False
        current_height, current_width = current_shape
        dx = int(target_width) - int(current_width)
        dy = int(target_height) - int(current_height)
        if abs(dx) <= 1 and abs(dy) <= 1:
            return False
        if self.isMaximized() or self.isFullScreen():
            return False
        minimum = self.minimumSize()
        new_width = max(int(minimum.width()), int(self.width()) + dx)
        new_height = max(int(minimum.height()), int(self.height()) + dy)
        if new_width == self.width() and new_height == self.height():
            return False
        self.resize(new_width, new_height)
        layout_manager = getattr(self, "layout_manager", None)
        refresh = getattr(layout_manager, "refresh_view_geometry", None)
        if callable(refresh):
            refresh()
        return True

    def _file_session_viewport_restore_ready(self) -> bool:
        if getattr(getattr(self, "view_state", None), "montage_axis", None) is None:
            return True
        session = getattr(self, "_montage_session", None)
        if session is None:
            return False
        if not bool(getattr(session, "display_committed", False)):
            return False
        return getattr(self, "_current_montage_geometry", None) is not None

    def _schedule_pending_file_session_viewport_restore(self) -> None:
        if (
            getattr(self, "_pending_file_session_viewport", None) is None
            or getattr(self, "_file_session_restore_applied", False)
        ):
            return
        Qt.QtCore.QTimer.singleShot(0, self._apply_pending_file_session_viewport)

    def _schedule_file_session_viewport_retarget(self) -> None:
        if getattr(getattr(self, "view_state", None), "montage_axis", None) is None:
            return
        scheduler = getattr(self, "_schedule_montage_viewport_update", None)
        if not callable(scheduler):
            return

        def schedule_once():
            scheduler(delay_ms=0)

        try:
            Qt.QtCore.QTimer.singleShot(0, schedule_once)
        except Exception:
            scheduler(delay_ms=0)

    def _revert_file_view_session_restore(self) -> None:
        defaults = getattr(self, "_file_session_restore_defaults", None)
        if not defaults:
            return
        recipe = defaults["recipe"]
        self.operation_coordinator.load_steps(recipe.steps)
        self._set_document(self.operation_coordinator.document)
        self._set_view_state(recipe.view_state.for_shape(self.data.shape, preserve_flags=True))
        self._apply_display_settings(recipe.display)
        self._restore_roi_session(defaults.get("rois", ()), selected_id=defaults.get("selected_roi_id"))
        self._pending_file_session_viewport = defaults.get("viewport")
        self._file_session_restore_applied = False
        self._file_session_restore_message_enabled = False
        self.render(reason="file-view-session-revert", force_autolevel=True)
        show_status_action(
            self,
            "Reverted saved view for this file.",
            "Don't save",
            self._disable_current_file_view_session_persistence,
            timeout=9000,
        )

    def _file_view_session_persistence_disabled(self, metadata=None) -> bool:
        metadata = self._current_file_session_metadata() if metadata is None else metadata
        if metadata is None:
            return False
        key = settings_key_for_metadata(metadata)
        return bool(self._settings.contains(key) and self._settings.value(key) is None)

    def _disable_current_file_view_session_persistence(self) -> None:
        metadata = self._current_file_session_metadata()
        if metadata is None:
            return
        self._settings.setValue(settings_key_for_metadata(metadata), None)
        self._settings.sync()
        show_status_message(self, "Saved view disabled for this file.", timeout=4000)

    def _enable_file_view_session_persistence(self, metadata=None) -> None:
        metadata = self._current_file_session_metadata() if metadata is None else metadata
        if metadata is None:
            return
        key = settings_key_for_metadata(metadata)
        if self._settings.contains(key) and self._settings.value(key) is None:
            self._settings.remove(key)
            self._settings.sync()
        show_status_message(self, "Saved view will be stored when this file closes.", timeout=5000)


def _file_view_session_config_dir() -> Path:
    try:
        settings_path = Path(Qt.QtCore.QSettings().fileName())
        if str(settings_path):
            return settings_path.parent
    except Exception:
        pass
    location = Qt.QtCore.QStandardPaths.writableLocation(
        Qt.QtCore.QStandardPaths.StandardLocation.AppConfigLocation
    )
    if not location:
        location = Qt.QtCore.QDir.homePath()
    return Path(location)
