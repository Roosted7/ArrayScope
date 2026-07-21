from __future__ import annotations

import math
from pathlib import Path

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from arrayscope.core.roi import RoiSelection
from arrayscope.core.roi_store import RoiStore
from arrayscope.core.view_session import (
    FileViewSession,
    PanelSession,
    ViewportSession,
    canonical_file_exists,
    load_session_file,
    metadata_for_file,
    save_session_file,
    settings_key_for_metadata,
)
from arrayscope.display.viewport import ViewportMode, constrain_view_range
from arrayscope.ui.toasts import show_revert_action, show_status_action, show_status_message
from arrayscope.window.viewport_continuity import ViewportContinuityTransaction

FILE_SESSION_VIEWPORT_RESTORE_RETRY_MS = 16
FILE_SESSION_VIEWPORT_RESTORE_ATTEMPTS = 12


class FileViewSessionMixin:
    def _viewport_continuity_transaction(self) -> ViewportContinuityTransaction | None:
        tx = getattr(self, "_viewport_continuity", None)
        return tx if isinstance(tx, ViewportContinuityTransaction) else None

    def _begin_viewport_continuity(
        self,
        *,
        reason: str,
        viewport: ViewportSession | None = None,
        view_range=None,
        viewport_shape: tuple[int, int] | None = None,
        montage_columns: int | None = None,
        mode: str | None = None,
        semantic_key: object | None = None,
        profile_visible: bool = False,
        defaults: dict[str, object] | None = None,
        message_enabled: bool = True,
    ) -> ViewportContinuityTransaction:
        generation = int(getattr(self, "_viewport_continuity_generation", 0) or 0) + 1
        self._viewport_continuity_generation = generation
        tx = ViewportContinuityTransaction(
            reason=reason,
            generation=generation,
            viewport=viewport,
            view_range=view_range,
            viewport_shape=viewport_shape,
            montage_columns=montage_columns,
            mode=mode,
            semantic_key=semantic_key,
            profile_visible=profile_visible,
            defaults=defaults,
            message_enabled=message_enabled,
        )
        if tx.viewport_shape is not None and self._viewport_continuity_shape_matches(
            tx.viewport_shape
        ):
            tx.shape_settled = True
        self._viewport_continuity = tx
        return tx

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
            panels=PanelSession(
                operation_visible=bool(
                    hasattr(self, "operation_dock") and self.operation_dock.isVisible()
                ),
                inspection_visible=bool(
                    hasattr(self, "inspection_dock") and self.inspection_dock.isVisible()
                ),
                window_size=(int(self.width()), int(self.height())),
                window_maximized=bool(self.isMaximized()),
            ),
        )

    def _current_viewport_session(self):
        tx = self._viewport_continuity_transaction()
        if (
            tx is not None
            and not tx.released
            and tx.view_range is not None
            and not tx.range_applied
            and tx.viewport is not None
        ):
            return tx.viewport
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
        if normalized is not None and self._viewport_continuity_content_rect() is None:
            # A ViewBox with no content bounds still reports a range: its
            # pristine default, x spanning the viewport aspect and y 0..1.
            # That is a placeholder for content that never arrived, not a
            # viewport anyone chose, and recording it as one is how a saved
            # view becomes a camera parked on data pixels (0,0)-(1,1) of
            # whatever is opened next.  Store no range; every restore path
            # already fits when the range is absent.
            normalized = None
        controller = getattr(self.img_view, "viewport_controller", None)
        mode = (
            getattr(getattr(controller, "mode", None), "value", None)
            or ViewportMode.AUTO_UNTOUCHED.value
        )
        montage_columns = None
        if getattr(getattr(self, "view_state", None), "montage_axis", None) is not None:
            plan = getattr(self, "_current_montage_plan", None)
            if plan is None:
                plan = getattr(getattr(self, "_frame_session", None), "plan", None)
            columns = getattr(plan, "columns", None)
            if columns is not None:
                montage_columns = max(1, int(columns))
        return ViewportSession(
            mode=str(mode),
            view_range=normalized,
            viewport_shape=self._current_image_viewport_shape(),
            montage_columns=montage_columns,
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
                lambda metadata=current_metadata: self._enable_file_view_session_persistence(
                    metadata
                ),
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
        previous_suspend_progressive = bool(getattr(self, "_suspend_progressive_dock_sync", False))
        self._suspend_progressive_dock_sync = True
        try:
            self.operation_coordinator.load_steps(session.recipe.steps)
            self._set_document(self.operation_coordinator.document)
            self._set_view_state(
                session.recipe.view_state.for_shape(self.data.shape, preserve_flags=True)
            )
            self._apply_display_settings(session.recipe.display)
            self._restore_roi_session(session.rois, selected_id=session.selected_roi_id)
            self._pending_file_session_panels = session.panels
            self._begin_viewport_continuity(
                reason="file-session-restore",
                viewport=session.viewport,
                profile_visible=bool(session.recipe.display.profile_visible),
                defaults=defaults,
            )
            if (
                session.viewport is not None
                and session.viewport.view_range is not None
                and getattr(getattr(self, "view_state", None), "montage_axis", None) is not None
                and str(session.viewport.mode) != ViewportMode.FIT.value
            ):
                controller = getattr(getattr(self, "img_view", None), "viewport_controller", None)
                if controller is not None:
                    controller.mode = ViewportMode.USER
            self._suppress_montage_autofit_revert_message = bool(
                session.viewport is not None and session.viewport.view_range is not None
            )
        except Exception as exc:
            self._suppress_montage_autofit_revert_message = False
            QtWidgets.QMessageBox.warning(
                self, "View Restore Error", f"Failed to restore saved view:\n{exc}"
            )
            return False
        finally:
            self._suspend_progressive_dock_sync = previous_suspend_progressive
        return True

    def _pending_viewport_continuity_range(self):
        tx = self._viewport_continuity_transaction()
        if tx is None or tx.released or tx.range_applied:
            return None
        return tx.view_range

    def _pending_viewport_continuity_columns(self):
        tx = self._viewport_continuity_transaction()
        if tx is None or tx.released or tx.range_applied:
            return None
        return tx.montage_columns

    def _active_viewport_continuity_range(self):
        tx = self._viewport_continuity_transaction()
        if tx is None or tx.released or tx.view_range is None:
            return None
        # A settled restore remains consumed after a later user resize changes
        # the live viewport shape. Layout transitions that must re-establish
        # the saved shape explicitly reopen the transaction by clearing
        # ``shape_settled`` in the shape-restore path.
        if tx.range_applied and tx.shape_settled:
            return None
        return tx.view_range

    def _release_viewport_continuity(self) -> None:
        tx = self._viewport_continuity_transaction()
        if tx is not None:
            tx.released = True

    def _viewport_continuity_shape_target(self) -> tuple[int, int] | None:
        tx = self._viewport_continuity_transaction()
        if tx is None or tx.released:
            return None
        if (
            tx.viewport_shape is not None
            and tx.shape_settled
            and self._viewport_continuity_shape_matches(tx.viewport_shape)
        ):
            return None
        viewport = None if tx is None else tx.viewport
        return None if viewport is None else viewport.viewport_shape

    def _restore_viewport_continuity_shape_after_layout(self) -> None:
        self._schedule_viewport_continuity_shape_restore()

    def _schedule_viewport_continuity_shape_restore(self) -> None:
        viewport_shape = self._viewport_continuity_shape_target()
        if viewport_shape is None:
            return
        tx = self._viewport_continuity_transaction()
        if tx is None or tx.viewport_shape is None or tx.shape_settled:
            if tx is None or tx.viewport_shape is None:
                return
            if tx.shape_settled and self._viewport_continuity_shape_matches(tx.viewport_shape):
                if tx.view_range is None:
                    tx.released = True
                    if tx.reason == "settings-restore":
                        self._viewport_continuity = None
                return
            tx.shape_settled = False
        if getattr(self, "_viewport_continuity_shape_generation", None) == tx.generation:
            return
        self._viewport_continuity_shape_generation = tx.generation
        # Timer category: UI cosmetic. Qt event-turn barrier. Viewport sizing must wait until the current
        # show/dock/layout pass has produced the real graphics viewport.
        Qt.QtCore.QTimer.singleShot(
            0,
            self,
            lambda shape=tuple(viewport_shape), generation=int(tx.generation): (
                self._restore_viewport_continuity_shape_step(
                    shape,
                    attempts=FILE_SESSION_VIEWPORT_RESTORE_ATTEMPTS,
                    generation=generation,
                )
            ),
        )

    def _restore_viewport_continuity_shape_step(
        self, viewport_shape, *, attempts: int, generation: int | None = None
    ) -> None:
        tx = self._viewport_continuity_transaction()
        if tx is None or (generation is not None and int(tx.generation) != int(generation)):
            return
        if bool(getattr(self, "_closing", False)):
            tx.shape_settled = True
            return
        try:
            self.isVisible()
        except RuntimeError:
            tx.shape_settled = True
            return
        layout_manager = getattr(self, "layout_manager", None)
        resize = getattr(layout_manager, "resize_to_dockless_viewport_shape", None)
        resized = False
        if callable(resize):
            resized = bool(resize(viewport_shape))
        if resized and attempts > 1:
            # A resize can produce a transient matching viewport before Qt has
            # completed the child-layout pass.  In particular, VisPy's stacked
            # native canvas settles one event turn after the outer window.  Do
            # not let that same-turn match release the continuity transaction.
            Qt.QtCore.QTimer.singleShot(
                FILE_SESSION_VIEWPORT_RESTORE_RETRY_MS,
                self,
                lambda shape=tuple(viewport_shape), attempts=int(attempts) - 1, generation=generation: (
                    self._restore_viewport_continuity_shape_step(
                        shape,
                        attempts=attempts,
                        generation=generation,
                    )
                ),
            )
            return
        if attempts <= 1 or self._viewport_continuity_shape_matches(viewport_shape):
            tx.shape_settled = True
            self._viewport_continuity_shape_generation = None
            if tx.view_range is None:
                tx.released = True
                if tx.reason == "settings-restore":
                    self._viewport_continuity = None
                return
            if tx is not None:
                if not tx.released and tx.view_range is not None:
                    self._apply_viewport_continuity_when_ready()
                else:
                    self._reapply_viewport_continuity_range_after_layout()
            return
        # Timer category: UI cosmetic. Qt layout retry. Bounded by FILE_SESSION_VIEWPORT_RESTORE_ATTEMPTS and
        # removable when viewport restore has a reliable post-layout signal.
        Qt.QtCore.QTimer.singleShot(
            FILE_SESSION_VIEWPORT_RESTORE_RETRY_MS,
            self,
            lambda shape=tuple(viewport_shape), attempts=int(attempts) - 1, generation=generation: (
                self._restore_viewport_continuity_shape_step(
                    shape,
                    attempts=attempts,
                    generation=generation,
                )
            ),
        )

    def _viewport_continuity_shape_matches(self, viewport_shape) -> bool:
        try:
            target_height = max(1, int(viewport_shape[0]))
            target_width = max(1, int(viewport_shape[1]))
            viewport = self.img_view.graphicsView.viewport()
            return (
                abs(int(viewport.width()) - target_width) <= 1
                and abs(int(viewport.height()) - target_height) <= 1
            )
        except Exception:
            return True

    def _apply_file_session_layout_intent(self) -> None:
        restore = self._viewport_continuity_transaction()
        if restore is not None and restore.reason != "file-session-restore":
            restore = None
        if restore is not None and restore.profile_visible:
            self._profile_dock_user_visible = True
        panels = getattr(self, "_pending_file_session_panels", None)
        if panels is None:
            return
        self._pending_file_session_panels = None
        if panels.window_maximized is not None:
            if bool(panels.window_maximized):
                self.showMaximized()
            else:
                self.showNormal()
                if panels.window_size is not None:
                    self.resize(int(panels.window_size[0]), int(panels.window_size[1]))
        if hasattr(self, "operation_dock"):
            self._operation_dock_user_visible = bool(panels.operation_visible)
            self.layout_manager.set_managed_dock_visible(
                self.operation_dock,
                bool(panels.operation_visible),
                reason="file-session-restore",
                preserve_canvas=False,
            )
        if hasattr(self, "inspection_dock"):
            self._inspection_dock_user_visible = bool(panels.inspection_visible)
            self.layout_manager.set_managed_dock_visible(
                self.inspection_dock,
                bool(panels.inspection_visible),
                reason="file-session-restore",
                preserve_canvas=False,
            )

    def _snapshot_file_session_defaults(self):
        return {
            "recipe": self._current_view_recipe(),
            "viewport": self._current_viewport_session(),
            "rois": tuple(getattr(getattr(self, "roi_store", None), "selections", ()) or ()),
            "selected_roi_id": getattr(getattr(self, "roi_store", None), "selected_id", None),
        }

    def _restore_roi_session(self, rois, *, selected_id=None) -> None:
        selections = tuple(
            selection for selection in tuple(rois or ()) if isinstance(selection, RoiSelection)
        )
        if selected_id is not None and str(selected_id) not in {
            selection.id for selection in selections
        }:
            selected_id = None
        setter = getattr(self.img_view, "setRoiSelections", None)
        if callable(setter):
            setter(selections, selected_id=selected_id)
        self.roi_store = RoiStore(selections=selections, selected_id=selected_id).replace_all(
            selections
        )
        if selected_id is not None:
            self.roi_store = self.roi_store.select(selected_id)
        if hasattr(self, "inspection_dock"):
            self.inspection_dock.set_rois(self.roi_store.selections)
        self._file_session_roi_refresh_pending = bool(selections)
        self._schedule_file_session_roi_refresh("file-session-restore")

    def _schedule_file_session_roi_refresh(self, reason: str) -> None:
        selections = tuple(getattr(getattr(self, "roi_store", None), "selections", ()) or ())
        if not selections:
            self._file_session_roi_refresh_pending = False
            return
        if reason != "file-session-restore" and not bool(
            getattr(self, "_file_session_roi_refresh_pending", False)
        ):
            return
        schedule_refresh = getattr(self, "_schedule_refresh_inspection_dock", None)
        if callable(schedule_refresh):
            schedule_refresh(reason)
        if reason != "file-session-restore":
            pending_values = getattr(self, "_montage_roi_values_pending", None)
            if not callable(pending_values) or not pending_values():
                self._file_session_roi_refresh_pending = False

    def _apply_viewport_continuity_when_ready(self) -> None:
        tx = self._viewport_continuity_transaction()
        viewport = None if tx is None else tx.viewport
        if viewport is None or (tx.range_applied and tx.released):
            return
        if not self._viewport_continuity_ready():
            return
        view = getattr(getattr(self, "img_view", None), "getView", lambda: None)()
        if view is None:
            return
        view_range = self._validated_viewport_continuity_range(viewport.view_range)
        controller = getattr(self.img_view, "viewport_controller", None)
        if controller is not None:
            try:
                controller.mode = ViewportMode(str(viewport.mode))
            except Exception:
                controller.mode = ViewportMode.USER
            if (
                getattr(getattr(self, "view_state", None), "montage_axis", None) is not None
                and controller.mode == ViewportMode.AUTO_UNTOUCHED
            ):
                controller.mode = ViewportMode.USER
            elif controller.mode == ViewportMode.AUTO_UNTOUCHED and view_range is not None:
                controller.last_auto_view_range = view_range
        if view_range is None:
            if controller is not None:
                controller.fit(view)
            tx.range_applied = True
            tx.released = True
            self._suppress_montage_autofit_revert_message = False
            sync_viewport = getattr(self.img_view, "_sync_vispy_camera_to_view", None)
            if callable(sync_viewport):
                sync_viewport()
            return
        first_apply = not tx.range_applied
        previous_applying = bool(getattr(self.img_view, "_viewport_applying", False))
        self.img_view._viewport_applying = True
        try:
            view.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)
        finally:
            self.img_view._viewport_applying = previous_applying
        sync_viewport = getattr(self.img_view, "_sync_vispy_camera_to_view", None)
        if callable(sync_viewport):
            sync_viewport()
        self._schedule_viewport_continuity_retarget()
        tx.range_applied = True
        if getattr(getattr(self, "view_state", None), "montage_axis", None) is None:
            tx.released = self._viewport_continuity_shape_settled()
        else:
            tx.released = False
        self._suppress_montage_autofit_revert_message = False
        if first_apply and tx.message_enabled and tx.reason == "file-session-restore":
            show_revert_action(
                self,
                "Restored saved view for this file.",
                self._revert_file_view_session_restore,
                timeout=7000,
            )

    def _reapply_viewport_continuity_range_after_layout(self) -> None:
        tx = self._viewport_continuity_transaction()
        viewport = None if tx is None else tx.viewport
        if viewport is None or viewport.view_range is None:
            return
        if not self._viewport_continuity_ready():
            return
        view = getattr(getattr(self, "img_view", None), "getView", lambda: None)()
        if view is None:
            return
        view_range = self._validated_viewport_continuity_range(viewport.view_range)
        if view_range is None:
            return
        previous_applying = bool(getattr(self.img_view, "_viewport_applying", False))
        self.img_view._viewport_applying = True
        try:
            view.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)
        finally:
            self.img_view._viewport_applying = previous_applying
        sync_viewport = getattr(self.img_view, "_sync_vispy_camera_to_view", None)
        if callable(sync_viewport):
            sync_viewport()
        self._schedule_viewport_continuity_retarget()

    def _viewport_continuity_shape_settled(self) -> bool:
        tx = self._viewport_continuity_transaction()
        if tx is None or tx.viewport_shape is None:
            return True
        return bool(tx.shape_settled and self._viewport_continuity_shape_matches(tx.viewport_shape))

    def _complete_viewport_continuity_if_settled(self) -> None:
        tx = self._viewport_continuity_transaction()
        if tx is None or not tx.range_applied or not self._viewport_continuity_shape_settled():
            return
        tx.released = True
        if tx.reason != "file-session-restore":
            self._viewport_continuity = None

    def _validated_viewport_continuity_range(self, view_range):
        normalized = _normalize_saved_view_range(view_range)
        if normalized is None:
            return None
        content_rect = self._viewport_continuity_content_rect()
        if content_rect is None:
            return normalized
        constrained = constrain_view_range(normalized, content_rect)
        return constrained if _view_range_overlaps_content(constrained, content_rect) else None

    def _viewport_continuity_content_rect(self):
        controller = getattr(getattr(self, "img_view", None), "viewport_controller", None)
        content_rect = getattr(controller, "last_display_rect", None)
        if content_rect is not None:
            try:
                x0, y0, x1, y1 = (float(value) for value in content_rect)
                if _finite_values(x0, y0, x1, y1) and abs(x1 - x0) > 0.0 and abs(y1 - y0) > 0.0:
                    return (x0, y0, x1, y1)
            except Exception:
                pass
        frame = getattr(self, "_committed_display_frame", None)
        geometry = getattr(frame, "geometry", None)
        display_shape = getattr(geometry, "display_shape", None)
        try:
            height, width = (int(value) for value in display_shape[:2])
        except Exception:
            return None
        if height < 1 or width < 1:
            return None
        return (0.0, 0.0, float(width), float(height))

    def _current_image_viewport_shape(self) -> tuple[int, int] | None:
        try:
            size = self.img_view.graphicsView.viewport().size()
            height = max(1, int(size.height()))
            width = max(1, int(size.width()))
        except Exception:
            return None
        return (height, width)

    def _viewport_continuity_ready(self) -> bool:
        if getattr(getattr(self, "view_state", None), "montage_axis", None) is None:
            frame = getattr(self, "_committed_display_frame", None)
            if frame is None:
                return False
            is_current = getattr(self, "_is_committed_display_frame_current", None)
            return bool(not callable(is_current) or is_current(frame))
        session = getattr(self, "_frame_session", None)
        if session is None:
            return False
        return getattr(session, "plan", None) is not None

    def _schedule_viewport_continuity_when_ready(self) -> None:
        tx = self._viewport_continuity_transaction()
        if tx is None or tx.viewport is None or (tx.range_applied and tx.released):
            return
        # Timer category: UI cosmetic. Qt event-turn barrier. The callback rechecks committed frame/session
        # readiness before applying the saved view range.
        Qt.QtCore.QTimer.singleShot(0, self, self._apply_viewport_continuity_when_ready)

    def _schedule_viewport_continuity_retarget(self) -> None:
        if getattr(getattr(self, "view_state", None), "montage_axis", None) is None:
            return
        scheduler = getattr(self, "retarget_montage_viewport", None)
        if not callable(scheduler):
            return

        def schedule_once():
            scheduler()

        try:
            # Timer category: UI cosmetic. Qt event-turn barrier. Retargeting follows the restored view range
            # after the ViewBox has accepted it.
            Qt.QtCore.QTimer.singleShot(0, self, schedule_once)
        except Exception:
            scheduler()

    def _revert_file_view_session_restore(self) -> None:
        restore = self._viewport_continuity_transaction()
        if restore is not None and restore.reason != "file-session-restore":
            restore = None
        defaults = None if restore is None else restore.defaults
        if not defaults:
            return
        recipe = defaults["recipe"]
        self.operation_coordinator.load_steps(recipe.steps)
        self._set_document(self.operation_coordinator.document)
        self._set_view_state(recipe.view_state.for_shape(self.data.shape, preserve_flags=True))
        self._apply_display_settings(recipe.display)
        self._restore_roi_session(
            defaults.get("rois", ()), selected_id=defaults.get("selected_roi_id")
        )
        self._begin_viewport_continuity(
            reason="file-session-restore",
            viewport=defaults.get("viewport"),
            profile_visible=bool(getattr(recipe.display, "profile_visible", False)),
            defaults=defaults,
            message_enabled=False,
        )
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
    def scoped_for_application(base: Path) -> Path:
        application_name = str(Qt.QtCore.QCoreApplication.applicationName() or "")
        if application_name and application_name != "ArrayScope" and base.name != application_name:
            return base / application_name
        return base

    try:
        settings_path = Path(Qt.QtCore.QSettings().fileName())
        if str(settings_path):
            return scoped_for_application(settings_path.parent)
    except Exception:
        pass
    location = Qt.QtCore.QStandardPaths.writableLocation(
        Qt.QtCore.QStandardPaths.StandardLocation.AppConfigLocation
    )
    if not location:
        location = Qt.QtCore.QDir.homePath()
    return scoped_for_application(Path(location))


def _normalize_saved_view_range(view_range):
    try:
        normalized = (
            (float(view_range[0][0]), float(view_range[0][1])),
            (float(view_range[1][0]), float(view_range[1][1])),
        )
    except Exception:
        return None
    values = (normalized[0][0], normalized[0][1], normalized[1][0], normalized[1][1])
    if not _finite_values(*values):
        return None
    if normalized[0][0] == normalized[0][1] or normalized[1][0] == normalized[1][1]:
        return None
    return normalized


def _view_range_overlaps_content(view_range, content_rect) -> bool:
    try:
        x_range, y_range = view_range
        x0, y0, x1, y1 = (float(value) for value in content_rect)
    except Exception:
        return False
    vx0, vx1 = sorted((float(x_range[0]), float(x_range[1])))
    vy0, vy1 = sorted((float(y_range[0]), float(y_range[1])))
    cx0, cx1 = sorted((x0, x1))
    cy0, cy1 = sorted((y0, y1))
    return max(vx0, cx0) < min(vx1, cx1) and max(vy0, cy0) < min(vy1, cy1)


def _finite_values(*values) -> bool:
    return all(math.isfinite(float(value)) for value in values)
