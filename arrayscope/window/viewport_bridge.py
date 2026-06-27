"""Viewport event bridge for render controllers."""

from __future__ import annotations

import pyqtgraph.Qt as Qt


class ViewportBridge:
    def __init__(self, owner):
        self.owner = owner

    def on_view_range_changed(self) -> None:
        applying = bool(getattr(getattr(self.owner, "img_view", None), "_viewport_applying", False))
        release_restore_camera = getattr(self.owner, "_release_file_session_restore_camera_lock", None)
        if callable(release_restore_camera) and not applying and _range_change_has_pointer_gesture():
            release_restore_camera()
        note_interaction = getattr(self.owner, "_note_viewport_interaction", None)
        if callable(note_interaction):
            note_interaction("range")
        self.owner._update_display_group_title()
        if self.owner.view_state.montage_axis is not None and not getattr(self.owner, "_montage_presentation_commit_active", False):
            self.owner._schedule_montage_viewport_update()


def _range_change_has_pointer_gesture() -> bool:
    try:
        return bool(Qt.QtWidgets.QApplication.mouseButtons())
    except Exception:
        return False
