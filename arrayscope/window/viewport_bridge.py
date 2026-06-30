"""Viewport event bridge for render controllers."""

from __future__ import annotations

import pyqtgraph.Qt as Qt


class ViewportBridge:
    def __init__(self, owner):
        self.owner = owner

    def on_view_range_changed(self) -> None:
        applying = bool(getattr(getattr(self.owner, "img_view", None), "_viewport_applying", False))
        pointer_gesture = bool(_range_change_has_pointer_gesture())
        release_continuity = getattr(self.owner, "_release_viewport_continuity", None)
        if callable(release_continuity) and not applying and pointer_gesture:
            release_continuity()
        note_interaction = getattr(self.owner, "_note_viewport_interaction", None)
        if callable(note_interaction):
            note_interaction("range-pointer" if pointer_gesture else "range-programmatic")
        self.owner._update_display_group_title()
        if not getattr(self.owner, "_montage_presentation_commit_active", False) and _owner_has_tiled_scene(self.owner):
            scheduler = getattr(self.owner, "_schedule_frame_viewport_update", None)
            if callable(scheduler):
                scheduler()
            elif self.owner.view_state.montage_axis is not None:
                self.owner._schedule_montage_viewport_update()


def _range_change_has_pointer_gesture() -> bool:
    try:
        return bool(Qt.QtWidgets.QApplication.mouseButtons())
    except Exception:
        return False


def _owner_has_tiled_scene(owner) -> bool:
    frame = getattr(owner, "_committed_display_frame", None)
    scene = getattr(frame, "scene", None)
    value_source = getattr(frame, "value_source", None)
    return scene is not None and value_source is not None and hasattr(value_source, "payloads")
