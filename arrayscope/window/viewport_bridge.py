"""Viewport event bridge for render controllers."""

from __future__ import annotations

import pyqtgraph.Qt as Qt


class ViewportBridge:
    def __init__(self, owner):
        self.owner = owner

    def on_view_range_changed(self) -> None:
        applying = bool(getattr(getattr(self.owner.win, "img_view", None), "_viewport_applying", False))
        pointer_gesture = bool(_range_change_has_pointer_gesture())
        release_continuity = getattr(self.owner.win, "_release_viewport_continuity", None)
        if callable(release_continuity) and not applying and pointer_gesture:
            release_continuity()
        note_interaction = getattr(self.owner.win, "_note_viewport_interaction", None)
        if callable(note_interaction):
            note_interaction("range-pointer" if pointer_gesture else "range-programmatic")
        self.owner._update_display_group_title()
        if not getattr(self.owner, "_montage_presentation_commit_active", False) and _owner_has_tiled_scene(self.owner):
            scheduler = getattr(self.owner, "_schedule_frame_viewport_update", None)
            if callable(scheduler):
                scheduler()
            elif self.owner.win.view_state.montage_axis is not None:
                self.owner.retarget_montage_viewport()


def _range_change_has_pointer_gesture() -> bool:
    try:
        return bool(Qt.QtWidgets.QApplication.mouseButtons())
    except Exception:
        return False


def _owner_has_tiled_scene(owner) -> bool:
    if getattr(getattr(owner.win, "view_state", None), "montage_axis", None) is not None:
        session = getattr(owner, "_montage_session", None)
        if session is not None and (
            bool(getattr(session, "display_committed", False))
            or bool(getattr(session, "display_tile_payloads", None))
            or bool(getattr(getattr(session, "tile_presentation_state", None), "payloads", None))
            or bool(getattr(getattr(session, "lifecycle", None), "presented_tiles", None))
        ):
            return True
    frame = getattr(owner.win, "_committed_display_frame", None)
    scene = getattr(frame, "scene", None)
    value_source = getattr(frame, "value_source", None)
    return scene is not None and value_source is not None and hasattr(value_source, "payloads")
