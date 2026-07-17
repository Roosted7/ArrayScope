"""Viewport event bridge for render controllers."""

from __future__ import annotations

from time import perf_counter

import pyqtgraph.Qt as Qt


class ViewportBridge:
    def __init__(self, owner):
        self.owner = owner

    def on_view_range_changed(self) -> None:
        started = perf_counter()
        image_view = getattr(self.owner.win, "img_view", None)
        applying = bool(getattr(image_view, "_viewport_applying", False))
        wheel_gesture = bool(getattr(image_view, "_viewport_wheel_range_pending", False))
        if wheel_gesture:
            image_view._viewport_wheel_range_pending = False
        pointer_gesture = bool(_range_change_has_pointer_gesture() or wheel_gesture)
        release_continuity = getattr(self.owner.win, "_release_viewport_continuity", None)
        if callable(release_continuity) and not applying and pointer_gesture:
            release_continuity()
        note_interaction = getattr(self.owner.win, "_note_viewport_interaction", None)
        if callable(note_interaction):
            note_interaction("range-pointer" if pointer_gesture else "range-programmatic")
        # The title depends on display mode, image shape, and viewport size,
        # never on camera range. Calling QGroupBox.setTitle here synchronously
        # entered style/layout work for every wheel and pan signal.
        if _owner_has_tiled_scene(self.owner):
            scheduler = getattr(self.owner, "_schedule_frame_viewport_update", None)
            montage = getattr(self.owner.win.view_state, "montage_axis", None) is not None
            if montage and getattr(self.owner, "_montage_presentation_commit_active", False):
                # The commit guard prevents re-entry, but it must not erase
                # camera intent.  Commit teardown already owns the one replay
                # obligation used by extent-induced range changes; user and
                # profile gestures crossing the same guard join that owner.
                self.owner._frame_viewport_retarget_after_commit = True
            elif montage and not pointer_gesture:
                # Programmatic range replay is a semantic obligation and the
                # V1 boundary reveal requires its visibility update before the
                # caller observes pixels. Only real input bursts are paced.
                self.owner.retarget_montage_viewport()
            elif montage and pointer_gesture:
                interactive_scheduler = getattr(
                    self.owner,
                    "_schedule_interactive_montage_viewport_update",
                    None,
                )
                if callable(interactive_scheduler):
                    interactive_scheduler()
                else:
                    self.owner.retarget_montage_viewport()
            elif callable(scheduler):
                scheduler(delay_ms=0)
            elif montage:
                self.owner.retarget_montage_viewport()
        elapsed_ms = (perf_counter() - started) * 1000.0
        if elapsed_ms >= 4.0:
            record = getattr(self.owner.win, "_record_ui_work", None)
            if callable(record):
                record(
                    "viewport_range_bridge",
                    elapsed_ms,
                    count=1,
                    work_class="viewport_input",
                )


def _range_change_has_pointer_gesture() -> bool:
    try:
        return bool(Qt.QtWidgets.QApplication.mouseButtons())
    except Exception:
        return False


def _owner_has_tiled_scene(owner) -> bool:
    frame = getattr(owner.win, "_committed_display_frame", None)
    scene = getattr(frame, "scene", None)
    value_source = getattr(frame, "value_source", None)
    return scene is not None and value_source is not None and hasattr(value_source, "payloads")
