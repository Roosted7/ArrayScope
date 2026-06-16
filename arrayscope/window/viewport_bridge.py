"""Viewport event bridge for render controllers."""

from __future__ import annotations


class ViewportBridge:
    def __init__(self, owner):
        self.owner = owner

    def on_view_range_changed(self) -> None:
        self.owner._update_display_group_title()
        if self.owner.view_state.montage_axis is not None and not getattr(self.owner, "_montage_canvas_commit_active", False):
            self.owner._schedule_montage_viewport_update()


