"""Montage render controller facade.

This module is the ownership target for progressive montage orchestration. The
current implementation delegates to the window while call sites migrate in
small, testable steps.
"""

from __future__ import annotations


class MontageRenderController:
    def __init__(self, owner):
        self.owner = owner

    def update_view(self, *, force_autolevel: bool = False, defer_side_panels: bool = False):
        return self.owner.update_montage_view(force_autolevel=force_autolevel, defer_side_panels=defer_side_panels)

    def on_tile_done(self, session_id, tile, result) -> None:
        self.owner._on_montage_tile_done(session_id, tile, result)

    def on_tile_error(self, session_id, tile, exc) -> None:
        self.owner._on_montage_tile_error(session_id, tile, exc)

    def on_stage_done(self, session_id, key, value) -> None:
        self.owner._on_montage_stage_done(session_id, key, value)

    def on_stage_error(self, session_id, key, exc) -> None:
        self.owner._on_montage_stage_error(session_id, key, exc)

    def schedule_viewport_update(self) -> None:
        self.owner._schedule_montage_viewport_update()


