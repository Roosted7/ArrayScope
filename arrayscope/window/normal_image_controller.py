"""Normal 2D image render controller facade."""

from __future__ import annotations


class NormalImageRenderController:
    def __init__(self, owner):
        self.owner = owner

    def update_view(self, *, force_autolevel: bool = False, defer_side_panels: bool = False):
        return self.owner.update_image_view(force_autolevel=force_autolevel, defer_side_panels=defer_side_panels)

