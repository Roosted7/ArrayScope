"""Graphics layer ownership for image-view items."""

from __future__ import annotations


Z_IMAGE = 0
Z_TILE_IMAGE = 0
Z_ROI = 40
Z_PROFILE_MARKER = 60
Z_MONTAGE_LOADING_OVERLAY = 70
Z_HUD_GRAPHICS = 90


class ViewLayerOwner:
    """Owns ViewBox item insertion and display z-order policy."""

    def __init__(self, view):
        self.view = view
        self._tile_items: dict[int, object] = {}
        self._roi_items: dict[str, object] = {}
        self._image_item = None
        self._montage_overlay_item = None

    def add_image_item(self, item) -> None:
        self._add_item(item, Z_IMAGE)
        self._image_item = item

    def add_tile_item(self, tile_number: int, item) -> None:
        tile_number = int(tile_number)
        existing = self._tile_items.get(tile_number)
        if existing is not None and existing is not item:
            self.remove_tile_item(tile_number)
        self._add_item(item, Z_TILE_IMAGE)
        self._tile_items[tile_number] = item

    def remove_tile_item(self, tile_number: int) -> None:
        item = self._tile_items.pop(int(tile_number), None)
        if item is not None:
            self._remove_item(item)

    def add_roi_item(self, roi_id: str, item) -> None:
        roi_id = str(roi_id)
        existing = self._roi_items.get(roi_id)
        if existing is not None and existing is not item:
            self.remove_roi_item(roi_id)
        self._add_item(item, Z_ROI)
        self._roi_items[roi_id] = item

    def remove_roi_item(self, roi_id: str) -> None:
        item = self._roi_items.pop(str(roi_id), None)
        if item is not None:
            self._remove_item(item)

    def add_profile_marker_items(self, *items) -> None:
        for item in items:
            if item is not None:
                self._add_item(item, Z_PROFILE_MARKER)

    def set_montage_overlay_item(self, item) -> None:
        if self._montage_overlay_item is not None and self._montage_overlay_item is not item:
            self._remove_item(self._montage_overlay_item)
        self._add_item(item, Z_MONTAGE_LOADING_OVERLAY)
        self._montage_overlay_item = item

    def _add_item(self, item, z_value: int) -> None:
        item.setZValue(float(z_value))
        if item.scene() is None:
            self.view.addItem(item)

    def _remove_item(self, item) -> None:
        try:
            self.view.removeItem(item)
        except Exception:
            pass
