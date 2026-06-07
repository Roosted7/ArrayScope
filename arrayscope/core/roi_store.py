"""Pure ROI selection store."""

from __future__ import annotations

from dataclasses import dataclass, replace

from arrayscope.core.roi import RoiSelection


DEFAULT_ROI_COLORS = (
    (230, 60, 30),
    (40, 120, 210),
    (40, 150, 90),
    (180, 90, 210),
    (220, 150, 40),
    (90, 140, 180),
)


@dataclass(frozen=True)
class RoiStore:
    selections: tuple[RoiSelection, ...] = ()
    selected_id: str | None = None

    def upsert(self, selection: RoiSelection) -> "RoiStore":
        selection = self._with_assigned_color(selection)
        selections = list(self.selections)
        for index, existing in enumerate(selections):
            if existing.id == selection.id:
                selections[index] = selection
                break
        else:
            selections.append(selection)
        selected = self.selected_id if self.selected_id in {item.id for item in selections} else selection.id
        return RoiStore(tuple(selections), selected)

    def replace_all(self, selections) -> "RoiStore":
        result = RoiStore(selected_id=self.selected_id)
        for selection in selections:
            result = result.upsert(selection)
        selected = result.selected_id if result.selected_id in {item.id for item in result.selections} else None
        return RoiStore(result.selections, selected)

    def remove(self, roi_id: str) -> "RoiStore":
        roi_id = str(roi_id)
        selections = tuple(selection for selection in self.selections if selection.id != roi_id)
        selected = self.selected_id if self.selected_id != roi_id else (selections[0].id if selections else None)
        return RoiStore(selections, selected)

    def clear(self) -> "RoiStore":
        return RoiStore()

    def select(self, roi_id: str | None) -> "RoiStore":
        if roi_id is None:
            return RoiStore(self.selections, None)
        roi_id = str(roi_id)
        if any(selection.id == roi_id for selection in self.selections):
            return RoiStore(self.selections, roi_id)
        return self

    def set_enabled(self, roi_id: str, enabled: bool) -> "RoiStore":
        selections = tuple(
            replace(selection, enabled=bool(enabled)) if selection.id == str(roi_id) else selection
            for selection in self.selections
        )
        return RoiStore(selections, self.selected_id)

    def get(self, roi_id: str) -> RoiSelection | None:
        for selection in self.selections:
            if selection.id == str(roi_id):
                return selection
        return None

    def _with_assigned_color(self, selection: RoiSelection) -> RoiSelection:
        used = {item.color for item in self.selections if item.id != selection.id}
        if selection.color not in used:
            return selection
        for color in DEFAULT_ROI_COLORS:
            if color not in used:
                return replace(selection, color=color)
        return selection
