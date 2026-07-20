"""Per-tile lifecycle truth overlays shared by both image backends."""

from __future__ import annotations

from collections.abc import Callable

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

from pyqtgraph.Qt import QtCore, QtWidgets


def tile_truth_overlay_row_text(row) -> str:
    """Format one lifecycle-owned identity row for its own tile overlay."""

    target_lod = row.get("target_lod") or {}
    acknowledged_lod = row.get("acknowledged_lod") or {}
    mapping = row.get("acknowledged_complex_mapping") or row.get("target_complex_mapping")
    mapping_text = "/".join(str(value) for value in tuple(mapping or ())) or "-"
    state = "DRAW" if bool(row.get("drawable")) else "LOAD"
    return "\n".join(
        (
            f"slot {int(row.get('tile', -1))}  {state}",
            f"src {row.get('target_source')} -> {row.get('acknowledged_source')}",
            f"tex {row.get('target_texture_kind')} -> {row.get('acknowledged_texture_kind')}",
            (
                f"phys {row.get('physical_texture_kind')}/"
                f"{row.get('physical_storage_mode')} map {row.get('physical_mapping_mode')} "
                f"comp {row.get('physical_component_mode')}"
            ),
            (
                f"phys-levels {row.get('physical_levels')} "
                f"shape {row.get('physical_texture_shape')} "
                f"dtype {row.get('physical_texture_dtype')}"
            ),
            (
                f"planes r {_plane_identity_text(row.get('real_plane_identity'))}  "
                f"i {_plane_identity_text(row.get('imag_plane_identity'))}"
            ),
            f"{row.get('target_channel')}  {mapping_text}",
            f"lod {target_lod.get('level')} -> {acknowledged_lod.get('level')}",
            f"sem {row.get('target_semantic_generation')} -> {row.get('acknowledged_semantic_generation')}",
            f"levels {row.get('levels_generation')}",
        )
    )


def _plane_identity_text(value) -> str:
    if not value:
        return "-"
    if not isinstance(value, dict):
        return str(value)
    pointer = value.get("pointer")
    pointer_text = "?" if pointer is None else hex(int(pointer))
    shape = "x".join(str(int(size)) for size in tuple(value.get("shape") or ())) or "?"
    return f"{pointer_text}:{shape}:{value.get('dtype') or '?'}"


def tile_truth_overlay_text(rows) -> str:
    """Return all visible per-tile texts for diagnostics and tests."""

    return "\n\n".join(tile_truth_overlay_row_text(row) for row in tuple(rows or ()))


class TileTruthOverlayLayer:
    """Pool and position one non-interactive identity label per visible tile."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        map_tile_rect: Callable[[tuple[float, float, float, float]], QtCore.QRect],
    ) -> None:
        self.parent = parent
        self._map_tile_rect = map_tile_rect
        self.rows: tuple[dict[str, object], ...] = ()
        self.labels: list[QtWidgets.QLabel] = []

    def set_rows(self, rows) -> None:
        self.rows = tuple(dict(row) for row in tuple(rows or ()))
        while len(self.labels) < len(self.rows):
            self.labels.append(self._make_label())
        for index, label in enumerate(self.labels):
            if index >= len(self.rows):
                label.hide()
                continue
            row = self.rows[index]
            label.setText(tile_truth_overlay_row_text(row))
            label.setToolTip(
                f"target: {row.get('target_identity')}\n"
                f"acknowledged: {row.get('acknowledged_identity')}"
            )
            label.setProperty("truthState", "draw" if bool(row.get("drawable")) else "load")
            label.setStyleSheet(_label_style(bool(row.get("drawable"))))
        self.reposition()

    def reposition(self) -> None:
        parent_rect = self.parent.rect()
        for label, row in zip(self.labels, self.rows, strict=False):
            tile_rect = row.get("tile_rect")
            if tile_rect is None:
                label.hide()
                continue
            screen_rect = self._map_tile_rect(tuple(float(value) for value in tile_rect))
            visible_rect = screen_rect.intersected(parent_rect)
            if visible_rect.width() < 16 or visible_rect.height() < 12:
                label.hide()
                continue
            label.adjustSize()
            label.setGeometry(
                visible_rect.left() + 2,
                visible_rect.top() + 2,
                min(label.sizeHint().width(), max(1, visible_rect.width() - 4)),
                min(label.sizeHint().height(), max(1, visible_rect.height() - 4)),
            )
            label.show()
            label.raise_()

    def clear(self) -> None:
        self.set_rows(())

    def visible_text(self) -> str:
        visible_rows = (
            row for label, row in zip(self.labels, self.rows, strict=False) if not label.isHidden()
        )
        return tile_truth_overlay_text(visible_rows)

    def _make_label(self) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(self.parent)
        label.setObjectName("TileTruthOverlay")
        label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        label.hide()
        return label


def _label_style(drawable: bool) -> str:
    border = "#22d3ee" if drawable else "#f59e0b"
    color = "#a5f3fc" if drawable else "#fde68a"
    return (
        "QLabel#TileTruthOverlay {"
        f"background: rgba(8, 18, 24, 210); color: {color}; border: 1px solid {border};"
        "border-radius: 3px; padding: 3px; font-family: monospace; font-size: 9px; }"
    )


__all__ = [
    "TileTruthOverlayLayer",
    "tile_truth_overlay_row_text",
    "tile_truth_overlay_text",
]
