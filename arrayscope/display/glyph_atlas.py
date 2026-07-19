"""CPU-baked glyph atlas feeding GPU-drawn overlay text (queue row 3 text gap).

Qt is allowed here on purpose: rasterization happens once per new
(font-key, pixel-size, glyph) triple — off the frame path — via
QPainter/QFontMetrics, exactly like any other asset bake.  Per-frame work is
GPU-only: the executor samples the uploaded alpha atlas; a frame with fully
cached glyphs performs zero atlas uploads (the ``glyph_atlas_uploads``
report counter is the oracle).

DPI awareness is the caller's contract: bake at ``pixel_size`` already
multiplied by the target's devicePixelRatio and place quads in physical
pixels — the cache key then distinguishes DPR 1 from DPR 2 rasters by
construction.

Growth is bounded and loud: the square atlas doubles up to ``max_size``;
beyond that every cached glyph is evicted at once with a
``wgpu_glyph_atlas_evicted`` trace and baking restarts from an empty atlas.
A working set that genuinely exceeds the bound therefore thrashes *audibly*
instead of failing silently or unboundedly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arrayscope.app.qt_binding import prefer_pyside6
from arrayscope.core.trace import emit_trace

prefer_pyside6()

from pyqtgraph.Qt import QtCore, QtGui

#: 1px transparent guard band around every cell so edge texels of one glyph
#: never bleed into a neighbour under exact-rect sampling.
_PAD = 1


@dataclass(frozen=True)
class GlyphEntry:
    """One baked glyph cell: atlas texel rect + horizontal advance."""

    x: int
    y: int
    width: int
    height: int
    advance: float


@dataclass(frozen=True)
class GlyphPlacement:
    """One laid-out glyph: entry + top-left pixel offset in the text block."""

    entry: GlyphEntry
    x: float
    y: float


@dataclass(frozen=True)
class TextLayout:
    """Placements plus the block's pixel extent (background/hit sizing)."""

    placements: tuple[GlyphPlacement, ...]
    width: float
    height: float


class GlyphAtlas:
    """Lazily-grown alpha atlas of rasterized glyph cells.

    ``version`` increments whenever atlas *pixels* change (new glyph, growth
    re-pack, eviction); consumers upload the image only when the version they
    last uploaded differs — the zero-per-frame-uploads oracle rests on this.
    """

    def __init__(self, *, initial_size: int = 256, max_size: int = 2048) -> None:
        initial_size = int(initial_size)
        max_size = int(max_size)
        if initial_size <= 0 or max_size < initial_size:
            raise ValueError(
                f"atlas bounds must satisfy 0 < initial <= max, got "
                f"{initial_size}/{max_size}"
            )
        self.max_size = max_size
        self._size = initial_size
        self._image = np.zeros((initial_size, initial_size), np.uint8)
        self._entries: dict[tuple[str, int, str], GlyphEntry] = {}
        self._shelves: list[list[int]] = []  # [y, height, x_cursor]
        self._shelf_cursor_y = 0
        self._version = 1
        self._evictions = 0
        self._metrics_cache: dict[tuple[str, int], QtGui.QFontMetricsF] = {}

    # ---- public surface -----------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    @property
    def version(self) -> int:
        return self._version

    @property
    def evictions(self) -> int:
        return self._evictions

    @property
    def image(self) -> np.ndarray:
        return self._image

    def image_bytes(self) -> bytes:
        return np.ascontiguousarray(self._image).tobytes()

    def glyph(self, font_key: str, pixel_size: int, char: str) -> GlyphEntry:
        """Return the baked cell for one glyph, rasterizing on first use."""

        key = (str(font_key), int(pixel_size), str(char))
        entry = self._entries.get(key)
        if entry is None:
            entry = self._bake(*key)
            self._entries[key] = entry
        return entry

    def line_height(self, font_key: str, pixel_size: int) -> float:
        return float(self._font_metrics(font_key, pixel_size).lineSpacing())

    def layout_text(self, text: str, font_key: str, pixel_size: int) -> TextLayout:
        """Lay out multi-line text as glyph placements in block pixel space."""

        metrics = self._font_metrics(font_key, pixel_size)
        line_step = float(metrics.lineSpacing())
        placements: list[GlyphPlacement] = []
        width = 0.0
        y = 0.0
        lines = str(text).split("\n")
        for line in lines:
            x = 0.0
            for char in line:
                if char.isspace():
                    x += float(metrics.horizontalAdvance(char))
                    continue
                entry = self.glyph(font_key, pixel_size, char)
                placements.append(GlyphPlacement(entry, x, y))
                x += entry.advance
            width = max(width, x)
            y += line_step
        height = max(y, line_step if lines else 0.0)
        return TextLayout(tuple(placements), width, height)

    # ---- baking -------------------------------------------------------------

    def _font_metrics(self, font_key: str, pixel_size: int) -> QtGui.QFontMetricsF:
        cache_key = (str(font_key), int(pixel_size))
        metrics = self._metrics_cache.get(cache_key)
        if metrics is None:
            metrics = QtGui.QFontMetricsF(self._font(*cache_key))
            self._metrics_cache[cache_key] = metrics
        return metrics

    def _font(self, font_key: str, pixel_size: int) -> QtGui.QFont:
        font = QtGui.QFont(font_key)
        if font_key == "monospace":
            font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        font.setPixelSize(max(1, int(pixel_size)))
        return font

    def _bake(self, font_key: str, pixel_size: int, char: str) -> GlyphEntry:
        font = self._font(font_key, pixel_size)
        metrics = self._font_metrics(font_key, pixel_size)
        advance = float(metrics.horizontalAdvance(char))
        cell_w = max(1, int(np.ceil(advance)))
        cell_h = max(1, int(np.ceil(float(metrics.height()))))

        image = QtGui.QImage(cell_w, cell_h, QtGui.QImage.Format.Format_Grayscale8)
        image.fill(0)
        painter = QtGui.QPainter(image)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
            painter.setFont(font)
            painter.setPen(QtGui.QColor(255, 255, 255))
            painter.drawText(QtCore.QPointF(0.0, float(metrics.ascent())), char)
        finally:
            painter.end()
        stride = image.bytesPerLine()
        raw = np.frombuffer(image.constBits(), np.uint8, count=stride * cell_h)
        cell = raw.reshape(cell_h, stride)[:, :cell_w].copy()

        x, y = self._allocate(cell_w, cell_h)
        self._image[y : y + cell_h, x : x + cell_w] = cell
        self._version += 1
        return GlyphEntry(x=x, y=y, width=cell_w, height=cell_h, advance=advance)

    # ---- packing ------------------------------------------------------------

    def _allocate(self, width: int, height: int) -> tuple[int, int]:
        padded_w = width + _PAD
        padded_h = height + _PAD
        if padded_w > self.max_size or padded_h > self.max_size:
            raise ValueError(
                f"glyph cell {width}x{height} exceeds the atlas bound "
                f"{self.max_size}"
            )
        while True:
            for shelf in self._shelves:
                shelf_y, shelf_h, cursor_x = shelf
                if padded_h <= shelf_h and cursor_x + padded_w <= self._size:
                    shelf[2] = cursor_x + padded_w
                    return cursor_x, shelf_y
            if self._shelf_cursor_y + padded_h <= self._size:
                shelf = [self._shelf_cursor_y, padded_h, padded_w]
                self._shelves.append(shelf)
                self._shelf_cursor_y += padded_h
                return 0, shelf[0]
            if self._size < self.max_size:
                self._grow()
                continue
            self._evict_all(width, height)

    def _grow(self) -> None:
        new_size = min(self.max_size, self._size * 2)
        grown = np.zeros((new_size, new_size), np.uint8)
        grown[: self._size, : self._size] = self._image
        self._image = grown
        self._size = new_size
        self._version += 1

    def _evict_all(self, want_w: int, want_h: int) -> None:
        # Bounded and loud: past max_size the whole cache resets; a working
        # set that truly needs more will re-trigger this every rebuild, and
        # the trace makes that thrash visible instead of silent.
        self._evictions += 1
        emit_trace(
            "wgpu_glyph_atlas_evicted",
            atlas_size=int(self._size),
            cached_glyphs=len(self._entries),
            wanted_cell=(int(want_w), int(want_h)),
            evictions=int(self._evictions),
        )
        self._entries.clear()
        self._shelves.clear()
        self._shelf_cursor_y = 0
        self._image = np.zeros((self._size, self._size), np.uint8)
        self._version += 1


__all__ = [
    "GlyphAtlas",
    "GlyphEntry",
    "GlyphPlacement",
    "TextLayout",
]
