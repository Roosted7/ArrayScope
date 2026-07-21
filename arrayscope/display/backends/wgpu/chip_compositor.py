"""Rasterize floating Qt chips so the screen path can draw them in-frame.

In the ``screen`` present path the swapchain lives on its own Wayland
subsurface stacked above the window, so any Qt pixel behind it is invisible
— including the floating chips that must appear *over* the image (first-run
hints, the evaluation indicator, the pixel HUD, the ROI info panel).

Qt cannot put those chips in front of that subsurface:

* promoting a chip to a native *child* window gives it no ARGB visual
  (``format().alphaBufferSize() == 0``), so its translucent rounded
  stylesheet renders as a flat opaque box — measured against the bitmap
  reference on 2026-07-21; and
* the swapchain subsurface cannot be restacked below the window, because
  ``QWindow.lower()`` emits no ``wl_subsurface.place_below`` at all.

So the chips are rasterized with ``QWidget.grab()`` and composited *inside*
the wgpu frame.  The pixels are Qt's own, produced by the same painter that
draws them on every other backend, which is what makes the result match the
bitmap path rather than merely approximate it.

The chips stay ordinary Qt widgets and keep painting normally.  That is
deliberate: a chip may extend past the canvas (the first-run hints chip
overlaps the histogram), and the swapchain only covers the canvas, so Qt
still draws the uncovered part.  Both halves come from one widget
rendering, so the seam is continuous.
"""

from __future__ import annotations

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

#: Events after which a chip's PIXELS may differ, so it must be re-rendered.
_REPAINT_EVENTS = frozenset(
    {
        QtCore.QEvent.Type.Paint,
        QtCore.QEvent.Type.Resize,
        QtCore.QEvent.Type.Show,
        QtCore.QEvent.Type.Hide,
        QtCore.QEvent.Type.LanguageChange,
    }
)

#: Events after which only a chip's POSITION may differ.  Re-rendering for
#: these is pure waste, and it is the common case: the hover readout emits a
#: Move for every pointer sample, so re-rasterizing here made the chip lag
#: the cursor.
_MOVE_EVENTS = frozenset({QtCore.QEvent.Type.Move})


class ChipPlacement:
    """One chip's atlas slice and where it belongs on the canvas."""

    __slots__ = ("offset", "size", "uv_rect")

    def __init__(self, offset, size, uv_rect):
        self.offset = offset  # (x, y) physical px from the canvas top-left
        self.size = size  # (w, h) physical px
        self.uv_rect = uv_rect  # (u0, v0, u1, v1) into the atlas

    def __hash__(self):
        return hash((self.offset, self.size, self.uv_rect))

    def __eq__(self, other):
        return (
            isinstance(other, ChipPlacement)
            and self.offset == other.offset
            and self.size == other.size
            and self.uv_rect == other.uv_rect
        )


class FloatingChipCompositor(QtCore.QObject):
    """Tracks floating chips and bakes them into one RGBA atlas."""

    def __init__(self, canvas_provider, on_invalidate=None):
        super().__init__()
        self._canvas_provider = canvas_provider
        # Called when a chip first goes stale, so the view can schedule the
        # frame that will actually show the new pixels.  Without it a chip
        # only reaches the screen when something ELSE happens to draw:
        # dragging a ROI or moving the cursor left the chip frozen until a
        # pan, while hovering a ROI worked purely because the hover state
        # change re-submitted overlay geometry and drew as a side effect.
        self._on_invalidate = on_invalidate
        self._widgets: list = []
        self._repaint_dirty: set = set()
        self._repaint_all = True
        self._layout_dirty = True
        self._baking = False
        self._rasters: dict = {}
        self._version = 0
        self._atlas: tuple[int, int, bytes] | None = None
        self._placements: tuple[ChipPlacement, ...] = ()

    # ---- registration --------------------------------------------------------

    def register(self, widget) -> None:
        """Start compositing ``widget``; idempotent."""

        if widget is None or any(existing is widget for existing in self._widgets):
            return
        self._widgets.append(widget)
        self._watch_subtree(widget)
        widget.destroyed.connect(lambda *_a, w=widget: self._forget(widget=w))
        self.invalidate()

    def _watch_subtree(self, widget) -> None:
        """Watch ``widget`` AND its descendants for repaints.

        The chips are composite widgets: the hover readout and the ROI panel
        put their text in child ``QLabel``s.  A child repainting does not
        deliver a Paint event to its parent, so watching only the registered
        widget left the atlas holding whatever the chip looked like when it
        was first baked — the hover chip photographed EMPTY because it was
        rasterized before its labels had text.
        """

        widget.installEventFilter(self)
        for child in widget.findChildren(QtWidgets.QWidget):
            child.installEventFilter(self)

    def _forget(self, *, widget) -> None:
        self._widgets = [existing for existing in self._widgets if existing is not widget]
        self.invalidate()

    def eventFilter(self, obj, event):
        # Rasterizing repaints the chip, which would re-dirty us from inside
        # our own bake and spin forever.  Invalidations raised while baking
        # are ours, by construction, so drop them.
        if self._baking:
            return False
        event_type = event.type()
        if event_type == QtCore.QEvent.Type.ChildAdded:
            # A chip that grows a new child (a row appended to the ROI
            # panel) must have that child watched too, or its repaints go
            # unseen exactly like the label-text case above.
            child = event.child()
            if child is not None and child.isWidgetType():
                child.installEventFilter(self)
            self.invalidate(self._owner_of(obj))
        elif event_type in _REPAINT_EVENTS:
            self.invalidate(self._owner_of(obj))
        elif event_type in _MOVE_EVENTS:
            self.invalidate(self._owner_of(obj), pixels=False)
        return False

    def _owner_of(self, obj):
        """The registered chip ``obj`` belongs to, or ``None`` if unknown."""

        widget = obj
        while widget is not None:
            if any(existing is widget for existing in self._widgets):
                return widget
            widget = widget.parentWidget() if widget.isWidgetType() else None
        return None

    def invalidate(self, widget=None, *, pixels: bool = True) -> None:
        """Mark work pending.  ``pixels=False`` means "it only moved"."""

        self._layout_dirty = True
        if pixels:
            if widget is None:
                self._repaint_all = True
            else:
                self._repaint_dirty.add(widget)
        # Always ask, and let the view coalesce: it already drops a request
        # while a draw is pending.  Suppressing on the clean -> dirty edge
        # here looked like a cheap optimisation but silently skipped the
        # FIRST request of all — the compositor starts dirty, so a freshly
        # registered chip never asked for the frame that would show it.
        if self._on_invalidate is not None:
            self._on_invalidate()

    @property
    def is_dirty(self) -> bool:
        return bool(self._layout_dirty or self._repaint_all or self._repaint_dirty)

    # ---- baking --------------------------------------------------------------

    @property
    def version(self) -> int:
        return self._version

    @property
    def atlas(self) -> tuple[int, int, bytes] | None:
        return self._atlas

    @property
    def placements(self) -> tuple[ChipPlacement, ...]:
        return self._placements

    def rebuild_if_needed(self) -> bool:
        """Re-bake what changed.  True when the ATLAS revision moved.

        Returning False still leaves ``placements`` current: a chip that only
        moved needs new quad offsets but the very same pixels, so the caller
        rewrites the (tiny) overlay buffer and skips the atlas upload
        entirely.  Re-rendering and re-uploading on every pointer sample is
        what made the hover chip lag the cursor.
        """

        if not self.is_dirty:
            return False
        canvas = self._canvas_provider()
        if canvas is None:
            return False
        repaint_all = self._repaint_all
        repaint = self._repaint_dirty
        self._repaint_all = False
        self._repaint_dirty = set()
        self._layout_dirty = False

        self._baking = True
        try:
            rasters = []
            for widget in tuple(self._widgets):
                geometry = self._geometry_for(widget, canvas)
                if geometry is None:
                    self._rasters.pop(widget, None)
                    continue
                offset, size = geometry
                cached = self._rasters.get(widget)
                stale = (
                    repaint_all
                    or widget in repaint
                    or cached is None
                    or cached[0] != size  # a resize changes the pixels too
                )
                if stale:
                    image = self._render(widget, size)
                    if image is None:
                        self._rasters.pop(widget, None)
                        continue
                    self._rasters[widget] = (size, image)
                else:
                    image = cached[1]
                rasters.append((offset, size, image))
        finally:
            self._baking = False

        atlas, placements = self._pack(rasters)
        atlas_changed = atlas != self._atlas
        self._placements = placements
        if not atlas_changed:
            return False
        self._atlas = atlas
        self._version += 1
        return True

    def _geometry_for(self, widget, canvas):
        """``(offset, size)`` in physical pixels, or ``None`` if it shows nothing.

        Cheap: no rasterizing.  This is the only work a pointer-driven move
        needs, so it stays out of the expensive path.
        """

        try:
            if not widget.isVisible():
                return None
            width = int(widget.width())
            height = int(widget.height())
            if width < 1 or height < 1:
                return None
            # Canvas-relative placement, via global coordinates so it works
            # whatever the chip is parented to.
            top_left = canvas.mapFromGlobal(widget.mapToGlobal(QtCore.QPoint(0, 0)))
            ratio = float(canvas.devicePixelRatio() or 1.0)
        except RuntimeError:  # C++ side already deleted
            return None
        size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        return (float(top_left.x()) * ratio, float(top_left.y()) * ratio), size

    def _render(self, widget, size):
        """Rasterize one chip to straight-RGBA at ``size`` physical pixels."""

        physical_width, physical_height = size
        try:
            ratio = float(widget.devicePixelRatio() or 1.0)
            # The paint target must be ARGB32_PREMULTIPLIED — Qt's raster
            # engine composites natively in premultiplied alpha, and painting
            # a rounded stylesheet straight into a non-premultiplied format
            # loses the corners' partial coverage.  Convert to straight RGBA
            # afterwards, because the overlay pipeline blends
            # src-alpha/one-minus-src-alpha.  NOT ``grab()``: that renders
            # into an opaque QPixmap pre-filled with the palette background.
            image = QtGui.QImage(
                physical_width,
                physical_height,
                QtGui.QImage.Format.Format_ARGB32_Premultiplied,
            )
            image.setDevicePixelRatio(ratio)
            image.fill(QtCore.Qt.GlobalColor.transparent)
            # DrawChildren WITHOUT DrawWindowBackground: these chips carry
            # ``WA_StyledBackground``, so their stylesheet background is
            # painted by their own paint event; asking for the window
            # background as well fills an extra SQUARE rect underneath it,
            # which squares off the border-radius and composites the
            # translucent background twice (1-(1-0.843)^2 = 0.975, so alpha
            # 215 rendered as ~249 and looked opaque).
            flags = QtWidgets.QWidget.RenderFlag.DrawChildren
            if not widget.testAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground):
                flags |= QtWidgets.QWidget.RenderFlag.DrawWindowBackground
            widget.render(image, QtCore.QPoint(), QtGui.QRegion(), flags)
            return image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        except RuntimeError:  # C++ side already deleted
            return None

    def _pack(self, rasters):
        """Stack chips vertically into one RGBA buffer.

        A shelf allocator would be overkill: there are at most a handful of
        chips and they are re-baked only when one of them actually changes.
        """

        if not rasters:
            return None, ()
        atlas_width = max(size[0] for _offset, size, _image in rasters)
        atlas_height = sum(size[1] for _offset, size, _image in rasters)
        buffer = bytearray(atlas_width * atlas_height * 4)
        placements = []
        y_cursor = 0
        for offset, (width, height), image in rasters:
            bits = image.constBits()
            source = bytes(bits)[: image.sizeInBytes()]
            stride = int(image.bytesPerLine())
            for row in range(height):
                start = row * stride
                destination = ((y_cursor + row) * atlas_width) * 4
                buffer[destination : destination + width * 4] = source[start : start + width * 4]
            placements.append(
                ChipPlacement(
                    offset,
                    (float(width), float(height)),
                    (
                        0.0,
                        y_cursor / atlas_height,
                        width / atlas_width,
                        (y_cursor + height) / atlas_height,
                    ),
                )
            )
            y_cursor += height
        return (atlas_width, atlas_height, bytes(buffer)), tuple(placements)
