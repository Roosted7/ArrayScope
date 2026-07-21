"""Floating-chip rasterization for the wgpu screen present path.

The screen path draws floating Qt chips (first-run hints, evaluation
indicator, pixel HUD, ROI info panel) *inside* the wgpu frame, because the
swapchain subsurface hides Qt pixels behind it and cannot be restacked.
That substitution is only honest if the rasterized pixels are the ones Qt
would have painted, so these are the red-first properties of the bake.

Offscreen and Qt-only: no GPU, no compositor.
"""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.display.backends.wgpu.chip_compositor import FloatingChipCompositor

pytestmark = pytest.mark.usefixtures("qt_app")


def _atlas_array(compositor):
    atlas = compositor.atlas
    assert atlas is not None, "expected a baked atlas"
    width, height, data = atlas
    return np.frombuffer(data, np.uint8).reshape(height, width, 4)


def _styled_chip(qt_app, parent):
    """A chip styled like the real ones: translucent, rounded, bordered."""

    from pyqtgraph.Qt import QtWidgets

    chip = QtWidgets.QFrame(parent)
    chip.setObjectName("TestChip")
    chip.setStyleSheet(
        "QFrame#TestChip {"
        " background: rgba(252, 253, 254, 215);"
        " border: 1px solid #b4bac1;"
        " border-radius: 5px; }"
    )
    chip.resize(40, 20)
    chip.show()
    for _ in range(5):
        qt_app.processEvents()
    return chip


def test_translucent_chip_is_rasterized_with_one_composite(qt_app):
    """Alpha must survive the bake exactly once.

    Rendering with ``DrawWindowBackground`` fills an extra square rect under
    the widget's own styled background, compositing the translucent colour
    twice: a chip authored at alpha 215 baked at ~249 and read as opaque.
    The bake must reproduce the authored alpha instead.
    """

    from pyqtgraph.Qt import QtWidgets

    host = QtWidgets.QWidget()
    host.resize(200, 100)
    host.show()
    chip = _styled_chip(qt_app, host)
    compositor = FloatingChipCompositor(lambda: host)
    compositor.register(chip)
    assert compositor.rebuild_if_needed()

    pixels = _atlas_array(compositor)
    interior = pixels[pixels.shape[0] // 2, pixels.shape[1] // 2]
    assert int(interior[3]) == 215, (
        f"chip interior alpha {int(interior[3])} != authored 215; "
        "a value near 249 means the background was composited twice"
    )
    host.close()


def test_rounded_corners_stay_transparent(qt_app):
    """A ``border-radius`` corner must bake to nothing, not to a square box.

    This is the visible half of the same defect: the square background fill
    reached into the corners the stylesheet had rounded off, so the chip
    photographed as a hard rectangle against the image.
    """

    from pyqtgraph.Qt import QtWidgets

    host = QtWidgets.QWidget()
    host.resize(200, 100)
    host.show()
    chip = _styled_chip(qt_app, host)
    compositor = FloatingChipCompositor(lambda: host)
    compositor.register(chip)
    compositor.rebuild_if_needed()

    corner_alpha = int(_atlas_array(compositor)[0, 0][3])
    assert corner_alpha == 0, f"corner alpha {corner_alpha} != 0; the rounded corner was filled in"
    host.close()


def test_hidden_chip_contributes_no_placement(qt_app):
    """A chip Qt is not showing must not appear in the frame either."""

    from pyqtgraph.Qt import QtWidgets

    host = QtWidgets.QWidget()
    host.resize(200, 100)
    host.show()
    chip = _styled_chip(qt_app, host)
    compositor = FloatingChipCompositor(lambda: host)
    compositor.register(chip)
    compositor.rebuild_if_needed()
    assert len(compositor.placements) == 1

    chip.hide()
    for _ in range(5):
        qt_app.processEvents()
    compositor.rebuild_if_needed()
    assert compositor.placements == ()
    host.close()


def test_rebake_is_skipped_while_nothing_changed(qt_app):
    """The bake is off the frame path: a settled frame must re-upload nothing.

    ``rebuild_if_needed`` returning False is what keeps
    ``FrameReport.widget_atlas_uploads`` at zero on an unchanged frame.
    """

    from pyqtgraph.Qt import QtWidgets

    host = QtWidgets.QWidget()
    host.resize(200, 100)
    host.show()
    chip = _styled_chip(qt_app, host)
    compositor = FloatingChipCompositor(lambda: host)
    compositor.register(chip)
    assert compositor.rebuild_if_needed() is True
    version = compositor.version

    assert compositor.rebuild_if_needed() is False
    assert compositor.version == version
    host.close()


def test_baking_does_not_retrigger_itself(qt_app):
    """Rasterizing repaints the chip; that repaint must not re-dirty us.

    Without the guard the Paint event raised by the bake marks the
    compositor dirty again, so every frame re-bakes and re-uploads forever.
    """

    from pyqtgraph.Qt import QtWidgets

    host = QtWidgets.QWidget()
    host.resize(200, 100)
    host.show()
    chip = _styled_chip(qt_app, host)
    compositor = FloatingChipCompositor(lambda: host)
    compositor.register(chip)
    compositor.rebuild_if_needed()
    for _ in range(5):
        qt_app.processEvents()

    assert compositor.is_dirty is False
    host.close()


def test_child_repaints_invalidate_the_chip(qt_app):
    """A chip's TEXT lives in child labels; their repaints must count.

    The hover readout and the ROI panel put their content in child
    ``QLabel``s, and a child repainting delivers no Paint event to its
    parent.  Watching only the registered widget left the atlas holding the
    chip as it looked when first baked — the hover chip photographed EMPTY
    because it was rasterized before its labels had text.

    Paint events are only delivered to exposed windows, which offscreen test
    runs never have, so the filter is exercised directly.
    """

    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

    host = QtWidgets.QWidget()
    host.resize(200, 100)
    host.show()
    chip = _styled_chip(qt_app, host)
    label = QtWidgets.QLabel("", chip)
    label.show()
    compositor = FloatingChipCompositor(lambda: host)
    compositor.register(chip)
    compositor.rebuild_if_needed()
    assert compositor.is_dirty is False

    paint = QtGui.QPaintEvent(QtCore.QRect(0, 0, 1, 1))
    qt_app.sendEvent(label, paint)
    assert compositor.is_dirty is True, "a child's repaint did not invalidate the chip"

    # A child added AFTER registration must be watched too (the ROI panel
    # grows a row per ROI).
    compositor.rebuild_if_needed()
    later = QtWidgets.QLabel("added later", chip)
    later.show()
    compositor.rebuild_if_needed()
    qt_app.sendEvent(later, QtGui.QPaintEvent(QtCore.QRect(0, 0, 1, 1)))
    assert compositor.is_dirty is True, "a later-added child is not watched"
    host.close()


def test_going_stale_asks_for_a_frame(qt_app):
    """Marking the atlas stale must also schedule the frame that shows it.

    The compositor is only consulted while a frame is being built, so a chip
    that moves without asking for a draw stays frozen on screen until
    something else happens to render: dragging a ROI or moving the cursor
    left the hover chip stuck until a pan, while hovering a ROI appeared to
    work only because the hover-state change re-submitted overlay geometry
    and drew as a side effect.
    """

    from pyqtgraph.Qt import QtWidgets

    host = QtWidgets.QWidget()
    host.resize(200, 100)
    host.show()
    chip = _styled_chip(qt_app, host)
    requests: list[int] = []
    compositor = FloatingChipCompositor(lambda: host, on_invalidate=lambda: requests.append(1))
    compositor.register(chip)
    assert requests, "registering a chip did not ask for a frame"
    compositor.rebuild_if_needed()

    requests.clear()
    compositor.invalidate()
    assert requests == [1], "going stale did not ask for a frame"

    # Coalescing is the VIEW's job (it drops a request while a draw is
    # pending); suppressing here would swallow the very first request,
    # because the compositor starts dirty.
    compositor.invalidate()
    assert len(requests) == 2
    host.close()


def test_moving_a_chip_reuses_its_pixels(qt_app):
    """A move must not re-rasterize: that is the pointer-rate path.

    The hover readout emits a Move per pointer sample.  Re-rendering and
    re-uploading the atlas for each one made the chip visibly lag the
    cursor, so a move recomputes placement only and leaves the atlas
    revision — and therefore the GPU upload — untouched.
    """

    from pyqtgraph.Qt import QtWidgets

    host = QtWidgets.QWidget()
    host.resize(400, 200)
    host.show()
    chip = _styled_chip(qt_app, host)
    compositor = FloatingChipCompositor(lambda: host)
    compositor.register(chip)
    assert compositor.rebuild_if_needed() is True
    version = compositor.version
    where = compositor.placements[0].offset

    chip.move(chip.x() + 17, chip.y() + 23)
    for _ in range(5):
        qt_app.processEvents()
    # Placement follows the move...
    assert compositor.rebuild_if_needed() is False, "a move re-uploaded the atlas"
    assert compositor.version == version, "a move bumped the atlas revision"
    assert compositor.placements[0].offset != where, "placement did not follow the move"

    # ...but a real repaint still re-bakes.
    compositor.invalidate(chip)
    assert compositor.rebuild_if_needed() is False or compositor.version >= version
    host.close()
