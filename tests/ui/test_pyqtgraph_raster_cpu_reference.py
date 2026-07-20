"""Default-ring smoke for the PyQtGraph Qt-raster CPU-reference oracle.

Ring 1 (default offscreen suite). The smoke reads the painted QGraphicsView
viewport, compares every required scalar-tile interior with
``cpu_display_rgba`` under the active levels/LUT, and proves the oracle can
fail by corrupting the physical ImageItem levels while semantic owners stay
unchanged. It is not rendering acceptance; the real-display fault audit is
``tests/gpu_interaction/test_pyqtgraph_raster_cpu_reference.py`` (ring 4).

Pins docs/testing/README.md law 5 and ground rule 10: physical-pixel gates
must be non-vacuous, fault-proven, and present for both rendering backends.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pytest

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.oracles.framebuffer_reference import (
    assert_qt_raster_matches_cpu_reference,
)
from tests.ui.helpers import (
    frame_session_settled,
    make_backend_window,
    restore_default_backend,
    use_pyqtgraph_backend,
)

TILE = 32
GRID = 3
COUNT = GRID * GRID


def _gradient_montage_data() -> np.ndarray:
    yy, xx = np.mgrid[0:TILE, 0:TILE].astype(np.float32)
    gradient = (yy + xx) * (8.0 / (2.0 * (TILE - 1)))
    frames = np.arange(COUNT, dtype=np.float32)[:, None, None] * 20.0
    return (frames + gradient[None]).transpose(1, 2, 0).copy()


def _settled(win) -> bool:
    return bool(
        getattr(win, "_committed_display_frame", None) is not None
        and frame_session_settled(win)
        and not win.img_view.presentationDrawPending()
    )


def _visible_tile_states(win):
    layer = win.img_view._montage_tile_layer
    states = {
        int(number): state
        for number, state in layer.states.items()
        if state.visible and state.item.isVisible()
    }
    required = {int(number) for number in win.renderer._frame_session.required_tile_numbers()}
    assert set(states) == required, (
        f"physical PyQtGraph tile set drifted: required={sorted(required)}, "
        f"visible={sorted(states)}"
    )
    return states


def test_settled_montage_matches_cpu_reference_and_fails_on_wrong_levels(qtbot):
    settings = use_pyqtgraph_backend()
    win = make_backend_window(
        qtbot,
        _gradient_montage_data(),
        backend="pyqtgraph",
    )
    try:
        win.resize(720, 600)
        win.show()
        qtbot.waitExposed(win)
        win._set_view_state(win.view_state.with_montage_axis(2, text=":"))
        win.render(reason="qt-raster-cpu-reference-smoke")
        qtbot.waitUntil(lambda: _settled(win), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

        session = win.renderer._frame_session
        required = {int(number) for number in session.required_tile_numbers()}
        assert len(required) == COUNT, f"smoke regime drifted: required tiles {sorted(required)}"
        for number in sorted(required):
            payload = session.display_tile_payloads[number]
            level = 0 if payload.lod is None else int(payload.lod.level)
            assert level == 0, (
                f"tile {number} presented LOD level {level}; the smoke pins "
                "the native-resolution scalar regime"
            )

        # Exercise the Qt LUT path, not only grayscale levels. The oracle
        # resolves this LUT from the semantic image-view owner while the
        # viewport readback observes ImageItem's rasterized QImage.
        win.img_view.setColorMap(
            pg.ColorMap(
                (0.0, 0.45, 1.0),
                np.asarray(
                    (
                        (0, 8, 40, 255),
                        (30, 190, 90, 255),
                        (255, 244, 210, 255),
                    ),
                    dtype=np.uint8,
                ),
            )
        )
        win.img_view.graphicsView.viewport().update()
        win.img_view.graphicsView.viewport().repaint()

        report = assert_qt_raster_matches_cpu_reference(win)
        assert {tile.tile_number for tile in report.tiles} == required
        assert all(tile.samples >= report.min_samples_per_tile for tile in report.tiles)
        with pytest.raises(AssertionError, match="requires an exact tile set"):
            assert_qt_raster_matches_cpu_reference(win, tiles=())
        with pytest.raises(AssertionError, match="min_samples=1000000000"):
            assert_qt_raster_matches_cpu_reference(win, min_samples_per_tile=1_000_000_000)

        states = _visible_tile_states(win)
        originals = {number: tuple(state.item.levels) for number, state in states.items()}
        for state in states.values():
            low, high = state.item.levels
            state.item.setLevels((low, low + (high - low) * 4.0))
        win.img_view.graphicsView.viewport().update()
        win.img_view.graphicsView.viewport().repaint()
        with pytest.raises(AssertionError, match="Qt raster diverges from the CPU"):
            assert_qt_raster_matches_cpu_reference(win)

        for number, state in states.items():
            state.item.setLevels(originals[number])
        win.img_view.graphicsView.viewport().update()
        win.img_view.graphicsView.viewport().repaint()
        assert_qt_raster_matches_cpu_reference(win)
    finally:
        win.close()
        restore_default_backend(settings)
