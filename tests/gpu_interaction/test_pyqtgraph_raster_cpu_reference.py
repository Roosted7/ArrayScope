"""Real-display PyQtGraph Qt-raster CPU-reference gate (ring 4).

The gate reads the painted QGraphicsView viewport and compares every required
scalar tile with ``cpu_display_rgba`` of committed payload values under the
semantic levels/LUT. It closes the PyQtGraph half of ground rule 10: Qt's
raster pixels now have the same non-vacuous physical-vs-CPU law as VisPy.

Fault-injection audit (docs/testing/README.md law 5):

* wrong levels -- corrupt each live ImageItem's raster levels only;
* stale pixmap -- replace one ImageItem's cached QImage while its source
  array, identity, and committed payload remain current;
* swapped tiles -- exchange two live ImageItems' physical positions while
  all semantic tile ownership remains unchanged.

Each corruption must fail the oracle, and reversible faults must turn green
again after repair. Offscreen coverage lives in
``tests/ui/test_pyqtgraph_raster_cpu_reference.py`` and is not acceptance.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pytest

from tests.gpu_interaction.conftest import COUNT, TILE, Harness


def gradient_montage_data() -> np.ndarray:
    yy, xx = np.mgrid[0:TILE, 0:TILE].astype(np.float32)
    gradient = (yy + xx) * (8.0 / (2.0 * (TILE - 1)))
    frames = np.arange(COUNT, dtype=np.float32)[:, None, None] * 10.0
    return (frames + gradient[None]).transpose(1, 2, 0).copy()


@pytest.fixture
def pyqtgraph_gradient_montage_window():
    """Production PyQtGraph window under an isolated QSettings namespace."""

    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.launch import _prepare_qt_environment
    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.window import ArrayScopeWindow

    _prepare_qt_environment()
    app = pg.mkQApp()
    previous_names = (str(app.organizationName()), str(app.applicationName()))
    app.setOrganizationName("ArrayScope")
    app.setApplicationName("ArrayScopeQtRasterOracleHarness")
    settings = QtCore.QSettings()
    settings.clear()
    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.PYQTGRAPH.value)
    settings.sync()
    win = ArrayScopeWindow(gradient_montage_data())
    win.setWindowTitle("pyqtgraph-raster-cpu-oracle")
    win.show()
    try:
        harness = Harness(app, win)
        harness.pump(0.3)
        win._set_view_state(win.view_state.with_montage_axis(2, text=":"))
        win.render(reason="pyqtgraph-raster-cpu-oracle-montage")
        assert harness.wait_settled(), (
            f"montage never settled after open: {harness.settlement_diagnostics()}"
        )
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
        win.img_view.graphicsView.viewport().repaint()
        yield harness
    finally:
        win.close()
        for _ in range(50):
            app.processEvents()
        settings.clear()
        settings.sync()
        app.setOrganizationName(previous_names[0])
        app.setApplicationName(previous_names[1])


def _require_pyqtgraph_layer(harness):
    assert getattr(harness.win.img_view, "_vispy_canvas", None) is None
    layer = getattr(harness.win.img_view, "_montage_tile_layer", None)
    if layer is None:
        pytest.skip("Qt-raster CPU-reference oracle needs the PyQtGraph tile layer")
    return layer


def _visible_states(harness):
    layer = _require_pyqtgraph_layer(harness)
    states = {
        int(number): state
        for number, state in layer.states.items()
        if state.visible and state.item.isVisible()
    }
    required = {int(number) for number in harness.session.required_tile_numbers()}
    assert set(states) == required, (
        f"physical PyQtGraph tile set drifted: required={sorted(required)}, "
        f"visible={sorted(states)}"
    )
    return states


def _settled_healthy_report(harness):
    harness.fit_plan_view()
    harness.pump(0.3)
    assert harness.wait_settled(), f"scene never settled: {harness.settlement_diagnostics()}"
    for number in harness.session.required_tile_numbers():
        payload = harness.session.display_tile_payloads[int(number)]
        level = 0 if payload.lod is None else int(payload.lod.level)
        assert level == 0, (
            f"tile {number} presented LOD level {level}; this gate pins the "
            "native-resolution scalar regime"
        )
    return harness.assert_tile_matches_cpu_reference()


def test_settled_scene_matches_cpu_reference(pyqtgraph_gradient_montage_window):
    harness = pyqtgraph_gradient_montage_window
    report = _settled_healthy_report(harness)
    required = {int(number) for number in harness.session.required_tile_numbers()}
    assert {tile.tile_number for tile in report.tiles} == required
    assert len(report.tiles) == COUNT
    assert all(tile.samples >= report.min_samples_per_tile for tile in report.tiles), (
        "oracle sample floor not met -- comparison would be vacuous"
    )


def test_wrong_levels_fail_oracle_and_recover(pyqtgraph_gradient_montage_window):
    harness = pyqtgraph_gradient_montage_window
    _settled_healthy_report(harness)
    states = _visible_states(harness)
    originals = {number: tuple(state.item.levels) for number, state in states.items()}
    for state in states.values():
        low, high = state.item.levels
        state.item.setLevels((low, low + (high - low) * 4.0))
    harness.win.img_view.graphicsView.viewport().update()
    harness.win.img_view.graphicsView.viewport().repaint()
    with pytest.raises(AssertionError, match="Qt raster diverges from the CPU"):
        harness.assert_tile_matches_cpu_reference()

    for number, state in states.items():
        state.item.setLevels(originals[number])
    harness.win.img_view.graphicsView.viewport().update()
    harness.win.img_view.graphicsView.viewport().repaint()
    harness.assert_tile_matches_cpu_reference()


def test_stale_tile_pixmap_fails_oracle_and_recovers(
    pyqtgraph_gradient_montage_window,
):
    harness = pyqtgraph_gradient_montage_window
    _settled_healthy_report(harness)
    states = _visible_states(harness)
    tile_a, tile_b = sorted(states)[:2]
    stale_state = states[tile_a]
    donor_state = states[tile_b]
    stale_state.item.render()
    donor_state.item.render()
    original = stale_state.item.qimage.copy()
    stale_state.item.qimage = donor_state.item.qimage.copy()
    stale_state.item._renderRequired = False
    harness.win.img_view.graphicsView.viewport().update()
    harness.win.img_view.graphicsView.viewport().repaint()

    with pytest.raises(AssertionError, match="Qt raster diverges from the CPU") as excinfo:
        harness.assert_tile_matches_cpu_reference()
    assert f"tile {tile_a}:" in str(excinfo.value)

    stale_state.item.qimage = original
    stale_state.item._renderRequired = False
    harness.win.img_view.graphicsView.viewport().update()
    harness.win.img_view.graphicsView.viewport().repaint()
    harness.assert_tile_matches_cpu_reference()


def test_swapped_tile_positions_fail_oracle_and_recover(
    pyqtgraph_gradient_montage_window,
):
    harness = pyqtgraph_gradient_montage_window
    _settled_healthy_report(harness)
    states = _visible_states(harness)
    tile_a, tile_b = sorted(states)[:2]
    state_a = states[tile_a]
    state_b = states[tile_b]
    position_a = state_a.item.pos()
    position_b = state_b.item.pos()
    state_a.item.setPos(position_b)
    state_b.item.setPos(position_a)
    harness.win.img_view.graphicsView.viewport().update()
    harness.win.img_view.graphicsView.viewport().repaint()

    with pytest.raises(AssertionError, match="Qt raster diverges from the CPU") as excinfo:
        harness.assert_tile_matches_cpu_reference()
    message = str(excinfo.value)
    assert f"tile {tile_a}:" in message
    assert f"tile {tile_b}:" in message

    state_a.item.setPos(position_a)
    state_b.item.setPos(position_b)
    harness.win.img_view.graphicsView.viewport().update()
    harness.win.img_view.graphicsView.viewport().repaint()
    harness.assert_tile_matches_cpu_reference()
