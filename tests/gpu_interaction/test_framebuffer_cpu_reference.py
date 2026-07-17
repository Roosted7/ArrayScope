"""Real-GL framebuffer-to-CPU reference oracle gate (ring 4).

The general oracle mandated by docs/testing/stress-and-trace-strategy.md
(addendum law 2 — *intent is not pixels*): a settled scene's framebuffer must
match a CPU-computed reference of the same semantic values (component/scale/
levels/LUT through ``arrayscope.display.shader_mapping``), with tolerance
only for GPU rounding.  This closes the "visibly wrong while every label is
truthful" gap class: the injected faults below corrupt ONLY physical GPU
state (a uniform, an atlas texture, a texcoord buffer) while payloads, trace
identity, and UI levels all stay correct — the identity/trace oracles cannot
see any of them by construction.

Fault-injection audit (testing law 5 — an oracle that has never failed on an
injected fault is unproven):

* wrong uniform      — ``set_levels`` on the live page visual;
* stale page         — overwrite the atlas texture texels behind fresh
                       mapping keys (no app path re-uploads: the identity is
                       clean, which is exactly why only pixels can catch it);
* swapped tile       — two tiles' texcoord spans exchanged in the vertex
                       buffer (reused/aliased atlas slot presentation).

Ring: tests/gpu_interaction only (real display, real GL).  The default-ring
smoke lives in tests/ui/test_framebuffer_cpu_reference.py; it keeps the
oracle honest offscreen but is never acceptance for rendering claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.gpu_interaction.conftest import (
    COUNT,
    TILE,
    Harness,
)


def gradient_montage_data() -> np.ndarray:
    """(64, 64, 36): frame k = 10*k + smooth x+y gradient (span ~8).

    Distinct per-tile offsets make stale/swapped content visible; the
    in-tile gradient makes wrong geometry (flip, shifted texcoords, wrong
    LOD placement) visible where the constant identity-ramp tiles of
    ``synthetic_montage_data`` cannot be.
    """

    yy, xx = np.mgrid[0:TILE, 0:TILE].astype(np.float32)
    gradient = (yy + xx) * (8.0 / (2.0 * (TILE - 1)))
    frames = np.arange(COUNT, dtype=np.float32)[:, None, None] * 10.0 + gradient[None]
    return frames.transpose(1, 2, 0).copy()


@pytest.fixture()
def gradient_montage_window():
    """Production window pinned to the VisPy backend in a dedicated QSettings
    namespace (profile-harness pattern) — the user's real ArrayScope settings
    stay untouched."""

    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.launch import _prepare_qt_environment
    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.window import ArrayScopeWindow

    _prepare_qt_environment()
    app = pg.mkQApp()
    previous_names = (str(app.organizationName()), str(app.applicationName()))
    app.setOrganizationName("ArrayScope")
    app.setApplicationName("ArrayScopeGpuOracleHarness")
    settings = QtCore.QSettings()
    settings.clear()
    settings.setValue(
        "image_rendering_backend", ImageRenderingBackendChoice.VISPY.value
    )
    settings.sync()
    win = ArrayScopeWindow(gradient_montage_data())
    win.setWindowTitle("gpu-fb-cpu-oracle")
    win.show()
    try:
        harness = Harness(app, win)
        harness.pump(0.3)
        vs = win.view_state
        win._set_view_state(vs.with_montage_axis(2, text=":"))
        win.render(reason="gpu-fb-cpu-oracle-montage")
        assert harness.wait_settled(), (
            f"montage never settled after open: {harness.settlement_diagnostics()}"
        )
        yield harness
    finally:
        win.close()
        for _ in range(50):
            app.processEvents()
        settings.clear()
        settings.sync()
        app.setOrganizationName(previous_names[0])
        app.setApplicationName(previous_names[1])


def _require_vispy_layer(harness):
    layer = getattr(harness.win.img_view, "_vispy_gpu_montage_layer", None)
    if layer is None:
        pytest.skip("framebuffer CPU-reference oracle needs the VisPy GPU atlas layer")
    return layer


def _settled_healthy_report(harness):
    harness.fit_plan_view()
    harness.pump(0.3)
    assert harness.wait_settled(), (
        f"scene never settled: {harness.settlement_diagnostics()}"
    )
    # Regime guard (strategy law 3): this gate pins the native-LOD scalar
    # regime; silently running reduced would weaken every assertion below.
    session = harness.session
    for number in session.required_tile_numbers():
        payload = session.display_tile_payloads[int(number)]
        level = 0 if payload.lod is None else int(payload.lod.level)
        assert level == 0, (
            f"tile {number} presented LOD level {level}; this gate pins the "
            "native-resolution regime"
        )
    return harness.assert_tile_matches_cpu_reference()


def _visible_visuals(layer):
    visuals = [
        (index, visual)
        for index, visual in enumerate(layer._visuals_by_page)
        if bool(getattr(visual, "visible", False))
    ]
    assert visuals, "no visible VisPy tile page visual"
    return visuals


def test_settled_scene_matches_cpu_reference(gradient_montage_window):
    harness = gradient_montage_window
    _require_vispy_layer(harness)
    report = _settled_healthy_report(harness)
    required = set(harness.session.required_tile_numbers())
    assert {tile.tile_number for tile in report.tiles} == required
    assert len(report.tiles) == COUNT
    assert all(
        tile.samples >= report.min_samples_per_tile for tile in report.tiles
    ), "oracle sample floor not met — comparison would be vacuous"


def test_wrong_levels_uniform_fails_oracle_and_recovers(gradient_montage_window):
    harness = gradient_montage_window
    layer = _require_vispy_layer(harness)
    _settled_healthy_report(harness)

    visuals = _visible_visuals(layer)
    originals = [tuple(visual._levels) for _index, visual in visuals]
    for _index, visual in visuals:
        low, high = visual._levels
        visual.set_levels((low, low + (high - low) * 4.0))
    with pytest.raises(AssertionError, match="diverges from the CPU"):
        harness.assert_tile_matches_cpu_reference()

    # Restoring the uniform restores the oracle: the failure was caused by
    # the injected fault, not by comparison noise.
    for (_index, visual), levels in zip(visuals, originals):
        visual.set_levels(levels)
    harness.assert_tile_matches_cpu_reference()


def test_stale_page_content_fails_oracle(gradient_montage_window):
    harness = gradient_montage_window
    layer = _require_vispy_layer(harness)
    _settled_healthy_report(harness)

    # Overwrite every visible page's scalar atlas texels while all mapping
    # keys, payloads, and acknowledgements stay fresh — the pure physical
    # staleness class no identity oracle can see.
    for index, _visual in _visible_visuals(layer):
        page = layer._pool.pages[index]
        assert page.scalar_is_atlas
        page.scalar_texture.set_data(
            np.full(page.atlas_shape + (1,), 177.0, dtype=np.float32)
        )
    with pytest.raises(AssertionError, match="diverges from the CPU"):
        harness.assert_tile_matches_cpu_reference()


def test_swapped_tile_texcoords_fail_oracle_and_recover(gradient_montage_window):
    harness = gradient_montage_window
    layer = _require_vispy_layer(harness)
    _settled_healthy_report(harness)

    from arrayscope.display.backends.vispy.tiles import _tile_quad_rects

    page_index, visual = _visible_visuals(layer)[0]
    payloads = layer._page_payloads_by_index[page_index]

    # Locate each tile's vertex span exactly the way the buffers were built
    # (sorted tile order, 6 vertices per quad — mirrors _quad_buffers).
    spans: dict[int, tuple[int, int]] = {}
    offset = 0
    for tile_number in sorted(int(key) for key in payloads):
        quads = _tile_quad_rects(
            tile_number,
            layer._last_layout,
            layer._pool.tile_uvs,
            layer._pool.tile_draw_parts,
            page_index=page_index,
        )
        count = 6 * len(quads)
        spans[tile_number] = (offset, count)
        offset += count
    swappable = [
        (number, span) for number, span in spans.items() if span[1] == 6
    ]
    assert len(swappable) >= 2, f"need two single-quad tiles to swap: {spans}"
    (tile_a, (offset_a, count_a)), (tile_b, (offset_b, count_b)) = swappable[:2]

    original = np.asarray(visual.texcoord_data, dtype=np.float32).copy()
    swapped = original.copy()
    swapped[offset_a : offset_a + count_a] = original[offset_b : offset_b + count_b]
    swapped[offset_b : offset_b + count_b] = original[offset_a : offset_a + count_a]
    visual.set_geometry(visual.vertex_data, swapped, visual.mode_data)

    with pytest.raises(AssertionError, match="diverges from the CPU") as excinfo:
        harness.assert_tile_matches_cpu_reference()
    message = str(excinfo.value)
    assert f"tile {tile_a}:" in message and f"tile {tile_b}:" in message, (
        f"swapped tiles {tile_a}/{tile_b} must both be reported: {message}"
    )

    # Swap back: the oracle recovers, proving the failure was the injected
    # aliasing and nothing else.
    visual.set_geometry(visual.vertex_data, original, visual.mode_data)
    harness.assert_tile_matches_cpu_reference()
