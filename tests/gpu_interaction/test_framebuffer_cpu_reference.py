"""Real-GPU WGPU framebuffer-to-CPU reference oracle gate (ring 4).

The production WGPU target must match an independently computed CPU reference
while all semantic payloads and identities remain current.  The injected
faults corrupt only physical rendering state: mapping commands, resident page
bytes, and submitted tile-instance geometry.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.gpu_interaction.conftest import COUNT, TILE, Harness


def gradient_montage_data() -> np.ndarray:
    yy, xx = np.mgrid[0:TILE, 0:TILE].astype(np.float32)
    gradient = (yy + xx) * (8.0 / (2.0 * (TILE - 1)))
    frames = np.arange(COUNT, dtype=np.float32)[:, None, None] * 10.0 + gradient[None]
    return frames.transpose(1, 2, 0).copy()


@pytest.fixture
def gradient_montage_window():
    """Production window pinned to exact WGPU storage in isolated settings."""

    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.launch import _prepare_qt_environment
    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.display.backend_contract import image_view_backend_capabilities
    from arrayscope.window import ArrayScopeWindow

    _prepare_qt_environment()
    app = pg.mkQApp()
    previous_names = (str(app.organizationName()), str(app.applicationName()))
    app.setOrganizationName("ArrayScope")
    app.setApplicationName("ArrayScopeGpuOracleHarness")
    settings = QtCore.QSettings()
    settings.clear()
    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.WGPU.value)
    settings.setValue("texture_codec", "off")
    settings.sync()
    win = ArrayScopeWindow(gradient_montage_data())
    win.setWindowTitle("wgpu-fb-cpu-oracle")
    win.show()
    try:
        if image_view_backend_capabilities(win.img_view).name != "wgpu":
            pytest.skip("WGPU backend unavailable in this display environment")
        harness = Harness(app, win)
        harness.pump(0.3)
        win._set_view_state(win.view_state.with_montage_axis(2, text=":"))
        win.render(reason="wgpu-fb-cpu-oracle-montage")
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


def _settled_healthy_report(harness):
    harness.fit_plan_view()
    harness.pump(0.3)
    assert harness.wait_settled(), f"scene never settled: {harness.settlement_diagnostics()}"
    for number in harness.session.required_tile_numbers():
        payload = harness.session.display_tile_payloads[int(number)]
        level = 0 if payload.lod is None else int(payload.lod.level)
        assert level == 0, f"tile {number} presented unexpected LOD level {level}"
    return harness.assert_tile_matches_cpu_reference()


def test_settled_scene_matches_cpu_reference(gradient_montage_window):
    report = _settled_healthy_report(gradient_montage_window)
    required = set(gradient_montage_window.session.required_tile_numbers())
    assert {tile.tile_number for tile in report.tiles} == required
    assert len(report.tiles) == COUNT
    assert all(tile.samples >= report.min_samples_per_tile for tile in report.tiles)


def test_wrong_mapping_command_fails_oracle_and_recovers(
    gradient_montage_window,
    monkeypatch,
):
    harness = gradient_montage_window
    view = harness.win.img_view
    _settled_healthy_report(harness)

    from arrayscope.gpu.command_protocol import SetDisplayMapping

    original_submit = view._submit_wgpu
    mapping = view._wgpu_mapping_state
    bad_mapping = replace(
        mapping,
        level_hi=mapping.level_lo + 4.0 * (mapping.level_hi - mapping.level_lo),
    )

    def corrupt_mapping(commands, **kwargs):
        corrupted = tuple(
            SetDisplayMapping(bad_mapping) if isinstance(command, SetDisplayMapping) else command
            for command in commands
        )
        return original_submit(corrupted, **kwargs)

    monkeypatch.setattr(view, "_submit_wgpu", corrupt_mapping)
    with pytest.raises(AssertionError, match="diverges from the CPU"):
        harness.assert_tile_matches_cpu_reference()
    monkeypatch.setattr(view, "_submit_wgpu", original_submit)
    harness.assert_tile_matches_cpu_reference()


def test_stale_resident_page_content_fails_oracle(gradient_montage_window):
    harness = gradient_montage_window
    view = harness.win.img_view
    _settled_healthy_report(harness)

    from arrayscope.gpu.keys import SCALAR_R32F
    from arrayscope.gpu.wgpu_executor import PAGE

    executor = view._wgpu_executor
    stale = np.full((PAGE, PAGE), 177.0, dtype=np.float32)
    overwritten = 0
    for key in executor.page_table.resident_keys():
        if key.representation != SCALAR_R32F:
            continue
        slot = executor.page_table.lookup(key)
        assert slot is not None
        pool = executor._pool_for_slot(slot)
        executor.device.queue.write_texture(
            {"texture": pool.texture, "origin": (0, 0, slot.page_index)},
            stale,
            {"bytes_per_row": PAGE * 4, "rows_per_image": PAGE},
            (PAGE, PAGE, 1),
        )
        overwritten += 1
    assert overwritten > 0
    with pytest.raises(AssertionError, match="diverges from the CPU"):
        harness.assert_tile_matches_cpu_reference()


def test_swapped_tile_instances_fail_oracle_and_recover(
    gradient_montage_window,
    monkeypatch,
):
    harness = gradient_montage_window
    view = harness.win.img_view
    _settled_healthy_report(harness)

    original_instances = view._wgpu_tile_instances()
    assert len(original_instances) >= 2
    first, second = original_instances[:2]
    assert first.plane_index != second.plane_index
    swapped = (
        replace(first, plane_index=second.plane_index),
        replace(second, plane_index=first.plane_index),
        *original_instances[2:],
    )
    monkeypatch.setattr(view, "_wgpu_tile_instances", lambda: swapped)
    with pytest.raises(AssertionError, match="diverges from the CPU"):
        harness.assert_tile_matches_cpu_reference()
    monkeypatch.setattr(view, "_wgpu_tile_instances", lambda: original_instances)
    harness.assert_tile_matches_cpu_reference()
