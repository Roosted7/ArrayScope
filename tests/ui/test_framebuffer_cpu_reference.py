"""Default-ring smoke for the WGPU framebuffer-to-CPU reference oracle."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from arrayscope.tools.framebuffer_reference import assert_wgpu_frame_matches_cpu_reference
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import (
    frame_session_settled,
    make_backend_window,
    restore_default_backend,
    use_wgpu_backend,
)

TILE = 32
GRID = 3
COUNT = GRID * GRID
_PAL_RELAXED_ORANGE = np.asarray((249, 127, 16), dtype=np.int16)


def _gradient_montage_data() -> np.ndarray:
    yy, xx = np.mgrid[0:TILE, 0:TILE].astype(np.float32)
    gradient = (yy + xx) * (8.0 / (2.0 * (TILE - 1)))
    frames = np.arange(COUNT, dtype=np.float32)[:, None, None] * 20.0 + gradient[None]
    return frames.transpose(1, 2, 0).copy()


def _orange_pixel_count(frame: np.ndarray) -> int:
    rgb = np.asarray(frame)[..., :3].astype(np.int16)
    return int(np.count_nonzero(np.all(np.abs(rgb - _PAL_RELAXED_ORANGE) <= 16, axis=-1)))


def test_settled_montage_matches_cpu_reference_and_rejects_bad_mapping(qtbot, monkeypatch):
    settings = use_wgpu_backend(extra_settings={"texture_codec": "off"})
    win = make_backend_window(qtbot, _gradient_montage_data(), backend="wgpu")
    try:
        win.resize(720, 600)
        win.show()
        qtbot.waitExposed(win)
        win._set_view_state(win.view_state.with_montage_axis(2, text=":"))
        win.render(reason="wgpu-fb-cpu-reference-smoke")
        qtbot.waitUntil(
            lambda: frame_session_settled(win),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )

        session = win.renderer._frame_session
        required = set(session.required_tile_numbers())
        assert len(required) == COUNT, f"smoke regime drifted: {sorted(required)}"
        report = assert_wgpu_frame_matches_cpu_reference(win)
        assert {tile.tile_number for tile in report.tiles} == required
        assert all(tile.samples >= report.min_samples_per_tile for tile in report.tiles)

        from arrayscope.gpu.command_protocol import SetDisplayMapping

        original_submit = win.img_view._submit_wgpu
        mapping = win.img_view._wgpu_mapping_state
        bad_mapping = replace(
            mapping,
            level_hi=mapping.level_lo + 4.0 * (mapping.level_hi - mapping.level_lo),
        )

        def corrupt_mapping(commands, **kwargs):
            corrupted = tuple(
                SetDisplayMapping(bad_mapping)
                if isinstance(command, SetDisplayMapping)
                else command
                for command in commands
            )
            return original_submit(corrupted, **kwargs)

        monkeypatch.setattr(win.img_view, "_submit_wgpu", corrupt_mapping)
        with pytest.raises(AssertionError, match="diverges from the CPU"):
            assert_wgpu_frame_matches_cpu_reference(win)
        monkeypatch.setattr(win.img_view, "_submit_wgpu", original_submit)
        assert_wgpu_frame_matches_cpu_reference(win)
    finally:
        win.close()
        restore_default_backend(settings)


def test_phase_color_zero_magnitude_is_physically_black_and_repairs_mapping(qtbot):
    """Zero magnitude modulates the phase LUT to black in production WGPU."""

    settings = use_wgpu_backend(extra_settings={"texture_codec": "off"})
    data = np.zeros((128, 128), dtype=np.complex64)
    data[8:24, 8:24] = 40.0 + 0.0j
    win = make_backend_window(qtbot, data, backend="wgpu")
    try:
        win.resize(720, 600)
        win.show()
        qtbot.waitExposed(win)
        win._on_channel_clicked("complex")
        win.render(reason="wgpu-phase-zero-background")
        qtbot.waitUntil(
            lambda: frame_session_settled(win),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )

        assert_wgpu_frame_matches_cpu_reference(win)
        view = win.img_view
        mapping = view._wgpu_mapping_state
        assert mapping.mode == "magnitude"
        assert mapping.phase_color is True
        healthy = view._wgpu_executor.read_target()
        assert _orange_pixel_count(healthy) == 0

        from arrayscope.gpu.command_protocol import SetDisplayMapping, UpdateTileInstances

        bad_mapping = replace(mapping, phase_color=False)
        view._submit_wgpu(
            (
                SetDisplayMapping(bad_mapping),
                view._wgpu_camera_command(),
                UpdateTileInstances(view._wgpu_tile_instances()),
            )
        )
        corrupted = view._wgpu_executor.read_target()
        assert _orange_pixel_count(corrupted) > 100

        view._submit_wgpu(
            (
                SetDisplayMapping(mapping),
                view._wgpu_camera_command(),
                UpdateTileInstances(view._wgpu_tile_instances()),
            )
        )
        recovered = view._wgpu_executor.read_target()
        assert _orange_pixel_count(recovered) == 0
        assert_wgpu_frame_matches_cpu_reference(win)
    finally:
        win.close()
        restore_default_backend(settings)
