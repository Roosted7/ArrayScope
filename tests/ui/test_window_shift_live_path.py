"""Live WGPU source-anchoring and slice-prefetch handoff gates."""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.gpu import DataChunkKey
from arrayscope.tools.framebuffer_reference import assert_wgpu_frame_matches_cpu_reference
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import (
    apply_plane,
    committed_value,
    make_backend_window,
    plane_settled,
    restore_default_backend,
    use_wgpu_backend,
)

CHUNK = 256


def _plane_content_key(win, index: int):
    from arrayscope.display.source_anchoring import source_anchoring_for_view

    anchoring = source_anchoring_for_view(win.document, win.view_state.with_slice(0, index))
    assert anchoring is not None
    return anchoring.content_key


def _resident_chunk_keys_for_content(executor, content_key) -> set[DataChunkKey]:
    return {
        key
        for key in executor.page_table.resident_keys()
        if isinstance(key, DataChunkKey)
        and key.document_generation == ("wgpu-source-plane", content_key)
    }


def test_fixed_index_scroll_forward_hits_warm_wgpu_residency(qtbot):
    """Adjacent-slice prefetch warms WGPU pages and makes 0→1 upload-free."""

    from dataclasses import replace

    settings = use_wgpu_backend(extra_settings={"texture_codec": "off"})
    rng = np.random.default_rng(31)
    data = rng.standard_normal((3, 2 * CHUNK, 4 * CHUNK)).astype(np.float32)
    win = make_backend_window(qtbot, data, backend="wgpu")
    try:
        state = win.view_state.with_image_axes(1, 2)
        win._set_view_state(state.with_slice(0, 0))
        win.render(reason="test-plane-initial")
        qtbot.waitUntil(
            lambda: plane_settled(win, 0),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )

        executor = win.img_view._wgpu_executor
        assert executor is not None
        assert executor.page_table.resident_keys(), "plane 0 did not establish WGPU residency"

        win.app_settings = replace(win.app_settings, prefetch_nearby_slices=True)
        win._active_slice_axis = 0
        win.renderer._prefetch_nearby_slices(win.view_state, None)

        plane_1_key = _plane_content_key(win, 1)
        expected_chunks = (data.shape[1] // CHUNK) * (data.shape[2] // CHUNK)
        qtbot.waitUntil(
            lambda: len(_resident_chunk_keys_for_content(executor, plane_1_key)) >= expected_chunks,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        rows = win.img_view.tileTruthPhysicalRows()
        assert rows
        assert {
            int(identity.source_index)
            for row in rows.values()
            if (identity := row.get("physical_acknowledged_identity")) is not None
        } == {0}

        uploads_before = executor.uploads_total
        apply_plane(win, 1, reason="test-plane-forward")
        qtbot.waitUntil(
            lambda: plane_settled(win, 1),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )

        assert executor.uploads_total == uploads_before
        assert_wgpu_frame_matches_cpu_reference(win)
        value = committed_value(win, CHUNK // 2, CHUNK // 2)
        assert value == pytest.approx(float(data[1, CHUNK // 2, CHUNK // 2]))
    finally:
        win.close()
        restore_default_backend(settings)
