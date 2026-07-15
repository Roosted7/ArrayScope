"""G3 live gate: source-anchored window-shift fast path through the real window.

The pool-level halves of this gate prove the residency arithmetic offscreen:

* tests/display/test_window_shift_gate.py — anchored PLANS: per-chunk region
  payloads with window-invariant source ids skip interior uploads.
* tests/display/test_vispy_chunked_residency.py — anchored PAYLOADS (ADR 0055
  G3b-2): the live flow ships ONE window-sized exact payload stamped with a
  ``PayloadSourceAnchor``; the atlas pool chunks it privately into
  origin-anchored 256x256 slots and re-uploads only boundary strips.

This module drives the same scenario through the REAL ArrayScopeWindow flow —
view-state window mutation, render scheduling, frame session, tiled commit —
and measures actual texture uploads, chunk residency, and pixel truth.

Live wiring (2026-07-15): the frame plan itself stays classic (single
window-sized tile, ``source_content_key=None``); the fast path engages one
layer down. ``frame_controller._session_source_anchoring`` gives the session
a window-free content identity on gpu_atlas backends,
``frame_session._payload_source_anchor`` stamps it on exact non-montage
payloads, and ``TextureAtlasPool.update_payloads`` takes the chunked
residency path for eligible payloads (exact, gutter-free, native LOD, plane
larger than one chunk).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.ui.helpers import clear_arrayscope_settings

_WAIT_TIMEOUT_MS = 15_000

CHUNK = 256  # == display.frame_planner.ANCHORED_CHUNK_SHAPE[1]; asserted below
# Four chunks wide so the shifted window keeps interior chunks strictly
# outnumbering the two boundary strips per row (uploads < total / 2).
EXTENT = 4 * CHUNK
START_A = 100
START_B = 101


def _use_vispy_backend():
    from pyqtgraph.Qt import QtCore
    from arrayscope.app.settings_state import ImageRenderingBackendChoice

    clear_arrayscope_settings()
    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.VISPY.value)
    settings.sync()
    return settings


def _restore_default_backend(settings):
    from arrayscope.app.settings_state import ImageRenderingBackendChoice

    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.PYQTGRAPH.value)
    settings.sync()


def _make_window(qtbot, data):
    from arrayscope.display.backend_contract import image_view_backend_capabilities
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    capabilities = image_view_backend_capabilities(win.img_view)
    if capabilities.name != "vispy":
        win.close()
        pytest.skip("VisPy backend unavailable in this Qt environment")
    assert capabilities.tile_residency_kind == "gpu_atlas"
    return win


def _apply_window(win, start: int, *, reason: str) -> None:
    state = win.view_state.with_axis_range(
        1,
        indices=tuple(range(start, start + EXTENT)),
        text=f"{start}:{start + EXTENT}",
    )
    win._set_view_state(state)
    win.render(reason=reason)


def _window_settled(win, start: int) -> bool:
    frame = getattr(win, "_committed_display_frame", None)
    if frame is None:
        return False
    indices = frame.geometry.view_state.axis_range_indices[1]
    if not indices or int(indices[0]) != start or len(indices) != EXTENT:
        return False
    session = getattr(win.renderer, "_frame_session", None)
    if session is None:
        return False
    return (
        session.visible_plan_complete()
        and not win.montage_tile_evaluation_controller.is_busy()
        and session.required_target_settled()
    )


def _wait_for_window(win, qtbot, start: int) -> None:
    qtbot.waitUntil(lambda: _window_settled(win, start), timeout=_WAIT_TIMEOUT_MS)


def _committed_value(win, view_x: int, view_y: int):
    """Probe the committed frame the way live hover does (render.py)."""

    geometry = win.renderer.display_geometry
    if geometry is None:
        return None
    context = geometry.context_for_view_point(float(view_x), float(view_y))
    if context is None:
        return None
    return win.renderer._hover_value_from_display(context.mapping)


def _pool(win):
    return win.img_view._vispy_gpu_montage_layer._pool


def _resident_chunks(pool) -> set:
    return {key for keys in pool.tile_chunk_residency.values() for key in keys}


def _chunk_rect(chunk_key) -> tuple[int, int, int, int]:
    # ("anchored-chunk", content_key, (y0, y1, x0, x1), kind, dtype, lod)
    return tuple(int(value) for value in chunk_key[2])


def test_window_shift_live_pixels_stay_correct(qtbot):
    """E2E sanity half: a one-pixel window shift commits truthful pixels.

    This must stay green regardless of whether the upload fast path engages —
    it proves the harness (window construction, view-state window mutation,
    settle detection, committed-value probing) that the fast-path half
    stands on.
    """

    pytest.importorskip("vispy")
    settings = _use_vispy_backend()
    rng = np.random.default_rng(23)
    data = rng.standard_normal((2 * CHUNK, 8 * CHUNK)).astype(np.float32)
    win = _make_window(qtbot, data)
    try:
        _apply_window(win, START_A, reason="test-window-initial")
        _wait_for_window(win, qtbot, START_A)

        _apply_window(win, START_B, reason="test-window-shift")
        _wait_for_window(win, qtbot, START_B)

        # Interior probe points, well inside chunk boundaries on both axes.
        probes = ((CHUNK // 2, CHUNK // 2), (EXTENT // 2 + 7, CHUNK + 11), (EXTENT - CHUNK, 2 * CHUNK - 1))
        for view_x, view_y in probes:
            qtbot.waitUntil(
                lambda x=view_x, y=view_y: _committed_value(win, x, y) is not None,
                timeout=_WAIT_TIMEOUT_MS,
            )
            value = _committed_value(win, view_x, view_y)
            expected = data[view_y, START_B + view_x]
            assert value == pytest.approx(float(expected)), (
                f"committed value at view ({view_x}, {view_y}) is {value!r}, "
                f"expected source pixel data[{view_y}, {START_B + view_x}] = {expected!r}"
            )
    finally:
        win.close()
        _restore_default_backend(settings)


def test_window_shift_live_uploads_only_boundary_strips(qtbot, monkeypatch):
    """E2E fast-path half: a one-pixel shift re-uploads only boundary strips."""

    pytest.importorskip("vispy")
    from arrayscope.display.frame_planner import ANCHORED_CHUNK_SHAPE
    import arrayscope.display.backends.vispy.tiles as vispy_tiles

    assert ANCHORED_CHUNK_SHAPE == (CHUNK, CHUNK)

    settings = _use_vispy_backend()

    uploads: list[tuple[int, int]] = []
    original_upload = vispy_tiles._upload_texture_plane

    def counting_upload(texture, plane, *, offset, copy):
        shape = tuple(int(value) for value in np.shape(plane)[:2])
        uploads.append(shape)
        return original_upload(texture, plane, offset=offset, copy=copy)

    monkeypatch.setattr(vispy_tiles, "_upload_texture_plane", counting_upload)

    rng = np.random.default_rng(23)
    data = rng.standard_normal((2 * CHUNK, 8 * CHUNK)).astype(np.float32)
    win = _make_window(qtbot, data)
    try:
        _apply_window(win, START_A, reason="test-window-initial")
        _wait_for_window(win, qtbot, START_A)

        pool = _pool(win)
        # Divergence gate: the exact full-window payload must have taken the
        # chunked residency path (ADR 0055 G3b-2) for any of the arithmetic
        # below to be meaningful. If this fails, the live payload was not
        # source-anchored (frame_session._payload_source_anchor) or the pool
        # rejected it (_payload_chunked_eligible in vispy/tiles.py).
        chunks_a = _resident_chunks(pool)
        assert chunks_a, (
            "live commit did not engage chunked residency: no anchored chunks "
            f"resident (tile_slots={dict(pool.tile_slots)!r})"
        )
        rows = {(_chunk_rect(key)[0], _chunk_rect(key)[1]) for key in chunks_a}
        columns = {(_chunk_rect(key)[2], _chunk_rect(key)[3]) for key in chunks_a}
        total = len(chunks_a)
        # First and last column strips per chunk row are clipped by the
        # window, so their content identity legitimately changes on a shift.
        expected_boundary = 2 * len(rows)
        assert len(columns) > 2, "window must contain interior chunk columns"
        content_keys_a = {key[1] for key in chunks_a}
        assert len(content_keys_a) == 1, "one window-invariant content key expected"

        uploads.clear()
        _apply_window(win, START_B, reason="test-window-shift")
        _wait_for_window(win, qtbot, START_B)

        chunks_b = _resident_chunks(pool)
        assert len(chunks_b) == total
        # The content key is window-invariant: the shift renames nothing but
        # the clipped boundary rects.
        assert {key[1] for key in chunks_b} == content_keys_a

        # Residency: interior chunks survive the shift byte-identical.
        overlap = chunks_a & chunks_b
        assert len(overlap) >= total - expected_boundary, (
            f"interior chunks did not survive the shift: overlap={len(overlap)}, "
            f"total={total}, expected_boundary={expected_boundary}"
        )

        # Uploads: only boundary strips may hit the GPU. Count native-height
        # planes so tiny LOD-preview thumbnails cannot mask a full re-upload,
        # and bound the uploaded AREA against the boundary-strip area.
        native_uploads = [shape for shape in uploads if shape[0] >= CHUNK]
        assert len(native_uploads) <= expected_boundary + 2, (
            f"shift uploaded {len(native_uploads)} native strips "
            f"(expected <= {expected_boundary + 2}); all uploads: {uploads}"
        )
        assert len(native_uploads) < total / 2, (
            f"shift re-uploaded {len(native_uploads)} of {total} chunks — "
            "fast path did not engage"
        )
        window_area = EXTENT * data.shape[0]
        native_area = sum(h * w for h, w in native_uploads)
        boundary_area_bound = expected_boundary * CHUNK * CHUNK
        assert native_area <= boundary_area_bound, (
            f"shift uploaded {native_area} px of native planes "
            f"(boundary strips bound: {boundary_area_bound}); uploads: {uploads}"
        )
        assert native_area < window_area / 2, (
            f"shift re-uploaded {native_area} of {window_area} window px — "
            "fast path did not engage"
        )
    finally:
        win.close()
        _restore_default_backend(settings)
