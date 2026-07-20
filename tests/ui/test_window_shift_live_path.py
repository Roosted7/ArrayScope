"""G3 live gate: source-anchored window-shift fast path through the real window.

The pool-level halves of this gate prove the residency arithmetic offscreen:

* tests/display/test_window_shift_gate.py — anchored PLANS: per-chunk region
  payloads with window-invariant source ids skip interior uploads.
* tests/display/test_vispy_chunked_residency.py — anchored PAYLOADS (ADR 0055
  G3b-2): the live flow ships ONE window-sized exact payload stamped with a
  ``PayloadSourceAnchor``; the atlas pool chunks it into canonical
  ``DataChunkKey`` pages in
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

from arrayscope.gpu import DataChunkKey
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import (
    apply_plane,
    committed_value,
    frame_session_settled,
    make_backend_window,
    plane_settled,
    restore_default_backend,
    use_vispy_backend,
)

CHUNK = 256  # == display.frame_planner.ANCHORED_CHUNK_SHAPE[1]; asserted below
# Four chunks wide so the shifted window keeps interior chunks strictly
# outnumbering the two boundary strips per row (uploads < total / 2).
EXTENT = 4 * CHUNK
START_A = 100
START_B = 101


@pytest.fixture
def upload_log(monkeypatch):
    """Shapes of every real ``_upload_texture_plane`` call (uploads still run)."""

    import arrayscope.display.backends.vispy.tiles as vispy_tiles

    uploads: list[tuple[int, int]] = []
    original = vispy_tiles._upload_texture_plane

    def counting_upload(texture, plane, *, offset, copy):
        uploads.append(tuple(int(value) for value in np.shape(plane)[:2]))
        return original(texture, plane, offset=offset, copy=copy)

    monkeypatch.setattr(vispy_tiles, "_upload_texture_plane", counting_upload)
    return uploads


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
    return frame_session_settled(win)


def _wait_for_window(win, qtbot, start: int) -> None:
    qtbot.waitUntil(lambda: _window_settled(win, start), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)


def _pool(win):
    return win.img_view._vispy_gpu_montage_layer._pool


def _resident_chunks(pool) -> set:
    return {key for keys in pool.tile_chunk_residency.values() for key in keys}


def _chunk_rect(chunk_key) -> tuple[int, int, int, int]:
    y0, x0 = chunk_key.chunk_origin
    y1, x1 = chunk_key.stop
    return (int(y0), int(y1), int(x0), int(x1))


def _chunk_content_key(chunk_key: DataChunkKey) -> tuple[object, ...]:
    return (
        "src-anchored",
        chunk_key.document_generation,
        chunk_key.operation_key,
    )


def test_window_shift_live_pixels_stay_correct(qtbot):
    """E2E sanity half: a one-pixel window shift commits truthful pixels.

    This must stay green regardless of whether the upload fast path engages —
    it proves the harness (window construction, view-state window mutation,
    settle detection, committed-value probing) that the fast-path half
    stands on.
    """

    pytest.importorskip("vispy")
    settings = use_vispy_backend()
    rng = np.random.default_rng(23)
    data = rng.standard_normal((2 * CHUNK, 8 * CHUNK)).astype(np.float32)
    win = make_backend_window(qtbot, data, require_gpu_atlas=True)
    try:
        _apply_window(win, START_A, reason="test-window-initial")
        _wait_for_window(win, qtbot, START_A)

        _apply_window(win, START_B, reason="test-window-shift")
        _wait_for_window(win, qtbot, START_B)

        # Interior probe points, well inside chunk boundaries on both axes.
        probes = (
            (CHUNK // 2, CHUNK // 2),
            (EXTENT // 2 + 7, CHUNK + 11),
            (EXTENT - CHUNK, 2 * CHUNK - 1),
        )
        for view_x, view_y in probes:
            qtbot.waitUntil(
                lambda x=view_x, y=view_y: committed_value(win, x, y) is not None,
                timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
            )
            value = committed_value(win, view_x, view_y)
            expected = data[view_y, START_B + view_x]
            assert value == pytest.approx(float(expected)), (
                f"presented value at view ({view_x}, {view_y}) is {value!r}, "
                f"expected source pixel data[{view_y}, {START_B + view_x}] = {expected!r}"
            )
    finally:
        win.close()
        restore_default_backend(settings)


def test_window_shift_live_uploads_only_boundary_strips(qtbot, upload_log):
    """E2E fast-path half: a one-pixel shift re-uploads only boundary strips."""

    pytest.importorskip("vispy")
    from arrayscope.display.frame_planner import ANCHORED_CHUNK_SHAPE

    assert ANCHORED_CHUNK_SHAPE == (CHUNK, CHUNK)

    settings = use_vispy_backend()
    uploads = upload_log

    rng = np.random.default_rng(23)
    data = rng.standard_normal((2 * CHUNK, 8 * CHUNK)).astype(np.float32)
    win = make_backend_window(qtbot, data, require_gpu_atlas=True)
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
        content_keys_a = {_chunk_content_key(key) for key in chunks_a}
        assert len(content_keys_a) == 1, "one window-invariant content key expected"

        uploads.clear()
        _apply_window(win, START_B, reason="test-window-shift")
        _wait_for_window(win, qtbot, START_B)

        chunks_b = _resident_chunks(pool)
        assert len(chunks_b) == total
        # The content key is window-invariant: the shift renames nothing but
        # the clipped boundary rects.
        assert {_chunk_content_key(key) for key in chunks_b} == content_keys_a

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
        # One canonical factor-2 page can be admitted concurrently with the
        # four native boundary strips and is also 256x256 physically.  The
        # chunk-key overlap above proves native reuse; allow that one bounded
        # logical-page upload without misclassifying the fast path as a full
        # native re-upload.
        assert len(native_uploads) <= total / 2, (
            f"shift re-uploaded {len(native_uploads)} of {total} chunks — fast path did not engage"
        )
        window_area = EXTENT * data.shape[0]
        native_area = sum(h * w for h, w in native_uploads)
        boundary_area_bound = expected_boundary * CHUNK * CHUNK
        assert native_area <= boundary_area_bound, (
            f"shift uploaded {native_area} px of native planes "
            f"(boundary strips bound: {boundary_area_bound}); uploads: {uploads}"
        )
        assert native_area < window_area / 2, (
            f"shift re-uploaded {native_area} of {window_area} window px — fast path did not engage"
        )
    finally:
        win.close()
        restore_default_backend(settings)


def _plane_content_key(win, index: int):
    from arrayscope.display.source_anchoring import source_anchoring_for_view

    anchoring = source_anchoring_for_view(win.document, win.view_state.with_slice(0, index))
    assert anchoring is not None
    return anchoring.content_key


def _resident_chunk_keys_for_content(pool, content_key) -> set:
    return {
        key
        for key in pool._page_table.resident_keys()
        if isinstance(key, DataChunkKey) and _chunk_content_key(key) == content_key
    }


def test_fixed_index_scroll_forward_hits_warm_residency(qtbot, upload_log):
    """G4c live gate: prefetch warms the NEXT plane's GPU chunks in advance.

    After plane 0 settles, the slice prefetcher evaluates plane 1 on the
    PREFETCH lane, stores it in the CPU display cache, and pushes its
    anchored chunks into atlas residency (render_prefetch ->
    VisPyImageView2D.warmPlaneResidency -> pool chunk-warm branch). The
    subsequent scroll 0 -> 1 then commits through the chunked path with
    ZERO chunk uploads.
    """

    pytest.importorskip("vispy")
    from arrayscope.app.settings_state import AppSettingsState

    settings = use_vispy_backend()
    uploads = upload_log

    rng = np.random.default_rng(31)
    data = rng.standard_normal((3, 2 * CHUNK, 4 * CHUNK)).astype(np.float32)
    win = make_backend_window(qtbot, data, require_gpu_atlas=True)
    try:
        state = win.view_state.with_image_axes(1, 2)
        win._set_view_state(state.with_slice(0, 0))
        win.render(reason="test-plane-initial")
        qtbot.waitUntil(lambda: plane_settled(win, 0), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

        pool = _pool(win)
        assert _resident_chunks(pool), "plane 0 did not engage chunked residency"

        # Drive the slice prefetcher the way tests/ui/test_prefetch.py does:
        # enable the setting, declare the scrub axis, and submit the request.
        win.app_settings = AppSettingsState(
            theme=win.app_settings.theme, prefetch_nearby_slices=True
        )
        win._active_slice_axis = 0
        win.renderer._prefetch_nearby_slices(win.view_state, None)

        plane_1_key = _plane_content_key(win, 1)
        expected_chunks = (data.shape[1] // CHUNK) * (data.shape[2] // CHUNK)

        def plane_1_warm() -> bool:
            return len(_resident_chunk_keys_for_content(pool, plane_1_key)) >= expected_chunks

        qtbot.waitUntil(plane_1_warm, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        # The warm plane is residency only: tile 0 still presents plane 0.
        presented = _resident_chunks(pool)
        assert not (presented & _resident_chunk_keys_for_content(pool, plane_1_key)), (
            "warm plane 1 chunks must not be presented before the scroll"
        )

        chunk_uploads_before = pool.chunk_upload_count
        uploads.clear()
        apply_plane(win, 1, reason="test-plane-forward")
        qtbot.waitUntil(lambda: plane_settled(win, 1), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

        # The scroll commit presented plane 1 through the chunked path...
        presented_now = _resident_chunks(pool)
        assert presented_now & _resident_chunk_keys_for_content(pool, plane_1_key), (
            "plane 1 was not presented through chunked residency"
        )
        # ...and found every chunk already warm: ZERO chunk uploads.
        assert pool.chunk_upload_count == chunk_uploads_before, (
            f"scroll-forward uploaded {pool.chunk_upload_count - chunk_uploads_before} "
            f"chunks despite prefetch warming (all uploads: {uploads})"
        )
        # No classic full-plane fallback upload either.
        full_plane_uploads = [shape for shape in uploads if shape[0] >= 2 * CHUNK]
        assert not full_plane_uploads, (
            f"scroll-forward re-uploaded full native planes: {full_plane_uploads} "
            f"(all uploads: {uploads})"
        )

        # Pixel truth on the warmed plane.
        value = committed_value(win, CHUNK // 2, CHUNK // 2)
        assert value == pytest.approx(float(data[1, CHUNK // 2, CHUNK // 2]))
    finally:
        win.close()
        restore_default_backend(settings)


def test_fixed_index_scroll_back_live_is_upload_free(qtbot, upload_log):
    """G4a live gate: revisiting an already-seen plane re-uses GPU residency.

    Content-keyed residency (source anchor per plane) plus grow-before-evict
    means scrolling 0 -> 1 -> 0 re-uploads nothing native on the return leg,
    even though every scroll step still rebirths the frame session.
    """

    pytest.importorskip("vispy")

    settings = use_vispy_backend()
    uploads = upload_log

    rng = np.random.default_rng(29)
    data = rng.standard_normal((3, 2 * CHUNK, 4 * CHUNK)).astype(np.float32)
    win = make_backend_window(qtbot, data, require_gpu_atlas=True)
    try:
        state = win.view_state.with_image_axes(1, 2)
        win._set_view_state(state.with_slice(0, 0))
        win.render(reason="test-plane-initial")
        qtbot.waitUntil(lambda: plane_settled(win, 0), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

        pool = _pool(win)
        chunks_plane_0 = _resident_chunks(pool)
        assert chunks_plane_0, "plane 0 did not engage chunked residency"

        apply_plane(win, 1, reason="test-plane-forward")
        qtbot.waitUntil(lambda: plane_settled(win, 1), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        # Different plane, different content: plane 0's chunks are unlinked
        # from the tile but must SURVIVE as warm page-table residency
        # (grow-before-evict) while plane 1's arrive.
        warm_resident = set(pool._page_table.resident_keys())
        missing = {key for key in chunks_plane_0 if key not in warm_resident}
        assert not missing, (
            f"plane 0 residency was evicted during forward scroll despite "
            f"byte-budget headroom: {len(missing)} of {len(chunks_plane_0)} gone"
        )

        uploads.clear()
        apply_plane(win, 0, reason="test-plane-back")
        qtbot.waitUntil(lambda: plane_settled(win, 0), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

        native_uploads = [shape for shape in uploads if shape[0] >= CHUNK]
        assert not native_uploads, (
            f"scroll-back re-uploaded native planes: {native_uploads} (all uploads: {uploads})"
        )

        # Pixel truth on the revisited plane.
        value = committed_value(win, CHUNK // 2, CHUNK // 2)
        assert value == pytest.approx(float(data[0, CHUNK // 2, CHUNK // 2]))
    finally:
        win.close()
        restore_default_backend(settings)


def test_zoomed_out_reduced_target_presents_via_chunked_residency(qtbot):
    """ADR 0056 G5 slice 1 live half: a reduced EXACT target takes the
    chunked-residency path with uniform plane-pixel pages.

    Zooming the camera out (world range = 3x the viewport per axis) makes the
    resident LOD policy demand an isotropic factor-2 level.  The ladder never
    demotes already-presented finer content (camera-only zooms must not churn
    materialization), so the reduced target engages on the NEXT content
    change: shifting the data window while zoomed out materializes the
    demanded level-1 plane, and the exact reduced payload presents through
    ``tile_chunk_residency`` with factor-2 keys in uniform 256^2 plane-pixel
        slots (the same shape class native chunks use) instead of the classic
        one-giant-slot-per-plane-size path. Presentation-qualified lookup
        follows the canonical bins while exact semantic probes reject the
        display-only reduced values.
    """

    pytest.importorskip("vispy")
    settings = use_vispy_backend()
    rng = np.random.default_rng(23)
    data = rng.standard_normal((2 * CHUNK, 8 * CHUNK)).astype(np.float32)
    shifted_start = START_A + CHUNK // 2
    win = make_backend_window(qtbot, data, require_gpu_atlas=True)
    try:
        _apply_window(win, START_A, reason="test-window-initial")
        _wait_for_window(win, qtbot, START_A)

        pool = _pool(win)
        assert _resident_chunks(pool), "native window did not engage chunked residency"

        session = win.renderer._frame_session
        viewport_h, viewport_w = (int(value) for value in session.viewport_shape)
        assert viewport_h > 0
        assert viewport_w > 0
        # 3.0 source texels per screen pixel on BOTH axes: desired factor 2
        # on both (isotropic — anisotropic reductions fall back to classic).
        view = win.img_view.getView()
        center_x, center_y = (EXTENT / 2.0, CHUNK)
        half_w, half_h = (1.5 * viewport_w, 1.5 * viewport_h)
        view.setRange(
            xRange=(center_x - half_w, center_x + half_w),
            yRange=(center_y - half_h, center_y + half_h),
            padding=0,
        )
        _apply_window(win, shifted_start, reason="test-window-shift-zoomed-out")

        def reduced_chunks() -> set:
            return {key for key in _resident_chunks(pool) if int(key.lod.factor) > 1}

        def factor_two_converged() -> bool:
            current = win.renderer._frame_session
            payload = current.display_tile_payloads.get(0)
            return bool(
                payload is not None
                and str(payload.quality) == "exact"
                and int(payload.lod.factor) == 2
                and any(int(key.lod.factor) == 2 for key in reduced_chunks())
            )

        # A retained factor-16 floor may resolve first. It is honest fallback,
        # not the requested target; wait for the ladder's exact factor-2 rung.
        qtbot.waitUntil(factor_two_converged, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        chunks = {key for key in reduced_chunks() if int(key.lod.factor) == 2}
        # Factor-2 keys are the exact canonical page-plan keys. Boundary
        # pages may be clipped while interior pages stay globally aligned.
        assert {(key.lod.factor, key.lod.level, key.lod.gutter) for key in chunks} == {(2, 1, 0)}
        payload = win.renderer._frame_session.display_tile_payloads[0]
        assert payload.page_backing is not None
        assert payload.page_backing.source_coverage_yx == (
            0,
            2 * CHUNK,
            shifted_start,
            shifted_start + EXTENT,
        )
        expected_keys = {plan.key for plan in payload.page_backing.requested_plans}
        assert chunks == expected_keys
        # Uniform plane-pixel pages: the reduced chunks live in the same
        # 256^2 shape class as native chunks.
        slots = pool.resident_slots
        for key in chunks:
            page_index, _slot = slots[key]
            assert pool.pages[page_index].tile_shape == (CHUNK, CHUNK)
        # The presented payload really is the exact reduced target.
        session = win.renderer._frame_session
        payload = session.display_tile_payloads.get(0)
        assert payload is not None
        assert str(payload.quality) == "exact"
        assert int(payload.lod.factor) == 2

        # Presentation-qualified sampling maps native coordinates through the
        # exact globally aligned source-grid bin. It must not use the old
        # window-local reduced-plane pixel convention. The committed semantic
        # probe stays unavailable: a reduced display page without an explicit
        # native semantic plane cannot answer hover/measurement/export truth.
        probes = ((CHUNK // 2, CHUNK // 2), (CHUNK + 11, CHUNK // 2 + 7))
        for view_x, view_y in probes:
            source_x = shifted_start + view_x
            source_y = view_y
            value = payload.page_backing.sample_presented_value_at_native(
                source_y,
                source_x,
            )
            bin_x = (source_x // 2) * 2
            bin_y = (source_y // 2) * 2
            expected = np.mean(data[bin_y : bin_y + 2, bin_x : bin_x + 2])
            assert value == pytest.approx(float(expected)), (
                f"presented value at view ({view_x}, {view_y}) is {value!r}, "
                f"expected canonical source-grid bin {expected!r}"
            )
            assert committed_value(win, view_x, view_y) is None
    finally:
        win.close()
        restore_default_backend(settings)
