"""G3 gate (offscreen half): window shifts preserve canonical chunk identity.

Drives the real chain — anchoring decision → frame plan → region payloads →
payload identities — across a ±1 same-extent window shift. WGPU's physical
zero-upload/rebind behavior is pinned in ``test_wgpu_imageview2d`` and
``test_resident_crop_rebind``; this file pins the backend-neutral identity
arithmetic those paths consume.
"""

import numpy as np

from arrayscope.core.frame_targets import FrameTarget
from arrayscope.core.view_state import ViewState
from arrayscope.display.backend_contract import WGPU_CAPABILITIES
from arrayscope.display.frame_planner import ANCHORED_CHUNK_SHAPE, FramePlanner
from arrayscope.display.region_source import EagerDisplayRegionSource
from arrayscope.display.slice_engine import DisplayImage
from arrayscope.display.source_anchoring import source_anchoring_for_view
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT

TARGET = FrameTarget(
    semantic_key="gate", viewport_key=None, presentation_key=None, quality="exact-visible"
)
CHUNK = ANCHORED_CHUNK_SHAPE[1]


def windowed_state(shape, x_start, x_extent):
    return (
        ViewState.from_shape(shape)
        .with_image_axes(0, 1)
        .with_axis_range(1, range(x_start, x_start + x_extent))
    )


def payloads_for_window(document, data, x_start, x_extent):
    state = windowed_state(data.shape, x_start, x_extent)
    anchoring = source_anchoring_for_view(document, state)
    plan = FramePlanner().plan(
        target=TARGET,
        view_state=state,
        display_shape=(data.shape[0], x_extent),
        backend_capabilities=WGPU_CAPABILITIES,
        source_anchoring=anchoring,
    )
    windowed = np.ascontiguousarray(data[:, x_start : x_start + x_extent])
    source = EagerDisplayRegionSource(
        DisplayImage(windowed),
        source_key=("request", x_start, x_extent),
        content_key=plan.source_content_key,
    )
    payloads = {
        int(region.region_id): source.read_region(region, quality="exact-visible")
        for region in plan.regions
    }
    return plan, payloads


def _source_ids(payloads):
    return {payload.source_id for payload in payloads.values()}


def test_one_pixel_window_shift_reuses_only_interior_chunk_identities():
    rng = np.random.default_rng(7)
    data = rng.standard_normal((2 * CHUNK, 8 * CHUNK)).astype(np.float32)
    document = ArrayDocument(data)
    extent = 4 * CHUNK

    _plan_a, payloads_a = payloads_for_window(document, data, 100, extent)
    plan_b, payloads_b = payloads_for_window(document, data, 101, extent)

    columns = {region.source_rect[2:] for region in plan_b.regions}
    rows = {region.source_rect[:2] for region in plan_b.regions}
    boundary_tiles = 2 * len(rows)  # first and last column strips per row
    shared = _source_ids(payloads_a) & _source_ids(payloads_b)
    assert len(shared) == len(payloads_b) - boundary_tiles
    # Sanity: the window genuinely contains interior columns.
    assert len(columns) > 2

    # Re-deriving the original window from fresh arrays keeps every canonical
    # identity stable; the WGPU executor may therefore rebind it upload-free.
    _plan_again, payloads_a_again = payloads_for_window(document, data, 100, extent)
    assert _source_ids(payloads_a_again) == _source_ids(payloads_a)


def test_shared_source_rect_payloads_are_pixel_identical():
    rng = np.random.default_rng(11)
    data = rng.standard_normal((CHUNK, 6 * CHUNK)).astype(np.float32)
    document = ArrayDocument(data)
    extent = 3 * CHUNK

    plan_a, payloads_a = payloads_for_window(document, data, 100, extent)
    plan_b, payloads_b = payloads_for_window(document, data, 101, extent)

    rects_a = {region.source_rect: region.region_id for region in plan_a.regions}
    shared = 0
    for region in plan_b.regions:
        tile_a = rects_a.get(region.source_rect)
        if tile_a is None:
            continue
        shared += 1
        a = payloads_a[tile_a]
        b = payloads_b[int(region.region_id)]
        # Equal source_id must mean equal pixels — the invariant the whole
        # fast path stands on.
        assert a.source_id == b.source_id
        assert np.array_equal(np.asarray(a.image), np.asarray(b.image))
    assert shared > 0


def test_fft_along_shifted_axis_rekeys_everything():
    rng = np.random.default_rng(13)
    data = rng.standard_normal((CHUNK, 6 * CHUNK)).astype(np.float32) + 0j
    document = ArrayDocument(data.astype(np.complex64), operations=(CenteredFFT(axis=1),))
    extent = 3 * CHUNK

    _plan_a, payloads_a = payloads_for_window(document, data.real, 100, extent)
    _plan_b, payloads_b = payloads_for_window(document, data.real, 101, extent)

    # The x-window is folded into every tile's identity when the chain
    # consumes it: nothing may be reused across the shift.
    assert _source_ids(payloads_a).isdisjoint(_source_ids(payloads_b))


def scrolled_state(shape, index):
    return ViewState.from_shape(shape).with_image_axes(1, 2).with_slice(0, index)


def payloads_for_plane(document, data, index, *, extent):
    state = scrolled_state(data.shape, index)
    anchoring = source_anchoring_for_view(document, state)
    assert anchoring is not None
    plan = FramePlanner().plan(
        target=TARGET,
        view_state=state,
        display_shape=(data.shape[1], extent),
        backend_capabilities=WGPU_CAPABILITIES,
        source_anchoring=anchoring,
    )
    plane = np.ascontiguousarray(data[index, :, :extent])
    source = EagerDisplayRegionSource(
        DisplayImage(plane),
        source_key=("request", index),
        content_key=anchoring.content_key,
    )
    return anchoring, {
        int(region.region_id): source.read_region(region, quality="exact-visible")
        for region in plan.regions
    }


def test_fixed_index_scroll_back_restores_the_same_chunk_identities():
    """G4a: revisiting an already-seen plane can re-use resident chunks."""

    rng = np.random.default_rng(3)
    data = rng.standard_normal((3, CHUNK, 4 * CHUNK)).astype(np.float32)
    document = ArrayDocument(data)
    extent = 3 * CHUNK

    anchor_0, plane_0 = payloads_for_plane(document, data, 0, extent=extent)
    anchor_1, plane_1 = payloads_for_plane(document, data, 1, extent=extent)
    # Different fixed indexes are different content.
    assert anchor_0.content_key != anchor_1.content_key

    assert _source_ids(plane_0).isdisjoint(_source_ids(plane_1))
    # Re-derive plane 0 from scratch (fresh buffers): reuse must come from
    # content identity, not object identity.
    _anchor, plane_0_again = payloads_for_plane(document, data, 0, extent=extent)
    assert _source_ids(plane_0_again) == _source_ids(plane_0)


def test_non_windowable_chain_still_gets_scroll_back_identity():
    """Content-keyed residency is independent of window anchoring: an FFT
    along both display axes anchors no axis, but revisiting the identical
    plane still re-uses resident chunks."""

    rng = np.random.default_rng(5)
    data = (rng.standard_normal((2, CHUNK, 3 * CHUNK)) + 0j).astype(np.complex64)
    document = ArrayDocument(data, operations=(CenteredFFT(axis=1), CenteredFFT(axis=2)))
    extent = 3 * CHUNK

    anchoring = source_anchoring_for_view(document, scrolled_state(data.shape, 0))
    assert anchoring is not None
    assert anchoring.anchored_starts == (None, None)
    assert anchoring.any_anchored is False

    _anchor_a, plane_a = payloads_for_plane(
        document, data.real.astype(np.float32), 0, extent=extent
    )
    _anchor_b, plane_b = payloads_for_plane(
        document, data.real.astype(np.float32), 1, extent=extent
    )
    assert _source_ids(plane_a).isdisjoint(_source_ids(plane_b))
    _anchor, plane_a_again = payloads_for_plane(
        document, data.real.astype(np.float32), 0, extent=extent
    )
    assert _source_ids(plane_a_again) == _source_ids(plane_a)
