"""G3 gate (offscreen half): window shifts upload only boundary chunks.

Drives the real chain — anchoring decision → frame plan → region payloads →
atlas residency — across a ±1 same-extent window shift and counts actual
texture uploads. The real-display harness scenario complements this with
pixel/trace evidence; this test pins the residency arithmetic.
"""

import numpy as np

from arrayscope.core.frame_targets import FrameTarget
from arrayscope.core.view_state import ViewState
from arrayscope.display.backend_contract import VISPY_CAPABILITIES
from arrayscope.display.backends.vispy.tiles import TextureAtlasPool
from arrayscope.display.frame_planner import ANCHORED_CHUNK_SHAPE, FramePlanner
from arrayscope.display.region_source import EagerDisplayRegionSource
from arrayscope.display.source_anchoring import source_anchoring_for_view
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT


class FakeTexture2D:
    def __init__(self, data=None, *, shape=None, **kwargs):
        self.shape = tuple(shape) if shape is not None else tuple(np.shape(data))

    def set_data(self, data, *, offset=None, copy=True):
        pass


class FakeGloo:
    Texture2D = FakeTexture2D


class FakeDisplayImage:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.histogram_data = None


TARGET = FrameTarget(semantic_key="gate", viewport_key=None, presentation_key=None, quality="exact-visible")
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
        backend_capabilities=VISPY_CAPABILITIES,
        source_anchoring=anchoring,
    )
    windowed = np.ascontiguousarray(data[:, x_start : x_start + x_extent])
    source = EagerDisplayRegionSource(
        FakeDisplayImage(windowed),
        source_key=("request", x_start, x_extent),
        content_key=plan.source_content_key,
    )
    payloads = {
        int(region.region_id): source.read_region(region, quality="exact-visible")
        for region in plan.regions
    }
    return plan, payloads


def commit(pool, payloads, tile_shape):
    _uvs, stats = pool.update_payloads(
        payloads,
        tile_shape=tile_shape,
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=len(payloads),
    )
    return stats


def test_one_pixel_window_shift_uploads_only_boundary_chunks():
    rng = np.random.default_rng(7)
    data = rng.standard_normal((2 * CHUNK, 8 * CHUNK)).astype(np.float32)
    document = ArrayDocument(data)
    extent = 4 * CHUNK

    plan_a, payloads_a = payloads_for_window(document, data, 100, extent)
    plan_b, payloads_b = payloads_for_window(document, data, 101, extent)

    pool = TextureAtlasPool(FakeGloo())
    cold = commit(pool, payloads_a, plan_a.tile_shape)
    assert cold.items_updated == len(payloads_a)

    warm = commit(pool, payloads_b, plan_b.tile_shape)
    columns = {region.source_rect[2:] for region in plan_b.regions}
    rows = {region.source_rect[:2] for region in plan_b.regions}
    boundary_tiles = 2 * len(rows)  # first and last column strips per row
    assert warm.items_updated == boundary_tiles
    assert warm.items_skipped == len(payloads_b) - boundary_tiles
    # Sanity: the window genuinely contains interior columns.
    assert len(columns) > 2

    # Scrolling back to the original window re-uses the boundary strips that
    # stayed resident: zero uploads.
    back = commit(pool, dict(payloads_a), plan_a.tile_shape)
    assert back.items_updated == 0


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


def test_fft_along_shifted_axis_reuploads_everything():
    rng = np.random.default_rng(13)
    data = rng.standard_normal((CHUNK, 6 * CHUNK)).astype(np.float32) + 0j
    document = ArrayDocument(data.astype(np.complex64), operations=(CenteredFFT(axis=1),))
    extent = 3 * CHUNK

    plan_a, payloads_a = payloads_for_window(document, data.real, 100, extent)
    plan_b, payloads_b = payloads_for_window(document, data.real, 101, extent)

    pool = TextureAtlasPool(FakeGloo())
    commit(pool, payloads_a, plan_a.tile_shape)
    warm = commit(pool, payloads_b, plan_b.tile_shape)
    # The x-window is folded into every tile's identity when the chain
    # consumes it: nothing may be reused across the shift.
    assert warm.items_updated == len(payloads_b)
    assert warm.items_skipped == 0
