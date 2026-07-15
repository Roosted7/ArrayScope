"""Source-anchored single-image tile identity (ADR 0055 G3b-1)."""

import numpy as np

from arrayscope.core.frame_targets import FrameTarget
from arrayscope.core.view_state import ViewState
from arrayscope.display.backend_contract import VISPY_CAPABILITIES
from arrayscope.display.frame_planner import ANCHORED_CHUNK_SHAPE, FramePlanner, _axis_origins
from arrayscope.display.region_source import EagerDisplayRegionSource
from arrayscope.display.source_anchoring import (
    contiguous_range_start,
    source_anchoring_for_view,
)
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT, Conjugate

from tests.display.vispy_test_utils import FakeDisplayImage


TARGET = FrameTarget(semantic_key="test", viewport_key=None, presentation_key=None, quality="exact-visible")


def windowed_state(shape, *, x_range=None, y_range=None):
    state = ViewState.from_shape(shape).with_image_axes(0, 1)
    if y_range is not None:
        state = state.with_axis_range(0, range(*y_range))
    if x_range is not None:
        state = state.with_axis_range(1, range(*x_range))
    return state


def plan_for(view_state, display_shape, anchoring):
    return FramePlanner().plan(
        target=TARGET,
        view_state=view_state,
        display_shape=display_shape,
        backend_capabilities=VISPY_CAPABILITIES,
        source_anchoring=anchoring,
    )


class TestAxisOrigins:
    def test_unanchored_matches_classic_grid(self):
        assert _axis_origins(100, 64, None) == (0, 64)
        assert _axis_origins(64, 64, None) == (0,)
        assert _axis_origins(0, 64, None) == ()

    def test_anchored_origins_align_to_source_boundaries(self):
        # Window rows 100:200, chunk 64: source boundaries at 128, 192.
        assert _axis_origins(100, 64, 100) == (0, 28, 92)
        # Aligned start behaves like the classic grid.
        assert _axis_origins(100, 64, 128) == (0, 64)


class TestAnchoringDecision:
    def test_contiguous_range_start(self):
        assert contiguous_range_start(range(100, 200)) == 100
        assert contiguous_range_start((5, 6, 7)) == 5
        assert contiguous_range_start((5, 7, 9)) is None
        assert contiguous_range_start(()) is None

    def test_raw_view_anchors_both_axes(self):
        document = ArrayDocument(np.zeros((512, 512), dtype=np.float32))
        state = windowed_state((512, 512), x_range=(100, 200))
        anchoring = source_anchoring_for_view(document, state)
        assert anchoring is not None
        assert anchoring.anchored_starts == (0, 100)

    def test_fft_on_displayed_axis_blocks_that_axis_only(self):
        document = ArrayDocument(np.zeros((512, 512), dtype=np.complex64), operations=(CenteredFFT(axis=1),))
        state = windowed_state((512, 512), x_range=(100, 200), y_range=(50, 400))
        anchoring = source_anchoring_for_view(document, state)
        assert anchoring is not None
        assert anchoring.anchored_starts == (50, None)

    def test_elementwise_chain_still_anchors(self):
        document = ArrayDocument(np.zeros((512, 512), dtype=np.complex64), operations=(Conjugate(),))
        state = windowed_state((512, 512), x_range=(100, 200))
        anchoring = source_anchoring_for_view(document, state)
        assert anchoring is not None and anchoring.any_anchored

    def test_non_contiguous_window_does_not_anchor_its_axis(self):
        document = ArrayDocument(np.zeros((512, 512), dtype=np.float32))
        state = ViewState.from_shape((512, 512)).with_image_axes(0, 1).with_axis_range(1, (1, 5, 9))
        anchoring = source_anchoring_for_view(document, state)
        assert anchoring is not None
        assert anchoring.anchored_starts == (0, None)

    def test_content_key_is_window_shift_invariant(self):
        document = ArrayDocument(np.zeros((512, 512), dtype=np.float32))
        a = source_anchoring_for_view(document, windowed_state((512, 512), x_range=(100, 200)))
        b = source_anchoring_for_view(document, windowed_state((512, 512), x_range=(101, 201)))
        assert a.content_key == b.content_key
        # A data revision bump changes content identity.
        bumped = source_anchoring_for_view(document.mark_base_data_changed(), windowed_state((512, 512), x_range=(100, 200)))
        assert bumped.content_key != a.content_key


class TestAnchoredPlan:
    def test_interior_regions_survive_window_shift(self):
        chunk = ANCHORED_CHUNK_SHAPE[1]
        extent = 4 * chunk  # window wide enough to contain interior chunks
        document = ArrayDocument(np.zeros((2048, 4096), dtype=np.float32))
        old_state = windowed_state((2048, 4096), x_range=(100, 100 + extent))
        new_state = windowed_state((2048, 4096), x_range=(101, 101 + extent))
        old_plan = plan_for(old_state, (2048, extent), source_anchoring_for_view(document, old_state))
        new_plan = plan_for(new_state, (2048, extent), source_anchoring_for_view(document, new_state))
        assert old_plan.source_content_key == new_plan.source_content_key
        old_rects = {region.source_rect for region in old_plan.regions}
        new_rects = {region.source_rect for region in new_plan.regions}
        shared = old_rects & new_rects
        # The interior source chunks are identical across the shift; only
        # boundary strips differ.
        assert shared
        assert len(shared) >= len(new_rects) - 2 * len({r[:2] for r in new_rects})

    def test_anchored_edge_regions_are_clipped_to_the_window(self):
        document = ArrayDocument(np.zeros((1024, 1024), dtype=np.float32))
        state = windowed_state((1024, 1024), x_range=(100, 612))
        plan = plan_for(state, (1024, 512), source_anchoring_for_view(document, state))
        first = plan.regions[0]
        assert first.source_rect[2] == 100  # x starts at the source window start
        widths = {region.source_rect[3] - region.source_rect[2] for region in plan.regions}
        assert ANCHORED_CHUNK_SHAPE[1] in widths  # interior chunks are full
        # Region bounds stay window-relative for drawing.
        assert plan.regions[0].bounds[0] == 0.0

    def test_unanchored_plan_unchanged(self):
        state = windowed_state((512, 512), x_range=(100, 200))
        plan = plan_for(state, (512, 100), None)
        assert plan.source_content_key is None
        assert all(region.source_rect is None for region in plan.regions)


class TestAnchoredSourceIds:
    def test_source_id_is_buffer_independent_for_anchored_regions(self):
        document = ArrayDocument(np.zeros((1024, 2048), dtype=np.float32))
        state = windowed_state((1024, 2048), x_range=(0, 1024))
        anchoring = source_anchoring_for_view(document, state)
        plan = plan_for(state, (1024, 1024), anchoring)
        region = plan.regions[0]
        first = EagerDisplayRegionSource(
            FakeDisplayImage(np.zeros((1024, 1024), dtype=np.float32)),
            source_key=("request", "window-a"),
            content_key=plan.source_content_key,
        ).read_region(region, quality="exact-visible")
        second = EagerDisplayRegionSource(
            FakeDisplayImage(np.zeros((1024, 1024), dtype=np.float32)),
            source_key=("request", "window-b"),
            content_key=plan.source_content_key,
        ).read_region(region, quality="exact-visible")
        assert first.source_id == second.source_id

    def test_legacy_source_id_keeps_buffer_identity(self):
        state = windowed_state((512, 512))
        plan = plan_for(state, (512, 512), None)
        region = plan.regions[0]
        image = FakeDisplayImage(np.zeros((512, 512), dtype=np.float32))
        first = EagerDisplayRegionSource(image, source_key="k").read_region(region, quality="exact-visible")
        other = EagerDisplayRegionSource(
            FakeDisplayImage(np.zeros((512, 512), dtype=np.float32)), source_key="k"
        ).read_region(region, quality="exact-visible")
        assert first.source_id != other.source_id
