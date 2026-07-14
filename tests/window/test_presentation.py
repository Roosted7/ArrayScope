from types import SimpleNamespace

import numpy as np

from pyqtgraph.Qt import QtCore

from arrayscope.core.scheduler import FrameTarget
from arrayscope.core.view_state import ViewState
from arrayscope.display.backend_contract import PYQTGRAPH_CAPABILITIES
from arrayscope.display.frame_planner import FramePlanner
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.slice_engine import DisplayImage
from arrayscope.display.shader_mapping import ShaderComponent, ShaderMapping
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.display.model.frame import (
    CommittedDisplayFrame,
    DisplayFrameKey,
    DisplayTilePayload,
    TilePresentationDelta,
    TilePresentationState,
    TiledValueSource,
)
from arrayscope.display.planning import LevelSource, LevelSourceRank, decide_presentation
from arrayscope.window.display_presenter import DisplayPresentationMixin
from arrayscope.display.model.commit import (
    CommitKind,
    DisplayPayload,
    DisplayTiledPresentation,
    PresentationInput,
    RenderRequestContext,
)


class _FakeTimer:
    def __init__(self, _parent):
        self.timeout = self
        self.active = False
        self.remaining_ms = -1
        self.starts = []

    def setSingleShot(self, _single_shot):
        pass

    def connect(self, _callback):
        pass

    def isActive(self):
        return self.active

    def remainingTime(self):
        return self.remaining_ms

    def start(self, delay_ms):
        self.active = True
        self.remaining_ms = int(delay_ms)
        self.starts.append(int(delay_ms))


def test_interactive_viewport_update_is_cadenced_not_debounced(monkeypatch):
    monkeypatch.setattr(QtCore, "QTimer", _FakeTimer)

    presenter = SimpleNamespace()
    presenter._run_interactive_montage_viewport_update = lambda: None

    DisplayPresentationMixin._schedule_interactive_montage_viewport_update(presenter)
    timer = presenter._interactive_montage_viewport_timer
    DisplayPresentationMixin._schedule_interactive_montage_viewport_update(presenter)

    assert timer.starts == [16]


def _geometry(shape=(2, 2)):
    return DisplayGeometry(view_state=ViewState.from_shape(shape), display_shape=shape)


def _context():
    return RenderRequestContext(document_key=("doc", 1), request_key=("image", 1), render_generation=1, semantic_key="levels")


def _payload(data, *, histogram_data=None):
    image = DisplayImage(np.asarray(data, dtype=np.float32), histogram_data=None if histogram_data is None else np.asarray(histogram_data, dtype=np.float32))
    frame_plan = FramePlanner().plan(
        target=FrameTarget(("test", image.data.shape), None, None, "exact-visible"),
        view_state=ViewState.from_shape(image.data.shape[:2]).with_image_axes(0, 1),
        display_shape=image.data.shape[:2],
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
    )
    payloads = _payloads_for_frame_plan(image, frame_plan)
    tile_state = TilePresentationState(payloads)
    tile_delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=payloads,
        active_tiles=frame_plan.active_region_ids,
        planned_tiles=frame_plan.planned_region_ids,
        near_tiles=frame_plan.near_region_ids,
    )
    return DisplayPayload(
        image=image,
        geometry=frame_plan.geometry,
        viewport_policy=ViewportPolicy.PRESERVE,
        frame_plan=frame_plan,
        tile_state=tile_state,
        base_tile_state=TilePresentationState(),
        tile_delta=tile_delta,
    )


def _frame(*, levels=(10.0, 20.0), histogram_range=(0.0, 100.0)):
    data = np.zeros((2, 2), dtype=np.float32)
    payload = DisplayTilePayload(0, 0, data, data.copy(), ("frame", 0), semantic_data=data, semantic_histogram_data=data.copy(), source_shape=data.shape)
    return CommittedDisplayFrame(
        data=data,
        histogram_data=data.copy(),
        geometry=_geometry(),
        levels=levels,
        histogram_range=histogram_range,
        key=DisplayFrameKey(("doc", 1), ("image", 0), 1, "levels"),
        value_source=TiledValueSource({0: payload}),
    )


def _payloads_for_frame_plan(image: DisplayImage, frame_plan) -> dict[int, DisplayTilePayload]:
    data = np.asarray(image.data)
    histogram = None if image.histogram_data is None else np.asarray(image.histogram_data)
    payloads = {}
    for region in frame_plan.regions:
        y_slice, x_slice = region.data_slices
        tile_data = data[y_slice, x_slice, ...]
        tile_histogram = None if histogram is None else histogram[y_slice, x_slice]
        payloads[int(region.region_id)] = DisplayTilePayload(
            int(region.region_id),
            int(region.region_id),
            tile_data,
            tile_histogram,
            ("payload", int(region.region_id), tuple(tile_data.shape), str(tile_data.dtype)),
            semantic_data=tile_data,
            semantic_histogram_data=tile_histogram,
            source_shape=tile_data.shape[:2],
        )
    return payloads


def _input(
    payload,
    *,
    previous_frame=None,
    force_auto=False,
    kind=CommitKind.FULL_FRAME_INITIAL,
    semantic_source=None,
    applied_level_source=None,
    window_mode="relative",
    user_levels=None,
):
    return PresentationInput(
        payload=payload,
        context=_context(),
        previous_frame=previous_frame,
        window_mode=window_mode,
        force_auto=force_auto,
        commit_kind=kind,
        semantic_source=semantic_source,
        applied_level_source=applied_level_source,
        user_levels=user_levels,
    )


def _frame_level_source(bounds=(200.0, 300.0)):
    return LevelSource(bounds, bounds, LevelSourceRank.MONTAGE_COMPLETE, source_count=1, expected_count=1, semantic_key="levels")


def test_frame_relative_level_reuse_uses_committed_frame():
    decision = decide_presentation(
        _input(
            _payload([[200, 300], [200, 300]]),
            previous_frame=_frame(levels=(25, 75), histogram_range=(0, 100)),
            semantic_source=_frame_level_source(),
        )
    )

    assert decision.levels == (225.0, 275.0)
    assert decision.histogram_range == (200.0, 300.0)


def test_frame_presentation_preserves_frame_plan_semantics():
    payload = _payload(np.zeros((4, 4), dtype=np.float32))
    frame_plan = FramePlanner(internal_tile_shape=(2, 2)).plan(
        target=FrameTarget("semantic", "viewport", "presentation", "exact-visible"),
        view_state=payload.geometry.view_state.with_image_axes(0, 1),
        display_shape=(4, 4),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
    )
    payloads = _payloads_for_frame_plan(payload.image, frame_plan)
    tile_state = TilePresentationState(payloads)
    tile_delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=payloads,
        active_tiles=frame_plan.active_region_ids,
        planned_tiles=frame_plan.planned_region_ids,
        near_tiles=frame_plan.near_region_ids,
    )
    payload = DisplayPayload(
        image=payload.image,
        geometry=frame_plan.geometry,
        viewport_policy=payload.viewport_policy,
        frame_plan=frame_plan,
        tile_state=tile_state,
        base_tile_state=TilePresentationState(),
        tile_delta=tile_delta,
    )

    decision = decide_presentation(_input(payload))

    assert decision.display_presentation.frame_plan is frame_plan


def test_frame_absolute_level_reuse_uses_committed_frame():
    decision = decide_presentation(
        _input(
            _payload([[200, 300], [200, 300]]),
            previous_frame=_frame(levels=(25, 75), histogram_range=(0, 100)),
            window_mode="absolute",
            semantic_source=_frame_level_source(),
        )
    )

    assert decision.levels == (25.0, 75.0)
    assert decision.histogram_range == (200.0, 300.0)


def test_frame_relative_restore_levels_are_not_a_durable_user_lock():
    decision = decide_presentation(
        _input(
            _payload([[200, 300], [200, 300]]),
            previous_frame=_frame(levels=(25, 75), histogram_range=(0, 100)),
            user_levels=(210, 240),
            semantic_source=_frame_level_source(),
        )
    )

    assert decision.levels == (210.0, 240.0)
    assert decision.histogram_range == (200.0, 300.0)
    assert decision.level_source_rank == int(LevelSourceRank.MONTAGE_COMPLETE)


def test_explicit_auto_window_wins_over_queued_restore_levels():
    decision = decide_presentation(
        _input(
            _payload([[200, 300], [200, 300]]),
            force_auto=True,
            user_levels=(210, 240),
            semantic_source=_frame_level_source(),
        )
    )

    assert decision.levels == (200.0, 300.0)


def test_explicit_auto_window_accepts_partial_montage_source():
    source = LevelSource((100.0, 200.0), (100.0, 200.0), LevelSourceRank.MONTAGE_VISIBLE_SUBSET, source_count=1, expected_count=4, semantic_key="levels")
    decision = decide_presentation(
        _input(
            _payload(np.full((2, 2), 1000.0)),
            previous_frame=_frame(),
            force_auto=True,
            kind=CommitKind.EXPLICIT_AUTO_WINDOW,
            semantic_source=source,
        )
    )

    assert decision.levels == (100.0, 200.0)
    assert decision.level_source_rank == int(LevelSourceRank.MONTAGE_VISIBLE_SUBSET)


def test_progressive_frame_patch_accepts_partial_implicit_source_monotonically():
    source = LevelSource((100.0, 200.0), (100.0, 200.0), LevelSourceRank.MONTAGE_VISIBLE_SUBSET, source_count=1, expected_count=4, semantic_key="levels")
    decision = decide_presentation(
        _input(
            _payload(np.full((2, 2), 1000.0)),
            previous_frame=_frame(levels=(2.0, 8.0), histogram_range=(0.0, 10.0)),
            kind=CommitKind.PROGRESSIVE_FRAME_PATCH,
            semantic_source=source,
        )
    )

    assert decision.levels == (40.0, 160.0)
    assert decision.histogram_range == (0.0, 200.0)


def test_progressive_frame_patch_accepts_complete_source():
    source = LevelSource((0.0, 300.0), (0.0, 300.0), LevelSourceRank.MONTAGE_COMPLETE, source_count=4, expected_count=4, semantic_key="levels")
    decision = decide_presentation(
        _input(
            _payload(np.full((2, 2), 1000.0)),
            previous_frame=_frame(levels=(2.0, 8.0), histogram_range=(0.0, 10.0)),
            kind=CommitKind.PROGRESSIVE_FRAME_PATCH,
            semantic_source=source,
        )
    )

    assert decision.levels == (60.0, 240.0)
    assert decision.histogram_range == (0.0, 300.0)


def test_degenerate_complete_source_does_not_shrink_previous_levels():
    source = LevelSource((5.0, 5.0), (float("nan"), float("nan")), LevelSourceRank.MONTAGE_COMPLETE, source_count=4, expected_count=4, semantic_key="levels")
    decision = decide_presentation(
        _input(
            _payload(np.full((2, 2), np.nan)),
            previous_frame=_frame(levels=(2.0, 8.0), histogram_range=(0.0, 10.0)),
            kind=CommitKind.PROGRESSIVE_FRAME_PATCH,
            semantic_source=source,
        )
    )

    assert decision.levels == (2.0, 8.0)
    assert decision.histogram_range == (0.0, 10.0)


def test_user_locked_montage_levels_are_not_overridden_by_complete_source():
    user = LevelSource((20.0, 40.0), (0.0, 100.0), LevelSourceRank.EXPLICIT_USER, semantic_key="levels")
    complete = LevelSource((0.0, 300.0), (0.0, 300.0), LevelSourceRank.MONTAGE_COMPLETE, source_count=4, expected_count=4, semantic_key="levels")

    decision = decide_presentation(
        _input(
            _payload(np.full((2, 2), 1000.0)),
            previous_frame=_frame(levels=(2.0, 8.0), histogram_range=(0.0, 10.0)),
            kind=CommitKind.PROGRESSIVE_FRAME_PATCH,
            semantic_source=complete,
            applied_level_source=user,
        )
    )

    assert decision.levels == (20.0, 40.0)
    assert decision.histogram_range == (0.0, 300.0)
    assert decision.level_source_rank == int(LevelSourceRank.EXPLICIT_USER)


def test_montage_restore_levels_bind_to_the_current_semantic_source():
    partial = LevelSource(
        (100.0, 200.0),
        (100.0, 200.0),
        LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
        source_count=1,
        expected_count=4,
        semantic_key="levels",
    )
    decision = decide_presentation(
        _input(
            _payload(np.full((2, 2), 1000.0)),
            kind=CommitKind.FULL_FRAME_INITIAL,
            semantic_source=partial,
            user_levels=(120.0, 140.0),
        )
    )

    assert decision.levels == (120.0, 140.0)
    assert decision.histogram_range == (100.0, 200.0)
    assert decision.level_source_key == "levels"
    assert decision.level_source_rank == int(LevelSourceRank.MONTAGE_VISIBLE_SUBSET)


def test_montage_absolute_preserves_numeric_levels_while_histogram_improves():
    absolute = LevelSource((20.0, 40.0), (0.0, 100.0), LevelSourceRank.EXPLICIT_USER, semantic_key="levels")
    complete = LevelSource((0.0, 300.0), (0.0, 300.0), LevelSourceRank.MONTAGE_COMPLETE, source_count=4, expected_count=4, semantic_key="levels")

    decision = decide_presentation(
        _input(
            _payload(np.full((2, 2), 1000.0)),
            kind=CommitKind.PROGRESSIVE_FRAME_PATCH,
            semantic_source=complete,
            applied_level_source=absolute,
            window_mode="absolute",
        )
    )

    assert decision.levels == (20.0, 40.0)
    assert decision.histogram_range == (0.0, 300.0)


def test_explicit_auto_clears_user_lock_and_uses_best_available_source():
    user = LevelSource((20.0, 40.0), (0.0, 100.0), LevelSourceRank.EXPLICIT_USER, semantic_key="levels")
    partial = LevelSource((100.0, 200.0), (100.0, 200.0), LevelSourceRank.MONTAGE_VISIBLE_SUBSET, source_count=1, expected_count=4, semantic_key="levels")

    decision = decide_presentation(
        _input(
            _payload(np.full((2, 2), 1000.0)),
            previous_frame=_frame(),
            force_auto=True,
            kind=CommitKind.EXPLICIT_AUTO_WINDOW,
            semantic_source=partial,
            applied_level_source=user,
        )
    )

    assert decision.levels == (100.0, 200.0)
    assert decision.level_source_rank == int(LevelSourceRank.MONTAGE_VISIBLE_SUBSET)


def test_montage_dirty_tiles_pass_through_presentation():
    payload = _payload(np.zeros((2, 2), dtype=np.float32), histogram_data=np.zeros((2, 2), dtype=np.float32))

    decision = decide_presentation(
        _input(
            payload,
            previous_frame=_frame(),
            kind=CommitKind.PROGRESSIVE_FRAME_PATCH,
        )
    )

    assert tuple(decision.display_presentation.tile_delta.upserts) == (0,)


def test_typed_tile_payloads_create_first_class_tiled_presentation():
    state = ViewState.from_shape((2, 2, 1)).with_image_axes(0, 1).with_montage_axis(2, columns=1, indices=(0,))
    geometry = DisplayGeometry(
        view_state=state,
        display_shape=(2, 2),
        montage=MontageGeometry(indices=(0,), tile_shape=(2, 2), columns=1, rows=1, gap=0),
        montage_tile_states=("loaded",),
    )
    tile = DisplayTilePayload(0, 0, np.ones((2, 2), dtype=np.float32), None, ("tile", 0))
    tile_state = TilePresentationState({0: tile})
    tile_delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts={0: tile},
        active_tiles=(0,),
        planned_tiles=(0,),
        near_tiles=(0,),
    )
    payload = DisplayPayload(
        image=DisplayImage(np.zeros((2, 2), dtype=np.float32)),
        geometry=geometry,
        viewport_policy=ViewportPolicy.PRESERVE,
        tile_state=tile_state,
        tile_delta=tile_delta,
        tile_residency_budget_bytes=64 * 1024 * 1024,
    )

    decision = decide_presentation(
        _input(payload, kind=CommitKind.FULL_FRAME_INITIAL)
    )

    presentation = decision.display_presentation
    assert isinstance(presentation, DisplayTiledPresentation)
    assert presentation.tile_state.payloads == {0: tile}
    assert presentation.tile_delta.upserts == {0: tile}
    assert presentation.tile_delta.active_tiles == (0,)
    assert not hasattr(presentation, "data")


def test_tiled_presentation_owns_one_explicit_shader_mapping():
    state = ViewState.from_shape((2, 2, 2)).with_image_axes(0, 1).with_montage_axis(2, columns=2, indices=(0, 1))
    geometry = DisplayGeometry(
        view_state=state,
        display_shape=(2, 4),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=0),
        montage_tile_states=("loaded", "loaded"),
    )
    mapping = ShaderMapping(component=ShaderComponent.IMAG)
    tiles = {
        index: DisplayTilePayload(
            index,
            index,
            np.ones((2, 2), dtype=np.float32),
            None,
            ("tile", index),
            shader_mapping=mapping,
        )
        for index in range(2)
    }
    tile_state = TilePresentationState(tiles)
    tile_delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=tiles,
        active_tiles=(0, 1),
        planned_tiles=(0, 1),
    )
    payload = DisplayPayload(
        image=DisplayImage(np.zeros((2, 4), dtype=np.float32)),
        geometry=geometry,
        viewport_policy=ViewportPolicy.PRESERVE,
        tile_state=tile_state,
        tile_delta=tile_delta,
    )

    presentation = decide_presentation(_input(payload, kind=CommitKind.FULL_FRAME_INITIAL)).display_presentation

    assert presentation.shader_mapping is mapping


def test_tiled_presentation_rejects_conflicting_payload_shader_mappings():
    state = ViewState.from_shape((2, 2, 2)).with_image_axes(0, 1).with_montage_axis(2, columns=2, indices=(0, 1))
    geometry = DisplayGeometry(
        view_state=state,
        display_shape=(2, 4),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=0),
        montage_tile_states=("loaded", "loaded"),
    )
    tiles = {
        0: DisplayTilePayload(0, 0, np.ones((2, 2), dtype=np.float32), None, ("tile", 0), shader_mapping=ShaderMapping(component=ShaderComponent.REAL)),
        1: DisplayTilePayload(1, 1, np.ones((2, 2), dtype=np.float32), None, ("tile", 1), shader_mapping=ShaderMapping(component=ShaderComponent.IMAG)),
    }
    tile_state = TilePresentationState(tiles)
    tile_delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=tiles,
        active_tiles=(0, 1),
        planned_tiles=(0, 1),
    )
    payload = DisplayPayload(
        image=DisplayImage(np.zeros((2, 4), dtype=np.float32)),
        geometry=geometry,
        viewport_policy=ViewportPolicy.PRESERVE,
        tile_state=tile_state,
        tile_delta=tile_delta,
    )

    with np.testing.assert_raises_regex(ValueError, "conflicting shader mappings"):
        decide_presentation(_input(payload, kind=CommitKind.FULL_FRAME_INITIAL))
