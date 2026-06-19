import numpy as np

from arrayscope.core.view_state import ViewState
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.slice_engine import DisplayImage
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.window.display_frame import (
    CommittedDisplayFrame,
    DisplayFrameKey,
    DisplayTilePayload,
    TilePresentationDelta,
    TilePresentationState,
)
from arrayscope.window.presentation import LevelSource, LevelSourceRank, decide_presentation
from arrayscope.window.render_model import (
    CommitKind,
    DisplayPayload,
    DisplayTiledPresentation,
    PresentationInput,
    RenderRequestContext,
)


def _geometry(shape=(2, 2)):
    return DisplayGeometry(view_state=ViewState.from_shape(shape), display_shape=shape)


def _context():
    return RenderRequestContext(document_key=("doc", 1), request_key=("image", 1), render_generation=1, semantic_key="levels")


def _payload(data, *, histogram_data=None):
    image = DisplayImage(np.asarray(data, dtype=np.float32), histogram_data=None if histogram_data is None else np.asarray(histogram_data, dtype=np.float32))
    return DisplayPayload(image=image, geometry=_geometry(image.data.shape[:2]), viewport_policy=ViewportPolicy.PRESERVE)


def _frame(*, levels=(10.0, 20.0), histogram_range=(0.0, 100.0)):
    data = np.zeros((2, 2), dtype=np.float32)
    return CommittedDisplayFrame(
        data=data,
        histogram_data=data.copy(),
        geometry=_geometry(),
        levels=levels,
        histogram_range=histogram_range,
        key=DisplayFrameKey(("doc", 1), ("image", 0), 1, "levels"),
    )


def _input(payload, *, previous_frame=None, force_auto=False, kind=CommitKind.FULL_NORMAL, semantic_source=None, applied_level_source=None, window_mode="relative"):
    return PresentationInput(
        payload=payload,
        context=_context(),
        previous_frame=previous_frame,
        window_mode=window_mode,
        force_auto=force_auto,
        commit_kind=kind,
        semantic_source=semantic_source,
        applied_level_source=applied_level_source,
    )


def test_normal_relative_level_reuse_uses_committed_frame():
    decision = decide_presentation(_input(_payload([[200, 300], [200, 300]]), previous_frame=_frame(levels=(25, 75), histogram_range=(0, 100))))

    assert decision.levels == (225.0, 275.0)
    assert decision.histogram_range == (200.0, 300.0)


def test_normal_absolute_level_reuse_uses_committed_frame():
    decision = decide_presentation(
        _input(
            _payload([[200, 300], [200, 300]]),
            previous_frame=_frame(levels=(25, 75), histogram_range=(0, 100)),
            window_mode="absolute",
        )
    )

    assert decision.levels == (25.0, 75.0)
    assert decision.histogram_range == (200.0, 300.0)


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


def test_progressive_montage_patch_accepts_partial_implicit_source_monotonically():
    source = LevelSource((100.0, 200.0), (100.0, 200.0), LevelSourceRank.MONTAGE_VISIBLE_SUBSET, source_count=1, expected_count=4, semantic_key="levels")
    decision = decide_presentation(
        _input(
            _payload(np.full((2, 2), 1000.0)),
            previous_frame=_frame(levels=(2.0, 8.0), histogram_range=(0.0, 10.0)),
            kind=CommitKind.PROGRESSIVE_MONTAGE_PATCH,
            semantic_source=source,
        )
    )

    assert decision.levels == (120.0, 180.0)
    assert decision.histogram_range == (100.0, 200.0)


def test_progressive_montage_patch_accepts_complete_source():
    source = LevelSource((0.0, 300.0), (0.0, 300.0), LevelSourceRank.MONTAGE_COMPLETE, source_count=4, expected_count=4, semantic_key="levels")
    decision = decide_presentation(
        _input(
            _payload(np.full((2, 2), 1000.0)),
            previous_frame=_frame(levels=(2.0, 8.0), histogram_range=(0.0, 10.0)),
            kind=CommitKind.PROGRESSIVE_MONTAGE_PATCH,
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
            kind=CommitKind.PROGRESSIVE_MONTAGE_PATCH,
            semantic_source=source,
        )
    )

    assert decision.levels == (4.7, 5.3)
    assert decision.histogram_range == (4.5, 5.5)


def test_user_locked_montage_levels_are_not_overridden_by_complete_source():
    user = LevelSource((20.0, 40.0), (0.0, 100.0), LevelSourceRank.EXPLICIT_USER, semantic_key="levels")
    complete = LevelSource((0.0, 300.0), (0.0, 300.0), LevelSourceRank.MONTAGE_COMPLETE, source_count=4, expected_count=4, semantic_key="levels")

    decision = decide_presentation(
        _input(
            _payload(np.full((2, 2), 1000.0)),
            previous_frame=_frame(levels=(2.0, 8.0), histogram_range=(0.0, 10.0)),
            kind=CommitKind.PROGRESSIVE_MONTAGE_PATCH,
            semantic_source=complete,
            applied_level_source=user,
        )
    )

    assert decision.levels == (20.0, 40.0)
    assert decision.histogram_range == (0.0, 300.0)
    assert decision.level_source_rank == int(LevelSourceRank.EXPLICIT_USER)


def test_montage_absolute_preserves_numeric_levels_while_histogram_improves():
    absolute = LevelSource((20.0, 40.0), (0.0, 100.0), LevelSourceRank.EXPLICIT_USER, semantic_key="levels")
    complete = LevelSource((0.0, 300.0), (0.0, 300.0), LevelSourceRank.MONTAGE_COMPLETE, source_count=4, expected_count=4, semantic_key="levels")

    decision = decide_presentation(
        _input(
            _payload(np.full((2, 2), 1000.0)),
            kind=CommitKind.PROGRESSIVE_MONTAGE_PATCH,
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
    payload = DisplayPayload(
        image=DisplayImage(np.zeros((2, 2), dtype=np.float32), histogram_data=np.zeros((2, 2), dtype=np.float32)),
        geometry=_geometry((2, 2)),
        viewport_policy=ViewportPolicy.PRESERVE,
        montage_dirty_tiles=(3,),
    )

    decision = decide_presentation(
        _input(
            payload,
            previous_frame=_frame(),
            kind=CommitKind.PROGRESSIVE_MONTAGE_PATCH,
        )
    )

    assert decision.display_presentation.montage_dirty_tiles == (3,)


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
        _input(payload, kind=CommitKind.FULL_MONTAGE_INITIAL)
    )

    presentation = decision.display_presentation
    assert isinstance(presentation, DisplayTiledPresentation)
    assert presentation.tile_state.payloads == {0: tile}
    assert presentation.tile_delta.upserts == {0: tile}
    assert presentation.tile_delta.active_tiles == (0,)
    assert not hasattr(presentation, "data")
