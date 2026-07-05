from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.display.backend_contract import ImageViewBackendCapabilities
from arrayscope.window.montage_backend import choose_montage_backend
from arrayscope.window.montage_payload_cache import (
    base_tile_source_id as _base_tile_source_id,
    payload_lod_matches as _payload_lod_matches,
    payload_compatible_with_tile as _payload_compatible_with_tile,
    RetainedTiledPayloadStore,
)
from arrayscope.window.montage_viewport import (
    MontageViewportPlan,
    montage_session_key,
    montage_viewport_update_delay_ms as _montage_viewport_update_delay_ms,
)



def _window_ns(**kwargs):
    ns = SimpleNamespace(**kwargs)
    ns.win = ns
    return ns


def _geometry():
    return SimpleNamespace(montage=object())


def _committed_tiled_frame(geometry, *, key):
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import CommittedDisplayFrame, DisplayTilePayload, TiledValueSource

    shape = tuple(int(value) for value in geometry.display_shape[:2])
    data = np.zeros(shape, dtype=np.float32)
    payloads = {}
    montage = geometry.montage
    for tile_number, source_index in enumerate(tuple(montage.indices)):
        state = geometry.montage_tile_states[tile_number]
        if state is not MontageTileState.LOADED:
            continue
        tile_shape = tuple(int(value) for value in montage.tile_shape[:2])
        tile_data = np.zeros(tile_shape, dtype=np.float32)
        payloads[int(tile_number)] = DisplayTilePayload(
            int(tile_number),
            int(source_index),
            tile_data,
            tile_data.copy(),
            ("frame-tile", int(tile_number), int(source_index)),
            semantic_data=tile_data,
            semantic_histogram_data=tile_data.copy(),
            source_shape=tile_shape,
        )
    return CommittedDisplayFrame(
        data=data,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogram_range=(0.0, 1.0),
        key=key,
        value_source=TiledValueSource(payloads),
    )


def test_known_montage_level_source_is_not_resampled(monkeypatch):
    from arrayscope.display.model.montage_levels import MontageLevelTracker, TileLevelStats
    import arrayscope.window.frame_renderer as frame_renderer
    from arrayscope.window.frame_renderer import FrameRenderMixin

    class Window(FrameRenderMixin):
        def __init__(self):
            self.win = self
            self._tracker = MontageLevelTracker()

        def _montage_level_tracker(self):
            return self._tracker

    key = ("levels",)
    win = Window()
    win._tracker.ensure_expected(key, (3,))
    win._tracker.update_from_stats(
        key,
        TileLevelStats(3, (0.0, 1.0), np.asarray([0.0, 1.0], dtype=np.float32), refined=False),
        aggregate=False,
    )
    calls = []
    monkeypatch.setattr(
        frame_renderer,
        "sample_tile_level_stats",
        lambda *_args, **_kwargs: calls.append("sample") or None,
    )
    rendered = SimpleNamespace(
        tile=SimpleNamespace(source_index=3),
        level_stats=None,
        level_data=None,
        histogram_data=np.arange(16, dtype=np.float32),
        image=np.arange(16, dtype=np.float32),
    )

    win._update_montage_level_bounds_from_rendered(key, rendered, expected_indices=(3,))

    assert calls == []


def test_payload_level_stats_are_bounded_and_deferred(monkeypatch):
    from arrayscope.display.model.montage_levels import MontageLevelTracker
    import arrayscope.window.frame_renderer as frame_renderer
    from arrayscope.window.frame_renderer import FrameRenderMixin

    class Window(FrameRenderMixin):
        def __init__(self):
            self.win = self
            self._tracker = MontageLevelTracker()
            self.scheduled = 0

        def _montage_level_tracker(self):
            return self._tracker

        def _schedule_montage_cached_level_stats(self, session):
            self.scheduled += 1

    calls = []
    monkeypatch.setattr(
        frame_renderer,
        "sample_tile_level_stats",
        lambda *_args, **_kwargs: calls.append("sample") or None,
    )
    rendered = {
        index: SimpleNamespace(
            tile=SimpleNamespace(source_index=index),
            level_stats=None,
            level_data=None,
            histogram_data=np.arange(64, dtype=np.float32),
            image=np.arange(64, dtype=np.float32),
        )
        for index in range(32)
    }
    session = SimpleNamespace(
        level_key=("levels", "bounded"),
        level_expected_indices=tuple(range(32)),
        rendered_tiles=rendered,
        plan=SimpleNamespace(tiles=tuple(SimpleNamespace(source_index=index) for index in range(32))),
        pending_level_tiles=deque(),
        pending_level_sources=set(),
        pending_refined_level_tiles=deque(),
        pending_refined_level_sources=set(),
        level_scan_cursor=0,
        level_scan_remaining_tiles=0,
    )
    win = Window()

    merged = win._queue_montage_level_stats_for_payloads(session, {index: object() for index in range(32)})

    assert merged == 0
    assert calls == []
    assert len(session.pending_level_tiles) <= frame_renderer.MONTAGE_LEVEL_STATS_COMMIT_BATCH
    assert session.level_scan_remaining_tiles == 32
    assert win.scheduled == 1


def test_prepared_payload_level_stats_merge_without_background_sampling(monkeypatch):
    from arrayscope.display.model.montage_levels import MontageLevelTracker, TileLevelStats
    import arrayscope.window.frame_renderer as frame_renderer
    from arrayscope.window.frame_renderer import FrameRenderMixin

    class Window(FrameRenderMixin):
        def __init__(self):
            self.win = self
            self._tracker = MontageLevelTracker()
            self.scheduled = 0

        def _montage_level_tracker(self):
            return self._tracker

        def _schedule_montage_cached_level_stats(self, session):
            self.scheduled += 1

    calls = []
    monkeypatch.setattr(
        frame_renderer,
        "sample_tile_level_stats",
        lambda *_args, **_kwargs: calls.append("sample") or None,
    )
    rendered = {
        index: SimpleNamespace(
            tile=SimpleNamespace(source_index=index),
            level_stats=TileLevelStats(index, (float(index), float(index + 1)), np.asarray([float(index)], dtype=np.float32)),
            level_data=None,
            histogram_data=None,
            image=np.asarray([float(index)], dtype=np.float32),
        )
        for index in range(4)
    }
    session = SimpleNamespace(
        level_key=("levels", "prepared"),
        level_expected_indices=tuple(range(4)),
        rendered_tiles=rendered,
        plan=SimpleNamespace(tiles=tuple(SimpleNamespace(source_index=index) for index in range(4))),
        pending_level_tiles=deque(),
        pending_level_sources=set(),
        pending_refined_level_tiles=deque(),
        pending_refined_level_sources=set(),
        level_scan_cursor=0,
        level_scan_remaining_tiles=0,
    )
    win = Window()

    merged = win._queue_montage_level_stats_for_payloads(session, {index: object() for index in range(4)})

    assert merged == 4
    assert calls == []
    assert len(session.pending_level_tiles) == 0
    assert win._tracker.summary_for(session.level_key).source_indices == frozenset(range(4))


def test_display_tile_payload_retains_prepared_level_stats_for_reuse():
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.model.montage_levels import TileLevelStats
    from arrayscope.window.frame_renderer import _rendered_tile_from_previous_payload

    stats = TileLevelStats(5, (2.0, 8.0), np.asarray([2.0, 8.0], dtype=np.float32))
    level_data = np.asarray([2.0, 8.0], dtype=np.float32)
    image = np.ones((2, 2), dtype=np.float32)
    payload = DisplayTilePayload(
        0,
        5,
        image,
        image,
        ("tile", 5),
        level_data=level_data,
        level_stats=stats,
    )
    rendered = _rendered_tile_from_previous_payload(
        SimpleNamespace(montage_index=0, source_index=5),
        payload,
    )

    assert rendered.level_data is level_data
    assert rendered.level_stats is stats


def test_pyqtgraph_auto_levels_wait_for_complete_semantic_source():
    from arrayscope.core.window_levels import LevelSourceRank
    from arrayscope.display.model.montage_levels import MontageLevelStats
    from arrayscope.window.frame_renderer import _tile_layer_auto_levels_wait_for_complete_source

    window = SimpleNamespace(img_view=SimpleNamespace(rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph")))
    window.win = window
    session = SimpleNamespace(
        pending_tiles=[],
        loading_tiles=set(),
        active_tile_requests=set(),
        pending_completed_tiles=deque(),
        pending_level_tiles=deque([object()]),
        level_scan_remaining_tiles=0,
    )
    partial = MontageLevelStats(
        bounds=(0.0, 1.0),
        source_indices=frozenset({0}),
        expected_indices=frozenset({0, 1}),
        rank=LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
    )
    complete = MontageLevelStats(
        bounds=(0.0, 1.0),
        source_indices=frozenset({0, 1}),
        expected_indices=frozenset({0, 1}),
        rank=LevelSourceRank.MONTAGE_COMPLETE,
    )
    sampled_full = MontageLevelStats(
        bounds=(0.0, 1.0),
        source_indices=frozenset({0, 1}),
        expected_indices=frozenset({0, 1}),
        rank=LevelSourceRank.MONTAGE_SAMPLED_FULL,
        refined=True,
    )

    assert _tile_layer_auto_levels_wait_for_complete_source(window, session, True, partial) is True
    assert _tile_layer_auto_levels_wait_for_complete_source(window, session, True, complete) is True
    assert _tile_layer_auto_levels_wait_for_complete_source(window, session, True, sampled_full) is False


def test_shader_auto_levels_do_not_wait_for_complete_cpu_window_source():
    from arrayscope.core.window_levels import LevelSourceRank
    from arrayscope.display.model.montage_levels import MontageLevelStats
    from arrayscope.window.frame_renderer import _tile_layer_auto_levels_wait_for_complete_source

    window = SimpleNamespace(
        img_view=SimpleNamespace(rendering_capabilities=ImageViewBackendCapabilities(name="vispy", shader_windowing=True))
    )
    window.win = window
    session = SimpleNamespace(pending_level_tiles=deque([object()]), level_scan_remaining_tiles=1)
    partial = MontageLevelStats(
        bounds=(0.0, 1.0),
        source_indices=frozenset({0}),
        expected_indices=frozenset({0, 1}),
        rank=LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
    )

    assert _tile_layer_auto_levels_wait_for_complete_source(window, session, True, partial) is False


def test_auto_small_scalar_montage_uses_tile_layer():
    decision = choose_montage_backend(_geometry(), np.zeros((64, 64), dtype=np.float32))

    assert decision.backend == "tile_layer"

    assert "tiled montage presentation" in decision.reason


def test_initial_montage_plan_uses_pending_restored_viewport_range():
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.viewport import ViewportMode
    from arrayscope.window.frame_renderer import FrameRenderMixin

    class Window(FrameRenderMixin):
        def __init__(self):
            self.win = self

    win = Window()
    win.win = win
    state = ViewState.from_shape((10, 10, 8)).with_montage_axis(2, columns=None, indices=tuple(range(8)), text=":")
    win.img_view = SimpleNamespace(
        image=None,
        viewport_controller=SimpleNamespace(
            mode=ViewportMode.USER,
            is_fit_locked=lambda: False,
            promote_near_auto=lambda _view_range: False,
            is_auto_active=lambda: False,
        ),
        graphicsView=SimpleNamespace(viewport=lambda: SimpleNamespace(size=lambda: QtCore.QSize(400, 200))),
        getView=lambda: SimpleNamespace(viewRange=lambda: ((0.0, 1.0), (0.0, 1.0))),
        rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph"),
    )
    win._pending_viewport_continuity_range = lambda: ((100.0, 120.0), (200.0, 220.0))
    win._pending_viewport_continuity_columns = lambda: 3

    viewport_plan = FrameRenderMixin._montage_viewport_plan(win, state)

    assert viewport_plan.view_range == ((100.0, 120.0), (200.0, 220.0))
    assert viewport_plan.plan.columns == 3


def test_initial_montage_plan_ignores_invalid_restored_columns():
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.viewport import ViewportMode
    from arrayscope.window.frame_renderer import FrameRenderMixin

    class Window(FrameRenderMixin):
        def __init__(self):
            self.win = self

    win = Window()
    win.win = win
    state = ViewState.from_shape((10, 10, 8)).with_montage_axis(2, columns=None, indices=tuple(range(8)), text=":")
    win.img_view = SimpleNamespace(
        image=None,
        viewport_controller=SimpleNamespace(
            mode=ViewportMode.USER,
            is_fit_locked=lambda: False,
            is_auto_active=lambda: False,
        ),
        graphicsView=SimpleNamespace(viewport=lambda: SimpleNamespace(size=lambda: QtCore.QSize(400, 200))),
        getView=lambda: SimpleNamespace(viewRange=lambda: ((0.0, 1.0), (0.0, 1.0))),
        rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph"),
    )
    win._pending_viewport_continuity_range = lambda: ((100.0, 120.0), (200.0, 220.0))
    win._pending_viewport_continuity_columns = lambda: "auto"

    viewport_plan = FrameRenderMixin._montage_viewport_plan(win, state)

    assert viewport_plan.view_range == ((100.0, 120.0), (200.0, 220.0))
    assert viewport_plan.plan.columns is not None


def test_montage_commit_reschedules_restored_roi_stats():
    from arrayscope.window.frame_renderer import FrameRenderMixin

    calls = []
    win = SimpleNamespace(
        _file_session_roi_refresh_pending=True,
        _schedule_viewport_continuity_when_ready=lambda: calls.append("viewport"),
        _schedule_file_session_roi_refresh=lambda reason: calls.append(("roi", reason)),
    )
    win.win = win

    FrameRenderMixin._notify_file_session_montage_committed(win)

    assert calls == ["viewport", ("roi", "montage-semantic-commit")]


def test_auto_large_scalar_montage_uses_tile_layer():
    data = np.zeros((1500, 1500), dtype=np.float32)

    decision = choose_montage_backend(_geometry(), data)
    slow = choose_montage_backend(_geometry(), data)

    assert decision.backend == "tile_layer"
    assert slow.backend == "tile_layer"


def test_auto_large_scalar_vispy_montage_uses_tile_layer_to_avoid_full_uploads():
    data = np.zeros((1500, 1500), dtype=np.float32)

    decision = choose_montage_backend(_geometry(), data, renderer_backend="vispy")

    assert decision.backend == "tile_layer"

    assert "tiled montage presentation" in decision.reason


def test_auto_small_scalar_vispy_montage_uses_tile_layer():
    data = np.zeros((64, 64), dtype=np.float32)

    decision = choose_montage_backend(_geometry(), data, renderer_backend="vispy")

    assert decision.backend == "tile_layer"

    assert "tiled montage presentation" in decision.reason


def test_vispy_persistent_upsert_limits_use_governed_upload_limit():
    import arrayscope.window.frame_renderer as frame_renderer

    session = SimpleNamespace()
    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(
                name="vispy",
                persistent_tile_residency=True,
                shader_windowing=True,
            )
        ),
        _viewport_interaction_active=False,
        _ui_work_decision=lambda *_args, **_kwargs: SimpleNamespace(batch_limit=11, byte_cap=2 * 1024 * 1024, budget_ms=2.0),
    )
    window.win = window

    limits = frame_renderer._persistent_tile_upsert_limits(window, session)

    assert limits["max_upserts"] == 11
    assert limits["max_upsert_bytes"] == 2 * 1024 * 1024


def test_vispy_persistent_upsert_limits_use_texture_upload_cost_without_raising_batch_limit():
    import arrayscope.window.frame_renderer as frame_renderer

    image = np.zeros((512, 512), dtype=np.float32)
    texture = np.zeros((512, 512), dtype=np.complex64)
    semantic = np.zeros((1024, 1024), dtype=np.complex64)
    payload = SimpleNamespace(
        image=image,
        texture_data=texture,
        histogram_data=None,
        semantic_data=semantic,
    )
    session = SimpleNamespace(
        rendered_tiles={0: SimpleNamespace(image=image, histogram_data=None, semantic_data=semantic, level_data=None)},
        display_tile_payloads={},
        dirty_payloads={0: None},
        pending_payload_upserts={},
    )
    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(
                name="vispy",
                persistent_tile_residency=True,
                shader_windowing=True,
            )
        ),
        _viewport_interaction_active=False,
        _ui_work_decision=lambda *_args, **_kwargs: SimpleNamespace(batch_limit=1, byte_cap=1024 * 1024, budget_ms=2.0),
    )
    window.win = window

    limits = frame_renderer._persistent_tile_upsert_limits(window, session)

    assert limits["max_upserts"] == 1
    assert limits["max_upsert_bytes"] == 1024 * 1024
    assert limits["upsert_cost_fn"](payload) == texture.nbytes


def test_pyqtgraph_tile_layer_upsert_limits_use_display_image_upload_cost():
    import arrayscope.window.frame_renderer as frame_renderer

    image = np.zeros((512, 512), dtype=np.float32)
    semantic = np.zeros((1024, 1024), dtype=np.complex64)
    payload = SimpleNamespace(
        image=image,
        texture_data=semantic,
        histogram_data=np.zeros((512, 512), dtype=np.float32),
        semantic_data=semantic,
    )
    session = SimpleNamespace(
        has_pending_level_update=lambda: True,
        has_stale_level_presentations=lambda: True,
        display_tile_payloads={},
    )
    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(
                name="pyqtgraph",
                persistent_tile_residency=False,
                shader_windowing=False,
            )
        ),
        _viewport_interaction_active=False,
        _ui_work_decision=lambda *_args, **_kwargs: SimpleNamespace(batch_limit=3, byte_cap=1024 * 1024, budget_ms=2.0),
    )
    window.win = window

    limits = frame_renderer._tile_layer_upsert_limits(window, session)

    assert limits["max_upserts"] == 3
    assert limits["max_upsert_bytes"] == 1024 * 1024
    assert limits["upsert_cost_fn"](payload) == image.nbytes


def test_tile_presentation_admission_uses_backend_cost_function():
    from arrayscope.display.montage import MontagePlan, MontageTile, RenderedTile
    from arrayscope.window.montage_session import MontageRenderSession

    tiles = tuple(
        MontageTile(
            montage_index=index,
            source_index=index,
            row=0,
            col=index,
            x0=index * 2,
            y0=0,
            width=2,
            height=2,
            view_state=None,
        )
        for index in range(2)
    )
    session = MontageRenderSession(
        session_id=1,
        key=("test",),
        render_generation=1,
        level_key=None,
        level_expected_indices=(0, 1),
        plan=MontagePlan(axis=0, tile_shape=(2, 2), grid_shape=(1, 2), columns=2, rows=1, gap=0, tiles=tiles),
        view_state=None,
        document=None,
        montage_axis=0,
        colormap_lut=None,
        viewport_shape=(2, 4),
        view_range=None,
        output_dtype=np.dtype("float32"),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )
    for index in range(2):
        image = np.zeros((2, 2), dtype=np.float32)
        semantic = np.zeros((64, 64), dtype=np.float32)
        session.rendered_tiles[index] = RenderedTile(
            tile=tiles[index],
            image=image,
            histogram_data=None,
            eval_ms=0.0,
            slab_shape=image.shape,
            slab_nbytes=image.nbytes,
            semantic_data=semantic,
        )
        session.dirty_payloads[index] = None

    state, delta = session.build_tile_presentation(
        {},
        max_upserts=2,
        max_upsert_bytes=2 * np.zeros((2, 2), dtype=np.float32).nbytes,
        upsert_cost_fn=lambda payload: np.asarray(payload.texture_data).nbytes,
    )

    assert tuple(state.active_payloads(delta)) == (0, 1)


def test_tile_presentation_limits_do_not_hide_acknowledged_resident_tiles():
    from arrayscope.display.montage import MontagePlan, MontageTile, RenderedTile
    from arrayscope.window.montage_session import MontageRenderSession

    tiles = tuple(
        MontageTile(
            montage_index=index,
            source_index=index,
            row=0,
            col=index,
            x0=index * 2,
            y0=0,
            width=2,
            height=2,
            view_state=None,
        )
        for index in range(4)
    )
    session = MontageRenderSession(
        session_id=1,
        key=("test",),
        render_generation=1,
        level_key=None,
        level_expected_indices=tuple(range(4)),
        plan=MontagePlan(axis=0, tile_shape=(2, 2), grid_shape=(1, 4), columns=4, rows=1, gap=0, tiles=tiles),
        view_state=None,
        document=None,
        montage_axis=0,
        colormap_lut=None,
        viewport_shape=(2, 8),
        view_range=None,
        output_dtype=np.dtype("float32"),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )
    for index, tile in enumerate(tiles):
        image = np.full((2, 2), index, dtype=np.float32)
        session.rendered_tiles[index] = RenderedTile(
            tile=tile,
            image=image,
            histogram_data=image,
            eval_ms=0.0,
            slab_shape=image.shape,
            slab_nbytes=image.nbytes,
        )
        session.dirty_payloads[index] = None

    state, delta = session.build_tile_presentation({})
    session.tile_presentation_state = state
    session.presented_tiles = {0}
    session.visible_tiles = tiles

    _state, delta = session.build_tile_presentation(
        {},
        max_upserts=1,
        max_upsert_bytes=1,
        upsert_cost_fn=lambda _payload: 1024 * 1024,
    )

    assert delta.upserts == {}
    assert delta.active_tiles == (0, 1, 2, 3)


def test_auto_policy_uses_renderer_backend_name():
    data = np.zeros((1500, 1500), dtype=np.float32)
    decision = choose_montage_backend(
        _geometry(),
        data,
        renderer_backend="future-gpu-backend",
    )

    assert decision.backend == "tile_layer"
    assert "future-gpu-backend" in decision.reason


def test_montage_policy_is_always_tiled():
    data = np.zeros((64, 64), dtype=np.float32)

    decision = choose_montage_backend(
        _geometry(),
        data,
        renderer_backend="future-gpu-backend",
    )

    assert decision.backend == "tile_layer"
    assert "tiled montage presentation" in decision.reason


def test_auto_preserves_vispy_tile_layer_mode():
    data = np.zeros((64, 64), dtype=np.float32)

    decision = choose_montage_backend(_geometry(), data)

    assert decision.backend == "tile_layer"
    assert "tiled montage presentation" in decision.reason


def test_interactive_montage_commit_is_timer_coalesced(qt_app, monkeypatch):
    from pyqtgraph.Qt import QtCore, QtWidgets
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window.frame_renderer import FrameRenderMixin
    from arrayscope.window.montage_session import MontageRenderSession

    class Window(QtCore.QObject, FrameRenderMixin):
        def __init__(self):
            super().__init__()
            self.win = self
            self.view_state = ViewState.from_shape((2, 2, 1)).with_montage_axis(2, indices=(0,), text=":")
            self._commits = 0

        def _commit_montage_session_presentation(self, session, *, force=False):
            self._commits += 1

    win = Window()
    plan = make_montage_plan(win.view_state, axis=2, indices=(0,), tile_shape=(2, 2), columns=1)
    session = MontageRenderSession(
        session_id=1,
        key=("session",),
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=(0,),
        plan=plan,
        view_state=win.view_state,
        document=object(),
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(2, 2),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )
    session.display_committed = True
    win._montage_session = session
    win._viewport_interaction_active = True

    win._schedule_montage_presentation_commit(session, force=True)

    assert win._commits == 0
    assert session.final_commit_pending is True
    assert win._montage_commit_timer.isActive()


def test_interactive_viewport_prunes_stale_montage_tile_work(qt_app):
    from pyqtgraph.Qt import QtCore
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import MontageTileState, make_montage_plan
    from arrayscope.window.frame_renderer import FrameRenderMixin
    from arrayscope.window.montage_session import MontageRenderSession

    class Controller:
        def __init__(self):
            self.groups = []

        def clear_group(self, group):
            self.groups.append(group)

    class Window(QtCore.QObject, FrameRenderMixin):
        def __init__(self):
            super().__init__()
            self.win = self

    state = ViewState.from_shape((2, 2, 8)).with_montage_axis(2, indices=tuple(range(8)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(8)), tile_shape=(2, 2), columns=8, gap=1)
    controller = Controller()
    session = MontageRenderSession(
        session_id=7,
        key=("session",),
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=tuple(range(8)),
        plan=plan,
        view_state=state,
        document=object(),
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(2, 2),
        view_range=((0.0, 2.0), (0.0, 2.0)),
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=(plan.tiles[0],),
        rendered_tiles={},
        loading_tiles={7},
        skipped_tiles=set(),
        pending_tiles=[plan.tiles[1], plan.tiles[7]],
    )
    session.active_tile_requests.add(7)
    session.tile_states = [MontageTileState.UNLOADED for _tile in plan.tiles]
    session.tile_states[7] = MontageTileState.LOADING
    win = Window()
    win._montage_session = session
    win.view_state = state
    win.montage_tile_evaluation_controller = controller
    win._viewport_interaction_active = True

    win._prune_stale_montage_tile_work(session)

    assert [int(tile.montage_index) for tile in session.pending_tiles] == [1]
    assert 7 in session.loading_tiles
    assert 7 in session.active_tile_requests
    assert session.tile_states[7] == MontageTileState.LOADING
    assert controller.groups == []


def test_interactive_viewport_expansion_chunks_cached_tile_resolution(qt_app):
    from pyqtgraph.Qt import QtCore
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.evaluator import _document_key
    from arrayscope.operations.pipeline import ArrayDocument
    from arrayscope.window.frame_renderer import FrameRenderMixin
    from arrayscope.window.montage_session import MontageRenderSession

    class Window(QtCore.QObject, FrameRenderMixin):
        def __init__(self, document, state, viewport_plan):
            super().__init__()
            self.win = self
            self.document = document
            self.view_state = state
            self._viewport_plan = viewport_plan
            self._viewport_interaction_active = True
            self._montage_viewport_addition_batch_size = 3
            self.resolved_batches = []
            self.tile_schedules = 0
            self.commits = 0
            self.img_view = SimpleNamespace(
                rendering_capabilities=ImageViewBackendCapabilities(
                    name="vispy",
                    persistent_tile_residency=True,
                    shader_windowing=True,
                ),
                montageDisplayMode=lambda: "vispy_tile_layer",
            )

        def _montage_viewport_plan(self, view_state, *, view_range=None):
            return self._viewport_plan

        def _evaluation_colormap_lut(self, view_state, *, shader_display=None):
            return None

        def _resolve_montage_tiles_from_cache(self, tiles, **_kwargs):
            batch = tuple(tiles)
            self.resolved_batches.append(tuple(int(tile.montage_index) for tile in batch))
            return (), batch

        def _schedule_montage_presentation_commit(self, session, *, force=False):
            self.commits += 1

        def _schedule_montage_tiles(self, session):
            self.tile_schedules += 1

    document = ArrayDocument(np.zeros((2, 2, 10), dtype=np.float32))
    state = ViewState.from_shape(document.current_shape).with_montage_axis(2, columns=10, indices=tuple(range(10)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(10)), tile_shape=(2, 2), columns=10)
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(10)),
        viewport_shape=(4, 40),
        tile_shape=(2, 2),
        plan=plan,
        view_range=((-1.0, 40.0), (-1.0, 4.0)),
        shader_display=True,
        persistent_tile_residency=True,
    )
    session = MontageRenderSession(
        session_id=11,
        key=montage_session_key(_document_key(document), state, viewport_plan, None),
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=tuple(range(10)),
        plan=plan,
        view_state=state,
        document=document,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(4, 40),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=(),
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )
    win = Window(document, state, viewport_plan)
    win._montage_session = session

    assert win._try_update_montage_viewport_only() is True

    assert len(win.resolved_batches) == 3
    resolved = [tile for batch in win.resolved_batches for tile in batch]
    assert len(resolved) == 3
    pending = [int(tile.montage_index) for tile in session.pending_tiles]
    assert pending == resolved
    assert pending[0] == 6
    assert session.loading_tiles == set()
    assert win.tile_schedules == 0
    assert win._montage_viewport_update_pending is True
    assert win._montage_viewport_continue_immediately is True
    assert win._last_montage_viewport_deferred_additions == 7


def test_quiet_viewport_update_schedules_deferred_missing_tiles(qt_app, monkeypatch):
    from pyqtgraph.Qt import QtCore, QtWidgets
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.evaluator import _document_key
    from arrayscope.operations.pipeline import ArrayDocument
    from arrayscope.window.frame_renderer import FrameRenderMixin
    from arrayscope.window.montage_session import MontageRenderSession

    class Window(QtCore.QObject, FrameRenderMixin):
        def __init__(self, document, state, viewport_plan):
            super().__init__()
            self.win = self
            self.document = document
            self.view_state = state
            self._viewport_plan = viewport_plan
            self.tile_schedules = 0
            self.img_view = SimpleNamespace(
                rendering_capabilities=ImageViewBackendCapabilities(
                    name="vispy",
                    persistent_tile_residency=True,
                    shader_windowing=True,
                ),
                montageDisplayMode=lambda: "vispy_tile_layer",
            )

        def _montage_viewport_plan(self, view_state, *, view_range=None):
            return self._viewport_plan

        def _evaluation_colormap_lut(self, view_state, *, shader_display=None):
            return None

        def _schedule_montage_tiles(self, session):
            self.tile_schedules += 1

        def _schedule_montage_presentation_commit(self, session, *, force=False):
            pass

    document = ArrayDocument(np.zeros((2, 2, 4), dtype=np.float32))
    state = ViewState.from_shape(document.current_shape).with_montage_axis(2, columns=4, indices=tuple(range(4)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(4)), tile_shape=(2, 2), columns=4)
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(4)),
        viewport_shape=(4, 16),
        tile_shape=(2, 2),
        plan=plan,
        view_range=((-1.0, 16.0), (-1.0, 4.0)),
        shader_display=True,
        persistent_tile_residency=True,
    )
    session = MontageRenderSession(
        session_id=12,
        key=montage_session_key(_document_key(document), state, viewport_plan, None),
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=tuple(range(4)),
        plan=plan,
        view_state=state,
        document=document,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(4, 16),
        view_range=viewport_plan.view_range,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles={0, 1, 2, 3},
        skipped_tiles=set(),
        pending_tiles=list(plan.tiles),
    )
    win = Window(document, state, viewport_plan)
    win._montage_session = session
    win._viewport_interaction_active = False
    monkeypatch.setattr(QtWidgets.QApplication, "mouseButtons", lambda: QtCore.Qt.MouseButton.NoButton)

    assert win._try_update_montage_viewport_only() is True

    assert win.tile_schedules == 1


def test_resize_retarget_commits_presentation_geometry_immediately(qt_app):
    from pyqtgraph.Qt import QtCore
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.backend_contract import ImageViewBackendCapabilities
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.pipeline import ArrayDocument
    from arrayscope.window.frame_renderer import FrameRenderMixin
    from arrayscope.window.montage_viewport import MontageViewportPlan

    class Planner:
        def plan(self, **_kwargs):
            return object()

    class Session:
        session_id = 1
        key = ("session",)
        window_mode = "relative"
        user_levels_override = None
        force_auto = False

        def __init__(self):
            self.retargeted = False

        def retarget_viewport(self, **_kwargs):
            self.retargeted = True
            return (), True

        def refresh_lod_for_viewport(self):
            return False

        pending_lod_requests = ()

    class Window(QtCore.QObject, FrameRenderMixin):
        def __init__(self):
            super().__init__()
            self.win = self
            self.view_state = ViewState.from_shape((4, 4, 4)).with_montage_axis(2, indices=tuple(range(4)), text=":")
            self.document = ArrayDocument(np.zeros((4, 4, 4), dtype=np.float32))
            self.img_view = SimpleNamespace(
                rendering_capabilities=ImageViewBackendCapabilities(
                    name="pyqtgraph",
                ),
                montageDisplayMode=lambda: "tile_layer",
            )
            self._montage_session = Session()
            self.commits = 0

        def _is_current_montage_session(self, session_id, key):
            return session_id == 1 and key == ("session",)

        def _montage_frame_planner(self):
            return Planner()

        def _commit_montage_session_presentation(self, session, *, force=False):
            assert session is self._montage_session
            assert force is True
            self.commits += 1

        def _schedule_montage_presentation_commit(self, session, *, force=False):
            raise AssertionError("resize geometry should commit immediately when no commit is active")

    state = ViewState.from_shape((4, 4, 4)).with_montage_axis(2, indices=tuple(range(4)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(4)), tile_shape=(4, 4), columns=2)
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(4)),
        viewport_shape=(40, 80),
        tile_shape=(4, 4),
        plan=plan,
        view_range=((0.0, 9.0), (0.0, 9.0)),
        shader_display=False,
        persistent_tile_residency=False,
    )
    win = Window()

    assert win._retarget_montage_resize_payloads(viewport_plan) is True
    assert win._montage_session.retargeted is True
    assert win.commits == 1


def test_nonpersistent_tile_layer_viewport_update_preserves_level_target(qt_app):
    from pyqtgraph.Qt import QtCore
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.evaluator import _document_key
    from arrayscope.operations.pipeline import ArrayDocument
    from arrayscope.window.frame_renderer import FrameRenderMixin
    from arrayscope.window.montage_session import MontageRenderSession

    class Window(QtCore.QObject, FrameRenderMixin):
        def __init__(self, document, state, viewport_plan):
            super().__init__()
            self.win = self
            self.document = document
            self.view_state = state
            self._viewport_plan = viewport_plan
            self.tile_schedules = 0
            self.commits = 0
            self.img_view = SimpleNamespace(
                rendering_capabilities=ImageViewBackendCapabilities(
                    name="pyqtgraph",
                    persistent_tile_residency=False,
                    shader_windowing=False,
                ),
                montageDisplayMode=lambda: "tile_layer",
            )

        def _montage_viewport_plan(self, view_state, *, view_range=None):
            return self._viewport_plan

        def _evaluation_colormap_lut(self, view_state, *, shader_display=None):
            return None

        def _schedule_montage_tiles(self, session):
            self.tile_schedules += 1

        def _schedule_montage_presentation_commit(self, session, *, force=False):
            self.commits += 1

    document = ArrayDocument(np.zeros((2, 2, 4), dtype=np.float32))
    state = ViewState.from_shape(document.current_shape).with_montage_axis(2, columns=4, indices=tuple(range(4)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(4)), tile_shape=(2, 2), columns=4)
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(4)),
        viewport_shape=(4, 16),
        tile_shape=(2, 2),
        plan=plan,
        view_range=((-1.0, 16.0), (-1.0, 4.0)),
        shader_display=False,
        persistent_tile_residency=False,
    )
    session = MontageRenderSession(
        session_id=13,
        key=montage_session_key(_document_key(document), state, viewport_plan, None),
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=tuple(range(4)),
        plan=plan,
        view_state=state,
        document=document,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(4, 16),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={int(tile.montage_index): SimpleNamespace(tile=tile) for tile in plan.tiles},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )
    session.level_generation.target_levels = (2.0, 4.0)
    session.set_level_update_pending(True)
    win = Window(document, state, viewport_plan)
    win._montage_session = session
    win._viewport_interaction_active = False

    assert win._try_update_montage_viewport_only() is True

    assert win._montage_session is session
    assert session.level_generation.target_levels == (2.0, 4.0)


def test_hover_priority_retarget_timer_changes_next_pending_tile(qt_app):
    from pyqtgraph.Qt import QtCore
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window.frame_renderer import FrameRenderMixin
    from arrayscope.window.montage_session import MontageRenderSession

    class Window(QtCore.QObject, FrameRenderMixin):
        def __init__(self, state, viewport_plan):
            super().__init__()
            self.win = self
            self.view_state = state
            self._viewport_plan = viewport_plan
            self.scheduled = []
            self.img_view = SimpleNamespace(rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph"))

        def _montage_viewport_plan(self, view_state, *, view_range=None):
            return self._viewport_plan

        def _is_current_montage_session(self, session_id, key):
            return True

        def _schedule_montage_tiles(self, session):
            tile = session.next_tile()
            if tile is not None:
                self.scheduled.append(int(tile.montage_index))

    state = ViewState.from_shape((2, 2, 4)).with_montage_axis(2, columns=4, indices=tuple(range(4)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(4)), tile_shape=(2, 2), columns=4)
    viewport_plan = SimpleNamespace(priority_focus=(10.0, 1.0))
    session = MontageRenderSession(
        session_id=12,
        key="key",
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=tuple(range(4)),
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(4, 16),
        view_range=((0.0, 12.0), (0.0, 4.0)),
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=list(plan.tiles),
    )
    win = Window(state, viewport_plan)
    win._montage_session = session

    win._run_montage_priority_retarget()

    assert win.scheduled == [3]


def test_tiled_commit_syncs_hover_geometry_after_backend_ack(qt_app):
    from dataclasses import replace
    from pyqtgraph.Qt import QtCore
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayFrameKey
    from arrayscope.window.frame_renderer import FrameRenderMixin

    class Window(QtCore.QObject, FrameRenderMixin):
        def __init__(self):
            super().__init__()
            self.win = self

        def _set_committed_display_frame(self, frame):
            self._committed_display_frame = frame

    state = ViewState.from_shape((2, 2, 1)).with_montage_axis(2, columns=1, indices=(0,), text=":")
    loading = DisplayGeometry(
        view_state=state,
        display_shape=(2, 2),
        montage=MontageGeometry(indices=(0,), tile_shape=(2, 2), columns=1, rows=1, gap=0),
        montage_tile_states=(MontageTileState.LOADING,),
    )
    loaded = replace(loading, montage_tile_states=(MontageTileState.LOADED,))
    frame = _committed_tiled_frame(loading, key=DisplayFrameKey(("doc",), ("view",), 1))
    win = Window()
    win.display_geometry = loading
    win._committed_display_frame = frame

    win._sync_committed_montage_geometry(loaded)

    assert win.display_geometry.montage_tile_states == (MontageTileState.LOADED,)
    assert win._committed_display_frame.geometry.montage_tile_states == (MontageTileState.LOADED,)
    assert win._committed_display_frame.scene.geometry == loaded


def test_loading_only_tiled_commit_does_not_mutate_committed_semantic_geometry(qt_app):
    from dataclasses import replace
    from pyqtgraph.Qt import QtCore
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayFrameKey
    from arrayscope.window.frame_renderer import FrameRenderMixin

    class Window(QtCore.QObject, FrameRenderMixin):
        def __init__(self):
            super().__init__()
            self.win = self

        def _set_committed_display_frame(self, frame):
            self._committed_display_frame = frame

    first_state = ViewState.from_shape((2, 2, 4)).with_montage_axis(2, columns=2, indices=(0, 1), text="0:2")
    second_state = first_state.with_axis_range(2, indices=(2, 3), text="2:4")
    committed = DisplayGeometry(
        view_state=first_state,
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    loading = DisplayGeometry(
        view_state=second_state,
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(2, 3), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADING, MontageTileState.LOADING),
    )
    frame = _committed_tiled_frame(committed, key=DisplayFrameKey(("doc",), ("view",), 1))
    win = Window()
    win.display_geometry = committed
    win._committed_display_frame = frame

    win._sync_committed_montage_geometry(loading, semantic_commit=False)

    assert win.display_geometry == loading
    assert win._committed_display_frame is frame
    assert win._committed_display_frame.geometry == committed


def test_persistent_tile_residency_defers_tile_discovery_behind_camera_updates():
    from arrayscope.window.montage_viewport import montage_viewport_retarget_policy
    from arrayscope.window.frame_renderer import _persistent_gpu_tile_residency_backend, _persistent_tile_residency_backend

    capabilities = ImageViewBackendCapabilities(
        name="vispy",
        persistent_tile_residency=True,
    )
    persistent_nonvispy = ImageViewBackendCapabilities(
        name="future-backend",
        persistent_tile_residency=True,
        shader_windowing=True,
    )
    direct_nonpersistent = ImageViewBackendCapabilities(
        name="pyqtgraph",
        persistent_tile_residency=False,
    )
    persistent_without_shader = ImageViewBackendCapabilities(
        name="resident-cpu-backend",
        persistent_tile_residency=True,
        shader_windowing=False,
    )
    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=capabilities,
            montageDisplayMode=lambda: "vispy_tile_layer",
        )
    )
    window.win = window
    fallback = _window_ns(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph"),
            montageDisplayMode=lambda: "tile_layer",
        )
    )

    assert _montage_viewport_update_delay_ms(window) == 90
    assert _montage_viewport_update_delay_ms(fallback) == 120
    assert montage_viewport_retarget_policy(capabilities, "vispy_tile_layer").coverage_margin_tiles == 1
    assert (
        _persistent_tile_residency_backend(
            _window_ns(img_view=SimpleNamespace(rendering_capabilities=persistent_nonvispy)),
            SimpleNamespace(),
        )
        is True
    )
    assert (
        _persistent_tile_residency_backend(
            _window_ns(img_view=SimpleNamespace(rendering_capabilities=persistent_without_shader)),
            SimpleNamespace(),
        )
        is True
    )
    assert (
        _persistent_gpu_tile_residency_backend(
            _window_ns(img_view=SimpleNamespace(rendering_capabilities=persistent_without_shader)),
            SimpleNamespace(),
        )
        is False
    )
    assert montage_viewport_retarget_policy(direct_nonpersistent, "tile_layer").enabled is True
    assert montage_viewport_retarget_policy(direct_nonpersistent, "tile_layer").coverage_margin_tiles == 0
    assert montage_viewport_retarget_policy(direct_nonpersistent, "tile_layer").enabled is True


def test_retained_payload_store_is_keyed_by_semantic_source_identity():
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.shader_mapping import TexturePlaneKind

    texture = np.ones((2, 2), dtype=np.complex64)
    payload = DisplayTilePayload(
        0,
        0,
        texture,
        np.ones((2, 2), dtype=np.float32),
        (("montage_tile", "doc", 2), "texture_kind", "complex_rg32f", "shader", None, "lod", 4, 2, 1),
        texture_data=texture,
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=texture,
    )
    other = DisplayTilePayload(1, 1, np.ones((2, 2), dtype=np.float32), None, ("plain", 1))

    store = RetainedTiledPayloadStore()
    store.remember_acknowledged({0: payload, 1: other}, limit=8)
    cache = store.payloads_by_base_source()

    assert _base_tile_source_id(payload.source_id) == ("montage_tile", "doc", 2)
    assert cache[("montage_tile", "doc", 2)] is payload
    assert cache[("plain", 1)] is other


def test_retained_payload_store_is_bounded_before_large_insert_batches():
    from arrayscope.display.model.frame import DisplayTilePayload

    store = RetainedTiledPayloadStore(limit=2)
    payloads = {
        index: DisplayTilePayload(
            index,
            index,
            np.ones((2, 2), dtype=np.float32),
            None,
            ("payload", index),
        )
        for index in range(5)
    }

    store.remember_acknowledged(payloads)

    retained = store.payloads_by_base_source()
    assert len(retained) == 2
    assert set(retained) == {("payload", 3), ("payload", 4)}


def test_retained_payload_store_reads_mapping_values_without_copying():
    from arrayscope.display.model.frame import DisplayTilePayload

    class ValuesOnlyPayloads:
        def __init__(self, values):
            self._values = values

        def values(self):
            return self._values

        def __iter__(self):
            raise AssertionError("remember_acknowledged must not copy the whole payload mapping")

    payload = DisplayTilePayload(
        0,
        0,
        np.ones((2, 2), dtype=np.float32),
        None,
        ("payload", 0),
    )
    store = RetainedTiledPayloadStore(limit=2)

    store.remember_acknowledged(ValuesOnlyPayloads((payload,)))

    assert store.payloads_by_base_source()[("payload", 0)] is payload


def test_recent_payload_cache_requires_matching_lod_factor():
    lod4 = SimpleNamespace(factor=4)
    lod1 = SimpleNamespace(factor=1)

    assert _payload_lod_matches(SimpleNamespace(lod=lod4), 4)
    assert not _payload_lod_matches(SimpleNamespace(lod=lod1), 4)


def test_montage_cache_resolver_accepts_single_display_image_cache_hit():
    from arrayscope.display.slice_engine import DisplayImage
    from arrayscope.window.frame_renderer import FrameRenderMixin

    image = np.arange(4, dtype=np.float32).reshape(2, 2)
    cached = DisplayImage(image, histogram_data=image.copy())
    tile = SimpleNamespace(view_state=object(), source_index=7, montage_index=0)

    class _Evaluator:
        def cached_montage_tile(self, *_args, **_kwargs):
            return cached

        def montage_tile_key_batch(self, **_kwargs):
            return lambda _tile_state: ("stub-key",)

        def cached_montage_tile_by_key(self, _key):
            return cached

    class _Window(FrameRenderMixin):
        def __init__(self):
            self.win = self

    win = _Window()
    win.operation_evaluator = _Evaluator()
    win._committed_display_frame = None

    cached_tiles, missing_tiles = win._resolve_montage_tiles_from_cache(
        (tile,),
        document=object(),
        axis=2,
        colormap_lut=None,
        shader_display=True,
    )

    assert missing_tiles == []
    assert len(cached_tiles) == 1
    assert cached_tiles[0].tile is tile
    np.testing.assert_array_equal(cached_tiles[0].image, image)
    np.testing.assert_array_equal(cached_tiles[0].histogram_data, image)


def test_montage_cache_resolver_uses_retained_payloads_when_current_frame_is_single():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.window.frame_renderer import FrameRenderMixin

    image = np.arange(4, dtype=np.float32).reshape(2, 2)
    histogram = image.copy()
    tile = SimpleNamespace(
        view_state=ViewState.from_shape((2, 2, 8)),
        source_index=7,
        montage_index=3,
    )
    tile_key = ("montage-tile-key", 7)
    payload = DisplayTilePayload(
        3,
        7,
        image,
        histogram,
        tile_key,
        semantic_data=image,
        semantic_histogram_data=histogram,
    )

    class _Evaluator:
        def cached_montage_tile(self, *_args, **_kwargs):
            return None

        def montage_tile_key(self, *_args, **_kwargs):
            return tile_key

        def montage_tile_key_batch(self, **_kwargs):
            return lambda _tile_state: tile_key

        def cached_montage_tile_by_key(self, _key):
            return None

    class _Window(FrameRenderMixin):
        def __init__(self):
            self.win = self

    win = _Window()
    win.operation_evaluator = _Evaluator()
    win._committed_display_frame = None
    win._retained_tiled_payload_store().remember_acknowledged({3: payload})

    cached_tiles, missing_tiles = win._resolve_montage_tiles_from_cache(
        (tile,),
        document=object(),
        axis=2,
        colormap_lut=None,
        shader_display=True,
    )

    assert missing_tiles == []
    assert len(cached_tiles) == 1
    assert cached_tiles[0].tile is tile
    np.testing.assert_array_equal(cached_tiles[0].image, image)
    np.testing.assert_array_equal(cached_tiles[0].histogram_data, histogram)


def test_previous_complex_shader_payload_must_carry_complex_texture():
    from arrayscope.core.view_state import ChannelMode, ViewState
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.shader_mapping import TexturePlaneKind

    state = ViewState.from_shape((2, 2, 4)).with_channel(ChannelMode.COMPLEX)
    stale_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    payload = DisplayTilePayload(
        0,
        0,
        stale_rgb,
        np.zeros((2, 2), dtype=np.float32),
        ("source", 0),
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=stale_rgb,
    )

    assert not _payload_compatible_with_tile(payload, state, shader_display=True)


def test_previous_complex_shader_payload_accepts_complex_texture():
    from arrayscope.core.view_state import ChannelMode, ViewState
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.shader_mapping import TexturePlaneKind

    state = ViewState.from_shape((2, 2, 4)).with_channel(ChannelMode.COMPLEX)
    texture = np.ones((2, 2), dtype=np.complex64)
    payload = DisplayTilePayload(
        0,
        0,
        texture,
        np.ones((2, 2), dtype=np.float32),
        ("source", 0),
        texture_data=texture,
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=texture,
    )

    assert _payload_compatible_with_tile(payload, state, shader_display=True)


def test_tiled_payload_source_id_changes_when_texture_content_changes():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import RenderedTile, make_montage_plan
    from arrayscope.window.montage_session import MontageRenderSession

    state = ViewState.from_shape((2, 2, 1)).with_montage_axis(2, columns=1, indices=(0,), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0,), tile_shape=(2, 2), columns=1)
    tile = plan.tiles[0]

    def rendered(value):
        image = np.full((2, 2), float(value), dtype=np.float32)
        return RenderedTile(
            tile=tile,
            image=image,
            histogram_data=image.copy(),
            eval_ms=0.0,
            slab_shape=image.shape,
            slab_nbytes=int(image.nbytes),
        )

    session = MontageRenderSession(
        session_id=1,
        key=("session",),
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=(0,),
        plan=plan,
        view_state=state,
        document=object(),
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(2, 2),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={0: rendered(1.0)},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )

    first = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]
    session.rendered_tiles[0] = rendered(2.0)
    second = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]

    assert _base_tile_source_id(first.source_id) == ("tile", 0)
    assert _base_tile_source_id(second.source_id) == ("tile", 0)
    assert first.source_id != second.source_id


def test_auto_large_rgb_montage_uses_tile_layer():
    data = np.zeros((1500, 1500, 3), dtype=np.uint8)

    decision = choose_montage_backend(_geometry(), data)

    assert decision.backend == "tile_layer"



def test_stage_wait_release_falls_back_to_direct_tile_evaluation():
    pytest.importorskip("pyqtgraph")
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.window.frame_renderer import FrameRenderMixin

    class _Window(FrameRenderMixin):
        def __init__(self):
            self.win = self

    class _Session:
        def __init__(self):
            self.stage_fan_in = StageFanInState(
                active_requests={"stage-key"},
                attached_requests={"stage-key"},
                waiting_tiles={"stage-key": [SimpleNamespace(montage_index=3)]},
                tile_stage_keys={3: "stage-key"},
            )
            self.rendered_tiles = {}
            self.skipped_tiles = set()
            self.pending_tiles = []
            self.loading_tiles = set()

        def mark_loading(self, tile):
            self.loading_tiles.add(int(tile.montage_index))

    session = _Session()

    _Window()._release_stage_waiting_tiles_to_direct(session, "stage-key")

    assert session.stage_fan_in.active_requests == set()
    assert session.stage_fan_in.attached_requests == set()
    assert session.stage_fan_in.waiting_tiles == {}
    assert session.stage_fan_in.tile_stage_keys == {}
    assert [tile.montage_index for tile in session.pending_tiles] == [3]
    assert session.loading_tiles == {3}


def test_stage_activation_uses_bounded_batches_without_losing_waiting_tiles():
    pytest.importorskip("pyqtgraph")
    from arrayscope.core.gui_callback_budget import GuiCallbackBudget
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.window.frame_renderer import FrameRenderMixin

    class _Window(FrameRenderMixin):
        def __init__(self):
            self.win = self

    class _Session:
        def __init__(self):
            self.stage_fan_in = StageFanInState(
                waiting_tiles={"stage-key": [SimpleNamespace(montage_index=index) for index in range(5)]},
                values={"stage-key": "stage-value"},
            )
            self.rendered_tiles = {}
            self.skipped_tiles = set()
            self.pending_tiles = []
            self.loading_tiles = set()

        def mark_loading(self, tile):
            self.loading_tiles.add(int(tile.montage_index))

    session = _Session()
    budget = GuiCallbackBudget("montage_stage_wait", item_cap=3)

    _Window()._activate_montage_stage_value(session, "stage-key", "stage-value", budget=budget)

    assert [tile.montage_index for tile in session.pending_tiles] == [0, 1, 2]
    assert [tile.montage_index for tile in session.stage_fan_in.waiting_tiles["stage-key"]] == [3, 4]
    assert budget.processed_items == 3


def test_ready_display_commit_refreshes_stale_commit_token_at_source(qt_app):
    pytest.importorskip("pyqtgraph")
    from pyqtgraph.Qt import QtCore
    from arrayscope.window.frame_renderer import FrameRenderMixin, _montage_work_token

    class _Window(QtCore.QObject, FrameRenderMixin):
        def __init__(self):
            super().__init__()
            self.win = self
            self.img_view = SimpleNamespace(rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph"))

        def _is_current_montage_session(self, session_id, key):
            return True

        def _flush_montage_presentation_commit(self):
            return None

    session = SimpleNamespace(
        session_id=1,
        key="session",
        render_generation=2,
        payload_revision=3,
        level_revision=4,
        viewport_revision=5,
        dirty_tiles=[0],
        dirty_rects=[],
        dirty_payloads={},
        pending_payload_upserts={},
        pending_removals=set(),
        display_committed=False,
        final_commit_pending=False,
        flush_pending=False,
        has_pending_level_update=lambda: False,
        has_stale_level_presentations=lambda: False,
    )
    win = _Window()
    win._montage_session = session
    win._montage_commit_token = ("commit", "stale")

    win._schedule_montage_ready_display_commit(session)

    assert win._montage_commit_token == _montage_work_token(session, "commit")
    assert session.final_commit_pending is True
    assert session.flush_pending is True


def test_montage_commit_token_ignores_viewport_only_revision_changes():
    from arrayscope.window.frame_renderer import _montage_work_token

    session = SimpleNamespace(
        session_id=1,
        key="session",
        render_generation=2,
        payload_revision=3,
        level_revision=4,
        viewport_revision=5,
    )
    commit_token = _montage_work_token(session, "commit")
    priority_token = _montage_work_token(session, "priority_retarget")

    session.viewport_revision += 1

    assert _montage_work_token(session, "commit") == commit_token
    assert _montage_work_token(session, "priority_retarget") != priority_token


def test_interactive_cache_hit_requires_committed_semantic_montage_mapping():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.evaluator import _document_key
    from arrayscope.operations.pipeline import ArrayDocument
    from arrayscope.window.frame_renderer import FrameRenderMixin

    class Evaluator:
        def cached_montage_tile(self, *_args, **_kwargs):
            return object()

    class Window(FrameRenderMixin):
        def __init__(self, document, state, viewport_plan):
            self.win = self
            self.document = document
            self.view_state = state
            self._viewport_plan = viewport_plan
            self.operation_evaluator = Evaluator()
            self.img_view = SimpleNamespace(
                rendering_capabilities=ImageViewBackendCapabilities(
                    name="pyqtgraph",
                    persistent_tile_residency=True,
                    shader_windowing=False,
                )
            )

        def _montage_viewport_plan(self, view_state, *, view_range=None):
            return self._viewport_plan

        def _evaluation_colormap_lut(self, view_state, *, shader_display=None):
            return None

    document = ArrayDocument(np.zeros((2, 2, 6), dtype=np.float32))
    old_state = ViewState.from_shape(document.current_shape).with_montage_axis(2, columns=2, indices=(0, 1), text="0:2")
    new_state = ViewState.from_shape(document.current_shape).with_montage_axis(2, columns=2, indices=(1, 2), text="1:3")
    old_plan = make_montage_plan(old_state, axis=2, indices=(0, 1), tile_shape=(2, 2), columns=2)
    new_plan = make_montage_plan(new_state, axis=2, indices=(1, 2), tile_shape=(2, 2), columns=2)
    old_viewport = MontageViewportPlan(2, (0, 1), (4, 8), (2, 2), old_plan, ((0.0, 4.0), (0.0, 2.0)), False, True)
    new_viewport = MontageViewportPlan(2, (1, 2), (4, 8), (2, 2), new_plan, ((0.0, 4.0), (0.0, 2.0)), False, True)
    old_key = montage_session_key(_document_key(document), old_state, old_viewport, None)
    new_key = montage_session_key(_document_key(document), new_state, new_viewport, None)

    win = Window(document, new_state, new_viewport)
    win._committed_display_frame = SimpleNamespace(key=SimpleNamespace(request_key=old_key))

    assert old_key != new_key
    assert win._interactive_frame_cache_hit() is False

    win._committed_display_frame = SimpleNamespace(key=SimpleNamespace(request_key=new_key))

    assert win._interactive_frame_cache_hit() is True


def test_montage_viewport_update_token_tracks_viewport_revision():
    from arrayscope.window.frame_renderer import _montage_work_token

    session = SimpleNamespace(
        session_id=1,
        key="session",
        render_generation=2,
        payload_revision=3,
        level_revision=4,
        viewport_revision=5,
    )
    viewport_token = _montage_work_token(session, "viewport_update")

    session.viewport_revision += 1

    assert _montage_work_token(session, "viewport_update") != viewport_token


def test_stale_montage_viewport_update_token_does_not_run():
    from arrayscope.window.frame_renderer import FrameRenderMixin

    class _Window(FrameRenderMixin):
        def __init__(self):
            self.win = self
            self.view_state = SimpleNamespace(montage_axis=2)
            self._montage_session = SimpleNamespace(
                session_id=1,
                key="session",
                render_generation=2,
                payload_revision=3,
                level_revision=4,
                viewport_revision=6,
            )
            self._montage_viewport_update_token = (
                "session",
                "session",
                2,
                5,
            )
            self.called = False

        def _try_update_montage_viewport_only(self):
            self.called = True
            return True

    win = _Window()

    win._run_montage_viewport_update()

    assert win.called is False


def test_loading_montage_profile_retry_waits_for_visibility_without_timer():
    from arrayscope.window.frame_renderer import FrameRenderMixin

    class _LiveProfile:
        def isChecked(self):
            return True

    class _ProfileDock:
        visible = False

        def isVisible(self):
            return bool(self.visible)

    class _Window(FrameRenderMixin):
        def __init__(self):
            self.win = self
            self.view_state = SimpleNamespace(montage_axis=2)
            self.widgets = {"buttons": {"display": {"live_profile": _LiveProfile()}}}
            self.profile_dock = _ProfileDock()
            self.updated = []

        def _update_live_profile_from_pending_pos(self):
            self.updated.append(self._pending_profile_point)

    win = _Window()

    win._schedule_loading_montage_profile_retry(1.0, 2.0)

    assert not hasattr(win, "_montage_profile_retry_timer")
    assert win.updated == []
    assert win._pending_montage_profile_retry == (1.0, 2.0)

    win.profile_dock.visible = True
    win._retry_loading_montage_profile()

    assert win.updated == [(1.0, 2.0)]


def test_native_tiled_payloads_publish_semantic_histogram_for_partial_commit():
    from arrayscope.core.window_levels import LevelSourceRank
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.model.montage_levels import MontageLevelStats
    from arrayscope.window.frame_renderer import (
        _should_publish_montage_histogram_plot,
        _tiled_payloads_require_semantic_histogram_plot,
    )

    payloads = {
        0: DisplayTilePayload(
            tile_number=0,
            source_index=0,
            image=np.zeros((2, 2), dtype=np.float32),
            histogram_data=None,
            source_id=("native", 0),
        )
    }
    stats = MontageLevelStats(
        bounds=(0.0, 1.0),
        source_indices=frozenset({0}),
        expected_indices=frozenset({0, 1}),
        rank=LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
        sample=np.asarray([0.0, 1.0], dtype=np.float32),
    )

    assert _tiled_payloads_require_semantic_histogram_plot(payloads) is True
    assert (
        _should_publish_montage_histogram_plot(
            False,
            False,
            stats,
            requires_semantic_plot=_tiled_payloads_require_semantic_histogram_plot(payloads),
        )
        is True
    )
    assert _should_publish_montage_histogram_plot(False, False, stats) is False


def test_stage_wait_in_flight_is_not_actionable_timer_work():
    pytest.importorskip("pyqtgraph")
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.window.frame_renderer import FrameRenderMixin

    class _Cache:
        def get(self, key):
            return None

    class _Materializer:
        _in_flight = {"stage-key": object()}

    class _Evaluator:
        stage_cache = _Cache()
        stage_materializer = _Materializer()

    class _Window(FrameRenderMixin):
        def __init__(self):
            self.win = self

        operation_evaluator = _Evaluator()

    session = SimpleNamespace(
        stage_fan_in=StageFanInState(waiting_tiles={"stage-key": [SimpleNamespace(montage_index=3)]}),
    )

    assert not _Window()._stage_wait_has_actionable_work(session, release_missing=True)


def test_stage_wait_completed_stage_is_actionable_timer_work():
    pytest.importorskip("pyqtgraph")
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.window.frame_renderer import FrameRenderMixin

    class _Cache:
        def get(self, key):
            return "stage-value" if key == "stage-key" else None

    class _Materializer:
        _in_flight = {}

    class _Evaluator:
        stage_cache = _Cache()
        stage_materializer = _Materializer()

    class _Window(FrameRenderMixin):
        def __init__(self):
            self.win = self

        operation_evaluator = _Evaluator()

    session = SimpleNamespace(
        stage_fan_in=StageFanInState(waiting_tiles={"stage-key": [SimpleNamespace(montage_index=3)]}),
    )

    assert _Window()._stage_wait_has_actionable_work(session, release_missing=True)


def test_montage_session_key_excludes_transient_viewport_range():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan

    state = ViewState.from_shape((4, 4, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(4, 4), columns=3)
    first = MontageViewportPlan(2, tuple(range(6)), (100, 100), (4, 4), plan, ((0, 10), (0, 10)), True, True)
    second = MontageViewportPlan(2, tuple(range(6)), (100, 100), (4, 4), plan, ((10, 20), (0, 10)), True, True)

    assert montage_session_key("doc", state, first, None) == montage_session_key("doc", state, second, None)


def test_montage_session_key_changes_with_population_but_not_layout_reflow():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan

    state = ViewState.from_shape((4, 4, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    plan3 = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(4, 4), columns=3)
    plan2 = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(4, 4), columns=2)
    base = MontageViewportPlan(2, tuple(range(6)), (100, 100), (4, 4), plan3, None, True, True)
    changed_population = MontageViewportPlan(2, tuple(range(5)), (100, 100), (4, 4), plan3, None, True, True)
    changed_layout = MontageViewportPlan(2, tuple(range(6)), (100, 100), (4, 4), plan2, None, True, True)

    key = montage_session_key("doc", state, base, None)
    assert key != montage_session_key("doc", state, changed_population, None)
    assert key == montage_session_key("doc", state, changed_layout, None)


def test_direct_tiled_payload_retarget_allows_only_safe_layout_reflow():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window.frame_renderer import _safe_tiled_payload_geometry_retarget

    state = ViewState.from_shape((4, 4, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    plan3 = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(4, 4), columns=3)
    plan2 = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(4, 4), columns=2)
    previous = DisplayGeometry(view_state=state, display_shape=plan3.display_shape, montage=plan3.geometry)
    reflow = DisplayGeometry(view_state=state, display_shape=plan2.display_shape, montage=plan2.geometry)
    changed_indices = make_montage_plan(state, axis=2, indices=tuple(range(5)), tile_shape=(4, 4), columns=3)
    incompatible = DisplayGeometry(
        view_state=state,
        display_shape=changed_indices.display_shape,
        montage=changed_indices.geometry,
    )

    assert _safe_tiled_payload_geometry_retarget(previous, reflow)
    assert not _safe_tiled_payload_geometry_retarget(previous, incompatible)
