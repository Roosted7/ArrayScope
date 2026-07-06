"""Renderer-side draining of pending LOD materializations (ADR 0050).

These tests exercise the real ``FrameRenderMixin`` scheduling methods with a
fake window/controller composition (``fake.win = fake``), so they need the Qt
import closure of ``frame_renderer`` and run in the host environment.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arrayscope.core.work_graph import WorkLane
from arrayscope.core.scheduler import EvalPriority, FrameTarget
from arrayscope.core.view_state import ChannelMode, ViewState
from arrayscope.display.lod import LOD_POLICY_RESIDENT
from arrayscope.display.model.frame import TiledValueSource
from arrayscope.display.pyramid import PyramidCache
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT, CenteredIFFT, FFTShift
from arrayscope.window.frame_renderer import FrameRenderMixin

from tests.window.test_montage_lod_residency import TILE, _cold_session, _session


class FakeController:
    def __init__(self, *, blocked=False):
        self.blocked = blocked
        self.calls = []
        self.capacity_waiters = {}

    def notify_when_capacity(self, key, fn):
        # ADR 0051 P2: blocked admissions arm a one-shot wakeup.
        self.capacity_waiters[key] = fn

    def start_latest(self, fn, **kwargs):
        self.calls.append(kwargs)
        if self.blocked:
            on_stale = kwargs.get("on_stale")
            if on_stale is not None:
                on_stale()
            return None
        result = fn(None) if kwargs.get("pass_token") else fn()
        on_done = kwargs.get("on_done")
        if on_done is not None:
            on_done(result)
        return len(self.calls)


def _renderer(session, *, blocked=False, current=True):
    fake = SimpleNamespace()
    fake.win = fake
    fake.montage_tile_evaluation_controller = FakeController(blocked=blocked)
    fake.visible_evaluation_controller = fake.montage_tile_evaluation_controller
    fake._montage_session = session if current else None
    fake._montage_session_is_current = lambda candidate: bool(current)
    fake._is_current_montage_session = lambda session_id, key: bool(current)
    fake._is_current_render_generation = lambda generation: True
    fake.commit_requests = []
    fake._schedule_montage_presentation_commit = (
        lambda session, force=False: fake.commit_requests.append(bool(force))
    )
    # Machine-derived dispatch (ADR 0051 P2): level-ready re-derives all
    # pumps; the commit pump is what this fixture observes.
    fake._dispatch_montage_work = FrameRenderMixin._dispatch_montage_work.__get__(fake)
    fake._schedule_montage_tiles = lambda session: None
    fake._schedule_montage_tile_result_flush = lambda session: None
    fake._schedule_montage_attached_stage_waits = lambda session: None
    fake._schedule_deferred_montage_planning = lambda session, delay_ms=0: None
    fake._ensure_montage_watchdog = lambda: None
    fake._schedule_montage_lod_materializations = (
        FrameRenderMixin._schedule_montage_lod_materializations.__get__(fake)
    )
    fake._on_montage_lod_level_ready = FrameRenderMixin._on_montage_lod_level_ready.__get__(fake)
    return fake


def test_drain_schedules_low_priority_supersedable_reductions_and_streams_results():
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(mode=LOD_POLICY_RESIDENT, pyramid=pyramid)
    session.build_tile_presentation({})
    assert len(session.pending_lod_requests) == 2
    renderer = _renderer(session)

    renderer._schedule_montage_lod_materializations(session)

    controller = renderer.montage_tile_evaluation_controller
    assert session.pending_lod_requests == []
    assert len(controller.calls) == 2
    for call in controller.calls:
        assert call["priority"] == EvalPriority.PREFETCH
        work_item = call["work_item"]
        assert work_item.lane == WorkLane.SPECULATIVE_RESIDENCY
        assert call["supersession_key"][0] == "montage-lod"
        assert call["supersession_value"][:2] == (session.key, int(session.session_id))
    # The fake controller ran the workers inline: each tile's chain admitted
    # the demanded level 2 plus the acceptable level 1 on the way (ADR 0050
    # level-chaining), and each completion re-derived dispatch (ADR 0051 P2).
    # With evaluation fully drained, the derived commit is forced — streamed
    # levels present immediately; the interaction gate inside the commit
    # scheduler still defers during a gesture.
    assert len(pyramid) == 4
    assert pyramid.pending_count == 0
    assert sorted(session.dirty_payloads) == [0, 1]
    assert renderer.commit_requests == [True, True]
    assert session.lod_materializations_completed == 2
    assert renderer._montage_lod_materializations_scheduled == 2
    assert renderer._montage_lod_materializations_completed == 2

    # The streamed levels now present through the normal build path.
    _state, delta = session.build_tile_presentation({})
    assert {payload.lod.level for payload in delta.upserts.values()} == {2}


def test_blocked_admission_releases_singleflight_claims_for_retry():
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(mode=LOD_POLICY_RESIDENT, pyramid=pyramid)
    session.build_tile_presentation({})
    renderer = _renderer(session, blocked=True)

    renderer._schedule_montage_lod_materializations(session)

    assert len(pyramid) == 0
    assert pyramid.pending_count == 0, "blocked work must release its claim"
    assert renderer._montage_lod_materializations_blocked == 2
    # ADR 0051 P2: a blocked admission must leave a wakeup armed — without
    # it, released levels waited for an unrelated pan/zoom (field report
    # 2026-07-05: tiles stuck on a coarser LOD at idle).
    controller = renderer.montage_tile_evaluation_controller
    assert ("montage-lod", session.key) in controller.capacity_waiters
    # The next presentation build can re-claim and re-queue the levels.
    session.dirty_payloads.update({0: None, 1: None})
    session.build_tile_presentation({})
    assert len(session.pending_lod_requests) == 2


def test_stale_session_releases_claims_without_scheduling():
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(mode=LOD_POLICY_RESIDENT, pyramid=pyramid)
    session.build_tile_presentation({})
    renderer = _renderer(session, current=False)

    renderer._schedule_montage_lod_materializations(session)

    assert renderer.montage_tile_evaluation_controller.calls == []
    assert pyramid.pending_count == 0
    assert session.pending_lod_requests == []


def _tile_worker_renderer(session, *, evaluated):
    """Fake window/renderer composition around the real tile scheduling method."""

    from arrayscope.operations.evaluator import EvaluationResult

    fake = SimpleNamespace()
    fake.win = fake
    fake.montage_tile_evaluation_controller = FakeController()
    fake.visible_evaluation_controller = fake.montage_tile_evaluation_controller
    fake._montage_session = session
    fake._montage_session_is_current = lambda candidate: True
    fake.completed = []
    fake._on_montage_tile_done = lambda session_id, tile, result, **kwargs: fake.completed.append(
        (int(tile.montage_index), result)
    )
    fake._on_montage_tile_error = lambda session_id, tile, exc: None
    fake._on_montage_tile_slow = lambda session_id: None
    fake._is_current_montage_session = lambda session_id, key: True
    fake._evaluation_context = lambda lane, token: None
    fake._schedule_montage_presentation_commit = lambda session, force=False: None
    fake._dispatch_montage_work = lambda session: None

    def _evaluate(session_arg, tile, token=None):
        image = np.arange(TILE * TILE, dtype=np.float32).reshape(TILE, TILE)
        value = SimpleNamespace(
            data=image,
            histogram_data=image,
            semantic_data=None,
            shader_mapping=None,
            texture_kind=None,
            lod=None,
            level_data=None,
            level_stats=None,
        )
        result = EvaluationResult(
            value=value,
            eval_ms=0.0,
            slab_shape=image.shape,
            slab_nbytes=int(image.nbytes),
        )
        evaluated.append(result)
        return result

    fake._evaluate_montage_tile_snapshot = _evaluate
    fake._schedule_next_montage_tile = FrameRenderMixin._schedule_next_montage_tile.__get__(fake)
    fake._schedule_montage_preview_tile = FrameRenderMixin._schedule_montage_preview_tile.__get__(fake)
    fake._schedule_montage_shared_preview_batch = FrameRenderMixin._schedule_montage_shared_preview_batch.__get__(fake)
    return fake


def test_cold_tile_worker_reduces_at_ingest_before_first_presentation():
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _cold_session(pyramid=pyramid)
    session.frame_plan = SimpleNamespace(
        target=SimpleNamespace(quality="final"), tile_shape=(TILE, TILE)
    )
    session.pending_tiles.append(session.plan.tiles[0])
    evaluated = []
    renderer = _tile_worker_renderer(session, evaluated=evaluated)

    assert renderer._schedule_next_montage_tile(session) is True

    # The reduction ran on the worker as part of tile materialization: the
    # demanded level was admitted before the done fan-in saw the result.
    assert len(evaluated) == 1
    assert len(pyramid) == 2
    assert pyramid.pending_count == 0
    assert renderer._montage_lod_ingest_reductions == 1
    assert len(renderer.completed) == 1

    # First presentation selects the reduced level; the tile never emits a
    # native payload and no post-hoc materialization is requested for it.
    tile_number, result = renderer.completed[0]
    from arrayscope.window.frame_renderer import _rendered_tile_from_evaluation_result

    session.mark_loaded(_rendered_tile_from_evaluation_result(session.plan.tiles[0], result))
    _state, delta = session.build_tile_presentation({})
    assert delta.upserts[tile_number].lod.level == 2
    assert delta.upserts[tile_number].texture_data.shape[:2] == (TILE // 4, TILE // 4)
    assert session.pending_lod_requests == []


def test_native_scale_scheduling_performs_no_ingest_reduction():
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _cold_session(pyramid=pyramid)
    session.view_range = ((0.0, float(2 * TILE)), (0.0, float(TILE)))
    session.build_tile_presentation({})
    assert session.ingest_lod_demand() is None
    session.frame_plan = SimpleNamespace(
        target=SimpleNamespace(quality="final"), tile_shape=(TILE, TILE)
    )
    session.pending_tiles.append(session.plan.tiles[0])
    renderer = _tile_worker_renderer(session, evaluated=[])

    assert renderer._schedule_next_montage_tile(session) is True

    assert len(pyramid) == 0
    assert int(getattr(renderer, "_montage_lod_ingest_reductions", 0)) == 0


def _non_display_transform_session_with_lazy_source():
    from dataclasses import replace

    from arrayscope.display.montage import MontagePlan

    class LazyArray:
        def __init__(self, data):
            self._data = np.asarray(data)
            self.shape = self._data.shape
            self.dtype = self._data.dtype
            self.reads = []

        def read_region(self, index, *, cancellation_token=None):
            self.reads.append(index)
            return self._data[index]

    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _cold_session(pyramid=pyramid)
    data = np.arange(TILE * TILE * 2, dtype=np.float32).reshape(TILE, TILE, 2)
    source = LazyArray(data)
    state = (
        ViewState.from_shape(data.shape)
        .with_image_axes(0, 1)
        .with_montage_axis(2, indices=(0, 1), columns=2, text=":")
    )
    tiles = tuple(
        replace(tile, view_state=state.tile_state_for_slice(2, tile.source_index))
        for tile in session.plan.tiles
    )
    session.plan = MontagePlan(
        axis=2,
        tile_shape=session.plan.tile_shape,
        grid_shape=session.plan.grid_shape,
        columns=2,
        rows=1,
        gap=session.plan.gap,
        tiles=tiles,
    )
    session.view_state = state
    session.montage_axis = 2
    session.document = ArrayDocument(
        source,
        steps=(CenteredFFT(axis=2), FFTShift(axis=2), CenteredIFFT(axis=2)),
    )
    session.frame_plan = SimpleNamespace(
        target=FrameTarget(("semantic",), ("viewport",), ("presentation",), "final"),
        tile_shape=(TILE, TILE),
    )
    return session, source


def test_non_display_transform_preview_is_not_scheduled_by_default_after_profile_regression():
    session, _source = _non_display_transform_session_with_lazy_source()
    session.pending_tiles.append(session.plan.tiles[0])
    renderer = _tile_worker_renderer(session, evaluated=[])

    assert renderer._schedule_next_montage_tile(session) is True

    assert len(renderer.montage_tile_evaluation_controller.calls) == 1
    assert int(getattr(renderer, "_montage_preview_reduced_scheduled", 0) or 0) == 0


def test_non_display_transform_preview_can_schedule_once_for_experimental_shared_tile_window(monkeypatch):
    monkeypatch.setenv("ARRAYSCOPE_SHARED_TRANSFORM_PREVIEW", "1")
    session, source = _non_display_transform_session_with_lazy_source()
    session.pending_tiles.append(session.plan.tiles[0])
    renderer = _tile_worker_renderer(session, evaluated=[])

    assert renderer._schedule_next_montage_tile(session) is True

    calls = renderer.montage_tile_evaluation_controller.calls
    assert len(calls) == 2
    assert calls[0]["work_item"].lane == WorkLane.VISIBLE_MATERIALIZATION
    assert calls[1]["work_item"].lane == WorkLane.DISPLAY_PREVIEW
    assert calls[1]["key"][0] == "montage_preview_batch"
    assert source.reads == [(slice(None, None, None), slice(None, None, None), slice(None, None, None))]
    assert int(getattr(renderer, "_montage_preview_reduced_scheduled", 0) or 0) == 1
    assert session.lod_preview_presentations == 2
    assert sorted(session.display_tile_payloads) == [0, 1]
    assert {payload.quality for payload in session.display_tile_payloads.values()} == {"preview"}
    assert {payload.texture_data.shape[:2] for payload in session.display_tile_payloads.values()} == {
        (TILE // 4, TILE // 4)
    }


def test_reduced_preview_evaluation_admits_floor_payload_without_exact_semantics():
    from dataclasses import replace

    from arrayscope.display.montage import MontagePlan
    from arrayscope.window.frame_renderer import _evaluate_montage_tile_preview_snapshot

    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _cold_session(pyramid=pyramid)
    data = np.arange(TILE * TILE * 2, dtype=np.float32).reshape(TILE, TILE, 2)
    state = (
        ViewState.from_shape(data.shape)
        .with_image_axes(0, 1)
        .with_montage_axis(2, indices=(0, 1), columns=2, text=":")
    )
    tiles = tuple(
        replace(tile, view_state=state.tile_state_for_slice(2, tile.source_index))
        for tile in session.plan.tiles
    )
    session.plan = MontagePlan(
        axis=2,
        tile_shape=session.plan.tile_shape,
        grid_shape=session.plan.grid_shape,
        columns=session.plan.columns,
        rows=session.plan.rows,
        gap=session.plan.gap,
        tiles=tiles,
    )
    session.view_state = state
    session.document = ArrayDocument(data)
    tile = session.plan.tiles[0]
    demand = session.ingest_lod_demand()

    preview = _evaluate_montage_tile_preview_snapshot(
        session,
        tile,
        demand=demand,
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert preview is not None
    key, plane, histogram = preview
    assert key.level_xy == (2, 2)
    assert plane.shape == (TILE // 4, TILE // 4)
    assert histogram is None
    pyramid.admit(key, plane)
    session._ensure_floor_payloads((0,))
    payload = session.display_tile_payloads[0]
    assert payload.quality == "preview"
    assert payload.semantic_data is None
    assert TiledValueSource({0: payload}).tile_region(tile, (slice(0, 1), slice(0, 1))) is None


def test_reduced_preview_evaluation_reads_only_tile_display_range():
    from dataclasses import replace

    from arrayscope.display.montage import MontagePlan
    from arrayscope.window.frame_renderer import _evaluate_montage_tile_preview_snapshot

    class LazyArray:
        def __init__(self, data):
            self._data = np.asarray(data)
            self.shape = self._data.shape
            self.dtype = self._data.dtype
            self.reads = []

        def read_region(self, index, *, cancellation_token=None):
            self.reads.append(index)
            return self._data[index]

    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _cold_session(pyramid=pyramid)
    data = np.arange((TILE * 2) * (TILE * 2) * 2, dtype=np.float32).reshape(TILE * 2, TILE * 2, 2)
    source = LazyArray(data)
    y_indices = tuple(range(5, 5 + TILE))
    x_indices = tuple(range(7, 7 + TILE))
    state = (
        ViewState.from_shape(data.shape)
        .with_image_axes(0, 1)
        .with_axis_range(0, y_indices, text="5:5+tile")
        .with_axis_range(1, x_indices, text="7:7+tile")
        .with_montage_axis(2, indices=(0, 1), columns=2, text=":")
    )
    tiles = tuple(
        replace(tile, view_state=state.tile_state_for_slice(2, tile.source_index))
        for tile in session.plan.tiles
    )
    session.plan = MontagePlan(
        axis=2,
        tile_shape=session.plan.tile_shape,
        grid_shape=session.plan.grid_shape,
        columns=session.plan.columns,
        rows=session.plan.rows,
        gap=session.plan.gap,
        tiles=tiles,
    )
    session.view_state = state
    session.document = ArrayDocument(source)
    tile = session.plan.tiles[0]

    preview = _evaluate_montage_tile_preview_snapshot(
        session,
        tile,
        demand=session.ingest_lod_demand(),
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert preview is not None
    assert source.reads == [(slice(5, 5 + TILE, 1), slice(7, 7 + TILE, 1), slice(0, 1, 1))]
    _key, plane, _histogram = preview
    assert plane.shape == (TILE // 4, TILE // 4)


def test_fft_over_montage_axis_preview_reduces_display_input_and_expands_transform_axis():
    from dataclasses import replace

    from arrayscope.display.montage import MontagePlan
    from arrayscope.window.frame_renderer import _evaluate_montage_tile_preview_snapshot

    class LazyArray:
        def __init__(self, data):
            self._data = np.asarray(data)
            self.shape = self._data.shape
            self.dtype = self._data.dtype
            self.reads = []

        def read_region(self, index, *, cancellation_token=None):
            self.reads.append(index)
            return self._data[index]

    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _cold_session(pyramid=pyramid)
    data = np.arange((TILE * 2) * (TILE * 2) * 2, dtype=np.float32).reshape(TILE * 2, TILE * 2, 2)
    source = LazyArray(data)
    y_indices = tuple(range(5, 5 + TILE))
    x_indices = tuple(range(7, 7 + TILE))
    state = (
        ViewState.from_shape(data.shape)
        .with_image_axes(0, 1)
        .with_axis_range(0, y_indices, text="5:5+tile")
        .with_axis_range(1, x_indices, text="7:7+tile")
        .with_montage_axis(2, indices=(0, 1), columns=2, text=":")
    )
    tiles = tuple(
        replace(tile, view_state=state.tile_state_for_slice(2, tile.source_index))
        for tile in session.plan.tiles
    )
    session.plan = MontagePlan(
        axis=2,
        tile_shape=session.plan.tile_shape,
        grid_shape=session.plan.grid_shape,
        columns=2,
        rows=1,
        gap=session.plan.gap,
        tiles=tiles,
    )
    session.view_state = state
    session.document = ArrayDocument(
        source,
        steps=(CenteredFFT(axis=2), FFTShift(axis=2), CenteredIFFT(axis=2)),
    )
    tile = session.plan.tiles[1]

    preview = _evaluate_montage_tile_preview_snapshot(
        session,
        tile,
        demand=session.ingest_lod_demand(),
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert preview is not None
    assert source.reads == [(slice(5, 5 + TILE, 1), slice(7, 7 + TILE, 1), slice(None, None, None))]
    _key, plane, _histogram = preview
    assert plane.shape == (TILE // 4, TILE // 4)


def test_fft_over_display_axis_preview_falls_back_to_native_output_reduction():
    from dataclasses import replace

    from arrayscope.display.montage import MontagePlan
    from arrayscope.window.frame_renderer import _evaluate_montage_tile_preview_snapshot

    class LazyArray:
        def __init__(self, data):
            self._data = np.asarray(data)
            self.shape = self._data.shape
            self.dtype = self._data.dtype
            self.reads = []

        def read_region(self, index, *, cancellation_token=None):
            self.reads.append(index)
            return self._data[index]

    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _cold_session(pyramid=pyramid)
    data = np.arange(TILE * TILE * 2, dtype=np.float32).reshape(TILE, TILE, 2)
    source = LazyArray(data)
    state = (
        ViewState.from_shape(data.shape)
        .with_image_axes(0, 1)
        .with_montage_axis(2, indices=(0, 1), columns=2, text=":")
    )
    tiles = tuple(
        replace(tile, view_state=state.tile_state_for_slice(2, tile.source_index))
        for tile in session.plan.tiles
    )
    session.plan = MontagePlan(
        axis=2,
        tile_shape=session.plan.tile_shape,
        grid_shape=session.plan.grid_shape,
        columns=2,
        rows=1,
        gap=session.plan.gap,
        tiles=tiles,
    )
    session.view_state = state
    session.document = ArrayDocument(source, steps=(CenteredFFT(axis=0),))
    tile = session.plan.tiles[0]

    preview = _evaluate_montage_tile_preview_snapshot(
        session,
        tile,
        demand=session.ingest_lod_demand(),
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert preview is not None
    assert source.reads == [(slice(None, None, None), slice(None, None, None), 0)]
    _key, plane, _histogram = preview
    assert plane.shape == (TILE // 4, TILE // 4)


def test_reduced_rgb_preview_floor_preserves_display_histogram_for_rewindowing():
    from dataclasses import replace

    from arrayscope.display.montage import MontagePlan
    from arrayscope.window import montage_lod
    from arrayscope.window.frame_renderer import _evaluate_montage_tile_preview_snapshot

    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _cold_session(pyramid=pyramid)
    real = np.arange(TILE * TILE * 2, dtype=np.float32).reshape(TILE, TILE, 2)
    imag = real + 10.0
    data = (real + 1j * imag).astype(np.complex64)
    state = (
        ViewState.from_shape(data.shape)
        .with_image_axes(0, 1)
        .with_channel(ChannelMode.COMPLEX)
        .with_montage_axis(2, indices=(0, 1), columns=2, text=":")
    )
    tiles = tuple(
        replace(tile, view_state=state.tile_state_for_slice(2, tile.source_index))
        for tile in session.plan.tiles
    )
    session.plan = MontagePlan(
        axis=2,
        tile_shape=session.plan.tile_shape,
        grid_shape=session.plan.grid_shape,
        columns=session.plan.columns,
        rows=session.plan.rows,
        gap=session.plan.gap,
        tiles=tiles,
    )
    session.view_state = state
    session.document = ArrayDocument(data)
    session.rgb = True
    tile = session.plan.tiles[0]

    preview = _evaluate_montage_tile_preview_snapshot(
        session,
        tile,
        demand=session.ingest_lod_demand(),
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert preview is not None
    key, plane, histogram = preview
    assert key.component == "rgb8"
    assert plane.shape == (TILE // 4, TILE // 4, 3)
    assert histogram is not None
    assert histogram.shape == (TILE // 4, TILE // 4)
    pyramid.admit(key, plane)
    pyramid.admit(montage_lod.histogram_key_for_level_key(key), histogram)
    session._ensure_floor_payloads((0,))
    payload = session.display_tile_payloads[0]
    assert payload.quality == "preview"
    assert payload.texture_kind == "rgb8"
    assert payload.histogram_data is not None
    assert payload.histogram_data.shape == (TILE // 4, TILE // 4)
    assert payload.semantic_data is None
    assert payload.semantic_histogram_data is None
    assert TiledValueSource({0: payload}).tile_region(tile, (slice(0, 1), slice(0, 1))) is None


def test_pyramid_budget_retains_the_reference_montage_working_set():
    """ADR 0050 gate 6 needs the CPU cache to survive threshold recrossings.

    Reference montage: 272 tiles of 336x336.  Both the float32 scene and the
    complex64 (FFT) scene share one renderer-level pyramid cache, and each
    needs levels 1 and 2 resident for hysteresis crossings to be cache hits.
    """

    fake = SimpleNamespace()
    fake.win = fake
    fake._memory_policy = lambda: SimpleNamespace(display_cache_budget_bytes=256 * 1024 * 1024)
    pyramid = FrameRenderMixin._montage_lod_pyramid.__get__(fake)()

    level_texels = 168 * 168 + 84 * 84
    float_scene = 272 * level_texels * 4
    complex_scene = 272 * level_texels * 8
    footprint = float_scene + complex_scene
    assert pyramid.max_bytes is not None
    assert pyramid.max_bytes >= 2 * footprint, (
        "pyramid budget must hold the reference working set with headroom, "
        f"got {pyramid.max_bytes} for footprint {footprint}"
    )
    # The same cache object is reused across sessions.
    assert FrameRenderMixin._montage_lod_pyramid.__get__(fake)() is pyramid
