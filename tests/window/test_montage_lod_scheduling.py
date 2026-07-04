"""Renderer-side draining of pending LOD materializations (ADR 0050).

These tests exercise the real ``FrameRenderMixin`` scheduling methods with a
fake window/controller composition (``fake.win = fake``), so they need the Qt
import closure of ``frame_renderer`` and run in the host environment.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arrayscope.core.work_graph import WorkLane
from arrayscope.core.scheduler import EvalPriority
from arrayscope.display.lod import LOD_POLICY_RESIDENT
from arrayscope.display.pyramid import PyramidCache
from arrayscope.window.frame_renderer import FrameRenderMixin

from tests.window.test_montage_lod_residency import TILE, _cold_session, _session


class FakeController:
    def __init__(self, *, blocked=False):
        self.blocked = blocked
        self.calls = []

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
    # The fake controller ran the workers inline: levels were admitted and
    # each completion marked its tile dirty and asked for an ordinary commit.
    assert len(pyramid) == 2
    assert pyramid.pending_count == 0
    assert sorted(session.dirty_payloads) == [0, 1]
    assert renderer.commit_requests == [False, False]
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
    assert len(pyramid) == 1
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
