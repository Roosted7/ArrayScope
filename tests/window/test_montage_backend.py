from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.memory_policy import MemoryProfileChoice
from arrayscope.core.resource_governor import ResourcePressure, ResourcePressureState
from arrayscope.display.backend_contract import ImageViewBackendCapabilities
from arrayscope.render.progressive_scheduling import SchedulingPhase, SchedulingVerdict
from arrayscope.window.montage_payload_cache import (
    RetainedTiledPayloadStore,
)
from arrayscope.window.montage_payload_cache import (
    base_tile_source_id as _base_tile_source_id,
)
from arrayscope.window.montage_payload_cache import (
    payload_compatible_with_tile as _payload_compatible_with_tile,
)
from arrayscope.window.montage_payload_cache import (
    payload_lod_matches as _payload_lod_matches,
)
from arrayscope.window.montage_prefetch import (
    _owner_memory_pressure_blocks_prefetch,
    _owner_prefetch_batch_limit,
)
from arrayscope.window.montage_viewport import (
    MontageViewportPlan,
    frame_session_key,
)


def _window_ns(**kwargs):
    ns = SimpleNamespace(**kwargs)
    ns.win = ns
    return ns


def _refine_scheduling_policy():
    return SimpleNamespace(verdict=SchedulingVerdict(1, SchedulingPhase.REFINE, ()))


def _coverage_scheduling_policy():
    policy = SimpleNamespace(
        verdict=SchedulingVerdict(1, SchedulingPhase.COVERAGE, ()),
        evidence_pending_calls=[],
    )
    policy.set_coverage_evidence_pending = lambda pending: policy.evidence_pending_calls.append(
        bool(pending)
    )
    return policy


def _geometry():
    return SimpleNamespace(montage=object())


def _prefetch_window(
    *, profile=MemoryProfileChoice.BALANCED, memory_pressure=ResourcePressure.NORMAL
):
    pressure = ResourcePressureState(
        ResourcePressure.NORMAL,
        0.5,
        memory_pressure,
        ResourcePressure.NORMAL,
        "test",
    )
    governor = SimpleNamespace(
        profile=profile, diagnostics=lambda: SimpleNamespace(pressure=pressure)
    )
    return SimpleNamespace(win=SimpleNamespace(resource_governor=governor))


def test_montage_prefetch_owner_uses_profile_batch_without_governor_decision():
    assert (
        _owner_prefetch_batch_limit(_prefetch_window(profile=MemoryProfileChoice.CONSERVATIVE)) == 1
    )
    assert _owner_prefetch_batch_limit(_prefetch_window(profile=MemoryProfileChoice.BALANCED)) == 2
    assert (
        _owner_prefetch_batch_limit(_prefetch_window(profile=MemoryProfileChoice.AGGRESSIVE)) == 4
    )


def test_montage_prefetch_owner_blocks_memory_pressure_from_telemetry():
    assert _owner_memory_pressure_blocks_prefetch(
        _prefetch_window(memory_pressure=ResourcePressure.ELEVATED)
    )
    assert not _owner_memory_pressure_blocks_prefetch(
        _prefetch_window(memory_pressure=ResourcePressure.NORMAL)
    )


def _committed_tiled_frame(geometry, *, key):
    from arrayscope.display.model.frame import (
        CommittedDisplayFrame,
        DisplayTilePayload,
        TiledValueSource,
    )
    from arrayscope.display.montage import MontageTileState

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


def test_pipeline_retarget_commits_swaps_for_its_final_lod_demand(monkeypatch):
    """A demand recomputed at pipeline ownership cannot lose its draw wakeup."""

    from arrayscope.render import lod as render_lod
    from arrayscope.window import frame_effects as montage_commit
    from arrayscope.window.frame_runtime import FrameRuntimeMixin

    calls = []
    session = SimpleNamespace(
        scheduling_policy=_refine_scheduling_policy(),
        lod_policy_decision=SimpleNamespace(demand=object()),
        pending_level_tiles=(),
        level_scan_remaining_tiles=0,
        active_tile_requests=(),
        loading_tiles=(),
        dirty_payloads={7: None},
        pending_payload_upserts={},
        histogram_aggregate_inflight=False,
        flush_pending=False,
        final_commit_pending=False,
        visible_first_pixels_presented=lambda: True,
        required_tile_numbers=lambda: (7,),
        first_pass_quality=None,
        first_pass_pixels_presented=lambda: False,
        is_complete=lambda: False,
        stage_fan_in=SimpleNamespace(
            active_requests=(),
            attached_requests=(),
            tile_stage_keys=(),
        ),
    )
    effects = SimpleNamespace(
        submit_shared_transform_floor=lambda _scope: 0,
    )
    pipeline = SimpleNamespace(
        effects=effects,
        retarget=lambda _intent, _demand, _scope: 0,
        last_plan_states=(),
        last_plan_steps=(),
    )
    runtime = SimpleNamespace()
    runtime._frame_session_is_current = lambda candidate: candidate is session
    runtime._montage_render_intent = lambda _session: object()
    runtime._lod_admission_scope = lambda _session, _intent: object()
    runtime._frame_pipeline_for_session = lambda _session: pipeline
    runtime.apply_ready_montage_display = lambda _session: calls.append("commit")
    runtime._finish_frame_session_if_complete = lambda _session: None
    runtime._ensure_montage_watchdog = lambda: None
    runtime._schedule_montage_cached_level_stats = lambda _session: None

    monkeypatch.setattr(
        render_lod, "selected_lod_factor", lambda _session: calls.append("select") or 8
    )
    monkeypatch.setattr(
        render_lod,
        "mark_ladder_swaps_for_current_demand",
        lambda _session: calls.append("mark") or True,
    )
    monkeypatch.setattr(montage_commit, "complete_deferred_stage_fan_in", lambda *_args: False)
    monkeypatch.setattr(montage_commit, "rearm_ready_stage_dependents", lambda *_args: None)

    submitted = FrameRuntimeMixin.retarget_frame_pipeline(runtime, session)

    assert submitted == 0
    assert calls[:2] == ["select", "mark"]
    assert calls.count("commit") == 1

    calls.clear()
    submitted = FrameRuntimeMixin.retarget_frame_pipeline(
        runtime,
        session,
        prepared_lod_swap_ready=True,
    )

    assert submitted == 0
    assert "select" not in calls
    assert "mark" not in calls
    assert calls.count("commit") == 1


def test_backend_refresh_obligation_crosses_clean_presentation_gate():
    from arrayscope.window.frame_controller import FrameControllerMixin

    session = SimpleNamespace(
        dirty_rects=(),
        dirty_tiles=(),
        dirty_payloads=(),
        pending_payload_upserts=(),
        pending_removals=(),
        backend_refresh_pending=True,
        has_pending_level_update=lambda: False,
        has_stale_level_presentations=lambda: False,
        final_commit_pending=False,
        flush_pending=False,
    )
    calls = []
    owner = SimpleNamespace(
        _frame_session_is_current=lambda candidate: candidate is session,
        apply_montage_presentation=lambda candidate: calls.append(candidate),
    )

    FrameControllerMixin.apply_ready_montage_display(owner, session)

    assert calls == [session]
    assert session.final_commit_pending is True
    assert session.flush_pending is True


def test_known_montage_level_source_is_not_resampled(monkeypatch):
    import arrayscope.render.level_stats as level_stats
    from arrayscope.display.model.montage_levels import MontageLevelTracker, TileLevelStats
    from arrayscope.window.frame_controller import FrameControllerMixin

    class Window(FrameControllerMixin):
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
        level_stats,
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


def test_montage_source_level_cache_reuses_overlapping_selection_and_keeps_refined():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.model.montage_levels import (
        LevelEvidenceQuality,
        TileLevelStats,
        montage_level_key,
    )
    from arrayscope.render.level_stats import LevelStatsService

    service = LevelStatsService()
    base = ViewState.from_shape((8, 8, 12))
    first_state = base.with_montage_axis(2, indices=tuple(range(8)), text="0:8")
    shifted_state = base.with_montage_axis(2, indices=tuple(range(1, 9)), text="1:9")
    first_key = montage_level_key("document", first_state)
    shifted_key = montage_level_key("document", shifted_state)
    refined = TileLevelStats(
        3,
        (10.0, 20.0),
        np.asarray([10.0, 20.0], dtype=np.float32),
        refined=True,
    )
    rough = TileLevelStats(
        3,
        (0.0, 1.0),
        np.asarray([0.0, 1.0], dtype=np.float32),
        evidence_quality=LevelEvidenceQuality.ROUGH_PREVIEW,
    )

    service._remember_montage_source_level_stats(first_key, refined)
    service._remember_montage_source_level_stats(shifted_key, rough)

    cached = service._cached_montage_source_level_stats(
        shifted_key,
        3,
        LevelEvidenceQuality.REFINED,
    )
    assert cached is refined
    assert cached.bounds == (10.0, 20.0)


@pytest.mark.parametrize(
    ("scheduling_policy", "first_pass_histogram_published"),
    [
        (_refine_scheduling_policy, True),
        (_coverage_scheduling_policy, False),
    ],
    ids=("refinement", "displayed-fallback-coverage"),
)
def test_histogram_aggregate_is_worker_derived_and_wakes_parked_presentation(
    scheduling_policy,
    first_pass_histogram_published,
):
    from arrayscope.display.backend_contract import WGPU_CAPABILITIES
    from arrayscope.display.model.montage_levels import MontageLevelTracker, TileLevelStats
    from arrayscope.render.level_stats import LevelStatsService

    class DeferredKernel:
        visible_backlog = 0

        def __init__(self):
            self.submissions = []

        def submit_speculative_batch(self, **kwargs):
            self.submissions.append(kwargs)
            return object()

        def submit(self, spec, *, on_done, on_stale, on_error):
            self.submissions.append(
                {
                    "fn": spec.fn,
                    "on_done": on_done,
                    "on_stale": on_stale,
                    "on_error": on_error,
                }
            )
            return object()

        def finish(self):
            submission = self.submissions.pop(0)
            submission["on_done"](submission["fn"]())

    requested = []
    session = SimpleNamespace(
        key=("frame",),
        session_id=3,
        level_key=("levels", "aggregate"),
        display_committed=True,
        scheduling_policy=scheduling_policy(),
        first_pass_histogram_published=first_pass_histogram_published,
        histogram_aggregate_inflight=False,
        histogram_aggregate_generation=None,
        pending_level_tiles=deque(),
        level_scan_remaining_tiles=0,
        semantic_level_evidence_progress=None,
        dirty_payloads={},
        pending_payload_upserts={},
        pending_removals=set(),
        flush_pending=True,
        final_commit_pending=True,
        pipeline=SimpleNamespace(
            effects=SimpleNamespace(request_presentation=lambda: requested.append("presentation"))
        ),
        required_target_settled=lambda: True,
    )

    class Window(LevelStatsService):
        def __init__(self):
            self.win = self
            self.img_view = SimpleNamespace(rendering_capabilities=WGPU_CAPABILITIES)
            self.kernel = DeferredKernel()
            self._frame_session = session
            self._tracker = MontageLevelTracker()

        def _frame_session_is_current(self, current):
            return current is self._frame_session

        def _montage_level_tracker(self):
            return self._tracker

    win = Window()
    win._tracker.ensure_expected(session.level_key, (0, 1))
    for source_index in (0, 1):
        values = np.asarray([source_index * 2.0, source_index * 2.0 + 1.0], dtype=np.float32)
        win._tracker.update_from_stats(
            session.level_key,
            TileLevelStats(source_index, (float(values[0]), float(values[-1])), values),
            aggregate=False,
        )

    assert win._montage_histogram_plot_data_for_session(session) is None
    assert session.histogram_aggregate_inflight
    assert len(win.kernel.submissions) == 1
    assert win._montage_histogram_plot_data_for_session(session) is None
    assert len(win.kernel.submissions) == 1

    win.kernel.finish()

    assert not session.histogram_aggregate_inflight
    assert requested == ["presentation"]
    assert np.array_equal(
        win._montage_histogram_plot_data_for_session(session),
        np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    )


def test_prepared_atomic_transaction_expires_when_level_generation_changes():
    from arrayscope.window.frame_effects import (
        _prepared_atomic_transaction_current,
        _shader_successor_transaction_payload_marker,
    )

    payload = SimpleNamespace(source_id=("source", 0), source_index=0)
    session = SimpleNamespace(
        session_id=7,
        level_generation=SimpleNamespace(revision=3),
        tile_presentation_state=SimpleNamespace(revision=11),
        visible_tile_numbers=(0,),
        skipped_tiles=(),
        display_tile_payloads={0: payload},
    )
    prepared = {
        "session_id": 7,
        "level_revision": 3,
        "marker_kind": "shader-successor",
        "tile_delta": SimpleNamespace(base_revision=11, active_tiles=(0,)),
        "payload_markers": {0: _shader_successor_transaction_payload_marker(payload)},
    }

    assert _prepared_atomic_transaction_current(session, prepared)
    session.level_generation.revision = 4
    assert not _prepared_atomic_transaction_current(session, prepared)


def test_prepared_atomic_transaction_uses_required_not_coverage_scope():
    """An offscreen retained shell cannot invalidate an on-screen handoff."""

    from arrayscope.window.frame_effects import (
        _prepared_atomic_transaction_current,
        _shader_successor_transaction_payload_marker,
    )

    payload = SimpleNamespace(source_id=("source", 0), source_index=0)
    session = SimpleNamespace(
        session_id=7,
        level_generation=SimpleNamespace(revision=3),
        tile_presentation_state=SimpleNamespace(revision=11),
        visible_tile_numbers=(0, 1),
        skipped_tiles=(),
        atomic_successor_required_scope=lambda: (0,),
        display_tile_payloads={0: payload},
    )
    prepared = {
        "session_id": 7,
        "level_revision": 3,
        "marker_kind": "shader-successor",
        "tile_delta": SimpleNamespace(base_revision=11, active_tiles=(0,)),
        "payload_markers": {0: _shader_successor_transaction_payload_marker(payload)},
    }

    assert _prepared_atomic_transaction_current(session, prepared)


def test_prepared_atomic_transaction_expires_when_semantic_target_changes():
    from arrayscope.display.model.tile_identity import TileIdentity
    from arrayscope.display.shader_mapping import TexturePlaneKind
    from arrayscope.window.frame_effects import (
        _prepared_atomic_transaction_current,
        _shader_successor_transaction_payload_marker,
    )

    def identity(semantic_generation):
        return TileIdentity(
            document_generation=("document", 1),
            operation_key=(),
            source_index=0,
            image_axes=(0, 1),
            axis_flips=(False, False, False),
            channel="real",
            complex_mapping=("scalar", "real", "mapped"),
            texture_kind=TexturePlaneKind.SCALAR_R32F,
            semantic_generation=semantic_generation,
        )

    old_target = identity(("crop", 95, 195))
    current_target = identity(("crop", 94, 194))
    payload = SimpleNamespace(
        source_id=("source", 0),
        source_index=0,
        tile_identity=old_target,
    )
    session = SimpleNamespace(
        session_id=7,
        level_generation=SimpleNamespace(revision=3),
        tile_presentation_state=SimpleNamespace(revision=11),
        visible_tile_numbers=(0,),
        skipped_tiles=(),
        display_tile_payloads={0: payload},
        tile_target_identities=lambda _required: {0: current_target},
    )
    delta = SimpleNamespace(
        base_revision=11,
        active_tiles=(0,),
        upserts={0: payload},
        target_identities={0: old_target},
    )
    prepared = {
        "session_id": 7,
        "level_revision": 3,
        "marker_kind": "shader-successor",
        "tile_delta": delta,
        "payload_markers": {0: _shader_successor_transaction_payload_marker(payload)},
    }

    assert not _prepared_atomic_transaction_current(session, prepared)


def test_payload_level_stats_are_bounded_and_deferred(monkeypatch):
    import arrayscope.render.level_stats as level_stats
    from arrayscope.display.model.montage_levels import MontageLevelTracker
    from arrayscope.window.frame_controller import FrameControllerMixin

    class Window(FrameControllerMixin):
        def __init__(self):
            from arrayscope.display.backend_contract import VISPY_CAPABILITIES

            self.win = self
            self.img_view = SimpleNamespace(rendering_capabilities=VISPY_CAPABILITIES)
            self._tracker = MontageLevelTracker()
            self.scheduled = 0

        def _montage_level_tracker(self):
            return self._tracker

        def _schedule_montage_cached_level_stats(self, session):
            self.scheduled += 1

    calls = []
    monkeypatch.setattr(
        level_stats,
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
        plan=SimpleNamespace(
            tiles=tuple(SimpleNamespace(source_index=index) for index in range(32))
        ),
        pending_level_tiles=deque(),
        pending_level_sources=set(),
        pending_refined_level_tiles=deque(),
        pending_refined_level_sources=set(),
        level_scan_cursor=0,
        level_scan_remaining_tiles=0,
    )
    win = Window()

    merged = win._queue_montage_level_stats_for_payloads(
        session, {index: object() for index in range(32)}
    )

    assert merged == 0
    assert calls == []
    assert len(session.pending_level_tiles) <= level_stats.MONTAGE_LEVEL_STATS_COMMIT_BATCH
    assert session.level_scan_remaining_tiles == 32
    assert win.scheduled == 1


def test_deferred_level_scan_owns_sparse_montage_indices():
    """The plan cursor must resolve canonical tile numbers, not dense ordinals."""

    from arrayscope.display.model.montage_levels import MontageLevelTracker
    from arrayscope.render.level_stats import LevelStatsService

    tile_numbers = tuple(100 + 2 * offset for offset in range(32))
    plan_tiles = tuple(
        SimpleNamespace(montage_index=tile_number, source_index=offset)
        for offset, tile_number in enumerate(tile_numbers)
    )
    rendered_tiles = {
        tile_number: SimpleNamespace(
            tile=plan_tile,
            quality="exact",
            level_stats=None,
        )
        for tile_number, plan_tile in zip(tile_numbers, plan_tiles, strict=True)
    }
    session = SimpleNamespace(
        level_key=("levels", "sparse-plan"),
        level_expected_indices=tuple(range(32)),
        plan=SimpleNamespace(tiles=plan_tiles),
        rendered_tiles=rendered_tiles,
        display_tile_payloads={},
        pending_level_tiles=deque(),
        pending_level_sources=set(),
        level_scan_cursor=0,
        level_scan_remaining_tiles=len(plan_tiles),
    )
    tracker = MontageLevelTracker()
    tracker.ensure_expected(session.level_key, session.level_expected_indices)
    service = LevelStatsService()
    service._montage_level_tracker = lambda: tracker
    service._cached_montage_source_level_stats = lambda *_args: None
    service._update_montage_level_bounds_from_prepared = lambda *_args, **_kwargs: False

    batch = service._take_montage_level_evidence_batch(
        session,
        expected=session.level_expected_indices,
        require_refined=True,
        batch_limit=8,
    )

    assert tuple(int(item.tile.source_index) for item in batch) == tuple(range(8))
    assert session.level_scan_cursor == 8
    assert session.level_scan_remaining_tiles == 24


def test_prepared_payload_level_stats_merge_without_background_sampling(monkeypatch):
    import arrayscope.render.level_stats as level_stats
    from arrayscope.display.model.montage_levels import MontageLevelTracker, TileLevelStats
    from arrayscope.window.frame_controller import FrameControllerMixin

    class Window(FrameControllerMixin):
        def __init__(self):
            from arrayscope.display.backend_contract import VISPY_CAPABILITIES

            self.win = self
            self.img_view = SimpleNamespace(rendering_capabilities=VISPY_CAPABILITIES)
            self._tracker = MontageLevelTracker()
            self.scheduled = 0

        def _montage_level_tracker(self):
            return self._tracker

        def _schedule_montage_cached_level_stats(self, session):
            self.scheduled += 1

    calls = []
    monkeypatch.setattr(
        level_stats,
        "sample_tile_level_stats",
        lambda *_args, **_kwargs: calls.append("sample") or None,
    )
    rendered = {
        index: SimpleNamespace(
            tile=SimpleNamespace(source_index=index),
            level_stats=TileLevelStats(
                index,
                (float(index), float(index + 1)),
                np.asarray([float(index)], dtype=np.float32),
            ),
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
        plan=SimpleNamespace(
            tiles=tuple(SimpleNamespace(source_index=index) for index in range(4))
        ),
        pending_level_tiles=deque(),
        pending_level_sources=set(),
        pending_refined_level_tiles=deque(),
        pending_refined_level_sources=set(),
        level_scan_cursor=0,
        level_scan_remaining_tiles=0,
    )
    win = Window()

    merged = win._queue_montage_level_stats_for_payloads(
        session, {index: object() for index in range(4)}
    )

    assert merged == 4
    assert calls == []
    assert len(session.pending_level_tiles) == 0
    assert win._tracker.summary_for(session.level_key).source_indices == frozenset(range(4))


def test_wgpu_prepared_payload_stats_bypass_resident_histogram_dispatch(monkeypatch):
    """CPU page summaries are the cheapest truthful evidence on every backend."""

    import arrayscope.render.level_stats as level_stats
    from arrayscope.display.backend_contract import WGPU_CAPABILITIES
    from arrayscope.display.model.montage_levels import MontageLevelTracker, TileLevelStats
    from arrayscope.render.level_stats import LevelStatsService

    resident_requests = []
    scheduled = []
    service = LevelStatsService()
    service.win = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=WGPU_CAPABILITIES,
            residentHistogramEvidence=lambda payloads: resident_requests.append(payloads) or (),
            acceptResidentHistogramEvidence=lambda _keys: None,
        )
    )
    tracker = MontageLevelTracker()
    service._montage_level_tracker = lambda: tracker
    service._schedule_montage_cached_level_stats = lambda session: scheduled.append(session)
    monkeypatch.setattr(
        level_stats,
        "sample_tile_level_stats",
        lambda *_args, **_kwargs: pytest.fail("prepared evidence must not resample pixels"),
    )
    rendered = {
        index: SimpleNamespace(
            tile=SimpleNamespace(source_index=index),
            quality="preview",
            level_stats=TileLevelStats(
                index,
                (float(index), float(index + 1)),
                np.asarray([float(index)], dtype=np.float32),
                evidence_quality=1,
            ),
            level_data=None,
        )
        for index in range(12)
    }
    payloads = {index: object() for index in rendered}
    session = SimpleNamespace(
        level_key=("levels", "wgpu-prepared"),
        level_expected_indices=tuple(rendered),
        rendered_tiles=rendered,
        plan=SimpleNamespace(
            tiles=tuple(
                SimpleNamespace(montage_index=index, source_index=index) for index in rendered
            )
        ),
        pending_level_tiles=deque(),
        pending_level_sources=set(),
        pending_refined_level_tiles=deque(),
        pending_refined_level_sources=set(),
        level_scan_cursor=0,
        level_scan_remaining_tiles=0,
        first_pass_histogram_published=False,
        first_pass_quality="preview",
        shader_display=True,
        display_committed=False,
        first_pass_accepts_quality=lambda quality: quality == "preview",
    )

    merged = service._queue_montage_level_stats_for_payloads(session, payloads)

    assert merged == len(rendered)
    assert resident_requests == []
    assert scheduled == [session]
    assert tracker.summary_for(session.level_key).source_indices == frozenset(rendered)


def test_wgpu_mixed_evidence_dispatches_only_sources_without_prepared_stats():
    from arrayscope.display.backend_contract import WGPU_CAPABILITIES
    from arrayscope.display.model.montage_levels import MontageLevelTracker, TileLevelStats
    from arrayscope.render.level_stats import LevelStatsService

    resident_requests = []
    service = LevelStatsService()
    service.win = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=WGPU_CAPABILITIES,
            residentHistogramEvidence=lambda payloads: (
                resident_requests.append(dict(payloads)) or ()
            ),
            acceptResidentHistogramEvidence=lambda _keys: None,
        )
    )
    tracker = MontageLevelTracker()
    service._montage_level_tracker = lambda: tracker
    service._schedule_montage_cached_level_stats = lambda _session: None
    service._cached_montage_source_level_stats = lambda *_args: None
    service._queue_wgpu_resident_histogram_evidence = lambda _session, payloads: (
        resident_requests.append(dict(payloads)) or 0
    )
    rendered = {
        index: SimpleNamespace(
            tile=SimpleNamespace(source_index=index),
            quality="preview",
            level_stats=(
                TileLevelStats(
                    index,
                    (float(index), float(index + 1)),
                    np.asarray([float(index)], dtype=np.float32),
                    evidence_quality=1,
                )
                if index != 2
                else None
            ),
            level_data=None,
        )
        for index in range(4)
    }
    payloads = {index: object() for index in rendered}
    session = SimpleNamespace(
        level_key=("levels", "wgpu-mixed"),
        level_expected_indices=tuple(rendered),
        rendered_tiles=rendered,
        plan=SimpleNamespace(
            tiles=tuple(
                SimpleNamespace(montage_index=index, source_index=index) for index in rendered
            )
        ),
        pending_level_tiles=deque(),
        pending_level_sources=set(),
        pending_refined_level_tiles=deque(),
        pending_refined_level_sources=set(),
        level_scan_cursor=0,
        level_scan_remaining_tiles=0,
        first_pass_histogram_published=False,
        first_pass_quality="preview",
        shader_display=True,
        display_committed=False,
        first_pass_accepts_quality=lambda quality: quality == "preview",
    )

    merged = service._queue_montage_level_stats_for_payloads(session, payloads)

    assert merged == 3
    assert resident_requests == [{2: payloads[2]}]


def test_level_stats_refresh_waits_for_pending_visible_upserts(monkeypatch):
    from arrayscope.core.gui_callback_budget import GuiCallbackBudget
    from arrayscope.display.model.montage_levels import MontageLevelTracker
    from arrayscope.render.level_stats import LevelStatsService

    class InlineKernel:
        visible_backlog = 0

        def submit_speculative_batch(self, *, fn, on_done, pass_token=False, **_kwargs):
            from arrayscope.kernel.task import CancellationToken

            on_done(fn(CancellationToken()) if pass_token else fn())
            return object()

    class Window(LevelStatsService):
        def __init__(self, session):
            from arrayscope.display.backend_contract import VISPY_CAPABILITIES

            self.win = self
            self.img_view = SimpleNamespace(rendering_capabilities=VISPY_CAPABILITIES)
            self.kernel = InlineKernel()
            self._frame_session = session
            self._tracker = MontageLevelTracker()
            self.applied = 0
            self.scheduled = 0

        def _frame_session_is_current(self, session):
            return session is self._frame_session

        def _montage_level_tracker(self):
            return self._tracker

        def _montage_callback_budget(self, *_args, **_kwargs):
            return GuiCallbackBudget("test", item_cap=8)

        def _record_gui_budget(self, _budget):
            return None

        def apply_montage_presentation(self, _session):
            self.applied += 1

        def _schedule_montage_cached_level_stats(self, _session):
            self.scheduled += 1

    rendered = SimpleNamespace(
        tile=SimpleNamespace(source_index=0),
        level_stats=None,
        level_data=np.asarray([0.0, 1.0], dtype=np.float32),
        histogram_data=np.asarray([0.0, 1.0], dtype=np.float32),
        image=np.asarray([0.0, 1.0], dtype=np.float32),
        slab_nbytes=8,
    )
    session = SimpleNamespace(
        key=("session",),
        session_id=1,
        level_key=("levels", "pending-upsert"),
        scheduling_policy=_refine_scheduling_policy(),
        level_revision=0,
        level_expected_indices=(0,),
        rendered_tiles={0: rendered},
        display_tile_payloads={},
        plan=SimpleNamespace(tiles=(SimpleNamespace(source_index=0),)),
        pending_level_tiles=deque([rendered]),
        pending_level_sources={0},
        pending_refined_level_tiles=deque(),
        pending_refined_level_sources=set(),
        level_scan_cursor=0,
        level_scan_remaining_tiles=0,
        dirty_payloads={},
        pending_payload_upserts={0: None},
        pending_removals=set(),
        display_committed=True,
        flush_pending=False,
        final_commit_pending=False,
        required_target_settled=lambda: True,
        visible_plan_complete=lambda: True,
    )
    win = Window(session)

    win._process_montage_cached_level_stats()

    assert win.applied == 0
    assert win.scheduled == 1


def test_preview_payload_level_data_updates_provisional_level_tracker(monkeypatch):
    import arrayscope.render.level_stats as level_stats
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.model.montage_levels import MontageLevelTracker
    from arrayscope.window.frame_controller import FrameControllerMixin

    class Window(FrameControllerMixin):
        def __init__(self):
            self.win = self
            self.img_view = SimpleNamespace(
                rendering_capabilities=ImageViewBackendCapabilities(
                    name="vispy", shader_windowing=True
                )
            )
            self._tracker = MontageLevelTracker()
            self.scheduled = 0

        def _montage_level_tracker(self):
            return self._tracker

        def _schedule_montage_cached_level_stats(self, session):
            self.scheduled += 1

    calls = []
    monkeypatch.setattr(
        level_stats,
        "sample_tile_level_stats",
        lambda *_args, **_kwargs: calls.append("sample") or None,
    )
    image = np.zeros((2, 2), dtype=np.float32)
    payload = DisplayTilePayload(
        0,
        7,
        image,
        np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        ("preview", 7),
        level_data=np.asarray([0.25, 4.0], dtype=np.float32),
        quality="preview",
    )
    session = SimpleNamespace(
        force_auto=True,
        level_key=("levels", "preview"),
        level_expected_indices=(7,),
        rendered_tiles={},
        plan=SimpleNamespace(tiles=(SimpleNamespace(montage_index=0, source_index=7),)),
        pending_level_tiles=deque(),
        pending_level_sources=set(),
        pending_refined_level_tiles=deque(),
        pending_refined_level_sources=set(),
        level_scan_cursor=0,
        level_scan_remaining_tiles=0,
    )
    win = Window()

    merged = win._queue_montage_level_stats_for_payloads(session, {0: payload})
    stats = win._tracker.summary_for(session.level_key)

    assert merged == 1
    assert calls == []
    assert stats.source_indices == frozenset({7})
    assert stats.bounds == (0.25, 4.0)
    assert stats.refined is False
    assert len(session.pending_level_tiles) == 0


def test_preview_level_evidence_is_not_promoted_to_refined_when_pyqtgraph_waits():
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.model.montage_levels import MontageLevelTracker
    from arrayscope.render.level_stats import _rendered_tile_from_previous_payload
    from arrayscope.window.frame_controller import FrameControllerMixin

    class Window(FrameControllerMixin):
        def __init__(self):
            self.win = self
            self._tracker = MontageLevelTracker()

        def _montage_level_tracker(self):
            return self._tracker

    image = np.zeros((2, 2), dtype=np.float32)
    tile = SimpleNamespace(montage_index=0, source_index=7)
    preview = DisplayTilePayload(
        0,
        7,
        image,
        image.copy(),
        ("preview", 7),
        level_data=np.asarray([1.0, 2.0], dtype=np.float32),
        quality="preview",
    )
    exact = DisplayTilePayload(
        0,
        7,
        image,
        np.asarray([[10.0, 20.0], [10.0, 20.0]], dtype=np.float32),
        ("exact", 7),
        semantic_data=image.copy(),
        semantic_histogram_data=np.asarray([[10.0, 20.0], [10.0, 20.0]], dtype=np.float32),
        level_data=np.asarray([10.0, 20.0], dtype=np.float32),
    )
    win = Window()
    key = ("levels", "preview-refined")

    win._update_montage_level_bounds_from_rendered(
        key,
        _rendered_tile_from_previous_payload(tile, preview),
        expected_indices=(7,),
        refined=True,
    )
    preview_stats = win._tracker.summary_for(key)
    win._update_montage_level_bounds_from_rendered(
        key,
        _rendered_tile_from_previous_payload(tile, exact),
        expected_indices=(7,),
        refined=True,
    )
    exact_stats = win._tracker.summary_for(key)

    assert preview_stats.bounds == (1.0, 2.0)
    assert preview_stats.refined is False
    assert exact_stats.bounds == (10.0, 20.0)
    assert exact_stats.refined is True


def test_preview_level_evidence_stays_provisional_on_shader_backend_until_exact():
    from types import SimpleNamespace

    from arrayscope.display.backend_contract import VISPY_CAPABILITIES
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.model.montage_levels import MontageLevelTracker
    from arrayscope.render.level_stats import _rendered_tile_from_previous_payload
    from arrayscope.window.frame_controller import FrameControllerMixin

    class Window(FrameControllerMixin):
        def __init__(self):
            self.win = self
            self.img_view = SimpleNamespace(rendering_capabilities=VISPY_CAPABILITIES)
            self._tracker = MontageLevelTracker()

        def _montage_level_tracker(self):
            return self._tracker

    tile = SimpleNamespace(montage_index=0, source_index=7)
    preview = DisplayTilePayload(
        0,
        7,
        np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        ("preview", 7),
        level_data=np.asarray([10.0, 40.0], dtype=np.float32),
        quality="preview",
    )
    win = Window()
    key = ("levels", "preview-refined-vispy")

    win._update_montage_level_bounds_from_rendered(
        key,
        _rendered_tile_from_previous_payload(tile, preview),
        expected_indices=(7,),
        refined=True,
    )
    preview_stats = win._tracker.summary_for(key)

    exact = DisplayTilePayload(
        0,
        7,
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32),
        ("exact", 7),
        semantic_data=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        semantic_histogram_data=np.asarray([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32),
        level_data=np.asarray([100.0, 400.0], dtype=np.float32),
    )
    win._update_montage_level_bounds_from_rendered(
        key,
        _rendered_tile_from_previous_payload(tile, exact),
        expected_indices=(7,),
        refined=True,
    )
    exact_stats = win._tracker.summary_for(key)

    assert preview_stats.refined is False
    assert preview_stats.bounds == (10.0, 40.0)
    assert exact_stats.refined is True
    assert exact_stats.bounds == (100.0, 400.0)


def test_finish_complete_montage_queues_exact_payloads_for_refined_levels():
    from arrayscope.display.backend_contract import VISPY_CAPABILITIES
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.model.montage_levels import MontageLevelTracker
    from arrayscope.render.level_stats import _rendered_tile_from_previous_payload
    from arrayscope.window.frame_controller import FrameControllerMixin

    class Window(FrameControllerMixin):
        def __init__(self):
            self.win = self
            self.img_view = SimpleNamespace(rendering_capabilities=VISPY_CAPABILITIES)
            self._tracker = MontageLevelTracker()
            self.scheduled = 0

        def _montage_level_tracker(self):
            return self._tracker

        def _frame_session_is_current(self, session):
            return True

        def _settle_montage_visible_plan_if_complete(self, session):
            return True

        def _schedule_montage_refined_level_stats(self, session):
            self.scheduled += 1

    tile = SimpleNamespace(montage_index=0, source_index=7)
    preview = DisplayTilePayload(
        0,
        7,
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        ("preview", 7),
        level_data=np.asarray([1.0, 4.0], dtype=np.float32),
        quality="preview",
    )
    exact = DisplayTilePayload(
        0,
        7,
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32),
        ("exact", 7),
        semantic_data=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        semantic_histogram_data=np.asarray([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32),
        level_data=np.asarray([100.0, 400.0], dtype=np.float32),
    )
    session = SimpleNamespace(
        scheduling_policy=_refine_scheduling_policy(),
        level_key=("levels", "finish-refined"),
        level_expected_indices=(7,),
        plan=SimpleNamespace(tiles=(tile,)),
        rendered_tiles={},
        display_tile_payloads={0: exact},
        pending_refined_level_tiles=deque(),
        pending_refined_level_sources=set(),
        is_complete=lambda: True,
    )
    win = Window()
    win._update_montage_level_bounds_from_rendered(
        session.level_key,
        _rendered_tile_from_previous_payload(tile, preview),
        expected_indices=(7,),
        refined=True,
    )

    assert win._finish_frame_session_if_complete(session) is True

    stats = win._tracker.summary_for(session.level_key)
    assert stats.refined is False
    assert len(session.pending_refined_level_tiles) == 1
    assert session.pending_refined_level_tiles[0].quality == "exact"
    assert win.scheduled == 1


def test_finish_complete_montage_seeds_rough_levels_from_display_payloads():
    from arrayscope.display.backend_contract import VISPY_CAPABILITIES
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.model.montage_levels import MontageLevelTracker
    from arrayscope.window.frame_controller import FrameControllerMixin

    class Window(FrameControllerMixin):
        def __init__(self):
            self.win = self
            self.img_view = SimpleNamespace(rendering_capabilities=VISPY_CAPABILITIES)
            self._tracker = MontageLevelTracker()
            self.cached_schedules = 0
            self.refined_schedules = 0

        def _montage_level_tracker(self):
            return self._tracker

        def _frame_session_is_current(self, session):
            return True

        def _settle_montage_visible_plan_if_complete(self, session):
            return True

        def _schedule_montage_cached_level_stats(self, session):
            self.cached_schedules += 1

        def _schedule_montage_refined_level_stats(self, session):
            self.refined_schedules += 1

    tile = SimpleNamespace(montage_index=0, source_index=7)
    preview = DisplayTilePayload(
        0,
        7,
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        ("preview", 7),
        level_data=np.asarray([1.0, 4.0], dtype=np.float32),
        quality="preview",
    )
    session = SimpleNamespace(
        scheduling_policy=_refine_scheduling_policy(),
        level_key=("levels", "finish-current"),
        level_expected_indices=(7,),
        plan=SimpleNamespace(tiles=(tile,)),
        rendered_tiles={},
        display_tile_payloads={0: preview},
        pending_level_tiles=deque(),
        pending_level_sources=set(),
        pending_refined_level_tiles=deque(),
        pending_refined_level_sources=set(),
        level_scan_cursor=0,
        level_scan_remaining_tiles=0,
        force_auto=True,
        is_complete=lambda: True,
    )
    win = Window()

    assert win._finish_frame_session_if_complete(session) is True

    stats = win._tracker.summary_for(session.level_key)
    assert stats.bounds == (1.0, 4.0)
    assert stats.refined is False
    assert stats.source_indices == frozenset({7})
    assert session.pending_refined_level_tiles == deque()
    assert win.cached_schedules == 1
    assert win.refined_schedules == 1


def test_preview_payloads_do_not_count_as_semantic_commits():
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.window.frame_effects import (
        tiled_payloads_can_commit_frame,
        tiled_payloads_include_semantics,
    )

    image = np.zeros((2, 2), dtype=np.float32)
    preview = DisplayTilePayload(
        0,
        0,
        image,
        image.copy(),
        ("preview", 0),
        quality="preview",
    )
    exact = DisplayTilePayload(
        1,
        1,
        image,
        image.copy(),
        ("exact", 1),
        semantic_data=image.copy(),
    )
    reduced_presentation_only = SimpleNamespace(
        quality="exact",
        semantic_data=None,
        page_backing=SimpleNamespace(resolved_page_set=object()),
    )

    assert tiled_payloads_include_semantics({0: preview}) is False
    assert tiled_payloads_include_semantics({0: reduced_presentation_only}) is False
    assert tiled_payloads_include_semantics({0: preview, 1: exact}) is True
    assert tiled_payloads_can_commit_frame({0: preview}) is False
    assert tiled_payloads_can_commit_frame({0: reduced_presentation_only}) is True
    assert tiled_payloads_can_commit_frame({0: preview, 1: exact}) is True


def test_montage_level_metadata_publishes_when_evidence_quality_improves():
    from arrayscope.core.window_levels import LevelSource, LevelSourceRank
    from arrayscope.display.model.montage_levels import LevelEvidenceQuality, MontageLevelStats
    from arrayscope.window.frame_controller import FrameControllerMixin

    class Window(FrameControllerMixin):
        def __init__(self):
            self.win = self

    session = SimpleNamespace(
        level_key=("levels", "quality"),
        rendered_tiles={0: object()},
        display_tile_payloads={},
        applied_level_source=LevelSource(
            levels=(0.0, 10.0),
            histogram_range=(0.0, 10.0),
            rank=LevelSourceRank.MONTAGE_COMPLETE,
            source_count=2,
            expected_count=2,
            semantic_key=("levels", "quality"),
            evidence_quality=int(LevelEvidenceQuality.ROUGH_PREVIEW),
        ),
    )
    stats = MontageLevelStats(
        bounds=(0.0, 10.0),
        source_indices=frozenset({0, 1}),
        expected_indices=frozenset({0, 1}),
        rank=LevelSourceRank.MONTAGE_COMPLETE,
        evidence_quality=LevelEvidenceQuality.ROUGH_TARGET,
    )

    assert Window()._should_publish_montage_level_metadata(session, stats) is True


def test_vispy_first_pass_level_metadata_publishes_on_rough_coverage_growth():
    from arrayscope.core.window_levels import LevelSource, LevelSourceRank
    from arrayscope.display.model.montage_levels import LevelEvidenceQuality, MontageLevelStats
    from arrayscope.window.frame_controller import FrameControllerMixin

    class Window(FrameControllerMixin):
        def __init__(self):
            self.win = self

    level_key = ("levels", "first-pass-growth")
    session = SimpleNamespace(
        level_key=level_key,
        rendered_tiles={},
        display_tile_payloads={0: object()},
        shader_display=True,
        first_pass_histogram_published=False,
        applied_level_source=LevelSource(
            levels=(100.0, 200.0),
            histogram_range=(100.0, 200.0),
            rank=LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
            source_count=1,
            expected_count=3,
            semantic_key=level_key,
            evidence_quality=int(LevelEvidenceQuality.ROUGH_PREVIEW),
        ),
    )
    stats = MontageLevelStats(
        bounds=(100.0, 500.0),
        source_indices=frozenset({0, 1}),
        expected_indices=frozenset({0, 1, 2}),
        rank=LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
        evidence_quality=LevelEvidenceQuality.ROUGH_PREVIEW,
    )

    assert Window()._should_publish_montage_level_metadata(session, stats) is True
    session.first_pass_histogram_published = True
    assert Window()._should_publish_montage_level_metadata(session, stats) is False


def _shader_level_session(level_key, *, applied_level_source=None):
    return SimpleNamespace(
        level_key=level_key,
        level_expected_indices=(7,),
        first_pass_histogram_published=False,
        applied_level_source=applied_level_source,
    )


def _settled_shader_levels(window, session, evidence_batches):
    """Replay commits the way _commit_tile_layer decides shader levels."""

    from arrayscope.core.window_levels import WindowLevelController
    from arrayscope.window.frame_effects import shader_commit_level_source

    tracker = window._montage_level_tracker()
    for published, stats in evidence_batches:
        session.first_pass_histogram_published = bool(published)
        if stats is not None:
            tracker.update_from_stats(session.level_key, stats, aggregate=False)
        candidate = shader_commit_level_source(window, session)
        if candidate is None:
            continue
        state = WindowLevelController().decide(
            previous=getattr(session, "applied_level_source", None),
            candidate=candidate,
            explicit_auto=False,
            mode="relative",
        )
        session.applied_level_source = state.as_level_source()
    return getattr(session.applied_level_source, "levels", None)


def test_shader_commit_gate_offers_mature_target_quality_after_publication():
    # The retention contract settles a window on the first COMPLETE population
    # at target quality (evidence_quality >= ROUGH_TARGET); full refinement
    # only improves it later (and on wgpu may never run at all). After the
    # first-pass histogram publishes, the commit gate must therefore offer a
    # mature target-quality summary instead of withholding everything short of
    # REFINED — withholding froze whatever anchored first.
    from arrayscope.display.model.montage_levels import LevelEvidenceQuality, TileLevelStats
    from arrayscope.render.level_stats import LevelStatsService
    from arrayscope.window.frame_effects import shader_commit_level_source

    window = LevelStatsService()
    window.win = window
    level_key = ("montage_levels", "doc", "slice-7", 0)
    session = _shader_level_session(level_key)
    session.first_pass_histogram_published = True
    window._montage_level_tracker().ensure_expected(level_key, (7,))
    window._montage_level_tracker().update_from_stats(
        level_key,
        TileLevelStats(
            7,
            (0.0, 100.0),
            np.linspace(0.0, 100.0, 64, dtype=np.float32),
            refined=False,
            evidence_quality=LevelEvidenceQuality.ROUGH_TARGET,
        ),
        aggregate=False,
    )

    candidate = shader_commit_level_source(window, session)

    assert candidate is not None
    assert candidate.levels == (0.0, 100.0)
    assert candidate.evidence_quality == int(LevelEvidenceQuality.ROUGH_TARGET)


def test_shader_settled_levels_are_navigation_path_independent():
    # Field defect 2026-07-24: reloading directly at a slice anchored the
    # window on preview/reduced-LOD evidence (averaging narrows the extremes)
    # and never re-anchored when the mature exact evidence arrived, while
    # scrolling to the same slice settled on the mature evidence — visibly
    # more clipping on the reload path. Settled levels must not depend on how
    # the user navigated to the data.
    from arrayscope.core.window_levels import LevelSource, LevelSourceRank
    from arrayscope.display.model.montage_levels import LevelEvidenceQuality, TileLevelStats
    from arrayscope.render.level_stats import LevelStatsService

    level_key = ("montage_levels", "doc", "slice-7", 0)

    def evidence_batches():
        preview = TileLevelStats(
            7,
            (45.0, 55.0),
            np.linspace(45.0, 55.0, 64, dtype=np.float32),
            refined=False,
            evidence_quality=LevelEvidenceQuality.ROUGH_PREVIEW,
        )
        exact = TileLevelStats(
            7,
            (0.0, 100.0),
            np.linspace(0.0, 100.0, 64, dtype=np.float32),
            refined=False,
            evidence_quality=LevelEvidenceQuality.ROUGH_TARGET,
        )
        # The preview commit anchors before the first-pass histogram
        # publishes; the exact evidence lands only afterwards (the trace-
        # confirmed cold-load ordering on wgpu).
        return ((False, preview), (True, exact))

    def fresh_window():
        window = LevelStatsService()
        window.win = window
        window._montage_level_tracker().ensure_expected(level_key, (7,))
        return window

    direct_session = _shader_level_session(level_key)
    direct = _settled_shader_levels(fresh_window(), direct_session, evidence_batches())

    predecessor = LevelSource(
        levels=(-20.0, 140.0),
        histogram_range=(-20.0, 140.0),
        rank=LevelSourceRank.MONTAGE_COMPLETE,
        source_count=1,
        expected_count=1,
        semantic_key=("montage_levels", "doc", "slice-6", 0),
        evidence_quality=int(LevelEvidenceQuality.ROUGH_TARGET),
    )
    scroll_session = _shader_level_session(level_key, applied_level_source=predecessor)
    scroll = _settled_shader_levels(fresh_window(), scroll_session, evidence_batches())

    assert direct == (0.0, 100.0)
    assert scroll == (0.0, 100.0)
    assert direct == scroll


def test_first_pass_rough_evidence_completion_uses_required_scope():
    from arrayscope.render.level_stats import LevelStatsService

    tiles = tuple(
        SimpleNamespace(montage_index=index, source_index=100 + index) for index in range(4)
    )
    summary = SimpleNamespace(source_indices=frozenset({100, 101}))
    service = SimpleNamespace(
        _montage_level_tracker=lambda: SimpleNamespace(
            summary_for=lambda _key: summary,
        )
    )
    session = SimpleNamespace(
        shader_display=True,
        level_key=("levels", "required-first-pass"),
        plan=SimpleNamespace(tiles=tiles),
        first_pass_pixels_presented=lambda: True,
        required_tile_numbers=lambda: (0, 1),
    )

    assert LevelStatsService._first_pass_level_evidence_complete(service, session)


def test_display_tile_payload_retains_prepared_level_stats_for_reuse():
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.model.montage_levels import TileLevelStats
    from arrayscope.window.frame_controller import _rendered_tile_from_previous_payload

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


def test_pyqtgraph_first_pixels_wait_for_complete_semantic_source():
    from arrayscope.core.window_levels import LevelSourceRank
    from arrayscope.display.model.montage_levels import LevelEvidenceQuality, MontageLevelStats
    from arrayscope.window.frame_effects import tile_layer_first_pixels_wait_for_level_source

    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph")
        )
    )
    window.win = window
    session = SimpleNamespace(
        loading_tiles=set(),
        active_tile_requests=set(),
        pending_level_tiles=deque([object()]),
        level_scan_remaining_tiles=0,
        plan=SimpleNamespace(
            tiles=(
                SimpleNamespace(montage_index=0, source_index=0),
                SimpleNamespace(montage_index=1, source_index=1),
            )
        ),
        required_tile_numbers=lambda: (0, 1),
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
        evidence_quality=LevelEvidenceQuality.ROUGH_TARGET,
    )
    sampled_full = MontageLevelStats(
        bounds=(0.0, 1.0),
        source_indices=frozenset({0, 1}),
        expected_indices=frozenset({0, 1}),
        rank=LevelSourceRank.MONTAGE_SAMPLED_FULL,
        refined=True,
    )

    assert tile_layer_first_pixels_wait_for_level_source(window, session, True, partial) is True
    assert tile_layer_first_pixels_wait_for_level_source(window, session, True, complete) is True
    assert (
        tile_layer_first_pixels_wait_for_level_source(window, session, True, sampled_full) is False
    )


def test_pyqtgraph_first_pixels_accept_refined_required_subset_honestly():
    from arrayscope.core.window_levels import LevelSourceRank
    from arrayscope.display.model.montage_levels import (
        LevelEvidenceQuality,
        MontageLevelStats,
    )
    from arrayscope.window.frame_effects import (
        tile_layer_first_pixels_wait_for_level_source,
    )

    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph")
        )
    )
    window.win = window
    tiles = tuple(
        SimpleNamespace(montage_index=tile_number, source_index=source_index)
        for tile_number, source_index in ((10, 100), (20, 200), (30, 300))
    )
    session = SimpleNamespace(
        plan=SimpleNamespace(tiles=tiles),
        required_tile_numbers=lambda: (10, 30),
    )
    missing_required = MontageLevelStats(
        bounds=(0.0, 1.0),
        source_indices=frozenset({100, 200}),
        expected_indices=frozenset({100, 200, 300}),
        rank=LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
        refined=True,
        evidence_quality=LevelEvidenceQuality.REFINED,
    )
    required_subset = MontageLevelStats(
        bounds=(0.0, 1.0),
        source_indices=frozenset({100, 300}),
        expected_indices=frozenset({100, 200, 300}),
        rank=LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
        refined=True,
        evidence_quality=LevelEvidenceQuality.REFINED,
    )

    assert tile_layer_first_pixels_wait_for_level_source(window, session, True, missing_required)
    assert not tile_layer_first_pixels_wait_for_level_source(window, session, True, required_subset)


def test_pyqtgraph_first_pixels_accept_provisional_refined_first_batch():
    """A cold scope larger than one evidence batch presents on the first batch.

    272-source montage entry held every evaluated floor behind the full
    refined sweep (~7 s black window, 2026-07-18 dossier). One refined batch
    (MONTAGE_LEVEL_STATS_FIRST_CPU_BATCH sources) is an honest provisional
    window for the whole frame; anything less, or rough-only evidence, still
    waits.
    """

    from arrayscope.core.window_levels import LevelSourceRank
    from arrayscope.display.model.montage_levels import (
        MONTAGE_LEVEL_STATS_FIRST_CPU_BATCH,
        LevelEvidenceQuality,
        MontageLevelStats,
    )
    from arrayscope.window.frame_effects import (
        tile_layer_first_pixels_wait_for_level_source,
    )

    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph")
        )
    )
    window.win = window
    tile_count = 272
    tiles = tuple(
        SimpleNamespace(montage_index=number, source_index=number) for number in range(tile_count)
    )
    session = SimpleNamespace(
        plan=SimpleNamespace(tiles=tiles),
        required_tile_numbers=lambda: tuple(range(tile_count)),
    )
    expected = frozenset(range(tile_count))
    batch = int(MONTAGE_LEVEL_STATS_FIRST_CPU_BATCH)

    def summary(count, *, refined):
        return MontageLevelStats(
            bounds=(0.0, 1.0),
            source_indices=frozenset(range(count)),
            expected_indices=expected,
            rank=LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
            refined=refined,
            evidence_quality=(
                LevelEvidenceQuality.REFINED if refined else LevelEvidenceQuality.ROUGH_TARGET
            ),
        )

    assert (
        tile_layer_first_pixels_wait_for_level_source(
            window, session, True, summary(batch - 1, refined=True)
        )
        is True
    )
    assert (
        tile_layer_first_pixels_wait_for_level_source(
            window, session, True, summary(batch, refined=False)
        )
        is True
    )
    assert (
        tile_layer_first_pixels_wait_for_level_source(
            window, session, True, summary(batch, refined=True)
        )
        is False
    )


def test_first_cpu_histogram_publishes_provisional_refined_first_batch():
    """The provisional window source publishes to the histogram/levels widgets.

    The first CPU pixels are windowed with the refined first batch, so the
    widgets must carry that same source; sub-batch or rough-only coverage
    stays unpublished.
    """

    import numpy as np

    from arrayscope.display.model.montage_levels import (
        MONTAGE_LEVEL_STATS_FIRST_CPU_BATCH,
        LevelEvidenceQuality,
        MontageLevelTracker,
        TileLevelStats,
    )
    from arrayscope.render.level_stats import LevelStatsService

    batch = int(MONTAGE_LEVEL_STATS_FIRST_CPU_BATCH)

    def publish_with(covered, *, quality):
        published = []
        tracker = MontageLevelTracker()
        level_key = ("levels", "provisional")
        tracker.ensure_expected(level_key, range(272))
        for source_index in range(covered):
            tracker.update_from_stats(
                level_key,
                TileLevelStats(
                    source_index=source_index,
                    bounds=(0.0, float(source_index + 1)),
                    sample=np.asarray([0.0, float(source_index + 1)], dtype=np.float32),
                    refined=quality >= LevelEvidenceQuality.REFINED,
                    evidence_quality=quality,
                ),
            )
        service = LevelStatsService()
        service.win = SimpleNamespace(
            img_view=SimpleNamespace(
                rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph"),
                applyHistogramMetadata=lambda **kwargs: (published.append(kwargs), True)[1],
            )
        )
        service._montage_level_tracker = lambda: tracker
        service._should_publish_montage_level_metadata = lambda _session, _summary: True
        service._schedule_montage_histogram_aggregate = lambda _session: True
        session = SimpleNamespace(
            level_key=level_key,
            display_committed=False,
            histogram_aggregate_inflight=False,
        )
        result = service._publish_first_cpu_histogram(session)
        return result, published

    result, published = publish_with(batch, quality=LevelEvidenceQuality.REFINED)
    assert result is True
    assert len(published) == 1
    assert published[0]["levels"] == (0.0, float(batch))

    result, published = publish_with(batch - 1, quality=LevelEvidenceQuality.REFINED)
    assert result is False
    assert published == []

    result, published = publish_with(batch, quality=LevelEvidenceQuality.ROUGH_TARGET)
    assert result is False
    assert published == []


def test_shader_first_pixels_wait_for_rough_source_but_not_complete_source():
    from arrayscope.core.window_levels import LevelSourceRank
    from arrayscope.display.model.montage_levels import LevelEvidenceQuality, MontageLevelStats
    from arrayscope.window.frame_effects import tile_layer_first_pixels_wait_for_level_source

    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(name="vispy", shader_windowing=True)
        )
    )
    window.win = window
    session = SimpleNamespace(pending_level_tiles=deque([object()]), level_scan_remaining_tiles=1)
    partial = MontageLevelStats(
        bounds=(0.0, 1.0),
        source_indices=frozenset({0}),
        expected_indices=frozenset({0, 1}),
        rank=LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
        evidence_quality=LevelEvidenceQuality.ROUGH_PREVIEW,
    )

    assert tile_layer_first_pixels_wait_for_level_source(window, session, True, None) is True
    assert tile_layer_first_pixels_wait_for_level_source(window, session, True, partial) is False


def test_refined_evidence_resumes_parked_first_commit_with_dirty_payloads():
    from arrayscope.render.level_stats import LevelStatsService

    requested = []

    session = SimpleNamespace(
        pending_level_tiles=deque(),
        level_scan_remaining_tiles=0,
        flush_pending=True,
        final_commit_pending=True,
        dirty_payloads={0: None},
        pending_payload_upserts={0: object()},
        pending_removals=set(),
        pipeline=SimpleNamespace(
            effects=SimpleNamespace(request_presentation=lambda: requested.append(True))
        ),
    )

    LevelStatsService()._maybe_publish_after_level_evidence(session, processed=1)

    assert requested == [True]


@pytest.mark.parametrize("shader_windowing", [False, True])
def test_first_display_level_scan_continuation_uses_visible_lane(shader_windowing):
    from arrayscope.kernel import Lane, Priority
    from arrayscope.render.level_stats import (
        FIRST_PIXEL_EVIDENCE_SCHEDULING_RANK,
        LevelStatsService,
    )

    submitted = []

    class Kernel:
        def submit(self, spec, **callbacks):
            submitted.append((spec, callbacks))
            return object()

        def submit_speculative_batch(self, **_kwargs):
            raise AssertionError(
                "first-frame correctness continuation must not use a speculative lane"
            )

    image_view = SimpleNamespace(
        rendering_capabilities=ImageViewBackendCapabilities(
            name="vispy" if shader_windowing else "pyqtgraph",
            shader_windowing=shader_windowing,
        )
    )
    service = LevelStatsService()
    service.win = SimpleNamespace(kernel=Kernel(), img_view=image_view)
    session = SimpleNamespace(
        key=("frame",),
        session_id=1,
        viewport_revision=0,
        level_key=("levels",),
        force_auto=True,
        user_levels_override=None,
        display_committed=False,
        scheduling_policy=_coverage_scheduling_policy(),
        level_evidence_inflight=False,
    )
    service._frame_session = session

    service._invite_montage_level_evidence_continuation(session)

    assert len(submitted) == 1
    spec, _callbacks = submitted[0]
    assert spec.lane == Lane.DISPLAY_PREVIEW
    # The continuation is the complement producer for the first-pixel wait:
    # at VISIBLE_IMAGE/UNRANKED it queued behind every INTERACTIVE-priority
    # tile evaluation, re-adding the full fill backlog to each evidence turn
    # (montage-entry blackout, 2026-07-18 dossier).
    assert spec.priority == Priority.INTERACTIVE
    assert spec.scheduling_rank == FIRST_PIXEL_EVIDENCE_SCHEDULING_RANK
    assert spec.presentation_phase == 1
    assert spec.coverage_pass_open is True


def test_first_shader_payload_level_evidence_uses_visible_lane():
    from arrayscope.kernel import Lane, Priority
    from arrayscope.render.level_stats import (
        FIRST_PIXEL_EVIDENCE_SCHEDULING_RANK,
        LevelStatsService,
    )

    submitted = []

    class Kernel:
        def submit(self, spec, **callbacks):
            submitted.append((spec, callbacks))
            return object()

        def submit_speculative_batch(self, **_kwargs):
            raise AssertionError("first-frame shader evidence must not use a speculative lane")

    service = LevelStatsService()
    service.win = SimpleNamespace(
        kernel=Kernel(),
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(name="vispy", shader_windowing=True)
        ),
    )
    session = SimpleNamespace(
        key=("frame",),
        session_id=1,
        viewport_revision=0,
        level_key=("levels",),
        force_auto=True,
        user_levels_override=None,
        display_committed=False,
        scheduling_policy=_coverage_scheduling_policy(),
        level_evidence_inflight=False,
        pending_level_tiles=deque([object()]),
        level_scan_remaining_tiles=0,
    )
    service._frame_session = session
    service._frame_session_is_current = lambda candidate: candidate is session
    service._montage_level_expected_indices = lambda _session: ()
    service._take_montage_level_evidence_batch = lambda *_args, **_kwargs: (object(),)

    service._process_montage_cached_level_stats()

    assert len(submitted) == 1
    spec, _callbacks = submitted[0]
    assert spec.lane == Lane.DISPLAY_PREVIEW
    assert spec.priority == Priority.INTERACTIVE
    assert spec.scheduling_rank == FIRST_PIXEL_EVIDENCE_SCHEDULING_RANK
    assert spec.presentation_phase == 1
    assert spec.coverage_pass_open is True


def test_wgpu_resident_histogram_evidence_uses_coverage_lane_and_shared_tracker():
    from arrayscope.display.backend_contract import WGPU_CAPABILITIES
    from arrayscope.display.model.montage_levels import (
        LevelEvidenceQuality,
        MontageLevelTracker,
    )
    from arrayscope.kernel import UNRANKED_SCHEDULING_RANK, Lane, Priority
    from arrayscope.render.level_stats import LevelStatsService

    submitted = []
    accepted = []
    requested = []
    waited = []

    class Kernel:
        def submit(self, spec, **callbacks):
            submitted.append((spec, callbacks))
            return object()

    class Readback:
        def resolve(self):
            assert waited == [True]
            return np.asarray([1, 2, 1], dtype=np.uint32), (-2.0, 4.0)

    evidence = SimpleNamespace(
        evidence_key=("resident", 7),
        source_index=7,
        wait_completed=lambda: waited.append(True),
        readback=Readback(),
    )

    def waiting_evidence(_payloads):
        return () if evidence.evidence_key in accepted else (evidence,)

    view = SimpleNamespace(
        rendering_capabilities=WGPU_CAPABILITIES,
        residentHistogramEvidence=waiting_evidence,
        acceptResidentHistogramEvidence=lambda keys: accepted.extend(keys),
    )
    session = SimpleNamespace(
        key=("frame",),
        session_id=3,
        viewport_revision=5,
        level_key=("levels", "wgpu-resident"),
        level_expected_indices=(7,),
        plan=SimpleNamespace(tiles=(SimpleNamespace(source_index=7),)),
        scheduling_policy=_coverage_scheduling_policy(),
        level_evidence_inflight=False,
        level_evidence_generation=None,
        first_pass_histogram_published=False,
        first_pass_quality="preview",
        flush_pending=False,
        final_commit_pending=False,
        pipeline=SimpleNamespace(
            effects=SimpleNamespace(
                request_presentation=lambda: requested.append(True),
            )
        ),
    )
    tracker = MontageLevelTracker()
    service = LevelStatsService()
    service.win = SimpleNamespace(kernel=Kernel(), img_view=view)
    service._frame_session = session
    service._montage_level_tracker = lambda: tracker
    service._remember_montage_source_level_stats = lambda *_args: None
    service._maybe_publish_after_level_evidence = lambda *_args, **_kwargs: None

    queued = service._queue_montage_level_stats_for_payloads(session, {0: object()})

    assert queued == 1
    assert len(submitted) == 1
    spec, callbacks = submitted[0]
    assert spec.lane == Lane.DISPLAY_PREVIEW
    assert spec.priority == Priority.VISIBLE_IMAGE
    assert spec.scheduling_rank == UNRANKED_SCHEDULING_RANK
    assert spec.presentation_phase == 1
    assert spec.coverage_pass_open is True
    from arrayscope.kernel.task import CancellationToken

    callbacks["on_done"](spec.fn(CancellationToken()) if spec.pass_token else spec.fn())

    stats = tracker.source_stats(session.level_key, 7)
    assert stats is not None
    assert stats.bounds == (-2.0, 4.0)
    assert stats.sample.size == 4
    assert stats.evidence_quality == LevelEvidenceQuality.ROUGH_PREVIEW
    assert accepted == [evidence.evidence_key]
    assert session.level_evidence_inflight is False
    assert session.flush_pending is True
    assert session.final_commit_pending is True
    assert requested == [True]


def test_wgpu_resident_histogram_evidence_defers_while_interaction_is_active():
    from arrayscope.render.level_stats import LevelStatsService

    resolved = []
    submitted = []
    evidence = SimpleNamespace(
        evidence_key=("resident", 7),
        source_index=7,
        wait_completed=lambda: resolved.append("waited"),
        readback=SimpleNamespace(resolve=lambda: resolved.append("resolved")),
    )
    session = SimpleNamespace(
        key=("frame",),
        session_id=3,
        viewport_revision=5,
        level_key=("levels", "wgpu-resident"),
        level_expected_indices=(7,),
        plan=SimpleNamespace(tiles=(SimpleNamespace(source_index=7),)),
        scheduling_policy=_coverage_scheduling_policy(),
        level_evidence_inflight=False,
        first_pass_histogram_published=False,
    )
    service = LevelStatsService()
    service.win = SimpleNamespace(
        _viewport_interaction_active=True,
        kernel=SimpleNamespace(submit=lambda *args, **kwargs: submitted.append(args)),
        img_view=SimpleNamespace(
            residentHistogramEvidence=lambda _payloads: (evidence,),
            acceptResidentHistogramEvidence=lambda _keys: None,
        ),
    )
    service._montage_level_tracker = lambda: SimpleNamespace(
        ensure_expected=lambda *_args: None,
    )

    queued = service._queue_wgpu_resident_histogram_evidence(session, {0: object()})

    assert queued == 0
    assert submitted == []
    assert resolved == []
    assert session._wgpu_histogram_evidence_deferred is True
    # The deferral is still phase-1 evidence debt: the coverage barrier arms.
    assert session.scheduling_policy.evidence_pending_calls == [True]


def test_wgpu_histogram_rearm_queues_acknowledged_evidence_before_noop_commit():
    """A tile can become resident while the preceding evidence batch runs.

    Its first queue attempt bails as in-flight.  Once that batch releases, a
    settled presentation has no backend report to pump the queue, so the
    evidence owner must re-scan the acknowledged population directly.
    """

    from arrayscope.render.level_stats import LevelStatsService

    requested = []
    acknowledged = {0: object(), 1: object()}
    session = SimpleNamespace(
        level_evidence_inflight=False,
        tile_presentation_state=SimpleNamespace(payloads=acknowledged),
        flush_pending=False,
        pipeline=SimpleNamespace(
            effects=SimpleNamespace(request_presentation=lambda: requested.append(True))
        ),
    )
    service = LevelStatsService()
    service._frame_session = session
    queued = []

    def queue(owner, payloads):
        queued.append((owner, payloads))
        owner.level_evidence_inflight = True
        return 1

    service._queue_wgpu_resident_histogram_evidence = queue

    service._rearm_wgpu_histogram_evidence()

    assert queued == [(session, acknowledged)]
    assert requested == []
    assert session.flush_pending is False


def test_wgpu_histogram_rearm_wakes_after_synchronous_evidence_reuse():
    """Retained evidence is progress, but it does not own a future wakeup.

    Attached crop/axis-switch stall ``/tmp/arrayscope-stall-334-2`` ended
    with 11 required preview tiles, zero work, and COVERAGE still open.
    The final histogram completion synchronously reused two retained exact
    rows; the integer return was mistaken for an asynchronous dispatch and
    the parked publication commit was never requested.
    """

    from arrayscope.render.level_stats import LevelStatsService

    requested = []
    acknowledged = {0: object(), 1: object()}
    session = SimpleNamespace(
        level_evidence_inflight=False,
        tile_presentation_state=SimpleNamespace(payloads=acknowledged),
        flush_pending=False,
        pipeline=SimpleNamespace(
            effects=SimpleNamespace(request_presentation=lambda: requested.append(True))
        ),
    )
    service = LevelStatsService()
    service._frame_session = session
    service._queue_wgpu_resident_histogram_evidence = lambda _owner, _payloads: 2

    service._rearm_wgpu_histogram_evidence()

    assert requested == [True]
    assert session.flush_pending is True


def test_deferred_cold_histogram_obligation_holds_coverage_and_dispatches_on_quiet_edge():
    """Codex review 2026-07-19 finding 1: the coverage-evidence bypass.

    A COLD histogram obligation deferred during interaction is still phase-1
    evidence debt. Without the barrier, first pixels close COVERAGE
    evidence-empty; the quiet-edge forced commit then runs in REFINE, where
    dispatch is gated off, and the rough histogram never runs — violating the
    progressive contract (phase 1 owns rough evidence; the phase owner is
    ``ProgressiveSchedulingPolicy``).

    Real phase owner, real deferral bail, real evidence-obligation
    configuration, real quiet-edge replan. Only the view/backend seam is
    modeled: dispatch happens iff the obligation is configured required, and
    evidence rows exist only after a dispatch.
    """

    from arrayscope.display.backend_contract import WGPU_CAPABILITIES
    from arrayscope.render.level_stats import LevelStatsService
    from arrayscope.render.progressive_scheduling import ProgressiveSchedulingPolicy
    from arrayscope.window.frame_effects import FramePipelineEffects
    from arrayscope.window.frame_runtime import FrameRuntimeMixin

    policy = ProgressiveSchedulingPolicy()
    assert policy.retarget(("scope", 1), (7,), progressive=True)
    assert policy.verdict.coverage_open

    class ViewSeam:
        rendering_capabilities = WGPU_CAPABILITIES

        def __init__(self):
            self.required_calls = []
            self.dispatched = False
            self.accepted = []

        def setResidentHistogramEvidenceRequired(self, required, obligation=None):
            self.required_calls.append((bool(required), obligation))
            if required:
                self.dispatched = True

        def residentHistogramEvidence(self, _payloads):
            if not self.dispatched:
                return ()
            return (
                SimpleNamespace(
                    evidence_key=("resident", 7),
                    source_index=7,
                    wait_completed=lambda: None,
                    readback=SimpleNamespace(
                        resolve=lambda: (
                            np.asarray([1, 2, 1], dtype=np.uint32),
                            (-2.0, 4.0),
                        )
                    ),
                ),
            )

        def acceptResidentHistogramEvidence(self, keys):
            self.accepted.extend(keys)

    view = ViewSeam()
    submitted = []
    win = SimpleNamespace(
        img_view=view,
        _viewport_interaction_active=True,
        kernel=SimpleNamespace(
            submit=lambda spec, **callbacks: submitted.append((spec, callbacks)) or object()
        ),
    )
    win.win = win
    session = SimpleNamespace(
        key=("frame",),
        session_id=3,
        viewport_revision=5,
        level_key=("levels", "wgpu-resident"),
        level_expected_indices=(7,),
        plan=SimpleNamespace(tiles=(SimpleNamespace(source_index=7),)),
        scheduling_policy=policy,
        level_evidence_inflight=False,
        level_evidence_generation=None,
        first_pass_histogram_published=False,
        first_pass_quality="preview",
        flush_pending=False,
        final_commit_pending=False,
        lifecycle=SimpleNamespace(first_pixels_presented=lambda _tiles: True),
        _interactive_residency_deferred=False,
        pipeline=SimpleNamespace(
            counters=SimpleNamespace(interactive_native_deferred=0),
        ),
    )
    renderer = SimpleNamespace(win=win)
    effects = FramePipelineEffects(renderer, session)
    service = LevelStatsService()
    service.win = win
    service._frame_session = session
    service._montage_level_tracker = lambda: SimpleNamespace(
        ensure_expected=lambda *_args: None,
        update_from_stats=lambda *_args, **_kwargs: None,
    )
    service._remember_montage_source_level_stats = lambda *_args: None
    service._maybe_publish_after_level_evidence = lambda *_args, **_kwargs: None

    # -- gesture: the cold obligation is deferred, no dispatch happens -------
    effects._configure_wgpu_evidence_obligation()
    assert view.dispatched is False
    assert service._queue_montage_level_stats_for_payloads(session, {0: object()}) == 0
    assert session._wgpu_histogram_evidence_deferred is True

    # First pixels present during the gesture. The deferred cold obligation is
    # phase-1 evidence debt: the barrier must hold COVERAGE open.
    assert policy.observe(session.lifecycle) is False
    assert policy.verdict.coverage_open, (
        "COVERAGE closed evidence-empty while a cold histogram obligation was deferred"
    )

    # -- quiet edge: the interaction stops and the forced commit re-runs the
    # evidence configuration and the level-stats producer, exactly as the
    # real retarget-driven commit does.
    win._viewport_interaction_active = False
    quiet_edge_queued = []
    settle_pumped = []

    def forced_commit(current, *, force_commit=False):
        assert force_commit is True
        assert current.backend_refresh_pending is True
        effects._configure_wgpu_evidence_obligation()
        quiet_edge_queued.append(
            service._queue_montage_level_stats_for_payloads(current, {0: object()})
        )

    def settle_pump(current, payloads):
        # The composed design (settle-edge pump, 1b300833) dispatches the
        # deferred evidence BEFORE the forced commit — strictly earlier than
        # the quiet-edge-only flow this test originally encoded.
        settle_pumped.append(service._queue_montage_level_stats_for_payloads(current, payloads))

    session.tile_presentation_state = SimpleNamespace(payloads={0: object()})
    owner = SimpleNamespace(
        _frame_session=session,
        _frame_session_is_current=lambda candidate: candidate is session,
        retarget_frame_pipeline=forced_commit,
        _queue_montage_level_stats_for_payloads=settle_pump,
    )
    assert FrameRuntimeMixin.replan_deferred_interactive_native_quality(owner)

    # Two dispatch edges exist in the composed design and BOTH sub-cases are
    # contract-legal with exactly one dispatch total: (a) an already-installed
    # obligation at the fill tail dispatches at the settle-edge pump (the
    # 1b300833 wgpu cold-fill stall case); (b) THIS case — a cold obligation
    # deferred before configuration — no-ops at the pump (0: nothing
    # installed yet) and dispatches when the forced commit re-runs the
    # evidence configuration (the codex finding-1 case).
    assert view.required_calls[-1] == (
        True,
        ("wgpu-resident-histogram", session.level_key),
    )
    assert view.dispatched is True
    assert settle_pumped == [0]
    assert quiet_edge_queued == [1]
    assert len(submitted) == 1
    assert session._wgpu_histogram_evidence_deferred is False

    # Publication releases the barrier and only then does coverage close.
    assert policy.observe(session.lifecycle) is False
    policy.set_coverage_evidence_pending(False)
    assert policy.observe(session.lifecycle) is True
    assert policy.verdict.refinement_admissible


def test_initial_montage_plan_uses_pending_restored_viewport_range():
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.viewport import ViewportMode
    from arrayscope.window.frame_controller import FrameControllerMixin

    class Window(FrameControllerMixin):
        def __init__(self):
            self.win = self

    win = Window()
    win.win = win
    state = ViewState.from_shape((10, 10, 8)).with_montage_axis(
        2, columns=None, indices=tuple(range(8)), text=":"
    )
    win.img_view = SimpleNamespace(
        image=None,
        viewport_controller=SimpleNamespace(
            mode=ViewportMode.USER,
            is_fit_locked=lambda: False,
            promote_near_auto=lambda _view_range: False,
            is_auto_active=lambda: False,
        ),
        graphicsView=SimpleNamespace(
            viewport=lambda: SimpleNamespace(size=lambda: QtCore.QSize(400, 200))
        ),
        getView=lambda: SimpleNamespace(viewRange=lambda: ((0.0, 1.0), (0.0, 1.0))),
        rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph"),
    )
    win._pending_viewport_continuity_range = lambda: ((100.0, 120.0), (200.0, 220.0))
    win._pending_viewport_continuity_columns = lambda: 3

    viewport_plan = FrameControllerMixin._montage_viewport_plan(win, state)

    assert viewport_plan.view_range == ((100.0, 120.0), (200.0, 220.0))
    assert viewport_plan.plan.columns == 3


def test_initial_montage_plan_ignores_invalid_restored_columns():
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.viewport import ViewportMode
    from arrayscope.window.frame_controller import FrameControllerMixin

    class Window(FrameControllerMixin):
        def __init__(self):
            self.win = self

    win = Window()
    win.win = win
    state = ViewState.from_shape((10, 10, 8)).with_montage_axis(
        2, columns=None, indices=tuple(range(8)), text=":"
    )
    win.img_view = SimpleNamespace(
        image=None,
        viewport_controller=SimpleNamespace(
            mode=ViewportMode.USER,
            is_fit_locked=lambda: False,
            is_auto_active=lambda: False,
        ),
        graphicsView=SimpleNamespace(
            viewport=lambda: SimpleNamespace(size=lambda: QtCore.QSize(400, 200))
        ),
        getView=lambda: SimpleNamespace(viewRange=lambda: ((0.0, 1.0), (0.0, 1.0))),
        rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph"),
    )
    win._pending_viewport_continuity_range = lambda: ((100.0, 120.0), (200.0, 220.0))
    win._pending_viewport_continuity_columns = lambda: "auto"

    viewport_plan = FrameControllerMixin._montage_viewport_plan(win, state)

    assert viewport_plan.view_range == ((100.0, 120.0), (200.0, 220.0))
    assert viewport_plan.plan.columns is not None


def test_initial_montage_plan_without_image_measures_startup_lod_from_layout():
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.lod import LOD_REASON_INVALID_VIEW, select_lod_demand
    from arrayscope.display.viewport import ViewportMode
    from arrayscope.window.frame_controller import FrameControllerMixin
    from arrayscope.window.montage_viewport import square_montage_fit_view_range

    class Window(FrameControllerMixin):
        def __init__(self):
            self.win = self

    win = Window()
    state = ViewState.from_shape((336, 336, 272)).with_montage_axis(
        2,
        columns=None,
        indices=tuple(range(49)),
        text=":",
    )
    win.img_view = SimpleNamespace(
        image=None,
        viewport_controller=SimpleNamespace(
            mode=ViewportMode.USER,
            is_fit_locked=lambda: False,
            is_auto_active=lambda: False,
        ),
        graphicsView=SimpleNamespace(
            viewport=lambda: SimpleNamespace(size=lambda: QtCore.QSize(200, 200))
        ),
        getView=lambda: SimpleNamespace(viewRange=lambda: ((0.0, 1.0), (0.0, 1.0))),
        rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph"),
    )

    viewport_plan = FrameControllerMixin._montage_viewport_plan(win, state)

    assert viewport_plan.view_range == square_montage_fit_view_range(
        viewport_plan.plan,
        viewport_plan.viewport_shape,
    )
    demand = select_lod_demand(
        viewport_plan.view_range,
        viewport_plan.viewport_shape,
        viewport_plan.tile_shape,
    )
    assert demand.reason != LOD_REASON_INVALID_VIEW
    assert demand.desired_level > 0


def test_auto_successor_plan_uses_successor_fit_while_predecessor_camera_is_retained():
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_state import ViewState
    from arrayscope.window.frame_controller import FrameControllerMixin
    from arrayscope.window.montage_viewport import square_montage_fit_view_range

    class Window(FrameControllerMixin):
        def __init__(self):
            self.win = self

    win = Window()
    state = ViewState.from_shape((336, 336, 272)).with_montage_axis(
        2,
        columns=21,
        indices=tuple(range(272)),
        text=":",
    )
    predecessor_range = ((0.0, 4380.0), (0.0, 2695.0))
    win._frame_session = SimpleNamespace(montage_axis=2, display_committed=True)
    win.img_view = SimpleNamespace(
        image=np.zeros((1, 1), dtype=np.float32),
        viewport_controller=SimpleNamespace(
            is_fit_locked=lambda: False,
            is_auto_active=lambda: True,
        ),
        graphicsView=SimpleNamespace(
            viewport=lambda: SimpleNamespace(size=lambda: QtCore.QSize(1245, 753))
        ),
        getView=lambda: SimpleNamespace(viewRange=lambda: predecessor_range),
        rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph"),
    )

    viewport_plan = FrameControllerMixin._montage_viewport_plan(win, state)

    assert viewport_plan.view_range == square_montage_fit_view_range(
        viewport_plan.plan,
        viewport_plan.viewport_shape,
    )
    assert len(viewport_plan.candidate_tiles()) == 272


def test_lod_policy_selects_producer_without_owning_target_debt():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.lod import (
        LOD_POLICY_NATIVE_ONLY,
        LOD_POLICY_RESIDENT,
        native_lod_policy,
        resident_lod_policy,
    )
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.render.lod import missing_tiles_require_native_target
    from arrayscope.window.frame_session import FrameSession

    def session_for(policy, decision):
        state = ViewState.from_shape((64, 64, 4)).with_montage_axis(
            2,
            columns=4,
            indices=(0, 1, 2, 3),
            text=":",
        )
        plan = make_montage_plan(
            state, axis=2, indices=(0, 1, 2, 3), tile_shape=(16, 16), columns=4
        )
        return FrameSession(
            session_id=1,
            key=("target-debt", policy),
            render_generation=1,
            level_key=("levels",),
            level_expected_indices=(0, 1, 2, 3),
            plan=plan,
            view_state=state,
            document=None,
            montage_axis=2,
            colormap_lut=None,
            view_range=((0.0, 1024.0), (0.0, 1024.0)),
            viewport_shape=(128, 128),
            output_dtype=np.dtype(np.float32),
            rgb=False,
            window_mode="relative",
            force_auto=False,
            visible_tiles=plan.tiles,
            rendered_tiles={},
            loading_tiles=set(),
            skipped_tiles=set(),
            lod_policy_mode=policy,
            lod_policy_decision=decision,
        )

    coarse_resident = session_for(
        LOD_POLICY_RESIDENT,
        resident_lod_policy(((0.0, 1024.0), (0.0, 1024.0)), (128, 128), (16, 16)),
    )
    native = session_for(
        LOD_POLICY_NATIVE_ONLY,
        native_lod_policy(((0.0, 16.0), (0.0, 16.0)), (128, 128), (16, 16)),
    )

    # Target debt has one owner regardless of which producer can satisfy it.
    assert coarse_resident.required_target_unsettled_tiles() == (0, 1, 2, 3)
    assert native.required_target_unsettled_tiles() == (0, 1, 2, 3)
    assert not missing_tiles_require_native_target(
        coarse_resident.lod_policy_mode,
        coarse_resident.lod_policy_decision.demand,
    )
    assert missing_tiles_require_native_target(
        native.lod_policy_mode,
        native.lod_policy_decision.demand,
    )


def test_montage_commit_reschedules_restored_roi_stats():
    from arrayscope.window.frame_controller import FrameControllerMixin

    calls = []
    win = SimpleNamespace(
        _file_session_roi_refresh_pending=True,
        _schedule_viewport_continuity_when_ready=lambda: calls.append("viewport"),
        _schedule_file_session_roi_refresh=lambda reason: calls.append(("roi", reason)),
    )
    win.win = win

    FrameControllerMixin._notify_file_session_montage_committed(win)

    assert calls == ["viewport", ("roi", "montage-semantic-commit")]


def test_vispy_persistent_upsert_limits_use_governed_upload_limit():
    from arrayscope.window import frame_effects as montage_commit

    session = SimpleNamespace()

    def resident(_payload):
        return True

    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(
                name="vispy",
                persistent_tile_residency=True,
                shader_windowing=True,
            ),
            tiledPayloadResident=resident,
        ),
        _viewport_interaction_active=False,
        resource_governor=SimpleNamespace(
            decide_commit_batch=lambda *, interactive: SimpleNamespace(
                batch_limit=11, byte_cap=2 * 1024 * 1024, budget_ms=2.0
            )
        ),
    )
    window.win = window

    limits = montage_commit._persistent_tile_upsert_limits(window, session)

    assert limits["max_upserts"] == 11
    assert limits["max_upsert_bytes"] == 2 * 1024 * 1024
    assert limits["physical_resident_fn"] is resident
    assert limits["pace_resident_retargets"] is False


def test_hidden_target_warm_does_not_wait_for_visible_target_settlement(monkeypatch):
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta
    from arrayscope.window import frame_effects

    resident: set[int] = set()
    warm_deltas = []
    payloads = {
        tile: DisplayTilePayload(
            tile,
            tile,
            np.full((2, 2), tile, dtype=np.float32),
            None,
            ("target", tile),
        )
        for tile in range(3)
    }

    def warm(**kwargs):
        warm_deltas.append(kwargs["tile_delta"])
        resident.update(id(payload) for payload in kwargs["payloads"].values())

    view = SimpleNamespace(
        warmTiledResidency=warm,
        tiledPayloadResident=lambda payload: id(payload) in resident,
    )
    session = SimpleNamespace(
        session_id=7,
        key=("session",),
        viewport_revision=3,
        _atomic_warm_job=None,
        final_commit_pending=False,
        flush_pending=False,
    )
    replans = []
    renderer = SimpleNamespace(
        win=SimpleNamespace(img_view=view),
        _frame_session_is_current=lambda candidate: candidate is session,
        _memory_policy=lambda: SimpleNamespace(
            visible_render_budget_bytes=1 << 30,
            display_cache_budget_bytes=1 << 30,
            user_render_cap_bytes=1 << 30,
        ),
        request_montage_replan=lambda candidate: replans.append(candidate),
    )
    monkeypatch.setattr(
        frame_effects,
        "_post_visible_path_callback",
        lambda _renderer, callback: callback(),
    )
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=3,
        upserts=payloads,
        active_tiles=tuple(payloads),
        planned_tiles=tuple(payloads),
    )

    ready = frame_effects._warm_atomic_successor_residency(
        renderer,
        session,
        _geometry(),
        delta,
        levels=(-1.0, 1.0),
        rgb_already_windowed=False,
        payloads=payloads,
        batch_size=2,
    )

    assert ready is False
    assert len(warm_deltas) == 2
    assert all(candidate.atomic_handoff for candidate in warm_deltas)
    assert resident == {id(payload) for payload in payloads.values()}
    assert session._atomic_warm_job is None
    assert session.final_commit_pending is True
    assert session.flush_pending is True
    assert replans == [session]


def test_hidden_target_warm_accepts_backend_residency_without_duplicate_markers(
    monkeypatch,
):
    """Physical residency is the sole warm owner when a backend exposes it."""

    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta
    from arrayscope.window import frame_effects

    payloads = {
        tile: DisplayTilePayload(
            tile,
            tile,
            np.full((2, 2), tile, dtype=np.float32),
            None,
            ("resident-target", tile),
        )
        for tile in range(50)
    }
    warm_calls = []
    callbacks = []
    replans = []
    view = SimpleNamespace(
        warmTiledResidency=lambda **kwargs: warm_calls.append(kwargs),
        tiledPayloadResident=lambda _payload: True,
    )
    session = SimpleNamespace(
        session_id=8,
        key=("resident-session",),
        viewport_revision=4,
        _atomic_warm_job=None,
        final_commit_pending=False,
        flush_pending=False,
    )
    renderer = SimpleNamespace(
        win=SimpleNamespace(img_view=view),
        _frame_session_is_current=lambda candidate: candidate is session,
        _memory_policy=lambda: SimpleNamespace(
            visible_render_budget_bytes=1 << 20,
            display_cache_budget_bytes=1 << 20,
            user_render_cap_bytes=1 << 20,
        ),
        request_montage_replan=lambda candidate: replans.append(candidate),
    )
    monkeypatch.setattr(
        frame_effects,
        "_post_visible_path_callback",
        lambda _renderer, callback: callbacks.append(callback),
    )
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=4,
        upserts=payloads,
        active_tiles=tuple(payloads),
        planned_tiles=tuple(payloads),
    )

    ready = frame_effects._warm_atomic_successor_residency(
        renderer,
        session,
        _geometry(),
        delta,
        levels=(-1.0, 1.0),
        rgb_already_windowed=False,
        payloads=payloads,
        batch_size=2,
    )

    assert ready is True
    assert callbacks == []
    assert warm_calls == []
    assert replans == []
    assert session._atomic_warm_job is None


def test_hidden_target_warm_rechecks_marked_payload_residency(monkeypatch):
    """A historical warm marker is not physical proof after pool eviction."""

    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta
    from arrayscope.window import frame_effects

    payload = DisplayTilePayload(
        0,
        0,
        np.zeros((2, 2), dtype=np.float32),
        None,
        ("evicted-target", 0),
    )
    level_key = (-1.0, 1.0)
    callbacks = []
    view = SimpleNamespace(
        warmTiledResidency=lambda **_kwargs: None,
        tiledPayloadResident=lambda _payload: False,
    )
    session = SimpleNamespace(
        session_id=9,
        key=("evicted-session",),
        viewport_revision=4,
        _atomic_warm_job=None,
        _atomic_warmed_payloads={0: (payload.source_id, level_key)},
        _atomic_warmed_identities={(payload.source_id, level_key)},
        final_commit_pending=False,
        flush_pending=False,
    )
    renderer = SimpleNamespace(
        win=SimpleNamespace(img_view=view),
        _frame_session_is_current=lambda candidate: candidate is session,
        _memory_policy=lambda: SimpleNamespace(
            visible_render_budget_bytes=1 << 20,
            display_cache_budget_bytes=1 << 20,
            user_render_cap_bytes=1 << 20,
        ),
        request_montage_replan=lambda _candidate: None,
    )
    monkeypatch.setattr(
        frame_effects,
        "_post_visible_path_callback",
        lambda _renderer, callback: callbacks.append(callback),
    )
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=4,
        upserts={0: payload},
        active_tiles=(0,),
        planned_tiles=(0,),
    )

    ready = frame_effects._warm_atomic_successor_residency(
        renderer,
        session,
        _geometry(),
        delta,
        levels=level_key,
        rgb_already_windowed=False,
        payloads={0: payload},
    )

    assert ready is False
    assert len(callbacks) == 1
    assert session._atomic_warm_job["pending"] == [0]


def test_partial_first_pixels_never_transfer_cold_tiles_to_hidden_warm():
    from arrayscope.window.frame_effects import (
        _cold_gpu_successor_requires_hidden_warm,
    )

    payload = object()
    session = SimpleNamespace(
        # Reproduce the field defect: a bounded first batch made this sticky
        # flag true while most required tiles still had no physical pixels.
        display_committed=True,
        required_first_pixels_presented=lambda: False,
    )

    assert not _cold_gpu_successor_requires_hidden_warm(
        session=session,
        cpu_backend=False,
        resident_predicate=lambda _payload: False,
        upserts={7: payload},
    )

    session.required_first_pixels_presented = lambda: True
    assert _cold_gpu_successor_requires_hidden_warm(
        session=session,
        cpu_backend=False,
        resident_predicate=lambda _payload: False,
        upserts={7: payload},
    )


def test_hidden_target_warm_zero_progress_rearms_visible_owner(monkeypatch):
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta
    from arrayscope.window import frame_effects

    payload = DisplayTilePayload(
        0,
        0,
        np.zeros((2, 2), dtype=np.float32),
        None,
        ("blocked-target", 0),
    )
    view = SimpleNamespace(
        warmTiledResidency=lambda **_kwargs: None,
        tiledPayloadResident=lambda _payload: False,
    )
    session = SimpleNamespace(
        session_id=8,
        key=("blocked-session",),
        viewport_revision=4,
        _atomic_warm_job=None,
        final_commit_pending=False,
        flush_pending=False,
    )
    replans = []
    renderer = SimpleNamespace(
        win=SimpleNamespace(img_view=view),
        _frame_session_is_current=lambda candidate: candidate is session,
        _memory_policy=lambda: SimpleNamespace(
            visible_render_budget_bytes=1 << 20,
            display_cache_budget_bytes=1 << 20,
            user_render_cap_bytes=1 << 20,
        ),
        request_montage_replan=lambda candidate: replans.append(candidate),
    )
    monkeypatch.setattr(
        frame_effects,
        "_post_visible_path_callback",
        lambda _renderer, callback: callback(),
    )
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=4,
        upserts={0: payload},
        active_tiles=(0,),
        planned_tiles=(0,),
    )

    ready = frame_effects._warm_atomic_successor_residency(
        renderer,
        session,
        _geometry(),
        delta,
        levels=(-1.0, 1.0),
        rgb_already_windowed=False,
        payloads={0: payload},
    )

    assert ready is False
    assert session._atomic_warm_job is None
    assert session.final_commit_pending is True
    assert session.flush_pending is True
    assert replans == [session]


def test_hidden_target_warm_accepts_visible_commit_slot_owner(monkeypatch):
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta
    from arrayscope.window import frame_effects

    payload = DisplayTilePayload(
        0,
        0,
        np.zeros((2, 2), dtype=np.float32),
        None,
        ("slot-owned-target", 0),
    )
    warm_calls = []
    view = SimpleNamespace(
        warmTiledResidency=lambda **kwargs: warm_calls.append(kwargs),
        tiledPayloadResident=lambda _payload: False,
        tiledPayloadCommitSlotOwned=lambda candidate: candidate is payload,
    )
    session = SimpleNamespace(
        session_id=10,
        key=("slot-owned-session",),
        viewport_revision=5,
        _atomic_warm_job=None,
        final_commit_pending=False,
        flush_pending=False,
    )
    replans = []
    renderer = SimpleNamespace(
        win=SimpleNamespace(img_view=view),
        _frame_session_is_current=lambda candidate: candidate is session,
        _memory_policy=lambda: SimpleNamespace(
            visible_render_budget_bytes=1 << 20,
            display_cache_budget_bytes=1 << 20,
            user_render_cap_bytes=1 << 20,
        ),
        request_montage_replan=lambda candidate: replans.append(candidate),
    )
    monkeypatch.setattr(
        frame_effects,
        "_post_visible_path_callback",
        lambda _renderer, callback: pytest.fail(
            f"slot-owned payload queued an unnecessary warm callback: {callback!r}"
        ),
    )
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=5,
        upserts={0: payload},
        active_tiles=(0,),
        planned_tiles=(0,),
    )

    ready = frame_effects._warm_atomic_successor_residency(
        renderer,
        session,
        _geometry(),
        delta,
        levels=(-1.0, 1.0),
        rgb_already_windowed=False,
        payloads={0: payload},
    )

    assert ready is True
    assert warm_calls == []
    assert session._atomic_warm_job is None
    assert session.final_commit_pending is False
    assert session.flush_pending is False
    assert replans == []


def test_vispy_atomic_successor_marker_ignores_lod_but_not_source_index():
    from arrayscope.window import frame_effects as montage_commit

    coarse = SimpleNamespace(
        source_id=(("semantic", 42), "texture_kind", "scalar", "lod", 3),
        source_index=42,
    )
    exact = SimpleNamespace(
        source_id=(("semantic", 42), "texture_kind", "scalar", "lod", 0),
        source_index=42,
    )
    wrong_source = SimpleNamespace(
        source_id=(("semantic", 43), "texture_kind", "scalar", "lod", 0),
        source_index=43,
    )

    marker = montage_commit._shader_successor_transaction_payload_marker(coarse)
    assert montage_commit._shader_successor_transaction_payload_marker(exact) == marker
    assert montage_commit._shader_successor_transaction_payload_marker(wrong_source) != marker


def test_vispy_first_persistent_upsert_limits_use_shared_commit_batch():
    from arrayscope.window import frame_effects as montage_commit

    session = SimpleNamespace(display_committed=False)

    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(
                name="vispy",
                persistent_tile_residency=True,
                shader_windowing=True,
            )
        ),
        _viewport_interaction_active=False,
        resource_governor=SimpleNamespace(
            decide_commit_batch=lambda *, interactive: SimpleNamespace(
                batch_limit=4, byte_cap=1024 * 1024, budget_ms=2.0
            )
        ),
    )
    window.win = window

    limits = montage_commit._persistent_tile_upsert_limits(window, session)

    assert limits["max_upserts"] == 4
    assert limits["max_upsert_bytes"] == 1024 * 1024


def test_vispy_persistent_upsert_limits_keep_minimum_cohort_under_fixed_transaction_cost():
    from arrayscope.window import frame_effects as montage_commit

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
        rendered_tiles={
            0: SimpleNamespace(
                image=image, histogram_data=None, semantic_data=semantic, level_data=None
            )
        },
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
        resource_governor=SimpleNamespace(
            decide_commit_batch=lambda *, interactive: SimpleNamespace(
                batch_limit=1, byte_cap=1024 * 1024, budget_ms=2.0
            )
        ),
    )
    window.win = window

    limits = montage_commit._persistent_tile_upsert_limits(window, session)

    assert limits["max_upserts"] == 4
    assert limits["max_upsert_bytes"] == 1024 * 1024
    assert limits["upsert_cost_fn"](payload) == texture.nbytes


def _wgpu_native_warm_payload(*, lod_level: int, source_rect):
    """A payload whose commit uploads its whole canonical plane, not its texture.

    The source anchor is not decoration here: the warm path keys the canonical
    pages off ``anchor.content_key``/``plane_shape``, so a whole-plane array
    without an anchor names no pages and warms nothing.
    """

    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.frame import DisplayTilePayload, PayloadSourceAnchor

    native = np.zeros((336, 336), dtype=np.float32)
    factor = 1 << int(lod_level)
    y0, y1, x0, x1 = source_rect
    window = ((y1 - y0) // factor, (x1 - x0) // factor)
    return DisplayTilePayload(
        0,
        0,
        np.zeros(window, dtype=np.float32),
        None,
        ("source", 0),
        lod=LodInfo(int(lod_level), factor, (y1 - y0, x1 - x0), window),
        native_residency_data=native,
        source_anchor=PayloadSourceAnchor(
            content_key=("src-anchored", 0),
            source_rect=tuple(source_rect),
            plane_shape=native.shape,
        ),
    )


def test_wgpu_native_source_prefetch_stays_in_bounded_two_tile_cohorts():
    from arrayscope.window import frame_effects as montage_commit

    native = np.zeros((336, 336), dtype=np.float32)
    payload = _wgpu_native_warm_payload(lod_level=2, source_rect=(0, 336, 0, 336))
    session = SimpleNamespace(
        display_committed=True,
        dirty_payloads=dict.fromkeys(range(50)),
        pending_payload_upserts={},
        display_tile_payloads=dict.fromkeys(range(50), payload),
    )
    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(
                name="wgpu",
                persistent_tile_residency=True,
                tile_residency_kind="gpu_atlas",
                shader_windowing=True,
            )
        ),
        _viewport_interaction_active=False,
        resource_governor=SimpleNamespace(
            decide_commit_batch=lambda *, interactive: SimpleNamespace(
                batch_limit=32, byte_cap=32 * 1024 * 1024, budget_ms=8.0
            )
        ),
    )
    window.win = window

    limits = montage_commit._persistent_tile_upsert_limits(window, session)

    assert limits["max_upserts"] == 2
    assert limits["max_upsert_bytes"] == 3 * 1024 * 1024
    assert limits["upsert_cost_fn"](payload) == native.nbytes


def test_wgpu_cropped_native_warm_is_costed_as_its_whole_plane():
    """A CROPPED exact payload warms the plane too, and must be costed as one.

    A view cropped from its first frame warms the canonical plane at level 0 —
    it presents a sub-rect of pixels it fully owns.  Costing that commit by its
    small crop texture would let the pacing admit 4-8 tiles per callback while
    each secretly uploads a whole plane, which is the accounting mistake the
    reduced path already learned (272 previews planned, 1088 native pages
    installed).  Cost and cohort follow the plane, not the window.
    """

    from arrayscope.window import frame_effects as montage_commit

    native = np.zeros((336, 336), dtype=np.float32)
    cropped = _wgpu_native_warm_payload(lod_level=0, source_rect=(38, 238, 66, 266))
    whole = _wgpu_native_warm_payload(lod_level=0, source_rect=(0, 336, 0, 336))

    assert montage_commit.wgpu_native_plane_warm_payload(cropped) is True
    assert montage_commit.wgpu_payload_upload_nbytes(cropped) == native.nbytes
    # An exact payload that already IS its whole plane uploads that plane
    # through its ordinary source-anchored binding; it is not a hidden warm.
    assert montage_commit.wgpu_native_plane_warm_payload(whole) is False
    assert montage_commit.wgpu_payload_upload_nbytes(whole) == native.nbytes

    session = SimpleNamespace(
        display_committed=True,
        dirty_payloads=dict.fromkeys(range(50)),
        pending_payload_upserts={},
        display_tile_payloads=dict.fromkeys(range(50), cropped),
    )
    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(
                name="wgpu",
                persistent_tile_residency=True,
                tile_residency_kind="gpu_atlas",
                shader_windowing=True,
            )
        ),
        _viewport_interaction_active=False,
        resource_governor=SimpleNamespace(
            decide_commit_batch=lambda *, interactive: SimpleNamespace(
                batch_limit=32, byte_cap=32 * 1024 * 1024, budget_ms=8.0
            )
        ),
    )
    window.win = window

    limits = montage_commit._persistent_tile_upsert_limits(window, session)
    assert limits["max_upserts"] == 2
    assert limits["max_upsert_bytes"] == 3 * 1024 * 1024
    assert limits["upsert_cost_fn"](cropped) == native.nbytes


def test_vispy_idle_upsert_cohort_scales_to_large_backlog():
    # The tiled commit's cost is fixed-dominated (full-plan classify + delta
    # walk + acknowledgement run once per commit regardless of item count),
    # so a latency-governed item clamp multiplies the fixed cost across a
    # large idle backlog instead of shortening any callback: the 272-tile
    # cold fill at 4 items/turn outran its settlement budget.  Idle commits
    # with a backlog larger than the governed limit take a fixed-cost
    # amortizing cohort; the byte cap stays authoritative for upload size
    # and the interactive clamp is untouched.
    from arrayscope.window import frame_effects as montage_commit

    def build_session(backlog: int):
        return SimpleNamespace(
            display_committed=True,
            dirty_payloads=dict.fromkeys(range(backlog)),
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
        resource_governor=SimpleNamespace(
            decide_commit_batch=lambda *, interactive: SimpleNamespace(
                batch_limit=4, byte_cap=32 * 1024 * 1024, budget_ms=2.0
            )
        ),
    )
    window.win = window

    assert (
        montage_commit._persistent_tile_upsert_limits(window, build_session(60))["max_upserts"]
        == 32
    )
    assert (
        montage_commit._persistent_tile_upsert_limits(window, build_session(10))["max_upserts"]
        == 10
    )
    assert (
        montage_commit._persistent_tile_upsert_limits(window, build_session(3))["max_upserts"] == 4
    )

    window._viewport_interaction_active = True
    assert (
        montage_commit._persistent_tile_upsert_limits(window, build_session(60))["max_upserts"] == 4
    )


def test_pyqtgraph_idle_commits_keep_governed_cohort_under_deep_backlog():
    # The CPU-windowed tile layer pays a real per-item cost, so the
    # idle-backlog cohort boost is reserved for the upload-only
    # shader-windowing path: a 32-item pyqtgraph commit is one long GUI
    # callback that delays the next gesture's pixels (journey-matrix
    # pyqtgraph scroll rows, 2026-07-18).
    from arrayscope.window import frame_effects as montage_commit

    def build_session(backlog: int):
        return SimpleNamespace(
            display_committed=True,
            dirty_payloads=dict.fromkeys(range(backlog)),
            pending_payload_upserts={},
            pending_removals=set(),
            has_pending_level_update=lambda: False,
            has_stale_level_presentations=lambda: False,
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
        resource_governor=SimpleNamespace(
            decide_commit_batch=lambda *, interactive: SimpleNamespace(
                batch_limit=2, byte_cap=4096, budget_ms=2.0
            )
        ),
    )
    window.win = window

    assert montage_commit.tile_layer_upsert_limits(window, build_session(60))["max_upserts"] == 2
    assert montage_commit.tile_layer_upsert_limits(window, build_session(1))["max_upserts"] == 2

    window._viewport_interaction_active = True
    assert montage_commit.tile_layer_upsert_limits(window, build_session(60))["max_upserts"] <= 8


def test_pyqtgraph_floor_progress_commits_stay_governed():
    # A floor-progress commit carries no dirty/pending work at limits
    # decision time — the build's floor pass materializes preview upserts
    # during assembly (zoom-in frontier tiles). With unsettled required
    # targets the commit must still be governed: the ungoverned batch
    # (max_upserts=0, unbounded_reason="") failed the journey matrix's
    # bounded-commit oracle (pyqtgraph zoom_in, v19/v11/2026-07-19 v2-v4).
    from arrayscope.window import frame_effects as montage_commit

    def build_session(unsettled):
        return SimpleNamespace(
            display_committed=True,
            dirty_payloads={},
            pending_payload_upserts={},
            pending_removals=set(),
            has_pending_level_update=lambda: False,
            has_stale_level_presentations=lambda: False,
            required_target_unsettled_tiles=lambda: tuple(unsettled),
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
        resource_governor=SimpleNamespace(
            decide_commit_batch=lambda *, interactive: SimpleNamespace(
                batch_limit=6, byte_cap=4096, budget_ms=2.0
            )
        ),
    )
    window.win = window

    governed = montage_commit.tile_layer_upsert_limits(window, build_session((3, 4, 5)))
    assert governed["max_upserts"] == 6

    assert montage_commit.tile_layer_upsert_limits(window, build_session(())) == {}


def test_pyqtgraph_tile_layer_upsert_limits_use_display_image_upload_cost():
    from arrayscope.window import frame_effects as montage_commit

    image = np.zeros((512, 512), dtype=np.float32)
    semantic = np.zeros((1024, 1024), dtype=np.complex64)
    payload = SimpleNamespace(
        image=image,
        texture_data=semantic,
        histogram_data=np.zeros((512, 512), dtype=np.float32),
        semantic_data=semantic,
    )
    session = SimpleNamespace(
        display_committed=True,
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
        resource_governor=SimpleNamespace(
            decide_commit_batch=lambda *, interactive: SimpleNamespace(
                batch_limit=3, byte_cap=1024 * 1024, budget_ms=2.0
            )
        ),
    )
    window.win = window

    limits = montage_commit.tile_layer_upsert_limits(window, session)

    assert limits["max_upserts"] == 3
    assert limits["max_upsert_bytes"] == 1024 * 1024
    assert limits["cold_deadline_ms"] == 2.0
    assert limits["upsert_cost_fn"](payload) == image.nbytes


def test_pyqtgraph_tile_layer_upsert_limits_apply_to_cold_dirty_payloads():
    from arrayscope.window import frame_effects as montage_commit

    session = SimpleNamespace(
        display_committed=True,
        dirty_payloads={0: None},
        pending_payload_upserts={},
        pending_removals=set(),
        has_pending_level_update=lambda: False,
        has_stale_level_presentations=lambda: False,
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
        resource_governor=SimpleNamespace(
            decide_commit_batch=lambda *, interactive: SimpleNamespace(
                batch_limit=2, byte_cap=4096, budget_ms=2.0
            )
        ),
    )
    window.win = window

    limits = montage_commit.tile_layer_upsert_limits(window, session)

    assert limits["max_upserts"] == 2
    assert limits["max_upsert_bytes"] == 4096
    assert limits["cold_deadline_ms"] == 2.0
    assert (
        limits["upsert_cost_fn"](SimpleNamespace(image=np.zeros((8, 8), dtype=np.float32)))
        == 8 * 8 * 4
    )


def test_pyqtgraph_first_frame_uses_bounded_batches():
    from arrayscope.window import frame_effects as montage_commit

    session = SimpleNamespace(
        scheduling_policy=_refine_scheduling_policy(),
        display_committed=False,
        dirty_payloads={0: None},
        pending_payload_upserts={0: None},
        pending_removals=set(),
        has_pending_level_update=lambda: False,
        has_stale_level_presentations=lambda: False,
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
        resource_governor=SimpleNamespace(
            decide_commit_batch=lambda *, interactive: SimpleNamespace(
                batch_limit=3,
                byte_cap=4096,
                budget_ms=2.0,
            )
        ),
    )
    window.win = window

    limits = montage_commit.tile_layer_upsert_limits(window, session)

    assert limits["max_upserts"] == 3
    assert limits["max_upsert_bytes"] == 4096
    assert limits["cold_deadline_ms"] == 2.0


def test_presentation_cold_walk_budget_keeps_callback_safety_margin():
    from arrayscope.window.frame_effects import presentation_upload_control_budget_ms

    decision = SimpleNamespace(budget_ms=100.0)
    window = SimpleNamespace()
    window.win = window

    assert (
        presentation_upload_control_budget_ms(
            window,
            "tile_layer_commit",
            decision,
            interactive=False,
        )
        == 12.0
    )


def test_tile_layer_commit_feedback_counts_acknowledged_level_upserts():
    from arrayscope.display.model.frame import TileCommitReport
    from arrayscope.window import frame_effects as montage_commit

    report = TileCommitReport(
        presented_tiles=(0, 1, 2),
        committed_upserts=(0, 1, 2),
        texture_uploads=0,
        texture_upload_bytes=0,
    )

    assert montage_commit.tile_layer_commit_processed_count(report) == 3


def test_retained_payload_store_receives_only_accepted_delta_payloads():
    from arrayscope.display.model.frame import (
        DisplayTilePayload,
        TileCommitReport,
        TilePresentationDelta,
    )
    from arrayscope.window import frame_effects as montage_commit

    payloads = {
        index: DisplayTilePayload(
            tile_number=index,
            source_index=index,
            image=np.full((2, 2), index, dtype=np.float32),
            histogram_data=None,
            source_id=("source", index),
        )
        for index in range(4)
    }
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        base_revision=4,
        target_revision=5,
        upserts={1: payloads[1], 3: payloads[3]},
    )
    report = TileCommitReport(
        presented_tiles=(0, 1, 2, 3),
        committed_upserts=(3,),
        delta_key=(4, 5),
    )

    retained = montage_commit.accepted_tiled_payloads(payloads, delta, report)

    assert retained == {3: payloads[3]}


def test_pyqtgraph_display_committed_tile_layer_can_use_direct_delta_commit():
    from arrayscope.window import frame_effects as montage_commit

    session = SimpleNamespace(display_committed=True)
    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(
                name="pyqtgraph",
                persistent_tile_residency=False,
                shader_windowing=False,
            )
        ),
        _viewport_interaction_active=False,
    )
    window.win = window

    assert montage_commit.direct_montage_tile_delta_commit_enabled(window, session) is True
    window._viewport_interaction_active = True
    assert montage_commit.direct_montage_tile_delta_commit_enabled(window, session) is True
    session.display_committed = False
    assert montage_commit.direct_montage_tile_delta_commit_enabled(window, session) is False


def test_vispy_persistent_tile_layer_can_direct_delta_first_session_commit():
    from arrayscope.window import frame_effects as montage_commit

    session = SimpleNamespace(display_committed=False)
    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(
                name="vispy",
                persistent_tile_residency=True,
                shader_windowing=True,
            )
        ),
        _viewport_interaction_active=False,
    )
    window.win = window

    assert (
        montage_commit.direct_montage_tile_delta_commit_enabled(
            window,
            session,
            allow_uncommitted_persistent=True,
        )
        is True
    )
    assert montage_commit.direct_montage_tile_delta_commit_enabled(window, session) is False


def test_pyqtgraph_tile_layer_uses_shared_commit_batch_without_cost_signature():
    from arrayscope.window import frame_effects as montage_commit

    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(
                name="pyqtgraph",
                persistent_tile_residency=False,
                shader_windowing=False,
            )
        ),
        _viewport_interaction_active=False,
        resource_governor=SimpleNamespace(
            decide_commit_batch=lambda *, interactive: SimpleNamespace(
                batch_limit=2, byte_cap=4096, budget_ms=2.0
            )
        ),
    )
    window.win = window
    session = SimpleNamespace(
        display_committed=True,
        dirty_payloads={0: None},
        pending_payload_upserts={},
        pending_removals=set(),
        rgb=False,
        output_dtype=np.dtype("float32"),
        has_pending_level_update=lambda: False,
        has_stale_level_presentations=lambda: False,
    )

    montage_commit.tile_layer_upsert_limits(window, session)
    session.rendered_tiles = {0: SimpleNamespace(image=np.zeros((8, 8), dtype=np.complex64))}
    limits = montage_commit.tile_layer_upsert_limits(window, session)

    assert limits["max_upserts"] == 2
    assert limits["max_upsert_bytes"] == 4096


def test_vispy_persistent_limits_use_shared_commit_batch_without_cost_signature():
    from arrayscope.window import frame_effects as montage_commit

    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(
                name="vispy",
                persistent_tile_residency=True,
                shader_windowing=True,
            )
        ),
        _viewport_interaction_active=False,
        resource_governor=SimpleNamespace(
            decide_commit_batch=lambda *, interactive: SimpleNamespace(
                batch_limit=8, byte_cap=8 * 1024 * 1024, budget_ms=8.0
            )
        ),
    )
    window.win = window
    session = SimpleNamespace(
        display_committed=True,
        dirty_payloads={0: None},
        pending_payload_upserts={},
        output_dtype=np.dtype("float32"),
        rgb=False,
        lifecycle=SimpleNamespace(
            presented_tiles=frozenset(),
            snapshot=lambda: SimpleNamespace(counts={}),
        ),
        rendered_tiles={},
    )

    montage_commit._persistent_tile_upsert_limits(window, session)
    session.output_dtype = np.dtype("complex64")
    session.rendered_tiles = {0: SimpleNamespace(image=np.zeros((8, 8), dtype=np.complex64))}
    limits = montage_commit._persistent_tile_upsert_limits(window, session)

    assert limits["max_upserts"] == 8
    assert limits["max_upsert_bytes"] == 8 * 1024 * 1024


def test_vispy_persistent_resident_remap_uses_shared_commit_batch():
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationState
    from arrayscope.window import frame_effects as montage_commit

    payload = DisplayTilePayload(
        tile_number=0,
        source_index=0,
        image=np.zeros((8, 8), dtype=np.complex64),
        histogram_data=np.zeros((8, 8), dtype=np.float32),
        source_id=("tile", 0, "complex"),
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
        resource_governor=SimpleNamespace(
            decide_commit_batch=lambda *, interactive: SimpleNamespace(
                batch_limit=1, byte_cap=1024, budget_ms=4.0
            )
        ),
    )
    window.win = window
    session = SimpleNamespace(
        display_committed=True,
        dirty_payloads={0: None},
        pending_payload_upserts={0: None},
        display_tile_payloads={0: payload},
        tile_presentation_state=TilePresentationState({0: payload}),
        acknowledged_source_ids={payload.source_id},
        output_dtype=np.dtype("complex64"),
        rgb=False,
        lifecycle=SimpleNamespace(
            presented_tiles=frozenset({0}),
            backend_presented_identities={0: payload.source_id},
        ),
        rendered_tiles={0: SimpleNamespace(image=payload.image)},
    )

    limits = montage_commit._persistent_tile_upsert_limits(window, session)

    assert limits["max_upserts"] == 4
    assert limits["max_upsert_bytes"] == 1024


def test_pyqtgraph_level_update_follows_delta_priority_order(qt_app):
    from pyqtgraph.graphicsItems.ImageItem import ImageItem

    from arrayscope.display.backends.pyqtgraph.tiles import (
        MontageTileLayer,
        TileLayerItemState,
        _direct_payload_source_id,
    )
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.model.frame import DisplayTilePayload

    class Owner:
        def add_tile_item(self, *_args):
            pass

        def remove_tile_item(self, *_args):
            pass

        def move_tile_item(self, *_args):
            pass

    geometry = DisplayGeometry(
        view_state=None,
        display_shape=(4, 4),
        montage=MontageGeometry(indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=2, rows=2, gap=0),
    )
    layer = MontageTileLayer(
        Owner(),
        set_image_item_data=lambda *_args, **_kwargs: None,
        record_upload_timing=lambda *_args, **_kwargs: None,
        histogram_levels_for_display=lambda levels: levels,
        is_rgb_image=lambda _image: False,
    )
    payloads = {}
    for tile_number in range(4):
        image = np.full((2, 2), tile_number, dtype=np.float32)
        payload = DisplayTilePayload(
            tile_number=tile_number,
            source_index=tile_number,
            image=image,
            histogram_data=None,
            source_id=("source", tile_number),
            semantic_data=image,
            source_shape=image.shape,
        )
        payloads[tile_number] = payload
        item = ImageItem(axisOrder="row-major")
        source_id = _direct_payload_source_id(payload.source_id, payload)
        layer.states[tile_number] = TileLayerItemState(
            tile_number=tile_number,
            source_index=tile_number,
            item=item,
            local_rect=(0, 0, 2, 2),
            world_rect=(-1, -1, -1, -1),
            source_array_id=source_id,
            histogram_array_id=None,
            levels=(0.0, 1.0),
            rgb_already_windowed=False,
            visible=True,
            display_cache=image,
        )

    order = []

    def update_levels(state, levels, **_payload_metadata):
        order.append(int(state.tile_number))
        state.levels = levels
        return False, False

    layer._update_tile_levels = update_levels
    tile_delta = SimpleNamespace(
        active_tiles=(3, 1, 2, 0),
        upserts={3: payloads[3], 1: payloads[1]},
        removals=(),
        force_refresh=False,
        cold_deadline_ms=None,
    )

    stats = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.25, 0.75),
        rgb_already_windowed=False,
        dirty_tiles=(),
        tile_payloads=payloads,
        tile_delta=tile_delta,
    )

    assert order == [3, 1]
    assert stats.presented_identities == {
        tile_number: payload.source_id for tile_number, payload in payloads.items()
    }


def test_level_only_drain_resolves_only_the_upsert_slice(qt_app, monkeypatch):
    # On a large complex montage the dominant per-commit cost is page-assembling
    # and re-windowing every resident active payload.  A ``level_only_drain``
    # commit only re-levels already-resident tiles, so the CPU-windowing backend
    # must resolve ONLY the emitted upsert slice, not every active tile.  Before
    # the fast path this loop ran over all active payloads (here: 4) regardless
    # of how few were being committed (here: 2).
    from pyqtgraph.graphicsItems.ImageItem import ImageItem

    from arrayscope.display.backends.pyqtgraph import tiles as tiles_module
    from arrayscope.display.backends.pyqtgraph.tiles import (
        MontageTileLayer,
        TileLayerItemState,
        _direct_payload_source_id,
    )
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta

    class Owner:
        def add_tile_item(self, *_args):
            pass

        def remove_tile_item(self, *_args):
            pass

        def move_tile_item(self, *_args):
            pass

    geometry = DisplayGeometry(
        view_state=None,
        display_shape=(4, 4),
        montage=MontageGeometry(indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=2, rows=2, gap=0),
    )
    layer = MontageTileLayer(
        Owner(),
        set_image_item_data=lambda *_args, **_kwargs: None,
        record_upload_timing=lambda *_args, **_kwargs: None,
        histogram_levels_for_display=lambda levels: levels,
        is_rgb_image=lambda _image: False,
    )
    payloads = {}
    for tile_number in range(4):
        image = np.full((2, 2), tile_number, dtype=np.float32)
        payload = DisplayTilePayload(
            tile_number=tile_number,
            source_index=tile_number,
            image=image,
            histogram_data=None,
            source_id=("source", tile_number),
            semantic_data=image,
            source_shape=image.shape,
        )
        payloads[tile_number] = payload
        item = ImageItem(axisOrder="row-major")
        source_id = _direct_payload_source_id(payload.source_id, payload)
        layer.states[tile_number] = TileLayerItemState(
            tile_number=tile_number,
            source_index=tile_number,
            item=item,
            local_rect=(0, 0, 2, 2),
            world_rect=(0, 0, 2, 2),
            source_array_id=source_id,
            histogram_array_id=None,
            levels=(0.0, 1.0),
            rgb_already_windowed=False,
            visible=True,
            display_cache=image,
        )

    resolved: list[int] = []
    original_resolve = tiles_module._resolve_page_backed_payload

    def counting_resolve(payload, *, levels=None):
        resolved.append(int(payload.tile_number))
        return original_resolve(payload, levels=levels)

    monkeypatch.setattr(tiles_module, "_resolve_page_backed_payload", counting_resolve)

    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        base_revision=0,
        active_tiles=(0, 1, 2, 3),
        upserts={3: payloads[3], 1: payloads[1]},
        removals=(),
        cold_deadline_ms=None,
        level_only_drain=True,
    )

    layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.25, 0.75),
        rgb_already_windowed=False,
        dirty_tiles=(),
        tile_payloads=payloads,
        tile_delta=delta,
    )

    # Only the two emitted upserts were resolved/re-windowed — the two
    # untouched resident tiles never entered the page-assembly path.
    assert sorted(resolved) == [1, 3]


def test_level_only_drain_commits_the_whole_upsert_slice(qt_app):
    # A level_only_drain commit must land every tile in its already-bounded
    # upsert slice: the resolve pass paid to re-window each of them, so a
    # mid-loop deadline must not drop the tail and force a wasteful re-resolve
    # next commit.
    from pyqtgraph.graphicsItems.ImageItem import ImageItem

    from arrayscope.display.backends.pyqtgraph.tiles import (
        MontageTileLayer,
        TileLayerItemState,
        _direct_payload_source_id,
    )
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta

    class Owner:
        def add_tile_item(self, *_args):
            pass

        def remove_tile_item(self, *_args):
            pass

        def move_tile_item(self, *_args):
            pass

    geometry = DisplayGeometry(
        view_state=None,
        display_shape=(4, 4),
        montage=MontageGeometry(indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=2, rows=2, gap=0),
    )
    layer = MontageTileLayer(
        Owner(),
        set_image_item_data=lambda *_args, **_kwargs: None,
        record_upload_timing=lambda *_args, **_kwargs: None,
        histogram_levels_for_display=lambda levels: levels,
        is_rgb_image=lambda _image: False,
    )
    payloads = {}
    for tile_number in range(4):
        image = np.full((2, 2), tile_number, dtype=np.float32)
        payload = DisplayTilePayload(
            tile_number=tile_number,
            source_index=tile_number,
            image=image,
            histogram_data=None,
            source_id=("source", tile_number),
            semantic_data=image,
            source_shape=image.shape,
        )
        payloads[tile_number] = payload
        item = ImageItem(axisOrder="row-major")
        source_id = _direct_payload_source_id(payload.source_id, payload)
        layer.states[tile_number] = TileLayerItemState(
            tile_number=tile_number,
            source_index=tile_number,
            item=item,
            local_rect=(0, 0, 2, 2),
            world_rect=(0, 0, 2, 2),
            source_array_id=source_id,
            histogram_array_id=None,
            levels=(0.0, 1.0),
            rgb_already_windowed=False,
            visible=True,
            display_cache=image,
        )

    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        base_revision=0,
        active_tiles=(0, 1, 2, 3),
        upserts={tile: payloads[tile] for tile in (0, 1, 2, 3)},
        removals=(),
        # A collapsed cold deadline that would normally stop the loop after the
        # first committed tile — the fast path ignores it for the bounded slice.
        cold_deadline_ms=0.0,
        level_only_drain=True,
    )

    stats = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.25, 0.75),
        rgb_already_windowed=False,
        dirty_tiles=(),
        tile_payloads=payloads,
        tile_delta=delta,
    )

    assert sorted(stats.committed_upserts) == [0, 1, 2, 3]


def test_tile_presentation_admission_uses_backend_cost_function():
    from arrayscope.display.montage import MontagePlan, MontageTile, RenderedTile
    from arrayscope.window.frame_session import FrameSession

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
    session = FrameSession(
        session_id=1,
        key=("test",),
        render_generation=1,
        level_key=None,
        level_expected_indices=(0, 1),
        plan=MontagePlan(
            axis=0, tile_shape=(2, 2), grid_shape=(1, 2), columns=2, rows=1, gap=0, tiles=tiles
        ),
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
    assert delta.force_refresh is False

    session.backend_refresh_pending = True
    _state, refresh_delta = session.build_tile_presentation({})
    assert refresh_delta.force_refresh is True


def test_tile_presentation_limits_do_not_hide_acknowledged_resident_tiles():
    from arrayscope.display.model.frame import TileCommitReport
    from arrayscope.display.model.tile_identity import tile_ack_identity
    from arrayscope.display.montage import MontagePlan, MontageTile, RenderedTile
    from arrayscope.window.frame_session import FrameSession

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
    session = FrameSession(
        session_id=1,
        key=("test",),
        render_generation=1,
        level_key=None,
        level_expected_indices=tuple(range(4)),
        plan=MontagePlan(
            axis=0, tile_shape=(2, 2), grid_shape=(1, 4), columns=4, rows=1, gap=0, tiles=tiles
        ),
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

    _state, delta = session.build_tile_presentation({})
    session.acknowledge_tile_presentation(
        delta,
        TileCommitReport(
            presented_tiles=frozenset(delta.upserts),
            committed_upserts=frozenset(delta.upserts),
            delta_key=(delta.base_revision, delta.target_revision),
            presented_identities={
                tile: tile_ack_identity(payload) for tile, payload in delta.upserts.items()
            },
        ),
    )
    session.mark_presented(tuple(delta.upserts))
    session.visible_tiles = tiles

    _state, delta = session.build_tile_presentation(
        {},
        max_upserts=1,
        max_upsert_bytes=1,
        upsert_cost_fn=lambda _payload: 1024 * 1024,
    )

    assert delta.upserts == {}
    assert delta.active_tiles == (0, 1, 2, 3)


def test_tile_presentation_limits_cap_resident_retarget_upserts():
    from arrayscope.display.montage import MontagePlan, MontageTile, RenderedTile
    from arrayscope.window.frame_session import FrameSession

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
    session = FrameSession(
        session_id=1,
        key=("test",),
        render_generation=1,
        level_key=None,
        level_expected_indices=tuple(range(4)),
        plan=MontagePlan(
            axis=0, tile_shape=(2, 2), grid_shape=(1, 4), columns=4, rows=1, gap=0, tiles=tiles
        ),
        view_state=None,
        document=None,
        montage_axis=0,
        colormap_lut=None,
        viewport_shape=(2, 8),
        view_range=((0.0, 8.0), (0.0, 2.0)),
        output_dtype=np.dtype("float32"),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
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
    session.lifecycle.presentation_confirmed(tuple(delta.upserts))
    for tile_number, payload in state.payloads.items():
        session.acknowledged_source_ids.add(payload.source_id)
        session.dirty_payloads[int(tile_number)] = None
        session.pending_payload_upserts[int(tile_number)] = None

    _state, delta = session.build_tile_presentation(
        {},
        max_upserts=2,
        max_upsert_bytes=1,
        upsert_cost_fn=lambda _payload: 0,
        # PyQtGraph semantics: remaps are byte-free but not time-free, so the
        # item cap paces them in priority order. Persistent GPU residency can
        # explicitly opt out when a remap is a single page-geometry update.
        pace_resident_retargets=True,
    )

    assert tuple(delta.upserts) == (1, 2)


def test_interactive_viewport_prunes_stale_montage_tile_work(qt_app):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import MontageTileState, make_montage_plan
    from arrayscope.window.frame_controller import FrameControllerMixin
    from arrayscope.window.frame_session import FrameSession

    class Controller:
        def __init__(self):
            self.groups = []

        def clear_group(self, group):
            self.groups.append(group)

    class Window(QtCore.QObject, FrameControllerMixin):
        def __init__(self):
            super().__init__()
            self.win = self

    state = ViewState.from_shape((2, 2, 8)).with_montage_axis(2, indices=tuple(range(8)), text=":")
    plan = make_montage_plan(
        state, axis=2, indices=tuple(range(8)), tile_shape=(2, 2), columns=8, gap=1
    )
    controller = Controller()
    session = FrameSession(
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
    )
    session.active_tile_requests.add(7)
    session.tile_states = [MontageTileState.UNLOADED for _tile in plan.tiles]
    session.tile_states[7] = MontageTileState.LOADING
    win = Window()
    win._frame_session = session
    win.view_state = state
    win.montage_tile_evaluation_controller = controller
    win._viewport_interaction_active = True

    win._prune_stale_montage_tile_work(session)

    assert session.required_target_unsettled_tiles() == (0,)
    assert 7 in session.loading_tiles
    assert 7 in session.active_tile_requests
    assert session.tile_states[7] == MontageTileState.LOADING
    assert controller.groups == []


def test_interactive_viewport_expansion_admits_only_required_tiles(qt_app, monkeypatch):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.evaluator import _document_key
    from arrayscope.operations.pipeline import ArrayDocument
    from arrayscope.window import frame_effects as montage_commit
    from arrayscope.window.frame_controller import FrameControllerMixin
    from arrayscope.window.frame_session import FrameSession

    class Window(QtCore.QObject, FrameControllerMixin):
        def __init__(self, document, state, viewport_plan):
            super().__init__()
            self.win = self
            self.document = document
            self.view_state = state
            self._viewport_plan = viewport_plan
            self._viewport_interaction_active = True
            self._montage_viewport_addition_batch_size = 3
            self.resolved_batches = []
            self.pipeline_retargets = 0
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

        def apply_montage_presentation(self, session):
            self.commits += 1

        def retarget_frame_pipeline(self, session, **_kwargs):
            self.pipeline_retargets += 1

    document = ArrayDocument(np.zeros((20, 20, 16), dtype=np.float32))
    state = ViewState.from_shape(document.current_shape).with_montage_axis(
        2,
        columns=4,
        indices=tuple(range(16)),
        text=":",
    )
    plan = make_montage_plan(
        state,
        axis=2,
        indices=tuple(range(16)),
        tile_shape=(20, 20),
        columns=4,
    )
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(16)),
        viewport_shape=(200, 200),
        tile_shape=(20, 20),
        plan=plan,
        view_range=((21.0, 39.0), (21.0, 39.0)),
        shader_display=True,
        persistent_tile_residency=True,
    )
    session = FrameSession(
        session_id=11,
        key=frame_session_key(_document_key(document), state, viewport_plan, None),
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=tuple(range(16)),
        plan=plan,
        view_state=state,
        document=document,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(200, 200),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=(),
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
    )
    win = Window(document, state, viewport_plan)
    win._frame_session = session
    submitted_stage_plans = []
    monkeypatch.setattr(
        montage_commit,
        "build_stage_fan_in_plan",
        lambda *_args, **_kwargs: pytest.fail(
            "interactive viewport update must not plan stage fan-in"
        ),
    )
    monkeypatch.setattr(
        "arrayscope.window.frame_controller.montage_commit.submit_deferred_stage_fan_in_plan",
        lambda _renderer, _session, tiles: submitted_stage_plans.append(tuple(tiles)) or True,
    )

    assert win._try_update_montage_viewport_only() is True

    required = set(session.required_tile_numbers())
    assert required
    assert required == set(session.frame_plan.active_region_ids)
    assert len(required) < len(plan.tiles), "the fixture needs a non-required coverage shell"
    assert set(session.scheduling_policy.verdict.required_tiles) == required
    assert len(win.resolved_batches) == len(required)
    resolved = [tile for batch in win.resolved_batches for tile in batch]
    assert set(resolved) == required
    assert set(session.required_target_unsettled_tiles()) == required
    assert session.loading_tiles == set()
    assert win.pipeline_retargets == 2
    assert submitted_stage_plans == [session.deferred_missing_tiles]
    assert {int(tile.montage_index) for tile in session.deferred_missing_tiles} == required
    assert session.stage_planning_deferred is True
    assert not getattr(win, "_montage_viewport_update_pending", False)
    assert not hasattr(win, "_montage_viewport_continue_immediately")
    assert win._last_montage_viewport_deferred_additions == 0


def test_viewport_update_retains_existing_deferred_tiles_without_quiet_gate(qt_app, monkeypatch):
    from pyqtgraph.Qt import QtCore, QtWidgets

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.evaluator import _document_key
    from arrayscope.operations.pipeline import ArrayDocument
    from arrayscope.window.frame_controller import FrameControllerMixin
    from arrayscope.window.frame_session import FrameSession

    class Window(QtCore.QObject, FrameControllerMixin):
        def __init__(self, document, state, viewport_plan):
            super().__init__()
            self.win = self
            self.document = document
            self.view_state = state
            self._viewport_plan = viewport_plan
            self.pipeline_retargets = 0
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

        def retarget_frame_pipeline(self, session, **_kwargs):
            self.pipeline_retargets += 1

        def apply_montage_presentation(self, session):
            pass

    document = ArrayDocument(np.zeros((2, 2, 4), dtype=np.float32))
    state = ViewState.from_shape(document.current_shape).with_montage_axis(
        2, columns=4, indices=tuple(range(4)), text=":"
    )
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
    session = FrameSession(
        session_id=12,
        key=frame_session_key(_document_key(document), state, viewport_plan, None),
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
        stage_planning_deferred=True,
        deferred_missing_tiles=tuple(plan.tiles),
    )
    win = Window(document, state, viewport_plan)
    win._frame_session = session
    win._viewport_interaction_active = False
    monkeypatch.setattr(
        QtWidgets.QApplication, "mouseButtons", lambda: QtCore.Qt.MouseButton.NoButton
    )

    assert win._try_update_montage_viewport_only() is True

    assert win.pipeline_retargets == 1
    assert session.stage_planning_deferred is True
    assert session.deferred_missing_tiles == tuple(plan.tiles)


def test_interactive_index_window_retarget_defers_stage_fan_in_without_planning(
    qt_app, monkeypatch
):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.evaluator import _document_key
    from arrayscope.operations.pipeline import ArrayDocument
    from arrayscope.window import frame_effects as montage_commit
    from arrayscope.window.frame_controller import FrameControllerMixin
    from arrayscope.window.frame_session import FrameSession

    class Planner:
        def plan(self, **kwargs):
            return SimpleNamespace(target=kwargs["target"])

    class Evaluator:
        def montage_tile_key_batch(self, **_kwargs):
            def key_for(view_state):
                return ("src", int(view_state.slice_indices[2]))

            return key_for

    class Surface:
        def __init__(self):
            self.invalidations = []
            self.capabilities = ImageViewBackendCapabilities(
                name="vispy",
                persistent_tile_residency=True,
                shader_windowing=True,
            )
            self.widget = None

        def present_tiled(self, _presentation):
            raise AssertionError("presentation is owned by the test window")

        def invalidate_tiled_presentation(self, reason, *, hide_pixels=True):
            self.invalidations.append(str(reason))

        def hide_tiled_presentation(self, _reason):
            pass

        def reset_tiled_residency(self, _reason):
            pass

        def set_profile_bounds(self, _bounds):
            pass

        def apply_camera(self, _image_shape, _viewport_policy, **_kwargs):
            pass

        def map_scene_to_overlay(self, scene_pos):
            return scene_pos

        def current_viewport_rect(self):
            return None

        def presentation_diagnostics(self):
            return {}

        def interaction_event_owner(self):
            return "test"

        def sync_interaction_state(self, _state):
            pass

        def reset_surface(self, _reason):
            pass

        def teardown_surface(self):
            pass

    class Window(QtCore.QObject, FrameControllerMixin):
        def __init__(self, document, old_state):
            super().__init__()
            self.win = self
            self.document = document
            self.view_state = old_state
            self.operation_evaluator = Evaluator()
            self.surface = Surface()
            self.img_view = SimpleNamespace(
                surface=self.surface,
                rendering_capabilities=self.surface.capabilities,
            )
            self._viewport_interaction_active = True
            self.pipeline_retargets = 0
            self.commits = 0
            self._last_montage_viewport_plan_ms = 0.0
            self._last_montage_cache_resolve_ms = 0.0

        def _montage_frame_planner(self):
            return Planner()

        def _montage_quality_policy_mode(self):
            return self._frame_session.lod_policy_mode

        def _capture_render_generation(self):
            return 2

        def _ensure_montage_watchdog(self):
            pass

        def _ensure_montage_level_stats(self, *_args, **_kwargs):
            pass

        def _queue_montage_cached_level_stats(self, *_args, **_kwargs):
            pass

        def commit_frame_session_presentation(self, session):
            assert session is self._frame_session
            self.commits += 1

        def retarget_frame_pipeline(self, session, **_kwargs):
            assert session is self._frame_session
            self.pipeline_retargets += 1

    document = ArrayDocument(np.zeros((2, 2, 6), dtype=np.float32))
    old_state = ViewState.from_shape(document.current_shape).with_montage_axis(
        2, columns=2, indices=(0, 1), text="0:2"
    )
    new_state = ViewState.from_shape(document.current_shape).with_montage_axis(
        2, columns=2, indices=(1, 2), text="1:3"
    )
    old_plan = make_montage_plan(old_state, axis=2, indices=(0, 1), tile_shape=(2, 2), columns=2)
    new_plan = make_montage_plan(new_state, axis=2, indices=(1, 2), tile_shape=(2, 2), columns=2)
    old_viewport = MontageViewportPlan(
        2, (0, 1), (4, 8), (2, 2), old_plan, ((0.0, 4.0), (0.0, 2.0)), True, True
    )
    new_viewport = MontageViewportPlan(
        2, (1, 2), (4, 8), (2, 2), new_plan, ((0.0, 4.0), (0.0, 2.0)), True, True
    )
    session = FrameSession(
        session_id=1,
        key=frame_session_key(_document_key(document), old_state, old_viewport, None),
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=(0, 1),
        plan=old_plan,
        view_state=old_state,
        document=document,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(4, 8),
        view_range=old_viewport.view_range,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=old_plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        display_committed=True,
    )
    session.shader_display = True
    win = Window(document, old_state)
    win._frame_session = session
    submitted_stage_plans = []
    monkeypatch.setattr(
        montage_commit,
        "build_stage_fan_in_plan",
        lambda *_args, **_kwargs: pytest.fail(
            "active index-window retarget must not plan stage fan-in"
        ),
    )
    monkeypatch.setattr(
        "arrayscope.window.frame_controller.montage_commit.submit_deferred_stage_fan_in_plan",
        lambda _renderer, _session, tiles: submitted_stage_plans.append(tuple(tiles)) or True,
    )

    handled = win._maybe_retarget_frame_session(
        session,
        document=document,
        axis=2,
        view_state=new_state,
        viewport_plan=new_viewport,
        plan=new_plan,
        policy=None,
        colormap_lut=None,
        window_mode="relative",
        force_auto=False,
        user_levels=None,
        output_dtype=np.dtype(np.float32),
        shader_display=True,
        cached_tiles=(),
        missing_tiles=(new_plan.tiles[1],),
        skipped_tiles=(),
        all_indices=(1, 2),
        display_tiles=new_plan.tiles,
        current_range=new_viewport.view_range,
        viewport_shape=(4, 8),
    )

    assert handled is True
    assert session.stage_planning_deferred is True
    assert session.stage_planning_async is False
    assert session.deferred_missing_tiles == (new_plan.tiles[1],)
    assert session.retained_stage_decision == "deferred-interaction"
    assert submitted_stage_plans == [(new_plan.tiles[1],)]
    assert win.surface.invalidations == []
    assert win.commits == 1
    assert win.pipeline_retargets == 1


def test_same_key_view_range_change_uses_viewport_retarget_not_session_rebirth(qt_app):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.evaluator import _document_key
    from arrayscope.operations.pipeline import ArrayDocument
    from arrayscope.window.frame_controller import FrameControllerMixin
    from arrayscope.window.frame_session import FrameSession

    class Window(QtCore.QObject, FrameControllerMixin):
        def __init__(self, document):
            super().__init__()
            self.win = self
            self.document = document
            self.viewport_retargets = 0

        def _montage_quality_policy_mode(self):
            return self._frame_session.lod_policy_mode

        def _try_update_montage_viewport_only(self):
            self.viewport_retargets += 1
            return True

    document = ArrayDocument(np.zeros((2, 2, 6), dtype=np.float32))
    state = ViewState.from_shape(document.current_shape).with_montage_axis(
        2, columns=3, indices=tuple(range(6)), text=":"
    )
    plan = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(2, 2), columns=3)
    old_viewport = MontageViewportPlan(
        2, tuple(range(6)), (4, 12), (2, 2), plan, ((0.0, 6.0), (0.0, 4.0)), True, True
    )
    new_viewport = MontageViewportPlan(
        2, tuple(range(6)), (4, 12), (2, 2), plan, ((1.0, 3.0), (0.5, 2.5)), True, True
    )
    session = FrameSession(
        session_id=1,
        key=frame_session_key(_document_key(document), state, old_viewport, None),
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=tuple(range(6)),
        plan=plan,
        view_state=state,
        document=document,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=old_viewport.viewport_shape,
        view_range=old_viewport.view_range,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        display_committed=True,
    )
    session.shader_display = True
    win = Window(document)
    win._frame_session = session

    handled = win._maybe_retarget_frame_session(
        session,
        document=document,
        axis=2,
        view_state=state,
        viewport_plan=new_viewport,
        plan=plan,
        policy=None,
        colormap_lut=None,
        window_mode="relative",
        force_auto=False,
        user_levels=None,
        output_dtype=np.dtype(np.float32),
        shader_display=True,
        cached_tiles=(),
        missing_tiles=(),
        skipped_tiles=(),
        all_indices=tuple(range(6)),
        display_tiles=plan.tiles,
        current_range=new_viewport.view_range,
        viewport_shape=new_viewport.viewport_shape,
    )

    assert handled is True
    assert win.viewport_retargets == 1
    assert getattr(win, "_frame_session_retarget_last_reject", "") != "view-range"


def test_resize_retarget_requests_presentation_through_gate(qt_app):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.backend_contract import ImageViewBackendCapabilities
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.pipeline import ArrayDocument
    from arrayscope.window.frame_controller import FrameControllerMixin
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

        def mark_ladder_swaps_for_viewport(self, **_kwargs):
            return False

        def sync_lifecycle_scope(self):
            return None

        pending_rung_materializations = ()

    class Window(QtCore.QObject, FrameControllerMixin):
        def __init__(self):
            super().__init__()
            self.win = self
            self.view_state = ViewState.from_shape((4, 4, 4)).with_montage_axis(
                2, indices=tuple(range(4)), text=":"
            )
            self.document = ArrayDocument(np.zeros((4, 4, 4), dtype=np.float32))
            self.img_view = SimpleNamespace(
                rendering_capabilities=ImageViewBackendCapabilities(
                    name="pyqtgraph",
                ),
                montageDisplayMode=lambda: "tile_layer",
            )
            self._frame_session = Session()
            self.commits = 0
            self.presentation_requests = 0

        def _is_current_frame_session(self, session_id, key):
            return session_id == 1 and key == ("session",)

        def _montage_frame_planner(self):
            return Planner()

        def commit_frame_session_presentation(self, session):
            assert session is self._frame_session
            self.commits += 1

        def retarget_frame_pipeline(self, session, **_kwargs):
            assert session is self._frame_session

        def apply_montage_presentation(self, session):
            assert session is self._frame_session
            self.presentation_requests += 1

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
    assert win._frame_session.retargeted is True
    assert win.commits == 0
    assert win.presentation_requests == 1


def test_nonpersistent_tile_layer_viewport_update_preserves_level_target(qt_app):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.evaluator import _document_key
    from arrayscope.operations.pipeline import ArrayDocument
    from arrayscope.window.frame_controller import FrameControllerMixin
    from arrayscope.window.frame_session import FrameSession

    class Window(QtCore.QObject, FrameControllerMixin):
        def __init__(self, document, state, viewport_plan):
            super().__init__()
            self.win = self
            self.document = document
            self.view_state = state
            self._viewport_plan = viewport_plan
            self.pipeline_retargets = 0
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

        def retarget_frame_pipeline(self, session, **_kwargs):
            self.pipeline_retargets += 1

        def apply_montage_presentation(self, session):
            self.commits += 1

    document = ArrayDocument(np.zeros((2, 2, 4), dtype=np.float32))
    state = ViewState.from_shape(document.current_shape).with_montage_axis(
        2, columns=4, indices=tuple(range(4)), text=":"
    )
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
    session = FrameSession(
        session_id=13,
        key=frame_session_key(_document_key(document), state, viewport_plan, None),
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
    )
    session.level_generation.target_levels = (2.0, 4.0)
    session.set_level_update_pending(True)
    win = Window(document, state, viewport_plan)
    win._frame_session = session
    win._viewport_interaction_active = False

    assert win._try_update_montage_viewport_only() is True

    assert win._frame_session is session
    assert session.level_generation.target_levels == (2.0, 4.0)


def test_hover_priority_retarget_changes_canonical_pipeline_order(qt_app):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.model.tile_priority import prioritize_tiles
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window.frame_controller import FrameControllerMixin
    from arrayscope.window.frame_session import FrameSession

    class Window(QtCore.QObject, FrameControllerMixin):
        def __init__(self, state, viewport_plan):
            super().__init__()
            self.win = self
            self.view_state = state
            self._viewport_plan = viewport_plan
            self.scheduled = []
            self.img_view = SimpleNamespace(
                rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph")
            )

        def _montage_viewport_plan(self, view_state, *, view_range=None):
            return self._viewport_plan

        def _is_current_frame_session(self, session_id, key):
            return True

        def retarget_frame_pipeline(self, session, **_kwargs):
            ordered = prioritize_tiles(
                session.plan.tiles,
                context=session.tile_priority_context(),
            )
            self.scheduled.append(int(ordered[0].montage_index))

    state = ViewState.from_shape((2, 2, 4)).with_montage_axis(
        2, columns=4, indices=tuple(range(4)), text=":"
    )
    plan = make_montage_plan(state, axis=2, indices=tuple(range(4)), tile_shape=(2, 2), columns=4)
    viewport_plan = SimpleNamespace(priority_focus=(10.0, 1.0))
    session = FrameSession(
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
    )
    win = Window(state, viewport_plan)
    win._frame_session = session

    win.apply_montage_priority_retarget()

    assert win.scheduled == [3]


def test_tiled_commit_syncs_hover_geometry_after_backend_ack(qt_app):
    from dataclasses import replace

    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.model.frame import DisplayFrameKey
    from arrayscope.display.montage import MontageTileState
    from arrayscope.window.frame_controller import FrameControllerMixin

    class Window(QtCore.QObject, FrameControllerMixin):
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
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.model.frame import DisplayFrameKey
    from arrayscope.display.montage import MontageTileState
    from arrayscope.window.frame_controller import FrameControllerMixin

    class Window(QtCore.QObject, FrameControllerMixin):
        def __init__(self):
            super().__init__()
            self.win = self

        def _set_committed_display_frame(self, frame):
            self._committed_display_frame = frame

    first_state = ViewState.from_shape((2, 2, 4)).with_montage_axis(
        2, columns=2, indices=(0, 1), text="0:2"
    )
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
    from arrayscope.window.frame_effects import (
        persistent_gpu_tile_residency_backend,
        persistent_tile_residency_backend,
    )
    from arrayscope.window.montage_viewport import montage_viewport_retarget_policy

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
    assert (
        montage_viewport_retarget_policy(capabilities, "vispy_tile_layer").coverage_margin_tiles
        == 1
    )
    assert (
        persistent_tile_residency_backend(
            _window_ns(img_view=SimpleNamespace(rendering_capabilities=persistent_nonvispy)),
            SimpleNamespace(),
        )
        is True
    )
    assert (
        persistent_tile_residency_backend(
            _window_ns(img_view=SimpleNamespace(rendering_capabilities=persistent_without_shader)),
            SimpleNamespace(),
        )
        is True
    )
    assert (
        persistent_gpu_tile_residency_backend(
            _window_ns(img_view=SimpleNamespace(rendering_capabilities=persistent_without_shader)),
            SimpleNamespace(),
        )
        is False
    )
    assert montage_viewport_retarget_policy(direct_nonpersistent, "tile_layer").enabled is True
    assert (
        montage_viewport_retarget_policy(direct_nonpersistent, "tile_layer").coverage_margin_tiles
        == 0
    )
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
        (
            ("montage_tile", "doc", 2),
            "texture_kind",
            "complex_rg32f",
            "shader",
            None,
            "lod",
            4,
            2,
            1,
        ),
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


def test_retained_payload_store_is_bounded_by_physical_bytes():
    from arrayscope.display.model.frame import DisplayTilePayload

    payloads = {
        index: DisplayTilePayload(
            index,
            index,
            np.ones((4, 4), dtype=np.float32),
            None,
            ("payload", index),
        )
        for index in range(4)
    }
    one_payload = next(iter(payloads.values())).nbytes
    store = RetainedTiledPayloadStore(limit=10)

    store.remember_acknowledged(payloads, max_bytes=2 * one_payload)

    assert store.bytes_used <= 2 * one_payload
    assert set(store.payloads_by_base_source()) == {("payload", 2), ("payload", 3)}


def test_retained_payload_store_has_a_safe_default_byte_cap():
    store = RetainedTiledPayloadStore()

    assert store.max_bytes is not None
    assert store.max_bytes > 0


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
    from arrayscope.window.frame_controller import FrameControllerMixin

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

    class _Window(FrameControllerMixin):
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
    from arrayscope.window.frame_controller import FrameControllerMixin

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

    class _Window(FrameControllerMixin):
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


def test_retained_payload_wrapper_rejects_an_old_display_axis_crop():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.model.tile_identity import TileIdentity
    from arrayscope.display.shader_mapping import TexturePlaneKind

    old_state = ViewState.from_shape((8, 8, 1)).with_axis_range(
        0,
        indices=(1, 2, 3, 4),
        text="1:5",
    )
    new_state = old_state.with_axis_range(
        0,
        indices=(2, 3, 4, 5),
        text="2:6",
    )
    texture = np.ones((4, 8), dtype=np.float32)
    identity = TileIdentity(
        document_generation="doc",
        operation_key=(),
        source_index=0,
        image_axes=old_state.image_axes,
        axis_flips=old_state.axis_flipped,
        channel=old_state.channel,
        complex_mapping=("scalar", "real", "mapped"),
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_generation=(
            old_state.shape,
            old_state.slice_indices,
            old_state.axis_range_indices,
            old_state.axis_fftshifted,
        ),
    )
    payload = DisplayTilePayload(
        0,
        0,
        texture,
        None,
        ("canonical-source", 0),
        tile_identity=identity,
    )

    assert _payload_compatible_with_tile(payload, old_state, shader_display=True)
    assert not _payload_compatible_with_tile(payload, new_state, shader_display=True)


def test_tiled_payload_source_id_follows_semantic_materialization_identity():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import RenderedTile, make_montage_plan
    from arrayscope.window.frame_session import FrameSession

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

    session = FrameSession(
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
    )

    first = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]
    session.rendered_tiles[0] = rendered(2.0)
    second = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]
    third = session.snapshot_display_tile_payloads({0: ("tile", 0, "revision", 2)})[0]

    assert _base_tile_source_id(first.source_id) == ("tile", 0)
    assert _base_tile_source_id(second.source_id) == ("tile", 0)
    assert first.source_id == second.source_id
    assert third.source_id != second.source_id


def test_initial_loading_only_tile_layer_commit_is_skipped(qt_app):
    pytest.importorskip("pyqtgraph")
    from arrayscope.display.geometry import MontageGeometry
    from arrayscope.display.model.frame import TilePresentationDelta, TilePresentationState
    from arrayscope.window.frame_controller import FrameControllerMixin

    class _Window(FrameControllerMixin):
        def __init__(self):
            self.win = self
            self.img_view = SimpleNamespace(
                rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph")
            )
            self.commits = 0

        def _frame_session_is_current(self, _session):
            return True

        def _montage_tile_source_ids(self, _session):
            return {}

        def _display_committer(self):
            self.commits += 1
            raise AssertionError("loading-only first commit must not reach backend")

        def request_montage_replan(self, _session):
            return None

        def _montage_level_source_for_session(self, _session, *, allow_partial=False):
            return None

    geometry = MontageGeometry(indices=(0,), tile_shape=(2, 2), columns=1, rows=1, gap=0)
    delta = TilePresentationDelta(
        structure_revision=0,
        payload_revision=0,
        visibility_revision=0,
        level_revision=0,
        histogram_revision=0,
        viewport_revision=0,
        base_revision=0,
        target_revision=0,
        active_tiles=(),
        planned_tiles=(0,),
    )
    session = SimpleNamespace(
        scheduling_policy=_refine_scheduling_policy(),
        display_committed=False,
        atomic_successor_pending=False,
        force_auto=False,
        tile_presentation_state=TilePresentationState(),
        consume_dirty_tiles=lambda: (),
        ensure_tile_states=lambda: (),
        build_tile_presentation=lambda *_args, **_kwargs: (TilePresentationState(), delta),
        _selected_lod_factor=lambda: 1,
        has_pending_level_update=lambda: False,
        has_stale_level_presentations=lambda: False,
        level_generation=SimpleNamespace(target_levels=None),
        user_levels_override=None,
        final_commit_pending=True,
        flush_pending=True,
        plan=SimpleNamespace(geometry=geometry, display_shape=(2, 2)),
        view_state=None,
        rgb=False,
        lifecycle=SimpleNamespace(
            presented_tiles=frozenset(),
            snapshot=lambda: SimpleNamespace(counts={}),
        ),
    )

    win = _Window()
    win.commit_frame_session_presentation(session)

    assert win.commits == 0
    assert session.final_commit_pending is False
    assert session.flush_pending is False


def test_replacement_plan_keeps_committed_content_extent_until_backend_commit():
    from arrayscope.window.frame_controller import FrameControllerMixin

    published = []

    class Window(FrameControllerMixin):
        def __init__(self):
            self.win = self
            self._committed_display_frame = object()

        def _publish_montage_content_extent(self, plan):
            published.append(plan)

    win = Window()
    replacement_plan = object()

    assert win._publish_first_frame_content_extent(replacement_plan) is False
    assert published == []

    win._committed_display_frame = None
    assert win._publish_first_frame_content_extent(replacement_plan) is True
    assert published == [replacement_plan]


def test_interactive_cache_hit_requires_committed_semantic_montage_mapping():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.evaluator import _document_key
    from arrayscope.operations.pipeline import ArrayDocument
    from arrayscope.window.frame_controller import FrameControllerMixin

    class Evaluator:
        def cached_montage_tile(self, *_args, **_kwargs):
            return object()

    class Window(FrameControllerMixin):
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
    old_state = ViewState.from_shape(document.current_shape).with_montage_axis(
        2, columns=2, indices=(0, 1), text="0:2"
    )
    new_state = ViewState.from_shape(document.current_shape).with_montage_axis(
        2, columns=2, indices=(1, 2), text="1:3"
    )
    old_plan = make_montage_plan(old_state, axis=2, indices=(0, 1), tile_shape=(2, 2), columns=2)
    new_plan = make_montage_plan(new_state, axis=2, indices=(1, 2), tile_shape=(2, 2), columns=2)
    old_viewport = MontageViewportPlan(
        2, (0, 1), (4, 8), (2, 2), old_plan, ((0.0, 4.0), (0.0, 2.0)), False, True
    )
    new_viewport = MontageViewportPlan(
        2, (1, 2), (4, 8), (2, 2), new_plan, ((0.0, 4.0), (0.0, 2.0)), False, True
    )
    old_key = frame_session_key(_document_key(document), old_state, old_viewport, None)
    new_key = frame_session_key(_document_key(document), new_state, new_viewport, None)

    win = Window(document, new_state, new_viewport)
    win._committed_display_frame = SimpleNamespace(key=SimpleNamespace(request_key=old_key))

    assert old_key != new_key
    assert win._interactive_frame_cache_hit() is False

    win._committed_display_frame = SimpleNamespace(key=SimpleNamespace(request_key=new_key))

    assert win._interactive_frame_cache_hit() is True


def test_montage_viewport_update_token_tracks_viewport_revision():
    from arrayscope.window.render_contract import montage_work_token as _montage_work_token

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
    from arrayscope.window.frame_controller import FrameControllerMixin

    class _Window(FrameControllerMixin):
        def __init__(self):
            self.win = self
            self.view_state = SimpleNamespace(montage_axis=2)
            self._frame_session = SimpleNamespace(
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

    win.apply_montage_viewport_retarget()

    assert win.called is False


def test_montage_viewport_continuation_yields_to_receiver_owned_gate():
    from arrayscope.window.frame_controller import FrameControllerMixin

    class _Window(FrameControllerMixin):
        def __init__(self):
            self.win = self
            self.view_state = SimpleNamespace(montage_axis=2)
            self._frame_session = SimpleNamespace(
                session_id=1,
                key="session",
                render_generation=2,
                payload_revision=3,
                level_revision=4,
                viewport_revision=5,
            )
            self.calls = 0
            self.public_retargets = 0
            self.scheduled = []

        def _try_update_montage_viewport_only(self):
            self.calls += 1
            if self.calls < 3:
                self._montage_viewport_update_pending = True
                self._montage_viewport_continue_immediately = True
            return True

        def retarget_montage_viewport(self):
            self.public_retargets += 1
            super().retarget_montage_viewport()

        def _schedule_frame_viewport_update(self, *, delay_ms=None):
            self.scheduled.append(delay_ms)

    win = _Window()

    win.apply_montage_viewport_retarget()

    assert win.calls == 1
    assert win.scheduled == [1]
    assert win.public_retargets == 0
    assert not getattr(win, "_montage_viewport_update_running", False)


def test_loading_montage_profile_retry_waits_for_visibility_without_timer():
    from arrayscope.window.frame_controller import FrameControllerMixin

    class _LiveProfile:
        def isChecked(self):
            return True

    class _ProfileDock:
        visible = False

        def isVisible(self):
            return bool(self.visible)

    class _Window(FrameControllerMixin):
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


def test_frame_session_key_excludes_transient_viewport_range():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan

    state = ViewState.from_shape((4, 4, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(4, 4), columns=3)
    first = MontageViewportPlan(
        2, tuple(range(6)), (100, 100), (4, 4), plan, ((0, 10), (0, 10)), True, True
    )
    second = MontageViewportPlan(
        2, tuple(range(6)), (100, 100), (4, 4), plan, ((10, 20), (0, 10)), True, True
    )

    assert frame_session_key("doc", state, first, None) == frame_session_key(
        "doc", state, second, None
    )


def test_frame_session_key_changes_with_population_but_not_layout_reflow():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan

    state = ViewState.from_shape((4, 4, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    plan3 = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(4, 4), columns=3)
    plan2 = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(4, 4), columns=2)
    base = MontageViewportPlan(2, tuple(range(6)), (100, 100), (4, 4), plan3, None, True, True)
    changed_population = MontageViewportPlan(
        2, tuple(range(5)), (100, 100), (4, 4), plan3, None, True, True
    )
    changed_layout = MontageViewportPlan(
        2, tuple(range(6)), (100, 100), (4, 4), plan2, None, True, True
    )

    key = frame_session_key("doc", state, base, None)
    assert key != frame_session_key("doc", state, changed_population, None)
    assert key == frame_session_key("doc", state, changed_layout, None)


def test_direct_tiled_payload_retarget_allows_only_safe_layout_reflow():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window.frame_effects import safe_tiled_payload_geometry_retarget

    state = ViewState.from_shape((4, 4, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    plan3 = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(4, 4), columns=3)
    plan2 = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(4, 4), columns=2)
    previous = DisplayGeometry(
        view_state=state, display_shape=plan3.display_shape, montage=plan3.geometry
    )
    reflow = DisplayGeometry(
        view_state=state, display_shape=plan2.display_shape, montage=plan2.geometry
    )
    changed_indices = make_montage_plan(
        state, axis=2, indices=tuple(range(5)), tile_shape=(4, 4), columns=3
    )
    incompatible = DisplayGeometry(
        view_state=state,
        display_shape=changed_indices.display_shape,
        montage=changed_indices.geometry,
    )

    assert safe_tiled_payload_geometry_retarget(previous, reflow)
    assert not safe_tiled_payload_geometry_retarget(previous, incompatible)


def test_automatic_level_generation_target_is_not_reclassified_as_user_intent():
    from arrayscope.window.frame_effects import session_requested_levels

    session = SimpleNamespace(
        level_generation=SimpleNamespace(target_levels=(-0.5, 7.5)),
        user_levels_override=None,
    )
    assert session_requested_levels(session) is None

    session.user_levels_override = (2.0, 4.0)
    assert session_requested_levels(session) == (2.0, 4.0)


# ---------------------------------------------------------------------------
# First-pass histogram publication: evidence-side obligation (idle-stall fix)
#
# Trace-proven deadlock (2026-07-15, /tmp/arrayscope-stall-18-1 seq 9703-9708
# and /tmp/arrayscope-stall-65-2 seq 37576-37585): after an index-window
# retarget resets ``first_pass_histogram_published``, the shared-transform
# DESIRED pass is barred behind that flag, and the flag is only set by an
# acknowledgement commit.  When the last backend-ack commit lands BEFORE the
# rough level evidence for the newly scrolled sources completes, the ack-time
# arm cannot fire and the settled-metadata refresh never opens (the required
# tiles are unsettled precisely BY the barred pass) — a closed wait cycle with
# an idle kernel.  ``_maybe_publish_after_level_evidence`` must arm the parked
# flush itself when it observes the completed first pass unpublished.
# ---------------------------------------------------------------------------


def _late_evidence_service(covered, *, drain_active=False):
    from arrayscope.render.level_stats import LevelStatsService

    service = LevelStatsService()
    service.win = SimpleNamespace(kernel=SimpleNamespace(visible_backlog=0))
    service._montage_level_tracker = lambda: SimpleNamespace(
        summary_for=lambda _key: SimpleNamespace(source_indices=frozenset(covered)),
    )
    service._montage_commit_drain_active = bool(drain_active)
    return service


def _late_evidence_session(**overrides):
    tiles = tuple(
        SimpleNamespace(montage_index=index, source_index=100 + index) for index in range(4)
    )
    presentation_requests = []
    session = SimpleNamespace(
        shader_display=True,
        first_pass_quality="preview",
        first_pass_histogram_published=False,
        flush_pending=False,
        final_commit_pending=False,
        display_committed=True,
        level_key=("levels", "late-first-pass"),
        plan=SimpleNamespace(tiles=tiles),
        first_pass_pixels_presented=lambda: True,
        required_tile_numbers=lambda: (0, 1),
        required_target_settled=lambda: False,
        pending_level_tiles=(),
        level_scan_remaining_tiles=0,
        histogram_aggregate_inflight=False,
        semantic_level_evidence_progress=None,
        dirty_payloads={},
        pending_payload_upserts={},
        pending_removals=(),
        pipeline=SimpleNamespace(
            effects=SimpleNamespace(
                request_presentation=lambda: presentation_requests.append(True),
            ),
        ),
    )
    session._presentation_requests = presentation_requests
    for name, value in overrides.items():
        setattr(session, name, value)
    return session


def test_late_first_pass_evidence_arms_publication_flush_and_resumes_presentation():
    """Evidence completing after the last ack commit must still publish."""

    service = _late_evidence_service({100, 101})
    session = _late_evidence_session()

    service._maybe_publish_after_level_evidence(session, processed=1)

    assert session.flush_pending is True
    assert session.final_commit_pending is True
    assert session._presentation_requests == [True]


def test_late_first_pass_evidence_defers_to_active_commit_drain():
    service = _late_evidence_service({100, 101}, drain_active=True)
    session = _late_evidence_session()

    service._maybe_publish_after_level_evidence(session, processed=1)

    # The obligation is parked on the session; the active commit's normal
    # backlog rearm owns the continuation (no receiver event from inside its
    # own handler).
    assert session.flush_pending is True
    assert session.final_commit_pending is True
    assert session._presentation_requests == []
    assert service._montage_gate_last_backlog is None


def test_late_first_pass_evidence_keeps_obligation_while_evidence_remains():
    service = _late_evidence_service({100, 101})
    session = _late_evidence_session(pending_level_tiles=(3,))

    service._maybe_publish_after_level_evidence(session, processed=1)

    # More evidence batches are in flight: the flush stays parked for the
    # final drain's resume path instead of committing per batch.
    assert session.flush_pending is True
    assert session.final_commit_pending is True
    assert session._presentation_requests == []


def test_published_first_pass_does_not_rearm_publication_flush():
    service = _late_evidence_service({100, 101})
    session = _late_evidence_session(first_pass_histogram_published=True)

    service._maybe_publish_after_level_evidence(session, processed=1)

    assert session.flush_pending is False
    assert session.final_commit_pending is False
    assert session._presentation_requests == []


def test_incomplete_first_pass_evidence_does_not_arm_publication_flush():
    service = _late_evidence_service({100})
    session = _late_evidence_session()

    service._maybe_publish_after_level_evidence(session, processed=1)

    assert session.flush_pending is False
    assert session.final_commit_pending is False
    assert session._presentation_requests == []


# ---------------------------------------------------------------------------
# Superseded wgpu histogram tasks: results are content-keyed evidence, never
# discarded (2026-07-18 journey-matrix v4: the discard latched
# coverage_evidence_pending on settled sessions), and refined remembered
# evidence satisfies the rough obligation without a fresh dispatch.
# ---------------------------------------------------------------------------


def _wgpu_evidence_service(session, *, cached=None):
    from arrayscope.render.level_stats import LevelStatsService

    service = LevelStatsService()
    accepted = []
    service.win = SimpleNamespace(
        img_view=SimpleNamespace(
            acceptResidentHistogramEvidence=lambda keys: accepted.extend(keys),
            residentHistogramEvidence=lambda _payloads: (),
        ),
        kernel=SimpleNamespace(visible_backlog=0),
    )
    tracker_updates = []
    service._montage_level_tracker = lambda: SimpleNamespace(
        ensure_expected=lambda key, expected: None,
        update_from_stats=lambda key, stats, aggregate=True: tracker_updates.append((key, stats)),
    )
    remembered = []
    service._remember_montage_source_level_stats = lambda level_key, stats: remembered.append(
        (level_key, stats)
    )
    service._cached_montage_source_level_stats = lambda level_key, source_index, quality: (
        cached or {}
    ).get(int(source_index))
    published = []
    service._maybe_publish_after_level_evidence = lambda current, processed: published.append(
        processed
    )
    service._frame_session = session
    service._probes = SimpleNamespace(
        accepted=accepted,
        tracker_updates=tracker_updates,
        remembered=remembered,
        published=published,
    )
    return service


def _wgpu_evidence_rows(*source_indices):
    return tuple(
        (
            SimpleNamespace(evidence_key=("ev", index), source_index=index),
            SimpleNamespace(source_index=index, refined=False, evidence_quality=1),
        )
        for index in source_indices
    )


def test_superseded_wgpu_histogram_results_absorb_into_current_session():
    session = _late_evidence_session()
    service = _wgpu_evidence_service(session)

    processed = service._absorb_late_wgpu_histogram_evidence(
        _wgpu_evidence_rows(100, 101),
        source_level_key=session.level_key,
        elapsed_ms=5.0,
    )

    probes = service._probes
    assert processed == 2
    assert [key for key, _ in probes.tracker_updates] == [session.level_key] * 2
    assert probes.accepted == [("ev", 100), ("ev", 101)]
    assert probes.published == [2]
    assert session.flush_pending
    assert session._presentation_requests  # re-arm turn requested


def test_superseded_wgpu_results_for_other_level_are_remembered_not_tracked():
    session = _late_evidence_session()
    service = _wgpu_evidence_service(session)

    processed = service._absorb_late_wgpu_histogram_evidence(
        _wgpu_evidence_rows(100),
        source_level_key=("levels", "previous-level"),
        elapsed_ms=5.0,
    )

    probes = service._probes
    assert processed == 0
    assert probes.tracker_updates == []
    assert [key for key, _ in probes.remembered] == [("levels", "previous-level")]
    assert probes.accepted == [("ev", 100)]
    assert probes.published == []
    assert session._presentation_requests  # settled session still un-latches


def test_refined_remembered_evidence_satisfies_rough_obligation_without_dispatch():
    refined = SimpleNamespace(source_index=100, refined=True, evidence_quality=3)
    session = _late_evidence_session(
        level_expected_indices=(100, 101),
        level_evidence_inflight=False,
        scheduling_policy=SimpleNamespace(verdict=SimpleNamespace(admits=lambda work: True)),
    )
    service = _wgpu_evidence_service(session, cached={100: refined, 101: refined})
    rows = tuple(row for row, _ in _wgpu_evidence_rows(100, 101))
    service.win.img_view.residentHistogramEvidence = lambda payloads: rows
    service.win.kernel.submit = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no dispatch expected when refined evidence is cached")
    )

    reused = service._queue_wgpu_resident_histogram_evidence(session, payloads={})

    probes = service._probes
    assert reused == 2
    assert probes.accepted == [("ev", 100), ("ev", 101)]
    assert [stats for _, stats in probes.tracker_updates] == [refined, refined]
    assert probes.published == [2]
