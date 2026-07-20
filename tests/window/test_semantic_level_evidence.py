from types import SimpleNamespace

import numpy as np

from arrayscope.core.view_state import ViewState
from arrayscope.core.window_levels import LevelSourceRank
from arrayscope.display.backend_contract import PYQTGRAPH_CAPABILITIES, VISPY_CAPABILITIES
from arrayscope.display.montage import make_montage_plan
from arrayscope.kernel import Lane, Priority
from arrayscope.operations.evaluator import OperationEvaluator
from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.render.level_stats import LevelStatsService
from arrayscope.window.frame_session import FrameSession


class _Token:
    cancelled = False


class _CapturingKernel:
    def __init__(self):
        self.tasks = []
        self.visible_backlog = 0

    def submit_speculative_batch(self, **kwargs):
        self.tasks.append(kwargs)
        return object()

    def submit(self, spec, **callbacks):
        task = {
            "fn": spec.fn,
            "lane": spec.lane,
            "priority": spec.priority,
            "pass_token": spec.pass_token,
            "scope": spec.scope,
            "generation": spec.supersession.value,
            **callbacks,
        }
        self.tasks.append(task)
        return object()

    def run_next(self):
        task = self.tasks.pop(0)
        fn = task["fn"]
        value = fn(_Token()) if task.get("pass_token") else fn()
        task.setdefault("max_items", len(getattr(value, "sources", ())))
        task["on_done"](value)
        return task


_DEFAULT_SELECTED = tuple(range(20))


def _session(data, *, session_id=1, selected=_DEFAULT_SELECTED, level_key=None):
    document = ArrayDocument(np.asarray(data))
    state = ViewState.from_shape(document.current_shape).with_montage_axis(
        2,
        columns=5,
        indices=selected,
        text=":",
    )
    plan = make_montage_plan(
        state,
        axis=2,
        indices=selected,
        tile_shape=document.current_shape[:2],
        columns=5,
    )
    session = FrameSession(
        session_id=int(session_id),
        key=("frame", int(session_id)),
        render_generation=int(session_id),
        level_key=level_key or ("levels", int(session_id)),
        level_expected_indices=tuple(int(index) for index in selected),
        plan=plan,
        view_state=state,
        document=document,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(640, 384),
        view_range=(
            (0.0, float(document.current_shape[1])),
            (0.0, float(document.current_shape[0])),
        ),
        output_dtype=document.base_data.dtype,
        rgb=False,
        window_mode=None,
        force_auto=True,
        visible_tiles=(plan.tiles[0],),
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
    )
    session.frame_plan = SimpleNamespace(active_region_ids=(0,))
    session.pipeline = SimpleNamespace(effects=SimpleNamespace(request_presentation=lambda: None))
    return session


def _service(session, *, capabilities=PYQTGRAPH_CAPABILITIES):
    kernel = _CapturingKernel()
    service = LevelStatsService()
    service.win = SimpleNamespace(
        kernel=kernel,
        operation_evaluator=OperationEvaluator(session.document),
        img_view=SimpleNamespace(rendering_capabilities=capabilities),
    )
    service._frame_session = session
    service._frame_session_is_current = lambda candidate: candidate is service._frame_session
    service._should_publish_montage_level_metadata = lambda _session, _summary: False
    return service, kernel


def _close_coverage_phase(session):
    session.scheduling_policy.observe(
        SimpleNamespace(first_pixels_presented=lambda _required: True)
    )


def test_semantic_owner_covers_full_population_without_admitting_offscreen_tiles():
    data = np.arange(32 * 48 * 20, dtype=np.float32).reshape(32, 48, 20)
    session = _session(data)
    service, kernel = _service(session)

    service._schedule_semantic_level_evidence(session)
    submitted = []
    while kernel.tasks:
        submitted.append(kernel.run_next())

    summary = service._montage_level_tracker().summary_for(session.level_key)
    target = session.semantic_level_evidence_target
    progress = session.semantic_level_evidence_progress

    assert session.required_tile_numbers() == (0,)
    assert tuple(tile.montage_index for tile in session.visible_tiles) == (0,)
    assert session.lifecycle.snapshot().visible_tiles == 1
    assert summary.rank == LevelSourceRank.MONTAGE_SAMPLED_FULL
    assert summary.source_indices == frozenset(range(20))
    assert target.target_population == 20
    assert target.pixel_limit == 8192
    assert target.blocking_batch_limit == 16
    assert target.background_batch_limit == 2
    assert progress.covered_sources == set(range(20))
    assert progress.pending_batches == 0
    assert progress.inflight_generation is None
    assert progress.blocking_reason == "ready"
    assert all(task["lane"] == Lane.DISPLAY_PREVIEW for task in submitted)
    # The visible-lane sweep is the complement producer for the first-pixel
    # wait; it carries the same INTERACTIVE priority as the tile evaluations
    # it gates (montage-entry blackout, 2026-07-18 dossier).
    assert all(task["priority"] == Priority.INTERACTIVE for task in submitted)
    assert all(task["pass_token"] is True for task in submitted)
    assert max(task["max_items"] for task in submitted) <= 16
    assert service.win.operation_evaluator.image_evaluations == 0
    assert session.rendered_tiles == {}
    assert session.display_tile_payloads == {}
    assert session.lifecycle.presented_tiles == frozenset()


def test_semantic_owner_rejects_superseded_generation_results():
    predecessor = _session(
        np.zeros((8, 10, 20), dtype=np.float32),
        session_id=1,
        level_key=("levels", "predecessor"),
    )
    service, kernel = _service(predecessor)
    publications = []
    predecessor.pipeline.effects.request_presentation = lambda: publications.append("predecessor")
    service._schedule_semantic_level_evidence(predecessor)
    old_task = kernel.tasks.pop(0)

    successor = _session(
        np.full((8, 10, 20), 100.0, dtype=np.float32),
        session_id=2,
        level_key=("levels", "successor"),
    )
    successor.pipeline.effects.request_presentation = lambda: publications.append("successor")
    service._frame_session = successor
    service.win.operation_evaluator.set_document(successor.document)
    service._schedule_semantic_level_evidence(successor)
    assert old_task["scope"] == kernel.tasks[0]["scope"] == "montage:semantic-level-evidence"
    assert old_task["generation"] != kernel.tasks[0]["generation"]

    old_value = old_task["fn"](_Token())
    old_task["on_done"](old_value)

    assert (
        service._montage_level_tracker().summary_for(successor.level_key).rank
        == LevelSourceRank.NONE
    )
    assert predecessor.semantic_level_evidence_progress.inflight_generation is None
    assert publications == []

    while kernel.tasks:
        kernel.run_next()

    summary = service._montage_level_tracker().summary_for(successor.level_key)
    assert summary.rank == LevelSourceRank.MONTAGE_SAMPLED_FULL
    assert summary.bounds == (99.0, 101.0)
    assert successor.semantic_level_evidence_progress.covered_sources == set(range(20))
    assert predecessor.semantic_level_evidence_progress.covered_sources == set()


def test_vispy_uses_background_batches_but_converges_to_the_same_population():
    data = np.arange(10 * 12 * 20, dtype=np.float32).reshape(10, 12, 20)
    session = _session(data)
    service, kernel = _service(session, capabilities=VISPY_CAPABILITIES)
    _close_coverage_phase(session)

    service._schedule_semantic_level_evidence(session)
    first = kernel.run_next()
    first_summary = service._montage_level_tracker().summary_for(session.level_key)

    assert first["max_items"] == 2
    assert first["lane"] == Lane.HISTOGRAM_REFINEMENT
    assert first_summary.rank == LevelSourceRank.MONTAGE_VISIBLE_SUBSET
    assert first_summary.source_indices == frozenset({0, 1})

    while kernel.tasks:
        kernel.run_next()

    final = service._montage_level_tracker().summary_for(session.level_key)
    assert final.rank == LevelSourceRank.MONTAGE_SAMPLED_FULL
    assert final.source_indices == frozenset(range(20))
    assert service.win.operation_evaluator.image_evaluations == 0
    assert session.display_tile_payloads == {}


def test_vispy_semantic_background_batches_publish_once_after_full_population():
    """Refinement batches must not replay the settled tiled presentation."""

    data = np.arange(10 * 12 * 20, dtype=np.float32).reshape(10, 12, 20)
    session = _session(data)
    service, kernel = _service(session, capabilities=VISPY_CAPABILITIES)
    publications = []
    session.shader_display = True
    session.display_committed = True
    session.first_pass_histogram_published = True
    session.required_target_settled = lambda: True
    _close_coverage_phase(session)
    session.pipeline.effects.request_presentation = lambda: publications.append(
        len(session.semantic_level_evidence_progress.covered_sources)
    )

    service._schedule_semantic_level_evidence(session)
    first = kernel.run_next()

    assert first["max_items"] == 2
    assert len(session.semantic_level_evidence_progress.covered_sources) == 2
    assert publications == []

    while kernel.tasks:
        kernel.run_next()

    assert publications == [20]


def test_semantic_evidence_diagnostics_are_constant_time_progress_truth():
    data = np.arange(12 * 16 * 20, dtype=np.float32).reshape(12, 16, 20)
    session = _session(data)
    service, kernel = _service(session)

    service._schedule_semantic_level_evidence(session)
    initial = session.semantic_level_evidence_diagnostics()

    assert initial == {
        "target_population": 20,
        "covered_sources": (),
        "covered_source_count": 0,
        "pending_batches": 2,
        "inflight_generation": session.semantic_level_evidence_target.generation,
        "blocking_reason": "worker-in-flight",
        "source_batch_limit": 16,
        "pixel_limit": 8192,
    }

    first = kernel.run_next()
    after_first = session.semantic_level_evidence_diagnostics()
    assert first["max_items"] == 16
    assert after_first["covered_source_count"] == 16
    assert after_first["pending_batches"] == 1
    assert len(after_first["covered_sources"]) == 16
    assert service._semantic_level_evidence_last_merged == 16
