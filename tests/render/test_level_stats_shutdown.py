"""Kernel shutdown must not wait out long level-evidence evaluations.

The 2026-07-19 shutdown change bounds the close callback under one global
join deadline, but that bound only holds if a currently running evidence
sweep observes its cancellation token at tile boundaries. These tests pin
the whole-process exit gate at the unit seam: shutdown during a long sweep
returns within the deadline with no leaked-thread diagnostics, and the
cancelled task's callbacks fire exactly once.
"""

from __future__ import annotations

import threading
import time
import warnings
from collections import deque
from time import monotonic
from types import SimpleNamespace

import pytest

from arrayscope.kernel import Kernel, ThreadWorkerBackend
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.render import level_stats
from arrayscope.render.level_stats import LevelStatsService
from arrayscope.tools.interaction_budget import bounded_interaction_settle_timeout_s


class _Verdict:
    coverage_open = True

    def admits(self, work) -> bool:
        return True


def _fake_session(batch_size: int):
    return SimpleNamespace(
        key="session-key",
        session_id=7,
        viewport_revision=0,
        level_key="level-key",
        # First display not committed: payload evidence is a visible
        # dependency, so the sweep takes the plain kernel.submit branch.
        display_committed=False,
        scheduling_policy=SimpleNamespace(verdict=_Verdict()),
        pending_level_tiles=deque(range(batch_size)),
        pending_level_sources=set(),
        level_scan_remaining_tiles=0,
        level_evidence_inflight=False,
        level_evidence_generation=None,
        first_pass_quality="preview",
    )


def _fake_renderer(kernel, session, batch):
    calls = SimpleNamespace(
        scan_pending=0,
        reschedules=0,
        publishes=0,
        semantic=0,
        ui_exceptions=[],
    )
    renderer = SimpleNamespace(
        _frame_session=session,
        _frame_session_is_current=lambda current: current is session,
        _take_montage_level_evidence_batch=(
            lambda current, *, expected, require_refined, batch_limit: tuple(batch)
        ),
        _montage_level_expected_indices=lambda current: tuple(range(len(batch))),
        _schedule_semantic_level_evidence=(
            lambda current: setattr(calls, "semantic", calls.semantic + 1)
        ),
        _mark_montage_level_scan_pending=(
            lambda current: setattr(calls, "scan_pending", calls.scan_pending + 1)
        ),
        _schedule_montage_cached_level_stats=(
            lambda current: setattr(calls, "reschedules", calls.reschedules + 1)
        ),
        _maybe_publish_after_level_evidence=(
            lambda current, *, processed: setattr(calls, "publishes", calls.publishes + 1)
        ),
        _invite_montage_level_evidence_continuation=lambda current: None,
        _requeue_montage_level_evidence=lambda current, batch: None,
        _queue_montage_level_refinement=lambda current, rendered: None,
        _remember_montage_source_level_stats=lambda level_key, stats: None,
        _montage_level_tracker=lambda: SimpleNamespace(
            update_from_stats=lambda *a, **k: None,
            record_vacuous_source=lambda *a, **k: None,
            has_source_quality=lambda *a, **k: False,
            ensure_expected=lambda *a, **k: None,
        ),
        _montage_pending_level_tiles_last_session=0,
        _last_montage_level_stats_ms=0.0,
        win=SimpleNamespace(kernel=kernel),
    )
    return renderer, calls


def _rendered_tile(index: int):
    return SimpleNamespace(tile=SimpleNamespace(source_index=int(index)))


@pytest.fixture
def sweep_environment(monkeypatch):
    """A real kernel driving the real sweep scheduler over a slow sampler."""

    kernel = Kernel(ThreadWorkerBackend(workers=1, name="test-level-evidence-shutdown"))
    batch = tuple(_rendered_tile(index) for index in range(30))
    session = _fake_session(len(batch))
    renderer, calls = _fake_renderer(kernel, session, batch)

    first_sample_started = threading.Event()
    sampled = []

    def slow_sample(rendered, *, refined, evidence_quality=None):
        sampled.append(int(rendered.tile.source_index))
        first_sample_started.set()
        time.sleep(0.15)
        return None

    monkeypatch.setattr(level_stats, "_sample_rendered_level_evidence", slow_sample)
    monkeypatch.setattr(
        level_stats,
        "_rendered_level_evidence_quality_for_session",
        lambda current, rendered, *, refined: level_stats.LevelEvidenceQuality.ROUGH_PREVIEW,
    )
    monkeypatch.setattr(
        level_stats, "_montage_level_evidence_requires_refined", lambda window, current: False
    )
    monkeypatch.setattr(level_stats.montage_commit, "rendered_tile_nbytes", lambda rendered: 0)
    monkeypatch.setattr(level_stats, "_complete_inline_work", lambda renderer, item: None)
    monkeypatch.setattr(
        level_stats,
        "handle_ui_exception",
        lambda context, exc: calls.ui_exceptions.append((context, exc)),
    )

    try:
        yield kernel, renderer, session, calls, first_sample_started, sampled, batch
    finally:
        kernel.shutdown(timeout=6.0)


def _drain(kernel) -> list:
    outcomes = []
    while True:
        event = kernel.completions.pop()
        if event is None:
            return outcomes
        outcomes.append((event.spec.key, kernel.dispatch_event(event)))


def test_shutdown_during_level_evidence_batch_stops_at_tile_boundary(sweep_environment):
    kernel, renderer, session, _calls, first_sample_started, sampled, _batch = sweep_environment

    LevelStatsService._process_montage_cached_level_stats(renderer)
    assert session.level_evidence_inflight is True
    assert first_sample_started.wait(timeout=5.0)

    started = monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kernel.shutdown(timeout=1.0)
    elapsed = monotonic() - started

    # A cooperative sweep bails at the next tile boundary (~0.15 s), well
    # inside the join deadline. Waiting out the deadline means the current
    # evaluation never observed cancellation.
    assert elapsed < 0.6, f"shutdown join waited {elapsed:.3f}s for a running sweep"
    shutdown_warnings = [
        warning for warning in caught if issubclass(warning.category, RuntimeWarning)
    ]
    assert shutdown_warnings == [], (
        f"leaked-thread diagnostics on shutdown: {[str(w.message) for w in shutdown_warnings]}"
    )
    assert kernel.last_shutdown_diagnostics == ()
    # Cooperative stop, not a mid-item interrupt: the sweep quit at a tile
    # edge after at most a couple of samples, never the whole batch.
    assert 1 <= len(sampled) <= 3


def test_cancelled_level_evidence_batch_fires_stale_callback_exactly_once(sweep_environment):
    kernel, renderer, session, calls, first_sample_started, _sampled, _batch = sweep_environment

    LevelStatsService._process_montage_cached_level_stats(renderer)
    assert first_sample_started.wait(timeout=5.0)
    kernel.shutdown(timeout=1.0)

    # The join deadline guarantees the completion event is queued once the
    # worker exits; poll briefly in case the current item is still finishing.
    deadline = monotonic() + bounded_interaction_settle_timeout_s(6.0)
    outcomes = []
    while monotonic() < deadline:
        outcomes.extend(_drain(kernel))
        if outcomes:
            break
        time.sleep(0.02)

    assert len(outcomes) == 1
    # Cancellation completes the task through the existing stale machinery:
    # exactly one stale delivery, no done/error delivery, no new paths.
    assert calls.scan_pending == 1
    assert calls.publishes == 0
    assert calls.ui_exceptions == []
    assert session.level_evidence_inflight is False
    assert session.level_evidence_generation is None
    # Nothing replays the same completion later.
    assert _drain(kernel) == []
    assert calls.scan_pending == 1


def test_wgpu_histogram_resolve_observes_cancellation_between_rows():
    """The fence-wait/resolve loops stop at row boundaries once cancelled."""

    resolve_calls = []
    waits = []

    class _Token:
        cancelled = False

    token = _Token()

    def _row(index: int):
        def wait_completed():
            waits.append(index)

        def resolve():
            resolve_calls.append(index)
            # Cancel after the first row resolves: the second row must not
            # be resolved, and the loop must exit at the row edge.
            token.cancelled = True
            return (None, None)

        return SimpleNamespace(
            wait_completed=wait_completed,
            readback=SimpleNamespace(resolve=resolve),
            source_index=index,
            evidence_key=("evidence", index),
        )

    rows = (_row(0), _row(1))
    with pytest.raises(EvaluationCancelled):
        level_stats.resolve_wgpu_histogram_evidence(
            rows,
            quality=level_stats.LevelEvidenceQuality.ROUGH_PREVIEW,
            cancellation_token=token,
        )
    assert resolve_calls == [0]
