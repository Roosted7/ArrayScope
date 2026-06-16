from types import SimpleNamespace

import numpy as np

from arrayscope.core.compute_policy import ComputeLane, EvaluationContext
from arrayscope.core.view_state import ViewState
from arrayscope.operations.coordinator import OperationCoordinator
from arrayscope.operations.pipeline import CenteredFFT
from arrayscope.window.stage_warmup import schedule_stage_warmup


class _Controller:
    def __init__(self, busy=False):
        self.busy = bool(busy)
        self.calls = []

    def is_busy(self):
        return self.busy

    def start_latest(self, fn, **kwargs):
        self.calls.append((fn, kwargs))
        return len(self.calls)


class _Window:
    def __init__(self, *, stage_budget=1024 * 1024, visible_busy=False, montage_busy=False):
        data = np.zeros((8, 8, 8), dtype=np.float32)
        coordinator = OperationCoordinator(data, operations=(CenteredFFT(axis=2),))
        self.document = coordinator.document
        self.operation_evaluator = coordinator.evaluator
        self.visible_evaluation_controller = _Controller(visible_busy)
        self.montage_tile_evaluation_controller = _Controller(montage_busy)
        self.stage_evaluation_controller = _Controller()
        self.render_coordinator = SimpleNamespace(has_pending_render=False)
        self._policy = SimpleNamespace(stage_cache_budget_bytes=stage_budget)
        self._generation = 1

    def _memory_policy(self):
        return self._policy

    def _capture_render_generation(self):
        return self._generation

    def _is_current_render_generation(self, generation):
        return int(generation) == int(self._generation)

    def _evaluation_context(self, lane, token=None):
        return EvaluationContext(ComputeLane(lane), token, 2, self._policy)


def test_stage_warmup_schedules_fitting_fft_candidate():
    window = _Window()
    view_state = ViewState.from_shape(window.document.current_shape)

    decision = schedule_stage_warmup(window, view_state)

    assert decision.decision == "scheduled"
    assert len(window.stage_evaluation_controller.calls) == 1


def test_stage_warmup_refuses_candidate_over_stage_budget():
    window = _Window(stage_budget=1)
    view_state = ViewState.from_shape(window.document.current_shape)

    decision = schedule_stage_warmup(window, view_state)

    assert decision.decision == "blocked_budget"
    assert window.stage_evaluation_controller.calls == []


def test_stage_warmup_attaches_to_in_flight_stage():
    window = _Window()
    view_state = ViewState.from_shape(window.document.current_shape)

    first = schedule_stage_warmup(window, view_state)
    second = schedule_stage_warmup(window, view_state)

    assert first.decision == "scheduled"
    assert second.decision == "in_flight"
    assert len(window.stage_evaluation_controller.calls) == 1


def test_stage_warmup_blocks_while_visible_or_montage_work_is_busy():
    view_state = ViewState.from_shape((8, 8, 8))

    assert schedule_stage_warmup(_Window(visible_busy=True), view_state).decision == "blocked_idle"
    assert schedule_stage_warmup(_Window(montage_busy=True), view_state).decision == "blocked_idle"
