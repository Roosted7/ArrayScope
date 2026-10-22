"""Operation history restores semantic edits without retaining obsolete sources."""

from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass

import numpy as np
import pytest

from arrayscope.operations.coordinator import OperationCoordinator
from arrayscope.operations.input_slots import ResolvedSlot, SlotBinding
from arrayscope.operations.pipeline import Crop, OperationStep, ReverseAxis, Scale, SplitAxis


def test_every_operation_edit_restores_exact_steps_and_values_in_both_directions():
    source = np.arange(4 * 6, dtype=np.float64).reshape(4, 6)
    original = source.copy()
    coordinator = OperationCoordinator(source)
    states = [(coordinator.document.steps, source.copy())]

    def remember():
        states.append((coordinator.document.steps, coordinator.document.materialize().copy()))

    coordinator.append_created_operation(Crop(1, 1, 5))
    remember()
    coordinator.append_created_operation(ReverseAxis(0))
    remember()
    coordinator.move(1, -1)
    remember()
    coordinator.set_enabled(0, False)
    remember()
    coordinator.replace_operation(1, "crop", axis=1, parameters={"start": 2, "stop": 6})
    remember()
    coordinator.reorder((1, 0))
    remember()
    coordinator.delete(1)
    remember()
    coordinator.load_steps((OperationStep(Scale(3)),))
    remember()
    coordinator.clear()
    remember()

    for steps, values in reversed(states[:-1]):
        assert coordinator.can_undo
        coordinator.undo()
        assert coordinator.document.steps == steps
        np.testing.assert_array_equal(coordinator.document.materialize(), values)
        assert coordinator.document.revision == 0, "ordinary edits must not change the source epoch"
    assert not coordinator.can_undo
    for steps, values in states[1:]:
        assert coordinator.can_redo
        coordinator.redo()
        assert coordinator.document.steps == steps
        np.testing.assert_array_equal(coordinator.document.materialize(), values)
    assert not coordinator.can_redo
    np.testing.assert_array_equal(source, original)


def test_parameter_undo_restores_the_previous_parameter_instead_of_removing_the_operation():
    coordinator = OperationCoordinator(np.arange(20).reshape(4, 5))
    coordinator.append_created_operation(Crop(1, 0, 4))
    previous = coordinator.document.steps
    coordinator.replace_operation(0, "crop", axis=1, parameters={"start": 1, "stop": 3})

    coordinator.undo()

    assert coordinator.document.steps == previous
    assert coordinator.shape == (4, 4)


def test_noops_and_rejected_edits_preserve_redo_and_runtime_step_ids():
    coordinator = OperationCoordinator(np.arange(12).reshape(3, 4))
    coordinator.append_created_operation(Scale(2))
    steps = coordinator.document.steps
    coordinator.append_created_operation(ReverseAxis(0))
    coordinator.undo()
    coordinator.set_enabled(0, True)
    coordinator.move(0, -1)
    coordinator.reorder((0,))
    coordinator.load_steps((OperationStep(Scale(2)),))
    assert coordinator.document.steps == steps
    assert coordinator.can_redo

    with pytest.raises(ValueError, match="exactly once"):
        coordinator.reorder((0, 0))
    assert coordinator.document.steps == steps
    assert coordinator.can_redo
    coordinator.append_created_operation(Crop(1, 1, 3))
    assert not coordinator.can_redo


def test_history_keeps_at_most_64_changes_and_initial_steps_are_a_baseline():
    coordinator = OperationCoordinator(np.ones((2, 2)), operations=(Scale(1),))
    assert not coordinator.can_undo
    for factor in range(2, 72):
        coordinator.replace_operation(0, "scale", parameters={"factor": factor})
    undone = 0
    while coordinator.can_undo:
        coordinator.undo()
        undone += 1
    assert undone == 64
    assert coordinator.document.enabled_operations == (Scale(7),)


@pytest.mark.parametrize("replacement", ["replace", "reload", "notify", "materialize"])
def test_source_content_changes_reset_both_history_directions(replacement):
    coordinator = OperationCoordinator(np.arange(12).reshape(3, 4))
    coordinator.append_created_operation(Scale(2))
    coordinator.append_created_operation(ReverseAxis(0))
    coordinator.undo()
    previous_epoch = coordinator.document.revision
    if replacement == "replace":
        coordinator.replace_base_data(np.ones((3, 4)))
    elif replacement == "reload":
        coordinator.reload_base_data(np.ones((3, 4)))
    elif replacement == "notify":
        coordinator.mark_base_data_changed()
    else:
        coordinator.materialize()
    assert coordinator.document.revision > previous_epoch
    assert not coordinator.can_undo
    assert not coordinator.can_redo


def test_replacing_the_primary_source_releases_history_without_copying_source_pixels():
    source = np.ones((3, 4))
    reference = weakref.ref(source)
    coordinator = OperationCoordinator(source)
    coordinator.append_created_operation(Scale(2))
    coordinator.append_created_operation(ReverseAxis(0))
    coordinator.undo()
    coordinator.replace_base_data(np.zeros((2, 2)))
    del source
    gc.collect()
    assert reference() is None


def test_new_edits_retire_stale_source_reset_explanations():
    coordinator = OperationCoordinator(np.ones((3, 4)))
    coordinator.mark_base_data_changed()
    assert coordinator.history_reset_reason
    coordinator.append_created_operation(ReverseAxis(0))
    assert coordinator.can_undo
    assert not coordinator.can_redo
    assert not coordinator.history_reset_reason


@dataclass(frozen=True)
class _BoundAdd:
    resolved_slots: tuple
    value: float = 1.0

    def apply(self, data):
        return data + self.value

    def output_shape(self, shape):
        return tuple(shape)

    def output_dtype(self, dtype):
        return dtype


def _bound(source, identity):
    slot = ResolvedSlot(
        SlotBinding("open-document", source_id=str(identity)),
        shape=(1,),
        dtype=np.dtype(float).str,
        source_identity=identity,
        source=source,
    )
    return _BoundAdd((("aux", slot),))


def test_multi_input_append_remains_undoable_and_small_input_redo_is_preserved():
    coordinator = OperationCoordinator(np.ones((2, 2)))
    operation = _bound(np.ones(8), "small-input")
    coordinator.append_created_operation(operation)
    assert coordinator.can_undo
    coordinator.undo()
    assert coordinator.document.steps == ()
    assert coordinator.can_redo
    coordinator.redo()
    assert coordinator.document.enabled_operations == (operation,)


def test_historical_input_budget_counts_shared_array_owners_once(monkeypatch):
    from arrayscope.operations import coordinator as module

    source = np.ones(64)
    monkeypatch.setattr(module, "HISTORY_EXTRA_INPUT_BUDGET_BYTES", source.nbytes)
    coordinator = OperationCoordinator(np.zeros((2, 2)))
    coordinator.append_created_operation(_bound(source[:4], "first-view"))
    coordinator.append_created_operation(_bound(source[4:8], "second-view"))
    coordinator.undo()
    coordinator.undo()

    assert coordinator.can_redo
    coordinator.redo()
    assert coordinator.can_redo
    coordinator.redo()
    assert len(coordinator.document.steps) == 2


def test_undo_releases_a_large_auxiliary_source_when_redo_would_exceed_the_budget(monkeypatch):
    from arrayscope.operations import coordinator as module

    auxiliary = np.ones(32)
    reference = weakref.ref(auxiliary)
    monkeypatch.setattr(module, "HISTORY_EXTRA_INPUT_BUDGET_BYTES", auxiliary.nbytes - 1)
    coordinator = OperationCoordinator(np.zeros((2, 2)))
    coordinator.append_created_operation(_bound(auxiliary, "large-input"))
    coordinator.undo()
    del auxiliary
    gc.collect()

    assert coordinator.document.steps == ()
    assert not coordinator.can_redo
    assert "memory" in coordinator.history_reset_reason.lower()
    assert reference() is None


def test_small_array_view_cannot_hide_a_large_retained_buffer_owner(monkeypatch):
    from arrayscope.operations import coordinator as module

    backing = bytearray(1000)
    source = np.ndarray((1,), dtype=np.float64, buffer=memoryview(backing))
    monkeypatch.setattr(module, "HISTORY_EXTRA_INPUT_BUDGET_BYTES", 999)
    coordinator = OperationCoordinator(np.zeros((2, 2)))
    coordinator.append_created_operation(_bound(source, "buffer-view"))
    coordinator.undo()
    assert not coordinator.can_redo
    assert "memory" in coordinator.history_reset_reason.lower()


def test_unknown_size_auxiliary_history_is_pruned_without_reading_lazy_pixels():
    class LazySource:
        def __array__(self, *args, **kwargs):
            pytest.fail("history accounting tried to read a lazy source")

    coordinator = OperationCoordinator(np.zeros((2, 2)))
    coordinator.append_created_operation(_bound(LazySource(), "unknown-size"))
    assert coordinator.can_undo
    coordinator.undo()
    assert not coordinator.can_redo


def test_rank_increasing_branch_copies_history_and_records_the_new_step_in_the_successor():
    source = np.arange(24).reshape(4, 6)
    parent = OperationCoordinator(source)
    parent.mark_base_data_changed()
    parent.append_created_operation(Crop(1, 0, 4))
    parent.append_created_operation(ReverseAxis(0))
    parent.undo()
    previous = parent.document.steps
    child = OperationCoordinator(source, steps=(*previous, OperationStep(SplitAxis(1, 2))))

    child.inherit_history_from(parent, description="Add operation")

    assert parent.document.steps == previous
    assert parent.can_redo
    assert child.can_undo
    assert not child.can_redo
    assert child.document.revision == parent.document.revision == 1
    child.undo()
    assert child.document.steps == previous
    child.redo()
    assert child.shape == (4, 2, 2)
    np.testing.assert_array_equal(child.document.materialize(), source[:, :4].reshape(4, 2, 2))
    other = OperationCoordinator(source.copy(), steps=child.document.steps)
    with pytest.raises(ValueError, match="source"):
        other.inherit_history_from(child)
