from types import SimpleNamespace

from arrayscope.operations.stage_fanin import StageFanInState


def _tile(index):
    return SimpleNamespace(montage_index=index)


def test_stage_fanin_activates_value_in_bounded_batches():
    state = StageFanInState(
        waiting_tiles={"stage": [_tile(0), _tile(1)]},
        active_requests={"stage"},
        attached_requests={"stage"},
    )

    first = state.activate_value("stage", object(), max_items=1)
    second = state.activate_value("stage", object(), max_items=1)

    assert [tile.montage_index for tile in first.tiles] == [0]
    assert first.complete is False
    assert [tile.montage_index for tile in second.tiles] == [1]
    assert second.complete is True
    assert state.waiting_tiles == {}
    assert state.active_requests == set()
    assert state.attached_requests == set()


def test_stage_fanin_release_missing_clears_tile_stage_keys():
    state = StageFanInState(
        waiting_tiles={"stage": [_tile(3)]},
        tile_stage_keys={3: "stage"},
    )

    batch = state.release_missing("stage")

    assert [tile.montage_index for tile in batch.tiles] == [3]
    assert state.tile_stage_keys == {}
    assert state.waiting_tiles == {}


def test_stage_fanin_failure_returns_waiting_tiles_and_clears_requests():
    state = StageFanInState(
        waiting_tiles={"stage": [_tile(3)]},
        active_requests={"stage"},
        attached_requests={"stage"},
        tile_stage_keys={3: "stage"},
    )

    waiting = state.fail("stage")

    assert [tile.montage_index for tile in waiting] == [3]
    assert state.waiting_tiles == {}
    assert state.active_requests == set()
    assert state.attached_requests == set()
    assert state.tile_stage_keys == {}


def test_stage_fanin_zero_item_batch_keeps_waiting_work():
    state = StageFanInState(waiting_tiles={"stage": [_tile(0), _tile(1)]})

    batch = state.activate_value("stage", object(), max_items=0)

    assert batch.tiles == ()
    assert batch.complete is False
    assert [tile.montage_index for tile in state.waiting_tiles["stage"]] == [0, 1]
