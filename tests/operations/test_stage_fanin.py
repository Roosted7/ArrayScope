from arrayscope.operations.stage_fanin import StageFanInState


def test_stage_fanin_records_stage_value_and_clears_request_sets():
    state = StageFanInState(
        active_requests={"stage"},
        attached_requests={"stage"},
        tile_stage_keys={0: "stage", 1: "stage"},
    )

    batch = state.activate_value("stage", object(), max_items=1)

    assert batch.tiles == (0, 1)
    assert batch.complete is True
    assert state.active_requests == set()
    assert state.attached_requests == set()
    assert "stage" in state.values
    assert state.tile_stage_keys == {}


def test_stage_fanin_release_missing_clears_request_sets_and_waiting_tiles():
    state = StageFanInState(
        active_requests={"stage"},
        attached_requests={"stage"},
        tile_stage_keys={3: "stage"},
    )

    batch = state.release_missing("stage")

    assert batch.tiles == (3,)
    assert batch.complete is True
    assert state.active_requests == set()
    assert state.attached_requests == set()
    assert state.tile_stage_keys == {}


def test_stage_fanin_failure_returns_bound_tile_numbers_and_clears_requests():
    state = StageFanInState(
        active_requests={"stage"},
        attached_requests={"stage"},
        tile_stage_keys={3: "stage", 4: "other"},
    )

    blocked = state.fail("stage")

    assert blocked == (3,)
    assert state.active_requests == set()
    assert state.attached_requests == set()
    assert state.tile_stage_keys == {4: "other"}


def test_stage_fanin_merge_records_dependency_keys():
    state = StageFanInState()

    state.merge_plan(
        {
            "tile_stage_keys": {0: "stage"},
            "attached_stage_keys": {"stage"},
            "stage_values": {"cached": object()},
        }
    )

    assert state.tile_stage_keys == {0: "stage"}
    assert state.attached_requests == {"stage"}
    assert "cached" in state.values


def test_stage_fanin_detaches_requests_after_last_tile_binding_is_removed():
    state = StageFanInState(attached_requests={"stage"}, tile_stage_keys={0: "stage"})

    state.tile_stage_keys.pop(0)
    state.detach_unbound_requests()

    assert state.attached_requests == set()
