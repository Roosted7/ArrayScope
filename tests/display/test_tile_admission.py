from arrayscope.display.model.tile_admission import TileAdmissionQueue, TilePriorityContext


def test_tile_admission_respects_item_cap_and_retained_active():
    queue = TileAdmissionQueue(TilePriorityContext.from_tiles(visible_tiles=(0, 1, 2)))

    decision = queue.admit((0, 1, 2), retained=(9,), max_items=1)

    assert decision.admitted == (0,)
    assert decision.deferred == (1, 2)
    assert decision.active == (9, 0)


def test_tile_admission_respects_byte_cap_after_first_item():
    queue = TileAdmissionQueue(TilePriorityContext.from_tiles(visible_tiles=(0, 1, 2)))

    decision = queue.admit((0, 1, 2), cost_fn=lambda tile: 10, max_bytes=15)

    assert decision.admitted == (0,)
    assert decision.deferred == (1, 2)
    assert decision.admitted_bytes == 10


def test_tile_admission_orders_visible_before_near():
    queue = TileAdmissionQueue(
        TilePriorityContext.from_tiles(
            visible_tiles=(2,),
            near_tiles=(1,),
        )
    )

    assert queue.admit((0, 1, 2)).admitted[:2] == (2, 1)


def test_tile_admission_deadline_keeps_first_progress(monkeypatch):
    times = iter((0.0, 0.01, 0.02))
    monkeypatch.setattr("arrayscope.display.model.tile_admission.perf_counter", lambda: next(times))
    queue = TileAdmissionQueue(TilePriorityContext.from_tiles(visible_tiles=(0, 1, 2)))

    decision = queue.admit((0, 1, 2), deadline_ms=1.0)

    assert decision.admitted == (0,)
    assert decision.deferred == (1, 2)


def test_tile_admission_bounded_progress_preserves_deferred_order():
    queue = TileAdmissionQueue(TilePriorityContext.from_tiles(visible_tiles=(0, 1, 2, 3)))

    decision = queue.admit((0, 1, 2, 3), max_items=2)

    assert decision.admitted == (0, 1)
    assert decision.deferred == (2, 3)
    assert decision.active == (0, 1)
