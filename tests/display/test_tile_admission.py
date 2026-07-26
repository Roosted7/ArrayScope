from arrayscope.display.model.tile_admission import TileAdmissionQueue, TilePriorityContext


def test_tile_admission_respects_item_cap_and_retained_active():
    queue = TileAdmissionQueue(TilePriorityContext.from_tiles(visible_tiles=(0, 1, 2)))

    decision = queue.admit((0, 1, 2), retained=(9,), max_items=1)

    assert decision.admitted == (0,)
    assert decision.deferred == (1, 2)
    assert decision.active == (9, 0)


def test_tile_admission_item_free_entries_bypass_item_cap_but_pay_bytes():
    queue = TileAdmissionQueue(TilePriorityContext.from_tiles(visible_tiles=(0, 1, 2)))

    decision = queue.admit(
        (0, 1, 2),
        cost_fn=lambda tile: 10,
        item_free_fn=lambda tile: int(tile) in {0, 1},
        max_items=1,
        max_bytes=25,
    )

    assert decision.admitted == (0, 1)
    assert decision.deferred == (2,)
    assert decision.admitted_bytes == 20


def test_tile_admission_item_free_entries_respect_burst_cap():
    queue = TileAdmissionQueue(TilePriorityContext.from_tiles(visible_tiles=(0, 1, 2, 3)))

    decision = queue.admit(
        (0, 1, 2, 3),
        cost_fn=lambda tile: 1,
        item_free_fn=lambda _tile: True,
        max_item_free=2,
        max_items=1,
        max_bytes=10,
    )

    assert decision.admitted == (0, 1)
    assert decision.deferred == (2, 3)


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


def test_tile_admission_names_the_cap_that_bit_or_says_none_did():
    """A small batch means nothing until you know whether a cap caused it.

    ``max_upserts`` in the commit trace reports the cap that was in FORCE, not
    the cap that BIT, so a batch of four tiles looks identical whether four
    were all that existed or four were all that fit.  That ambiguity is why an
    intermittent supply-starved refinement (49 batches of 3-8 tiles over 8.0 s
    against the usual 2 batches over 0.28 s) reads as run-to-run variance
    instead of as a defect with a name — see
    docs/redesign/per-commit-transaction-count-2026-07-26.md §7.
    """

    queue = TileAdmissionQueue(TilePriorityContext.from_tiles(visible_tiles=(0, 1, 2, 3)))

    # Supply-bound: everything offered was admitted, so no cap can be blamed.
    supply = queue.admit((0, 1), max_items=8, max_bytes=1 << 20, cost_fn=lambda _tile: 16)
    assert supply.admitted == (0, 1)
    assert supply.deferred == ()
    assert supply.limit == ""

    # Item-bound.
    items = queue.admit((0, 1, 2, 3), max_items=2)
    assert items.deferred == (2, 3)
    assert items.limit == "items"

    # Byte-bound: the item cap is slack, the byte cap is what stops the walk.
    byte_bound = queue.admit((0, 1, 2, 3), max_items=99, max_bytes=100, cost_fn=lambda _tile: 60)
    assert byte_bound.admitted == (0,)
    assert byte_bound.limit == "bytes"

    # A free item bypasses every cap and therefore blames none of them.
    free = queue.admit((0, 1), max_items=0, free_fn=lambda _tile: True)
    assert free.admitted == (0, 1)
    assert free.limit == ""


def test_tile_admission_deadline_is_named_as_the_cap(monkeypatch):
    times = iter((0.0, 0.01, 0.02))
    monkeypatch.setattr("arrayscope.display.model.tile_admission.perf_counter", lambda: next(times))
    queue = TileAdmissionQueue(TilePriorityContext.from_tiles(visible_tiles=(0, 1, 2)))

    decision = queue.admit((0, 1, 2), deadline_ms=1.0)

    assert decision.admitted == (0,)
    assert decision.limit == "deadline"


def test_tile_admission_zero_cap_names_itself_without_walking():
    queue = TileAdmissionQueue(TilePriorityContext.from_tiles(visible_tiles=(0, 1)))

    assert queue.admit((0, 1), max_items=0).limit == "items"
    assert queue.admit((0, 1), max_bytes=0).limit == "bytes"
