from arrayscope.display.model.tile_priority import MontageTilePriorityQueue, TilePriorityContext, prioritize_tile_numbers


def _plan(count=9, columns=3):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan

    state = ViewState.from_shape((10, 10, count)).with_montage_axis(2, indices=tuple(range(count)), text=":")
    return make_montage_plan(state, axis=2, indices=tuple(range(count)), tile_shape=(10, 10), columns=columns, gap=1)


def _context(*, focus=None, visible=(), near=(), priority=(), view_range=((0.0, 32.0), (0.0, 32.0))):
    return TilePriorityContext.from_tiles(
        view_range=view_range,
        focus=focus,
        visible_tiles=visible,
        near_tiles=near,
        priority_tiles=priority,
    )


def test_priority_queue_retargets_focus_without_reinserting_tiles():
    plan = _plan()
    queue = MontageTilePriorityQueue(
        plan.tiles,
        context=_context(focus=(16.0, 16.0), visible=range(9)),
    )

    assert queue.pop().montage_index == 4

    queue.set_context(_context(focus=(27.0, 27.0), visible=range(9), priority=(8,)), max_items=3)

    assert queue.pop().montage_index == 8


def test_priority_queue_explicit_focus_tile_precedes_nearer_visible_tiles():
    plan = _plan()
    queue = MontageTilePriorityQueue(
        plan.tiles,
        context=_context(
            focus=(16.0, 16.0),
            visible=range(9),
            priority=(8,),
        ),
    )

    assert queue.pop().montage_index == 8


def test_prioritize_tile_numbers_honors_explicit_focus_tile():
    plan = _plan()

    ordered = prioritize_tile_numbers(
        range(9),
        plan_tiles=plan.tiles,
        context=_context(
            focus=(16.0, 16.0),
            visible=range(9),
            priority=(8,),
        ),
    )

    assert ordered[0] == 8


def test_priority_queue_orders_visible_before_near_and_waiting():
    plan = _plan()
    queue = MontageTilePriorityQueue(
        plan.tiles,
        context=_context(
            focus=(27.0, 27.0),
            visible=(8,),
            near=(7,),
        ),
    )

    assert queue.pop().montage_index == 8
    assert queue.pop().montage_index == 7


def test_priority_queue_bulk_drain_stays_in_priority_order():
    # Regression: a "fairness aging" mechanism used to pop the oldest-inserted
    # tile after every few priority pops, so any bulk drain (stage fan-in
    # activation, per-commit upsert admission) degenerated to insertion order
    # after the first few items — the montage visibly filled from the sides.
    plan = _plan(count=12, columns=12)
    focus = (126.0, 5.0)
    queue = MontageTilePriorityQueue(
        plan.tiles,
        context=_context(
            focus=focus,
            visible=range(12),
            view_range=((0.0, 132.0), (0.0, 10.0)),
        ),
    )
    popped = [queue.pop().montage_index for _ in range(12)]

    def distance(index):
        tile = plan.tiles[index]
        center = (float(tile.x0) + float(tile.width) * 0.5, float(tile.y0) + float(tile.height) * 0.5)
        return (center[0] - focus[0]) ** 2 + (center[1] - focus[1]) ** 2

    assert popped == sorted(popped, key=distance)


def test_priority_queue_prunes_stale_entries_lazily():
    plan = _plan()
    queue = MontageTilePriorityQueue(
        plan.tiles,
        context=_context(focus=(16.0, 16.0), visible=range(9)),
    )

    removed = queue.prune({8})
    queue.set_context(_context(focus=(16.0, 16.0), visible=(8,)), max_items=1)

    assert removed == 8
    assert len(queue) == 1
    assert queue.pop().montage_index == 8
    assert not queue


def test_priority_queue_retarget_takes_full_effect_on_every_pop():
    # The architectural invariant behind montage fill order: pop() always
    # returns the best tile under the CURRENT context, regardless of which
    # context was live when each tile was pushed. An earlier design re-keyed
    # only a bounded batch per set_context, so with several actors
    # retargeting (hover, viewport restore, stage activation) the pop order
    # depended on push timing and the montage visibly filled from stale
    # anchors.
    plan = _plan(count=48, columns=12)
    stale = _context(
        focus=(6.0, 6.0),
        visible=range(48),
        view_range=((0.0, 132.0), (0.0, 44.0)),
    )
    queue = MontageTilePriorityQueue(plan.tiles, context=stale)

    focus = (126.0, 38.0)
    queue.set_context(
        _context(
            focus=focus,
            visible=range(48),
            view_range=((0.0, 132.0), (0.0, 44.0)),
        )
    )

    popped = []
    while queue:
        popped.append(queue.pop().montage_index)

    def distance(index):
        tile = plan.tiles[index]
        center = (float(tile.x0) + float(tile.width) * 0.5, float(tile.y0) + float(tile.height) * 0.5)
        return ((center[0] - focus[0]) / 132.0) ** 2 + ((center[1] - focus[1]) / 44.0) ** 2

    assert popped == sorted(popped, key=lambda index: (distance(index), index))
