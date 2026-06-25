from arrayscope.display.model.tile_priority import MontageTilePriorityQueue, TilePriorityContext


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


def test_priority_queue_aging_completes_distant_visible_tiles():
    plan = _plan(count=12, columns=12)
    queue = MontageTilePriorityQueue(
        plan.tiles,
        context=_context(
            focus=(126.0, 5.0),
            visible=range(12),
            view_range=((0.0, 132.0), (0.0, 10.0)),
        ),
        aging_after=2,
    )
    popped = [queue.pop().montage_index for _ in range(3)]

    assert 0 in popped
    assert queue.fairness_pops >= 1


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
