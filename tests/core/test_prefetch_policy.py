from arrayscope.core.prefetch_policy import SliceScrubMomentum, prefetch_deltas


def test_first_observation_plans_single_neighbor():
    momentum = SliceScrubMomentum()
    momentum.observe(5, now=0.0)

    plan = momentum.plan()

    assert plan.direction == 0
    assert plan.depth == 1
    assert plan.deltas == (-1, 1)


def test_single_step_plans_shallow_directional_warmup():
    momentum = SliceScrubMomentum()
    momentum.observe(5, now=0.0)
    momentum.observe(6, now=0.1)

    plan = momentum.plan()

    assert plan.direction == 1
    assert plan.depth == 2
    assert plan.deltas == (1, 2, -1)


def test_sustained_scrub_deepens_ahead_of_motion():
    momentum = SliceScrubMomentum()
    for step, index in enumerate((5, 6, 7, 8, 9)):
        momentum.observe(index, now=0.1 * step)

    plan = momentum.plan()

    assert plan.direction == 1
    assert plan.depth == 4
    assert plan.deltas == (1, 2, 3, 4, -1)


def test_direction_reversal_resets_streak():
    momentum = SliceScrubMomentum()
    for step, index in enumerate((5, 6, 7, 8)):
        momentum.observe(index, now=0.1 * step)
    momentum.observe(7, now=0.4)

    plan = momentum.plan()

    assert plan.direction == -1
    assert plan.depth == 2
    assert plan.deltas == (-1, -2, 1)


def test_pause_resets_streak():
    momentum = SliceScrubMomentum()
    for step, index in enumerate((5, 6, 7, 8)):
        momentum.observe(index, now=0.1 * step)
    momentum.observe(9, now=5.0)

    plan = momentum.plan()

    assert plan.direction == 1
    assert plan.depth == 2


def test_repeated_same_index_keeps_momentum():
    momentum = SliceScrubMomentum()
    momentum.observe(5, now=0.0)
    momentum.observe(6, now=0.1)
    momentum.observe(6, now=0.2)

    plan = momentum.plan()

    assert plan.direction == 1
    assert plan.depth == 2


def test_plan_depth_is_clamped_by_axis_size():
    momentum = SliceScrubMomentum()
    for step, index in enumerate((0, 1, 2, 3, 4)):
        momentum.observe(index, now=0.1 * step)

    plan = momentum.plan(size=3)

    assert plan.depth == 2


def test_prefetch_deltas_keep_one_reversal_guard():
    assert prefetch_deltas(1, 3) == (1, 2, 3, -1)
    assert prefetch_deltas(-1, 3) == (-1, -2, -3, 1)
    assert prefetch_deltas(0, 2) == (-1, 1, -2, 2)
