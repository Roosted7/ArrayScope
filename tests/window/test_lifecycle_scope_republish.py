"""A camera move syncs the lifecycle scope twice; only one of them is work.

``retarget_viewport`` publishes the new targets (and the *previous* frame
plan's required scope); the controller syncs again after the planner installs
the new ``frame_plan``, because ``required_tile_numbers`` reads
``frame_plan.active_region_ids`` and nothing before that point can know it.
Neither call is removable.  What was duplicated is the target rebuild — one
typed identity per visible tile — which the second call now adopts from the
first when the inputs it was built from have not moved.

These tests pin both halves: that the second call still republishes the new
required scope, and that the guard which lets it skip the rebuild declines the
moment any input to a target actually changes.

The second group pins what that republication does to the COVERAGE -> REFINE
phase owner.  A pan over a montage that was never uncovered used to open
COVERAGE, only for the very next verdict -- inside the same
``retarget_frame_pipeline`` call -- to close it again and fire the replan
wakeup, buying a second full retarget for a scope that had been complete the
whole time (trace 5 phase 4: ``montage_retarget`` 47.9 % of wall clock, 95 % of
those calls submitting nothing).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.trace import TRACE, close_trace, configure_trace
from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationState
from arrayscope.display.montage import MontagePlan, MontageTile
from arrayscope.render.progressive_scheduling import SchedulingPhase
from arrayscope.window.frame_session import FrameSession

TILE = 32
COUNT = 6
VIEWPORT = (TILE, 3 * TILE)


def _tiles(count=COUNT):
    return tuple(
        MontageTile(
            montage_index=index,
            source_index=index,
            row=0,
            col=index,
            x0=index * TILE,
            y0=0,
            width=TILE,
            height=TILE,
            view_state=None,
        )
        for index in range(count)
    )


def _plan(tiles):
    return MontagePlan(
        axis=0,
        tile_shape=(TILE, TILE),
        grid_shape=(1, len(tiles)),
        columns=len(tiles),
        rows=1,
        gap=0,
        tiles=tiles,
    )


@pytest.fixture
def session() -> FrameSession:
    tiles = _tiles()
    plan = _plan(tiles)
    return FrameSession(
        session_id=1,
        key=("scope-republish",),
        render_generation=1,
        level_key=None,
        level_expected_indices=tuple(range(COUNT)),
        plan=plan,
        view_state=None,
        document=SimpleNamespace(base_data=np.zeros(4), revision=1, steps=()),
        montage_axis=0,
        colormap_lut=None,
        viewport_shape=VIEWPORT,
        view_range=((0.0, 3.0 * TILE), (0.0, float(TILE))),
        output_dtype=np.dtype("float32"),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=tiles[:3],
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
    )


def _counting(session, name):
    calls: list[int] = []
    original = getattr(session, name)

    def wrapper(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    setattr(session, name, wrapper)
    return calls


def test_second_sync_publishes_the_new_frame_plan_scope_without_rebuilding_targets(session):
    """The controller's second sync is the only one that sees the new plan."""

    session.frame_plan = SimpleNamespace(active_region_ids=(0, 1, 2))
    session.sync_lifecycle_scope()
    assert tuple(session.scheduling_policy.verdict.required_tiles) == (0, 1, 2)

    builds = _counting(session, "_sync_lifecycle_targets")

    # A pan: retarget_viewport publishes targets and syncs once (call one),
    # still against the previous plan's required scope.
    session.retarget_viewport(
        view_range=((3.0 * TILE, 6.0 * TILE), (0.0, float(TILE))),
        viewport_shape=VIEWPORT,
    )
    assert len(builds) == 1
    stale_scope = tuple(session.scheduling_policy.verdict.required_tiles)
    panned_to = tuple(sorted(session.visible_tile_numbers))
    assert stale_scope != panned_to

    # The planner installs the new plan; the controller syncs again.  This is
    # the call that carries the new required scope, and there is no earlier
    # point at which it could have been known.
    session.frame_plan = SimpleNamespace(active_region_ids=panned_to)
    session.sync_lifecycle_scope()

    assert tuple(session.scheduling_policy.verdict.required_tiles) == panned_to
    # No second typed-identity rebuild: the published set was still current.
    assert len(builds) == 1


def test_second_sync_rebuilds_when_a_target_input_moved(session):
    """Adoption is guarded, not assumed."""

    session.frame_plan = SimpleNamespace(active_region_ids=(0, 1, 2))
    session.retarget_viewport(
        view_range=((0.0, 3.0 * TILE), (0.0, float(TILE))),
        viewport_shape=VIEWPORT,
    )
    builds = _counting(session, "_sync_lifecycle_targets")

    session.skipped_tiles.add(1)
    session.sync_lifecycle_scope()

    assert len(builds) == 1


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda s: setattr(s, "semantic_key", ("moved",)), id="semantic_key"),
        pytest.param(lambda s: setattr(s, "key", ("moved",)), id="key"),
        pytest.param(lambda s: setattr(s, "output_dtype", np.dtype("float64")), id="output_dtype"),
        pytest.param(lambda s: setattr(s, "shader_display", True), id="shader_display"),
        pytest.param(
            lambda s: setattr(s, "canonical_orientation", True), id="canonical_orientation"
        ),
        pytest.param(
            lambda s: setattr(s.document, "revision", 99),
            id="document_revision",
        ),
        pytest.param(
            lambda s: setattr(s.document, "steps", ("fft",)),
            id="document_steps",
        ),
        pytest.param(
            lambda s: setattr(s.document, "base_data", np.zeros(4)),
            id="document_base_data",
        ),
        pytest.param(
            lambda s: setattr(
                s, "document", SimpleNamespace(base_data=np.zeros(4), revision=1, steps=())
            ),
            id="document",
        ),
        pytest.param(lambda s: s.skipped_tiles.add(1), id="skipped_tiles"),
        pytest.param(
            # A rebuilt-but-equal tuple declines: the check is identity, so
            # "unsure" always falls back to a recompute.
            lambda s: setattr(s, "visible_tiles", tuple(tile for tile in s.visible_tiles)),
            id="visible_tiles",
        ),
    ],
)
def test_every_target_input_invalidates_the_published_set(session, mutate):
    """Each value a target is built from must be able to decline the guard.

    The snapshot used to carry only the three values a caller mid-method could
    move, and leaned on the caller's own control flow for the rest.  Adopting
    a published set across a planner call means the snapshot has to stand on
    its own, so every input gets a case here — including ``base_data``, which
    is held by object rather than by ``id()`` precisely so a recycled address
    cannot age a stale snapshot into a false positive.
    """

    session.retarget_viewport(
        view_range=((0.0, 3.0 * TILE), (0.0, float(TILE))),
        viewport_shape=VIEWPORT,
    )
    snapshot = session._lifecycle_target_inputs()
    assert session._lifecycle_targets_still_current(snapshot)

    mutate(session)

    assert not session._lifecycle_targets_still_current(snapshot)


def test_required_scope_is_derived_once_per_set_of_inputs(session):
    """The one scope every commit reads ~27 times is derived when it moves.

    Each answer rebuilt two montage-wide sets and a sorted tuple. All three
    inputs -- the frame plan, the visible set, and the skip index -- are
    replaced or revised wholesale, so the latch has an exact edge for each and
    no way to keep a stale answer past one of them.
    """

    session.visible_tile_numbers = frozenset(range(COUNT))
    first = session.required_tile_numbers()
    assert first == tuple(range(COUNT))
    assert session.required_tile_numbers() is first

    session.visible_tile_numbers = frozenset(range(COUNT - 1))
    moved = session.required_tile_numbers()
    assert moved is not first
    assert moved == tuple(range(COUNT - 1))

    session.lifecycle.plan_applied([0])
    session.lifecycle.tile_skipped(0)
    skipped = session.required_tile_numbers()
    assert skipped == tuple(range(1, COUNT - 1)), "a newly skipped tile leaves the scope"

    session.lifecycle.tile_unskipped(0)
    assert session.required_tile_numbers() == tuple(range(COUNT - 1))

    session.frame_plan = SimpleNamespace(active_region_ids=(3, 4))
    assert session.required_tile_numbers() == (3, 4), "the frame plan owns the scope when it exists"


# --- The COVERAGE -> REFINE phase this republication opens ----------------


@pytest.fixture
def progressive_session() -> FrameSession:
    """The same montage, with the whole grid visible and progressive on.

    ``sync_lifecycle_scope`` only opens COVERAGE at all when the session is
    progressive (``shader_display`` or a resident-LOD policy), so the phase
    tests below would be vacuous on the default fixture.
    """

    tiles = _tiles()
    plan = _plan(tiles)
    return FrameSession(
        session_id=1,
        key=("scope-republish",),
        render_generation=1,
        level_key=None,
        level_expected_indices=tuple(range(COUNT)),
        plan=plan,
        view_state=None,
        document=SimpleNamespace(base_data=np.zeros(4), revision=1, steps=()),
        montage_axis=0,
        colormap_lut=None,
        viewport_shape=VIEWPORT,
        view_range=((0.0, 3.0 * TILE), (0.0, float(TILE))),
        output_dtype=np.dtype("float32"),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        shader_display=True,
    )


def _present(session, tile_numbers):
    """Drive one tile through the whole physical presentation crossing.

    Both coverage owners read different halves of this: the lifecycle wants a
    first pixel, and the session's first-pass predicate additionally wants the
    display payload, the backend acknowledgement identity and the plan's
    source index to agree.
    """

    payloads = dict(getattr(session.tile_presentation_state, "payloads", {}) or {})
    numbers = tuple(int(tile) for tile in tile_numbers)
    for index in numbers:
        tile = session.plan.tiles[index]
        source_index = int(tile.source_index)
        image = np.ones((TILE, TILE), dtype=np.float32)
        payload = DisplayTilePayload(
            tile_number=index,
            source_index=source_index,
            image=image,
            histogram_data=image,
            source_id=("tile", source_index, "exact"),
        )
        session.display_tile_payloads[index] = payload
        session.record_tile_payload(payload)
        payloads[index] = payload
        session.lifecycle.commit_emitted({index: payload})
        session.lifecycle.backend_ack({index: payload})
        session.lifecycle.acknowledge_presented(index, payload.source_id, payload.quality, 0)
    session.tile_presentation_state = TilePresentationState(payloads)
    session.lifecycle.presentation_confirmed(numbers)


def _settle(session, *, scope=(0, 1, 2), latch="exact", presented=None):
    """Bring the montage to "first pass complete over ``scope``"."""

    session.frame_plan = SimpleNamespace(active_region_ids=tuple(scope))
    session.sync_lifecycle_scope()
    if latch is not None:
        session.first_pass_quality = latch
    _present(session, tuple(range(COUNT)) if presented is None else presented)
    return session


def _pan_to(session, scope):
    """The controller's post-plan sync: a new required scope, same targets."""

    session.frame_plan = SimpleNamespace(active_region_ids=tuple(scope))
    generation_before = session.scheduling_policy.verdict.generation
    session.sync_lifecycle_scope()
    assert session.scheduling_policy.verdict.generation == generation_before + 1, (
        "the pan must actually adopt a new scope generation, or the test proves nothing"
    )
    return session.scheduling_policy.verdict


def test_a_pan_over_a_first_pass_complete_montage_does_not_reopen_coverage(progressive_session):
    """The trace-5 phase-4 defect: coverage opened and closed within one call.

    Every tile of the panned-to scope is already presented and acknowledged at
    the latched first-pass quality, so both observe owners would have closed
    COVERAGE on their first look.  Opening it would have cost a
    ``coverage_closed`` edge, its ``on_refinement_replan`` wakeup, and the
    second full retarget that wakeup buys.
    """

    session = _settle(progressive_session)
    assert session.first_pass_pixels_presented(), "precondition: the first pass is complete"

    configure_trace(None, ring_events=512)
    try:
        verdict = _pan_to(session, (2, 3, 4))
        events = TRACE.snapshot()
    finally:
        close_trace()

    assert verdict.required_tiles == (2, 3, 4)
    assert verdict.coverage_open is False
    assert verdict.phase is SchedulingPhase.REFINE
    started = [
        event
        for event in events
        if event.get("kind") == "scheduling_phase" and event.get("event") == "scope_started"
    ]
    assert [event["generation"] for event in started] == [verdict.generation]
    assert started[0]["already_covered"] is True
    assert not [
        event
        for event in events
        if event.get("kind") == "scheduling_phase"
        and event.get("event") == "coverage_closed"
        and event.get("generation") == verdict.generation
    ], "a scope that never opened coverage must not close it"


def test_a_pan_that_reveals_an_unpresented_tile_still_opens_coverage(progressive_session):
    """The negative arm: one missing tile is the whole point of the phase."""

    session = _settle(progressive_session, presented=range(COUNT - 1))

    verdict = _pan_to(session, (3, 4, COUNT - 1))

    assert verdict.required_tiles == (3, 4, COUNT - 1)
    assert verdict.coverage_open is True
    assert verdict.phase is SchedulingPhase.COVERAGE


def test_the_predicate_is_the_conjunction_not_the_lifecycle_half(progressive_session):
    """Lifecycle first pixels alone may never stand in for the first pass.

    ``frame_effects.scheduling_verdict`` observes with the lifecycle and the
    commit path observes with the session, and either may be the first to see
    a generation.  Shortcutting on the cheap half alone would open REFINE on a
    scope the session-side owner would have held in COVERAGE -- with no wakeup
    left to correct it, because the montage watchdog only emits evidence and
    never mutates scheduling state.  R2/R4 depend on this staying a
    conjunction.
    """

    session = _settle(progressive_session, latch=None)
    assert session.lifecycle.first_pixels_presented((2, 3, 4)), "the cheap half holds"
    assert session.first_pass_quality is None, "the session-side half does not"

    verdict = _pan_to(session, (2, 3, 4))

    assert verdict.coverage_open is True
    assert verdict.phase is SchedulingPhase.COVERAGE


def test_a_payload_disagreeing_with_the_plan_source_still_opens_coverage(progressive_session):
    """The session-side half is per-tile agreement, not a session-wide flag.

    The lifecycle has its first pixel for every tile in the panned-to scope
    and the first pass is latched, yet one displayed payload no longer names
    the plan's source for its slot.  That tile's pixels are not the first
    pass's pixels, so coverage is still owed.
    """

    session = _settle(progressive_session)
    displaced = dataclasses.replace(
        session.display_tile_payloads[3],
        source_index=int(session.plan.tiles[3].source_index) + 100,
    )
    session.display_tile_payloads[3] = displaced

    assert session.lifecycle.first_pixels_presented((2, 3, 4)), "the cheap half still holds"
    assert session.first_pass_quality == "exact"
    assert not session.first_pass_pixels_presented_for((2, 3, 4))

    verdict = _pan_to(session, (2, 3, 4))

    assert verdict.coverage_open is True
    assert verdict.phase is SchedulingPhase.COVERAGE


def test_a_freshly_constructed_session_opens_coverage(progressive_session):
    """Inert on a cold fill: there is no latched first pass to be complete."""

    assert progressive_session.first_pass_quality is None
    progressive_session.frame_plan = SimpleNamespace(active_region_ids=(0, 1, 2))

    progressive_session.sync_lifecycle_scope()

    verdict = progressive_session.scheduling_policy.verdict
    assert verdict.required_tiles == (0, 1, 2)
    assert verdict.phase is SchedulingPhase.COVERAGE
    assert progressive_session.first_pass_pixels_presented_for((0, 1, 2)) is False


def test_an_empty_scope_is_not_vacuously_covered(progressive_session):
    """``lifecycle.first_pixels_presented(())`` is True; the conjunction is not."""

    session = _settle(progressive_session)

    assert session.lifecycle.first_pixels_presented(()) is True
    assert session.first_pass_pixels_presented_for(()) is False


def test_an_index_window_retarget_still_opens_coverage(progressive_session):
    """Inert on a session rebirth's cheaper cousin.

    ``retarget_index_window`` resets ``first_pass_quality`` to ``None`` before
    it syncs the scope, exactly so the new window has to prove its own first
    pass.  The shortcut reads that same field, so it cannot survive the reset —
    this is the property that keeps R2/R4 intact across an index scrub.
    """

    session = _settle(progressive_session)
    assert session.first_pass_pixels_presented()

    shifted = _plan(
        tuple(
            dataclasses.replace(tile, source_index=int(tile.source_index) + COUNT)
            for tile in _tiles()
        )
    )
    session.retarget_index_window(
        session_id=session.session_id + 1,
        key=("scope-republish", "retargeted"),
        semantic_key=("semantic", "retargeted"),
        level_key=None,
        render_generation=session.render_generation + 1,
        view_state=None,
        source_anchoring=session.source_anchoring,
        plan=shifted,
        frame_plan=SimpleNamespace(active_region_ids=(0, 1, 2)),
        all_indices=tuple(int(tile.source_index) for tile in shifted.tiles),
        new_source_ids={
            int(tile.montage_index): ("source", int(tile.source_index)) for tile in shifted.tiles
        },
        cached_tiles={},
        visible_tiles=tuple(shifted.tiles),
    )

    assert session.first_pass_quality is None
    verdict = session.scheduling_policy.verdict
    assert verdict.required_tiles == (0, 1, 2)
    assert verdict.phase is SchedulingPhase.COVERAGE


def test_one_pan_over_a_settled_montage_costs_one_planner_pass_not_two(progressive_session):
    """The counting arm: the self-generated retarget pair is gone.

    ``retarget_frame_pipeline`` syncs the lifecycle scope and then, inside
    ``FramePipeline.retarget``, reads ``effects.scheduling_verdict()`` -- the
    observe site that closes coverage and fires ``on_refinement_replan``.  That
    wakeup arms the 1 ms replan gate, which re-enters the planner.  The driver
    below is those two statements plus the gate's at-most-one-per-turn
    coalescing; everything it touches (the session, the phase owner, the
    verdict site) is the product's own.

    Before the fix this montage cost two planner passes for one pan: the
    second one is the whole 47.9 % of trace 5 phase 4 that submitted nothing.
    """

    from arrayscope.window.frame_effects import FramePipelineEffects

    session = _settle(progressive_session)

    entries: list[tuple[int, ...]] = []
    replans: list[object] = []
    renderer = SimpleNamespace(
        win=None,
        _frame_session_is_current=lambda candidate: candidate is session,
        request_montage_replan=replans.append,
    )
    effects = FramePipelineEffects(renderer, session)

    def planner_pass(scope):
        entries.append(tuple(scope))
        session.frame_plan = SimpleNamespace(active_region_ids=tuple(scope))
        session.sync_lifecycle_scope()
        return effects.scheduling_verdict()

    planner_pass((2, 3, 4))
    while replans:
        replans.clear()
        planner_pass((2, 3, 4))

    assert entries == [(2, 3, 4)]
    assert session.scheduling_policy.verdict.phase is SchedulingPhase.REFINE
