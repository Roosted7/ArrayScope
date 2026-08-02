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
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.display.montage import MontagePlan, MontageTile
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
