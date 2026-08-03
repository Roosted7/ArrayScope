"""The presentation planner is pure; the applier owns every write.

ADR 0065 stage 2 slice D's prerequisite.  Slice C's three-arm bisect
(``docs/redesign/presentation-owner-stage2-2026-08-03.md``) put commit
corruption in the *builder*, not in the take, the validation or the mailbox:
``build_tile_presentation`` planned and mutated in one pass, so a transaction
that was prepared and then dropped — which a take-miss makes an ordinary
outcome — had already spent the session's dirty entries, retired payloads and
advanced the lifecycle.

Three properties make that unreachable, and this module owns all three:

1. **Planning writes nothing.**  Plan, drop, and the session fingerprint is
   byte-identical.  This is also the completeness check for
   ``_PRESENTATION_PLAN_CONTAINERS``: a container the planner mutates in place
   without a private copy shows up here as a fingerprint drift, and nowhere
   else.
2. **Planning twice is planning once.**  ``plan -> plan -> apply`` and
   ``plan -> drop -> plan -> apply`` both land exactly where ``plan -> apply``
   lands, in state and in delta.
3. **The applier is the only writer.**  Every family it installs matters: the
   mutation checks below drop one at a time and demand a red.

Property 1 is the one that could not even be *expressed* against the old
builder, which is why it is written as the new API's contract rather than as a
differential test against the old one.  The differential equality pin was run
out of band over 284 builder calls in 151 tests (whole ``tests/window`` +
``tests/presentation`` + ``tests/ui/test_frame_session.py``) and produced zero
divergence in delta or session state.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from arrayscope.core.view_state import ViewState
from arrayscope.display.lod import LodInfo
from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationState
from arrayscope.display.model.tile_identity import TileLodIdentity, tile_ack_identity
from arrayscope.display.montage import RenderedTile, make_montage_plan
from arrayscope.display.shader_mapping import TexturePlaneKind
from arrayscope.window.frame_session import FrameSession

TILE_SHAPE = (4, 4)
SOURCES = (0, 1, 2, 3, 4, 5, 6, 7)


def _session(**overrides) -> FrameSession:
    state = ViewState.from_shape((4, 4, len(SOURCES))).with_montage_axis(
        2, indices=SOURCES, text=":"
    )
    plan = make_montage_plan(state, axis=2, indices=SOURCES, tile_shape=TILE_SHAPE, columns=4)
    fields = {
        "session_id": 1,
        "key": "key",
        "render_generation": 1,
        "level_key": "levels",
        "level_expected_indices": SOURCES,
        "plan": plan,
        "view_state": state,
        "document": None,
        "montage_axis": 2,
        "colormap_lut": None,
        "viewport_shape": (16, 16),
        "view_range": None,
        "output_dtype": np.dtype(np.float32),
        "rgb": False,
        "window_mode": None,
        "force_auto": False,
        "visible_tiles": plan.tiles,
        "rendered_tiles": {},
        "loading_tiles": set(),
        "skipped_tiles": set(),
        "shader_display": True,
        "retain_out_of_view_payloads": True,
        "tile_residency_budget_bytes": 64 * 1024 * 1024,
    }
    fields.update(overrides)
    return FrameSession(**fields)


def _payload(session: FrameSession, tile_number: int) -> DisplayTilePayload:
    tile = session.plan.tiles[int(tile_number)]
    image = np.full(TILE_SHAPE, float(tile.source_index), dtype=np.float32)
    return DisplayTilePayload(
        tile_number=int(tile_number),
        source_index=int(tile.source_index),
        image=image,
        histogram_data=image,
        source_id=("tile", int(tile.source_index), "exact"),
        lod=LodInfo(level=0, factor=1, source_shape=TILE_SHAPE, texture_shape=TILE_SHAPE),
        tile_identity=session._tile_identity(
            tile,
            texture_kind=TexturePlaneKind.SCALAR_R32F,
            complex_mapping=None,
            lod=TileLodIdentity(),
            quality="exact",
        ),
    )


def _present(session: FrameSession, *tile_numbers: int) -> None:
    """Put payloads on screen the way an acknowledged commit does."""

    session.sync_lifecycle_scope()
    state_payloads = dict(session.tile_presentation_state.payloads)
    for tile_number in tile_numbers:
        index = int(tile_number)
        payload = _payload(session, index)
        session.display_tile_payloads[index] = payload
        session.record_tile_payload(payload)
        session.lifecycle.commit_emitted({index: payload})
        session.lifecycle.backend_ack({index: payload})
        session.lifecycle.acknowledge_presented(index, tile_ack_identity(payload), "exact", 0)
        state_payloads[index] = payload
    session.tile_presentation_state = TilePresentationState(state_payloads)


def _render(session: FrameSession, *tile_numbers: int) -> None:
    for tile_number in tile_numbers:
        tile = session.plan.tiles[int(tile_number)]
        image = np.full(TILE_SHAPE, float(tile.source_index), dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))


def _look_at(session: FrameSession, tile_numbers) -> None:
    wanted = {int(tile) for tile in tile_numbers}
    session.visible_tiles = tuple(
        tile for tile in session.plan.tiles if int(tile.montage_index) in wanted
    )
    session.visible_tile_numbers = frozenset(wanted)
    session.sync_lifecycle_scope()


# --------------------------------------------------------------- fingerprints


#: Identities carry the address of the plane they describe, and two sessions
#: built from the same recipe hold different arrays at different addresses.
#: The comparisons here are *between sessions*, so the address is noise —
#: everything else in the identity still has to match exactly.
_ADDRESSES = re.compile(r"\b(pointer|id)=\d+|0x[0-9a-f]{6,}|\barray:[0-9a-f]{32}\b")


def _scrub(text: str) -> str:
    return _ADDRESSES.sub("<addr>", text)


def _canonical(value, depth: int = 0):
    if depth > 8:
        return "<deep>"
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, str(value.dtype), value.tobytes())
    if isinstance(value, DisplayTilePayload):
        return (
            "payload",
            int(value.tile_number),
            int(value.source_index),
            _scrub(repr(value.source_id)),
            str(value.quality),
            _scrub(repr(value.tile_identity)),
            _canonical(value.image, depth + 1),
        )
    if isinstance(value, dict):
        return tuple(
            (_scrub(repr(key)), _canonical(item, depth + 1))
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_scrub(repr(item)) for item in value))
    if isinstance(value, (tuple, list)):
        return tuple(_canonical(item, depth + 1) for item in value)
    return _scrub(repr(value))


_RECORD_FIELDS = (
    "semantic",
    "presentation",
    "emitted_source_id",
    "presented_source_id",
    "presented_quality",
    "presented_level",
    "backend_source_id",
    "parked_reason",
    "load_intent",
    "request_active",
    "stage_key",
    "evaluation_claim",
    "target",
    "task_claim",
    "stage_producer_key",
    "failed_reason",
    "satisfaction_traced",
)


def _lifecycle_fingerprint(lifecycle) -> tuple:
    records = []
    for tile_number, record in sorted(lifecycle._records.items()):
        records.append(
            (
                tile_number,
                tuple(_scrub(repr(getattr(record, name))) for name in _RECORD_FIELDS),
                _canonical(record.presentable_payloads),
                tuple(sorted(_scrub(repr(key)) for key in record.levels)),
                tuple(sorted(_scrub(repr(entry)) for entry in record.levels.values())),
                tuple(sorted(_scrub(repr(pair)) for pair in record.resigned)),
                _canonical(record.preview_claims),
            )
        )
    return (
        tuple(records),
        tuple(sorted(lifecycle._parked)),
        tuple(sorted(lifecycle._evaluating)),
        tuple(sorted(lifecycle._presented)),
        tuple(sorted(lifecycle._loading)),
        tuple(sorted(lifecycle._active_requests)),
        tuple(sorted(lifecycle._skipped)),
        int(lifecycle._skipped_revision),
        int(lifecycle._level_order),
        int(lifecycle._level_revision),
        int(lifecycle._identity_rejections),
        _canonical(lifecycle._backend_identities),
        _canonical({key: tuple(sorted(value)) for key, value in lifecycle._stage_blocked.items()}),
        _canonical(lifecycle._identity_rejection_counts),
    )


#: Everything a build is allowed to touch, plus the collections it must not.
_FINGERPRINTED_ATTRS = (
    "display_tile_payloads",
    "dirty_payloads",
    "pending_payload_upserts",
    "pending_removals",
    "tile_source_ids",
    "_identity_retry_attempts",
    "retained_out_of_view",
    "retained_out_of_view_bytes",
    "retained_out_of_view_evictions",
    "retained_out_of_view_returns",
    "retained_out_of_view_admitted",
    "_retention_departure_stamp",
    "deferred_out_of_view_invalidations",
    "deferred_out_of_view_returned",
    "deferred_out_of_view_deferred",
    "deferred_out_of_view_drained",
    "tile_presentation_state",
    "payload_revision",
    "visibility_revision",
    "viewport_revision",
    "structure_revision",
    "presentation_geometry_changed",
    "_layout_geometry_changed_pending",
    "_last_active_tiles",
    "_last_planned_tiles",
    "_last_near_tiles",
    "_last_viewport_identity",
    "_atomic_fast_reject_reason",
    "atomic_successor_pending",
    "lod_stats_cross_level_reuses",
    "lod_stats_recomputes",
    "lod_floor_presentations",
    "acknowledged_source_ids",
    "lod_preview_metadata",
    "rendered_tiles",
    # ADR 0065 stage 2 slice D2c: the retarget's stage phase plans too, and the
    # resident-crop-rebind seed is its only writer of these.  They are in the
    # fingerprint so a seed that escapes a dropped plan is visible here.
    "resident_crop_rebound_source_ids",
    "resident_crop_fallback_source_ids",
    "resident_crop_governor_stats",
    "resident_crop_rebind_stats",
    "resident_crop_rebind_seeded",
    "level_evidence_reanchor",
    "tile_state_revision",
)


def _session_fingerprint(session: FrameSession) -> tuple:
    attrs = tuple(
        (name, _canonical(getattr(session, name, "<missing>"))) for name in _FINGERPRINTED_ATTRS
    )
    state = session.tile_presentation_state
    return (
        attrs,
        int(getattr(state, "revision", 0) or 0),
        _canonical(dict(getattr(state, "payloads", {}) or {})),
        _lifecycle_fingerprint(session.lifecycle),
        _canonical(session.level_generation.tile_revisions),
        _canonical(session.level_generation.tile_values),
        _canonical(session.level_generation.stale_active_tiles),
        int(session.level_generation.revision),
        _scrub(repr(session.level_generation.target_levels)),
        _scrub(repr(session.level_generation.semantic_key)),
        tuple(sorted(session.level_generation.active_tiles)),
    )


def _delta_fingerprint(delta) -> tuple:
    if delta is None:
        return ("<no delta>",)
    return tuple(
        (name, _canonical(getattr(delta, name)))
        for name in sorted(vars(delta))
        if not name.startswith("__")
    )


# ----------------------------------------------------------------- scenarios


def _scenario_cold_fill() -> FrameSession:
    """Nothing presented, four rendered tiles waiting for their first commit."""

    session = _session()
    _look_at(session, range(4))
    _render(session, 0, 1, 2, 3)
    return session


def _scenario_capped_fill() -> FrameSession:
    """More debt than one governed transaction can admit."""

    session = _session()
    _look_at(session, SOURCES)
    _render(session, *SOURCES)
    return session


def _scenario_settled() -> FrameSession:
    """Everything on screen and acknowledged: the build should be a near no-op."""

    session = _session()
    _look_at(session, range(4))
    _render(session, 0, 1, 2, 3)
    _present(session, 0, 1, 2, 3)
    return session


def _scenario_out_of_view_retention() -> FrameSession:
    """Four tiles presented, then the camera moves off two of them."""

    session = _session()
    _look_at(session, range(4))
    _render(session, 0, 1, 2, 3)
    _present(session, 0, 1, 2, 3)
    session.lifecycle.backend_presented_snapshot(
        {index: tile_ack_identity(_payload(session, index)) for index in range(4)}
    )
    _look_at(session, (0, 1))
    return session


def _scenario_stale_backend_identity() -> FrameSession:
    """The backend draws an identity the session no longer wants."""

    session = _session()
    _look_at(session, range(4))
    _render(session, 0, 1, 2, 3)
    _present(session, 0, 1, 2, 3)
    session.lifecycle.backend_presented_snapshot({0: ("stale", 0), 1: ("stale", 1)})
    return session


def _scenario_discarded_out_of_scope() -> FrameSession:
    """Stale backend slots outside the planned scope: the discard path.

    ``presentation_discarded`` is the one builder event that shrinks the
    machine's presented set, and out-of-scope slots reach it without the
    ``may_remove_visible`` gate.
    """

    session = _session()
    _look_at(session, range(4))
    _render(session, 0, 1, 2, 3)
    _present(session, 0, 1, 2, 3)
    _look_at(session, (0, 1))
    session.lifecycle.backend_presented_snapshot({2: ("stale", 2), 3: ("stale", 3)})
    return session


def _scenario_pending_removals() -> FrameSession:
    """Tiles queued for physical removal alongside live upserts."""

    session = _session()
    _look_at(session, range(4))
    _render(session, 0, 1, 2, 3)
    _present(session, 0, 1, 2, 3)
    _look_at(session, (0, 1))
    session.pending_removals.update({2, 3})
    session.dirty_payloads[0] = None
    return session


def _scenario_dirty_current_payload() -> FrameSession:
    """A presented, already-current tile carrying a dirty entry.

    The builder's "the dirty record's promise is already kept" pop — the one
    that lets a commit loop settle — is the only writer of ``dirty_payloads``
    an ordinary settled build reaches.
    """

    session = _session()
    _look_at(session, range(4))
    _render(session, 0, 1, 2, 3)
    _present(session, 0, 1, 2, 3)
    for index in range(4):
        session.dirty_payloads[index] = None
    return session


def _scenario_departed_tiles() -> FrameSession:
    """Presented tiles whose results are gone and whose camera moved off them.

    Drives the drop pass: ``_drop_display_payload`` retires the slot, queues the
    physical removal and forgets the tile's level generation — three families
    the scenarios above never touch.
    """

    session = _session()
    _look_at(session, range(4))
    _render(session, 0, 1, 2, 3)
    _present(session, 0, 1, 2, 3)
    for index in (2, 3):
        session.rendered_tiles.pop(index, None)
    _look_at(session, (0, 1))
    return session


def _scenario_dropped_tiles_without_retention() -> FrameSession:
    """``departed-tiles`` with retention off, so the drop really drops.

    Retention is what keeps the departed slots alive in the scenario above;
    switched off, the same setup reaches ``pending_removals`` and
    ``level_generation.forget_tile``.
    """

    session = _session(retain_out_of_view_payloads=False)
    _look_at(session, range(4))
    _render(session, 0, 1, 2, 3)
    _present(session, 0, 1, 2, 3)
    for index in (2, 3):
        session.rendered_tiles.pop(index, None)
    _look_at(session, (0, 1))
    return session


def _scenario_rehydrated_state() -> FrameSession:
    """Backend truth ahead of the acknowledged state: the rehydration write.

    The one path that assigns ``tile_presentation_state`` inside a build.
    """

    session = _session()
    _look_at(session, range(4))
    _render(session, 0, 1, 2, 3)
    _present(session, 0, 1, 2, 3)
    session.lifecycle.backend_presented_snapshot(
        {index: tile_ack_identity(_payload(session, index)) for index in range(4)}
    )
    session.tile_presentation_state = TilePresentationState()
    return session


def _scenario_deferred_invalidations() -> FrameSession:
    """Out-of-view invalidations deferred, then their tiles come back in scope."""

    session = _session()
    _look_at(session, range(4))
    _render(session, 0, 1, 2, 3)
    _present(session, 0, 1, 2, 3)
    for index in (2, 3):
        session.defer_out_of_view_invalidation(index)
    _look_at(session, range(4))
    return session


SCENARIOS = {
    "cold-fill": (_scenario_cold_fill, {}),
    "cold-fill-capped": (_scenario_capped_fill, {"max_upserts": 2}),
    "cold-fill-byte-capped": (_scenario_capped_fill, {"max_upserts": 4, "max_upsert_bytes": 64}),
    "settled": (_scenario_settled, {}),
    "out-of-view-retention": (_scenario_out_of_view_retention, {}),
    "stale-backend-identity": (_scenario_stale_backend_identity, {}),
    "pending-removals": (_scenario_pending_removals, {}),
    "deferred-invalidations": (_scenario_deferred_invalidations, {}),
    "dirty-current-payload": (_scenario_dirty_current_payload, {}),
    "departed-tiles": (_scenario_departed_tiles, {}),
    "dropped-tiles": (_scenario_dropped_tiles_without_retention, {}),
    "rehydrated-state": (_scenario_rehydrated_state, {}),
    "discarded-out-of-scope": (_scenario_discarded_out_of_scope, {}),
}


def _source_ids() -> dict[int, object]:
    return {index: ("tile", index) for index in SOURCES}


@pytest.fixture(params=sorted(SCENARIOS))
def scenario(request):
    build, kwargs = SCENARIOS[request.param]
    return build, dict(kwargs)


# --------------------------------------------------------------------- pins


def test_planning_a_transaction_writes_nothing_to_the_session(scenario):
    """Property 1: a prepared-and-dropped transaction leaves no trace.

    This is the pin the whole slice exists for, and the completeness check for
    the staged-container table: a container the planner reaches without a
    private copy fails here.
    """

    build, kwargs = scenario
    session = build()
    before = _session_fingerprint(session)

    plan = session.plan_tile_presentation(_source_ids(), **kwargs)

    assert plan.delta is not None
    assert _session_fingerprint(session) == before


def test_planning_twice_then_applying_once_equals_planning_once(scenario):
    """Property 2, the exact sequence a take-miss produces on the owner."""

    build, kwargs = scenario
    once, twice = build(), build()
    assert _session_fingerprint(once) == _session_fingerprint(twice)

    once.apply_tile_presentation_plan(once.plan_tile_presentation(_source_ids(), **kwargs))

    twice.plan_tile_presentation(_source_ids(), **kwargs)
    second = twice.plan_tile_presentation(_source_ids(), **kwargs)
    twice.apply_tile_presentation_plan(second)

    assert _session_fingerprint(twice) == _session_fingerprint(once)


def test_planning_dropping_and_replanning_equals_planning_once(scenario):
    """Property 2's other order: the dropped plan is garbage, not history."""

    build, kwargs = scenario
    once, dropped = build(), build()

    reference = once.plan_tile_presentation(_source_ids(), **kwargs)
    once.apply_tile_presentation_plan(reference)

    del_me = dropped.plan_tile_presentation(_source_ids(), **kwargs)
    del del_me
    replanned = dropped.plan_tile_presentation(_source_ids(), **kwargs)
    dropped.apply_tile_presentation_plan(replanned)

    assert _session_fingerprint(dropped) == _session_fingerprint(once)
    assert _delta_fingerprint(replanned.delta) == _delta_fingerprint(reference.delta)


def test_plan_then_apply_is_the_monolithic_build(scenario):
    """The inline anchor: ``build_tile_presentation`` is plan-then-apply."""

    build, kwargs = scenario
    split, inline = build(), build()

    plan = split.plan_tile_presentation(_source_ids(), **kwargs)
    split_state, split_delta = split.apply_tile_presentation_plan(plan)
    inline_state, inline_delta = inline.build_tile_presentation(_source_ids(), **kwargs)

    assert _delta_fingerprint(split_delta) == _delta_fingerprint(inline_delta)
    assert _canonical(dict(split_state.payloads)) == _canonical(dict(inline_state.payloads))
    assert _session_fingerprint(split) == _session_fingerprint(inline)


def test_the_applier_keeps_the_machines_and_their_views_identical():
    """Installing in place, not swapping: identity across a commit is load-bearing.

    A ``TileRecord`` handed out before a commit and a ``loading_tiles`` view
    bound at session build must both still describe the session afterwards.
    Swapping the lifecycle object satisfied every state assertion and silently
    stranded both.
    """

    session = _scenario_cold_fill()
    lifecycle = session.lifecycle
    record = session.lifecycle.record(0)
    loading = session.loading_tiles
    rendered = session.rendered_tiles

    session.build_tile_presentation(_source_ids())

    assert session.lifecycle is lifecycle
    assert session.loading_tiles is loading
    assert session.rendered_tiles is rendered
    assert session.lifecycle.record(0) is record
    assert record.presentable_payloads == session.lifecycle.record(0).presentable_payloads
    assert set(loading) == set(session.lifecycle.loading_tiles)


def test_a_plan_refuses_to_apply_to_another_session():
    """A plan is bound to the session generation it was decided against."""

    source = _scenario_cold_fill()
    other = _scenario_cold_fill()
    plan = source.plan_tile_presentation(_source_ids())

    with pytest.raises(RuntimeError, match="does not belong"):
        other.apply_tile_presentation_plan(plan)


def test_the_atomic_successor_plan_is_pure_too():
    """The fast path plans and declines without writing its reason.

    ``_atomic_fast_reject_reason`` is a session write the prepare seam reads
    immediately afterwards, so the declining case has to travel in a plan like
    everything else.
    """

    session = _scenario_settled()
    session.atomic_successor_pending = True
    session._layout_geometry_changed_pending = True
    before = _session_fingerprint(session)

    plan = session.plan_atomic_successor_presentation()

    assert plan.delta is None
    assert plan.path == "atomic-declined"
    assert _session_fingerprint(session) == before
    session.apply_tile_presentation_plan(plan)
    assert session._atomic_fast_reject_reason == "layout"


# ---------------------------------------------------------- mutation checks


def _effective_families(session: FrameSession, plan) -> tuple[str, ...]:
    """Staged families this scenario actually changes, and the pins can see.

    A family whose staged value already equals the live one is a no-op for the
    scenario, and dropping it proves nothing; the memo caches
    (``_priority_context``, ``_best_floor_key_memo`` …) travel in the record
    but are not session truth and are deliberately outside the fingerprint.
    Deriving the list instead of writing one down means a family added to the
    builder tomorrow is mutation-checked without anyone remembering to add it.
    """

    return tuple(
        name
        for name, value in plan.staged.items()
        if name in _FINGERPRINTED_ATTRS
        and _canonical(value) != _canonical(getattr(session, name, "<missing>"))
    )


def test_every_applier_family_the_scenario_exercises_is_load_bearing(scenario):
    """Mutation check: drop one family from the record and the outcome moves.

    A pin that stays green when the applier stops writing something is pinning
    nothing.
    """

    build, kwargs = scenario
    reference = build()
    reference.build_tile_presentation(_source_ids(), **kwargs)

    probe = build()
    families = _effective_families(probe, probe.plan_tile_presentation(_source_ids(), **kwargs))
    assert families, "this scenario changes nothing, so it checks nothing"

    for family in families:
        session = build()
        partial = session.plan_tile_presentation(_source_ids(), **kwargs)
        partial.staged.pop(family)
        session.apply_tile_presentation_plan(partial)
        assert _session_fingerprint(session) != _session_fingerprint(reference), family


def test_dropping_the_lifecycle_install_is_visible():
    """Mutation check for the family that does not travel by assignment."""

    session = _scenario_capped_fill()
    reference = _scenario_capped_fill()
    reference.build_tile_presentation(_source_ids(), max_upserts=2)

    plan = session.plan_tile_presentation(_source_ids(), max_upserts=2)
    for name, value in plan.staged.items():
        setattr(session, name, value)

    assert _session_fingerprint(session) != _session_fingerprint(reference)


# ------------------------------------------------- the seam, not the builder
#
# Everything above is the *builder's* contract.  Slice D moves the property one
# level out: the whole prepare seam — level decisions, the atomic-successor
# block, the metadata publication decision, every bail's flags — plans against
# a staged session too, and `_take_prepared_outcome` is the only writer.  That
# needs a live window, because the seam is where the renderer-facing calls are
# and the whole point of the separation is which session each of them gets.


@pytest.fixture
def prepare_seam_window(qtbot):
    import numpy as np

    from arrayscope.window import ArrayScopeWindow
    from tests.settle import deliver_posted_events, settle_window

    win = ArrayScopeWindow(np.arange(8 * 8 * 4, dtype=np.float32).reshape(8, 8, 4))
    qtbot.addWidget(win)
    win._set_view_state(
        win.view_state.with_montage_axis(2, columns=2, indices=(0, 1, 2, 3), text=":")
    )
    win.render(reason="test-prepare-seam-purity")
    settle_window(win, lambda: getattr(win.renderer, "_frame_session", None) is not None)
    session = win.renderer._frame_session
    settle_window(win, lambda: session.visible_plan_complete())
    deliver_posted_events()
    try:
        yield win
    finally:
        win.close()


def _seam_effects(win):
    from arrayscope.window.frame_effects import FramePipelineEffects

    session = win.renderer._frame_session
    pipeline = getattr(session, "pipeline", None)
    return session, (
        pipeline.effects if pipeline is not None else FramePipelineEffects(win.renderer, session)
    )


def _prepare_once(effects, session):
    direct = effects.direct_tile_layer_presentation()
    return effects._prepare_tile_layer_transaction(direct, dirty_tiles=session.peek_dirty_tiles())


def test_the_prepare_seam_writes_nothing_when_its_outcome_is_dropped(prepare_seam_window):
    """Slice D's contract, at the seam.

    A dropped prepare is the ordinary outcome of a take-miss, and slice C's
    bisect proved it is the *second* prepare that corrupts: it plans against
    the first one's writes.  So the first one must leave none.
    """

    pytest.importorskip("pytestqt")
    session, effects = _seam_effects(prepare_seam_window)
    before = _session_fingerprint(session)

    outcome = _prepare_once(effects, session)

    assert outcome is not None, "the seam decided nothing, so nothing was observed"
    assert outcome.plan is not None
    assert _session_fingerprint(session) == before
    # And per attribute, because the fingerprint covers what the *builder*
    # owns: a settled window's seam legitimately decides a bail whose whole
    # staged set is bookkeeping the fingerprint never looks at, and against
    # that scenario the fingerprint alone cannot fail. Removing the staging
    # makes exactly this loop red.
    staged = dict(outcome.plan.staged)
    assert staged, "the seam decided no session write, so nothing was observed"
    for name, value in staged.items():
        assert getattr(session, name, None) is not value, name


def test_preparing_twice_and_dropping_the_first_decides_the_same_transaction(
    prepare_seam_window,
):
    """Property 2 at the seam: the second prepare cannot see the first."""

    pytest.importorskip("pytestqt")
    session, effects = _seam_effects(prepare_seam_window)

    first = _prepare_once(effects, session)
    second = _prepare_once(effects, session)

    assert first is not None
    assert second is not None
    assert type(first) is type(second)
    assert _canonical(getattr(first, "outcome", None)) == _canonical(
        getattr(second, "outcome", None)
    )
    first_delta = getattr(first, "tile_delta", None)
    second_delta = getattr(second, "tile_delta", None)
    if first_delta is not None:
        assert _delta_fingerprint(first_delta) == _delta_fingerprint(second_delta)


def test_the_take_is_the_seams_only_writer(prepare_seam_window):
    """Read per attribute, not by fingerprint.

    The fingerprint answers "did anything the *builder* owns move", and a
    settled window's seam legitimately decides a bail whose whole staged set is
    bookkeeping the fingerprint does not cover (commit counters, the applied
    level source). Asking the plan itself which attributes it carries makes the
    claim exact and keeps it honest on every scenario: whatever the seam
    decided this time, none of it was installed before the take and all of it
    was installed after.
    """

    pytest.importorskip("pytestqt")
    session, effects = _seam_effects(prepare_seam_window)

    outcome = _prepare_once(effects, session)
    staged = dict(outcome.plan.staged)
    assert staged, "the seam decided no session write, so nothing was observed"
    unwritten = [
        name for name, value in staged.items() if getattr(session, name, None) is not value
    ]
    assert unwritten, "prepare installed every write it planned"

    effects._take_prepared_outcome(outcome)

    for name, value in staged.items():
        assert getattr(session, name) is value, name


def test_a_prepared_outcome_from_another_generation_is_refused(prepare_seam_window):
    """The identity gate, which only becomes reachable once prepare is posted."""

    pytest.importorskip("pytestqt")
    from dataclasses import replace as _replace

    win = prepare_seam_window
    session, effects = _seam_effects(win)
    outcome = _prepare_once(effects, session)
    assert outcome is not None

    stale = _replace(
        outcome,
        identity=_replace(outcome.identity, session_id=int(outcome.identity.session_id) + 1),
    )
    before = _session_fingerprint(session)
    refused_before = int(getattr(win.renderer, "_montage_prepare_refused_stale", 0) or 0)

    assert effects._take_prepared_outcome(stale) is None
    assert _session_fingerprint(session) == before
    assert int(win.renderer._montage_prepare_refused_stale) == refused_before + 1


# ------------------------------------------- the retarget seam, not the commit
#
# ADR 0065 stage 2 slice D2c moves the same property one seam further out.  D2b
# measured the retarget's remaining GUI-side hold and named it exactly:
# ``effects.tile_states`` (2223 ms per three fills) and
# ``mark_ladder_swaps_for_current_demand`` (1816 ms), against an unlocked
# ``ladder.plan`` of single-digit milliseconds.  Neither is arithmetic — both
# are session WRITES (evaluation-claim releases, crop-rebind seeding, dirty swap
# marks) — so neither could leave the lock by being moved, only by being
# planned.


def test_the_seeds_own_containers_travel_with_the_plan():
    """Completeness check for the three families slice D2c added to the table.

    ``rebind_resident_crop_tiles`` is the only writer of these, and building a
    real resident rebind needs a backend residency seam and a committed
    predecessor frame.  The property under test is not the rebind's decision but
    the *containment* of its writes, so the writes are made directly: a family
    missing from ``_PRESENTATION_PLAN_CONTAINERS`` is shared with the live
    session, and the first assertion below sees it immediately.  Removing any
    one of the three entries is the mutation check.
    """

    session = _scenario_settled()
    before = _session_fingerprint(session)

    staged = session._staged_presentation_session()
    staged.resident_crop_rebound_source_ids[0] = ("rebound", 0)
    staged.resident_crop_fallback_source_ids[0] = ("fallback", 0)
    staged.resident_crop_governor_stats["seed_callbacks"] = 3
    staged.level_evidence_reanchor = True
    plan = session._harvest_presentation_plan(staged, None, None, path="retarget-tile-states")

    assert _session_fingerprint(session) == before

    session.apply_tile_presentation_plan(plan)

    assert session.resident_crop_rebound_source_ids == {0: ("rebound", 0)}
    assert session.resident_crop_fallback_source_ids == {0: ("fallback", 0)}
    assert session.resident_crop_governor_stats["seed_callbacks"] == 3
    assert session.level_evidence_reanchor is True


def _retarget_seam(win):
    renderer = win.renderer
    session = renderer._frame_session
    pipeline = renderer._frame_pipeline_for_session(session)
    intent = renderer._montage_render_intent(session)
    scope = renderer._lod_admission_scope(session, intent)
    return session, pipeline, pipeline.effects, intent, scope


def test_the_retarget_stage_phase_plans_without_writing_the_session(prepare_seam_window):
    """Slice D2c's contract, at the retarget seam.

    Every write the stage phase used to make under the lock — the claim
    releases, the crop-rebind seed and the swap marks — now lands on a private
    copy, and a plan nobody applies leaves nothing behind.  The mutation is
    planning against the live session (which is exactly what ``staged=False``
    selects), and it turns the fingerprint assertion red.
    """

    pytest.importorskip("pytestqt")
    from arrayscope.render import lod as render_lod

    win = prepare_seam_window
    session, _pipeline, effects, intent, scope = _retarget_seam(win)
    before = _session_fingerprint(session)
    memo_before = dict(getattr(win.renderer, "_resident_crop_canonical_planes", None) or {})

    staging = effects.stage_tile_states(intent, session.lod_policy_decision.demand, scope)
    assert staging is not None, "the seam staged nothing, so nothing was observed"
    assert staging.staged, "the session was not staged, so purity is not what ran"
    assert staging.session is not session
    render_lod.mark_ladder_swaps_for_current_demand(staging.session)
    planned = effects.plan_tile_states(staging)

    assert planned.states, "the plan decided no tile states, so nothing was observed"
    assert _session_fingerprint(session) == before
    assert dict(getattr(win.renderer, "_resident_crop_canonical_planes", None) or {}) == memo_before
    for name, value in dict(planned.plan.staged).items():
        assert getattr(session, name, None) is not value, name


def test_the_retargets_applier_is_its_only_writer(prepare_seam_window):
    """Whatever the stage phase decided, all of it lands on apply and none before."""

    pytest.importorskip("pytestqt")
    win = prepare_seam_window
    session, _pipeline, effects, intent, scope = _retarget_seam(win)

    staging = effects.stage_tile_states(intent, session.lod_policy_decision.demand, scope)
    assert staging is not None
    assert staging.staged
    planned = effects.plan_tile_states(staging)
    staged = dict(planned.plan.staged)
    assert staged, "the stage phase decided no session write, so nothing was observed"
    assert [name for name, value in staged.items() if getattr(session, name, None) is not value]

    assert effects.apply_tile_states_plan(planned) is True

    for name, value in staged.items():
        assert getattr(session, name) is value, name


def test_a_retarget_plan_overtaken_by_a_live_write_is_refused(prepare_seam_window):
    """The conservation refusal, which only exists because the middle is unlocked.

    The applier installs whole containers, so an admission landing between the
    staging and the apply would be reverted — a lost admission, which is the
    settle wedge D2 measured (35 stale tiles, nothing loading).  One is staged
    deliberately here.  Without the check the admission's payload is gone and
    the counter stays empty, which is the mutation.
    """

    pytest.importorskip("pytestqt")
    win = prepare_seam_window
    session, _pipeline, effects, intent, scope = _retarget_seam(win)

    staging = effects.stage_tile_states(intent, session.lod_policy_decision.demand, scope)
    assert staging is not None
    assert staging.staged
    planned = effects.plan_tile_states(staging)

    # The interleaving: an admission installs a newer payload for a tile while
    # the retarget was planning against a copy that does not have it.
    successor = object()
    session.display_tile_payloads[9999] = successor

    assert effects.apply_tile_states_plan(planned) is False
    assert session.display_tile_payloads.get(9999) is successor
    assert int(win.renderer._montage_retarget_stage_refusals.get("interleaved", 0)) == 1


def test_a_retarget_plan_from_another_generation_is_refused(prepare_seam_window):
    """The identity gate, the same one the commit's take applies."""

    pytest.importorskip("pytestqt")
    win = prepare_seam_window
    session, _pipeline, effects, intent, scope = _retarget_seam(win)

    staging = effects.stage_tile_states(intent, session.lod_policy_decision.demand, scope)
    assert staging is not None
    assert staging.staged
    planned = effects.plan_tile_states(staging)
    object.__setattr__(planned.plan, "session_id", int(session.session_id) + 1)
    before = _session_fingerprint(session)

    assert effects.apply_tile_states_plan(planned) is False
    assert _session_fingerprint(session) == before
    assert int(win.renderer._montage_retarget_stage_refusals.get("generation", 0)) == 1
