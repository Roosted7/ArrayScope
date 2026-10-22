"""``FrameSession.mutation_revision`` must move for every mutation there is.

A retarget gate that skips planning when the epoch has not moved turns a
missed bump into a **lost wakeup**, and ``_montage_watchdog_tick`` observes
without repairing, so a lost wakeup here is permanent rather than late.  The
only defence that survives the next person adding a field is a test that does
not know the field names.

So this walks the class by reflection against a LIVE settled montage session.
For every public callable it takes a deep, field-reflected snapshot of the
session and its ledger, invokes the method with arguments taken from that same
live session, snapshots again, and asserts:

    the observable state changed  =>  ``mutation_revision`` moved

Three properties make it un-rottable:

* the snapshot enumerates ``dataclasses.fields(FrameSession)``, the session's
  own ``__dict__`` and the ledger's records, so a **new field** is compared
  without being named here;
* a method whose parameters this file cannot supply is a **failure**, not a
  skip.  A new method with an unfamiliar parameter forces its author to teach
  ``_arguments`` what a plausible value is, and while they do that the walk
  covers their method;
* the arguments are derived from the live session (its own tiles, payloads,
  identities and plan) rather than hand-built fakes, so they stay plausible as
  the product's types change.

The converse direction (the epoch moving when nothing changed) is deliberately
NOT asserted *in general*: the counter fails callers open on anything it cannot
decide cheaply, and a spurious bump costs one replan while a missing one costs
a frame that never arrives.  The one case it does decide is a store of the
object the slot already holds, which is not a write at all -- see
``core.mutation_clock``'s no-op rule and
``tests/core/test_mutation_clock_noop_writes.py``.

That rule is why the walks below assign a value that is *different*.  They
used to assert the hook by writing a field back to itself, which reads as "any
assignment bumps" and is now false; the property they were built to defend --
a field, container or sub-object whose write the epoch cannot see is a lost
wakeup -- is unchanged and is what they still assert.
"""

from __future__ import annotations

import inspect
from collections import deque
from collections.abc import Collection
from dataclasses import dataclass, is_dataclass
from dataclasses import fields as dataclass_fields
from hashlib import blake2b

import numpy as np
import pytest

from arrayscope.core.mutation_clock import (
    CONTAINER_MUTATORS,
    CONTAINER_READERS,
    TRACKED_CONTAINER_TYPES,
    ClockedState,
    MutationClock,
)
from arrayscope.display.model.frame import (
    TileCommitReport,
    TilePresentationDelta,
)
from arrayscope.display.montage import MontageTileState
from arrayscope.window.frame_session import FrameSession, LifecycleRungMaterializations
from tests.ui.helpers import clear_arrayscope_settings  # noqa: F401  (autouse-style import)
from tests.ui.helpers import settle_quiescent as _settle_quiet

DEPTH = 16
COLUMNS = 4


def _build(qtbot):
    from pyqtgraph.Qt import QtCore

    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", "pyqtgraph")
    settings.sync()
    win = ArrayScopeWindow(np.arange(12 * 12 * DEPTH, dtype=np.float32).reshape(12, 12, DEPTH))
    qtbot.addWidget(win)
    win.resize(520, 420)
    win.show()
    _settle_quiet(win)
    win._set_view_state(
        win.view_state.with_montage_axis(2, columns=COLUMNS, indices=tuple(range(DEPTH)), text=":")
    )
    win.update_image_view()
    _settle_quiet(win)
    session = win.renderer._frame_session
    assert session is not None
    assert session.display_tile_payloads, "the walk needs a session with real payloads"
    return win, session


def _preview_pages(session):
    """One checked materialized page, the shape ``admit_preview_plane`` demands."""

    from arrayscope.display.pyramid import MaterializedLodPage, plan_source_grid_pages
    from arrayscope.gpu.keys import SCALAR_R32F
    from arrayscope.render.lod import LodPageSetKey

    height, width = (int(value) for value in tuple(session.plan.tile_shape)[:2])
    page_plan = plan_source_grid_pages(
        content_key=("mutation-walk",),
        valid_source_rect_yx=(0, height, 0, width),
        reduction_yx=(1, 1),
        stored_page_shape=(max(1, height // 2), max(1, width // 2)),
        dtype="float32",
        representation=SCALAR_R32F,
        reducer="mean",
    )[0]
    pages = (MaterializedLodPage(page_plan, np.zeros(page_plan.stored_shape, dtype=np.float32)),)
    page_set_key = LodPageSetKey(
        source_id=("mutation-walk",),
        tile_id=0,
        level_xy=(1, 1),
        reducer="mean",
        plans=(page_plan,),
    )
    return pages, page_set_key


def _arguments(win, session) -> dict[str, object]:
    """Plausible values BY PARAMETER NAME, taken from the live session.

    An unknown name fails the walk on purpose (see the module docstring).
    """

    tile_number, payload = next(iter(sorted(session.display_tile_payloads.items())))
    tile_number = int(tile_number)
    tile = next(item for item in session.plan.tiles if int(item.montage_index) == tile_number)
    rendered = session.rendered_tiles.get(tile_number) or next(
        iter(session.rendered_tiles.values()), None
    )
    identity = session.tile_target_identity(tile, lod_level=0)
    preview_pages, page_set_key = _preview_pages(session)
    delta = TilePresentationDelta(
        structure_revision=int(session.structure_revision),
        payload_revision=int(session.payload_revision),
        visibility_revision=int(session.visibility_revision),
        level_revision=int(session.level_revision),
        histogram_revision=int(session.histogram_revision),
        viewport_revision=int(session.viewport_revision),
        base_revision=int(session.tile_presentation_state.revision),
        target_revision=int(session.tile_presentation_state.revision) + 1,
        upserts={tile_number: payload},
        active_tiles=tuple(sorted(session.display_tile_payloads)),
    )
    report = TileCommitReport(
        presented_tiles=(tile_number,),
        committed_upserts=(tile_number,),
        removed_tiles=(),
    )
    return {
        "acknowledged": {tile_number: identity},
        "active": (tile_number,),
        "active_tiles": tuple(sorted(session.display_tile_payloads)),
        "admission": None,
        "all_indices": tuple(int(item.source_index) for item in session.plan.tiles),
        "backend_identities": {tile_number: identity},
        "cached_tiles": dict(session.rendered_tiles),
        "canonical_by_tile": {},
        "candidates": 1,
        "delta": delta,
        "focus": (0.0, 0.0),
        "frame_plan": session.frame_plan,
        "histogram": None,
        "key": session.level_key,
        "level_data": None,
        "level_key": session.level_key,
        "level_stats": None,
        "levels": (0.0, 1.0),
        "limit": 4,
        "lod": 0,
        "lod_level": 0,
        "max_count": 2,
        "native_shape": tuple(session.plan.tile_shape),
        "near_tiles": (tile_number,),
        "new_source_ids": {tile_number: session.tile_semantic_source_id(tile.source_index)},
        "payload": payload,
        "payloads": {tile_number: payload},
        "pending": (tile_number,),
        "physical_resident_fn": lambda *_args, **_kwargs: True,
        "plan": session.plan,
        "plan_tiles_by_number": {int(item.montage_index): item for item in session.plan.tiles},
        "planned_numbers": tuple(sorted(session.display_tile_payloads)),
        "plane": preview_pages,
        "previous_by_tile": {},
        "previous_payloads": {tile_number: payload},
        "priority_focus": (0.0, 0.0),
        "quality": "exact",
        "removals": (),
        "render_generation": int(session.render_generation) + 1,
        "rendered": rendered,
        "report": report,
        "semantic_key": session.semantic_key,
        "session_id": int(session.session_id) + 1,
        "shader_mapping": None,
        "source": None,
        "source_anchoring": session.source_anchoring,
        "source_ids": {tile_number: session.tile_semantic_source_id(tile.source_index)},
        "source_index": int(tile.source_index),
        "state": MontageTileState.LOADED,
        "target_level": 0,
        "targets": None,
        "texture_data": np.zeros(tuple(session.plan.tile_shape), dtype=np.float32),
        "texture_kind": getattr(payload, "texture_kind", None),
        "tile": tile,
        "tile_number": tile_number,
        "tile_numbers": (tile_number,),
        # ``camera_pending_tiles`` walks plan tiles, not tile numbers: it reads
        # ``montage_index`` AND ``source_index`` off each one, because the plan
        # is the authority on which source a slot shows.
        "tiles": tuple(session.plan.tiles),
        "upsert_cost_fn": lambda *_args, **_kwargs: 1,
        "view_range": session.view_range,
        "view_state": session.view_state,
        "viewport_shape": tuple(session.viewport_shape),
        "visible_tiles": tuple(session.visible_tiles),
        # Per-METHOD overrides, for a parameter name the product overloads.
        # Keyed by ``(method, parameter)``, checked before the by-name table.
        ("admit_preview_plane", "key"): page_set_key,
        ("attach_stage_fan_in", "state"): session.stage_fan_in,
    }


class _SnapshotValues:
    """Reflect the state graph once, without formatting shared pixel planes.

    References address this snapshot's flat node table. Repeated objects and
    cycles therefore cost one visit, and equality does not expand a shared
    graph back into a tree. Every snapshot gets a fresh table so in-place
    writes remain visible. Mutation clocks and explicitly derived record
    memos are bookkeeping rather than the state whose clock is under test.
    """

    def __init__(self):
        self.seen = {}
        self.nodes = []
        self.owners = []

    def value(self, value):
        if type(value) in (type(None), bool, int, str, bytes):
            return (type(value), value)
        if isinstance(value, (float, np.generic)):
            # A NaN must compare equal to its own unchanged snapshot.
            return (type(value), repr(value))
        if isinstance(value, MutationClock):
            return (MutationClock,)
        identity = id(value)
        if identity in self.seen:
            return ("ref", self.seen[identity])
        number = len(self.nodes)
        self.seen[identity] = number
        self.nodes.append(None)
        # Iterated lifecycle facades may yield temporary wrappers. Retain them
        # during this snapshot so their ids cannot be reused for another node.
        self.owners.append(value)
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                contents = tuple(self.value(item) for item in value.flat)
            else:
                digest = blake2b(digest_size=32)
                if value.size and value.itemsize:
                    with np.nditer(
                        value,
                        flags=["external_loop", "buffered", "zerosize_ok"],
                        op_flags=["readonly", "contig"],
                        order="C",
                        buffersize=max(1, 65536 // value.itemsize),
                    ) as iterator:
                        for block in iterator:
                            digest.update(block.view(np.uint8))
                contents = digest.digest()
            state = (value.shape, repr(value.dtype.descr), contents)
        elif isinstance(value, dict):
            state = tuple(
                (self.value(key), self.value(item))
                for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
            )
        elif isinstance(value, (set, frozenset)):
            state = tuple(self.value(item) for item in sorted(value, key=repr))
        elif isinstance(value, (tuple, list, deque)):
            state = tuple(self.value(item) for item in value)
        elif is_dataclass(value) and not isinstance(value, type):
            declared = dataclass_fields(value)
            attributes = dict(getattr(value, "__dict__", {}))
            for item in declared:
                attributes.setdefault(item.name, getattr(value, item.name))
            derived = {item.name for item in declared if item.metadata.get("derived")}
            state = tuple(
                (name, self.value(item))
                for name, item in sorted(attributes.items())
                if name != "_mutations" and name not in derived
            )
        elif isinstance(value, ClockedState):
            state = tuple(
                (name, self.value(item))
                for name, item in sorted(value.__dict__.items())
                if name != "_mutations"
            )
        elif inspect.ismethod(value):
            # Bound-method repr recursively formats its owning FrameSession.
            state = (self.value(value.__func__), self.value(value.__self__))
        elif isinstance(value, (Collection, LifecycleRungMaterializations)):
            # Lifecycle's set/list facades expose real ledger contents through
            # iteration. Their repr would format every shared payload again.
            state = tuple(self.value(item) for item in value)
        else:
            state = repr(value)
        self.nodes[number] = (type(value), state)
        return ("ref", number)


def _snapshot(session):
    """Everything the session and its ledger can be asked about, by reflection.

    Read from ``__dict__``, never through the attributes: several properties
    memoize, so reading them would be a mutation of its own.
    """

    values = _SnapshotValues()
    declared = tuple(item.name for item in dataclass_fields(FrameSession))
    own = tuple(
        (name, values.value(value))
        for name, value in sorted(session.__dict__.items())
        # The clock is the thing under test; the ledger is expanded below.
        if name not in {"_mutations", "lifecycle"}
    )
    lifecycle = session.lifecycle
    ledger = tuple(
        (name, values.value(value))
        for name, value in sorted(lifecycle.__dict__.items())
        if name not in {"_mutations", "_records"}
    )
    records = tuple(
        (number, values.value(record)) for number, record in sorted(lifecycle._records.items())
    )
    # ``declared`` is carried so a field REMOVED from the dataclass while an
    # instance attribute of that name lingers is still compared.
    return (declared, own, ledger, records, tuple(values.nodes))


def _observe_mutation(session, call, *args, **kwargs):
    before_state = _snapshot(session)
    before_revision = session.mutation_revision
    call(*args, **kwargs)
    after_revision = session.mutation_revision
    if after_revision > before_revision:
        # The implication is already satisfied. A second deep comparison
        # cannot expose a missed bump once the revision has visibly moved.
        return True, None
    return False, _snapshot(session) != before_state


def _public_methods():
    for name, member in sorted(vars(FrameSession).items()):
        if name.startswith("_"):
            continue
        if isinstance(member, property) or not callable(member):
            continue
        yield name


def _call_arguments(method_name, method, table):
    """Plausible arguments for ``method``, or the names we cannot supply."""

    signature = inspect.signature(method)
    args: list[object] = []
    kwargs: dict[str, object] = {}
    unknown: list[str] = []
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if (method_name, name) in table:
            value = table[(method_name, name)]
        elif name in table:
            value = table[name]
        elif parameter.default is not parameter.empty:
            continue
        else:
            unknown.append(name)
            continue
        if parameter.kind is parameter.KEYWORD_ONLY:
            kwargs[name] = value
        else:
            args.append(value)
    return args, kwargs, unknown


def test_snapshot_detects_unclocked_fields_and_pixels_without_recursive_repr():
    from arrayscope.presentation import TileLifecycle

    @dataclass
    class Watched:
        plane: np.ndarray
        reason: str

        def __repr__(self):
            raise AssertionError("snapshot formatted a nested state object")

    session = object.__new__(FrameSession)
    lifecycle = TileLifecycle()
    record = lifecycle.record(0)
    plane = np.array([1.0, 2.0])
    watched = Watched(plane, "initial")
    state = {"nested": watched, "shared": watched, "plane_alias": plane}
    state["cycle"] = state
    iterator = iter([1, 2])
    state["opaque_iterator"] = iterator
    object.__setattr__(session, "_mutations", MutationClock())
    object.__setattr__(session, "lifecycle", lifecycle)
    object.__setattr__(session, "unlisted_field", state)
    _snapshot(session)
    assert next(iterator) == 1, "snapshot consumed an opaque iterator"

    def rearrange_pixels():
        # Same shape, dtype and sum: the old nansum signature missed this.
        plane[:] = plane[::-1]

    for mutate in (
        lambda: setattr(watched, "reason", "unclocked nested write"),
        rearrange_pixels,
        lambda: object.__setattr__(record, "failed_reason", "unclocked ledger write"),
        lambda: object.__setattr__(session, "future_field", "unclocked new field"),
    ):
        assert _observe_mutation(session, lambda: None) == (False, False)
        assert _observe_mutation(session, mutate) == (False, True)


def test_reflected_walk_discovers_a_new_mutator_and_rejects_unknown_arguments(monkeypatch):
    def future_mutator(self, previously_unknown_parameter):
        raise AssertionError("an unsupplied mutator must not be invoked")

    monkeypatch.setattr(FrameSession, "future_mutator", future_mutator, raising=False)

    assert "future_mutator" in set(_public_methods())
    args, kwargs, unknown = _call_arguments("future_mutator", future_mutator, {})
    assert args == []
    assert kwargs == {}
    assert unknown == ["previously_unknown_parameter"]


@pytest.fixture(scope="module")
def _walk_session(qtbot_module_session):
    return qtbot_module_session


def test_every_public_mutator_moves_the_mutation_revision(qtbot):
    """One live session, every public method, one assertion each.

    Written as a single test rather than a parametrization because building a
    settled montage window per method would cost ~77 window builds; the walk
    re-snapshots between methods, so each method is still adjudicated alone.
    """

    win, session = _build(qtbot)
    table = _arguments(win, session)
    unknown_by_method: dict[str, list[str]] = {}
    missed: list[str] = []
    moved: list[str] = []

    for name in sorted(_public_methods()):
        args, kwargs, unknown = _call_arguments(name, getattr(FrameSession, name), table)
        if unknown:
            unknown_by_method[name] = unknown
            continue
        revision_moved, state_changed = _observe_mutation(
            session, getattr(session, name), *args, **kwargs
        )
        if revision_moved:
            moved.append(name)
        elif state_changed:
            missed.append(name)

    assert not unknown_by_method, (
        f"these FrameSession methods take parameters this walk cannot supply: "
        f"{unknown_by_method}. Add plausible values to ``_arguments`` in this "
        "file -- a method the walk cannot call is a method whose mutations "
        "nothing checks, and a missed mutation_revision bump is a permanent "
        "lost wakeup."
    )
    assert not missed, (
        f"{missed} changed session state without moving mutation_revision. A "
        "retarget gate keyed on that epoch would skip the replan those "
        "mutations are owed, and nothing else comes back for it."
    )
    assert len(moved) >= 25, (
        f"only {len(moved)} of the {len(list(_public_methods()))} walked methods "
        f"mutated anything ({sorted(moved)}); the settled montage no longer puts "
        "the session in a state where its methods bite, so this file has stopped "
        "testing."
    )


def test_a_declared_container_field_write_outside_the_session_bumps(qtbot):
    """The containers are handed out live, so their writes must count too."""

    _win, session = _build(qtbot)

    before = session.mutation_revision
    session.display_tile_payloads[999] = object()
    assert session.mutation_revision > before, "a dict field mutation was not counted"

    before = session.mutation_revision
    session.pending_removals.add(999)
    assert session.mutation_revision > before, "a set field mutation was not counted"

    before = session.mutation_revision
    session.dirty_payloads[999] = None
    assert session.mutation_revision > before, "an OrderedDict field mutation was not counted"

    before = session.mutation_revision
    session.presented_order.append((0, 0, 0, 0))
    assert session.mutation_revision > before, "a list field mutation was not counted"

    before = session.mutation_revision
    session.pending_level_tiles.appendleft(None)
    session.pending_level_tiles.popleft()
    assert session.mutation_revision > before, "a deque field mutation was not counted"


class _Different:
    """A value no session field can already be holding.

    The walk below has to change a field whose type it does not know, so it
    stores an object that is by construction not the one already there and
    puts the original back afterwards.  ``__setattr__`` never inspects the
    value beyond its exact type, so an alien object exercises the same path a
    real one would.
    """

    __slots__ = ()


def test_a_changing_scalar_field_write_bumps(qtbot):
    """Including a field that did not exist when this test was written."""

    _win, session = _build(qtbot)
    for name in (item.name for item in dataclass_fields(FrameSession)):
        original = getattr(session, name)
        before = session.mutation_revision
        setattr(session, name, _Different())
        # Restored BEFORE the assertion, and without a second bump.  The epoch
        # is the sum of the session's counter and its ledger's, so while
        # ``lifecycle`` holds the sentinel the property cannot even be read;
        # and putting it back through ``setattr`` would fold the sentinel's
        # (absent) reading into the session counter.  Counters only rise, so
        # reading after the restore still measures the write above.
        object.__setattr__(session, name, original)
        assert session.mutation_revision > before, (
            f"assigning FrameSession.{name} did not move mutation_revision; "
            "a write the epoch cannot see is a lost wakeup"
        )


def test_a_field_write_of_the_object_already_held_does_not_bump(qtbot):
    """The no-op rule, stated over the same reflected field walk.

    The counterpart of the test above and the reason it had to change: this
    walk is exactly what it used to do, and the expectation is inverted.
    """

    _win, session = _build(qtbot)
    for name in (item.name for item in dataclass_fields(FrameSession)):
        before = session.mutation_revision
        setattr(session, name, getattr(session, name))
        assert session.mutation_revision == before, (
            f"storing the object FrameSession.{name} already holds moved "
            "mutation_revision; the graph is identical, so there is nothing "
            "for a consumer of the epoch to notice"
        )


def test_a_wholesale_container_replacement_stays_tracked(qtbot):
    """A plain dict assigned to a declared field must be tracked from then on."""

    _win, session = _build(qtbot)
    session.tile_source_ids = {}
    before = session.mutation_revision
    session.tile_source_ids[7] = ("source", 7)
    assert session.mutation_revision > before, (
        "a container assigned after construction was not tracked, so its "
        "later mutations are invisible to the epoch"
    )


def test_owned_state_objects_bump_through_their_own_writes(qtbot):
    """Session-owned mutable sub-objects share the session's clock."""

    _win, session = _build(qtbot)

    before = session.mutation_revision
    session.scheduling_policy._phase = _Different()
    assert session.mutation_revision > before, "ProgressiveSchedulingPolicy is not clocked"

    before = session.mutation_revision
    session.round_load_shedding.rounds_shed += 1
    assert session.mutation_revision > before, "RoundLoadShedding is not clocked"

    before = session.mutation_revision
    session.level_generation.tile_revisions[999] = 1
    assert session.mutation_revision > before, "PresentationGenerationTracker is not clocked"

    before = session.mutation_revision
    session.stage_fan_in.values[("stage", 999)] = object()
    assert session.mutation_revision > before, "StageFanInState is not clocked"


def test_the_ledgers_epoch_is_a_term_of_the_sessions(qtbot):
    """A ledger mutation with no session write must still move the epoch."""

    _win, session = _build(qtbot)
    before = session.mutation_revision
    session.lifecycle.record(0).semantic_reason = "walked"
    assert session.mutation_revision > before


def test_replacing_the_ledger_keeps_the_epoch_monotonic(qtbot):
    """The epoch is a SUM, so a ledger swap must not walk it backwards.

    A consumer compares two readings; one that can decrease would compare
    equal across a swap and skip the replan a whole new ledger is owed.
    """

    from arrayscope.presentation import TileLifecycle

    _win, session = _build(qtbot)
    before = session.mutation_revision
    assert before > 0, "the built session mutated nothing, so this proves nothing"
    session.lifecycle = TileLifecycle(trace_session_id=99)
    assert session.mutation_revision > before, (
        "a fresh ledger's epoch starts at zero, so the session must absorb the "
        "outgoing ledger's reading or the sum goes backwards"
    )


def test_every_mutable_container_a_live_session_holds_is_tracked(qtbot):
    """The residual, pinned: ad-hoc attributes are counted on ASSIGNMENT only.

    ``__setattr__`` replaces a plain container assigned to a DECLARED field
    with a tracked one, and deliberately does not do that for an ad-hoc
    attribute -- an ad-hoc attribute may be a generation token compared by
    identity (``_atomic_warm_job``), and replacing it stalls the montage.

    So every ad-hoc attribute that holds a mutable container is a member whose
    in-place mutations the epoch does NOT count.  Each one below is classified
    with the reason it is safe; an attribute that is not classified fails this
    test, which is what stops the residual growing silently.
    """

    _win, session = _build(qtbot)
    declared = {item.name for item in dataclass_fields(FrameSession)}
    tracked = tuple(TRACKED_CONTAINER_TYPES.values())

    untracked_declared = sorted(
        name
        for name, value in session.__dict__.items()
        if name in declared
        and type(value) in TRACKED_CONTAINER_TYPES
        and not isinstance(value, tracked)
    )
    assert not untracked_declared, (
        f"declared fields {untracked_declared} hold plain containers, so their "
        "in-place mutations are invisible to the epoch"
    )

    #: Ad-hoc attributes allowed to hold an untracked container, and why.
    allowed_adhoc = {
        # Memos and caches: derived from state that is itself counted, so a
        # stale memo cannot make a plan believe there is nothing to do.
        "_lifecycle_payload_objects": "memo of the payload object last reported per slot",
        "_level_rehydrate_offsets": "round-robin cursor into the level cache",
        "_best_floor_key_memo": "memo over lifecycle levels, which are counted",
        "_lod_demand_sig_cache": "memo of the LOD demand signature",
        "_lod_resident_levels_memo": "memo over lifecycle levels, which are counted",
        "_montage_level_expected_latch": "latch over level_key, which is counted",
        "_required_tile_numbers_latch": "latch over the ledger revision it caches",
        "_published_lifecycle_targets": "mirror of the ledger's own targets",
        "_source_anchor_starts_memo": "memo of anchor starts for one crop",
        # In-flight job state.  Counted on assignment and on clear; while it is
        # live the work it describes is in flight, which is itself the reason a
        # gate must not skip.
        "_atomic_prepared_transaction": "prepared atomic handoff, assigned and cleared whole",
        "_atomic_warm_job": "identity-compared warm generation token",
        "_atomic_warmed_payloads": "warm bookkeeping for the job above",
        "_atomic_warmed_identities": "warm bookkeeping for the job above",
    }
    unclassified = sorted(
        name
        for name, value in session.__dict__.items()
        if name not in declared
        and isinstance(value, (dict, set, list))
        and not isinstance(value, tracked)
        and name not in allowed_adhoc
    )
    assert not unclassified, (
        f"ad-hoc session attributes {unclassified} hold untracked mutable "
        "containers. Their in-place mutations do NOT move mutation_revision. "
        "Either make the attribute a declared dataclass field (which tracks it) "
        "or classify it in ``allowed_adhoc`` above with the reason a consumer "
        "of the epoch cannot lose a wakeup to it."
    )


def test_the_tracked_container_types_classify_every_method_of_their_base():
    """A method a future Python adds must fail here, not go uncounted."""

    for base in CONTAINER_MUTATORS:
        public = {name for name in dir(base) if not name.startswith("_")}
        classified = {
            name
            for name in (CONTAINER_MUTATORS[base] | CONTAINER_READERS[base])
            if not name.startswith("_")
        }
        assert public == classified, (
            f"{base.__name__} has unclassified members {sorted(public - classified)} "
            f"and stale entries {sorted(classified - public)}. Every member must be "
            "declared a mutator (which generates a bumping override) or a reader."
        )
        tracked = TRACKED_CONTAINER_TYPES[base]
        for name in CONTAINER_MUTATORS[base]:
            assert name in vars(tracked), (
                f"{tracked.__name__} does not override {base.__name__}.{name}, "
                "so that mutation would not be counted"
            )


def test_the_owned_state_classes_carry_the_clock_hook():
    """The sub-objects a session owns must all be ``ClockedState``."""

    from arrayscope.display.model.presentation_generation import PresentationGenerationTracker
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.render.progressive_scheduling import (
        ProgressiveSchedulingPolicy,
        RoundLoadShedding,
    )
    from arrayscope.window.frame_session import SemanticLevelEvidenceProgress

    for klass in (
        PresentationGenerationTracker,
        StageFanInState,
        ProgressiveSchedulingPolicy,
        RoundLoadShedding,
        SemanticLevelEvidenceProgress,
    ):
        assert issubclass(klass, ClockedState), (
            f"{klass.__name__} is session-owned mutable state without the "
            "mutation-clock hook; its writes would not move the epoch"
        )
