"""``TileLifecycle.mutation_revision`` must move for every mutation there is.

A retarget gate that skips planning when the revision has not moved turns a
missed bump into a **lost wakeup**, and ``_montage_watchdog_tick`` observes
without repairing, so a lost wakeup here is permanent rather than late.  The
only defence that survives the next person adding a method is a test that
does not know the method names.

So this walks the class by reflection.  For every public callable it takes a
deep, field-reflected snapshot of the ledger, invokes the method with
plausible arguments, snapshots again, and asserts:

    the observable state changed  =>  ``mutation_revision`` moved

Two properties make it un-rottable:

* the snapshot enumerates ``dataclasses.fields(TileRecord)`` and the ledger's
  own ``__dict__``, so a **new field** is compared without being named here;
* a method whose parameters this file cannot supply is a **failure**, not a
  skip.  A new method with an unfamiliar parameter forces its author to teach
  ``_ARGUMENTS`` what a plausible value is, and while they do that the walk
  covers their method.

The converse direction (the revision moving when nothing changed) is
deliberately NOT asserted: the counter fails callers *open*, and a spurious
bump costs one replan, while a missing one costs a frame that never arrives.
"""

from __future__ import annotations

import inspect
from dataclasses import fields as dataclass_fields

import pytest

from arrayscope.presentation import (
    ClaimOwner,
    LevelPhase,
    Semantic,
    TileLifecycle,
    TilePayloadRef,
    TileTarget,
)
from arrayscope.presentation.tile_lifecycle import TileRecord

TILE = 3
SOURCE_INDEX = 11
SOURCE_ID = ("semantic-source", SOURCE_INDEX)
STAGE_KEY = ("stage", 1)
PRODUCER_KEY = ("producer", 1)
LEVEL_KEY = ("level", 0)


def _target(level: int = 0) -> TileTarget:
    return TileTarget(
        tile_number=TILE,
        source_index=SOURCE_INDEX,
        semantic_source_id=SOURCE_ID,
        lod_level=level,
        visible=True,
    )


def _payload() -> TilePayloadRef:
    return TilePayloadRef(
        source_id=SOURCE_ID,
        quality="exact",
        lod_level=0,
        source_index=SOURCE_INDEX,
    )


class _Request:
    """A materialization request shaped like the ones the ladder submits."""

    def __init__(self) -> None:
        self.chain = ((TILE, LEVEL_KEY),)
        self.tile_number = TILE
        self.level_key = LEVEL_KEY


_REQUEST = _Request()

#: Plausible values BY PARAMETER NAME, so a new method that reuses the
#: ledger's existing vocabulary is covered without touching this file.  An
#: unknown name fails the walk on purpose (see the module docstring).
_ARGUMENTS: dict[str, object] = {
    "tile_number": TILE,
    "tile_numbers": (TILE,),
    "targets": {TILE: _target()},
    "payload": _payload(),
    "upserts": {TILE: _payload()},
    "identities": {TILE: SOURCE_ID},
    "acknowledged": {TILE: SOURCE_ID},
    "accepted_payloads": {TILE: _payload()},
    "backend_payloads": {TILE: _payload()},
    "emitted_tiles": (TILE,),
    "accepted_tiles": (TILE,),
    "removed_tiles": (),
    "payload_identity": SOURCE_ID,
    "semantic_key": SOURCE_ID,
    "task_key": ("task", 1),
    "source_id": SOURCE_ID,
    "session_id": 7,
    "stage_key": STAGE_KEY,
    "producer_key": PRODUCER_KEY,
    "bindings": {STAGE_KEY: (TILE,)},
    "level_key": LEVEL_KEY,
    "owner": ClaimOwner.EVALUATION,
    "request": _REQUEST,
    "claim": ("task", 1),
    "rung": 0,
    "level": 0,
    "active_scope": (TILE,),
    "memory_pressure": False,
    "reason": "test",
    "presented": True,
    "quality": "exact",
    "lod_level": 0,
    "identity": SOURCE_ID,
    "key": LEVEL_KEY,
    "value": 1,
}


def _snapshot(lifecycle: TileLifecycle):
    """Everything the ledger can be asked about, by reflection."""

    record_fields = tuple(
        field.name for field in dataclass_fields(TileRecord) if field.name != "mutation_clock"
    )
    records = tuple(
        (
            number,
            tuple(
                (name, _value_repr(getattr(record, name, None))) for name in sorted(record_fields)
            ),
        )
        for number, record in sorted(lifecycle._records.items())
    )
    own = tuple(
        (name, _value_repr(value))
        for name, value in sorted(lifecycle.__dict__.items())
        # The clock is the thing under test, and ``_records`` is expanded above.
        if name not in {"_mutations", "_records"}
    )
    return (records, own)


def _value_repr(value):
    if isinstance(value, dict):
        return tuple(sorted((repr(key), _value_repr(item)) for key, item in value.items()))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(repr(item) for item in value))
    return repr(value)


def _seeded() -> TileLifecycle:
    """A ledger with one tile in a mid-flight state, so most methods bite."""

    lifecycle = TileLifecycle(trace_session_id=1)
    lifecycle.retarget({TILE: _target()})
    lifecycle.plan_applied([TILE])
    lifecycle.evaluation_requested(TILE)
    lifecycle.evaluation_claimed(TILE, ("task", 1))
    lifecycle.evaluation_started(TILE)
    lifecycle.task_admitted(TILE, ("task", 1), stage_key=STAGE_KEY)
    lifecycle.stage_attached(TILE, STAGE_KEY)
    lifecycle.level_claimed(TILE, LEVEL_KEY, ClaimOwner.EVALUATION, request=_REQUEST)
    lifecycle.load_marked(TILE)
    lifecycle.remember_presentable(TILE, _payload())
    lifecycle.fallback_ready(TILE, _payload())
    return lifecycle


def _public_methods():
    for name, member in sorted(vars(TileLifecycle).items()):
        if name.startswith("_"):
            continue
        if isinstance(member, property) or not callable(member):
            continue
        yield name


def _call_arguments(method):
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
        if name not in _ARGUMENTS:
            if parameter.default is not parameter.empty:
                continue
            unknown.append(name)
            continue
        if parameter.kind is parameter.KEYWORD_ONLY:
            kwargs[name] = _ARGUMENTS[name]
        else:
            args.append(_ARGUMENTS[name])
    return args, kwargs, unknown


@pytest.mark.parametrize("name", sorted(_public_methods()))
def test_every_public_mutator_moves_the_mutation_revision(name):
    lifecycle = _seeded()
    method = getattr(TileLifecycle, name)
    args, kwargs, unknown = _call_arguments(method)
    assert not unknown, (
        f"TileLifecycle.{name} takes {unknown}, which this walk cannot supply. "
        "Add a plausible value to _ARGUMENTS in this file -- a method the walk "
        "cannot call is a method whose mutations nothing checks, and a missed "
        "mutation_revision bump is a permanent lost wakeup."
    )

    before_state = _snapshot(lifecycle)
    before_revision = lifecycle.mutation_revision
    getattr(lifecycle, name)(*args, **kwargs)
    after_state = _snapshot(lifecycle)
    after_revision = lifecycle.mutation_revision

    if after_state == before_state:
        return
    assert after_revision > before_revision, (
        f"TileLifecycle.{name} changed ledger state without moving "
        "mutation_revision. A retarget gate keyed on that revision would skip "
        "the replan this mutation is owed, and nothing else comes back for it."
    )


def test_the_walk_actually_reaches_the_mutators():
    """Guard the guard: a walk that calls nothing proves nothing."""

    moved = []
    for name in _public_methods():
        lifecycle = _seeded()
        args, kwargs, unknown = _call_arguments(getattr(TileLifecycle, name))
        if unknown:
            continue
        before = lifecycle.mutation_revision
        getattr(lifecycle, name)(*args, **kwargs)
        if lifecycle.mutation_revision > before:
            moved.append(name)
    assert len(moved) >= 20, (
        f"only {len(moved)} of the walked methods mutated anything "
        f"({sorted(moved)}); the seeded ledger no longer puts the machine in a "
        "state where its methods bite, so this file has stopped testing."
    )


def test_a_record_field_write_outside_the_ledger_still_bumps():
    """Records are handed out by ``record()``; a caller write must count."""

    lifecycle = _seeded()
    before = lifecycle.mutation_revision
    lifecycle.record(TILE).semantic = Semantic.DECLINED
    assert lifecycle.mutation_revision > before


def test_a_record_collection_mutation_still_bumps():
    """The collections are handed out live, so their writes must count too."""

    lifecycle = _seeded()
    before = lifecycle.mutation_revision
    lifecycle.record(TILE).presentable_payloads["extra"] = object()
    assert lifecycle.mutation_revision > before

    before = lifecycle.mutation_revision
    lifecycle.record(TILE).preview_claims[0] = (0, SOURCE_ID)
    assert lifecycle.mutation_revision > before

    before = lifecycle.mutation_revision
    lifecycle.record(TILE).resigned.add(("a", "b"))
    assert lifecycle.mutation_revision > before


def test_a_level_entry_phase_write_still_bumps():
    """``entry.phase = RESIDENT`` is a mutation no container hook can see.

    Written against the entry directly rather than through
    ``level_materializing``: that method also bumps ``_level_revision``, so
    routing through it would pass with the entry's own hook removed.
    """

    lifecycle = _seeded()
    entry = lifecycle.record(TILE).levels[LEVEL_KEY]
    before = lifecycle.mutation_revision
    entry.phase = LevelPhase.RESIDENT
    assert lifecycle.mutation_revision > before


def test_a_wholesale_collection_replacement_stays_tracked():
    """``retarget`` rebuilds ``presentable_payloads``; the new one must track."""

    lifecycle = _seeded()
    lifecycle.retarget(
        {
            TILE: TileTarget(
                tile_number=TILE,
                source_index=SOURCE_INDEX + 1,
                semantic_source_id=("semantic-source", SOURCE_INDEX + 1),
                lod_level=0,
                visible=True,
            )
        }
    )
    before = lifecycle.mutation_revision
    lifecycle.record(TILE).presentable_payloads["after-replacement"] = object()
    assert lifecycle.mutation_revision > before
