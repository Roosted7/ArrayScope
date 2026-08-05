"""A store of the object already held is not a mutation -- and nothing more.

``core.mutation_clock`` counts changes to an object graph so a consumer can
ask "did anything happen since I last looked?".  It answers that question by
counting *calls*, which is the safe direction: over-counting costs a redundant
pass, under-counting costs a frame that never arrives.

That safety has a cost, and it was being paid at field scale.  Measured on a
settled 272-tile montage (``tests/render/test_commit_cost_scaling.py`` pins the
number): one commit that created nothing, updated nothing and uploaded nothing
moved ``FrameSession.mutation_revision`` by **2460**, and every one of those
2460 calls left the graph identical -- 1088 record attribute stores of the
object already there, 816 ``set.add``/``discard`` of members and non-members,
544 ``dict.pop`` of absent keys.  A genuine one-tile delta moved it by 2465.
Five parts signal in 2465 is not an epoch; ``viewport_invariant_key`` carries
that reading as ``lifecycle_revision`` and its epoch arm could not hold across
any commit at all.

So the guard.  It fires only where the no-op is decidable in O(1) *without
comparing state*, and this module is where that boundary is pinned, because
the boundary is what a future reader will get wrong:

* **identity, for slots.**  ``d[k] = v`` when ``d[k] is v``, and
  ``obj.attr = v`` when the attribute already holds that exact object.  A
  DIFFERENT object that merely compares equal still bumps and is still stored
  -- ``==`` is never called, and a slot that swaps one equal object for
  another has genuinely changed which object it holds.
* **membership, for sets.**  ``s.add(x)`` when ``x in s`` does not bump,
  ``s.discard(x)`` when it is absent does not bump.  Here equality really is
  the whole story: CPython's ``set.add`` of an equal member keeps the element
  already stored and changes nothing observable, so the guard matches the
  container's own semantics rather than overriding them.  This is the one
  place the two rules differ and it is not an inconsistency.
* **presence, for the rest.**  ``pop`` of an absent key, ``setdefault`` on a
  present one, ``clear()`` when already empty.

Every other mutator still fails open: ``update``, ``__ior__``,
``difference_update`` and friends can be no-ops too, and deciding that would
mean comparing state, which is the thing the counter must never do.

The direction this module does NOT own is completeness -- "every real mutation
still bumps".  That is asserted by reflection walks over the live product in
``tests/window/test_frame_session_mutation_revision.py`` and
``tests/presentation/test_tile_lifecycle_mutation_revision.py``, which do not
know any field or method names and so cannot rot around a new one.
"""

from __future__ import annotations

from collections import OrderedDict, deque

import pytest

from arrayscope.core.mutation_clock import (
    CONTAINER_MUTATORS,
    NOTHING_HELD,
    TRACKED_CONTAINER_TYPES,
    ClockedState,
    MutationClock,
    TrackedDeque,
    TrackedDict,
    TrackedList,
    TrackedOrderedDict,
    TrackedSet,
)


class _Equal:
    """Distinct objects that compare and hash equal.

    The whole point of the identity rule is that these are NOT interchangeable
    in a slot, and the whole point of the membership rule is that they ARE
    interchangeable in a set.  One class exercises both.
    """

    __slots__ = ("tag",)

    def __init__(self, tag: str = "same") -> None:
        self.tag = tag

    def __eq__(self, other) -> bool:
        return isinstance(other, _Equal)

    def __hash__(self) -> int:
        return hash("_Equal")

    def __repr__(self) -> str:
        return f"_Equal({self.tag!r})"


class _Owner(ClockedState):
    """The smallest thing that owns a clock."""

    def __init__(self, clock: MutationClock) -> None:
        object.__setattr__(self, "_mutation_clock", clock)


@pytest.fixture
def clock() -> MutationClock:
    return MutationClock()


# ------------------------------------------------------- the no-op direction


def test_the_empty_sentinel_is_one_object_that_names_itself(clock):
    """It stands for "absent", which ``None`` cannot: ``None`` is storable.

    The ``repr`` is the reason it is a class and not a bare ``object()`` --
    one of these reaching a traceback or a diagnostic dump has to say what it
    is, because by then something has stored a value that can never be read
    back as itself.
    """

    tracked = TrackedDict(clock)
    tracked["k"] = None
    before = clock.value
    tracked["k"] = None
    assert clock.value == before, "``None`` stored over ``None`` is still a no-op"
    assert tracked.get("absent", NOTHING_HELD) is NOTHING_HELD
    assert repr(NOTHING_HELD) == "<nothing held>"


def test_storing_the_object_a_key_already_holds_does_not_bump(clock):
    for tracked in (TrackedDict(clock), TrackedOrderedDict(clock)):
        value = object()
        tracked["k"] = value
        before = clock.value
        tracked["k"] = value
        assert clock.value == before, (
            f"{type(tracked).__name__} counted a store of the object the key "
            "already held; the mapping is identical afterwards"
        )
        assert tracked["k"] is value


def test_storing_the_object_an_attribute_already_holds_does_not_bump(clock):
    owner = _Owner(clock)
    value = object()
    owner.thing = value
    before = clock.value
    owner.thing = value
    assert clock.value == before
    assert owner.thing is value


def test_adding_a_member_and_discarding_a_non_member_do_not_bump(clock):
    tracked = TrackedSet(clock)
    tracked.add(7)
    before = clock.value
    tracked.add(7)
    tracked.discard(9)
    assert clock.value == before, (
        "a set that already contains the element, and one that never "
        "contained it, are both unchanged by these calls"
    )
    assert tracked == {7}


def test_popping_an_absent_key_and_setdefaulting_a_present_one_do_not_bump(clock):
    for tracked in (TrackedDict(clock), TrackedOrderedDict(clock)):
        tracked["k"] = 1
        before = clock.value
        assert tracked.pop("absent", "default") == "default"
        assert tracked.setdefault("k", 2) == 1
        assert clock.value == before
        assert dict(tracked) == {"k": 1}


def test_clearing_an_already_empty_container_does_not_bump(clock):
    for tracked in (
        TrackedDict(clock),
        TrackedOrderedDict(clock),
        TrackedSet(clock),
        TrackedList(clock),
        TrackedDeque(clock),
    ):
        before = clock.value
        tracked.clear()
        assert clock.value == before, f"{type(tracked).__name__}.clear() on an empty container"


# ----------------------------------------- the asymmetry a reader will trip on


def test_a_different_but_equal_object_in_a_slot_is_still_a_mutation(clock):
    """Identity, not equality -- and ``__eq__`` is never even called.

    This is the half most likely to surprise: two objects that compare equal
    are still two objects, and a slot that now holds the other one has
    changed.  A consumer that memoised anything derived from the FIRST object
    -- a payload ref, a residency key, an identity tuple -- would be reading a
    stale memo if this did not bump.
    """

    first, second = _Equal("first"), _Equal("second")
    assert first == second
    assert hash(first) == hash(second)

    for tracked in (TrackedDict(clock), TrackedOrderedDict(clock)):
        tracked["k"] = first
        before = clock.value
        tracked["k"] = second
        assert clock.value > before, (
            f"{type(tracked).__name__} treated a swap between two equal-but-"
            "distinct objects as a no-op; the key now holds a different object"
        )
        assert tracked["k"] is second

    owner = _Owner(clock)
    owner.thing = first
    before = clock.value
    owner.thing = second
    assert clock.value > before
    assert owner.thing is second


def test_adding_a_different_but_equal_member_to_a_set_is_not_a_mutation(clock):
    """The deliberate exception, and it is CPython's rule rather than ours.

    ``set.add`` of an element that compares equal to one already present keeps
    the element already stored -- it does not replace it.  So the set really is
    unchanged, and bumping would report a change that did not happen.
    """

    first, second = _Equal("first"), _Equal("second")
    tracked = TrackedSet(clock)
    tracked.add(first)
    before = clock.value
    tracked.add(second)
    assert clock.value == before
    assert next(iter(tracked)) is first, (
        "CPython changed set.add semantics; if an equal element now REPLACES "
        "the stored one this guard is wrong and must bump"
    )


# ------------------------------------------------- the guard is not too greedy


@pytest.mark.parametrize(
    ("make", "mutate"),
    [
        pytest.param(TrackedDict, lambda c: c.__setitem__("k", object()), id="dict-setitem"),
        pytest.param(TrackedDict, lambda c: (c.update({"k": 1}), c.pop("k")), id="dict-pop"),
        pytest.param(TrackedDict, lambda c: c.setdefault("fresh", 1), id="dict-setdefault-absent"),
        pytest.param(TrackedDict, lambda c: (c.update({"k": 1}), c.clear()), id="dict-clear"),
        pytest.param(
            TrackedOrderedDict, lambda c: c.__setitem__("k", object()), id="odict-setitem"
        ),
        pytest.param(TrackedSet, lambda c: c.add(object()), id="set-add"),
        pytest.param(TrackedSet, lambda c: (c.add(1), c.discard(1)), id="set-discard"),
        pytest.param(TrackedSet, lambda c: (c.add(1), c.clear()), id="set-clear"),
        pytest.param(TrackedList, lambda c: c.append(1), id="list-append"),
        pytest.param(TrackedDeque, lambda c: c.append(1), id="deque-append"),
    ],
)
def test_a_guarded_operation_that_changes_something_still_bumps(clock, make, mutate):
    """Every branch the guards touch, driven through its changing case."""

    container = make(clock)
    before = clock.value
    mutate(container)
    assert clock.value > before, "a real mutation went uncounted"


def test_the_guards_did_not_drop_any_mutator_override():
    """The specialization must not shrink the set of overridden mutators.

    Adding the ``pop``/``setdefault``/``clear`` guards meant filtering the
    specialization table by base, and a name filtered out of BOTH the
    specialization and the generated namespace would silently stop counting.
    """

    for base, names in CONTAINER_MUTATORS.items():
        tracked = TRACKED_CONTAINER_TYPES[base]
        missing = sorted(name for name in names if name not in vars(tracked))
        assert not missing, f"{tracked.__name__} stopped overriding {missing}"


@pytest.mark.parametrize(
    ("base", "tracked"),
    [(dict, TrackedDict), (OrderedDict, TrackedOrderedDict), (set, TrackedSet)],
)
def test_a_guarded_mutator_keeps_its_bases_return_value(base, tracked, clock):
    """The guards wrap calls; they must not change what a caller receives."""

    container = tracked(clock)
    plain = base()
    if base is set:
        assert container.add(1) == plain.add(1)
        assert container.discard(1) == plain.discard(1)
        return
    container["k"] = 1
    plain["k"] = 1
    assert container.pop("k") == plain.pop("k")
    assert container.pop("gone", "d") == plain.pop("gone", "d")
    assert container.setdefault("k", 2) == plain.setdefault("k", 2)
    assert container.setdefault("k", 3) == plain.setdefault("k", 3)
    with pytest.raises(KeyError):
        container.pop("gone")


def test_a_list_index_write_still_fails_open(clock):
    """Slices make "what is already there" ambiguous, so lists do not guard.

    Pinned so that a future attempt to extend the identity rule to sequences
    has to confront the slice case rather than discover it in production.
    """

    for tracked in (TrackedList(clock, [1, 2, 3]), TrackedDeque(clock, deque([1, 2, 3]))):
        before = clock.value
        tracked[0] = tracked[0]
        assert clock.value > before, f"{type(tracked).__name__} guarded an index write"
