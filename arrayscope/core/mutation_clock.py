"""One monotonic counter for "did anything in this object graph change?".

A gate that skips work when a revision has not moved turns a **missed bump**
into a lost wakeup, and in the render path there is no backstop for one
(``_montage_watchdog_tick`` observes and emits evidence; it never repairs).
So the bump may not be installed on a hand-maintained list of mutating
methods: the next method anyone adds would silently not bump.

It is installed on the *containers* instead.  Every attribute write goes
through a ``__setattr__`` that bumps; every mutable collection an owner holds
is a tracked subclass whose mutators bump; and the tracked subclasses are
GENERATED from :data:`CONTAINER_MUTATORS`, whose complement
:data:`CONTAINER_READERS` a test requires to cover ``dir(base)`` exactly.  So
a new field, a new collection, or a mutator added by a future Python bumps --
or fails a test -- **by construction**, whatever it is called.

One thing is deliberately NOT by construction.  Replacing a plain container
with a tracked one on assignment makes a COPY, and a copy is a different
object, which breaks two idioms silently:

    x = {}; obj.attr = x; x[k] = v            # writes land in the discard
    obj.attr = job; ...; if obj.attr is job:  # generation guard, never true

Both are live in this codebase (four of the first, and ``_atomic_warm_job``
of the second -- it stalled ten montage tests).  So replacement is confined to
attributes an owner DECLARES, whose assignment sites are bounded and
auditable; see :func:`track_state_object` for the other half and the owning
class for what that leaves as a residual.

The counter fails its callers *open*: a mutation that writes back an
identical value still bumps.  Equal values mean no mutation was attempted,
never that two states compare equal.  Over-bumping costs one redundant pass;
under-bumping costs a frame that never arrives.
"""

from __future__ import annotations

from collections import OrderedDict, deque

__all__ = [
    "CONTAINER_MUTATORS",
    "CONTAINER_READERS",
    "IMMUTABLE_VALUE_TYPES",
    "NULL_CLOCK",
    "TRACKED_CONTAINER_TYPES",
    "ClockedState",
    "MutationClock",
    "TrackedDeque",
    "TrackedDict",
    "TrackedList",
    "TrackedOrderedDict",
    "TrackedSet",
    "attach_mutation_clock",
    "bind_existing_state",
    "track_mutable",
    "track_state_object",
]


class MutationClock:
    """One monotonic counter shared by an owner and everything it owns."""

    __slots__ = ("value",)

    def __init__(self, value: int = 0) -> None:
        self.value = int(value)

    def __repr__(self) -> str:
        # Deliberately value-free.  Differential oracles dump owned state
        # field by field via ``repr``; the clock is bookkeeping, not observable
        # state, and leaking its identity or its count into those dumps would
        # make two owners that agree on every field compare unequal.
        return "<mutation clock>"


#: Shared by anything built before its owner attached a real clock.  Such an
#: object never reaches a gate, so its mutations need not be counted.
NULL_CLOCK = MutationClock()


#: Every public method of each tracked base type, classified as one that can
#: change the container's contents or one that cannot.  The mutator half is
#: what the tracked subclasses are GENERATED from, and a test asserts the two
#: halves together cover ``dir(base)`` exactly -- so a method added by a future
#: Python is an unclassified name and a test failure, not a silent hole.
CONTAINER_MUTATORS: dict[type, frozenset[str]] = {
    dict: frozenset(
        {"clear", "pop", "popitem", "setdefault", "update", "__setitem__", "__delitem__", "__ior__"}
    ),
    set: frozenset(
        {
            "add",
            "clear",
            "difference_update",
            "discard",
            "intersection_update",
            "pop",
            "remove",
            "symmetric_difference_update",
            "update",
            "__iand__",
            "__ior__",
            "__isub__",
            "__ixor__",
        }
    ),
    list: frozenset(
        {
            "append",
            "clear",
            "extend",
            "insert",
            "pop",
            "remove",
            "reverse",
            "sort",
            "__setitem__",
            "__delitem__",
            "__iadd__",
            "__imul__",
        }
    ),
    deque: frozenset(
        {
            "append",
            "appendleft",
            "clear",
            "extend",
            "extendleft",
            "insert",
            "pop",
            "popleft",
            "remove",
            "reverse",
            "rotate",
            "__setitem__",
            "__delitem__",
            "__iadd__",
            "__imul__",
        }
    ),
    OrderedDict: frozenset(
        {
            "clear",
            "move_to_end",
            "pop",
            "popitem",
            "setdefault",
            "update",
            "__setitem__",
            "__delitem__",
            "__ior__",
        }
    ),
}

#: The other half.  ``copy`` and ``fromkeys`` return a *new* plain container:
#: it is not owned state until something assigns it, and assignment tracks it.
CONTAINER_READERS: dict[type, frozenset[str]] = {
    dict: frozenset({"copy", "fromkeys", "get", "items", "keys", "values"}),
    set: frozenset(
        {
            "copy",
            "difference",
            "intersection",
            "isdisjoint",
            "issubset",
            "issuperset",
            "symmetric_difference",
            "union",
        }
    ),
    list: frozenset({"copy", "count", "index"}),
    deque: frozenset({"copy", "count", "index", "maxlen"}),
    OrderedDict: frozenset({"copy", "fromkeys", "get", "items", "keys", "values"}),
}


def _bumping_namespace(base: type) -> dict[str, object]:
    """Override every mutator of ``base`` with one that bumps first."""

    namespace: dict[str, object] = {}
    for name in sorted(CONTAINER_MUTATORS[base]):
        inherited = getattr(base, name)

        def _make(inherited=inherited):
            def _bump(self, *args, **kwargs):
                self._clock.value += 1
                return inherited(self, *args, **kwargs)

            return _bump

        namespace[name] = _make()
    return namespace


#: Hand-written overrides for the mutators that run per tile per pass, where
#: the generated ``*args`` wrapper's packing is a measurable share of the cost
#: (dict ``__setitem__`` into a 272-entry dict: 15 ns plain, 228 ns through
#: the generated wrapper, 114 ns through the specialization below).
#: They are an optimisation ONLY: the generated wrapper is correct, the guard
#: test still requires every name in :data:`CONTAINER_MUTATORS` to be
#: overridden, and a name here that is not a mutator changes nothing.
def _specialized(base: type) -> dict[str, object]:
    def __setitem__(self, key, value) -> None:
        self._clock.value += 1
        base.__setitem__(self, key, value)

    def __delitem__(self, key) -> None:
        self._clock.value += 1
        base.__delitem__(self, key)

    def add(self, value) -> None:
        self._clock.value += 1
        base.add(self, value)

    def discard(self, value) -> None:
        self._clock.value += 1
        base.discard(self, value)

    def append(self, value) -> None:
        self._clock.value += 1
        base.append(self, value)

    return {
        name: member
        for name, member in (
            ("__setitem__", __setitem__),
            ("__delitem__", __delitem__),
            ("add", add),
            ("discard", discard),
            ("append", append),
        )
        if name in CONTAINER_MUTATORS[base]
    }


def _tracked_container(base: type, name: str) -> type:
    namespace = _bumping_namespace(base)
    namespace.update(_specialized(base))
    namespace["__slots__"] = ("_clock",)
    namespace["__doc__"] = f"A {base.__name__} that bumps a MutationClock on every mutation."

    def __init__(self, clock: MutationClock, data=()) -> None:
        # The clock FIRST: ``OrderedDict.__init__`` seeds through
        # ``__setitem__``, which is one of the bumping overrides, so a
        # clock bound afterwards would raise on any non-empty seed.
        self._clock = clock
        base.__init__(self, data)

    def __repr__(self) -> str:
        # Indistinguishable from the plain container it replaces, so every
        # existing diagnostic dump and repr-based oracle reads unchanged.
        return base.__repr__(self)

    namespace["__init__"] = __init__
    namespace["__repr__"] = __repr__
    return type(name, (base,), namespace)


TrackedDict = _tracked_container(dict, "TrackedDict")
TrackedSet = _tracked_container(set, "TrackedSet")
TrackedList = _tracked_container(list, "TrackedList")
TrackedDeque = _tracked_container(deque, "TrackedDeque")
TrackedOrderedDict = _tracked_container(OrderedDict, "TrackedOrderedDict")

#: Exact type -> tracked replacement.  The lookup is by EXACT type, never by
#: ``isinstance``: a subclass (``LifecycleRenderedTiles``, the lifecycle set
#: views) carries behaviour of its own and reports through its own owner.
TRACKED_CONTAINER_TYPES: dict[type, type] = {
    dict: TrackedDict,
    set: TrackedSet,
    list: TrackedList,
    deque: TrackedDeque,
    OrderedDict: TrackedOrderedDict,
}

#: Exact value types that cannot hold mutable state, so a write of one needs
#: the bump and nothing else.  A SPEED short-circuit on the hottest attribute
#: write paths in the application, never a completeness decision: anything not
#: listed falls through to :func:`track_mutable`, which is what decides.
#: Adding a type here that can be mutated in place would be the lost wakeup;
#: every entry is immutable in CPython.
IMMUTABLE_VALUE_TYPES = frozenset(
    {bool, int, float, complex, str, bytes, tuple, frozenset, type(None)}
)

_TRACKED_CONTAINERS = (
    TrackedDict,
    TrackedSet,
    TrackedList,
    TrackedDeque,
    TrackedOrderedDict,
)


class ClockedState:
    """Mixin for a mutable object owned by something that keeps an epoch.

    Every attribute write bumps, including one to an attribute added after
    this was written, and every mutable collection the write installs is
    tracked.  Subclasses need no per-field registration -- that is the whole
    point: a registration list is the thing that rots.
    """

    __slots__ = ()

    def __setattr__(self, name, value) -> None:
        clock = self.__dict__.get("_mutation_clock")
        if clock is None or name == "_mutation_clock":
            object.__setattr__(self, name, value)
            return
        clock.value += 1
        if type(value) not in IMMUTABLE_VALUE_TYPES:
            value = track_mutable(value, clock)
        object.__setattr__(self, name, value)


def track_mutable(value, clock: MutationClock):
    """Bind ``value`` to ``clock`` if it is mutable state worth counting."""

    tracked = TRACKED_CONTAINER_TYPES.get(type(value))
    if tracked is not None:
        return tracked(clock, value)
    if isinstance(value, _TRACKED_CONTAINERS):
        # Already tracked -- re-point it at its new owner.  The previous owner
        # is a superseded session whose gate nobody consults again.
        value._clock = clock
        return value
    if isinstance(value, ClockedState):
        return attach_mutation_clock(value, clock)
    return value


def track_state_object(value, clock: MutationClock):
    """Bind ``value`` to ``clock`` WITHOUT ever replacing a container.

    ``track_mutable`` replaces a plain ``dict``/``set``/``list`` with a tracked
    copy, and a copy is a different object.  Two idioms in this codebase break
    on that, both silently:

    * ``x = {}; obj.attr = x; x[k] = v`` -- the writes land in the discarded
      original (four sites, all fixed to read the attribute back);
    * ``obj.attr = job; ...; if obj.attr is not job: return`` -- a generation
      guard that becomes permanently true, which stalled ten montage tests
      before this split existed (``_atomic_warm_job``).

    So replacement is confined to the attributes an owner DECLARES, whose
    assignment sites are bounded and auditable.  Ad-hoc attributes get the
    write counted and their nested state objects bound, but a container they
    hold keeps its identity and its contents are not counted.
    """

    if isinstance(value, ClockedState):
        return attach_mutation_clock(value, clock)
    return value


def bind_existing_state(obj, clock: MutationClock, *, skip=()) -> None:
    """Track everything already in ``obj.__dict__``.

    Used once, when an owner adopts a clock after its attributes were bound
    (a dataclass ``__init__`` runs before ``__post_init__``).
    """

    for name, value in tuple(obj.__dict__.items()):
        if name in skip or name == "_mutation_clock":
            continue
        bound = track_mutable(value, clock)
        if bound is not value:
            object.__setattr__(obj, name, bound)


def attach_mutation_clock(obj, clock: MutationClock):
    """Give ``obj`` (and everything it already holds) ``clock``."""

    object.__setattr__(obj, "_mutation_clock", clock)
    bind_existing_state(obj, clock)
    return obj
