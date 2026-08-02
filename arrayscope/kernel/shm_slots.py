"""A bounded pool of uniform shared-memory slots for cross-process payloads.

One segment, ``slots`` fixed-size slices of it, and a strict per-slot state
machine: ``FREE -> LEASED (exactly one writer) -> READY -> HELD (one consumer)
-> FREE``. A slot is immutable from READY until it is released, so publication
can be a descriptor small enough to pickle down a pipe.

The state array is shared, so a worker process leases and publishes for itself;
only the *segment* has an owner, and that owner is always the GUI process — it
creates and unlinks, children attach and close. Exhaustion is not an error:
``lease`` returns ``None``, the caller pickles the payload instead, and the
pool counts the fallback.
"""

from __future__ import annotations

import multiprocessing
from contextlib import suppress
from dataclasses import dataclass, fields, is_dataclass, replace
from multiprocessing import shared_memory
from typing import Any

import numpy as np

DEFAULT_SLOT_BYTES = 1 << 20
DEFAULT_SLOT_COUNT = 64
# Below this an ndarray is cheaper to pickle than to lease, write, describe,
# and release; the pilot families' results (bin counts, bounded samples) all
# sit under it and never touch a slot.
SLOT_PAYLOAD_MIN_BYTES = 64 * 1024

FREE = 0
LEASED = 1
READY = 2
HELD = 3

_STATE_NAMES = {FREE: "free", LEASED: "leased", READY: "ready", HELD: "held"}


@dataclass(frozen=True)
class SlotDescriptor:
    """What travels over the pipe in place of the array itself."""

    index: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class SlotPoolHandle:
    """Everything a spawned child needs to attach. Passed as a process arg."""

    name: str
    slot_bytes: int
    slots: int
    states: Any
    owners: Any
    fallbacks: Any


class SlotLease:
    """A writer's exclusive hold on one slot, from LEASED to READY."""

    __slots__ = ("_pool", "_view", "capacity", "index")

    def __init__(self, pool: SlotPool, index: int) -> None:
        self._pool = pool
        self.index = int(index)
        self.capacity = int(pool.slot_bytes)
        self._view = None

    def view(self, shape, dtype) -> np.ndarray:
        """A writable numpy view over the slot, for producing in place."""

        array = self._pool._slot_array(self.index, shape, dtype, writable=True)
        self._view = array
        return array

    def publish(self, array: np.ndarray) -> SlotDescriptor:
        """Freeze the slot at READY and describe it. Copies unless in place."""

        array = np.ascontiguousarray(array)
        if array.nbytes > self.capacity:
            raise ValueError("payload exceeds the slot size")
        if array is not self._view:
            target = self._pool._slot_array(self.index, array.shape, array.dtype, writable=True)
            np.copyto(target, array)
        self._pool._advance(self.index, LEASED, READY)
        return SlotDescriptor(
            index=self.index,
            nbytes=int(array.nbytes),
            shape=tuple(int(size) for size in array.shape),
            dtype=array.dtype.str,
        )

    def release(self) -> None:
        self._view = None
        self._pool.release(self.index)


class SlotPool:
    """The slot allocator. One instance per process; one segment per pool."""

    def __init__(
        self,
        *,
        slot_bytes: int = DEFAULT_SLOT_BYTES,
        slots: int = DEFAULT_SLOT_COUNT,
        context=None,
    ) -> None:
        context = context if context is not None else multiprocessing.get_context("spawn")
        self.slot_bytes = max(1, int(slot_bytes))
        self.slots = max(1, int(slots))
        self._memory = shared_memory.SharedMemory(create=True, size=self.slot_bytes * self.slots)
        self._states = context.Array("b", self.slots)
        self._owners = context.Array("i", self.slots, lock=False)
        self._fallbacks = context.Value("q", 0)
        self._owns_segment = True
        self._cursor = 0
        self._closed = False

    @classmethod
    def attach(cls, handle: SlotPoolHandle) -> SlotPool:
        """Map an existing pool from a child. Never unlinks."""

        pool = cls.__new__(cls)
        pool.slot_bytes = int(handle.slot_bytes)
        pool.slots = int(handle.slots)
        pool._memory = shared_memory.SharedMemory(name=handle.name, create=False, track=False)
        pool._states = handle.states
        pool._owners = handle.owners
        pool._fallbacks = handle.fallbacks
        pool._owns_segment = False
        pool._cursor = 0
        pool._closed = False
        return pool

    def handle(self) -> SlotPoolHandle:
        return SlotPoolHandle(
            name=self._memory.name,
            slot_bytes=self.slot_bytes,
            slots=self.slots,
            states=self._states,
            owners=self._owners,
            fallbacks=self._fallbacks,
        )

    @property
    def fallbacks(self) -> int:
        """How often a payload had to pickle because no slot was free.

        Read without the lock: a monotone diagnostic counter has no invariant
        a torn read could break, and this is read on every remote dispatch.
        """

        return int(self._fallbacks.value)

    def lease(self, owner: int = 0) -> SlotLease | None:
        """Claim the next free slot, or count a fallback and return ``None``."""

        with self._states.get_lock():
            for step in range(self.slots):
                index = (self._cursor + step) % self.slots
                if self._states[index] == FREE:
                    self._states[index] = LEASED
                    self._owners[index] = int(owner)
                    self._cursor = (index + 1) % self.slots
                    return SlotLease(self, index)
        with self._fallbacks.get_lock():
            self._fallbacks.value += 1
        return None

    def take(self, descriptor: SlotDescriptor) -> np.ndarray:
        """Move a READY slot to HELD and return the consumer's read-only view."""

        self._advance(descriptor.index, READY, HELD)
        return self._slot_array(
            descriptor.index, descriptor.shape, np.dtype(descriptor.dtype), writable=False
        )

    def release(self, target) -> None:
        index = int(getattr(target, "index", target))
        with self._states.get_lock():
            self._states[index] = FREE
            self._owners[index] = 0

    def reclaim(self, owner: int) -> int:
        """Free every slot tagged ``owner``. The answer to a dead worker."""

        owner = int(owner)
        if owner == 0:
            return 0
        reclaimed = 0
        with self._states.get_lock():
            for index in range(self.slots):
                if self._owners[index] == owner and self._states[index] != FREE:
                    self._states[index] = FREE
                    self._owners[index] = 0
                    reclaimed += 1
        return reclaimed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_segment:
            self._memory.unlink()
        # A consumer may still hold a view. The segment is already unlinked,
        # so the name is gone either way; the mapping dies with the views.
        with suppress(BufferError):
            self._memory.close()

    def _advance(self, index: int, expected: int, target: int) -> None:
        index = int(index)
        with self._states.get_lock():
            current = int(self._states[index])
            if current != expected:
                raise RuntimeError(
                    f"slot {index} is {_STATE_NAMES[current]}, expected {_STATE_NAMES[expected]}"
                )
            self._states[index] = target

    def _slot_array(self, index: int, shape, dtype, *, writable: bool) -> np.ndarray:
        dtype = np.dtype(dtype)
        array = np.ndarray(
            tuple(shape),
            dtype=dtype,
            buffer=self._memory.buf,
            offset=int(index) * self.slot_bytes,
        )
        if not writable:
            array.flags.writeable = False
        return array


def slottable(value) -> bool:
    """Whether an ndarray is worth (and safe to) route through a slot."""

    return (
        isinstance(value, np.ndarray)
        and value.nbytes >= SLOT_PAYLOAD_MIN_BYTES
        and not value.dtype.hasobject
        and value.dtype.fields is None
    )


def wrap_payload(value, pool: SlotPool | None, *, owner: int = 0, depth: int = 4):
    """Replace large ndarrays inside ``value`` with slot descriptors.

    Returns the rewritten payload plus the leases the caller must hold until
    the far side has read them.
    """

    leases: list[SlotLease] = []
    if pool is None:
        return value, ()
    return _wrap(value, pool, owner, depth, leases), tuple(leases)


def unwrap_payload(value, pool: SlotPool | None, *, copy: bool = False):
    """Rebuild ``value`` from slot descriptors.

    ``copy=True`` detaches the arrays from their slots and returns the indices
    the caller should release; ``copy=False`` hands back read-only views that
    stay valid only while the producer holds the leases.
    """

    taken: list[int] = []
    if pool is None:
        return value, ()
    return _unwrap(value, pool, copy, taken), tuple(taken)


def _wrap(value, pool: SlotPool, owner: int, depth: int, leases: list[SlotLease]):
    if slottable(value):
        lease = pool.lease(owner)
        if lease is None or value.nbytes > lease.capacity:
            if lease is not None:
                lease.release()
            return value
        leases.append(lease)
        target = lease.view(value.shape, value.dtype)
        np.copyto(target, value)
        return lease.publish(target)
    if depth <= 0:
        return value
    if isinstance(value, tuple):
        return tuple(_wrap(item, pool, owner, depth - 1, leases) for item in value)
    if isinstance(value, list):
        return [_wrap(item, pool, owner, depth - 1, leases) for item in value]
    if isinstance(value, dict):
        return {key: _wrap(item, pool, owner, depth - 1, leases) for key, item in value.items()}
    return _wrap_dataclass(value, pool, owner, depth, leases)


def _unwrap(value, pool: SlotPool, copy: bool, taken: list[int]):
    if isinstance(value, SlotDescriptor):
        array = pool.take(value)
        taken.append(int(value.index))
        return np.array(array, copy=True) if copy else array
    if isinstance(value, tuple):
        return tuple(_unwrap(item, pool, copy, taken) for item in value)
    if isinstance(value, list):
        return [_unwrap(item, pool, copy, taken) for item in value]
    if isinstance(value, dict):
        return {key: _unwrap(item, pool, copy, taken) for key, item in value.items()}
    return _unwrap_dataclass(value, pool, copy, taken)


def _wrap_dataclass(value, pool: SlotPool, owner: int, depth: int, leases: list[SlotLease]):
    names = _rewritable_dataclass_fields(value)
    if names is None:
        return value
    changed = {}
    for name in names:
        item = getattr(value, name)
        rewritten = _wrap(item, pool, owner, depth - 1, leases)
        if rewritten is not item:
            changed[name] = rewritten
    return _replace_or_original(value, changed)


def _unwrap_dataclass(value, pool: SlotPool, copy: bool, taken: list[int]):
    names = _rewritable_dataclass_fields(value)
    if names is None:
        return value
    changed = {}
    for name in names:
        item = getattr(value, name)
        rewritten = _unwrap(item, pool, copy, taken)
        if rewritten is not item:
            changed[name] = rewritten
    return _replace_or_original(value, changed)


def _rewritable_dataclass_fields(value) -> tuple[str, ...] | None:
    if not is_dataclass(value) or isinstance(value, type):
        return None
    names = tuple(field.name for field in fields(value) if field.init)
    return names if len(names) == len(fields(value)) else None


def _replace_or_original(value, changed: dict):
    if not changed:
        return value
    try:
        return replace(value, **changed)
    except Exception:
        # A dataclass with a validating __post_init__ that rejects a
        # descriptor keeps its arrays and pickles whole; correctness first.
        return value


__all__ = [
    "DEFAULT_SLOT_BYTES",
    "DEFAULT_SLOT_COUNT",
    "SLOT_PAYLOAD_MIN_BYTES",
    "SlotDescriptor",
    "SlotLease",
    "SlotPool",
    "SlotPoolHandle",
    "slottable",
    "unwrap_payload",
    "wrap_payload",
]
