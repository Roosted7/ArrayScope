"""Software page table: logical chunk → physical page slot (ADR 0055).

The page table is passive state, not an actor (ADR 0053 forbids new
schedulers). Owners consult it on the GUI thread; materialization and upload
work still flows through the kernel/pipeline. "Not resident" is an explicit
answer (``lookup`` returns ``None``), never an implicit upload.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arrayscope.gpu.keys import DataChunkKey


@dataclass(frozen=True)
class PageSlot:
    """Physical placement of one chunk: a slot inside a page of a pool."""

    pool_id: str
    page_index: int
    slot_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "pool_id", str(self.pool_id))
        object.__setattr__(self, "page_index", int(self.page_index))
        object.__setattr__(self, "slot_index", int(self.slot_index))


@dataclass
class ResidencyEntry:
    slot: PageSlot
    nbytes: int
    generation: int
    last_use: int
    pinned: bool = False


@dataclass
class PageTable:
    """Mapping of resident chunks with LRU order, pins, and generations.

    The generation increments on every binding change (not on touches), so a
    consumer can cheaply detect "did residency change since I last planned".
    ``last_use`` is a monotonic counter, not wall time — deterministic under
    test and immune to clock adjustment.
    """

    _entries: dict[DataChunkKey, ResidencyEntry] = field(default_factory=dict)
    _slots: dict[PageSlot, DataChunkKey] = field(default_factory=dict)
    _generation: int = 0
    _use_counter: int = 0

    @property
    def generation(self) -> int:
        return self._generation

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: DataChunkKey) -> bool:
        return key in self._entries

    def lookup(self, key: DataChunkKey) -> PageSlot | None:
        """Slot holding ``key``, or ``None`` (explicit not-resident)."""

        entry = self._entries.get(key)
        return None if entry is None else entry.slot

    def bind(self, key: DataChunkKey, slot: PageSlot, *, nbytes: int, pinned: bool = False) -> None:
        """Record that ``key``'s values now live in ``slot``."""

        occupant = self._slots.get(slot)
        if occupant is not None and occupant != key:
            raise ValueError(f"slot {slot} already holds {occupant}")
        previous = self._entries.get(key)
        if previous is not None and previous.slot != slot:
            del self._slots[previous.slot]
        self._use_counter += 1
        self._generation += 1
        self._entries[key] = ResidencyEntry(
            slot=slot,
            nbytes=int(nbytes),
            generation=self._generation,
            last_use=self._use_counter,
            pinned=bool(pinned),
        )
        self._slots[slot] = key

    def unbind(self, key: DataChunkKey) -> PageSlot | None:
        """Drop ``key``; returns the freed slot (``None`` if absent)."""

        entry = self._entries.pop(key, None)
        if entry is None:
            return None
        del self._slots[entry.slot]
        self._generation += 1
        return entry.slot

    def touch(self, key: DataChunkKey) -> None:
        entry = self._entries.get(key)
        if entry is not None:
            self._use_counter += 1
            entry.last_use = self._use_counter

    def last_use(self, key: DataChunkKey, default: int = -1) -> int:
        """Monotonic last-use stamp for LRU tiebreaks (``default`` if absent)."""

        entry = self._entries.get(key)
        return default if entry is None else entry.last_use

    def slot_items(self) -> tuple[tuple[DataChunkKey, PageSlot], ...]:
        return tuple((key, entry.slot) for key, entry in self._entries.items())

    def remap_slots(self, transform) -> None:
        """Rewrite every binding's slot via ``transform(slot) -> PageSlot``.

        Used when pages are dropped/compacted and surviving page indices
        shift. LRU state and pins survive; the generation bumps once.
        """

        new_slots: dict[PageSlot, DataChunkKey] = {}
        for key, entry in self._entries.items():
            slot = transform(entry.slot)
            if slot in new_slots:
                raise ValueError(f"remap collides on slot {slot}")
            entry.slot = slot
            new_slots[slot] = key
        self._slots = new_slots
        self._generation += 1

    def pin(self, key: DataChunkKey, pinned: bool = True) -> None:
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(f"cannot pin non-resident chunk {key}")
        entry.pinned = bool(pinned)

    def chunk_in_slot(self, slot: PageSlot) -> DataChunkKey | None:
        return self._slots.get(slot)

    def resident_keys(self) -> tuple[DataChunkKey, ...]:
        return tuple(self._entries)

    def resident_bytes(self) -> int:
        return sum(entry.nbytes for entry in self._entries.values())

    def eviction_candidates(self) -> tuple[DataChunkKey, ...]:
        """Unpinned resident chunks, least recently used first."""

        return tuple(
            key
            for key, entry in sorted(self._entries.items(), key=lambda item: item[1].last_use)
            if not entry.pinned
        )
