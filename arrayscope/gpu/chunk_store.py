"""Chunk store: page table + physical slot pool behind one facade (ADR 0055).

``ChunkStore.ensure`` answers the only question a renderer needs at draw
time: *where does this chunk live, and do I have to upload it first?* It
never performs the upload — that is backend work scheduled through the
kernel — it only reserves the destination and tracks the bookkeeping.

This is the software-virtual-texturing half of the design; hardware sparse
resources (G7) would replace the pool internals, not this interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arrayscope.gpu.keys import DataChunkKey
from arrayscope.gpu.page_table import PageSlot, PageTable


@dataclass(frozen=True)
class Residency:
    """Result of ``ensure``: destination slot plus what the caller must do."""

    key: DataChunkKey
    slot: PageSlot
    needs_upload: bool
    evicted: tuple[DataChunkKey, ...] = ()


@dataclass(frozen=True)
class ChunkStoreDiagnostics:
    pool_id: str
    pages: int
    slots_per_page: int
    resident: int
    free_slots: int
    resident_bytes: int
    hits: int
    misses: int
    evictions: int
    capacity_denials: int
    generation: int


class CapacityError(RuntimeError):
    """Raised when a chunk cannot be placed even after evicting everything unpinned."""


@dataclass
class SlotPool:
    """Fixed-size physical slots organized in pages, grown up to a budget.

    Mirrors the VisPy atlas structure (uniform slots in growable pages) so
    the atlas can adopt it in G2, but knows nothing about textures: a slot
    is an address, not storage.
    """

    pool_id: str
    slots_per_page: int
    max_pages: int
    _free: list[PageSlot] = field(default_factory=list)
    _pages: int = 0

    def __post_init__(self) -> None:
        self.pool_id = str(self.pool_id)
        self.slots_per_page = int(self.slots_per_page)
        self.max_pages = int(self.max_pages)
        if self.slots_per_page <= 0:
            raise ValueError("slots_per_page must be positive")
        if self.max_pages <= 0:
            raise ValueError("max_pages must be positive")

    @property
    def pages(self) -> int:
        return self._pages

    def free_count(self) -> int:
        return len(self._free)

    def capacity(self) -> int:
        return self.max_pages * self.slots_per_page

    def acquire(self) -> PageSlot | None:
        """A free slot, growing by one page if allowed; ``None`` when full."""

        if not self._free and self._pages < self.max_pages:
            page_index = self._pages
            self._pages += 1
            # Stack order: low slot indices come off first, which keeps early
            # allocations dense within a page (good locality for debugging
            # and for later per-page visibility culling).
            self._free.extend(
                PageSlot(self.pool_id, page_index, slot)
                for slot in reversed(range(self.slots_per_page))
            )
        return self._free.pop() if self._free else None

    def release(self, slot: PageSlot) -> None:
        if slot.pool_id != self.pool_id:
            raise ValueError(f"slot {slot} does not belong to pool {self.pool_id!r}")
        self._free.append(slot)


@dataclass
class ChunkStore:
    """One pool of uniform slots plus the page table over it."""

    pool: SlotPool
    table: PageTable = field(default_factory=PageTable)
    _hits: int = 0
    _misses: int = 0
    _evictions: int = 0
    _capacity_denials: int = 0

    def ensure(self, key: DataChunkKey, *, nbytes: int, pin: bool = False) -> Residency:
        """Reserve residency for ``key``; report whether an upload is needed.

        A hit touches LRU state and returns ``needs_upload=False``. A miss
        acquires a slot — evicting least-recently-used unpinned chunks if the
        pool is exhausted — and returns ``needs_upload=True``; the caller
        uploads into ``slot`` and the binding is already recorded. Raises
        :class:`CapacityError` when every slot is pinned.
        """

        existing = self.table.lookup(key)
        if existing is not None:
            self._hits += 1
            self.table.touch(key)
            if pin:
                self.table.pin(key)
            return Residency(key=key, slot=existing, needs_upload=False)

        self._misses += 1
        evicted: list[DataChunkKey] = []
        slot = self.pool.acquire()
        while slot is None:
            candidates = self.table.eviction_candidates()
            if not candidates:
                self._capacity_denials += 1
                raise CapacityError(
                    f"pool {self.pool.pool_id!r} exhausted: {len(self.table)} resident, all pinned"
                )
            victim = candidates[0]
            freed = self.table.unbind(victim)
            self.pool.release(freed)
            evicted.append(victim)
            self._evictions += 1
            slot = self.pool.acquire()
        self.table.bind(key, slot, nbytes=nbytes, pinned=pin)
        return Residency(key=key, slot=slot, needs_upload=True, evicted=tuple(evicted))

    def release(self, key: DataChunkKey) -> None:
        """Explicitly drop a chunk (e.g. superseded generation)."""

        slot = self.table.unbind(key)
        if slot is not None:
            self.pool.release(slot)

    def unpin(self, key: DataChunkKey) -> None:
        if key in self.table:
            self.table.pin(key, False)

    def replace_pins(self, owner: object, keys) -> None:
        """Replace one consumer's coverage set without disturbing others."""

        self.table.replace_pin_set(owner, keys)

    def diagnostics(self) -> ChunkStoreDiagnostics:
        return ChunkStoreDiagnostics(
            pool_id=self.pool.pool_id,
            pages=self.pool.pages,
            slots_per_page=self.pool.slots_per_page,
            resident=len(self.table),
            free_slots=self.pool.free_count(),
            resident_bytes=self.table.resident_bytes(),
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            capacity_denials=self._capacity_denials,
            generation=self.table.generation,
        )
