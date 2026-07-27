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


@dataclass(frozen=True)
class PageResolution:
    """One CPU-resolved target → resident-page binding (ADR 0056 §5).

    ``scale`` and ``offset`` describe the nominal aligned-grid transform from
    target stored samples into actual stored samples.  They are exact for
    complete interior bins.  A clipped boundary bin is not representable by
    one affine transform, so presentation must use the immutable target and
    actual ``LodPagePlan`` draw blocks for exact edges; physical truth reports
    both this nominal transform and the submitted draw geometry.  The binding
    generation belongs to that physical association, so compaction or slot
    reuse invalidates a cached resolution even when the logical key survives.
    """

    target_key: DataChunkKey
    actual_key: DataChunkKey
    slot: PageSlot
    scale: tuple[float, ...]
    offset: tuple[float, ...]
    binding_generation: int


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
    _family_entries: dict[tuple[object, ...], dict[DataChunkKey, ResidencyEntry]] = field(
        default_factory=dict
    )
    _generation: int = 0
    _use_counter: int = 0
    _pin_sets: dict[object, frozenset[DataChunkKey]] = field(default_factory=dict)

    @property
    def generation(self) -> int:
        return self._generation

    def __len__(self) -> int:
        return len(self._entries)

    def rebind_from(self, other: PageTable) -> None:
        """Adopt ``other``'s bindings so this table can diverge from them.

        For a consumer that publishes a fresh table per residency revision and
        changes only a few pages between them: rebuilding by ``bind`` costs a
        ``ResidencyEntry`` construction and four index writes per resident
        page, which on a montage-sized set dwarfs the handful of pages that
        actually moved.

        ``ResidencyEntry`` values are SHARED with ``other``, not copied, so
        this is only sound when neither table will mutate an entry in place --
        ``touch``, ``pin``, and ``remap`` all do. ``bind`` and ``unbind``
        replace or drop whole entries and stay safe. Callers that need a
        table they can touch or remap must build their own.
        """

        self._entries = dict(other._entries)
        self._slots = dict(other._slots)
        self._family_entries = {
            family: dict(entries) for family, entries in other._family_entries.items()
        }
        self._pin_sets = dict(other._pin_sets)
        self._generation = other._generation
        self._use_counter = other._use_counter

    def __contains__(self, key: DataChunkKey) -> bool:
        return key in self._entries

    def lookup(self, key: DataChunkKey) -> PageSlot | None:
        """Slot holding ``key``, or ``None`` (explicit not-resident)."""

        entry = self._entries.get(key)
        return None if entry is None else entry.slot

    def resolve(self, target: DataChunkKey) -> PageResolution | None:
        """Resolve ``target`` to its finest compatible resident ancestor.

        Logical chunk geometry is native-source space.  An ancestor must
        cover the complete target footprint and be at least as reduced on
        every axis; semantic generations, representation, dtype, and reducer
        family must match exactly.  Resolution is performed once on the CPU,
        never as a per-fragment shader search.
        """

        exact = self._entries.get(target)
        if exact is not None:
            return _page_resolution(target, target, exact)
        target_reduction = _reduction_vector(target)
        candidates: list[tuple[tuple[object, ...], DataChunkKey, ResidencyEntry]] = []
        family = self._family_entries.get(_value_family_key(target), {})
        for sequence, (key, entry) in enumerate(family.items()):
            actual_reduction = _reduction_vector(key)
            if not page_key_can_cover(target, key):
                continue
            delta = tuple(
                int(actual) - int(wanted)
                for actual, wanted in zip(actual_reduction, target_reduction, strict=False)
            )
            rank = (
                sum(delta),
                max(delta, default=0),
                delta,
                _source_volume(key),
                sum(
                    int(target_start) - int(actual_start)
                    for target_start, actual_start in zip(
                        target.chunk_origin,
                        key.chunk_origin,
                        strict=False,
                    )
                ),
                -int(entry.last_use),
                int(sequence),
            )
            candidates.append((rank, key, entry))
        if not candidates:
            return None
        _rank, actual, entry = min(candidates, key=lambda row: row[0])
        return _page_resolution(target, actual, entry)

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
            pinned=bool(pinned or self._owners_for(key)),
        )
        if isinstance(key, DataChunkKey):
            self._family_entries.setdefault(_value_family_key(key), {})[key] = self._entries[key]
        self._slots[slot] = key
        if pinned:
            self._replace_owner_key(_LEGACY_PIN_OWNER, key, True)

    def unbind(self, key: DataChunkKey) -> PageSlot | None:
        """Drop ``key``; returns the freed slot (``None`` if absent)."""

        entry = self._entries.pop(key, None)
        if entry is None:
            return None
        del self._slots[entry.slot]
        if isinstance(key, DataChunkKey):
            family_key = _value_family_key(key)
            family = self._family_entries.get(family_key)
            if family is not None:
                family.pop(key, None)
                if not family:
                    self._family_entries.pop(family_key, None)
        for owner, keys in tuple(self._pin_sets.items()):
            if key not in keys:
                continue
            remaining = frozenset(value for value in keys if value != key)
            if remaining:
                self._pin_sets[owner] = remaining
            else:
                self._pin_sets.pop(owner, None)
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

    def is_pinned(self, key: DataChunkKey) -> bool:
        """Whether any owner currently protects this resident binding."""

        entry = self._entries.get(key)
        return bool(entry is not None and entry.pinned)

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
        self._generation += 1
        for entry in self._entries.values():
            entry.generation = self._generation
        self._slots = new_slots

    def pin(self, key: DataChunkKey, pinned: bool = True) -> None:
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(f"cannot pin non-resident chunk {key}")
        self._replace_owner_key(_LEGACY_PIN_OWNER, key, bool(pinned))

    def replace_pin_set(self, owner: object, keys) -> None:
        """Atomically replace one owner's resident coverage pins."""

        requested = frozenset(keys or ())
        missing = tuple(key for key in requested if key not in self._entries)
        if missing:
            raise KeyError(f"cannot pin non-resident chunks: {missing!r}")
        previous = self._pin_sets.get(owner, frozenset())
        if requested:
            self._pin_sets[owner] = requested
        else:
            self._pin_sets.pop(owner, None)
        for key in previous | requested:
            entry = self._entries.get(key)
            if entry is not None:
                entry.pinned = bool(self._owners_for(key))

    def pin_set(self, owner: object) -> frozenset[DataChunkKey]:
        """Return one owner's resident pins for diagnostics and capacity planning."""

        return self._pin_sets.get(owner, frozenset())

    def pin_owner_counts(self) -> tuple[tuple[str, int], ...]:
        """Return stable, compact pin attribution for failure evidence."""

        return tuple(
            sorted((str(owner), len(keys)) for owner, keys in self._pin_sets.items() if keys)
        )

    def _replace_owner_key(self, owner: object, key: DataChunkKey, pinned: bool) -> None:
        values = set(self._pin_sets.get(owner, frozenset()))
        if pinned:
            values.add(key)
        else:
            values.discard(key)
        self.replace_pin_set(owner, values)

    def _owners_for(self, key: DataChunkKey) -> tuple[object, ...]:
        return tuple(owner for owner, keys in self._pin_sets.items() if key in keys)

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


_LEGACY_PIN_OWNER = object()


def _reduction_vector(key: DataChunkKey) -> tuple[int, ...]:
    reduction = tuple(int(step) for step in key.lod.reduction)
    if len(reduction) < key.rank:
        reduction += (0,) * (key.rank - len(reduction))
    return reduction[: key.rank]


def _same_value_family(target: DataChunkKey, actual: DataChunkKey) -> bool:
    return bool(
        target.rank == actual.rank
        and target.document_generation == actual.document_generation
        and target.operation_key == actual.operation_key
        and target.dtype == actual.dtype
        and target.representation == actual.representation
        and target.lod.reducer == actual.lod.reducer
        and target.lod.gutter == actual.lod.gutter
    )


def _value_family_key(key: DataChunkKey) -> tuple[object, ...]:
    return (
        key.rank,
        key.document_generation,
        key.operation_key,
        key.dtype,
        key.representation,
        key.lod.reducer,
        key.lod.gutter,
    )


def _page_resolution(
    target: DataChunkKey,
    actual: DataChunkKey,
    entry: ResidencyEntry,
) -> PageResolution:
    target_reduction = _reduction_vector(target)
    actual_reduction = _reduction_vector(actual)
    target_scale = tuple(float(1 << step) for step in target_reduction)
    actual_scale = tuple(float(1 << step) for step in actual_reduction)
    return PageResolution(
        target_key=target,
        actual_key=actual,
        slot=entry.slot,
        scale=tuple(
            target_step / actual_step
            for target_step, actual_step in zip(target_scale, actual_scale, strict=False)
        ),
        offset=tuple(
            (float(target_start) - float(actual_start)) / actual_step
            for target_start, actual_start, actual_step in zip(
                target.chunk_origin,
                actual.chunk_origin,
                actual_scale,
                strict=False,
            )
        ),
        binding_generation=int(entry.generation),
    )


def page_key_can_cover(
    target: DataChunkKey,
    actual: DataChunkKey,
    *,
    require_coverage: bool = True,
) -> bool:
    """Return whether ``actual`` is a semantic/anisotropic ancestor of ``target``.

    This is the single value-family and ancestry predicate used by resolution
    and by checked page-backed payload admission.  Geometry stays in native
    source axis order throughout.
    """

    if not _same_value_family(target, actual):
        return False
    target_reduction = _reduction_vector(target)
    actual_reduction = _reduction_vector(actual)
    if any(
        value < wanted for value, wanted in zip(actual_reduction, target_reduction, strict=False)
    ):
        return False
    if not require_coverage:
        return True
    return all(
        actual_start <= target_start and actual_stop >= target_stop
        for actual_start, actual_stop, target_start, target_stop in zip(
            actual.chunk_origin,
            actual.stop,
            target.chunk_origin,
            target.stop,
            strict=False,
        )
    )


def _source_volume(key: DataChunkKey) -> int:
    volume = 1
    for extent in key.chunk_shape:
        volume *= int(extent)
    return volume
