"""PageTable residency bookkeeping (ADR 0055 G1): explicit lookup misses,
bind/unbind roundtrips, LRU eviction order that skips pins.

These used to be driven through ``ChunkStore``/``SlotPool``, a facade whose
``ensure`` evicted eagerly at reservation time. The shipped WGPU executor never
adopted it — it owns its own layer pools and evicts on demand from
``_ensure_free_pool_layer`` — so the facade was deleted and the assertions were
moved onto ``PageTable`` itself, which the executor really does use.
"""

import pytest

from arrayscope.gpu import ChunkLod, DataChunkKey, PageSlot, PageTable


def chunk(n, generation=1):
    return DataChunkKey(
        document_generation=("doc", generation),
        operation_key=None,
        lod=ChunkLod(),
        chunk_origin=(n * 128,),
        chunk_shape=(128,),
        dtype="float32",
    )


class TestPageTable:
    def test_lookup_miss_is_explicit_none(self):
        table = PageTable()
        assert table.lookup(chunk(0)) is None
        assert chunk(0) not in table

    def test_bind_lookup_unbind_roundtrip(self):
        table = PageTable()
        slot = PageSlot("p", 0, 0)
        table.bind(chunk(0), slot, nbytes=64)
        assert table.lookup(chunk(0)) == slot
        assert table.chunk_in_slot(slot) == chunk(0)
        assert table.resident_bytes() == 64
        assert table.unbind(chunk(0)) == slot
        assert table.lookup(chunk(0)) is None
        assert table.unbind(chunk(0)) is None

    def test_slot_conflict_rejected(self):
        table = PageTable()
        slot = PageSlot("p", 0, 0)
        table.bind(chunk(0), slot, nbytes=1)
        with pytest.raises(ValueError):
            table.bind(chunk(1), slot, nbytes=1)

    def test_rebinding_same_chunk_moves_it(self):
        table = PageTable()
        table.bind(chunk(0), PageSlot("p", 0, 0), nbytes=1)
        table.bind(chunk(0), PageSlot("p", 0, 1), nbytes=1)
        assert table.lookup(chunk(0)) == PageSlot("p", 0, 1)
        assert table.chunk_in_slot(PageSlot("p", 0, 0)) is None

    def test_generation_counts_binding_changes_not_touches(self):
        table = PageTable()
        table.bind(chunk(0), PageSlot("p", 0, 0), nbytes=1)
        generation = table.generation
        table.touch(chunk(0))
        assert table.generation == generation
        table.unbind(chunk(0))
        assert table.generation == generation + 1

    def test_eviction_order_is_lru_and_skips_pinned(self):
        # Successive victims, each taken in turn, are the whole LRU order:
        # asking for the single victim and then unbinding it walks the same
        # sequence the executor walks under sustained pool pressure.
        table = PageTable()
        for n in range(3):
            table.bind(chunk(n), PageSlot("p", 0, n), nbytes=1)
        table.touch(chunk(0))  # 1 is now least recently used
        table.pin(chunk(1))
        walked = []
        while (victim := table.eviction_candidate()) is not None:
            walked.append(victim)
            table.unbind(victim)
        assert walked == [chunk(2), chunk(0)]  # the pinned 1 is never offered
        with pytest.raises(KeyError):
            table.pin(chunk(9))

    def test_the_victim_query_honours_the_executor_pool_filter(self):
        # ``_evict_one_unpinned`` needs a free layer in ONE physical pool: the
        # raw and BC pools of a representation have independent budgets, so a
        # caller that needs a BC layer must never be handed a freed raw one.
        table = PageTable()
        for n in range(4):
            table.bind(chunk(n), PageSlot("a" if n % 2 else "b", 0, n), nbytes=1)
        table.touch(chunk(0))  # 1 (pool a) and 2 (pool b) are now the oldest
        table.pin(chunk(1))
        assert table.eviction_candidate() == chunk(2)
        assert table.eviction_candidate(lambda key, slot: slot.pool_id == "a") == chunk(3)
        assert table.eviction_candidate(lambda key, slot: slot.pool_id == "b") == chunk(2)
        assert table.eviction_candidate(lambda key, slot: slot.pool_id == "z") is None

    def test_every_binding_pinned_leaves_no_eviction_candidate(self):
        # The condition the executor reads before it grows or gives up: a
        # fully pinned table offers nothing to evict, so the caller must widen
        # its pool rather than silently drop somebody's coverage.
        table = PageTable()
        table.bind(chunk(0), PageSlot("p", 0, 0), nbytes=1)
        table.pin(chunk(0))
        assert table.eviction_candidate() is None

    def test_superseded_generation_releases_its_slot_for_the_successor(self):
        # A document revision bump mints different chunk keys; superseded
        # generations are explicitly unbound, exactly like the montage payload
        # cache drops superseded source ids.
        table = PageTable()
        slot = PageSlot("p", 0, 0)
        table.bind(chunk(0, generation=1), slot, nbytes=1)
        assert table.unbind(chunk(0, generation=1)) == slot
        table.bind(chunk(0, generation=2), slot, nbytes=1)
        assert len(table) == 1
        assert table.lookup(chunk(0, generation=2)) == slot
        assert table.lookup(chunk(0, generation=1)) is None

    def test_eviction_stamp_moves_only_when_residency_can_be_lost(self):
        # The stamp is the "nothing I verified resident has gone away" proof
        # a bounded commit uses to skip re-walking the whole presented set.
        # Anything that only ADDS coverage or moves slots must therefore leave
        # it alone, and anything that can DROP coverage must move it.
        table = PageTable()
        stamp = table.eviction_stamp
        table.bind(chunk(0), PageSlot("p", 0, 0), nbytes=1)
        table.touch(chunk(0))
        table.pin(chunk(0))
        assert table.eviction_stamp == stamp
        table.remap_slots(lambda slot: PageSlot("p", 1, slot.slot_index))
        assert table.eviction_stamp == stamp
        table.unbind(chunk(0))
        assert table.eviction_stamp != stamp

    def test_eviction_stamp_never_validates_across_tables(self):
        # Same serial value, different table: the per-instance epoch keeps a
        # stamp from one table from proving anything about another — including
        # a table that wholesale-adopted this one's bindings.
        first = PageTable()
        second = PageTable()
        assert first.eviction_stamp != second.eviction_stamp
        first.bind(chunk(0), PageSlot("p", 0, 0), nbytes=1)
        stamp = first.eviction_stamp
        second.rebind_from(first)
        assert second.eviction_stamp != stamp

    def test_rebind_from_invalidates_the_adopters_own_stamps(self):
        # Adoption replaces residency wholesale, which can drop keys the
        # adopter used to hold without ever calling unbind on them.
        table = PageTable()
        table.bind(chunk(0), PageSlot("p", 0, 0), nbytes=1)
        stamp = table.eviction_stamp
        table.rebind_from(PageTable())
        assert table.eviction_stamp != stamp
        assert table.lookup(chunk(0)) is None
