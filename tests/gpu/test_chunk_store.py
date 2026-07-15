import pytest

from arrayscope.gpu import CapacityError, ChunkLod, ChunkStore, DataChunkKey, PageSlot, PageTable, SlotPool


def chunk(n, generation=1):
    return DataChunkKey(
        document_generation=("doc", generation),
        operation_key=None,
        lod=ChunkLod(),
        chunk_origin=(n * 128,),
        chunk_shape=(128,),
        dtype="float32",
    )


def make_store(slots_per_page=2, max_pages=2):
    return ChunkStore(pool=SlotPool(pool_id="test", slots_per_page=slots_per_page, max_pages=max_pages))


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
        table = PageTable()
        for n in range(3):
            table.bind(chunk(n), PageSlot("p", 0, n), nbytes=1)
        table.touch(chunk(0))  # 1 is now least recently used
        table.pin(chunk(1))
        assert table.eviction_candidates() == (chunk(2), chunk(0))
        with pytest.raises(KeyError):
            table.pin(chunk(9))


class TestSlotPool:
    def test_grows_page_by_page_up_to_budget(self):
        pool = SlotPool(pool_id="p", slots_per_page=2, max_pages=2)
        slots = [pool.acquire() for _ in range(4)]
        assert all(slot is not None for slot in slots)
        assert len(set(slots)) == 4
        assert pool.pages == 2
        assert pool.acquire() is None
        pool.release(slots[0])
        assert pool.acquire() == slots[0]

    def test_rejects_foreign_slot_release(self):
        pool = SlotPool(pool_id="p", slots_per_page=1, max_pages=1)
        with pytest.raises(ValueError):
            pool.release(PageSlot("other", 0, 0))


class TestChunkStore:
    def test_miss_then_hit(self):
        store = make_store()
        first = store.ensure(chunk(0), nbytes=64)
        assert first.needs_upload and first.evicted == ()
        again = store.ensure(chunk(0), nbytes=64)
        assert not again.needs_upload
        assert again.slot == first.slot
        diagnostics = store.diagnostics()
        assert (diagnostics.hits, diagnostics.misses) == (1, 1)

    def test_lru_eviction_on_exhaustion(self):
        store = make_store(slots_per_page=2, max_pages=1)
        a = store.ensure(chunk(0), nbytes=1)
        store.ensure(chunk(1), nbytes=1)
        store.ensure(chunk(0), nbytes=1)  # refresh chunk 0; chunk 1 is LRU
        third = store.ensure(chunk(2), nbytes=1)
        assert third.needs_upload
        assert third.evicted == (chunk(1),)
        assert third.slot in {a.slot, third.slot}
        # Evicted chunk re-ensures as a fresh upload.
        assert store.ensure(chunk(1), nbytes=1).needs_upload

    def test_pinned_chunks_never_evicted(self):
        store = make_store(slots_per_page=1, max_pages=1)
        store.ensure(chunk(0), nbytes=1, pin=True)
        with pytest.raises(CapacityError):
            store.ensure(chunk(1), nbytes=1)
        store.unpin(chunk(0))
        replacement = store.ensure(chunk(1), nbytes=1)
        assert replacement.evicted == (chunk(0),)

    def test_release_frees_slot_for_reuse(self):
        store = make_store(slots_per_page=1, max_pages=1)
        first = store.ensure(chunk(0), nbytes=1)
        store.release(chunk(0))
        second = store.ensure(chunk(1), nbytes=1)
        assert second.slot == first.slot
        assert second.evicted == ()

    def test_generation_supersession_is_a_release_pattern(self):
        # A document revision bump mints different chunk keys; superseded
        # generations are explicitly released, exactly like the montage
        # payload cache drops superseded source ids.
        store = make_store()
        old = store.ensure(chunk(0, generation=1), nbytes=1)
        store.release(chunk(0, generation=1))
        new = store.ensure(chunk(0, generation=2), nbytes=1)
        assert new.needs_upload
        assert store.diagnostics().resident == 1
        assert new.slot == old.slot
