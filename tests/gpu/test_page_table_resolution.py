"""Best-resident LOD resolution and coverage pins (ADR 0056, G5).

The page-table transform maps coordinates in the target chunk's stored-sample
space into the resolved chunk's stored-sample space::

    resolved = target * scale + offset

Chunk origins and shapes are source-space extents.  Consequently a target
whose reduction step is 0 resolving through a step-2 ancestor has scale 1/4;
an origin delta of 256 source samples becomes an offset of 64 stored samples.
"""

import pytest

from arrayscope.gpu import (
    CapacityError,
    ChunkLod,
    ChunkStore,
    DataChunkKey,
    PageSlot,
    PageTable,
    SlotPool,
)


def chunk(
    *,
    origin=(0, 0),
    shape=(256, 256),
    reduction=(0, 0),
    reducer="mean",
    gutter=0,
    operation_key=("op", "identity"),
):
    return DataChunkKey(
        document_generation=("doc", 1),
        operation_key=operation_key,
        lod=ChunkLod(reduction=reduction, reducer=reducer, gutter=gutter),
        chunk_origin=origin,
        chunk_shape=shape,
        dtype="float32",
    )


def bind(table, key, slot_index):
    slot = PageSlot("lod", 0, slot_index)
    table.bind(key, slot, nbytes=256 * 256 * 4)
    return slot


def assert_resolution(resolution, *, target, actual, slot, scale, offset):
    assert resolution is not None
    assert resolution.target_key == target
    assert resolution.actual_key == actual
    assert resolution.slot == slot
    assert resolution.scale == pytest.approx(scale)
    assert resolution.offset == pytest.approx(offset)
    assert resolution.binding_generation > 0


def test_exact_target_resolves_to_its_exact_binding():
    table = PageTable()
    target = chunk(origin=(256, 256))
    slot = bind(table, target, 0)

    assert_resolution(
        table.resolve(target),
        target=target,
        actual=target,
        slot=slot,
        scale=(1.0, 1.0),
        offset=(0.0, 0.0),
    )


def test_resolution_scans_only_the_targets_value_family(monkeypatch):
    import arrayscope.gpu.page_table as page_table_module

    table = PageTable()
    target = chunk(origin=(256, 256))
    ancestor = chunk(origin=(0, 0), shape=(1024, 1024), reduction=(2, 2))
    ancestor_slot = bind(table, ancestor, 0)
    for index in range(100):
        bind(table, chunk(operation_key=("unrelated", index)), index + 1)
    original = page_table_module.page_key_can_cover
    calls = 0

    def counted(wanted, actual, **kwargs):
        nonlocal calls
        calls += 1
        return original(wanted, actual, **kwargs)

    monkeypatch.setattr(page_table_module, "page_key_can_cover", counted)
    resolution = table.resolve(target)

    assert resolution.actual_key == ancestor
    assert resolution.slot == ancestor_slot
    assert calls == 1


def test_finest_compatible_covering_ancestor_wins():
    table = PageTable()
    target = chunk(origin=(256, 256))
    coarse = chunk(origin=(0, 0), shape=(1024, 1024), reduction=(3, 3))
    fine = chunk(origin=(0, 0), shape=(512, 512), reduction=(1, 1))
    bind(table, coarse, 0)
    fine_slot = bind(table, fine, 1)

    assert_resolution(
        table.resolve(target),
        target=target,
        actual=fine,
        slot=fine_slot,
        scale=(0.5, 0.5),
        offset=(128.0, 128.0),
    )


@pytest.mark.parametrize(
    "incompatible",
    [
        # Correct family and LOD direction, but it does not cover the target.
        chunk(origin=(0, 0), shape=(256, 256), reduction=(2, 2)),
        # X is a coarser ancestor, but Y is finer than the anisotropic target.
        chunk(origin=(0, 0), shape=(1024, 1024), reduction=(2, 0)),
        # Same geometry and reduction, but semantically different values.
        chunk(origin=(0, 0), shape=(1024, 1024), reduction=(2, 2), reducer="rms"),
        # Atlas-border samples are a different stored-value family.
        chunk(origin=(0, 0), shape=(1024, 1024), reduction=(2, 2), gutter=1),
    ],
    ids=("spatial-coverage", "anisotropic-direction", "reducer-family", "gutter-family"),
)
def test_incompatible_resident_pages_are_not_ancestors(incompatible):
    table = PageTable()
    target = chunk(origin=(256, 256), reduction=(1, 1))
    bind(table, incompatible, 0)

    assert table.resolve(target) is None


@pytest.mark.parametrize(
    ("target_reduction", "crossed_reduction", "expected_scale"),
    [
        ((1, 2), (2, 1), (0.5, 1.0)),
        ((2, 1), (1, 2), (1.0, 0.5)),
    ],
)
def test_asymmetric_yx_routes_use_componentwise_ancestry(
    target_reduction,
    crossed_reduction,
    expected_scale,
):
    target = chunk(reduction=target_reduction)
    crossed = chunk(reduction=crossed_reduction)

    crossed_only = PageTable()
    bind(crossed_only, crossed, 0)
    assert crossed_only.resolve(target) is None, (
        "(1, 2) and (2, 1) are not interchangeable ancestors"
    )

    table = PageTable()
    ancestor = chunk(reduction=(2, 2))
    slot = bind(table, ancestor, 1)
    assert_resolution(
        table.resolve(target),
        target=target,
        actual=ancestor,
        slot=slot,
        scale=expected_scale,
        offset=(0.0, 0.0),
    )


def test_unbinding_fine_page_immediately_falls_back_to_coarse():
    table = PageTable()
    target = chunk(origin=(256, 256))
    coarse = chunk(origin=(0, 0), shape=(1024, 1024), reduction=(2, 2))
    fine = chunk(origin=(256, 256), reduction=(1, 1))
    coarse_slot = bind(table, coarse, 0)
    bind(table, fine, 1)

    assert table.resolve(target).actual_key == fine
    table.unbind(fine)

    assert_resolution(
        table.resolve(target),
        target=target,
        actual=coarse,
        slot=coarse_slot,
        scale=(0.25, 0.25),
        offset=(64.0, 64.0),
    )


def test_remap_and_slot_reuse_mint_new_binding_generations():
    table = PageTable()
    target = chunk(origin=(256, 256))
    first = chunk(origin=(0, 0), shape=(1024, 1024), reduction=(3, 3))
    first_slot = bind(table, first, 0)
    initial = table.resolve(target)

    remapped_slot = PageSlot("lod", 1, 3)
    table.remap_slots(lambda slot: remapped_slot if slot == first_slot else slot)
    remapped = table.resolve(target)
    assert remapped.slot == remapped_slot
    assert remapped.binding_generation > initial.binding_generation

    table.unbind(first)
    replacement = chunk(origin=(0, 0), shape=(512, 512), reduction=(1, 1))
    table.bind(replacement, remapped_slot, nbytes=256 * 256 * 4)
    reused = table.resolve(target)
    assert reused.actual_key == replacement
    assert reused.slot == remapped_slot
    assert reused.binding_generation > remapped.binding_generation


def test_owner_pin_replacement_keeps_shared_coverage_until_last_owner_releases():
    store = ChunkStore(SlotPool(pool_id="lod", slots_per_page=3, max_pages=1))
    coarse = chunk(origin=(0, 0), shape=(1024, 1024), reduction=(3, 3))
    fine_a = chunk(origin=(0, 0), shape=(512, 512), reduction=(1, 1))
    fine_b = chunk(origin=(512, 0), shape=(512, 512), reduction=(1, 1))
    for key in (coarse, fine_a, fine_b):
        store.ensure(key, nbytes=1)

    store.replace_pins("view-a", (coarse,))
    store.replace_pins("view-b", (coarse,))
    store.replace_pins("view-a", (fine_a,))
    assert coarse not in store.table.eviction_candidates()
    assert fine_a not in store.table.eviction_candidates()

    store.replace_pins("view-b", (fine_b,))
    assert coarse in store.table.eviction_candidates()
    assert fine_a not in store.table.eviction_candidates()
    assert fine_b not in store.table.eviction_candidates()


def test_all_pinned_capacity_denial_preserves_resolvable_coverage():
    store = ChunkStore(SlotPool(pool_id="lod", slots_per_page=1, max_pages=1))
    coarse = chunk(origin=(0, 0), shape=(1024, 1024), reduction=(3, 3))
    target = chunk(origin=(256, 256))
    coarse_slot = store.ensure(coarse, nbytes=1).slot
    store.replace_pins("active-view", (coarse,))

    with pytest.raises(CapacityError):
        store.ensure(target, nbytes=1)

    assert len(store.table) == 1
    assert_resolution(
        store.table.resolve(target),
        target=target,
        actual=coarse,
        slot=coarse_slot,
        scale=(0.125, 0.125),
        offset=(32.0, 32.0),
    )
