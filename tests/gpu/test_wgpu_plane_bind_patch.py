"""In-place plane rebinds: a bounded delta must not rebuild the whole table.

A bounded montage commit re-sends the full plane list with a few indices
swapped.  ``_bind_planes`` used to answer anything but a pure append with a
whole-montage rebuild — grids, family indices, and the complete flat lookup
table — so a one-tile delta paid plane cost proportional to the montage.  The
patch path rewrites exactly the changed indices; these tests pin its physical
truth with the same pixel oracle the command-protocol suite uses, not with
timing.

Every scene uses per-plane value ramps (seeded planes), so a patched plane
drawing the wrong content is a pixel difference, never a silent pass.
"""

import pytest

wgpu = pytest.importorskip("wgpu")

from arrayscope.gpu.command_protocol import (
    BindContentPlanes,
    ContentPlane,
    FrameSubmission,
)
from arrayscope.gpu.keys import SCALAR_R32F
from arrayscope.gpu.wgpu_executor import PAGE, WgpuPlaneExecutor
from tests.gpu.test_wgpu_command_protocol import (
    SCALAR_LEVELS,
    SP_CANVAS,
    SP_H,
    SP_W,
    _adapter_available,
    _assert_matches,
    _ensure_scalar_plane,
    _montage_tiles,
    _scalar_key,
    _scalar_plane,
    _scalar_plane_binding,
    _scalar_reference,
    _shared_device,
)

pytestmark = pytest.mark.skipif(not _adapter_available(), reason="no wgpu adapter on this machine")

# Five planes: enough that a one-plane swap sits under the patch-vs-rebuild
# width threshold (a two-plane montage legitimately rebuilds).
DOCS = "ABCDE"


def _scene():
    planes = {doc: _scalar_plane(i + 1) for i, doc in enumerate(DOCS)}
    executor = WgpuPlaneExecutor(target_size=SP_CANVAS, pool_layers=32, device=_shared_device())
    commands = [BindContentPlanes(tuple(_scalar_plane_binding(doc) for doc in DOCS))]
    for doc in DOCS:
        commands.extend(_ensure_scalar_plane(doc, planes[doc]))
    executor.submit(FrameSubmission(0, commands))
    assert executor.plane_bind_rebuilds_total == 1
    return executor, planes


def _render(executor, tiles, generation):
    from arrayscope.gpu.command_protocol import (
        PresentGeneration,
        SetDisplayMapping,
        UpdateTileInstances,
    )

    report = executor.submit(
        FrameSubmission(
            generation,
            (
                SetDisplayMapping(SCALAR_LEVELS),
                UpdateTileInstances(tuple(tiles)),
                PresentGeneration(generation),
            ),
        )
    )
    assert report.presented
    return report


def _swapped_bindings(replaced_index, new_doc):
    bindings = [_scalar_plane_binding(doc) for doc in DOCS]
    bindings[replaced_index] = _scalar_plane_binding(new_doc)
    return tuple(bindings)


def test_in_place_plane_swap_is_a_patch_and_pixels_follow():
    executor, planes = _scene()
    planes["Z"] = _scalar_plane(9)
    view = _montage_tiles((0, 1))
    _render(executor, view, 1)
    _assert_matches(
        executor.read_target(),
        _scalar_reference([(t, planes[DOCS[t.plane_index]]) for t in view], SCALAR_LEVELS),
    )

    # The real commit shape: the swapped plane's pages arrive in the SAME
    # submission, after the bind.
    executor.submit(
        FrameSubmission(
            2,
            (
                BindContentPlanes(_swapped_bindings(1, "Z")),
                *_ensure_scalar_plane("Z", planes["Z"]),
            ),
        )
    )
    assert executor.plane_bind_patches_total == 1
    assert executor.plane_bind_rebuilds_total == 1

    _render(executor, view, 3)
    lookup = {0: "A", 1: "Z"}
    _assert_matches(
        executor.read_target(),
        _scalar_reference([(t, planes[lookup[t.plane_index]]) for t in view], SCALAR_LEVELS),
    )
    # The family index moved with the patch: the old plane's keys no longer
    # feed slot 1, the new plane's keys do.
    assert executor._flat_indices(_scalar_key("B", 0, 0)) == ()
    assert executor._flat_indices(_scalar_key("Z", 0, 0)) != ()


def test_patch_refills_from_pages_already_resident():
    executor, planes = _scene()
    planes["Z"] = _scalar_plane(9)
    # Warm the replacement's pages while its plane is still unbound — resident
    # pages of unbound planes are exactly what the page table keeps warm.
    executor.submit(FrameSubmission(1, _ensure_scalar_plane("Z", planes["Z"])))

    report = executor.submit(FrameSubmission(2, (BindContentPlanes(_swapped_bindings(1, "Z")),)))
    assert executor.plane_bind_patches_total == 1
    assert report.uploads == 0

    view = _montage_tiles((0, 1))
    _render(executor, view, 3)
    lookup = {0: "A", 1: "Z"}
    _assert_matches(
        executor.read_target(),
        _scalar_reference([(t, planes[lookup[t.plane_index]]) for t in view], SCALAR_LEVELS),
    )


def test_footprint_change_takes_the_rebuild_path():
    executor, planes = _scene()
    bindings = [_scalar_plane_binding(doc) for doc in DOCS]
    # Same source, deeper ladder: the flat span grows, every later plane's
    # base shifts, and only the rebuild derives that honestly.
    bindings[1] = ContentPlane("B", "op-live", (SP_H, SP_W), max_lod=1, representation=SCALAR_R32F)
    executor.submit(FrameSubmission(1, (BindContentPlanes(tuple(bindings)),)))
    assert executor.plane_bind_patches_total == 0
    assert executor.plane_bind_rebuilds_total == 2

    view = _montage_tiles((0, 1))
    _render(executor, view, 2)
    _assert_matches(
        executor.read_target(),
        _scalar_reference([(t, planes[DOCS[t.plane_index]]) for t in view], SCALAR_LEVELS),
    )


def test_wide_swap_takes_the_rebuild_path():
    executor, planes = _scene()
    for doc, seed in (("V", 11), ("W", 12), ("X", 13), ("Y", 14)):
        planes[doc] = _scalar_plane(seed)
    bindings = [_scalar_plane_binding(doc) for doc in "VWXYE"]
    executor.submit(
        FrameSubmission(
            1,
            (
                BindContentPlanes(tuple(bindings)),
                *(command for doc in "VWXY" for command in _ensure_scalar_plane(doc, planes[doc])),
            ),
        )
    )
    assert executor.plane_bind_patches_total == 0
    assert executor.plane_bind_rebuilds_total == 2

    view = _montage_tiles((0, 4))
    _render(executor, view, 2)
    lookup = {0: "V", 4: "E"}
    _assert_matches(
        executor.read_target(),
        _scalar_reference([(t, planes[lookup[t.plane_index]]) for t in view], SCALAR_LEVELS),
    )


def test_pin_replacement_repins_pages_that_arrive_later():
    # The memo in replace_resident_pin_set must not outlive a residency
    # change: a key requested while absent joins the pin set as soon as a
    # later call runs after its page arrived (the page-table generation moved,
    # so the memo cannot answer from cache).
    executor, planes = _scene()
    planes["Z"] = _scalar_plane(9)
    owner = "probe-owner"
    wanted = tuple(_scalar_key("Z", cx, 0) for cx in range(SP_W // PAGE))

    resident = executor.replace_resident_pin_set(owner, wanted)
    assert resident == frozenset()

    executor.submit(FrameSubmission(1, _ensure_scalar_plane("Z", planes["Z"])))
    resident = executor.replace_resident_pin_set(owner, wanted)
    assert resident == frozenset(wanted)
    assert all(executor.page_table.is_pinned(key) for key in wanted)

    # Unchanged keys and unchanged residency: the memo answers, and the pin
    # set it reports is still exactly what the table holds.
    assert executor.replace_resident_pin_set(owner, wanted) == frozenset(wanted)
    assert executor.page_table.pin_set(owner) == frozenset(wanted)
