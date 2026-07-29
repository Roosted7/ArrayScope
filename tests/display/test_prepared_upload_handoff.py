"""Ring 1. Backends take worker-prepared buffers only under the exact identity.

The hand-off seam described in `docs/architecture/progressive-render-contract.md`
(R5, "the budget is a bookkeeping budget"): a worker packs a payload's upload
buffer, the GUI thread submits it. This pins the half that can go wrong —
a buffer prepared for one payload must never be presented for another, and both
backends must fall back to preparing inline when the mailbox cannot help.

Ring 1 because the seam is pure array/mailbox logic: no compositor is needed to
tell a matched key from a mismatched one.
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace

import numpy as np
import pytest

from arrayscope.display.backends.pyqtgraph.tiles import (
    page_assembly_nbytes,
    resolve_page_backed_assembly,
)
from arrayscope.display.lod import LodInfo
from arrayscope.display.shader_mapping import ShaderMapping, ShaderScale
from arrayscope.display.model.frame import DisplayTilePayload, PageBackedPresentation
from arrayscope.display.pyramid import materialize_source_grid_pages, plan_source_grid_pages
from arrayscope.display.wgpu_imageview2d import (
    WgpuImageView2D,
    _wgpu_preparable_representation,
    wgpu_pack_saves_work,
    wgpu_packed_payload_texture,
)
from arrayscope.gpu.keys import COMPLEX_RG32F, SCALAR_R32F
from arrayscope.presentation.prepared_uploads import (
    PreparedUploadMailbox,
    cpu_mapping_preparation_variant,
    prepared_upload_key,
)

CONTENT = ("src-anchored", ("doc", 1), ("op", "identity"))
RECT = (0, 4, 0, 6)


def _page_backed_payload(*, tile_number: int = 0, source_id=("tile", 0), fill: float = 1.0):
    """A payload whose pixels must be assembled from pages before display."""

    values = np.full((RECT[1] - RECT[0], RECT[3] - RECT[2]), fill, dtype=np.float32)
    plans = plan_source_grid_pages(
        content_key=CONTENT,
        valid_source_rect_yx=RECT,
        reduction_yx=(0, 0),
        stored_page_shape=(2, 3),
        dtype="float32",
        representation=SCALAR_R32F,
        reducer="native",
    )
    pages = materialize_source_grid_pages(values, source_origin_yx=(0, 0), plans=plans)
    lod = LodInfo(0, 1, values.shape, values.shape, 0)
    return DisplayTilePayload(
        tile_number,
        tile_number,
        values,
        values,
        source_id,
        semantic_data=values,
        texture_data=values,
        lod=lod,
        page_backing=PageBackedPresentation(plans, pages, RECT, lod),
    )


# --- PyQtGraph: page assembly -------------------------------------------------


def test_pyqtgraph_takes_a_prepared_assembly_instead_of_assembling_inline():
    from arrayscope.display.backends.pyqtgraph.tiles import MontageTileLayer

    payload = _page_backed_payload()
    levels = (0.0, 2.0)
    mailbox = PreparedUploadMailbox()
    layer = MontageTileLayer.__new__(MontageTileLayer)
    layer._payload_prepare_ms = 0.0
    layer._prepared_uploads = mailbox
    layer._prepared_assembly_hits = 0

    prepared = resolve_page_backed_assembly(payload, levels=levels)
    mailbox.publish(
        0,
        prepared_upload_key(payload, cpu_mapping_preparation_variant(payload, levels)),
        prepared,
        nbytes=page_assembly_nbytes(prepared),
    )

    resolved = layer._resolve_payload(payload, levels=levels)

    assert resolved is prepared
    assert layer._prepared_assembly_hits == 1
    # Nothing was assembled on the calling thread.
    assert layer.consume_payload_prepare_ms() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("stale_source_id", "stale_levels"),
    [
        # A buffer prepared for a different payload identity.
        (("tile", 99), (0.0, 2.0)),
        # A buffer baked against different round levels (R3).
        (("tile", 0), (0.0, 5.0)),
    ],
)
def test_pyqtgraph_refuses_a_stale_assembly_and_assembles_inline(stale_source_id, stale_levels):
    """A stale prepared frame is dropped, never presented."""

    from arrayscope.display.backends.pyqtgraph.tiles import MontageTileLayer

    payload = _page_backed_payload(fill=1.0)
    committing_levels = (0.0, 2.0)
    mailbox = PreparedUploadMailbox()
    layer = MontageTileLayer.__new__(MontageTileLayer)
    layer._payload_prepare_ms = 0.0
    layer._prepared_uploads = mailbox
    layer._prepared_assembly_hits = 0

    other = _page_backed_payload(source_id=stale_source_id, fill=9.0)
    stale = resolve_page_backed_assembly(other, levels=stale_levels)
    # Published into THIS payload's slot under a key that does not match what
    # the commit will ask for -- exactly the race the mailbox exists to lose.
    mailbox.publish(
        0,
        prepared_upload_key(
            other,
            cpu_mapping_preparation_variant(other, stale_levels),
        ),
        stale,
        nbytes=page_assembly_nbytes(stale),
    )

    resolved = layer._resolve_payload(payload, levels=committing_levels)

    assert resolved is not stale
    assert layer._prepared_assembly_hits == 0
    assert mailbox.counters().stale == 1
    # The pixels are this payload's, not the stale buffer's.
    assert float(np.asarray(resolved.payload.image).ravel()[0]) == pytest.approx(1.0)
    # And it was genuinely assembled here.
    assert layer.consume_payload_prepare_ms() > 0.0


def test_pyqtgraph_assembles_inline_when_the_mailbox_is_empty():
    from arrayscope.display.backends.pyqtgraph.tiles import MontageTileLayer

    payload = _page_backed_payload()
    layer = MontageTileLayer.__new__(MontageTileLayer)
    layer._payload_prepare_ms = 0.0
    layer._prepared_uploads = PreparedUploadMailbox()
    layer._prepared_assembly_hits = 0

    resolved = layer._resolve_payload(payload, levels=(0.0, 2.0))

    assert resolved is not None
    assert layer._prepared_assembly_hits == 0
    assert float(np.asarray(resolved.payload.image).ravel()[0]) == pytest.approx(1.0)


# --- WGPU: upload packing -----------------------------------------------------


def test_wgpu_pack_is_pure_and_matches_what_a_worker_would_publish():
    """The worker and the inline path must produce the same plane."""

    payload = _page_backed_payload(fill=3.0)

    worker_result = wgpu_packed_payload_texture(payload, SCALAR_R32F)
    inline_result = wgpu_packed_payload_texture(payload, SCALAR_R32F)

    np.testing.assert_array_equal(worker_result, inline_result)
    assert worker_result.dtype == np.float32


def test_wgpu_takes_a_prepared_plane_only_under_its_own_representation():
    from arrayscope.display.wgpu_imageview2d import WgpuImageView2D

    payload = _page_backed_payload(fill=3.0)
    mailbox = PreparedUploadMailbox()
    view = WgpuImageView2D.__new__(WgpuImageView2D)
    view._prepared_tiled_uploads = mailbox

    prepared = wgpu_packed_payload_texture(payload, SCALAR_R32F)
    mailbox.publish(0, prepared_upload_key(payload, SCALAR_R32F), prepared)

    assert view._wgpu_payload_texture(payload, SCALAR_R32F) is prepared
    assert mailbox.counters().hits == 1


def test_wgpu_refuses_a_plane_prepared_for_another_representation():
    """Representation is part of the promise; a mismatch packs inline."""

    from arrayscope.display.wgpu_imageview2d import WgpuImageView2D

    payload = _page_backed_payload(fill=3.0)
    mailbox = PreparedUploadMailbox()
    view = WgpuImageView2D.__new__(WgpuImageView2D)
    view._prepared_tiled_uploads = mailbox

    prepared = wgpu_packed_payload_texture(payload, SCALAR_R32F)
    mailbox.publish(0, prepared_upload_key(payload, COMPLEX_RG32F), prepared)

    packed = view._wgpu_payload_texture(payload, SCALAR_R32F)

    assert packed is not prepared
    assert mailbox.counters().stale == 1
    np.testing.assert_array_equal(packed, wgpu_packed_payload_texture(payload, SCALAR_R32F))


def test_wgpu_refuses_a_plane_prepared_for_another_payload():
    from arrayscope.display.wgpu_imageview2d import WgpuImageView2D

    committing = _page_backed_payload(source_id=("tile", 0), fill=1.0)
    other = _page_backed_payload(source_id=("tile", 99), fill=9.0)
    mailbox = PreparedUploadMailbox()
    view = WgpuImageView2D.__new__(WgpuImageView2D)
    view._prepared_tiled_uploads = mailbox

    mailbox.publish(
        0,
        prepared_upload_key(other, SCALAR_R32F),
        wgpu_packed_payload_texture(other, SCALAR_R32F),
    )

    packed = view._wgpu_payload_texture(committing, SCALAR_R32F)

    assert mailbox.counters().stale == 1
    assert float(packed.ravel()[0]) == pytest.approx(1.0)


def test_the_representation_a_preparation_targets_matches_the_commit_plan():
    """Preparing under the wrong representation would only ever miss."""

    payload = _page_backed_payload()

    assert _wgpu_preparable_representation(payload, True) == SCALAR_R32F
    assert _wgpu_preparable_representation(payload, False) == SCALAR_R32F


def test_an_unpackable_payload_is_skipped_rather_than_guessed_at():
    class _Opaque:
        texture_data = None
        image = None

    assert _wgpu_preparable_representation(_Opaque(), False) is None


# --- Physical residency is not mapping residency ------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(
            {"levels": (0.0, 2.0)},
            {"levels": (0.0, 5.0)},
            id="levels",
        ),
        pytest.param(
            {"scale": ShaderScale.LINEAR},
            {"scale": ShaderScale.LOG},
            id="scale",
        ),
        pytest.param(
            {"scale": ShaderScale.SYMLOG, "symlog_constant": 1.0},
            {"scale": ShaderScale.SYMLOG, "symlog_constant": 4.0},
            id="symlog-constant",
        ),
        pytest.param(
            {"lut_identity": ("lut", "gray")},
            {"lut_identity": ("lut", "viridis")},
            id="lut",
        ),
        pytest.param(
            {"component": "real"},
            {"component": "abs"},
            id="complex-component",
        ),
    ],
)
def test_a_mapping_change_alone_changes_the_preparation_key(first, second):
    """`TileIdentity` is blind to the CPU mapping; the preparation key is not.

    Each of these changes the pixels a PyQtGraph assembly bakes while leaving
    semantic identity untouched. If any of them collided, a buffer baked under
    the old mapping could be taken for the new one — the torn frame the mailbox
    exists to prevent.
    """

    base = _page_backed_payload()
    levels_first = first.pop("levels", (0.0, 2.0))
    levels_second = second.pop("levels", (0.0, 2.0))
    one = dataclass_replace(base, shader_mapping=ShaderMapping(**first))
    other = dataclass_replace(base, shader_mapping=ShaderMapping(**second))

    assert prepared_upload_key(
        one, cpu_mapping_preparation_variant(one, levels_first)
    ) != prepared_upload_key(other, cpu_mapping_preparation_variant(other, levels_second))


def test_a_skipped_preparation_still_bakes_the_current_mapping_inline():
    """The residency filter may only cost a preparation, never a remap.

    Planning skips physically resident payloads, and physical residency says
    nothing about the mapping. So the case that matters is a source the backend
    already holds, presented through a mapping it was not baked under: the
    commit must bake the *current* mapping, exactly as it would with no
    preparation in the picture at all.
    """

    from arrayscope.display.backends.pyqtgraph.tiles import MontageTileLayer

    payload = _page_backed_payload(fill=3.0)
    mailbox = PreparedUploadMailbox()
    layer = MontageTileLayer.__new__(MontageTileLayer)
    layer._payload_prepare_ms = 0.0
    layer._prepared_uploads = mailbox
    layer._prepared_assembly_hits = 0

    # Nothing was prepared for this round: planning skipped it as resident.
    mailbox.note_resident()
    resolved = layer._resolve_payload(payload, levels=(0.0, 9.0))

    assert layer._prepared_assembly_hits == 0
    assert mailbox.counters().misses == 1
    assert layer.consume_payload_prepare_ms() > 0.0
    assert float(np.asarray(resolved.payload.image).ravel()[0]) == pytest.approx(3.0)


def test_a_fallback_payload_sharing_semantic_identity_keeps_its_own_key():
    """Two payloads can satisfy one target from different physical pages."""

    coarse = dataclass_replace(
        _page_backed_payload(source_id=("tile", "fallback")), tile_identity="same-target"
    )
    fine = dataclass_replace(
        _page_backed_payload(source_id=("tile", "exact")), tile_identity="same-target"
    )

    assert coarse.tile_identity == fine.tile_identity
    assert prepared_upload_key(coarse, "scalar") != prepared_upload_key(fine, "scalar")


# --- What is worth preparing at all ------------------------------------------


def test_a_contiguous_scalar_plane_is_not_worth_preparing():
    """The pack would hand back the array it was given.

    `pack_texture_data` on an already-contiguous float32 plane is
    `np.ascontiguousarray` over a contiguous array: the same object comes back.
    Preparing that ahead moves no work off the GUI thread, and it is not free —
    it takes a scheduler slot and holds a worker for as long as it runs, which
    is worker time the round's pixel-producing tasks could have had.
    """

    values = np.ascontiguousarray(np.zeros((4, 6), dtype=np.float32))
    payload = DisplayTilePayload(0, 0, values, values, ("tile", 0), texture_data=values)

    assert not wgpu_pack_saves_work(payload, SCALAR_R32F)
    # And the claim it rests on: packing really is the identity here.
    assert wgpu_packed_payload_texture(payload, SCALAR_R32F) is values


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.zeros((4, 6), dtype=np.float64), id="dtype-conversion"),
        pytest.param(np.zeros((4, 12), dtype=np.float32)[:, ::2], id="non-contiguous"),
    ],
)
def test_a_scalar_plane_needing_conversion_is_worth_preparing(values):
    payload = DisplayTilePayload(0, 0, values, values, ("tile", 0), texture_data=values)

    assert wgpu_pack_saves_work(payload, SCALAR_R32F)
    assert wgpu_packed_payload_texture(payload, SCALAR_R32F) is not values


def test_a_complex_plane_is_worth_preparing():
    """Complex → RG32F always allocates: real and imaginary are interleaved."""

    values = np.zeros((4, 6), dtype=np.complex64)
    payload = DisplayTilePayload(0, 0, values, values, ("tile", 0), texture_data=values)

    assert wgpu_pack_saves_work(payload, COMPLEX_RG32F)


def test_a_multi_page_payload_is_worth_preparing_even_when_its_pack_is_trivial():
    """Assembly is the work, and it happens before the representation pack."""

    payload = _page_backed_payload()

    assert wgpu_pack_saves_work(payload, SCALAR_R32F)


def test_the_wgpu_planner_skips_passthrough_payloads_and_counts_them():
    """The skip is visible in the counters, not silent."""

    values = np.ascontiguousarray(np.zeros((4, 6), dtype=np.float32))
    payload = DisplayTilePayload(3, 3, values, values, ("tile", 3), texture_data=values)
    view = _StubWgpuView()

    rows = WgpuImageView2D.tiledUploadPreparations(
        view, {3: payload}, levels=(0.0, 1.0), rgb_already_windowed=True
    )

    assert rows == ()
    counters = view.preparedTiledUploads.counters()
    assert counters.skipped_no_work == 1
    assert counters.submitted == 0


def test_the_wgpu_planner_still_prepares_a_payload_with_real_work():
    payload = _page_backed_payload(tile_number=2, source_id=("tile", 2))
    view = _StubWgpuView()

    rows = WgpuImageView2D.tiledUploadPreparations(
        view, {2: payload}, levels=(0.0, 1.0), rgb_already_windowed=True
    )

    assert len(rows) == 1
    slot, key, prepare = rows[0]
    assert slot == 2
    prepare()
    assert view.preparedTiledUploads.take(2, key) is not None


class _StubWgpuView:
    """Just the mailbox the planner touches; no GPU device is involved."""

    def __init__(self) -> None:
        self._prepared_tiled_uploads = PreparedUploadMailbox()

    @property
    def preparedTiledUploads(self) -> PreparedUploadMailbox:
        return self._prepared_tiled_uploads


# --- Physical residency is not mapping residency ------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(
            {"levels": (0.0, 2.0)},
            {"levels": (0.0, 5.0)},
            id="levels",
        ),
        pytest.param(
            {"scale": ShaderScale.LINEAR},
            {"scale": ShaderScale.LOG},
            id="scale",
        ),
        pytest.param(
            {"scale": ShaderScale.SYMLOG, "symlog_constant": 1.0},
            {"scale": ShaderScale.SYMLOG, "symlog_constant": 4.0},
            id="symlog-constant",
        ),
        pytest.param(
            {"lut_identity": ("lut", "gray")},
            {"lut_identity": ("lut", "viridis")},
            id="lut",
        ),
        pytest.param(
            {"component": "real"},
            {"component": "abs"},
            id="complex-component",
        ),
    ],
)
def test_a_mapping_change_alone_changes_the_preparation_key(first, second):
    """`TileIdentity` is blind to the CPU mapping; the preparation key is not.

    Each of these changes the pixels a PyQtGraph assembly bakes while leaving
    semantic identity untouched. If any of them collided, a buffer baked under
    the old mapping could be taken for the new one — the torn frame the mailbox
    exists to prevent.
    """

    base = _page_backed_payload()
    levels_first = first.pop("levels", (0.0, 2.0))
    levels_second = second.pop("levels", (0.0, 2.0))
    one = dataclass_replace(base, shader_mapping=ShaderMapping(**first))
    other = dataclass_replace(base, shader_mapping=ShaderMapping(**second))

    assert prepared_upload_key(
        one, cpu_mapping_preparation_variant(one, levels_first)
    ) != prepared_upload_key(other, cpu_mapping_preparation_variant(other, levels_second))


def test_a_skipped_preparation_still_bakes_the_current_mapping_inline():
    """The residency filter may only cost a preparation, never a remap.

    Planning skips physically resident payloads, and physical residency says
    nothing about the mapping. So the case that matters is a source the backend
    already holds, presented through a mapping it was not baked under: the
    commit must bake the *current* mapping, exactly as it would with no
    preparation in the picture at all.
    """

    from arrayscope.display.backends.pyqtgraph.tiles import MontageTileLayer

    payload = _page_backed_payload(fill=3.0)
    mailbox = PreparedUploadMailbox()
    layer = MontageTileLayer.__new__(MontageTileLayer)
    layer._payload_prepare_ms = 0.0
    layer._prepared_uploads = mailbox
    layer._prepared_assembly_hits = 0

    # Nothing was prepared for this round: planning skipped it as resident.
    mailbox.note_resident()
    resolved = layer._resolve_payload(payload, levels=(0.0, 9.0))

    assert layer._prepared_assembly_hits == 0
    assert mailbox.counters().misses == 1
    assert layer.consume_payload_prepare_ms() > 0.0
    assert float(np.asarray(resolved.payload.image).ravel()[0]) == pytest.approx(3.0)


def test_a_fallback_payload_sharing_semantic_identity_keeps_its_own_key():
    """Two payloads can satisfy one target from different physical pages."""

    coarse = dataclass_replace(
        _page_backed_payload(source_id=("tile", "fallback")), tile_identity="same-target"
    )
    fine = dataclass_replace(
        _page_backed_payload(source_id=("tile", "exact")), tile_identity="same-target"
    )

    assert coarse.tile_identity == fine.tile_identity
    assert prepared_upload_key(coarse, "scalar") != prepared_upload_key(fine, "scalar")
