"""G5 live-ladder page-target contract (ADR 0056).

The ladder must name desired logical pages before the VisPy pool resolves
them to physical residency.  Planning is a pure source-grid geometry step:
no Qt receiver, timer, scheduler, cache claim, or upload belongs here.
"""

from __future__ import annotations

from arrayscope.gpu import ChunkLod, DataChunkKey, PageSlot, PageTable
from arrayscope.render import lod as render_lod


CONTENT_KEY = ("src-anchored", ("doc", 7), ("request", "window-free"))
PAGE_SHAPE = (256, 256)


def page_targets(source_rect, *, reduction=(1, 1)):
    """Call the intended Qt-free render-LOD target planner."""

    return render_lod.plan_lod_page_targets(
        content_key=CONTENT_KEY,
        source_rect=source_rect,
        reduction=reduction,
        stored_page_shape=PAGE_SHAPE,
        dtype="float32",
        representation="scalar_r32f",
        reducer="mean",
    )


def key_rect(key: DataChunkKey) -> tuple[int, int, int, int]:
    y0, x0 = key.chunk_origin
    height, width = key.chunk_shape
    return (y0, y0 + height, x0, x0 + width)


def test_ladder_target_decomposes_on_canonical_global_source_grid():
    targets = page_targets((0, 512, 101, 1125))

    assert targets
    assert all(isinstance(key, DataChunkKey) for key in targets)
    assert {key_rect(key) for key in targets} == {
        (0, 512, 101, 512),
        (0, 512, 512, 1024),
        (0, 512, 1024, 1125),
    }
    for key in targets:
        assert key.document_generation == ("doc", 7)
        assert key.operation_key == (
            "source-grid-page",
            1,
            ("request", "window-free"),
        )
        assert key.dtype == "float32"
        assert key.representation == "scalar_r32f"
        assert key.lod.reduction == (1, 1)
        assert key.lod.reducer == "mean"


def test_factor_two_shifted_windows_share_aligned_interior_not_boundaries():
    first = set(page_targets((0, 512, 101, 1125)))
    shifted = set(page_targets((0, 512, 102, 1126)))

    shared = first & shifted
    assert {key_rect(key) for key in shared} == {(0, 512, 512, 1024)}
    assert {key_rect(key) for key in first - shifted} == {
        (0, 512, 101, 512),
        (0, 512, 1024, 1125),
    }
    assert {key_rect(key) for key in shifted - first} == {
        (0, 512, 102, 512),
        (0, 512, 1024, 1126),
    }


def test_desired_mean_page_identity_stays_separate_from_coarse_resolution():
    target = next(
        key
        for key in page_targets((0, 512, 101, 1125))
        if key_rect(key) == (0, 512, 512, 1024)
    )
    coarse = DataChunkKey(
        document_generation=target.document_generation,
        operation_key=target.operation_key,
        lod=ChunkLod(reduction=(2, 2), reducer="mean"),
        chunk_origin=(0, 0),
        chunk_shape=(1024, 1536),
        dtype=target.dtype,
        representation=target.representation,
    )
    table = PageTable()
    coarse_slot = PageSlot("vispy-atlas", 0, 3)
    table.bind(coarse, coarse_slot, nbytes=256 * 256 * 4)

    resolution = table.resolve(target)

    assert resolution is not None
    # Semantic target remains the demanded factor-two mean page.
    assert resolution.target_key == target
    assert resolution.target_key.lod.reduction == (1, 1)
    assert resolution.target_key.lod.reducer == "mean"
    # Physical truth separately names the sampled factor-four fallback.
    assert resolution.actual_key == coarse
    assert resolution.actual_key != resolution.target_key
    assert resolution.actual_key.lod.reduction == (2, 2)
    assert resolution.slot == coarse_slot


def test_page_target_planning_is_a_deterministic_value_transform():
    """The seam has no session/scheduler input and creates no residency."""

    source_rect = (13, 901, 29, 1301)
    first = page_targets(source_rect)
    second = page_targets(source_rect)

    assert first == second
    assert first is not second
    assert all(isinstance(key, DataChunkKey) for key in first)
