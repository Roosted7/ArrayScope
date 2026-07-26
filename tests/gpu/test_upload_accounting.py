"""Per-(lod level, representation) upload accounting — pure bookkeeping.

``uploads_total``/``texture_upload_bytes_total`` alone cannot say which page
keys a run uploaded.  The 2026-07-26 preview-LOD dossier had to infer it (1088
uploads on a 272-tile montage = 4 native pages per tile) because no per-level
counter existed.  These tests pin the counter that removes the inference; the
executor's device path is exercised by the real-GPU suites.
"""

from __future__ import annotations

import numpy as np

from arrayscope.gpu.keys import COMPLEX_RG32F, SCALAR_R32F
from arrayscope.gpu.wgpu_executor import PAGE, WgpuPlaneExecutor, plane_chunk_key

# Two native pages on a side, so a level-0 grid really has four members.
PLANE = (2 * PAGE, 2 * PAGE)


def _accounting() -> WgpuPlaneExecutor:
    """An executor with only its upload counters initialized (no device)."""

    executor = WgpuPlaneExecutor.__new__(WgpuPlaneExecutor)
    executor._uploads_total = 0
    executor._texture_upload_bytes_total = 0
    executor._uploads_by_level_rep = {}
    executor._upload_bytes_by_level_rep = {}
    return executor


def _key(
    level: int,
    *,
    chunk_x: int = 0,
    chunk_y: int = 0,
    representation: str = SCALAR_R32F,
    generation: object = ("doc", 0),
):
    return plane_chunk_key(
        generation,
        (),
        level,
        chunk_x,
        chunk_y,
        dtype="float32",
        representation=representation,
        plane_shape=PLANE,
    )


def test_no_uploads_reports_no_rows_rather_than_zero_rows():
    # A level with no row means "nothing uploaded there", never "0 uploaded":
    # a permanently-present zero reads as evidence it is not.
    assert _accounting().upload_rows_by_level() == ()


def test_uploads_are_attributed_to_their_key_level_and_representation():
    executor = _accounting()
    page_bytes = PAGE * PAGE * 4
    for chunk_y in (0, 1):  # the native page grid of one plane
        for chunk_x in (0, 1):
            executor._record_upload(_key(0, chunk_x=chunk_x, chunk_y=chunk_y), page_bytes)
    executor._record_upload(_key(2), page_bytes)
    executor._record_upload(_key(2, representation=COMPLEX_RG32F), page_bytes * 2)

    assert executor.uploads_total == 6
    assert executor.texture_upload_bytes_total == page_bytes * 7
    assert executor.upload_rows_by_level() == (
        {
            "level": 0,
            "representation": SCALAR_R32F,
            "uploads": 4,
            "bytes": page_bytes * 4,
        },
        {
            "level": 2,
            "representation": COMPLEX_RG32F,
            "uploads": 1,
            "bytes": page_bytes * 2,
        },
        {
            "level": 2,
            "representation": SCALAR_R32F,
            "uploads": 1,
            "bytes": page_bytes,
        },
    )


def test_per_level_totals_reconcile_with_the_aggregate_totals():
    # The breakdown is a partition of the aggregate, not a parallel estimate:
    # a level missing from the rows is a level that uploaded nothing.
    executor = _accounting()
    rng = np.random.default_rng(0)
    for index in range(50):
        key = _key(int(index % 5), generation=("doc", index))
        executor._record_upload(key, int(rng.integers(1, 4096)))

    rows = executor.upload_rows_by_level()
    assert sum(int(row["uploads"]) for row in rows) == executor.uploads_total == 50
    assert sum(int(row["bytes"]) for row in rows) == executor.texture_upload_bytes_total
    assert [int(row["level"]) for row in rows] == [0, 1, 2, 3, 4]
