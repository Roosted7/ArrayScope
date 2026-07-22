"""Real-GPU gate: histogram / auto-level bounds over native BC pools (G7 Phase B).

Companion to ``test_wgpu_compressed_render.py`` (which covers the *render* path).
This covers the *compute* path -- the GPU histogram and auto-level bounds -- and
the LOD-from-compressed handling, on a BC-capable adapter.  It proves:

* **Path A** (``histogram_codec_mode="gpu_compressed"``): the compute shaders
  sample the BC pools, so a compressed montage's histogram/auto-range have full
  coverage (no ``histogram_missing``) and the bounds match the raw-pool result
  within a stated tolerance -- while compression is provably engaged (the page
  table's BC ``pool_id`` and ``compressed_uploads_total``).
* **Path B posture** (``histogram_codec_mode="skip"``): the compute excludes
  compressed pages and reports them ``histogram_missing`` (their exact stats come
  from the CPU semantic plane / full-population refinement, not lossy texels).
* the OFF/raw histogram is byte-identical to a raw executor; and
* LOD generation from compressed children never crashes and yields a valid raw
  LOD page.

Skips cleanly where no BC adapter is present.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from arrayscope.gpu.command_protocol import (
    BindContentPlanes,
    ContentPlane,
    DispatchHistogram,
    DisplayMapping,
    EnsureChunkResident,
    FrameSubmission,
    GenerateLodPages,
    SetDisplayMapping,
)
from arrayscope.gpu.keys import COMPLEX_RG32F, SCALAR_R32F
from arrayscope.gpu.wgpu_executor import PAGE, WgpuPlaneExecutor, plane_chunk_key


def _bc_device():
    try:
        import wgpu  # noqa: F401

        from arrayscope.gpu.bc_gpu import create_compute_device
    except Exception:
        return None
    try:
        return create_compute_device("high-performance", features=["texture-compression-bc"])
    except Exception:
        with contextlib.suppress(Exception):
            return create_compute_device("low-power", features=["texture-compression-bc"])
        return None


_DEVICE = _bc_device()
pytestmark = pytest.mark.skipif(
    _DEVICE is None, reason="no BC-capable wgpu adapter on this machine"
)

BINS = 128


def _scalar_tile(seed: int) -> np.ndarray:
    y, x = np.mgrid[0:PAGE, 0:PAGE]
    return (np.sin((x + seed) / 40.0) * np.cos((y - seed) / 35.0) + x / float(PAGE)).astype(
        np.float32
    )


def _complex_tile() -> np.ndarray:
    y, x = np.mgrid[0:PAGE, 0:PAGE]
    re = np.sin(x / 40.0) * np.cos(y / 35.0)
    im = np.cos(x / 33.0) * np.sin(y / 50.0)
    return (re + 1j * im).astype(np.complex64)


def _scalar_executor(mode: str, hist_mode: str):
    ex = WgpuPlaneExecutor(
        (PAGE, 2 * PAGE),
        max_lod=0,
        target_size=(64, 64),
        device=_DEVICE,
        pool_layers={SCALAR_R32F: 4},
        compressed_textures=mode,
        histogram_codec_mode=hist_mode,
    )
    doc, op = "levels-scalar", "op"
    plane = ContentPlane(doc, op, (PAGE, 2 * PAGE), max_lod=0, representation=SCALAR_R32F)
    keys = [
        plane_chunk_key(
            doc,
            op,
            0,
            cx,
            0,
            dtype="float32",
            representation=SCALAR_R32F,
            plane_shape=(PAGE, 2 * PAGE),
        )
        for cx in range(2)
    ]
    ex.submit(
        FrameSubmission(
            0,
            [
                BindContentPlanes((plane,)),
                *(EnsureChunkResident(keys[cx], _scalar_tile(cx * 7)) for cx in range(2)),
            ],
        )
    )
    return ex, keys


def _dispatch(ex, keys, *, mode="real", lo=None, hi=None):
    report = ex.submit(
        FrameSubmission(
            1,
            (
                SetDisplayMapping(DisplayMapping(mode, 0.0, 1.0)),
                DispatchHistogram(
                    tuple(keys),
                    bins=BINS,
                    lo=lo,
                    hi=hi,
                    mode=mode,
                    scale="linear",
                    symlog_constant=0.0,
                ),
            ),
        )
    )
    report.wait_completed()
    result = report.histograms[1]
    if hasattr(result, "resolve"):
        counts, bounds = result.resolve()
    else:
        counts, bounds = result, report.histogram_bounds.get(1)
    return np.asarray(counts, np.int64), bounds, report.histogram_missing.get(1, ())


def test_gpu_histogram_over_bc_pool_covers_all_pages_and_matches_raw_bounds():
    engaged, ekeys = _scalar_executor("on", "gpu_compressed")
    raw, rkeys = _scalar_executor("off", "skip")

    # Compression provably engaged -- the loud channels, not a silent fallback.
    assert engaged.codec_engaged
    assert engaged.compressed_uploads_total == 2
    for key in ekeys:
        assert engaged.page_is_compressed(key)
        assert engaged.page_table.lookup(key).pool_id == "wgpu-scalar_r32f-bc-pool"

    counts_bc, bounds_bc, missing_bc = _dispatch(engaged, ekeys)
    counts_raw, bounds_raw, _missing_raw = _dispatch(raw, rkeys)

    # Path A gives FULL coverage -- no page reported missing, every texel binned.
    assert missing_bc == ()
    assert int(counts_bc.sum()) == 2 * PAGE * PAGE
    assert int(counts_raw.sum()) == 2 * PAGE * PAGE

    # Auto-level bounds from the lossy BC texels match the raw bounds within a
    # tolerance justified by BC4's measured display quality.
    span = bounds_raw[1] - bounds_raw[0]
    assert abs(bounds_bc[0] - bounds_raw[0]) <= 5e-3 * span
    assert abs(bounds_bc[1] - bounds_raw[1]) <= 5e-3 * span

    # Histogram shape is close (lossy texels shift a few samples across bins).
    per_bin = int(np.abs(counts_bc - counts_raw).max())
    assert per_bin < 0.01 * counts_raw.sum(), f"per-bin drift {per_bin} too large"


def test_gpu_histogram_skip_mode_reports_compressed_pages_missing():
    engaged, ekeys = _scalar_executor("on", "skip")
    counts, bounds, missing = _dispatch(engaged, ekeys)
    # Every page is compressed, so Path B posture reports them all partial and
    # never reads a lossy texel; the exact stats come from the CPU semantic path.
    assert set(missing) == set(ekeys)
    assert int(counts.sum()) == 0
    assert bounds is None


def test_complex_bc5_histogram_bounds_match_raw_within_tolerance():
    ex = WgpuPlaneExecutor(
        (PAGE, PAGE),
        max_lod=0,
        target_size=(64, 64),
        device=_DEVICE,
        pool_layers={COMPLEX_RG32F: 4},
        compressed_textures="on",
        histogram_codec_mode="gpu_compressed",
    )
    raw = WgpuPlaneExecutor(
        (PAGE, PAGE),
        max_lod=0,
        target_size=(64, 64),
        device=_DEVICE,
        pool_layers={COMPLEX_RG32F: 4},
        compressed_textures="off",
    )
    doc, op = "levels-complex", "op"
    for e in (ex, raw):
        plane = ContentPlane(doc, op, (PAGE, PAGE), max_lod=0, representation=COMPLEX_RG32F)
        key = plane_chunk_key(
            doc,
            op,
            0,
            0,
            0,
            dtype="complex64",
            representation=COMPLEX_RG32F,
            plane_shape=(PAGE, PAGE),
        )
        e.submit(
            FrameSubmission(
                0, [BindContentPlanes((plane,)), EnsureChunkResident(key, _complex_tile())]
            )
        )
        e._levels_key = key

    assert ex.compressed_uploads_total == 1
    assert ex.page_is_compressed(ex._levels_key)

    _cbc, bounds_bc, missing_bc = _dispatch(ex, [ex._levels_key], mode="magnitude")
    _craw, bounds_raw, _m = _dispatch(raw, [raw._levels_key], mode="magnitude")
    assert missing_bc == ()
    span = bounds_raw[1] - bounds_raw[0]
    assert abs(bounds_bc[0] - bounds_raw[0]) <= 5e-3 * span
    assert abs(bounds_bc[1] - bounds_raw[1]) <= 5e-3 * span


def test_off_histogram_is_byte_identical_to_raw_executor():
    a, akeys = _scalar_executor("off", "skip")
    b, bkeys = _scalar_executor("off", "gpu_compressed")  # inert when off
    ca, ba, _ = _dispatch(a, akeys)
    cb, bb, _ = _dispatch(b, bkeys)
    assert np.array_equal(ca, cb)
    assert ba == bb


def test_lod_generation_from_compressed_children_does_not_crash():
    y, x = np.mgrid[0 : 2 * PAGE, 0 : 2 * PAGE]
    src = (np.sin(x / 50.0) * np.cos(y / 47.0)).astype(np.float32)
    ex = WgpuPlaneExecutor(
        src.shape,
        max_lod=1,
        target_size=(64, 64),
        device=_DEVICE,
        pool_layers={SCALAR_R32F: 8},
        compressed_textures="on",
    )
    doc, op = "levels-lod", "op"
    sources = tuple(
        plane_chunk_key(
            doc, op, 0, cx, cy, dtype="float32", representation=SCALAR_R32F, plane_shape=src.shape
        )
        for cy in range(2)
        for cx in range(2)
    )
    dest = plane_chunk_key(
        doc, op, 1, 0, 0, dtype="float32", representation=SCALAR_R32F, plane_shape=src.shape
    )
    ex.submit(
        FrameSubmission(
            1,
            (
                BindContentPlanes(
                    (ContentPlane(doc, op, src.shape, max_lod=1, representation=SCALAR_R32F),)
                ),
                *(
                    EnsureChunkResident(
                        k, src[cy * PAGE : (cy + 1) * PAGE, cx * PAGE : (cx + 1) * PAGE]
                    )
                    for k, (cy, cx) in zip(sources, ((0, 0), (0, 1), (1, 0), (1, 1)), strict=True)
                ),
            ),
        )
    )
    assert all(ex.page_is_compressed(k) for k in sources)

    gen = ex.submit(FrameSubmission(2, (GenerateLodPages(sources, dest),)))
    gen.wait_completed()
    assert gen.lod_pages_generated == (dest,)
    assert ex.lod_compressed_source_reductions_total == 1
    # The generated LOD page is stored RAW (codec 0), so it renders/reduces
    # through the exact integer-coord paths downstream.
    assert not ex.page_is_compressed(dest)

    # It equals the box-mean of the (already lossy) decoded children -- exact
    # w.r.t. what the children physically hold.
    got = ex.read_resident_page(dest)
    decoded = np.zeros((2 * PAGE, 2 * PAGE), np.float32)
    for k, (cy, cx) in zip(sources, ((0, 0), (0, 1), (1, 0), (1, 1)), strict=True):
        decoded[cy * PAGE : (cy + 1) * PAGE, cx * PAGE : (cx + 1) * PAGE] = ex.read_resident_page(k)
    expected = decoded.reshape(PAGE, 2, PAGE, 2).mean(axis=(1, 3)).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def teardown_module(module):
    with contextlib.suppress(Exception):
        _DEVICE._destroy() if hasattr(_DEVICE, "_destroy") else None
