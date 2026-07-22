"""Real-GPU parity: native ASTC pools sampled in the live wgpu render path.

The integrated Intel adapter advertises ``texture-compression-astc`` (the discrete
NVIDIA one does not), and ASTC is the format the topology policy selects there.
This is the ASTC twin of ``test_wgpu_compressed_render.py``: on an ASTC-capable
integrated adapter it renders a montage through
:class:`~arrayscope.gpu.wgpu_executor.WgpuPlaneExecutor` twice -- once byte-identical
raw, once with compression engaged -- and

* proves the pages are actually in an **ASTC** pool (``codec_family == "astc"`` and
  the pool's texture format is ``astc-BxB-unorm``), not a BC pool or a raw fallback;
* asserts the engaged framebuffer matches raw within a PSNR-justified tolerance
  (ASTC is lossy; the loss is a number);
* checks auto-level bounds + histogram from the ASTC pages match raw within the
  same tolerance the BC path uses.

Skips cleanly where no ASTC adapter is present (e.g. a discrete-only or headless
software host), so CI stays green.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from arrayscope.gpu.astc_codec import astc_available
from arrayscope.gpu.command_protocol import (
    BindContentPlanes,
    ContentPlane,
    DispatchHistogram,
    DisplayMapping,
    EnsureChunkResident,
    FrameSubmission,
    PresentGeneration,
    SetDisplayMapping,
    TileInstance,
    UpdateTileInstances,
)
from arrayscope.gpu.keys import COMPLEX_RG32F, SCALAR_R32F
from arrayscope.gpu.wgpu_executor import PAGE, WgpuPlaneExecutor, plane_chunk_key


def _astc_device():
    if not astc_available():
        return None
    try:
        import wgpu  # noqa: F401

        from arrayscope.gpu.bc_gpu import create_compute_device
    except Exception:
        return None
    # ASTC lives on the integrated adapter here; low-power selects it.
    with contextlib.suppress(Exception):
        return create_compute_device("low-power", features=["texture-compression-astc"])
    return None


_DEVICE = _astc_device()
pytestmark = pytest.mark.skipif(
    _DEVICE is None, reason="no ASTC-capable wgpu adapter on this machine"
)

CANVAS = (384, 384)
BLOCK = (4, 4)
NX = 2


def _scalar_tile(seed: int = 0) -> np.ndarray:
    y, x = np.mgrid[0:PAGE, 0:PAGE]
    return (np.sin((x + seed) / 40.0) * np.cos((y - seed) / 35.0) + x / float(PAGE)).astype(
        np.float32
    )


def _complex_tile(seed: int = 0) -> np.ndarray:
    y, x = np.mgrid[0:PAGE, 0:PAGE]
    re = np.sin((x + seed) / 40.0) * np.cos(y / 35.0)
    im = np.cos(x / 33.0) * np.sin((y + seed) / 50.0)
    return (re + 1j * im).astype(np.complex64)


def _frame_psnr(engaged: np.ndarray, raw: np.ndarray) -> tuple[float, float]:
    a = engaged[..., :3].astype(np.float64)
    b = raw[..., :3].astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse <= 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)
    over = float(np.mean(np.any(np.abs(a - b) > 4, axis=-1)))
    return psnr, over


def _render_scalar(mode: str):
    ex = WgpuPlaneExecutor(
        (PAGE, NX * PAGE), max_lod=0, target_size=CANVAS, device=_DEVICE,
        pool_layers={SCALAR_R32F: NX + 1}, compressed_textures=mode, astc_block=BLOCK,
    )
    doc, op = "astc-scalar", "op"
    plane = ContentPlane(doc, op, (PAGE, NX * PAGE), max_lod=0, representation=SCALAR_R32F)
    keys = [
        plane_chunk_key(doc, op, 0, cx, 0, dtype="float32", representation=SCALAR_R32F,
                        plane_shape=(PAGE, NX * PAGE))
        for cx in range(NX)
    ]
    ensures = [EnsureChunkResident(keys[cx], _scalar_tile(cx * 11)) for cx in range(NX)]
    ex.submit(FrameSubmission(0, [BindContentPlanes((plane,)), *ensures]))
    tile = TileInstance(
        (0.0, 0.0, 1.0, 1.0), (0.0, 0.0), (float(NX * PAGE), float(PAGE)), 0, plane_index=0
    )
    rep = ex.submit(FrameSubmission(1, (
        SetDisplayMapping(DisplayMapping("real", -1.0, 2.0)),
        UpdateTileInstances((tile,)),
        DispatchHistogram(tuple(keys), bins=64),
        PresentGeneration(1),
    )))
    rep.wait_completed()
    return ex, keys, ex.read_target(), rep


def _render_complex(mode: str):
    ex = WgpuPlaneExecutor(
        (PAGE, NX * PAGE), max_lod=0, target_size=CANVAS, device=_DEVICE,
        pool_layers={COMPLEX_RG32F: NX + 1}, compressed_textures=mode, astc_block=BLOCK,
    )
    doc, op = "astc-complex", "op"
    plane = ContentPlane(doc, op, (PAGE, NX * PAGE), max_lod=0, representation=COMPLEX_RG32F)
    keys = [
        plane_chunk_key(doc, op, 0, cx, 0, dtype="complex64", representation=COMPLEX_RG32F,
                        plane_shape=(PAGE, NX * PAGE))
        for cx in range(NX)
    ]
    ensures = [EnsureChunkResident(keys[cx], _complex_tile(cx * 11)) for cx in range(NX)]
    ex.submit(FrameSubmission(0, [BindContentPlanes((plane,)), *ensures]))
    tile = TileInstance(
        (0.0, 0.0, 1.0, 1.0), (0.0, 0.0), (float(NX * PAGE), float(PAGE)), 0, plane_index=0
    )
    rep = ex.submit(FrameSubmission(1, (
        SetDisplayMapping(DisplayMapping("magnitude", -1.5, 1.5)),
        UpdateTileInstances((tile,)),
        DispatchHistogram(tuple(keys), bins=64),
        PresentGeneration(1),
    )))
    rep.wait_completed()
    return ex, keys, ex.read_target(), rep


def _is_integrated(device) -> bool:
    try:
        return "discrete" not in str(dict(device.adapter.info).get("adapter_type", "")).lower()
    except Exception:
        return True


def test_scalar_montage_uses_an_astc_pool_and_matches_raw():
    if not _is_integrated(_DEVICE):
        pytest.skip("ASTC device is not the integrated adapter")
    engaged, keys, frame_c, rep_c = _render_scalar("auto")
    raw, _keys, frame_raw, rep_raw = _render_scalar("off")

    # Loud channel: ASTC, not BC, not a raw fallback.
    assert engaged.codec_engaged
    assert engaged.codec_family == "astc"
    assert engaged.codec_pool_format(SCALAR_R32F) == f"astc-{BLOCK[0]}x{BLOCK[1]}-unorm"
    assert engaged.compressed_uploads_total == NX
    assert raw.compressed_uploads_total == 0
    for key in keys:
        assert engaged.page_is_compressed(key)
        assert engaged.page_table.lookup(key).pool_id == "wgpu-scalar_r32f-bc-pool"
        assert not raw.page_is_compressed(key)

    psnr, over = _frame_psnr(frame_c, frame_raw)
    assert psnr >= 45.0, f"ASTC scalar render PSNR {psnr:.1f} dB below the 45 dB floor"
    assert over < 0.01

    # Auto-level bounds from the ASTC pages match raw within the BC tolerance.
    b_c = rep_c.histogram_bounds.get(2)
    b_raw = rep_raw.histogram_bounds.get(2)
    assert b_c is not None
    assert b_raw is not None
    span = (b_raw[1] - b_raw[0]) or 1.0
    assert abs(b_c[0] - b_raw[0]) <= 0.02 * span
    assert abs(b_c[1] - b_raw[1]) <= 0.02 * span


def test_complex_montage_uses_an_astc_pool_and_matches_raw():
    if not _is_integrated(_DEVICE):
        pytest.skip("ASTC device is not the integrated adapter")
    engaged, keys, frame_c, _rep_c = _render_complex("auto")
    _raw, _keys, frame_raw, _rep_raw = _render_complex("off")

    assert engaged.codec_family == "astc"
    assert engaged.codec_pool_format(COMPLEX_RG32F) == f"astc-{BLOCK[0]}x{BLOCK[1]}-unorm"
    assert engaged.compressed_uploads_total == NX
    for key in keys:
        assert engaged.page_is_compressed(key)

    psnr, over = _frame_psnr(frame_c, frame_raw)
    assert psnr >= 45.0, f"ASTC magnitude render PSNR {psnr:.1f} dB below the 45 dB floor"
    assert over < 0.01


def test_off_mode_is_byte_identical_and_not_astc():
    off, _keys, frame_off, _rep = _render_scalar("off")
    _raw, _k, frame_raw, _r = _render_scalar("off")
    assert not off.codec_engaged
    assert off.codec_family == "none"
    assert np.array_equal(frame_off, frame_raw)


def test_lod_generation_from_astc_children_does_not_crash():
    """Point 5: an LOD whose children live in ASTC pools reduces on the CPU
    (decode -> box-mean -> store raw) exactly like the BC path, no crash."""

    from arrayscope.gpu.command_protocol import GenerateLodPages

    if not _is_integrated(_DEVICE):
        pytest.skip("ASTC device is not the integrated adapter")
    y, x = np.mgrid[0 : 2 * PAGE, 0 : 2 * PAGE]
    src = (np.sin(x / 50.0) * np.cos(y / 47.0)).astype(np.float32)
    ex = WgpuPlaneExecutor(
        src.shape, max_lod=1, target_size=(64, 64), device=_DEVICE,
        pool_layers={SCALAR_R32F: 8}, compressed_textures="on", astc_block=BLOCK,
    )
    doc, op = "astc-lod", "op"
    sources = tuple(
        plane_chunk_key(doc, op, 0, cx, cy, dtype="float32", representation=SCALAR_R32F,
                        plane_shape=src.shape)
        for cy in range(2)
        for cx in range(2)
    )
    dest = plane_chunk_key(doc, op, 1, 0, 0, dtype="float32", representation=SCALAR_R32F,
                           plane_shape=src.shape)
    ex.submit(FrameSubmission(1, (
        BindContentPlanes((ContentPlane(doc, op, src.shape, max_lod=1, representation=SCALAR_R32F),)),
        *(
            EnsureChunkResident(k, src[cy * PAGE : (cy + 1) * PAGE, cx * PAGE : (cx + 1) * PAGE])
            for k, (cy, cx) in zip(sources, ((0, 0), (0, 1), (1, 0), (1, 1)), strict=True)
        ),
    )))
    assert ex.codec_family == "astc"
    assert all(ex.page_is_compressed(k) for k in sources)

    gen = ex.submit(FrameSubmission(2, (GenerateLodPages(sources, dest),)))
    gen.wait_completed()
    assert gen.lod_pages_generated == (dest,)
    assert ex.lod_compressed_source_reductions_total == 1
    # The generated LOD page is stored RAW (codec 0), matching the BC path.
    assert not ex.page_is_compressed(dest)

    got = ex.read_resident_page(dest)
    decoded = np.zeros((2 * PAGE, 2 * PAGE), np.float32)
    for k, (cy, cx) in zip(sources, ((0, 0), (0, 1), (1, 0), (1, 1)), strict=True):
        decoded[cy * PAGE : (cy + 1) * PAGE, cx * PAGE : (cx + 1) * PAGE] = ex.read_resident_page(k)
    expected = decoded.reshape(PAGE, 2, PAGE, 2).mean(axis=(1, 3)).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def teardown_module(module):
    with contextlib.suppress(Exception):
        _DEVICE._destroy() if hasattr(_DEVICE, "_destroy") else None
