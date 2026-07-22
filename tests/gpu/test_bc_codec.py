"""G7 Phase B: BC/ASTC texture codecs (offscreen) + bitpack lossless fallback.

Pure-CPU proofs -- no GPU device is created here.  They pin: BC4/BC5 round-trip
quality and the decline policy, complex DISPLAY-domain quality (re/im storage,
no phase-wrap artifact), ASTC (skipped cleanly when the optional dep is absent),
and the retained lossless narrow-int ``bitpack`` codec.
"""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.gpu import bc_codec
from arrayscope.gpu.chunk_codec import BitpackCodec, get_codec, gpu_decodable_codec_names


def _smooth_tile(shape=(256, 256)) -> np.ndarray:
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (np.sin(x / 40.0) * np.cos(y / 35.0) + x / float(shape[1])).astype(np.float32)


# --------------------------------------------------------------------------- BC4


def test_bc4_roundtrip_smooth_is_high_quality_and_8x_smaller():
    tile = _smooth_tile()
    unit, _norm = bc_codec.normalize_tile(tile)
    data, h, w = bc_codec.bc4_encode(unit)
    decoded = bc_codec.bc4_decode(data, h, w)
    q = bc_codec.quality_of(unit, decoded)
    assert q.psnr_db > 45.0
    assert len(data) == (256 // 4) * (256 // 4) * bc_codec.BC4_BLOCK_BYTES
    assert tile.nbytes / len(data) == 8.0  # r32float -> BC4


def test_bc4_constant_tile_is_exact():
    tile = np.full((64, 64), 3.5, dtype=np.float32)
    unit, _ = bc_codec.normalize_tile(tile)
    data, h, w = bc_codec.bc4_encode(unit)
    decoded = bc_codec.bc4_decode(data, h, w)
    assert bc_codec.quality_of(unit, decoded).max_abs_diff == 0.0


def test_bc4_odd_shape_pads_and_restores_shape():
    tile = _smooth_tile((100, 97))
    unit, _ = bc_codec.normalize_tile(tile)
    data, h, w = bc_codec.bc4_encode(unit)
    decoded = bc_codec.bc4_decode(data, h, w)
    assert decoded.shape == (100, 97)
    assert bc_codec.quality_of(unit, decoded).psnr_db > 40.0


@pytest.mark.parametrize("shape", [(4, 4), (100, 97), (256, 256)])
def test_numba_bc4_accelerator_is_byte_identical_after_explicit_prewarm(shape):
    pytest.importorskip("numba")
    unit, _ = bc_codec.normalize_tile(_smooth_tile(shape))
    expected_data, height, width = bc_codec._bc4_encode_numpy(unit)
    expected_decoded = bc_codec.bc4_decode(expected_data, height, width)

    assert bc_codec.prewarm_numba_encoder()
    assert bc_codec.numba_encoder_ready()
    data, actual_height, actual_width = bc_codec.bc4_encode(unit)
    quality_data, quality_height, quality_width, quality = bc_codec.bc4_encode_with_quality(unit)

    assert data == expected_data
    assert (actual_height, actual_width) == (height, width)
    assert quality_data == expected_data
    assert (quality_height, quality_width) == (height, width)
    expected_quality = bc_codec.quality_of(unit, expected_decoded)
    assert quality.psnr_db == pytest.approx(expected_quality.psnr_db)
    assert quality.max_abs_diff == pytest.approx(expected_quality.max_abs_diff)
    assert quality.rmse == pytest.approx(expected_quality.rmse)


def test_bc4_plan_engages_on_smooth_declines_on_noise():
    smooth = _smooth_tile()
    noisy = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
    p_smooth = bc_codec.bc4_plan(smooth)
    p_noisy = bc_codec.bc4_plan(noisy)
    assert p_smooth.engage is True
    assert p_smooth.vram_ratio == 8.0
    assert p_noisy.engage is False  # loss too high -> keep raw
    assert "decline" in p_noisy.reason


# --------------------------------------------------------------------------- BC5


def test_bc5_two_channel_roundtrip_and_size():
    a = _smooth_tile()
    b = _smooth_tile()[::-1]
    ua, _ = bc_codec.normalize_tile(a)
    ub, _ = bc_codec.normalize_tile(b)
    data, h, w = bc_codec.bc5_encode(ua, ub)
    da, db = bc_codec.bc5_decode(data, h, w)
    assert len(data) == (256 // 4) * (256 // 4) * bc_codec.BC5_BLOCK_BYTES
    assert bc_codec.quality_of(ua, da).psnr_db > 45.0
    assert bc_codec.quality_of(ub, db).psnr_db > 45.0


def test_complex_display_quality_reim_has_no_phase_wrap_artifact():
    # A complex tile with real signed structure; store (re, im), measure DISPLAY.
    re = _smooth_tile()
    im = _smooth_tile()[:, ::-1] * 0.5
    ure, nre = bc_codec.normalize_tile(re)
    uim, nim = bc_codec.normalize_tile(im)
    data, h, w = bc_codec.bc5_encode(ure, uim)
    d0, d1 = bc_codec.bc5_decode(data, h, w)
    re_dec = bc_codec.denormalize_channel(d0, nre)
    im_dec = bc_codec.denormalize_channel(d1, nim)
    q = bc_codec.complex_display_quality(re, im, re_dec, im_dec)
    assert q.magnitude_psnr_db > 40.0
    # magnitude-weighted phase error is tiny; the unweighted max may approach pi
    # only at near-zero magnitude (not displayed) -- re/im storage guarantees the
    # weighted error stays small (no wrap smear).
    assert q.phase_weighted_rmse_rad < 0.1


# --------------------------------------------------------------------------- ASTC


def test_astc_scalar_block_size_is_the_quality_knob():
    astc = pytest.importorskip("astc_encoder")  # noqa: F841
    from arrayscope.gpu import astc_codec

    assert astc_codec.astc_available()
    unit, _ = bc_codec.normalize_tile(_smooth_tile())
    r44 = astc_codec.encode_scalar(unit, block=(4, 4))
    r66 = astc_codec.encode_scalar(unit, block=(6, 6))
    # smaller block = more bytes, higher quality; the knob is real.
    assert r44.bc_bytes > r66.bc_bytes
    q44 = bc_codec.quality_of(unit, r44.decoded[0])
    q66 = bc_codec.quality_of(unit, r66.decoded[0])
    assert q44.psnr_db >= q66.psnr_db
    assert r44.wgpu_format == "astc-4x4-unorm"


# ------------------------------------------------------------------ bitpack (fallback)


@pytest.mark.parametrize(
    ("dtype", "kbits"),
    [(np.int16, 12), (np.uint8, 5), (np.int8, 3), (np.int32, 20)],
)
def test_bitpack_lossless_roundtrip_in_range(dtype, kbits):
    rng = np.random.default_rng(2)
    lo = -7 if np.dtype(dtype).kind == "i" else 0
    a = (rng.integers(0, 2**kbits, size=(64, 64)).astype(np.int64) + lo).astype(dtype)
    codec = BitpackCodec()
    blob = codec.encode(a)
    assert codec.is_packed(blob)
    back = codec.decode(blob, shape=a.shape, dtype=a.dtype)
    assert np.array_equal(back, a)  # bit-exact
    assert len(blob) < a.nbytes  # actually compressed


def test_bitpack_declines_full_width_and_float_but_stays_lossless():
    codec = BitpackCodec()
    full = np.arange(256, dtype=np.uint8).reshape(16, 16)  # needs all 8 bits
    blob = codec.encode(full)
    assert not codec.is_packed(blob)  # declined to raw mode
    assert np.array_equal(codec.decode(blob, shape=full.shape, dtype=full.dtype), full)

    flt = np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)
    fblob = codec.encode(flt)
    assert not codec.is_packed(fblob)
    assert np.array_equal(codec.decode(fblob, shape=flt.shape, dtype=flt.dtype), flt)


def test_bitpack_is_the_registered_gpu_decodable_codec():
    assert "bitpack" in gpu_decodable_codec_names()
    assert get_codec("bitpack").gpu_decodable is True
    assert get_codec("raw").gpu_decodable is False


# ---------------------------------------------------------- unified texture layer


def test_chunk_texture_codec_dispatches_bc_and_measures_quality():
    from arrayscope.gpu.cache_policy import decide_texture_codec
    from arrayscope.gpu.chunk_texture_codec import (
        encode_complex_tile,
        encode_scalar_tile,
    )
    from arrayscope.gpu.device_topology import DeviceTopology

    discrete = DeviceTopology(kind="discrete", unified_memory=False)
    decision = decide_texture_codec(topology=discrete, enable=True, astc_supported=False)

    scalar = _smooth_tile()
    enc = encode_scalar_tile(scalar, decision)
    assert enc.family == "bc"
    assert enc.wgpu_format == "bc4-r-unorm"
    assert enc.vram_ratio == 8.0
    assert enc.quality.psnr_db > 45.0

    re = _smooth_tile()
    im = _smooth_tile()[:, ::-1] * 0.5
    cx = encode_complex_tile(re, im, decision)
    assert cx.wgpu_format == "bc5-rg-unorm"
    assert cx.quality.magnitude_psnr_db > 40.0
    assert cx.quality.phase_weighted_rmse_rad < 0.1
