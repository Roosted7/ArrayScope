"""Native block-compressed textures for the display path (G7 Phase B headline).

Phase A proved compressing host chunks saves RAM; the CPU-decode transport race
proved it cannot win the *transfer* on a fast PCIe link, because decode sits on
the CPU critical path.  The fix that actually wins is to hand the GPU a format it
decompresses *for free in the texture sampler* -- a native block-compressed
texture (BC).  Then there is **no decode pass at all**: the compressed bytes
cross PCIe, stay compressed resident in VRAM (an 8x saving vs r32float for BC4),
and the hardware sampler returns usable values at sample time.  The cost is that
BC is *lossy*; that is acceptable on the display path (window/level is applied at
sample time anyway) and the loss is *measured*, never assumed.

Formats implemented here (CPU reference encode/decode, matching the hardware):

* **BC4** -- one channel, 8 bytes per 4x4 block (two 8-bit endpoints + 16 3-bit
  indices).  For scalar tiles.  0.5 bytes/texel (8x vs r32float, 4x vs r16).
* **BC5** -- two BC4 channels, 16 bytes per block.  For two-channel tiles
  (complex real/imag, or magnitude/phase).

The tile holds RAW scientific values and the render shader applies window/level
at sample time, so a tile is first *normalized* to [0, 1] with per-tile
(min, max) -- the sampler returns the normalized value and the caller rescales by
(min, max).  Encoders operate on that normalized field.  A tile whose BC PSNR
falls below a threshold is *declined* (kept raw) -- :func:`bc4_plan` measures the
error and reports it so "the loss is small" is a number, not a hope.

BC6H (half-float, less loss for float tiles) is noted as follow-up: its encode is
materially harder (mode/partition search) than BC4/BC5's min/max endpoints.

Import health: importing this module never touches wgpu -- it is pure numpy.  The
GPU-side pieces (real BC textures, the WGSL BC4 compute encoder) live in
:mod:`arrayscope.gpu.bc_gpu`, imported lazily there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "BC4_BLOCK_BYTES",
    "BC5_BLOCK_BYTES",
    "BcQuality",
    "ComplexDisplayQuality",
    "bc4_decode",
    "bc4_encode",
    "bc4_plan",
    "bc5_decode",
    "bc5_encode",
    "complex_display_quality",
    "denormalize_channel",
    "normalize_tile",
    "psnr",
    "quality_of",
]

BC4_BLOCK_BYTES = 8
BC5_BLOCK_BYTES = 16


# ---------------------------------------------------------------------------
# normalization: RAW scientific tile <-> [0, 1] field the sampler returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileNorm:
    """The per-tile affine that maps raw values to/from the [0, 1] sampler field."""

    lo: float
    hi: float

    @property
    def span(self) -> float:
        return (self.hi - self.lo) or 1.0

    def denormalize(self, unit: np.ndarray) -> np.ndarray:
        return unit.astype(np.float32) * np.float32(self.span) + np.float32(self.lo)


def normalize_tile(tile: np.ndarray) -> tuple[np.ndarray, TileNorm]:
    """Map a real-valued tile to a float32 field in [0, 1] with its per-tile norm."""

    arr = np.ascontiguousarray(tile, dtype=np.float32)
    lo = float(np.nanmin(arr)) if arr.size else 0.0
    hi = float(np.nanmax(arr)) if arr.size else 1.0
    norm = TileNorm(lo=lo, hi=hi)
    unit = (arr - np.float32(lo)) / np.float32(norm.span)
    return np.clip(unit, 0.0, 1.0), norm


# ---------------------------------------------------------------------------
# BC4: single-channel, 8 bytes / 4x4 block
# ---------------------------------------------------------------------------


def _to_blocks(field_u8: np.ndarray) -> np.ndarray:
    """Reshape an (H, W) uint8 image (H, W multiples of 4) to (nby, nbx, 4, 4)."""

    h, w = field_u8.shape
    return field_u8.reshape(h // 4, 4, w // 4, 4).transpose(0, 2, 1, 3)


def _pad_to_multiple_of_4(field_u8: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = field_u8.shape
    ph = (-h) % 4
    pw = (-w) % 4
    if ph or pw:
        field_u8 = np.pad(field_u8, ((0, ph), (0, pw)), mode="edge")
    return field_u8, h, w


def _bc4_palette(red0: np.ndarray, red1: np.ndarray) -> np.ndarray:
    """8 interpolated levels per block for the red0 > red1 (6-interp) mode.

    Shapes: ``red0``/``red1`` are (nblocks,) uint16; returns (nblocks, 8) float.
    """

    r0 = red0.astype(np.float64)
    r1 = red1.astype(np.float64)
    levels = [r0, r1, *(((7 - j) * r0 + j * r1) / 7.0 for j in range(1, 7))]
    return np.stack(levels, axis=1)  # (nblocks, 8)


def bc4_encode(field_unit: np.ndarray) -> tuple[bytes, int, int]:
    """Encode a [0, 1] field to BC4 blocks.  Returns (bytes, height, width).

    Uses the high-precision (red0 > red1) mode with per-block min/max endpoints.
    Each 4x4 block is 8 bytes: ``red0``, ``red1``, then 16 3-bit indices packed
    LSB-first (texel ``t`` at bits ``[3t, 3t+3)`` of a 48-bit little-endian word).
    """

    field = np.ascontiguousarray(field_unit, dtype=np.float32)
    field = np.clip(field, 0.0, 1.0)
    u8 = np.rint(field * 255.0).astype(np.uint8)
    u8, h, w = _pad_to_multiple_of_4(u8)
    blocks = _to_blocks(u8).reshape(-1, 16)  # (nblocks, 16) row-major within block
    lo = blocks.min(axis=1).astype(np.uint16)
    hi = blocks.max(axis=1).astype(np.uint16)
    red0 = hi  # red0 > red1 selects the 8-value interpolated mode
    red1 = lo
    palette = _bc4_palette(red0, red1)  # (nblocks, 8)
    # nearest palette index per texel
    diff = np.abs(blocks[:, :, None].astype(np.float64) - palette[:, None, :])  # (nb,16,8)
    idx = diff.argmin(axis=2).astype(np.uint64)  # (nblocks, 16)
    shifts = (3 * np.arange(16, dtype=np.uint64))[None, :]
    index48 = np.bitwise_or.reduce(idx << shifts, axis=1)  # (nblocks,) uint64
    block_u64 = red0.astype(np.uint64) | (red1.astype(np.uint64) << np.uint64(8)) | (
        index48 << np.uint64(16)
    )
    data = block_u64.astype("<u8").tobytes()
    return data, h, w


def bc4_decode(data: bytes, height: int, width: int) -> np.ndarray:
    """Decode BC4 blocks back to a [0, 1] float32 field of shape (height, width)."""

    ph = (-height) % 4
    pw = (-width) % 4
    H, W = height + ph, width + pw
    block_u64 = np.frombuffer(data, dtype="<u8")
    red0 = (block_u64 & 0xFF).astype(np.uint16)
    red1 = ((block_u64 >> np.uint64(8)) & np.uint64(0xFF)).astype(np.uint16)
    index48 = block_u64 >> np.uint64(16)
    shifts = (3 * np.arange(16, dtype=np.uint64))[None, :]
    idx = ((index48[:, None] >> shifts) & np.uint64(0x7)).astype(np.intp)  # (nblocks,16)
    palette = _bc4_palette(red0, red1)  # (nblocks, 8)
    vals = np.take_along_axis(palette, idx, axis=1)  # (nblocks, 16)
    nby, nbx = H // 4, W // 4
    img = vals.reshape(nby, nbx, 4, 4).transpose(0, 2, 1, 3).reshape(H, W)
    field = (img / 255.0).astype(np.float32)
    return field[:height, :width]


# ---------------------------------------------------------------------------
# BC5: two BC4 channels (16 bytes / block), for two-channel tiles
# ---------------------------------------------------------------------------


def bc5_encode(field0_unit: np.ndarray, field1_unit: np.ndarray) -> tuple[bytes, int, int]:
    """Encode two [0, 1] channels to BC5 (two interleaved BC4 blocks per 4x4)."""

    d0, h, w = bc4_encode(field0_unit)
    d1, _h, _w = bc4_encode(field1_unit)
    a = np.frombuffer(d0, dtype="<u8")
    b = np.frombuffer(d1, dtype="<u8")
    inter = np.empty(a.size * 2, dtype="<u8")
    inter[0::2] = a
    inter[1::2] = b
    return inter.tobytes(), h, w


def bc5_decode(data: bytes, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode BC5 blocks back to two [0, 1] float32 fields (height, width)."""

    inter = np.frombuffer(data, dtype="<u8")
    d0 = inter[0::2].tobytes()
    d1 = inter[1::2].tobytes()
    return bc4_decode(d0, height, width), bc4_decode(d1, height, width)


# ---------------------------------------------------------------------------
# quality: measured, never assumed
# ---------------------------------------------------------------------------


def psnr(reference: np.ndarray, candidate: np.ndarray, *, peak: float = 1.0) -> float:
    """Peak SNR in dB between two arrays (``peak`` is the value range)."""

    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    mse = float(np.mean((ref - cand) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 20.0 * np.log10(peak) - 10.0 * np.log10(mse)


@dataclass(frozen=True)
class BcQuality:
    """Measured BC round-trip error in the [0, 1] normalized field."""

    psnr_db: float
    max_abs_diff: float
    rmse: float

    def acceptable(self, *, min_psnr_db: float, max_abs: float) -> bool:
        return bool(self.psnr_db >= min_psnr_db and self.max_abs_diff <= max_abs)


def quality_of(reference_unit: np.ndarray, decoded_unit: np.ndarray) -> BcQuality:
    """PSNR / max-abs / RMSE of a BC round-trip in the normalized [0, 1] field."""

    ref = np.asarray(reference_unit, dtype=np.float64)
    dec = np.asarray(decoded_unit, dtype=np.float64)
    err = np.abs(ref - dec)
    rmse = float(np.sqrt(np.mean((ref - dec) ** 2))) if ref.size else 0.0
    return BcQuality(
        psnr_db=psnr(ref, dec, peak=1.0),
        max_abs_diff=float(err.max()) if err.size else 0.0,
        rmse=rmse,
    )


def denormalize_channel(unit: np.ndarray, norm: TileNorm) -> np.ndarray:
    """Map a [0, 1] channel back to its raw (signed) values via ``norm``."""

    return norm.denormalize(unit)


@dataclass(frozen=True)
class ComplexDisplayQuality:
    """BC/ASTC round-trip error for a complex tile, in the DISPLAY domain.

    The user never sees raw (real, imag); they see magnitude and phase derived
    from them in the shader.  So error is measured there: magnitude PSNR (peak =
    the raw magnitude range) and the worst wrapped phase error in radians.  Storing
    (real, imag) -- not (mag, phase) -- keeps this error small and isotropic and
    avoids the phase-wrap discontinuity that block compression would smear.
    """

    magnitude_psnr_db: float
    magnitude_max_abs: float
    phase_max_abs_rad: float
    phase_rmse_rad: float
    # magnitude-weighted phase error: the honest DISPLAY metric.  Phase is
    # meaningless (random) where magnitude ~= 0 -- a tiny (re,im) error can flip it
    # by up to pi there, but the user sees near-black, so that pixel does not
    # matter.  These weight each phase error by magnitude, and the "significant"
    # figures restrict to pixels above a magnitude floor (what is actually shown).
    phase_weighted_rmse_rad: float
    phase_max_abs_significant_rad: float


def complex_display_quality(
    re_raw: np.ndarray,
    im_raw: np.ndarray,
    re_dec: np.ndarray,
    im_dec: np.ndarray,
    *,
    significant_frac: float = 0.1,
) -> ComplexDisplayQuality:
    """Magnitude/phase error of a decoded complex tile vs the raw (re, im) tile.

    ``significant_frac`` is the fraction of peak magnitude below which phase is
    treated as not-displayed (near-black) and excluded from the "significant"
    phase figures -- the honest measure of what the phase-color view shows.
    """

    re_raw = np.asarray(re_raw, dtype=np.float64)
    im_raw = np.asarray(im_raw, dtype=np.float64)
    re_dec = np.asarray(re_dec, dtype=np.float64)
    im_dec = np.asarray(im_dec, dtype=np.float64)
    mag_raw = np.hypot(re_raw, im_raw)
    mag_dec = np.hypot(re_dec, im_dec)
    peak = float(mag_raw.max()) or 1.0
    mag_err = np.abs(mag_raw - mag_dec)
    # wrapped phase difference in (-pi, pi]
    dphase = np.angle(np.exp(1j * (np.arctan2(im_dec, re_dec) - np.arctan2(im_raw, re_raw))))
    weights = mag_raw / peak
    wsum = float(weights.sum()) or 1.0
    weighted_rmse = float(np.sqrt(np.sum(weights * dphase**2) / wsum)) if dphase.size else 0.0
    sig = mag_raw >= significant_frac * peak
    sig_max = float(np.abs(dphase[sig]).max()) if sig.any() else 0.0
    return ComplexDisplayQuality(
        magnitude_psnr_db=psnr(mag_raw, mag_dec, peak=peak),
        magnitude_max_abs=float(mag_err.max()) if mag_err.size else 0.0,
        phase_max_abs_rad=float(np.abs(dphase).max()) if dphase.size else 0.0,
        phase_rmse_rad=float(np.sqrt(np.mean(dphase**2))) if dphase.size else 0.0,
        phase_weighted_rmse_rad=weighted_rmse,
        phase_max_abs_significant_rad=sig_max,
    )


@dataclass(frozen=True)
class Bc4Plan:
    """The BC4 decision for one scalar tile: engage (with its measured quality) or
    decline to raw when the loss exceeds the threshold."""

    engage: bool
    quality: BcQuality
    norm: TileNorm
    height: int
    width: int
    raw_bytes: int
    bc_bytes: int
    reason: str

    @property
    def vram_ratio(self) -> float:
        return self.raw_bytes / self.bc_bytes if self.bc_bytes else float("inf")


def bc4_plan(
    tile: np.ndarray,
    *,
    min_psnr_db: float = 40.0,
    max_abs: float = 0.05,
    raw_bytes_per_texel: int = 4,
) -> Bc4Plan:
    """Encode ``tile`` as BC4, measure the loss, and decide engage vs decline.

    ``min_psnr_db`` / ``max_abs`` gate engagement: a tile whose BC4 round-trip is
    worse than either threshold is declined (kept raw) so quality is never
    silently sacrificed.  ``raw_bytes_per_texel`` is the raw resident cost the BC
    texture is compared against (4 for r32float, 2 for r16)."""

    unit, norm = normalize_tile(tile)
    data, h, w = bc4_encode(unit)
    decoded = bc4_decode(data, h, w)
    q = quality_of(unit, decoded)
    n = int(tile.shape[-2] * tile.shape[-1]) if tile.ndim >= 2 else int(tile.size)
    raw_bytes = n * int(raw_bytes_per_texel)
    engage = q.acceptable(min_psnr_db=min_psnr_db, max_abs=max_abs)
    reason = (
        f"BC4 PSNR {q.psnr_db:.1f}dB, max|err| {q.max_abs_diff:.4f}: "
        + ("engage (within thresholds)" if engage else "decline -> raw (loss too high)")
    )
    return Bc4Plan(
        engage=engage,
        quality=q,
        norm=norm,
        height=h,
        width=w,
        raw_bytes=raw_bytes,
        bc_bytes=len(data),
        reason=reason,
    )
