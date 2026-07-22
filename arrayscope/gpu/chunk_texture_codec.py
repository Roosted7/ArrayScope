"""Unified native-texture codec layer for the display path (G7 Phase B).

Ties the pieces together behind one call: given a tile and a
:class:`~arrayscope.gpu.cache_policy.TextureCodecDecision` (topology-driven,
default OFF), encode it to the chosen native compressed-texture format, decode it
back with the reference decoder, and measure the loss in the domain the user
sees.  The decision picks the family:

* discrete (NVIDIA) -> BC4 (scalar) / BC5 (complex, holding real+imag)
* integrated (Intel) -> ASTC (scalar or two-channel) when available, else BC

Callers get a :class:`TextureEncoding` with the compressed bytes, the resident
VRAM byte count, the wgpu format string, and a measured-quality record.  Scalar
quality is PSNR/max-abs in the normalized [0, 1] field; complex quality is in the
display domain (magnitude PSNR + wrapped-phase error).  The raw path is never
touched unless a caller opts into ``decision.engage``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arrayscope.gpu import bc_codec
from arrayscope.gpu.cache_policy import TextureCodecDecision

__all__ = [
    "TextureEncoding",
    "encode_complex_tile",
    "encode_scalar_tile",
]


@dataclass(frozen=True)
class TextureEncoding:
    family: str
    wgpu_format: str
    data: bytes
    raw_bytes: int
    resident_bytes: int
    height: int
    width: int
    quality: object  # BcQuality (scalar) or ComplexDisplayQuality (complex)
    norms: tuple  # per-channel TileNorm(s), the shader rescale metadata

    @property
    def vram_ratio(self) -> float:
        return self.raw_bytes / self.resident_bytes if self.resident_bytes else float("inf")


def encode_scalar_tile(
    tile: np.ndarray,
    decision: TextureCodecDecision,
    *,
    raw_bytes_per_texel: int = 4,
) -> TextureEncoding:
    """Encode a scalar tile per the decision (BC4 or ASTC); measure quality."""

    unit, norm = bc_codec.normalize_tile(tile)
    h, w = int(tile.shape[-2]), int(tile.shape[-1])
    raw_bytes = h * w * int(raw_bytes_per_texel)

    if decision.family == "astc":
        from arrayscope.gpu import astc_codec

        block = decision.astc_block or (6, 6)
        res = astc_codec.encode_scalar(unit, block=block)
        quality = bc_codec.quality_of(unit, res.decoded[0][:h, :w])
        return TextureEncoding(
            family="astc",
            wgpu_format=res.wgpu_format,
            data=res.data,
            raw_bytes=raw_bytes,
            resident_bytes=res.bc_bytes,
            height=h,
            width=w,
            quality=quality,
            norms=(norm,),
        )

    data, bh, bw = bc_codec.bc4_encode(unit)
    decoded = bc_codec.bc4_decode(data, bh, bw)
    quality = bc_codec.quality_of(unit, decoded)
    return TextureEncoding(
        family="bc",
        wgpu_format=decision.scalar_format,
        data=data,
        raw_bytes=raw_bytes,
        resident_bytes=len(data),
        height=bh,
        width=bw,
        quality=quality,
        norms=(norm,),
    )


def encode_complex_tile(
    re: np.ndarray,
    im: np.ndarray,
    decision: TextureCodecDecision,
    *,
    raw_bytes_per_texel: int = 8,
) -> TextureEncoding:
    """Encode a complex tile as two channels (real, imag); measure DISPLAY loss."""

    unit_re, norm_re = bc_codec.normalize_tile(re)
    unit_im, norm_im = bc_codec.normalize_tile(im)
    h, w = int(re.shape[-2]), int(re.shape[-1])
    raw_bytes = h * w * int(raw_bytes_per_texel)

    if decision.family == "astc":
        from arrayscope.gpu import astc_codec

        block = decision.astc_block or (4, 4)
        res = astc_codec.encode_two_channel(unit_re, unit_im, block=block)
        re_dec = bc_codec.denormalize_channel(res.decoded[0][:h, :w], norm_re)
        im_dec = bc_codec.denormalize_channel(res.decoded[1][:h, :w], norm_im)
        quality = bc_codec.complex_display_quality(re, im, re_dec, im_dec)
        return TextureEncoding(
            family="astc",
            wgpu_format=res.wgpu_format,
            data=res.data,
            raw_bytes=raw_bytes,
            resident_bytes=res.bc_bytes,
            height=h,
            width=w,
            quality=quality,
            norms=(norm_re, norm_im),
        )

    data, bh, bw = bc_codec.bc5_encode(unit_re, unit_im)
    d0, d1 = bc_codec.bc5_decode(data, bh, bw)
    re_dec = bc_codec.denormalize_channel(d0, norm_re)
    im_dec = bc_codec.denormalize_channel(d1, norm_im)
    quality = bc_codec.complex_display_quality(re, im, re_dec, im_dec)
    return TextureEncoding(
        family="bc",
        wgpu_format=decision.complex_format,
        data=data,
        raw_bytes=raw_bytes,
        resident_bytes=len(data),
        height=bh,
        width=bw,
        quality=quality,
        norms=(norm_re, norm_im),
    )
