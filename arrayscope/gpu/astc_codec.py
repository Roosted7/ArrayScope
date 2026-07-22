"""ASTC texture compression for the integrated (Intel) display path (G7 Phase B).

BC is the only compressed-texture family the discrete NVIDIA adapter exposes, but
the integrated Intel UHD additionally advertises ``texture-compression-astc`` --
and ASTC is the better fit there: flexible block sizes (4x4 = 8 bpp down to
12x12 ~= 0.89 bpp) give a real size/quality knob, and its multi-channel handling
suits the two-channel complex case.  Intel samples ASTC in hardware, so like BC
the decode is free at sample time.

This module wraps ``astc_encoder`` (ARM astcenc bindings).  It is an *optional*
dependency: :func:`astc_available` reports ``False`` when it is not installed and
every entry point degrades cleanly, so importing arrayscope never requires it.

Encoding model (matches the BC path): a tile is normalized to [0, 1] per channel
first (the render shader rescales at sample time).  Scalar tiles are stored as a
replicated-luminance LDR image (value in R=G=B); two-channel complex tiles store
(real, imag) in (R, G).  Quality is measured in the same normalized field as the
BC path for an apples-to-apples cross-format comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ASTC_BLOCK_BYTES",
    "AstcResult",
    "astc_available",
    "astc_block_bytes",
    "astc_decode",
    "encode_scalar",
    "encode_two_channel",
    "wgpu_format_for_block",
]

ASTC_BLOCK_BYTES = 16  # every ASTC block is 128 bits regardless of block size


def astc_available() -> bool:
    try:
        import astc_encoder  # noqa: F401
    except Exception:
        return False
    return True


def astc_block_bytes(block: tuple[int, int], width: int, height: int) -> int:
    bx, by = block
    nblocks = ((width + bx - 1) // bx) * ((height + by - 1) // by)
    return nblocks * ASTC_BLOCK_BYTES


def wgpu_format_for_block(block: tuple[int, int]) -> str:
    """The wgpu ``TextureFormat`` name for an ASTC block size (unorm LDR)."""

    bx, by = block
    return f"astc-{bx}x{by}-unorm"


@dataclass(frozen=True)
class AstcResult:
    """An ASTC encode: the compressed bytes, the decoded [0, 1] field(s), sizes."""

    block: tuple[int, int]
    data: bytes
    decoded: tuple[np.ndarray, ...]  # one [0,1] field per stored channel
    width: int
    height: int
    bc_bytes: int  # resident/compressed byte count
    wgpu_format: str


def _quantize_rgba(channels: list[np.ndarray]) -> tuple[bytes, int, int]:
    """Pack up to 4 [0, 1] channels to an RGBA u8 buffer (unused channels 0)."""

    h, w = channels[0].shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    for i, ch in enumerate(channels[:4]):
        rgba[..., i] = np.rint(np.clip(ch, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgba.tobytes(), h, w


def _compress(rgba_bytes: bytes, w: int, h: int, block: tuple[int, int]):
    import astc_encoder as ae

    bx, by = block
    cfg = ae.ASTCConfig(ae.ASTCProfile.LDR, bx, by, quality=ae.ASTCQualityPreset.MEDIUM)
    ctx = ae.ASTCContext(cfg)
    sw = ae.ASTCSwizzle.from_str("RGBA")
    img = ae.ASTCImage(ae.ASTCType.U8, w, h, 1, data=rgba_bytes)
    comp = bytes(ctx.compress(img, sw))
    arr = _decompress(comp, w, h, block)
    return comp, arr


def _decompress(comp: bytes, w: int, h: int, block: tuple[int, int]) -> np.ndarray:
    """Decode ASTC block bytes to an (h, w, 4) float32 [0, 1] RGBA image."""

    import astc_encoder as ae

    bx, by = block
    cfg = ae.ASTCConfig(ae.ASTCProfile.LDR, bx, by, quality=ae.ASTCQualityPreset.MEDIUM)
    ctx = ae.ASTCContext(cfg)
    sw = ae.ASTCSwizzle.from_str("RGBA")
    out = ae.ASTCImage(ae.ASTCType.U8, w, h, 1)
    dec = ctx.decompress(comp, out, sw)
    return np.frombuffer(dec.data, dtype=np.uint8).reshape(h, w, 4).astype(np.float32) / 255.0


def astc_decode(
    comp: bytes,
    block: tuple[int, int],
    width: int,
    height: int,
    n_channels: int = 1,
) -> tuple[np.ndarray, ...]:
    """Decode ASTC block bytes to ``n_channels`` [0, 1] float32 fields (R, then G).

    The hardware sampler on an ASTC-capable device produces the same decoded
    texels (verified to match this CPU reference within ~60 dB), so this is the
    reference oracle for a page read back out of an ASTC GPU pool.
    """

    arr = _decompress(comp, width, height, block)
    return tuple(arr[..., i] for i in range(int(n_channels)))


def encode_scalar(unit_field: np.ndarray, block: tuple[int, int] = (4, 4)) -> AstcResult:
    """Encode a [0, 1] scalar field as a replicated-luminance ASTC image."""

    field = np.ascontiguousarray(np.clip(unit_field, 0.0, 1.0), dtype=np.float32)
    rgba, h, w = _quantize_rgba([field, field, field])
    comp, arr = _compress(rgba, w, h, block)
    return AstcResult(
        block=block,
        data=comp,
        decoded=(arr[..., 0],),
        width=w,
        height=h,
        bc_bytes=len(comp),
        wgpu_format=wgpu_format_for_block(block),
    )


def encode_two_channel(
    unit0: np.ndarray, unit1: np.ndarray, block: tuple[int, int] = (4, 4)
) -> AstcResult:
    """Encode two [0, 1] channels (e.g. normalized real, imag) into ASTC R, G."""

    c0 = np.ascontiguousarray(np.clip(unit0, 0.0, 1.0), dtype=np.float32)
    c1 = np.ascontiguousarray(np.clip(unit1, 0.0, 1.0), dtype=np.float32)
    rgba, h, w = _quantize_rgba([c0, c1])
    comp, arr = _compress(rgba, w, h, block)
    return AstcResult(
        block=block,
        data=comp,
        decoded=(arr[..., 0], arr[..., 1]),
        width=w,
        height=h,
        bc_bytes=len(comp),
        wgpu_format=wgpu_format_for_block(block),
    )
