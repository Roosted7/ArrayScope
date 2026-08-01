"""CPU mirrors of the WGSL the display shader runs.

The reference side of the shader-conformance tests: each function reproduces
one marker or filter the fragment shader applies, so a GPU render can be
compared against a value computed here rather than against a second copy of the
same shader. The WGSL is the caller-side twin -- nothing in the render path
imports these, which is exactly why they live in the harness package instead of
beside the mapping code they mirror.

The names are ``wgpu_``-prefixed because that is what they mirror; the module
is ``shader_reference`` because that is what it is.
"""

from __future__ import annotations

import numpy as np

from arrayscope.display.shader_mapping import (
    CLIP_OVER_RGB,
    CLIP_UNDER_RGB,
    MINIFY_FILTER_MAX_TAPS,
    MINIFY_FILTER_MIN_TEXELS_PER_PX,
    MINIFY_FILTER_RECT_EPS,
    MISSING_HATCH_SHADES,
    NAN_HATCH_SHADES,
    PIXEL_GRID_LINE_PX,
    PIXEL_GRID_MAX_PX_PER_TEXEL,
    PIXEL_GRID_MIN_PX_PER_TEXEL,
    PIXEL_GRID_STRENGTH,
    _hatch_shades,
    _shade_to_rgba8,
    _smoothstep,
)


def wgpu_nan_hatch_rgba(pos_x, pos_y) -> np.ndarray:
    """RGBA8 mirror of the WGSL ``nan_marker`` (A2)."""

    shade = _hatch_shades(pos_x, pos_y, angle_sign=1.0, shades=NAN_HATCH_SHADES)
    return _shade_to_rgba8(shade)


def wgpu_missing_hatch_rgba(pos_x, pos_y) -> np.ndarray:
    """RGBA8 mirror of the WGSL ``missing_marker`` (A3)."""

    shade = _hatch_shades(pos_x, pos_y, angle_sign=-1.0, shades=MISSING_HATCH_SHADES)
    return _shade_to_rgba8(shade)


def wgpu_clip_rgb(kind: str) -> tuple[int, int, int]:
    """RGB8 of a clip marker (A4): ``"under"`` or ``"over"``."""

    rgb = CLIP_UNDER_RGB if kind == "under" else CLIP_OVER_RGB
    return tuple(round(c * 255.0) for c in rgb)


def wgpu_pixel_grid_darken(rgb8, src_x, src_y, fw_x, fw_y, *, enabled: bool = True) -> np.ndarray:
    """Mirror of the WGSL ``pixel_grid`` darkening (A1) on an RGBA8 image.

    ``fw_x``/``fw_y`` are the source-texels-per-screen-pixel derivatives
    (``fwidth(in.src)`` on the GPU); for an axis-aligned affine tile these are
    the constant ratios ``src_size / dst_pixels`` per axis, which the GPU
    computes exactly.  Faded to zero below ``PIXEL_GRID_MIN_PX_PER_TEXEL``, so
    a normally-zoomed scene is returned byte-identical.
    """

    rgb8 = np.asarray(rgb8)
    if not enabled:
        return rgb8
    fw_x = np.maximum(np.asarray(fw_x, dtype=np.float64), 1e-8)
    fw_y = np.maximum(np.asarray(fw_y, dtype=np.float64), 1e-8)
    px_per_texel = 1.0 / np.maximum(fw_x, fw_y)
    fade = _smoothstep(PIXEL_GRID_MIN_PX_PER_TEXEL, PIXEL_GRID_MAX_PX_PER_TEXEL, px_per_texel)
    frac_x = np.mod(np.asarray(src_x, dtype=np.float64), 1.0)
    frac_y = np.mod(np.asarray(src_y, dtype=np.float64), 1.0)
    edge_x = np.minimum(frac_x, 1.0 - frac_x) / fw_x
    edge_y = np.minimum(frac_y, 1.0 - frac_y) / fw_y
    line = 1.0 - _smoothstep(0.0, PIXEL_GRID_LINE_PX, np.minimum(edge_x, edge_y))
    darken = (PIXEL_GRID_STRENGTH * line * fade)[..., np.newaxis]
    out = rgb8.astype(np.float64).copy()
    out[..., :3] = np.rint(out[..., :3] * (1.0 - darken))
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def wgpu_minification_tap_count(fw_x, fw_y, *, enabled: bool = True) -> int:
    """Taps per axis for the C1 minification box filter.

    ``fw_x``/``fw_y`` are the source-texels-per-screen-pixel derivatives, as
    for :func:`wgpu_pixel_grid_darken`.  One tap means "unchanged": the shader
    does the single ``textureLoad`` it always did.  Magnification always gets
    one tap — inventing values between samples is the one thing an inspection
    tool must not do — so only a minified draw filters, and only up to
    ``MINIFY_FILTER_MAX_TAPS`` per axis.

    The GPU evaluates this per fragment; the mirror takes the per-tile
    constant derivative, which is exact for an axis-aligned affine tile.
    """

    if not enabled:
        return 1
    factor = max(float(fw_x), float(fw_y))
    # Negated so a non-finite derivative falls to the nearest-sample path.
    if not factor > MINIFY_FILTER_MIN_TEXELS_PER_PX:
        return 1
    return min(int(np.ceil(factor)), MINIFY_FILTER_MAX_TAPS)


def wgpu_minification_taps(src_x, src_y, fw_x, fw_y, taps: int, *, src_rect=None) -> list:
    """Source coordinates of the C1 box-filter taps, uniformly weighted.

    Box quadrature over the fragment's footprint ``[src - fw/2, src + fw/2]``,
    clamped to ``src_rect`` (``(x0, y0, w, h)``, the tile's own source window)
    so no tap escapes the tile into a neighbouring montage cell or the zero
    padding past a page's ``stored_rect``.  Returns ``taps * taps`` coordinate
    pairs; the caller gathers and averages them, and is responsible for the
    residency and non-finite rules the shader applies (see ``sample_footprint``
    in ``wgpu_executor``).
    """

    src_x = np.asarray(src_x, dtype=np.float64)
    src_y = np.asarray(src_y, dtype=np.float64)
    taps = int(taps)
    if taps <= 1:
        return [(src_x, src_y)]
    fw_x = float(fw_x)
    fw_y = float(fw_y)
    if src_rect is None:
        lo_x = lo_y = -np.inf
        hi_x = hi_y = np.inf
    else:
        x0, y0, width, height = (float(v) for v in src_rect)
        lo_x, lo_y = x0, y0
        hi_x = x0 + width - MINIFY_FILTER_RECT_EPS
        hi_y = y0 + height - MINIFY_FILTER_RECT_EPS
    out = []
    for j in range(taps):
        for i in range(taps):
            unit_x = (i + 0.5) / taps - 0.5
            unit_y = (j + 0.5) / taps - 0.5
            out.append(
                (
                    np.clip(src_x + unit_x * fw_x, lo_x, hi_x),
                    np.clip(src_y + unit_y * fw_y, lo_y, hi_y),
                )
            )
    return out
