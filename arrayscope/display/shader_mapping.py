"""Shader-equivalent display mapping helpers.

The module is deliberately pure NumPy so tests and CPU fallbacks can compare
the same formulas used by GPU shader paths.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import numpy as np

from arrayscope.display import shader_kernels


class _ValueEnum(Enum):
    def __eq__(self, other):
        if isinstance(other, Enum):
            return self.value == getattr(other, "value", object())
        return self.value == other

    def __hash__(self):
        return hash(self.value)


class ShaderDisplayMode(_ValueEnum):
    SCALAR = "scalar"
    COMPLEX = "complex"
    PHASE_COLOR = "phase_color"
    RGB_WINDOWED = "rgb_windowed"
    RGB_DISPLAY_READY = "rgb_display_ready"


class ShaderComponent(_ValueEnum):
    REAL = "real"
    IMAG = "imag"
    ABS = "abs"
    ANGLE = "angle"
    COMPLEX_PHASE = "complex_phase"


class ShaderScale(_ValueEnum):
    LINEAR = "linear"
    LOG = "log"
    SYMLOG = "symlog"


class TexturePlaneKind(_ValueEnum):
    SCALAR_R32F = "scalar_r32f"
    COMPLEX_RG32F = "complex_rg32f"
    RGB8 = "rgb8"


@dataclass(frozen=True)
class ShaderMapping:
    component: ShaderComponent = ShaderComponent.REAL
    scale: ShaderScale = ShaderScale.LINEAR
    levels: tuple[float, float] | None = None
    display_mode: ShaderDisplayMode = ShaderDisplayMode.SCALAR
    lut_identity: object | None = None
    lut_data: np.ndarray | None = None
    histogram_source_policy: str = "mapped"
    symlog_constant: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", _coerce_enum(ShaderComponent, self.component))
        object.__setattr__(self, "scale", _coerce_enum(ShaderScale, self.scale))
        object.__setattr__(self, "display_mode", _coerce_enum(ShaderDisplayMode, self.display_mode))
        if self.levels is not None:
            low, high = self.levels
            object.__setattr__(self, "levels", (float(low), float(high)))
        if self.lut_data is not None:
            object.__setattr__(self, "lut_data", np.asarray(self.lut_data))
        object.__setattr__(self, "histogram_source_policy", str(self.histogram_source_policy))
        object.__setattr__(self, "symlog_constant", float(self.symlog_constant))

    @property
    def identity_key(self) -> tuple[Any, ...]:
        lut_key = self.lut_identity
        if lut_key is None and self.lut_data is not None:
            lut = np.asarray(self.lut_data)
            lut_key = (tuple(lut.shape), str(lut.dtype), lut.tobytes())
        return (
            self.display_mode.value,
            self.component.value,
            self.scale.value,
            None if self.levels is None else tuple(float(value) for value in self.levels),
            lut_key,
            self.histogram_source_policy,
            float(self.symlog_constant),
        )


def default_gray_lut(size: int = 256) -> np.ndarray:
    """Return a linear grayscale RGB lookup table."""

    values = np.linspace(0.0, 255.0, max(1, int(size)), dtype=np.float32)
    values = np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8)
    return np.repeat(values[:, np.newaxis], 3, axis=1)


def normalize_lut_rgb(lut: np.ndarray | None, *, phase_default: bool = False) -> np.ndarray:
    """Normalize a lookup table to contiguous ``uint8`` RGB rows."""

    default = default_phase_lut() if phase_default else default_gray_lut()
    lut_array = default if lut is None else np.asarray(lut)
    if lut_array.ndim != 2 or lut_array.shape[0] < 1 or lut_array.shape[1] < 3:
        raise ValueError("LUT must have shape (N, 3) or (N, 4)")
    color = lut_array[:, :3]
    if color.dtype != np.uint8:
        max_value = (
            1.0
            if np.issubdtype(color.dtype, np.floating)
            and color.size
            and float(np.nanmax(color)) <= 1.0
            else 255.0
        )
        color = np.clip(
            np.asarray(color, dtype=np.float32) * (255.0 / max_value), 0.0, 255.0
        ).astype(np.uint8)
    return np.ascontiguousarray(color)


def shader_mapping_with_lut(
    mapping: ShaderMapping | None,
    lut_data: np.ndarray,
    *,
    lut_identity: object | None = None,
) -> ShaderMapping:
    """Return ``mapping`` with one explicit frame-level display LUT."""

    base = ShaderMapping() if mapping is None else mapping
    if not isinstance(base, ShaderMapping):
        raise TypeError("mapping must be a ShaderMapping instance or None")
    lut = normalize_lut_rgb(lut_data)
    return replace(base, lut_identity=lut_identity, lut_data=lut)


def common_shader_mapping(mappings) -> ShaderMapping | None:
    """Return the one presentation mapping shared by a set of payloads.

    Shader state is frame-level presentation state.  It must never be inferred
    independently for each atlas page because page membership changes as tiles
    enter and leave residency.  Missing mappings are tolerated for scalar
    payloads without shader metadata; conflicting explicit mappings are rejected.
    """

    common = None
    common_key = None
    for mapping in mappings:
        if mapping is None:
            continue
        if not isinstance(mapping, ShaderMapping):
            raise TypeError("shader mappings must be ShaderMapping instances")
        if common is None:
            common = mapping
            common_key = mapping.identity_key
            continue
        if mapping is common:
            continue
        if mapping.identity_key != common_key:
            raise ValueError("a tiled presentation cannot contain conflicting shader mappings")
    return common


def extract_component(data, component: ShaderComponent | str) -> np.ndarray:
    component = _coerce_enum(ShaderComponent, component)
    arr = np.asarray(data)
    if component == ShaderComponent.REAL:
        return np.real(arr).astype(np.float32, copy=False)
    if component == ShaderComponent.IMAG:
        return np.imag(arr).astype(np.float32, copy=False)
    if component == ShaderComponent.ABS:
        return np.abs(arr).astype(np.float32, copy=False)
    if component in {ShaderComponent.ANGLE, ShaderComponent.COMPLEX_PHASE}:
        return np.angle(arr).astype(np.float32, copy=False)
    raise ValueError(f"unsupported shader component: {component!r}")


def shader_component_uniform(component: ShaderComponent | str | None) -> float:
    if component is None:
        return 0.0
    component = _coerce_enum(ShaderComponent, component)
    return {
        ShaderComponent.REAL: 0.0,
        ShaderComponent.IMAG: 1.0,
        ShaderComponent.ABS: 2.0,
        ShaderComponent.ANGLE: 3.0,
        ShaderComponent.COMPLEX_PHASE: 3.0,
    }[component]


def apply_scale(data, scale: ShaderScale | str, *, symlog_constant: float = 0.0) -> np.ndarray:
    scale = _coerce_enum(ShaderScale, scale)
    arr = np.asarray(data, dtype=np.float32)
    if scale == ShaderScale.LINEAR:
        return arr
    if scale == ShaderScale.LOG:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(np.maximum(arr, 0.0)).astype(np.float32, copy=False)
    if scale == ShaderScale.SYMLOG:
        c = float(symlog_constant)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (np.sign(arr) * np.log10(1.0 + np.abs(arr) / (10.0**c))).astype(
                np.float32, copy=False
            )
    raise ValueError(f"unsupported shader scale: {scale!r}")


def mapped_scalar(data, mapping: ShaderMapping) -> np.ndarray:
    component = extract_component(data, mapping.component)
    return apply_scale(component, mapping.scale, symlog_constant=mapping.symlog_constant)


def window_intensity(data, levels: tuple[float, float]) -> np.ndarray:
    low, high = levels
    span = max(float(high) - float(low), 1e-12)
    values = (np.asarray(data, dtype=np.float32) - float(low)) / span
    values = np.clip(values, 0.0, 1.0)
    return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def phase_lut_indices(data, lut_size: int) -> np.ndarray:
    if lut_size <= 0:
        raise ValueError("lut_size must be positive")
    phase = extract_component(data, ShaderComponent.ANGLE)
    position = (phase + np.pi) / (2.0 * np.pi)
    position = np.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
    upper_endpoint = np.isclose(position, 1.0, rtol=0.0, atol=np.finfo(np.float32).eps)
    position = np.where(upper_endpoint, 1.0, position)
    return np.clip((position * (int(lut_size) - 1)).astype(np.int64), 0, int(lut_size) - 1)


def apply_phase_lut(data, lut: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    lut_array = _lut_rgb_uint8(lut)
    phase = extract_component(data, ShaderComponent.ANGLE)
    position = (phase + np.pi) / (2.0 * np.pi)
    position = np.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
    color = _sample_lut_rgb(lut_array, np.clip(position, 0.0, 1.0))
    return color.astype(np.uint8, copy=False), np.abs(np.asarray(data)).astype(
        np.float32, copy=False
    )


_SCALE_CODES = {
    ShaderScale.LINEAR: shader_kernels._SCALE_LINEAR,
    ShaderScale.LOG: shader_kernels._SCALE_LOG,
    ShaderScale.SYMLOG: shader_kernels._SCALE_SYMLOG,
}


def _numba_cpu_display_rgba(data, mapping: ShaderMapping):
    """Fused numba fast path for the two dominant CPU display branches.

    Returns the RGBA result, or ``None`` (so the caller runs the NumPy
    reference) when numba is not yet warm or the mapping is one of the branches
    we deliberately leave on NumPy: a scalar mapping without explicit levels
    (needs ``finite_default_levels``), the phase-color magnitude branch, a
    size-1 LUT, or any non-SCALAR/PHASE display mode.
    """

    if not shader_kernels.ready():
        shader_kernels.ensure_prewarming()
        return None
    mode = mapping.display_mode
    if mode == ShaderDisplayMode.SCALAR:
        if mapping.levels is None:
            return None
        component = extract_component(data, mapping.component)
        lut = normalize_lut_rgb(mapping.lut_data, phase_default=False)
        low, high = mapping.levels
    elif mode == ShaderDisplayMode.PHASE_COLOR and mapping.component in {
        ShaderComponent.ANGLE,
        ShaderComponent.COMPLEX_PHASE,
    }:
        component = extract_component(data, ShaderComponent.ANGLE)
        lut = _lut_rgb_uint8(mapping.lut_data)
        low, high = mapping.levels if mapping.levels is not None else (-np.pi, np.pi)
    else:
        return None
    lut = np.asarray(lut, dtype=np.float32)
    if lut.ndim != 2 or lut.shape[0] < 2 or lut.shape[1] != 3:
        return None  # _sample_lut_rgb broadcasts a size-1 LUT; leave to NumPy
    span = max(float(high) - float(low), 1e-12)
    return shader_kernels.scalar_rgba(
        component, _SCALE_CODES[mapping.scale], mapping.symlog_constant, low, span, lut
    )


def cpu_display_rgba(data, mapping: ShaderMapping) -> np.ndarray:
    fast = _numba_cpu_display_rgba(data, mapping)
    if fast is not None:
        return fast
    if mapping.display_mode == ShaderDisplayMode.PHASE_COLOR:
        scalar = mapped_scalar(data, mapping)
        levels = mapping.levels
        if mapping.component in {ShaderComponent.ANGLE, ShaderComponent.COMPLEX_PHASE}:
            lut = _lut_rgb_uint8(mapping.lut_data)
            lut_position = window_intensity(scalar, levels or (-np.pi, np.pi))
            rgb = _sample_lut_rgb(lut, lut_position)
        else:
            color, _magnitude = apply_phase_lut(data, mapping.lut_data)
            intensity = (
                np.ones_like(scalar, dtype=np.float32)
                if levels is None
                else window_intensity(scalar, levels)
            )
            rgb = np.clip(color.astype(np.float32) * intensity[..., np.newaxis], 0.0, 255.0).astype(
                np.uint8
            )
        alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
        alpha[~np.isfinite(scalar), 0] = 0
        return np.concatenate((rgb, alpha), axis=-1)
    scalar = mapped_scalar(data, mapping)
    levels = mapping.levels or finite_default_levels(scalar)
    intensity = window_intensity(scalar, levels)
    alpha = np.full((*intensity.shape, 1), 255, dtype=np.uint8)
    alpha[~np.isfinite(scalar), 0] = 0
    lut = normalize_lut_rgb(mapping.lut_data, phase_default=False)
    rgb = _sample_lut_rgb(lut, intensity)
    return np.concatenate((rgb, alpha), axis=-1)


def pack_texture_data(data, texture_kind: TexturePlaneKind | str) -> np.ndarray:
    kind = _coerce_enum(TexturePlaneKind, texture_kind)
    arr = np.asarray(data)
    if kind == TexturePlaneKind.SCALAR_R32F:
        return np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    if kind == TexturePlaneKind.COMPLEX_RG32F:
        if np.iscomplexobj(arr):
            packed = np.empty((*arr.shape, 2), dtype=np.float32)
            packed[..., 0] = np.real(arr).astype(np.float32, copy=False)
            packed[..., 1] = np.imag(arr).astype(np.float32, copy=False)
            return np.ascontiguousarray(packed)
        packed = np.asarray(arr, dtype=np.float32)
        if packed.ndim < 3 or packed.shape[-1] != 2:
            raise ValueError("complex RG32F texture data must be complex or have trailing size 2")
        return np.ascontiguousarray(packed)
    if kind == TexturePlaneKind.RGB8:
        rgb = np.asarray(arr)
        if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
            raise ValueError("RGB8 texture data must have shape (H, W, 3|4)")
        if rgb.dtype != np.uint8:
            if np.issubdtype(rgb.dtype, np.floating) and rgb.size and float(np.nanmax(rgb)) <= 1.0:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb[..., :3])
    raise ValueError(f"unsupported texture kind: {kind!r}")


def finite_default_levels(data, fallback: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    arr = np.asarray(data, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (float(fallback[0]), float(fallback[1]))
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if high <= low:
        eps = max(abs(low) * 0.03, 0.5)
        return (low - eps, high + eps)
    return (low, high)


def default_phase_lut(size: int = 256) -> np.ndarray:
    values = np.linspace(0.0, 1.0, int(size), endpoint=False, dtype=np.float32)
    h = values * 6.0
    c = np.ones_like(h)
    x = 1.0 - np.abs(h % 2.0 - 1.0)
    rgb = np.zeros((int(size), 3), dtype=np.float32)
    masks = (
        (h >= 0) & (h < 1),
        (h >= 1) & (h < 2),
        (h >= 2) & (h < 3),
        (h >= 3) & (h < 4),
        (h >= 4) & (h < 5),
        (h >= 5) & (h < 6),
    )
    choices = ((c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x))
    for mask, choice in zip(masks, choices, strict=False):
        for channel, value in enumerate(choice):
            rgb[mask, channel] = value if np.isscalar(value) else value[mask]
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def _lut_rgb_uint8(lut: np.ndarray | None) -> np.ndarray:
    return normalize_lut_rgb(lut, phase_default=True)


def _sample_lut_rgb(lut: np.ndarray, position: np.ndarray) -> np.ndarray:
    lut = np.asarray(lut, dtype=np.float32)
    if lut.shape[0] == 1:
        return np.broadcast_to(lut[0].astype(np.uint8), (*np.asarray(position).shape, 3))
    scaled = np.clip(np.asarray(position, dtype=np.float32), 0.0, 1.0) * float(lut.shape[0] - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.clip(lower + 1, 0, lut.shape[0] - 1)
    weight = (scaled - lower.astype(np.float32))[..., np.newaxis]
    color = lut[lower] * (1.0 - weight) + lut[upper] * weight
    return np.clip(np.rint(color), 0.0, 255.0).astype(np.uint8)


def _coerce_enum(enum_type, value):
    if isinstance(value, enum_type):
        return value
    if hasattr(value, "value"):
        value = value.value
    return enum_type(value)


# --------------------------------------------------------------------------
# wgpu shader-legibility mirrors (Stage A trust signals + C1 minification)
#
# Pure NumPy mirrors of the fragment-shader visuals implemented in
# ``arrayscope.gpu.wgpu_executor._RENDER_WGSL.fs_main`` (and its BC-pool
# variant).  They are the single owner of the CPU-side formulas, so the wgpu
# executor's per-pixel oracle (``tests/gpu/test_wgpu_command_protocol.py``
# ``Scene.reference``) mirrors the WGSL without re-deriving the math.  These
# are deliberately NOT wired into :func:`cpu_display_rgba` — that is the
# maintained display paths, which Stage A does not touch. Colours here
# are the shader's normalized f32 space; callers reproduce the GPU's
# ``rgba8unorm`` store with ``round(x * 255)``.
# --------------------------------------------------------------------------

#: Screen px-per-texel below which the zoom-gated pixel grid is fully faded
#: out (0 contribution) and above which it is fully faded in.
PIXEL_GRID_MIN_PX_PER_TEXEL = 12.0
PIXEL_GRID_MAX_PX_PER_TEXEL = 24.0
#: Grid line half-width in screen pixels and its maximum darkening strength.
PIXEL_GRID_LINE_PX = 1.0
PIXEL_GRID_STRENGTH = 0.2
#: Diagonal-hatch period (screen px) shared by the NaN and missing markers.
TRUST_HATCH_PERIOD_PX = 8.0
#: NaN/Inf marker: opaque black/white diagonal (45°) — high contrast against
#: every colormap, so a non-finite texel can never masquerade as a LUT entry.
NAN_HATCH_SHADES = (0.0, 1.0)
#: Missing-page marker: dim gray hatch at the opposite angle (-45°), so a
#: not-yet-resident page never reads as an actual zero value.
MISSING_HATCH_SHADES = (0.12, 0.20)
#: Clip markers (only drawn when ``clip_indicator`` is enabled): cool below
#: the window, warm above it.
CLIP_UNDER_RGB = (0.0, 0.2, 0.8)
CLIP_OVER_RGB = (0.9, 0.1, 0.0)
#: Source texels per screen pixel above which the C1 minification filter
#: engages.  At or below it the draw is magnifying (or 1:1) and stays exactly
#: nearest — a filtered magnification would show values that are not in the
#: array, and would defeat the pixel grid.
MINIFY_FILTER_MIN_TEXELS_PER_PX = 1.0
#: Cap on taps per axis, so the filter costs a small constant.  2 on measured
#: evidence: a third ring of taps doubles the cost for two more percentage
#: points of aliasing reduction (see the WGSL ``minify_taps``).  Minification
#: heavier than 2x2 covers is the LOD ladder's problem, not this filter's.
MINIFY_FILTER_MAX_TAPS = 2
#: Shrink applied to the tile source rect's far edge before clamping taps, so
#: a tap on the boundary stays inside the last texel instead of rolling into
#: the next page.
MINIFY_FILTER_RECT_EPS = 1e-4


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((np.asarray(x, dtype=np.float64) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hatch_shades(pos_x, pos_y, *, angle_sign: float, shades: tuple[float, float]) -> np.ndarray:
    """Two-tone diagonal hatch value (normalized f32) keyed off screen pos."""

    diagonal = np.asarray(pos_x, dtype=np.float64) + angle_sign * np.asarray(
        pos_y, dtype=np.float64
    )
    stripe = np.mod(diagonal / TRUST_HATCH_PERIOD_PX, 1.0)
    return np.where(stripe < 0.5, shades[0], shades[1]).astype(np.float64)


def _shade_to_rgba8(shade: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.rint(np.asarray(shade)[..., np.newaxis] * 255.0), 0.0, 255.0)
    rgb = np.repeat(rgb, 3, axis=-1).astype(np.uint8)
    alpha = np.full((*rgb.shape[:-1], 1), 255, dtype=np.uint8)
    return np.concatenate((rgb, alpha), axis=-1)


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


__all__ = [
    "MINIFY_FILTER_MAX_TAPS",
    "MINIFY_FILTER_MIN_TEXELS_PER_PX",
    "ShaderComponent",
    "ShaderDisplayMode",
    "ShaderMapping",
    "ShaderScale",
    "TexturePlaneKind",
    "apply_phase_lut",
    "apply_scale",
    "common_shader_mapping",
    "cpu_display_rgba",
    "default_gray_lut",
    "default_phase_lut",
    "extract_component",
    "finite_default_levels",
    "mapped_scalar",
    "normalize_lut_rgb",
    "pack_texture_data",
    "phase_lut_indices",
    "shader_component_uniform",
    "shader_mapping_with_lut",
    "wgpu_clip_rgb",
    "wgpu_minification_tap_count",
    "wgpu_minification_taps",
    "wgpu_missing_hatch_rgba",
    "wgpu_nan_hatch_rgba",
    "wgpu_pixel_grid_darken",
    "window_intensity",
]
