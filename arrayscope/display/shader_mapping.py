"""Shader-equivalent display mapping helpers.

The module is deliberately pure NumPy so tests and CPU fallbacks can compare
the same formulas used by VisPy shader paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


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


def common_shader_mapping(mappings) -> ShaderMapping | None:
    """Return the one presentation mapping shared by a set of payloads.

    Shader state is frame-level presentation state.  It must never be inferred
    independently for each atlas page because page membership changes as tiles
    enter and leave residency.  Missing mappings are tolerated for legacy
    scalar payloads; conflicting explicit mappings are rejected.
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
            return (np.sign(arr) * np.log10(1.0 + np.abs(arr) / (10.0**c))).astype(np.float32, copy=False)
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
    return np.clip((position * (int(lut_size) - 1)).astype(np.int64), 0, int(lut_size) - 1)


def apply_phase_lut(data, lut: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    lut_array = _lut_rgb_uint8(lut)
    phase = extract_component(data, ShaderComponent.ANGLE)
    position = (phase + np.pi) / (2.0 * np.pi)
    position = np.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
    color = _sample_lut_rgb(lut_array, np.clip(position, 0.0, 1.0))
    return color.astype(np.uint8, copy=False), np.abs(np.asarray(data)).astype(np.float32, copy=False)


def cpu_display_rgba(data, mapping: ShaderMapping) -> np.ndarray:
    if mapping.display_mode == ShaderDisplayMode.PHASE_COLOR:
        scalar = mapped_scalar(data, mapping)
        levels = mapping.levels
        if mapping.component in {ShaderComponent.ANGLE, ShaderComponent.COMPLEX_PHASE}:
            lut = _lut_rgb_uint8(mapping.lut_data)
            lut_position = window_intensity(scalar, levels or (-np.pi, np.pi))
            rgb = _sample_lut_rgb(lut, lut_position)
        else:
            color, _magnitude = apply_phase_lut(data, mapping.lut_data)
            intensity = np.ones_like(scalar, dtype=np.float32) if levels is None else window_intensity(scalar, levels)
            rgb = np.clip(color.astype(np.float32) * intensity[..., np.newaxis], 0.0, 255.0).astype(np.uint8)
        alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
        alpha[~np.isfinite(scalar), 0] = 0
        return np.concatenate((rgb, alpha), axis=-1)
    scalar = mapped_scalar(data, mapping)
    levels = mapping.levels or finite_default_levels(scalar)
    intensity = window_intensity(scalar, levels)
    alpha = np.full(intensity.shape + (1,), 255, dtype=np.uint8)
    alpha[~np.isfinite(scalar), 0] = 0
    rgb = np.clip(intensity[..., np.newaxis] * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.concatenate((rgb, rgb, rgb, alpha), axis=-1)


def pack_texture_data(data, texture_kind: TexturePlaneKind | str) -> np.ndarray:
    kind = _coerce_enum(TexturePlaneKind, texture_kind)
    arr = np.asarray(data)
    if kind == TexturePlaneKind.SCALAR_R32F:
        return np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    if kind == TexturePlaneKind.COMPLEX_RG32F:
        if np.iscomplexobj(arr):
            packed = np.empty(arr.shape + (2,), dtype=np.float32)
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
        (0 <= h) & (h < 1),
        (1 <= h) & (h < 2),
        (2 <= h) & (h < 3),
        (3 <= h) & (h < 4),
        (4 <= h) & (h < 5),
        (5 <= h) & (h < 6),
    )
    choices = ((c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x))
    for mask, choice in zip(masks, choices):
        for channel, value in enumerate(choice):
            rgb[mask, channel] = value if np.isscalar(value) else value[mask]
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def _lut_rgb_uint8(lut: np.ndarray | None) -> np.ndarray:
    lut_array = default_phase_lut() if lut is None else np.asarray(lut)
    if lut_array.ndim != 2 or lut_array.shape[0] < 1 or lut_array.shape[1] < 3:
        raise ValueError("phase LUT must have shape (N, 3) or (N, 4)")
    color = lut_array[:, :3]
    if color.dtype != np.uint8:
        max_value = 1.0 if np.issubdtype(color.dtype, np.floating) and color.size and np.nanmax(color) <= 1.0 else 255.0
        color = np.clip(color.astype(np.float32) * (255.0 / max_value), 0.0, 255.0).astype(np.uint8)
    return color.astype(np.uint8, copy=False)


def _sample_lut_rgb(lut: np.ndarray, position: np.ndarray) -> np.ndarray:
    lut = np.asarray(lut, dtype=np.float32)
    if lut.shape[0] == 1:
        return np.broadcast_to(lut[0].astype(np.uint8), np.asarray(position).shape + (3,))
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


__all__ = [
    "ShaderDisplayMode",
    "ShaderComponent",
    "ShaderScale",
    "ShaderMapping",
    "common_shader_mapping",
    "TexturePlaneKind",
    "extract_component",
    "shader_component_uniform",
    "apply_scale",
    "mapped_scalar",
    "window_intensity",
    "phase_lut_indices",
    "apply_phase_lut",
    "cpu_display_rgba",
    "pack_texture_data",
    "finite_default_levels",
    "default_phase_lut",
]
