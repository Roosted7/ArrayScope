"""CPU-side tile LOD helpers for montage rendering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LodInfo:
    level: int
    factor: int
    source_shape: tuple[int, int]
    texture_shape: tuple[int, int]
    gutter: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", max(0, int(self.level)))
        factor = max(1, int(self.factor))
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "source_shape", _shape2(self.source_shape))
        object.__setattr__(self, "texture_shape", _shape2(self.texture_shape))
        object.__setattr__(self, "gutter", max(0, int(self.gutter)))


def select_lod_factor(
    view_range,
    viewport_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    target_min: float = 1.0,
    target_max: float = 2.0,
) -> int:
    """Choose a power-of-two source decimation factor for the current view."""

    del tile_shape
    if view_range is None:
        return 1
    try:
        x_range, y_range = view_range
        world_w = abs(float(x_range[1]) - float(x_range[0]))
        world_h = abs(float(y_range[1]) - float(y_range[0]))
    except Exception:
        return 1
    viewport_h, viewport_w = (max(1, int(viewport_shape[0])), max(1, int(viewport_shape[1])))
    texels_per_pixel = max(world_w / viewport_w, world_h / viewport_h)
    if not np.isfinite(texels_per_pixel) or texels_per_pixel <= float(target_max):
        return 1
    factor = 1
    while texels_per_pixel / (factor * 2) >= float(target_min):
        factor *= 2
    return max(1, int(factor))


def build_tile_lod_pyramid(data, *, max_level: int | None = None) -> tuple[np.ndarray, ...]:
    """Build a finite-aware 2x2 box-reduced pyramid.

    NaN and Inf samples are ignored. A reduced pixel is NaN only when all
    contributing samples are non-finite.
    """

    current = np.asarray(data)
    pyramid = [np.ascontiguousarray(current)]
    level = 0
    while min(current.shape[:2]) > 1 and (max_level is None or level < int(max_level)):
        current = _box_reduce_2x2(current)
        pyramid.append(np.ascontiguousarray(current))
        level += 1
    return tuple(pyramid)


def apply_tile_gutter(data, gutter: int = 1) -> np.ndarray:
    gutter = max(0, int(gutter))
    arr = np.asarray(data)
    if gutter == 0:
        return np.ascontiguousarray(arr)
    pad_width = [(gutter, gutter), (gutter, gutter)] + [(0, 0)] * max(0, arr.ndim - 2)
    return np.pad(arr, tuple(pad_width), mode="edge")


def inner_uv_for_gutter(texture_shape: tuple[int, int], gutter: int = 1) -> tuple[float, float, float, float]:
    height, width = _shape2(texture_shape)
    gutter = max(0, int(gutter))
    if gutter <= 0:
        return (0.0, 0.0, 1.0, 1.0)
    return (
        gutter / max(1, width),
        gutter / max(1, height),
        (width - gutter) / max(1, width),
        (height - gutter) / max(1, height),
    )


def lod_info_for_texture(
    *,
    level: int,
    factor: int,
    source_shape: tuple[int, int],
    texture_shape: tuple[int, int],
    gutter: int = 0,
) -> LodInfo:
    return LodInfo(level=level, factor=factor, source_shape=source_shape, texture_shape=texture_shape, gutter=gutter)


def _box_reduce_2x2(data) -> np.ndarray:
    arr = np.asarray(data)
    h, w = arr.shape[:2]
    out_shape = ((h + 1) // 2, (w + 1) // 2) + tuple(arr.shape[2:])
    work = np.full((out_shape[0] * 2, out_shape[1] * 2) + tuple(arr.shape[2:]), np.nan, dtype=_reduction_dtype(arr))
    work[:h, :w, ...] = arr.astype(work.dtype, copy=False)
    blocks = work.reshape(out_shape[0], 2, out_shape[1], 2, *arr.shape[2:])
    finite = np.isfinite(blocks)
    sums = np.where(finite, blocks, 0).sum(axis=(1, 3))
    counts = finite.sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        reduced = sums / counts
    reduced = np.where(counts > 0, reduced, np.nan)
    if np.iscomplexobj(arr):
        return reduced.astype(np.result_type(arr.dtype, np.complex64), copy=False)
    return reduced.astype(np.float32, copy=False)


def _reduction_dtype(arr) -> np.dtype:
    if np.iscomplexobj(arr):
        return np.dtype(np.complex64 if np.dtype(arr.dtype).itemsize <= 8 else np.complex128)
    return np.dtype(np.float32)


def _shape2(shape) -> tuple[int, int]:
    values = tuple(int(value) for value in tuple(shape)[:2])
    if len(values) != 2:
        raise ValueError("shape must have at least two dimensions")
    return values


__all__ = [
    "LodInfo",
    "select_lod_factor",
    "build_tile_lod_pyramid",
    "apply_tile_gutter",
    "inner_uv_for_gutter",
    "lod_info_for_texture",
]
