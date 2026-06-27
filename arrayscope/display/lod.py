"""Tile LOD demand and native-only production policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LOD_POLICY_NATIVE_ONLY = "native-only"
LOD_REASON_NATIVE_SCALE = "native-resolution texture is appropriate at the current scale"
LOD_REASON_ASYNC_RESIDENCY_REQUIRED = (
    "desired LOD is deferred until asynchronous multi-resolution residency can retain adjacent levels"
)
LOD_REASON_INVALID_VIEW = "native resolution selected because viewport LOD demand could not be measured"


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


@dataclass(frozen=True)
class LodDemand:
    desired_level: int
    desired_factor: int
    desired_factor_xy: tuple[int, int]
    acceptable_levels: tuple[int, ...]
    source_texels_per_pixel_xy: tuple[float, float]
    reason: str

    def __post_init__(self) -> None:
        level = max(0, int(self.desired_level))
        factor = max(1, int(self.desired_factor))
        factor_xy = _factor2(self.desired_factor_xy)
        texels_xy = _float2(self.source_texels_per_pixel_xy)
        levels = tuple(sorted({max(0, int(value)) for value in tuple(self.acceptable_levels or (level,))}))
        object.__setattr__(self, "desired_level", level)
        object.__setattr__(self, "desired_factor", factor)
        object.__setattr__(self, "desired_factor_xy", factor_xy)
        object.__setattr__(self, "acceptable_levels", levels or (level,))
        object.__setattr__(self, "source_texels_per_pixel_xy", texels_xy)
        object.__setattr__(self, "reason", str(self.reason))


@dataclass(frozen=True)
class LodPolicyDecision:
    demand: LodDemand
    applied_level: int = 0
    applied_factor: int = 1
    applied_factor_xy: tuple[int, int] = (1, 1)
    policy: str = LOD_POLICY_NATIVE_ONLY
    reason: str = LOD_REASON_NATIVE_SCALE

    def __post_init__(self) -> None:
        object.__setattr__(self, "applied_level", max(0, int(self.applied_level)))
        object.__setattr__(self, "applied_factor", max(1, int(self.applied_factor)))
        object.__setattr__(self, "applied_factor_xy", _factor2(self.applied_factor_xy))
        object.__setattr__(self, "policy", str(self.policy))
        object.__setattr__(self, "reason", str(self.reason))


def select_lod_demand(
    view_range,
    viewport_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    target_min: float = 1.0,
    target_max: float = 2.0,
    previous_factor: int | None = None,
    hysteresis: float = 0.15,
) -> LodDemand:
    """Return desired display quality without promising materialization."""

    tile_height, tile_width = (max(1, int(value)) for value in tile_shape)
    native = LodDemand(
        desired_level=0,
        desired_factor=1,
        desired_factor_xy=(min(1, tile_width), min(1, tile_height)),
        acceptable_levels=(0,),
        source_texels_per_pixel_xy=(0.0, 0.0),
        reason=LOD_REASON_INVALID_VIEW,
    )
    if view_range is None:
        return native
    try:
        x_range, y_range = view_range
        world_w = abs(float(x_range[1]) - float(x_range[0]))
        world_h = abs(float(y_range[1]) - float(y_range[0]))
    except Exception:
        return native
    viewport_h, viewport_w = (max(1, int(viewport_shape[0])), max(1, int(viewport_shape[1])))
    texels_x = world_w / viewport_w
    texels_y = world_h / viewport_h
    if not np.isfinite(texels_x) or not np.isfinite(texels_y) or texels_x < 0.0 or texels_y < 0.0:
        return native
    factor_x = _desired_factor_for_texels(texels_x, target_min=target_min, target_max=target_max)
    factor_y = _desired_factor_for_texels(texels_y, target_min=target_min, target_max=target_max)
    factor = max(factor_x, factor_y)
    texels_per_pixel = max(texels_x, texels_y)
    if previous_factor is None:
        desired = factor
    else:
        desired = _apply_scalar_hysteresis(
            factor,
            texels_per_pixel=texels_per_pixel,
            previous_factor=previous_factor,
            target_min=target_min,
            hysteresis=hysteresis,
        )
    desired_factor_xy = _hysteresis_adjusted_factor_xy((factor_x, factor_y), desired)
    level = _level_for_factor(desired)
    acceptable_levels = tuple(range(max(0, level - 1), level + 2))
    reason = _demand_reason((texels_x, texels_y), desired=desired)
    return LodDemand(
        desired_level=level,
        desired_factor=desired,
        desired_factor_xy=desired_factor_xy,
        acceptable_levels=acceptable_levels,
        source_texels_per_pixel_xy=(texels_x, texels_y),
        reason=reason,
    )


def native_lod_policy(
    view_range,
    viewport_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    *,
    previous_factor: int | None = None,
) -> LodPolicyDecision:
    """Return the production policy: demand may exceed native; applied never does."""

    demand = select_lod_demand(
        view_range,
        viewport_shape,
        tile_shape,
        previous_factor=previous_factor,
    )
    if demand.reason == LOD_REASON_INVALID_VIEW:
        reason = LOD_REASON_INVALID_VIEW
    elif demand.desired_factor > 1:
        reason = LOD_REASON_ASYNC_RESIDENCY_REQUIRED
    else:
        reason = LOD_REASON_NATIVE_SCALE
    return LodPolicyDecision(
        demand=demand,
        applied_level=0,
        applied_factor=1,
        applied_factor_xy=(1, 1),
        policy=LOD_POLICY_NATIVE_ONLY,
        reason=reason,
    )


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


def _desired_factor_for_texels(texels_per_pixel: float, *, target_min: float, target_max: float) -> int:
    if not np.isfinite(texels_per_pixel) or texels_per_pixel <= float(target_max):
        return 1
    factor = 1
    while texels_per_pixel / (factor * 2) >= float(target_min):
        factor *= 2
    return max(1, int(factor))


def _apply_scalar_hysteresis(
    factor: int,
    *,
    texels_per_pixel: float,
    previous_factor: int,
    target_min: float,
    hysteresis: float,
) -> int:
    previous = max(1, int(previous_factor))
    factor = max(1, int(factor))
    hysteresis = max(0.0, min(0.45, float(hysteresis)))
    if factor > previous:
        promote_at = 2.0 * previous * float(target_min) * (1.0 + hysteresis)
        if texels_per_pixel < promote_at:
            return previous
    elif factor < previous:
        demote_below = previous * float(target_min) * (1.0 - hysteresis)
        if texels_per_pixel >= demote_below:
            return previous
    return factor


def _level_for_factor(factor: int) -> int:
    factor = max(1, int(factor))
    return max(0, int(np.log2(factor)))


def _hysteresis_adjusted_factor_xy(raw_xy: tuple[int, int], desired: int) -> tuple[int, int]:
    desired = max(1, int(desired))
    raw_x, raw_y = _factor2(raw_xy)
    raw_max = max(raw_x, raw_y)
    if raw_max == desired:
        return (raw_x, raw_y)
    if raw_max > desired:
        return (min(raw_x, desired), min(raw_y, desired))
    return (
        desired if raw_x == raw_max else raw_x,
        desired if raw_y == raw_max else raw_y,
    )


def _demand_reason(texels_xy: tuple[float, float], *, desired: int) -> str:
    texels_x, texels_y = texels_xy
    if desired <= 1:
        return LOD_REASON_NATIVE_SCALE
    lo = max(1e-12, min(texels_x, texels_y))
    hi = max(texels_x, texels_y)
    if hi / lo >= 4.0:
        return "zoomed-out anisotropic viewport prefers coarser display LOD on one axis"
    return "zoomed-out viewport prefers coarser display LOD"


def _factor2(values) -> tuple[int, int]:
    x, y = tuple(values)[:2]
    return (max(1, int(x)), max(1, int(y)))


def _float2(values) -> tuple[float, float]:
    x, y = tuple(values)[:2]
    x = float(x)
    y = float(y)
    return (x if np.isfinite(x) else 0.0, y if np.isfinite(y) else 0.0)


def _shape2(shape) -> tuple[int, int]:
    values = tuple(int(value) for value in tuple(shape)[:2])
    if len(values) != 2:
        raise ValueError("shape must have at least two dimensions")
    return values


__all__ = [
    "LodInfo",
    "LodDemand",
    "LodPolicyDecision",
    "LOD_POLICY_NATIVE_ONLY",
    "LOD_REASON_NATIVE_SCALE",
    "LOD_REASON_ASYNC_RESIDENCY_REQUIRED",
    "LOD_REASON_INVALID_VIEW",
    "select_lod_demand",
    "native_lod_policy",
    "inner_uv_for_gutter",
]
