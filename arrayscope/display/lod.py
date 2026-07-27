"""Tile LOD demand and native-only production policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOD_POLICY_NATIVE_ONLY = "native-only"
LOD_REASON_NATIVE_SCALE = "native-resolution texture is appropriate at the current scale"
LOD_REASON_NATIVE_POLICY = (
    "desired LOD is not applied: the native-only montage LOD policy is selected"
)
LOD_REASON_BACKEND_ADOPTION_PENDING = (
    "desired LOD awaits resident-LOD adoption on this backend (ADR 0050)"
)
LOD_REASON_INVALID_VIEW = (
    "native resolution selected because viewport LOD demand could not be measured"
)


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
        levels = tuple(
            sorted({max(0, int(value)) for value in tuple(self.acceptable_levels or (level,))})
        )
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
    allow_anisotropy: bool = True,
) -> LodDemand:
    """Return desired display quality without promising materialization.

    ``allow_anisotropy=False`` squares the per-axis factors off at the
    dominant axis for backends that cannot bind an anisotropic page (see
    ``ImageViewBackendCapabilities.anisotropic_lod_pages``).  Every pyramid
    key derives from ``desired_factor_xy`` through ``factor_xy_for_level``,
    so clamping the demand here is what keeps such a backend from planning
    pages its presenter would then have to refuse.
    """

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
    desired_factor_xy = (
        _hysteresis_adjusted_factor_xy((factor_x, factor_y), desired)
        if allow_anisotropy
        else (desired, desired)
    )
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
    demand: LodDemand | None = None,
    previous_factor: int | None = None,
    deferred_reason: str | None = None,
    allow_anisotropy: bool = True,
) -> LodPolicyDecision:
    """Return the native-only policy: demand may exceed native; applied never does.

    ``deferred_reason`` states *why* a desired factor > 1 is not applied
    (user-selected native-only policy vs. resident LOD not yet adopted on the
    active backend); it defaults to the policy-selected wording. A supplied
    ``demand`` is a round-owned snapshot and is applied without re-derivation.
    """

    if demand is None:
        demand = select_lod_demand(
            view_range,
            viewport_shape,
            tile_shape,
            previous_factor=previous_factor,
            allow_anisotropy=allow_anisotropy,
        )
    if demand.reason == LOD_REASON_INVALID_VIEW:
        reason = LOD_REASON_INVALID_VIEW
    elif demand.desired_factor > 1:
        reason = str(deferred_reason or LOD_REASON_NATIVE_POLICY)
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


LOD_POLICY_RESIDENT = "resident"
LOD_REASON_RESIDENT_MATCH = "demanded LOD level is resident and presented"
LOD_REASON_RESIDENT_NATIVE_FALLBACK = "native fallback shown while the demanded LOD materializes"
LOD_REASON_RESIDENT_FINER = "finer resident level presented while the demanded level materializes"
LOD_REASON_RESIDENT_COARSER = (
    "coarser resident level presented while the demanded level materializes"
)


def factor_xy_for_level(demand: LodDemand, level: int) -> tuple[int, int]:
    """Return per-axis reduction factors for applying ``level`` to ``demand``.

    Anisotropy from the demand is preserved and shifted with the level so a
    coarser or finer applied level keeps the demanded aspect treatment.  Each
    axis is clamped to ``[1, 2**level]`` and the dominant axis always equals
    ``2**level``.
    """

    level = max(0, int(level))
    if level == 0:
        return (1, 1)
    if level == int(demand.desired_level):
        return demand.desired_factor_xy
    factor_cap = 2**level
    shift = level - int(demand.desired_level)
    factors = []
    for axis_factor in demand.desired_factor_xy:
        axis_factor = max(1, int(axis_factor))
        scaled = axis_factor << shift if shift >= 0 else max(1, axis_factor >> -shift)
        factors.append(max(1, min(factor_cap, int(scaled))))
    factor_x, factor_y = factors
    if max(factor_x, factor_y) != factor_cap:
        if factor_x >= factor_y:
            factor_x = factor_cap
        else:
            factor_y = factor_cap
    return (factor_x, factor_y)


def choose_resident_level(demand: LodDemand, resident_levels) -> int:
    """Return the finest resident level that is no coarser than acceptable.

    Candidates are explicit resident display levels no coarser than the
    demanded level.  A finer resident level is already valid display data;
    demotion to a coarser demanded level is a memory-residency decision, not
    an LOD correctness requirement.  Level 0 is the fallback when no explicit
    display LOD is resident.
    """

    resident = {int(level) for level in tuple(resident_levels or ()) if int(level) > 0}
    desired = max(0, int(demand.desired_level))
    candidates = [level for level in resident if level <= desired]
    return (
        min(candidates, key=lambda level: resident_presentation_rank(level, desired))
        if candidates
        else 0
    )


def resident_presentation_rank(level: int, desired: int) -> tuple[int, int]:
    """Rank presentable resident levels without authorizing quality loss.

    A level finer than the current demand remains strictly preferable to every
    coarser level: zooming out is a camera/coverage change, not permission to
    replace higher-quality pixels.  If no finer/equal level exists, the nearest
    coarser fallback wins so complete coverage remains preferable to black.
    Physical eviction pressure is backend-owned and therefore does not enter
    this pure presentation rank.
    """

    level = max(0, int(level))
    desired = max(0, int(desired))
    if level <= desired:
        return (0, level)
    return (1, level - desired)


def resident_lod_policy(
    view_range,
    viewport_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    *,
    demand: LodDemand | None = None,
    previous_factor: int | None = None,
    resident_levels=(),
    allow_anisotropy: bool = True,
) -> LodPolicyDecision:
    """Apply resident data no coarser than the demanded semantic target.

    The applied level is always materialized-and-resident (level 0 counts as
    implicitly resident).  Finer/equal resident LODs may satisfy or improve
    tile presentation; coarser-only resident levels stay physical cache and do
    not decide semantic completion. A supplied ``demand`` is a round-owned
    snapshot and is applied without re-derivation.
    """

    if demand is None:
        demand = select_lod_demand(
            view_range,
            viewport_shape,
            tile_shape,
            previous_factor=previous_factor,
            allow_anisotropy=allow_anisotropy,
        )
    if demand.reason == LOD_REASON_INVALID_VIEW:
        return LodPolicyDecision(
            demand=demand,
            applied_level=0,
            applied_factor=1,
            applied_factor_xy=(1, 1),
            policy=LOD_POLICY_RESIDENT,
            reason=LOD_REASON_INVALID_VIEW,
        )
    applied_level = choose_resident_level(demand, resident_levels)
    if applied_level == demand.desired_level:
        reason = LOD_REASON_RESIDENT_MATCH if applied_level > 0 else LOD_REASON_NATIVE_SCALE
    elif applied_level == 0:
        reason = LOD_REASON_RESIDENT_NATIVE_FALLBACK
    elif applied_level < demand.desired_level:
        reason = LOD_REASON_RESIDENT_FINER
    else:
        reason = LOD_REASON_RESIDENT_COARSER
    return LodPolicyDecision(
        demand=demand,
        applied_level=applied_level,
        applied_factor=2**applied_level,
        applied_factor_xy=factor_xy_for_level(demand, applied_level),
        policy=LOD_POLICY_RESIDENT,
        reason=reason,
    )


def inner_uv_for_gutter(
    texture_shape: tuple[int, int], gutter: int = 1
) -> tuple[float, float, float, float]:
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


def _desired_factor_for_texels(
    texels_per_pixel: float, *, target_min: float, target_max: float
) -> int:
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
    "LOD_POLICY_NATIVE_ONLY",
    "LOD_POLICY_RESIDENT",
    "LOD_REASON_BACKEND_ADOPTION_PENDING",
    "LOD_REASON_INVALID_VIEW",
    "LOD_REASON_NATIVE_POLICY",
    "LOD_REASON_NATIVE_SCALE",
    "LOD_REASON_RESIDENT_COARSER",
    "LOD_REASON_RESIDENT_FINER",
    "LOD_REASON_RESIDENT_MATCH",
    "LOD_REASON_RESIDENT_NATIVE_FALLBACK",
    "LodDemand",
    "LodInfo",
    "LodPolicyDecision",
    "choose_resident_level",
    "factor_xy_for_level",
    "inner_uv_for_gutter",
    "native_lod_policy",
    "resident_lod_policy",
    "select_lod_demand",
]
