"""Pure runtime memory policy for ArrayScope."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from arrayscope.core.memory_budget import format_bytes


MiB = 1024 * 1024
GiB = 1024 * MiB
DEFAULT_RENDER_CAP_MB = 512
MIN_RENDER_CAP_MB = 128
MAX_RENDER_CAP_MB = 8192


class MemoryProfileChoice(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass(frozen=True)
class SystemMemorySnapshot:
    total_bytes: int
    available_bytes: int
    process_rss_bytes: int
    source: str = "psutil"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryPolicy:
    profile: MemoryProfileChoice
    system_total_bytes: int
    system_available_bytes: int
    process_rss_bytes: int
    input_nbytes: int | None
    user_render_cap_bytes: int

    visible_render_budget_bytes: int
    montage_canvas_budget_bytes: int
    single_tile_budget_bytes: int
    image_cache_budget_bytes: int
    tile_cache_budget_bytes: int
    profile_cache_budget_bytes: int
    stage_cache_budget_bytes: int
    prefetch_budget_bytes: int
    operation_prefetch_peak_budget_bytes: int
    fft_prefetch_peak_budget_bytes: int

    cache_prefetch_skip_fraction: float = 0.80
    warnings: tuple[str, ...] = ()


def normalize_memory_profile_choice(value) -> MemoryProfileChoice:
    if isinstance(value, MemoryProfileChoice):
        return value
    candidate = getattr(value, "value", value)
    try:
        return MemoryProfileChoice(str(candidate))
    except Exception:
        return MemoryProfileChoice.BALANCED


def sample_system_memory(*, psutil_module=None) -> SystemMemorySnapshot:
    try:
        if psutil_module is None:
            import psutil as psutil_module
        virtual = psutil_module.virtual_memory()
        rss = psutil_module.Process().memory_info().rss
        return SystemMemorySnapshot(
            total_bytes=int(virtual.total),
            available_bytes=int(virtual.available),
            process_rss_bytes=int(rss),
            source="psutil",
        )
    except Exception:
        return SystemMemorySnapshot(
            total_bytes=8 * GiB,
            available_bytes=4 * GiB,
            process_rss_bytes=0,
            source="fallback",
            warnings=("psutil unavailable; using fallback memory policy inputs",),
        )


def input_nbytes_for(data) -> int | None:
    nbytes = getattr(data, "nbytes", None)
    if nbytes is None:
        return None
    try:
        return int(nbytes)
    except Exception:
        return None


def compute_memory_policy(
    *,
    profile: MemoryProfileChoice | str,
    render_cap_mb: int,
    input_nbytes: int | None,
    system: SystemMemorySnapshot | None = None,
) -> MemoryPolicy:
    profile = normalize_memory_profile_choice(profile)
    system = sample_system_memory() if system is None else system
    total = max(1, int(system.total_bytes))
    available = max(1, int(system.available_bytes))
    input_nbytes = None if input_nbytes is None else max(0, int(input_nbytes))
    render_cap_mb = _normalize_render_cap_mb(render_cap_mb)
    render_cap = int(render_cap_mb) * MiB

    if profile == MemoryProfileChoice.CONSERVATIVE:
        visible = _clamp(min(available * 0.05, total * 0.02), 128 * MiB, min(render_cap, 1 * GiB))
        image_cache = _clamp(min(available * 0.05, max(128 * MiB, visible)), 128 * MiB, 1 * GiB)
        tile_cache = _clamp(min(available * 0.08, max(256 * MiB, input_nbytes or visible)), 256 * MiB, 2 * GiB)
        profile_cache = _clamp(min(available * 0.02, 128 * MiB), 32 * MiB, 256 * MiB)
        stage_cache = _clamp(min(available * 0.15, max(input_nbytes or 0, 256 * MiB)), 256 * MiB, 4 * GiB)
        prefetch = min(128 * MiB, stage_cache * 0.05, max(1, visible // 2))
        operation_prefetch_peak = min(32 * MiB, max(1, visible // 10), prefetch)
        fft_prefetch_peak = min(16 * MiB, operation_prefetch_peak)
    elif profile == MemoryProfileChoice.AGGRESSIVE:
        visible = _clamp(min(available * 0.18, total * 0.08), 256 * MiB, min(render_cap, 4 * GiB))
        image_cache = _clamp(min(available * 0.15, max(512 * MiB, 2 * visible)), 512 * MiB, 4 * GiB)
        tile_cache = _clamp(min(available * 0.25, max(2 * visible, 2 * (input_nbytes or 0))), 512 * MiB, 8 * GiB)
        profile_cache = _clamp(min(available * 0.05, 512 * MiB), 128 * MiB, 1 * GiB)
        stage_cache = _clamp(min(available * 0.45, max(3 * (input_nbytes or 0), 1 * GiB)), 512 * MiB, 16 * GiB)
        prefetch = min(512 * MiB, stage_cache * 0.15, visible)
        operation_prefetch_peak = min(128 * MiB, max(1, visible // 6), prefetch)
        fft_prefetch_peak = min(64 * MiB, operation_prefetch_peak)
    elif profile == MemoryProfileChoice.CUSTOM:
        visible = render_cap
        image_cache = _clamp(min(available * 0.10, max(256 * MiB, render_cap)), 128 * MiB, 2 * GiB)
        tile_cache = _clamp(min(available * 0.15, max(render_cap, input_nbytes or 0)), 256 * MiB, 4 * GiB)
        profile_cache = _clamp(min(available * 0.03, 256 * MiB), 64 * MiB, 512 * MiB)
        stage_cache = _clamp(min(available * 0.30, max(2 * (input_nbytes or 0), 512 * MiB)), 256 * MiB, 8 * GiB)
        prefetch = min(256 * MiB, stage_cache * 0.10, max(1, render_cap // 2))
        operation_prefetch_peak = min(64 * MiB, max(1, render_cap // 8), prefetch)
        fft_prefetch_peak = min(32 * MiB, operation_prefetch_peak)
    else:
        visible = _clamp(min(available * 0.10, total * 0.04), 128 * MiB, min(render_cap, 2 * GiB))
        image_cache = _clamp(min(available * 0.10, max(256 * MiB, visible)), 256 * MiB, 2 * GiB)
        tile_cache = _clamp(min(available * 0.15, max(visible, input_nbytes or 0)), 256 * MiB, 4 * GiB)
        profile_cache = _clamp(min(available * 0.03, 256 * MiB), 64 * MiB, 512 * MiB)
        stage_cache = _clamp(min(available * 0.30, max(2 * (input_nbytes or 0), 512 * MiB)), 256 * MiB, 8 * GiB)
        prefetch = min(256 * MiB, stage_cache * 0.10, visible)
        operation_prefetch_peak = min(64 * MiB, max(1, visible // 8), prefetch)
        fft_prefetch_peak = min(32 * MiB, operation_prefetch_peak)

    visible = int(visible)
    return MemoryPolicy(
        profile=profile,
        system_total_bytes=total,
        system_available_bytes=available,
        process_rss_bytes=max(0, int(system.process_rss_bytes)),
        input_nbytes=input_nbytes,
        user_render_cap_bytes=render_cap,
        visible_render_budget_bytes=visible,
        montage_canvas_budget_bytes=visible,
        single_tile_budget_bytes=visible,
        image_cache_budget_bytes=int(image_cache),
        tile_cache_budget_bytes=int(tile_cache),
        profile_cache_budget_bytes=int(profile_cache),
        stage_cache_budget_bytes=int(stage_cache),
        prefetch_budget_bytes=int(prefetch),
        operation_prefetch_peak_budget_bytes=int(operation_prefetch_peak),
        fft_prefetch_peak_budget_bytes=int(fft_prefetch_peak),
        warnings=tuple(system.warnings),
    )


def apply_policy_hysteresis(
    previous: MemoryPolicy | None,
    current: MemoryPolicy,
    *,
    active_render: bool = False,
) -> MemoryPolicy:
    if previous is None:
        return current
    if (
        previous.profile != current.profile
        or previous.user_render_cap_bytes != current.user_render_cap_bytes
        or previous.input_nbytes != current.input_nbytes
    ):
        return current

    updates = {}
    if active_render:
        for name in (
            "visible_render_budget_bytes",
            "montage_canvas_budget_bytes",
            "single_tile_budget_bytes",
        ):
            updates[name] = max(getattr(previous, name), getattr(current, name))

    old_available = max(1, int(previous.system_available_bytes))
    change_fraction = abs(int(current.system_available_bytes) - old_available) / float(old_available)
    if change_fraction < 0.20:
        for name in (
            "image_cache_budget_bytes",
            "tile_cache_budget_bytes",
            "profile_cache_budget_bytes",
            "stage_cache_budget_bytes",
            "prefetch_budget_bytes",
            "operation_prefetch_peak_budget_bytes",
            "fft_prefetch_peak_budget_bytes",
        ):
            updates[name] = getattr(previous, name)

    if not updates:
        return current
    return replace(current, **updates)


def format_memory_policy(policy: MemoryPolicy) -> str:
    lines = [
        f"Profile: {policy.profile.value}",
        f"System total: {format_bytes(policy.system_total_bytes)}",
        f"System available: {format_bytes(policy.system_available_bytes)}",
        f"Process RSS: {format_bytes(policy.process_rss_bytes)}",
        f"Input: {'n/a' if policy.input_nbytes is None else format_bytes(policy.input_nbytes)}",
        f"Render cap: {format_bytes(policy.user_render_cap_bytes)}",
        f"Visible render budget: {format_bytes(policy.visible_render_budget_bytes)}",
        f"Montage canvas budget: {format_bytes(policy.montage_canvas_budget_bytes)}",
        f"Single tile budget: {format_bytes(policy.single_tile_budget_bytes)}",
        f"Image cache budget: {format_bytes(policy.image_cache_budget_bytes)}",
        f"Tile cache budget: {format_bytes(policy.tile_cache_budget_bytes)}",
        f"Profile cache budget: {format_bytes(policy.profile_cache_budget_bytes)}",
        f"Stage cache budget: {format_bytes(policy.stage_cache_budget_bytes)}",
        f"Prefetch budget: {format_bytes(policy.prefetch_budget_bytes)}",
        f"Operation prefetch peak: {format_bytes(policy.operation_prefetch_peak_budget_bytes)}",
        f"FFT prefetch peak: {format_bytes(policy.fft_prefetch_peak_budget_bytes)}",
    ]
    lines.extend(f"Warning: {warning}" for warning in policy.warnings)
    return "\n".join(lines)


def _normalize_render_cap_mb(value) -> int:
    try:
        mb = int(value)
    except Exception:
        return DEFAULT_RENDER_CAP_MB
    return max(MIN_RENDER_CAP_MB, min(MAX_RENDER_CAP_MB, mb))


def _clamp(value, low, high) -> int:
    low = int(low)
    high = int(high)
    if high < low:
        return high
    return int(max(low, min(high, int(value))))
