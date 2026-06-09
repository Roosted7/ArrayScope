from types import SimpleNamespace

import numpy as np

from arrayscope.core.memory_policy import (
    GiB,
    MiB,
    MemoryProfileChoice,
    SystemMemorySnapshot,
    apply_policy_hysteresis,
    compute_memory_policy,
    format_memory_policy,
    input_nbytes_for,
    sample_system_memory,
)


class _Psutil:
    @staticmethod
    def virtual_memory():
        return SimpleNamespace(total=16 * GiB, available=10 * GiB)

    @staticmethod
    def Process():
        return SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=123 * MiB))


class _BrokenPsutil:
    @staticmethod
    def virtual_memory():
        raise RuntimeError("nope")


def _system(available=8 * GiB, total=16 * GiB):
    return SystemMemorySnapshot(total_bytes=total, available_bytes=available, process_rss_bytes=256 * MiB)


def test_sample_system_memory_uses_psutil_values():
    snapshot = sample_system_memory(psutil_module=_Psutil)

    assert snapshot.total_bytes == 16 * GiB
    assert snapshot.available_bytes == 10 * GiB
    assert snapshot.process_rss_bytes == 123 * MiB
    assert snapshot.source == "psutil"


def test_sample_system_memory_fallback_is_deterministic():
    snapshot = sample_system_memory(psutil_module=_BrokenPsutil)

    assert snapshot.total_bytes == 8 * GiB
    assert snapshot.available_bytes == 4 * GiB
    assert snapshot.process_rss_bytes == 0
    assert snapshot.source == "fallback"
    assert snapshot.warnings


def test_balanced_policy_uses_render_cap_for_visible_and_montage():
    policy = compute_memory_policy(profile="balanced", render_cap_mb=512, input_nbytes=1 * GiB, system=_system())

    assert policy.visible_render_budget_bytes == 512 * MiB
    assert policy.montage_canvas_budget_bytes == 512 * MiB
    assert policy.single_tile_budget_bytes == 512 * MiB


def test_conservative_policy_has_smaller_budgets_than_balanced():
    conservative = compute_memory_policy(profile="conservative", render_cap_mb=2048, input_nbytes=1 * GiB, system=_system())
    balanced = compute_memory_policy(profile="balanced", render_cap_mb=2048, input_nbytes=1 * GiB, system=_system())

    assert conservative.visible_render_budget_bytes < balanced.visible_render_budget_bytes
    assert conservative.image_cache_budget_bytes < balanced.image_cache_budget_bytes


def test_aggressive_policy_has_larger_cache_budgets_than_balanced():
    aggressive = compute_memory_policy(profile="aggressive", render_cap_mb=4096, input_nbytes=1 * GiB, system=_system())
    balanced = compute_memory_policy(profile="balanced", render_cap_mb=4096, input_nbytes=1 * GiB, system=_system())

    assert aggressive.image_cache_budget_bytes > balanced.image_cache_budget_bytes
    assert aggressive.tile_cache_budget_bytes > balanced.tile_cache_budget_bytes


def test_custom_policy_uses_render_cap_as_visible_budget():
    policy = compute_memory_policy(profile="custom", render_cap_mb=1024, input_nbytes=1 * GiB, system=_system())

    assert policy.visible_render_budget_bytes == 1024 * MiB


def test_policy_hysteresis_keeps_cache_budgets_for_small_available_memory_change():
    previous = compute_memory_policy(profile="balanced", render_cap_mb=1024, input_nbytes=1, system=_system(available=8 * GiB))
    current = compute_memory_policy(profile="balanced", render_cap_mb=1024, input_nbytes=1, system=_system(available=7 * GiB))

    policy = apply_policy_hysteresis(previous, current)

    assert policy.system_available_bytes == current.system_available_bytes
    assert policy.image_cache_budget_bytes == previous.image_cache_budget_bytes
    assert policy.tile_cache_budget_bytes == previous.tile_cache_budget_bytes


def test_policy_hysteresis_does_not_shrink_active_render_budgets():
    previous = compute_memory_policy(profile="balanced", render_cap_mb=2048, input_nbytes=1, system=_system(available=8 * GiB))
    current = compute_memory_policy(profile="balanced", render_cap_mb=2048, input_nbytes=1, system=_system(available=1 * GiB))

    policy = apply_policy_hysteresis(previous, current, active_render=True)

    assert policy.visible_render_budget_bytes == previous.visible_render_budget_bytes
    assert policy.montage_canvas_budget_bytes == previous.montage_canvas_budget_bytes


def test_prefetch_thresholds_are_not_larger_than_prefetch_budget():
    for profile in MemoryProfileChoice:
        policy = compute_memory_policy(profile=profile, render_cap_mb=1024, input_nbytes=1 * GiB, system=_system())
        assert policy.operation_prefetch_peak_budget_bytes <= policy.prefetch_budget_bytes
        assert policy.fft_prefetch_peak_budget_bytes <= policy.prefetch_budget_bytes


def test_format_memory_policy_includes_profile_and_budget_names():
    policy = compute_memory_policy(profile="balanced", render_cap_mb=512, input_nbytes=input_nbytes_for(np.zeros((4, 4))), system=_system())

    text = format_memory_policy(policy)

    assert "Profile: balanced" in text
    assert "Visible render budget" in text
    assert "Tile cache budget" in text
