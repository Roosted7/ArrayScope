from arrayscope.app.settings_state import AppSettingsState, FFTWorkersChoice
from arrayscope.core.compute_policy import ComputeLane, compute_policy_from_settings


def test_auto_compute_policy_caps_tile_fft_workers_at_one():
    policy = compute_policy_from_settings(AppSettingsState(fft_workers=FFTWorkersChoice.AUTO), cpu_count=16)

    assert policy.montage_tile_workers == 8
    assert policy.fft_workers_tile == 1
    assert policy.montage_tile_workers * policy.fft_workers_tile <= 8


def test_visible_and_stage_use_resolved_auto_fft_workers():
    policy = compute_policy_from_settings(AppSettingsState(fft_workers=FFTWorkersChoice.AUTO), cpu_count=16)

    assert policy.fft_workers_visible == 8
    assert policy.fft_workers_stage == 8
    assert policy.fft_workers_for_lane(ComputeLane.VISIBLE) == 8
    assert policy.fft_workers_for_lane(ComputeLane.STAGE) == 8


def test_small_cpu_policy_keeps_tile_worker_product_conservative():
    policy = compute_policy_from_settings(AppSettingsState(fft_workers=FFTWorkersChoice.AUTO), cpu_count=2)

    assert policy.fft_workers_tile == 1
    assert policy.montage_tile_workers * policy.fft_workers_tile <= 2
