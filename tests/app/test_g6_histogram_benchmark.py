from arrayscope.tools.g6_histogram_benchmark import Measurement, _scenario_sources, _summaries


def _measurement(
    scenario: str,
    variant: str,
    *,
    submit_max_ms: float,
    total_wall_ms: float,
    heartbeat_max_gap_ms: float,
) -> Measurement:
    return Measurement(
        scenario=scenario,
        variant=variant,
        repetition=0,
        source_count=1 if scenario == "single_slice" else 60,
        resident_lod=0 if variant == "exact" else 2,
        bins=500 if variant == "exact" else 64,
        source_pixels=112_896,
        batch_sources=4,
        submit_wall_ms=submit_max_ms,
        submit_max_ms=submit_max_ms,
        fence_wall_ms=1.0,
        resolve_wall_ms=1.0,
        first_batch_wall_ms=10.0,
        total_wall_ms=total_wall_ms,
        gpu_compute_ms=1.0,
        heartbeat_max_gap_ms=heartbeat_max_gap_ms,
        reconstructed_values=8_192 if variant == "exact" else 512,
        finite_weight=112_896,
    )


def test_g6b_representative_sessions_pin_native_singleton_and_l2_montages():
    scenarios = _scenario_sources(272)

    assert [(name, len(sources), lod) for name, sources, lod in scenarios] == [
        ("single_slice", 1, 0),
        ("montage_60", 60, 2),
        ("montage_272", 272, 2),
    ]


def test_g6b_summary_rejects_a_montage_heartbeat_regression():
    rows = (
        _measurement(
            "single_slice",
            "exact",
            submit_max_ms=2.0,
            total_wall_ms=12.0,
            heartbeat_max_gap_ms=3.0,
        ),
        _measurement(
            "montage_60",
            "exact",
            submit_max_ms=8.0,
            total_wall_ms=700.0,
            heartbeat_max_gap_ms=20.0,
        ),
    )

    summaries = {
        (row["scenario"], row["variant"]): row for row in _summaries(rows)
    }

    assert summaries[("single_slice", "exact")]["fits_phase1_budget"] is True
    assert summaries[("montage_60", "exact")]["fits_phase1_budget"] is False
