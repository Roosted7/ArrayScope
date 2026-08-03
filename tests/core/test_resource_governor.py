import sys
import threading
from dataclasses import replace

from arrayscope.app.settings_state import AppSettingsState
from arrayscope.core.compute_policy import ComputeLane, compute_policy_from_settings
from arrayscope.core.gui_callback_budget import GuiCallbackObservation
from arrayscope.core.memory_policy import (
    MemoryPolicy,
    MemoryProfileChoice,
    SystemMemorySnapshot,
    compute_memory_policy,
)
from arrayscope.core.resource_governor import (
    ResourceGovernor,
    SchedulerBusyState,
    _render_pass_cost_model,
    _render_pass_extrapolation_cost_ms,
    _render_pass_extrapolation_weight,
    _render_pass_frame_aware_knees,
    _render_pass_latency_cost_ms,
    _render_pass_optimal_point,
    _render_pass_shared_terms,
    _RenderPassCostModel,
    _robust_linear_fit,
)
from arrayscope.core.resource_telemetry import CpuSnapshot, ResourceSnapshot


def _policy(profile=MemoryProfileChoice.BALANCED):
    return compute_policy_from_settings(AppSettingsState(memory_profile=profile), cpu_count=16)


def _memory(available_fraction=0.5) -> MemoryPolicy:
    total = 16 * 1024**3
    return compute_memory_policy(
        profile=MemoryProfileChoice.BALANCED,
        render_cap_mb=512,
        input_nbytes=128,
        system=SystemMemorySnapshot(total, int(total * available_fraction), 100),
    )


def _snapshot(memory: MemoryPolicy, cpu_percent=25.0) -> ResourceSnapshot:
    return ResourceSnapshot(
        memory=SystemMemorySnapshot(
            memory.system_total_bytes, memory.system_available_bytes, memory.process_rss_bytes
        ),
        cpu=CpuSnapshot(
            16,
            process_cpu_percent=20.0,
            system_cpu_percent=cpu_percent,
            load_average_1m=2.0,
            source="test",
        ),
        timestamp_monotonic=0.0,
    )


def test_bridge_drain_knob_uses_profile_feedback_defaults():
    governor = ResourceGovernor(
        _policy(MemoryProfileChoice.AGGRESSIVE), profile=MemoryProfileChoice.AGGRESSIVE
    )
    governor.update_telemetry(_snapshot(_memory()), _memory())

    decision = governor.decide_bridge_drain(interactive=False)

    assert decision.channel == "kernel_bridge_drain"
    assert decision.batch_limit == 18
    assert decision.budget_ms == 11.0
    assert decision.byte_cap == 0


def test_commit_batch_knob_covers_last_observed_bytes_per_item():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_present_total",
        12.0,
        item_count=6,
        byte_count=6 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="wgpu",
    )

    decision = governor.decide_commit_batch(interactive=False)

    assert decision.channel == "montage_present_total"
    assert decision.batch_limit >= 1
    assert decision.byte_cap >= decision.batch_limit * 1024 * 1024
    assert decision.interval_ms == 0
    assert decision.model == "ewma"


def test_render_pass_knob_owns_r5_target_and_adapts_per_item_work():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)

    cold = governor.decide_render_pass(interactive=False)
    governor.record_ui_observation(
        "montage_render_pass_preview",
        80.0,
        item_count=8,
        byte_count=8 * 1024,
        work_class="presentation_upsert",
        backend="pyqtgraph",
    )
    adapted = governor.decide_render_pass(interactive=False)

    assert cold.channel == "montage_render_pass_preview"
    assert cold.batch_limit == 1
    assert cold.budget_ms == 32.0
    assert adapted.batch_limit == 4
    assert adapted.budget_ms == 32.0
    assert adapted.model == "r5-feedback"


def test_render_pass_knob_still_shrinks_per_item_work_after_overrun():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_render_pass_target",
        80.0,
        item_count=8,
        byte_count=8 * 1024,
        work_class="presentation_upsert",
        backend="pyqtgraph",
    )

    adapted = governor.decide_render_pass(
        interactive=False,
        pass_kind="target",
    )

    assert adapted.batch_limit == 4
    assert adapted.model == "r5-feedback"


def test_retained_fallback_refinement_caps_admission_and_render_pass():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    for item_count in (8, 16, 32):
        governor.record_ui_observation(
            "montage_render_pass_target",
            20.0,
            item_count=item_count,
            byte_count=item_count * 1024,
            work_class="presentation_upsert",
            backend="wgpu",
        )

    ordinary = governor.decide_render_pass(
        interactive=False,
        pass_kind="target",
        remaining_items=272,
    )
    retained = governor.decide_render_pass(
        interactive=False,
        pass_kind="target",
        remaining_items=272,
        retained_fallback_refinement=True,
    )

    assert ordinary.batch_limit > 4
    assert retained.batch_limit == 4
    assert "retained fallback refinement cap=4" in retained.details
    assert (
        governor.decide_ladder_admission(
            default_limit=32,
            retained_fallback_refinement=True,
        )
        == 4
    )
    assert (
        governor.decide_ladder_admission(
            default_limit=32,
            retained_fallback_refinement=False,
        )
        == 32
    )
    rebind = governor.decide_resident_crop_rebind(remaining_items=272)
    assert rebind.batch_limit == 32
    assert rebind.model == "named-policy"


def test_render_pass_latches_item_independent_cost_and_recovers_floor():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_render_pass_preview",
        41.64,
        item_count=2,
        work_class="presentation_upsert",
        backend="wgpu",
    )
    assert governor.decide_render_pass(interactive=False).batch_limit == 1
    governor.record_ui_observation(
        "montage_render_pass_preview",
        41.48,
        item_count=1,
        work_class="presentation_upsert",
        backend="wgpu",
    )

    recovered = governor.decide_render_pass(interactive=False, remaining_items=272)

    assert recovered.batch_limit > 2
    assert any("fixed=" in detail and "item=" in detail for detail in recovered.details)
    governor.record_ui_observation(
        "montage_render_pass_preview",
        47.11,
        item_count=2,
        work_class="presentation_upsert",
        backend="wgpu",
    )
    assert governor.decide_render_pass(interactive=False, remaining_items=272).batch_limit >= 2


def test_render_pass_splits_identifiable_fixed_item_and_byte_costs():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    observations = (
        (1, 1, 15.0),
        (2, 1, 17.0),
        (1, 2, 18.0),
        (2, 2, 20.0),
    )
    for items, mib, elapsed_ms in observations:
        governor.record_ui_observation(
            "montage_render_pass_preview",
            elapsed_ms,
            item_count=items,
            byte_count=mib * 1024 * 1024,
            work_class="presentation_upsert",
            backend="wgpu",
        )
        decision = governor.decide_render_pass(
            interactive=False,
            remaining_items=272,
        )

    assert any(
        "fixed=10.00ms item=2.00ms bytes=3.00ms/MiB" in detail for detail in decision.details
    )
    assert any(
        "fit-rms=0.00ms raw-rms=0.00ms outliers=0 fit-var=0.000ms2 "
        "r2=1.000 item-byte-independence=1.000 samples=4" in detail
        for detail in decision.details
    )


def test_render_pass_rejects_nearly_collinear_byte_attribution():
    governor = ResourceGovernor(_policy())
    for items, byte_mib, elapsed_ms in (
        (1, 1.00, 12.0),
        (2, 2.01, 14.0),
        (4, 4.00, 18.0),
        (8, 8.01, 26.0),
    ):
        governor.record_ui_observation(
            "montage_render_pass_preview",
            elapsed_ms,
            item_count=items,
            byte_count=round(byte_mib * 1024 * 1024),
            work_class="presentation_upsert",
        )
        decision = governor.decide_render_pass(interactive=False, remaining_items=272)

    assert any("fixed=10.00ms item=2.00ms bytes=n/a/MiB" in row for row in decision.details)
    assert any("item-byte-independence=0.000" in row for row in decision.details)


def test_render_pass_fit_consumes_existing_measured_observation_log():
    governor = ResourceGovernor(_policy())
    for items, elapsed_ms in ((1, 10.0), (4, 34.0)):
        governor.record_ui_observation(
            "montage_render_pass_preview",
            elapsed_ms,
            item_count=items,
            work_class="presentation_upsert",
        )
        decision = governor.decide_render_pass(
            interactive=False,
            remaining_items=272,
        )

    assert any("fixed=2.00ms item=8.00ms" in detail for detail in decision.details)
    assert decision.batch_limit == 3
    assert "fit-observations=1@10.00ms@0B,4@34.00ms@0B" in decision.details


def test_render_pass_shares_structural_terms_and_warm_seeds_representation_terms():
    governor = ResourceGovernor(_policy())
    structure = ("wgpu", "persistent-delta", (336, 336), 17, 16, 1, 272)

    governor.begin_render_pass(
        "preview-a",
        pass_kind="preview",
        structural_key=structure,
        representation_key=("float32", "real"),
    )
    for items, mib, elapsed_ms in (
        (1, 1, 15.0),
        (2, 1, 17.0),
        (1, 2, 18.0),
        (2, 2, 20.0),
    ):
        governor.record_ui_observation(
            "montage_render_pass_preview",
            elapsed_ms,
            item_count=items,
            byte_count=mib * 1024 * 1024,
            work_class="presentation_upsert",
        )
        preview = governor.decide_render_pass(interactive=False, remaining_items=272)
    assert any("fixed=10.00ms item=2.00ms bytes=3.00ms/MiB" in row for row in preview.details)

    governor.begin_render_pass(
        "target-a",
        pass_kind="target",
        structural_key=structure,
        representation_key=("float32", "real"),
    )
    for items, mib, elapsed_ms in (
        (1, 1, 20.0),
        (2, 1, 27.0),
        (1, 2, 23.0),
        (2, 2, 30.0),
    ):
        governor.record_ui_observation(
            "montage_render_pass_target",
            elapsed_ms,
            item_count=items,
            byte_count=mib * 1024 * 1024,
            work_class="presentation_upsert",
        )
        target = governor.decide_render_pass(
            interactive=False,
            pass_kind="target",
            remaining_items=272,
        )
        if items == 1 and mib == 1:
            assert 4 <= target.batch_limit <= 16, (
                "one local cohort cannot identify the target item slope; the next decision "
                "must widen the evidence without discarding shared fixed/byte knowledge"
            )
    assert any("fixed=10.00ms item=7.00ms bytes=3.00ms/MiB" in row for row in target.details)

    governor.begin_render_pass(
        "preview-b",
        pass_kind="preview",
        structural_key=structure,
        representation_key=("complex64", "complex"),
    )
    warm = governor.decide_render_pass(interactive=False, remaining_items=272)
    assert any("fixed=10.00ms item=2.00ms bytes=3.00ms/MiB" in row for row in warm.details)
    assert any("seeded=1" in row and "uncertainty=" in row for row in warm.details)

    governor.begin_render_pass(
        "preview-c",
        pass_kind="preview",
        structural_key=("pyqtgraph", "cpu-delta", (336, 336), 17, 16, 1, 272),
        representation_key=("complex64", "complex"),
    )
    cold = governor.decide_render_pass(interactive=False, remaining_items=272)
    assert cold.batch_limit == 1
    assert any("fixed=n/a" in row for row in cold.details)


def test_render_pass_carries_model_seed_but_starts_fresh_round_evidence():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    structure = ("wgpu", "gpu-delta", (336, 336), 17, 16, 1, 272)
    representation = ("float32", "target")
    governor.begin_render_pass(
        "round-a",
        pass_kind="target",
        structural_key=structure,
        representation_key=representation,
    )
    for items, elapsed_ms in ((1, 20.0), (8, 27.0), (24, 43.0)):
        governor.record_ui_observation(
            "montage_render_pass_target",
            elapsed_ms,
            item_count=items,
            byte_count=items * 1024 * 1024,
            work_class="presentation_upsert",
        )
        governor.decide_render_pass(
            interactive=False,
            pass_kind="target",
            remaining_items=272,
        )

    governor.begin_render_pass(
        "round-b",
        pass_kind="target",
        structural_key=structure,
        representation_key=representation,
    )
    warm = governor.decide_render_pass(
        interactive=False,
        pass_kind="target",
        remaining_items=272,
    )

    assert any("seeded=1" in detail for detail in warm.details)
    assert "fit-observations=" in warm.details
    assert all("1@20.00ms" not in detail for detail in warm.details)


def test_render_pass_reports_curvature_as_residual_instead_of_fitting_an_unproven_term():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    for items in (1, 4, 8, 12):
        governor.record_ui_observation(
            "montage_render_pass_target",
            10.0 + items + 0.05 * items * items,
            item_count=items,
            work_class="presentation_upsert",
            backend="pyqtgraph",
        )
        decision = governor.decide_render_pass(
            interactive=False,
            pass_kind="target",
            remaining_items=272,
        )

    assert all("cohort2=" not in detail for detail in decision.details)
    assert any(
        "fit-rms=" in detail and "fit-rms=0.00ms" not in detail for detail in decision.details
    )


def test_render_pass_model_exposes_only_falsifiable_fixed_item_and_byte_terms():
    samples = [
        (20, 36.78, 20 * 1024 * 1024),
        (20, 33.16, 20 * 1024 * 1024),
        (24, 37.35, 24 * 1024 * 1024),
        (24, 41.06, 24 * 1024 * 1024),
        (24, 52.33, 24 * 1024 * 1024),
        (24, 47.79, 24 * 1024 * 1024),
        (1, 27.70, 0),
        (22, 47.87, 22 * 1024 * 1024),
    ]

    model = _render_pass_cost_model(
        samples,
        fixed_prior_ms=17.15,
    )

    assert not hasattr(model, "cohort_quadratic_ms")
    assert model.item_ms is not None
    assert model.residual_rms_ms is not None


def test_render_pass_linear_fit_rejects_one_high_leverage_stall():
    clean = _robust_linear_fit(
        [1.0, 4.0, 8.0, 16.0],
        [10.5, 12.0, 14.0, 18.0],
    )
    stalled = _robust_linear_fit(
        [1.0, 21.0, 3.0, 4.0, 4.0],
        [27.0, 167.0, 25.0, 25.0, 26.0],
    )

    assert clean == (10.0, 0.5)
    assert stalled is not None
    fixed_ms, item_ms = stalled
    assert 24.0 <= fixed_ms <= 27.0
    assert item_ms <= 0.5


def test_fixed_dominated_pass_escapes_a_small_cohort_seed_quickly():
    """The slow-fill attractor of 2026-07-31, encoded as a regression.

    A 272-tile refinement whose commits measured fixed-dominated (~40 ms per
    commit whether it carried 2 or 4 items) locked at 3-5 items per commit
    for 60+ commits: the exploration guard priced anything past ~1.1x the
    observed cohort as unexplored territory, small cohorts kept the observed
    range small, and the state re-confirmed itself on every observation.
    When the cost model itself says items are nearly free, exploration must
    widen by a real multiple per decision — a few commits to the amortizing
    cohort, not a 10% crawl the rolling window resets faster than it moves.
    """

    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    # A stable, tight, fixed-dominated fit over small cohorts: 40 ms plus
    # 0.1 ms per item, measured at 2-4 items across repeated commits.
    for count in (2, 3, 4, 2, 3, 4):
        governor.record_ui_observation(
            "montage_render_pass_target",
            40.0 + 0.1 * count,
            item_count=count,
            byte_count=count * 1024 * 1024,
            work_class="presentation_upsert",
            backend="wgpu",
        )
        decision = governor.decide_render_pass(
            interactive=False, pass_kind="target", remaining_items=272
        )
    assert decision.batch_limit >= 12, (
        "fixed-dominated evidence must widen the cohort by a real multiple "
        f"per decision, not pin it near the last small commit (got {decision.batch_limit})"
    )
    # And the escape must compound: measuring the decided cohort at the same
    # fixed-dominated cost must keep widening toward the amortizing optimum.
    governor.record_ui_observation(
        "montage_render_pass_target",
        40.0 + 0.1 * decision.batch_limit,
        item_count=decision.batch_limit,
        byte_count=decision.batch_limit * 1024 * 1024,
        work_class="presentation_upsert",
        backend="wgpu",
    )
    wider = governor.decide_render_pass(interactive=False, pass_kind="target", remaining_items=272)
    assert wider.batch_limit >= decision.batch_limit, (
        f"a measured-safe large cohort must not shrink back toward the seed "
        f"({decision.batch_limit} -> {wider.batch_limit})"
    )


def test_shared_byte_terms_pool_only_locally_identified_byte_fits():
    """A pooled byte prior must be evidence, not an echo.

    Byte and item counts are nearly collinear in a tiled fill (one page per
    tile), so most fits cannot split them. The pool once accepted byte terms
    that were themselves inherited from the pool, and one misattributed split
    then became every later pass's marginal cost: 4.14 ms/MiB steering vs
    0.52 ms/MiB measured on the same workload, which is the difference
    between 3-item and 13-item refinement cohorts (2026-07-31).
    """

    identified = _RenderPassCostModel(
        fixed_ms=20.0,
        item_ms=0.5,
        byte_ms_per_mib=0.6,
        byte_identified=True,
        samples=6,
        design_points=4,
        observed_item_min=2,
        observed_item_max=32,
        mean_elapsed_ms=40.0,
    )
    inherited = replace(identified, byte_ms_per_mib=4.0, byte_identified=False)

    _fixed, byte_ms = _render_pass_shared_terms((identified, inherited, inherited))

    assert byte_ms == 0.6, "inherited byte values must not vote in the pool"


def test_render_pass_weights_fill_time_against_continuous_latency_cost():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_render_pass_target",
        50.0,
        item_count=1,
        work_class="presentation_upsert",
        backend="wgpu",
    )
    governor.decide_render_pass(
        interactive=False,
        pass_kind="target",
        remaining_items=272,
    )
    governor.record_ui_observation(
        "montage_render_pass_target",
        50.1,
        item_count=2,
        work_class="presentation_upsert",
        backend="wgpu",
    )

    decision = governor.decide_render_pass(
        interactive=False,
        pass_kind="target",
        remaining_items=272,
    )

    # 50.0 ms at one item and 50.1 ms at two is an almost entirely fixed cost:
    # 49.9 ms fixed, 0.1 ms per item. With 272 remaining, the cohort that
    # actually minimizes the objective is ~136 items at ~63 ms per chunk --
    # deliberately over the 50 ms report line, because two 63 ms chunks beat
    # 272 single-item chunks of 50 ms each by two orders of magnitude of fill
    # time. Collapsing toward 1 here is the pathology, not the safe choice.
    #
    # Evidence only extends to two items, though, so the extrapolation term
    # must still hold the step short of that informed optimum: the governor
    # explores outward across successive decisions rather than trusting a
    # model fitted on two points at 68x the range it has measured.
    assert decision.batch_limit > 3, "fixed-dominated cost must not collapse the cohort"
    assert decision.batch_limit < 136, (
        "a two-point model must not jump straight to the informed optimum"
    )
    assert any(
        "steering=weighted-fill-latency-extrapolation" in detail for detail in decision.details
    )
    assert any("r5-achievable=1" in detail for detail in decision.details)

    # Widening the evidence must move the choice toward that optimum, not away.
    governor.record_ui_observation(
        "montage_render_pass_target",
        50.0 + 0.1 * decision.batch_limit,
        item_count=decision.batch_limit,
        work_class="presentation_upsert",
        backend="wgpu",
    )
    wider = governor.decide_render_pass(
        interactive=False,
        pass_kind="target",
        remaining_items=272,
    )
    assert wider.batch_limit > decision.batch_limit, (
        "the cohort must ratchet outward as the cost model gains range"
    )


def test_render_pass_latency_cost_is_zero_then_smooth_and_quadratic():
    assert _render_pass_latency_cost_ms(18.0) == 0.0
    assert 0.0 < _render_pass_latency_cost_ms(30.0) < _render_pass_latency_cost_ms(45.0)
    epsilon = 1e-4
    left_slope = (
        _render_pass_latency_cost_ms(45.0) - _render_pass_latency_cost_ms(45.0 - epsilon)
    ) / epsilon
    right_slope = (
        _render_pass_latency_cost_ms(45.0 + epsilon) - _render_pass_latency_cost_ms(45.0)
    ) / epsilon
    assert abs(left_slope - right_slope) < 1e-3
    assert _render_pass_latency_cost_ms(60.0) - _render_pass_latency_cost_ms(
        50.0
    ) > _render_pass_latency_cost_ms(50.0) - _render_pass_latency_cost_ms(45.0)


def test_render_pass_extrapolation_cost_is_smoothly_exponential():
    assert _render_pass_extrapolation_cost_ms(1.1) == 0.0
    at_two = _render_pass_extrapolation_cost_ms(2.0)
    at_ten = _render_pass_extrapolation_cost_ms(10.0)
    at_hundred = _render_pass_extrapolation_cost_ms(100.0)
    assert 0.0 < at_two < 5.0
    assert at_ten > 50.0 * at_two
    assert at_hundred > 50.0 * at_ten


def test_render_pass_extrapolation_tightens_as_fit_uncertainty_falls():
    cold = _RenderPassCostModel(
        fixed_ms=50.0,
        item_ms=0.1,
        residual_rms_ms=None,
        samples=2,
        design_points=1,
        observed_item_min=1,
        observed_item_max=1,
        mean_elapsed_ms=50.0,
    )
    informed = _RenderPassCostModel(
        fixed_ms=50.0,
        item_ms=0.1,
        residual_rms_ms=0.0,
        samples=8,
        design_points=8,
        observed_item_min=1,
        observed_item_max=16,
        mean_elapsed_ms=50.0,
    )

    assert _render_pass_extrapolation_weight(cold) < 0.2
    assert _render_pass_extrapolation_weight(informed) == 1.0


def test_repeated_single_cohort_does_not_masquerade_as_exploration():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    structure = ("wgpu", "gpu-delta", (336, 336), 17, 16, 1, 272)
    governor.begin_render_pass(
        "preview",
        pass_kind="preview",
        structural_key=structure,
        representation_key=("float32", "preview"),
    )
    for items, elapsed_ms in ((1, 8.0), (4, 8.4), (8, 8.8)):
        governor.record_ui_observation(
            "montage_render_pass_preview",
            elapsed_ms,
            item_count=items,
            byte_count=items * 256,
            work_class="presentation_upsert",
        )
        governor.decide_render_pass(interactive=False, remaining_items=272)

    governor.begin_render_pass(
        "target",
        pass_kind="target",
        structural_key=structure,
        representation_key=("float32", "target"),
    )
    decisions = []
    weights = []
    for elapsed_ms in (27.0, 25.0, 28.0, 26.0):
        governor.record_ui_observation(
            "montage_render_pass_target",
            elapsed_ms,
            item_count=1,
            byte_count=7056,
            work_class="presentation_upsert",
        )
        decision = governor.decide_render_pass(
            interactive=False,
            pass_kind="target",
            remaining_items=272,
        )
        decisions.append(decision.batch_limit)
        fit = next(detail for detail in decision.details if detail.startswith("fit-rms="))
        assert "design-points=1" in fit
        uncertainty = float(fit.split("uncertainty=", 1)[1].split()[0])
        weights.append(1.0 / (1.0 + uncertainty))

    assert min(decisions) > 1
    assert max(weights) - min(weights) < 1e-12, (
        "repetition may estimate noise but cannot tighten extrapolation without new design points"
    )


def test_render_pass_rejects_one_load_spike_and_tracks_a_sustained_shift():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.begin_render_pass("round-a", pass_kind="target")
    cohort = 1
    for load_ms in (0.0, 0.0, 0.0, 0.0):
        elapsed_ms = 19.0 + cohort + load_ms
        governor.record_ui_observation(
            "montage_render_pass_target",
            elapsed_ms,
            item_count=cohort,
            work_class="presentation_upsert",
        )
        decision = governor.decide_render_pass(
            interactive=False,
            pass_kind="target",
            remaining_items=272,
        )
        cohort = decision.batch_limit
    governor.begin_render_pass("round-b", pass_kind="target")
    stable_cohort = governor.decide_render_pass(
        interactive=False,
        pass_kind="target",
        remaining_items=272,
    ).batch_limit
    cohort = stable_cohort

    governor.record_ui_observation(
        "montage_render_pass_target",
        19.0 + cohort + 80.0,
        item_count=cohort,
        work_class="presentation_upsert",
    )
    isolated = governor.decide_render_pass(
        interactive=False,
        pass_kind="target",
        remaining_items=272,
    )
    assert isolated.batch_limit == stable_cohort
    assert any("load-offset=0.00ms" in detail for detail in isolated.details)

    cohort = isolated.batch_limit
    shifted = isolated
    for _ in range(3):
        governor.record_ui_observation(
            "montage_render_pass_target",
            19.0 + cohort + 80.0,
            item_count=cohort,
            work_class="presentation_upsert",
        )
        shifted = governor.decide_render_pass(
            interactive=False,
            pass_kind="target",
            remaining_items=272,
        )
        cohort = shifted.batch_limit

    assert shifted.batch_limit > stable_cohort
    assert shifted.batch_limit != 1
    assert any(
        "item=1.00ms" in detail and "load-offset=80.00ms" in detail for detail in shifted.details
    )

    for _ in range(2):
        governor.record_ui_observation(
            "montage_render_pass_target",
            19.0 + cohort,
            item_count=cohort,
            work_class="presentation_upsert",
        )
        recovered = governor.decide_render_pass(
            interactive=False,
            pass_kind="target",
            remaining_items=272,
        )
        cohort = recovered.batch_limit

    assert recovered.batch_limit == stable_cohort


def test_render_pass_responsiveness_weight_moves_one_shared_objective():
    model = _RenderPassCostModel(
        fixed_ms=45.0,
        item_ms=0.2,
        residual_rms_ms=0.0,
        samples=8,
        observed_item_min=1,
        observed_item_max=272,
        mean_elapsed_ms=45.2,
    )
    common = {
        "fixed_ms": 45.0,
        "marginal_ms": 0.2,
        "remaining_items": 10_000,
        "observed_item_max": 10_000,
        "observed_byte_max": 0,
        "bytes_per_item": 0.0,
        "model": model,
    }

    responsive = _render_pass_optimal_point(**common, responsiveness_weight=2.0)[0]
    balanced = _render_pass_optimal_point(**common, responsiveness_weight=1.0)[0]
    throughput = _render_pass_optimal_point(**common, responsiveness_weight=0.3)[0]

    assert responsive <= balanced <= throughput


def test_render_pass_requirement_does_not_adapt_downward():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_render_pass_target",
        92.0,
        item_count=1,
        work_class="presentation_upsert",
        backend="wgpu",
    )

    diagnostics = governor.diagnostics(channels=("montage_render_pass_target",))

    assert diagnostics.feedback_channels[0].budget_ms == 32.0
    assert diagnostics.recent_ui_work_observations[-1].target_ms == 32.0


def test_callback_observations_are_kept_without_decision_ring():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    observation = GuiCallbackObservation(
        channel="montage_commit",
        work_class="tile_upsert",
        backend="wgpu",
        target_ms=4.0,
        warning_ms=16.0,
        item_cap=12,
        byte_cap=8 * 1024 * 1024,
        elapsed_ms=18.0,
        processed_items=3,
        processed_bytes=4096,
    )

    governor.record_gui_callback_observation(observation)
    diagnostics = governor.diagnostics()

    assert diagnostics.recent_over_warning_callbacks == (observation,)
    assert diagnostics.recent_ui_work_observations[-1] == observation
    assert not hasattr(diagnostics, "ui_decisions")
    assert not hasattr(diagnostics, "recent_ui_work_decisions")


def test_histogram_workers_park_only_behind_runnable_visible_work():
    governor = ResourceGovernor(
        _policy(), profile=MemoryProfileChoice.BALANCED, min_worker_update_interval_ms=0
    )

    busy = governor.decide_lane_workers(
        ComputeLane.HISTOGRAM,
        interactive=True,
        busy_state=SchedulerBusyState(montage_busy=True, result_backlog=2),
    )
    between_bursts = governor.decide_lane_workers(
        ComputeLane.HISTOGRAM,
        interactive=True,
        busy_state=SchedulerBusyState(),
    )

    assert busy.target_workers == 0
    assert busy.min_workers == 0
    assert "runnable user-visible rendering" in busy.reason
    assert between_bursts.target_workers == between_bursts.max_workers

    bookkeeping_only = governor.decide_lane_workers(
        ComputeLane.HISTOGRAM,
        interactive=False,
        busy_state=SchedulerBusyState(result_backlog=2),
    )
    assert bookkeeping_only.target_workers == bookkeeping_only.max_workers


def test_interaction_immediately_parks_side_lanes_and_limits_montage():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    busy = SchedulerBusyState(montage_busy=True)

    montage = governor.decide_lane_workers(
        ComputeLane.MONTAGE_TILE,
        interactive=True,
        busy_state=busy,
    )
    prefetch = governor.decide_lane_workers(
        ComputeLane.PREFETCH,
        interactive=True,
        busy_state=busy,
    )
    profile = governor.decide_lane_workers(
        ComputeLane.PROFILE,
        interactive=True,
        busy_state=busy,
    )

    assert montage.target_workers == 2
    assert "bounded montage workers" in montage.reason
    assert prefetch.target_workers == 0
    assert profile.target_workers == 0


def test_first_lane_decision_starts_at_policy_target_without_damping_from_maximum():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)

    decision = governor.decide_lane_workers(
        ComputeLane.PREFETCH,
        interactive=True,
        busy_state=SchedulerBusyState(visible_busy=True),
    )

    assert decision.target_workers == 0


def test_blocking_semantic_evidence_keeps_one_histogram_worker():
    governor = ResourceGovernor(
        _policy(), profile=MemoryProfileChoice.BALANCED, min_worker_update_interval_ms=0
    )

    decision = governor.decide_lane_workers(
        ComputeLane.HISTOGRAM,
        interactive=True,
        busy_state=SchedulerBusyState(
            visible_busy=True,
            result_backlog=2,
            semantic_evidence_blocking=True,
        ),
    )

    assert decision.target_workers == 1
    assert decision.min_workers == 1
    assert "blocks first presentation" in decision.reason


def test_tile_worker_product_guard_still_holds():
    policy = _policy(MemoryProfileChoice.AGGRESSIVE)

    assert policy.fft_workers_tile == 1
    assert policy.montage_tile_workers * policy.fft_workers_tile <= 14


def test_enclosing_callback_charge_shrinks_the_cohort_and_is_free_when_absent():
    """R5's ceiling is per GUI callback, so a nested pass starts part-spent.

    Counting form, no clock: the same measured cost model is asked for its
    optimum twice, and the only difference is how much of the enclosing
    callback has already been spent. The zero case must be arithmetically
    identical to not passing the argument at all -- every commit outside an R6
    partial remap takes that branch, and repricing them is exactly the cohort
    collapse R5 warns about.
    """

    model = _RenderPassCostModel(
        fixed_ms=6.0,
        item_ms=0.6,
        residual_rms_ms=0.0,
        samples=8,
        design_points=8,
        observed_item_min=1,
        observed_item_max=60,
        mean_elapsed_ms=24.0,
    )
    common = {
        "fixed_ms": 6.0,
        "marginal_ms": 0.6,
        "remaining_items": 60,
        "observed_item_max": 60,
        "observed_byte_max": 0,
        "bytes_per_item": 0.0,
        "model": model,
    }

    unenclosed = _render_pass_optimal_point(**common)[0]
    declared_zero = _render_pass_optimal_point(**common, callback_elapsed_ms=0.0)[0]
    enclosed = _render_pass_optimal_point(**common, callback_elapsed_ms=38.0)[0]

    assert declared_zero == unenclosed
    assert enclosed < unenclosed


def test_render_pass_decision_reports_the_callback_it_is_nested_in():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)

    decision = governor.decide_render_pass(
        interactive=True,
        pass_kind="preview",
        remaining_items=60,
        callback_elapsed_ms=38.0,
    )

    assert sum("enclosing-callback=38.00ms" in detail for detail in decision.details) == 1


def _per_item_model() -> tuple[_RenderPassCostModel, dict]:
    """A cost model with a real per-item term, measured over its whole span."""

    model = _RenderPassCostModel(
        fixed_ms=6.0,
        item_ms=0.6,
        residual_rms_ms=0.0,
        samples=8,
        design_points=8,
        observed_item_min=1,
        observed_item_max=60,
        mean_elapsed_ms=24.0,
    )
    return model, {
        "fixed_ms": 6.0,
        "marginal_ms": 0.6,
        "remaining_items": 60,
        "observed_item_max": 60,
        "observed_byte_max": 0,
        "bytes_per_item": 0.0,
        "model": model,
    }


def test_frame_aware_knee_prices_a_chunk_against_the_display_period():
    """A frame-critical pass buys frames with cohort size; an idle one does not.

    One measured model priced twice: the fill-optimal curve and one 60 Hz
    refresh period. The frame-critical chunk must land at the period it was
    priced against, while the idle arm keeps the fill-optimal answer byte for
    byte. A 144 Hz frame is refused for this model -- 6.9 ms cannot hold two of
    its 6 ms fixed term -- so a cheaper transaction carries that half.
    """

    _model, common = _per_item_model()

    idle_items, idle_ms, _ = _render_pass_optimal_point(**common)
    at_60_items, at_60_ms, _ = _render_pass_optimal_point(**common, frame_budget_ms=1000.0 / 60.0)

    assert idle_items == _render_pass_optimal_point(**common, frame_budget_ms=0.0)[0]
    assert at_60_items < idle_items
    # The curve is a price, not a cap: a chunk may still overrun its frame when
    # the fill time it buys is worth more than the overrun costs. What must
    # hold is that the chosen chunk lands within half a frame of the period it
    # was priced against, which the fill-optimal answer does not.
    assert abs(at_60_ms - 1000.0 / 60.0) <= 0.5 * 1000.0 / 60.0
    assert idle_ms > 2.0 * 1000.0 / 60.0

    cheap = replace(_model, fixed_ms=1.5, item_ms=0.6, mean_elapsed_ms=19.5)
    cheap_common = {**common, "fixed_ms": 1.5, "model": cheap}
    cheap_idle = _render_pass_optimal_point(**cheap_common)[0]
    cheap_at_144_items, cheap_at_144_ms, _ = _render_pass_optimal_point(
        **cheap_common, frame_budget_ms=1000.0 / 144.0
    )

    assert cheap_at_144_items < cheap_idle
    assert abs(cheap_at_144_ms - 1000.0 / 144.0) <= 0.5 * 1000.0 / 144.0


def test_frame_aware_knee_refuses_to_collapse_a_fixed_dominated_cohort():
    """R5's cohort collapse: what the chunk cannot buy, it must not be charged.

    49.9 ms of fixed cost and 0.1 ms per item means no cohort size fits a
    16.7 ms frame, so pricing every candidate as an overrun would only shrink
    the cohort while every chunk still missed its frame. The frame-aware arm
    must return the same cohort as the fill-optimal one and report the overrun.
    """

    model = _RenderPassCostModel(
        fixed_ms=49.9,
        item_ms=0.1,
        residual_rms_ms=0.0,
        samples=8,
        design_points=8,
        observed_item_min=1,
        observed_item_max=60,
        mean_elapsed_ms=53.0,
    )
    common = {
        "fixed_ms": 49.9,
        "marginal_ms": 0.1,
        "remaining_items": 60,
        "observed_item_max": 60,
        "observed_byte_max": 0,
        "bytes_per_item": 0.0,
        "model": model,
    }

    idle = _render_pass_optimal_point(**common)
    frame_critical = _render_pass_optimal_point(**common, frame_budget_ms=1000.0 / 60.0)

    assert frame_critical[0] == idle[0]
    assert frame_critical[1] > 1000.0 / 60.0


def test_frame_aware_knees_floor_at_the_cheapest_chunk_the_model_can_make():
    assert _render_pass_frame_aware_knees(16.0, 2.0, 4.0) == (8.0, 16.0)
    # The fixed term does not shrink with the chunk: when one item already
    # costs more than a frame, the steep branch starts there instead.
    assert _render_pass_frame_aware_knees(16.0, 2.0, 40.0) == (20.0, 40.0)


def test_a_frame_a_fixed_cost_cannot_fit_is_not_for_sale():
    """Refuse the frame-aware price when the fixed term would multiply.

    Every chunk re-pays the fixed cost, so pacing a pass into frame-sized
    chunks multiplies that term by the chunk count -- R5's cohort collapse
    reached through the responsiveness price. Measured on the cold 272-tile
    WGPU montage (~11 ms fixed against a 16.7 ms frame): the profiler's 5 s
    fill window presented 34 tiles with the frame knee against 266 without it.
    """

    assert _render_pass_frame_aware_knees(16.67, 11.0, 12.5) is None
    assert _render_pass_frame_aware_knees(16.67, 4.0, 5.0) == (8.335, 16.67)

    model = _RenderPassCostModel(
        fixed_ms=11.0,
        item_ms=1.5,
        residual_rms_ms=0.0,
        samples=8,
        design_points=8,
        observed_item_min=1,
        observed_item_max=60,
        mean_elapsed_ms=30.0,
    )
    common = {
        "fixed_ms": 11.0,
        "marginal_ms": 1.5,
        "remaining_items": 272,
        "observed_item_max": 60,
        "observed_byte_max": 0,
        "bytes_per_item": 0.0,
        "model": model,
    }

    assert _render_pass_optimal_point(
        **common, frame_budget_ms=1000.0 / 60.0
    ) == _render_pass_optimal_point(**common)


def test_the_frame_a_60hz_pass_can_buy_costs_half_of_it():
    """Where the refusal turns over, in the terms the commit is measured in.

    The rule above is stated as an inequality; this is the number it puts on
    the render pass, so a change to the commit's fixed cost can be read against
    it without re-deriving the arithmetic. At 60 Hz the pass gets a frame-sized
    price exactly while its fixed term is under **half a frame, 8.33 ms**.

    The two arguments below are measured, not chosen (272-tile scalar WGPU
    montage, `render_pass_governor_probe`, 2026-08-02): the backend commit's
    own share of the callback is 4.8 ms and would be paced, while the whole
    render-pass callback fits at 10.5 ms and is not. What stands between them
    is the presentation work the callback does around the commit -- the payload
    build, the presentation decision and the state publish -- and until that
    comes down the knee stays refused however cheap the commit gets.
    """

    frame_ms = 1000.0 / 60.0

    assert _render_pass_frame_aware_knees(frame_ms, 0.5 * frame_ms, 1.0) is not None
    assert _render_pass_frame_aware_knees(frame_ms, 0.5 * frame_ms + 1e-9, 1.0) is None

    assert _render_pass_frame_aware_knees(frame_ms, 10.5, 12.0) is None
    assert _render_pass_frame_aware_knees(frame_ms, 4.8, 6.0) == (0.5 * frame_ms, frame_ms)


def test_frame_aware_knee_moves_the_enclosing_charge_instead_of_adding_one():
    """The two prices share one curve, so the same milliseconds pay once.

    The enclosing-callback charge is a DIFFERENCE on the responsiveness curve.
    Shifting the knee must move that difference, never stack a second price on
    top of it: a pass enclosed by nothing pays nothing under either knee, and
    the enclosed cohort stays no larger than the unenclosed one.
    """

    _model, common = _per_item_model()
    frame = {"frame_budget_ms": 1000.0 / 60.0}

    unenclosed = _render_pass_optimal_point(**common, **frame)
    declared_zero = _render_pass_optimal_point(**common, **frame, callback_elapsed_ms=0.0)
    enclosed = _render_pass_optimal_point(**common, **frame, callback_elapsed_ms=38.0)

    assert declared_zero == unenclosed
    assert enclosed[0] <= unenclosed[0]
    # And the enclosing charge itself reads the shifted curve: 38 ms of spent
    # callback is already past a 60 Hz frame, so it must cost more there than
    # on the fill-optimal curve, which is free until 18 ms.
    at_frame_knee = _render_pass_latency_cost_ms(38.0, knee_ms=8.335, steep_ms=16.67)
    assert at_frame_knee > _render_pass_latency_cost_ms(38.0)


def test_render_pass_decision_reports_the_frame_it_was_priced_against():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)

    idle = governor.decide_render_pass(interactive=False, pass_kind="target", remaining_items=60)
    paced = governor.decide_render_pass(
        interactive=False,
        pass_kind="target",
        remaining_items=60,
        frame_critical=True,
        refresh_hz=144.0,
    )

    assert sum("frame-critical=0 refresh=60.0Hz frame-budget=0.00ms" in row for row in idle.details)
    assert (
        sum("frame-critical=1 refresh=144.0Hz frame-budget=6.94ms" in row for row in paced.details)
        == 1
    )
    # A refresh rate the platform will not report is the governor's own
    # default, never a division by zero.
    unknown = governor.decide_render_pass(
        interactive=False,
        pass_kind="target",
        remaining_items=60,
        frame_critical=True,
        refresh_hz=0.0,
    )
    assert any("refresh=60.0Hz frame-budget=16.67ms" in row for row in unknown.details)


def _drain_observation(index: int) -> GuiCallbackObservation:
    """One presentation-owner drain observation, submitted off the GUI thread."""

    return GuiCallbackObservation(
        channel="presentation_owner_drain",
        work_class="completion_body",
        backend="wgpu",
        target_ms=8.0,
        warning_ms=16.0,
        item_cap=4,
        byte_cap=0,
        elapsed_ms=1.0 + float(index % 5),
        processed_items=1 + index % 3,
        processed_bytes=0,
    )


def test_owner_and_gui_families_run_concurrently_without_losing_observations():
    """ADR 0065 stage 1: two threads call this object, and both keep their work.

    The owner thread records its own drain observations and drives the
    render-pass family while the GUI thread keeps lanes, pressure, telemetry,
    and the diagnostics read.  Those families walk the same deques and dicts --
    ``decide_render_pass`` scans the observation deque the drain thread is
    appending to, and ``diagnostics`` iterates the lane decisions the lane
    thread is writing -- so the failure this pins is unsynchronized structural
    corruption, not slowness.  Bounded by iteration count; nothing here reads a
    clock.

    The switch interval is the lever that makes this a detector rather than a
    coincidence: at the 5 ms default the interpreter almost never changes
    thread inside one of these walks, and the unguarded governor survived every
    trial.  At 1 us it raised ``deque mutated during iteration`` in 3 of 3
    trials of the same loop.  It is restored either way -- it is process-wide.
    """

    governor = ResourceGovernor(
        _policy(), profile=MemoryProfileChoice.BALANCED, min_worker_update_interval_ms=0
    )
    governor.update_telemetry(_snapshot(_memory()), _memory())
    epoch = governor.begin_ui_observation_epoch()

    drain_iterations = 400
    decision_iterations = 200
    drain_threads = 2
    failures: list[BaseException] = []
    # The deque holds 4096; the submitted total stays well inside it so the
    # count below is a closed equation rather than a bound.
    assert drain_threads * drain_iterations < 4096

    def drain() -> None:
        for index in range(drain_iterations):
            governor.record_gui_callback_observation(_drain_observation(index))

    def render_pass(pass_kind: str):
        def body() -> None:
            for index in range(decision_iterations):
                # One round, many decisions -- the real shape, and the one that
                # exposes the deque: a decision scans the observation ring back
                # to the marker its round began at, so every drain recorded
                # since then is walked while the drain threads keep appending.
                if index % 8 == 0:
                    governor.begin_render_pass((pass_kind, index // 8), pass_kind=pass_kind)
                governor.decide_render_pass(
                    interactive=index % 2 == 0,
                    pass_kind=pass_kind,
                    remaining_items=64,
                )

        return body

    def gui_family() -> None:
        for index in range(decision_iterations):
            governor.decide_lane_workers(
                ComputeLane.MONTAGE_TILE,
                interactive=index % 2 == 0,
                busy_state=SchedulerBusyState(montage_busy=True, result_backlog=index % 8),
            )
            governor.decide_bridge_drain(interactive=False)
            governor.decide_commit_batch(interactive=True)
            governor.update_telemetry(_snapshot(_memory()), _memory())

    def read_diagnostics() -> None:
        for _ in range(decision_iterations):
            snapshot = governor.diagnostics()
            assert len(snapshot.recent_ui_work_observations) <= 64

    bodies = [
        drain,
        drain,
        render_pass("preview"),
        render_pass("target"),
        gui_family,
        read_diagnostics,
    ]
    start = threading.Barrier(len(bodies), timeout=60.0)

    def guarded(body):
        def run() -> None:
            try:
                start.wait()
                body()
            except BaseException as error:  # reported from the main thread
                failures.append(error)

        return run

    threads = [
        threading.Thread(target=guarded(body), name=f"governor-hammer-{index}", daemon=True)
        for index, body in enumerate(bodies)
    ]
    previous_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120.0)
    finally:
        sys.setswitchinterval(previous_switch_interval)

    assert [thread.name for thread in threads if thread.is_alive()] == []
    assert failures == []
    observations = tuple(governor._recent_ui_work_observations)
    assert len(observations) == drain_threads * drain_iterations
    assert {observation.channel for observation in observations} == {"presentation_owner_drain"}
    count, _, epoch_is_current = governor.ui_observation_epoch_evidence(epoch)
    assert epoch_is_current
    assert count == drain_threads * drain_iterations
    assert governor.ui_observation_epoch_max(epoch).channel == "presentation_owner_drain"
    assert ComputeLane.MONTAGE_TILE in {
        decision.lane for decision in governor.diagnostics().lane_decisions
    }


def test_one_caller_section_may_span_several_governor_entry_points():
    """The governor's lock is re-entrant, which is what makes a section legal.

    ADR 0065's migration bridge serializes owner pipeline sections against GUI
    session sections, so a caller holds a section across several entry points --
    each of which acquires the same lock again.  A non-re-entrant lock does not
    fail that with an exception, it deadlocks, so the section runs on a worker
    joined with a bounded wait and the verdict is whether it finished at all.
    """

    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.update_telemetry(_snapshot(_memory()), _memory())
    results: dict[str, object] = {}
    failures: list[BaseException] = []

    def section() -> None:
        try:
            with governor._lock:
                governor.begin_render_pass(("owner-section",))
                governor.record_gui_callback_observation(_drain_observation(0))
                with governor._lock:
                    results["decision"] = governor.decide_render_pass(
                        interactive=False, remaining_items=32
                    )
                results["diagnostics"] = governor.diagnostics()
        except BaseException as error:  # reported from the main thread
            failures.append(error)

    # Daemon: a lock that is not re-entrant deadlocks this worker, and a
    # deadlocked non-daemon thread would hold the whole suite at interpreter
    # exit instead of failing the one test that asked the question.
    worker = threading.Thread(target=section, name="governor-reentrant-section", daemon=True)
    worker.start()
    worker.join(timeout=60.0)

    assert not worker.is_alive()
    assert failures == []
    assert results["decision"].channel == "montage_render_pass_preview"
    assert results["diagnostics"].recent_ui_work_observations[-1].channel == (
        "presentation_owner_drain"
    )
