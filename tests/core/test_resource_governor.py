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
    _render_pass_latency_cost_ms,
    _render_pass_optimal_point,
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
