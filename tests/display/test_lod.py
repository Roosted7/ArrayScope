from arrayscope.display.lod import (
    LOD_POLICY_NATIVE_ONLY,
    LOD_REASON_ASYNC_RESIDENCY_REQUIRED,
    LOD_REASON_INVALID_VIEW,
    inner_uv_for_gutter,
    native_lod_policy,
    select_lod_demand,
)


def test_lod_demand_keeps_native_when_zoomed_in():
    demand = select_lod_demand(((0.0, 64.0), (0.0, 64.0)), (128, 128), (64, 64))

    assert demand.desired_factor == 1
    assert demand.desired_factor_xy == (1, 1)
    assert demand.source_texels_per_pixel_xy == (0.5, 0.5)
    assert demand.acceptable_levels == (0, 1)


def test_lod_demand_records_per_axis_texels_for_zoomed_out_view():
    demand = select_lod_demand(((0.0, 1024.0), (0.0, 512.0)), (128, 256), (64, 64))

    assert demand.desired_factor == 4
    assert demand.desired_factor_xy == (4, 4)
    assert demand.source_texels_per_pixel_xy == (4.0, 4.0)
    assert demand.desired_level == 2
    assert demand.acceptable_levels == (1, 2, 3)
    assert "zoomed-out" in demand.reason


def test_lod_demand_hysteresis_is_independent_of_materialization():
    previous = select_lod_demand(
        ((0.0, 256.0), (0.0, 256.0)),
        (128, 128),
        (64, 64),
        previous_factor=1,
    )
    stable = select_lod_demand(
        ((0.0, 260.0), (0.0, 260.0)),
        (128, 128),
        (64, 64),
        previous_factor=previous.desired_factor,
    )
    promoted = select_lod_demand(
        ((0.0, 320.0), (0.0, 320.0)),
        (128, 128),
        (64, 64),
        previous_factor=previous.desired_factor,
    )

    assert previous.desired_factor == 1
    assert stable.desired_factor == 1
    assert stable.desired_factor_xy == (1, 1)
    assert promoted.desired_factor == 2
    assert promoted.desired_factor_xy == (2, 2)


def test_lod_demand_factor_xy_matches_final_hysteresis_factor():
    held_high = select_lod_demand(
        ((0.0, 300.0), (0.0, 300.0)),
        (128, 128),
        (64, 64),
        previous_factor=4,
    )
    held_low = select_lod_demand(
        ((0.0, 260.0), (0.0, 260.0)),
        (128, 128),
        (64, 64),
        previous_factor=1,
    )

    assert max(held_high.desired_factor_xy) == held_high.desired_factor
    assert max(held_low.desired_factor_xy) == held_low.desired_factor


def test_lod_demand_invalid_view_falls_back_to_native():
    demand = select_lod_demand(None, (128, 128), (64, 64))

    assert demand.desired_factor == 1
    assert demand.desired_factor_xy == (1, 1)
    assert demand.source_texels_per_pixel_xy == (0.0, 0.0)
    assert demand.reason == LOD_REASON_INVALID_VIEW


def test_native_policy_preserves_invalid_view_reason():
    decision = native_lod_policy(None, (128, 128), (64, 64))

    assert decision.demand.reason == LOD_REASON_INVALID_VIEW
    assert decision.applied_factor == 1
    assert decision.policy == LOD_POLICY_NATIVE_ONLY
    assert decision.reason == LOD_REASON_INVALID_VIEW


def test_lod_demand_records_anisotropic_extreme_aspect_case():
    demand = select_lod_demand(((0.0, 2048.0), (0.0, 64.0)), (128, 128), (64, 64))

    assert demand.desired_factor > 1
    assert demand.desired_factor_xy[0] > demand.desired_factor_xy[1]
    assert demand.source_texels_per_pixel_xy == (16.0, 0.5)
    assert "anisotropic" in demand.reason


def test_native_policy_reports_desired_and_applied_separately():
    decision = native_lod_policy(((0.0, 1024.0), (0.0, 1024.0)), (128, 128), (64, 64))

    assert decision.demand.desired_factor > 1
    assert decision.applied_factor == 1
    assert decision.applied_factor_xy == (1, 1)
    assert decision.applied_level == 0
    assert decision.policy == LOD_POLICY_NATIVE_ONLY
    assert decision.reason == LOD_REASON_ASYNC_RESIDENCY_REQUIRED


def test_inner_uv_for_gutter_excludes_border_pixels():
    assert inner_uv_for_gutter((4, 6), gutter=1) == (1 / 6, 1 / 4, 5 / 6, 3 / 4)
