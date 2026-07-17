from arrayscope.display.lod import LOD_POLICY_NATIVE_ONLY, LOD_POLICY_RESIDENT, select_lod_demand
from arrayscope.render.lod import missing_tiles_require_native_target


def test_missing_tiles_require_native_target_only_for_native_demand():
    coarse_demand = select_lod_demand(
        ((0.0, 1024.0), (0.0, 1024.0)),
        (256, 256),
        (64, 64),
    )
    native_demand = select_lod_demand(
        ((0.0, 64.0), (0.0, 64.0)),
        (256, 256),
        (64, 64),
    )

    assert coarse_demand.desired_level > 0
    assert not missing_tiles_require_native_target(LOD_POLICY_RESIDENT, coarse_demand)
    assert missing_tiles_require_native_target(LOD_POLICY_RESIDENT, native_demand)
    assert missing_tiles_require_native_target(LOD_POLICY_NATIVE_ONLY, coarse_demand)
