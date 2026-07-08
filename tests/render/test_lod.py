from arrayscope.display.lod import LOD_POLICY_NATIVE_ONLY, LOD_POLICY_RESIDENT, select_lod_demand
from arrayscope.render.lod import native_missing_tile_queue_required


def test_native_missing_tile_queue_is_only_required_for_native_demand():
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
    assert not native_missing_tile_queue_required(LOD_POLICY_RESIDENT, coarse_demand)
    assert native_missing_tile_queue_required(LOD_POLICY_RESIDENT, native_demand)
    assert native_missing_tile_queue_required(LOD_POLICY_NATIVE_ONLY, coarse_demand)
