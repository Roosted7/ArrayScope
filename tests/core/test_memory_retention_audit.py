from arrayscope.core.memory_policy import GiB, SystemMemorySnapshot
from arrayscope.tools.memory_retention_audit import run_audit


def _system() -> SystemMemorySnapshot:
    return SystemMemorySnapshot(
        total_bytes=64 * GiB,
        available_bytes=40 * GiB,
        process_rss_bytes=1 * GiB,
        source="test",
    )


def test_audit_separates_display_roi_and_gpu_page_owners():
    result = run_audit(
        input_bytes=4 * GiB,
        profile="balanced",
        render_cap_mb=512,
        source_mode="lazy",
        system=_system(),
    )
    owners = {owner["name"]: owner for owner in result["owners"]}

    assert owners["display_materialization"]["budget_bytes"] == 4 * GiB
    assert owners["roi_region_demand"]["budget_bytes"] == 4 * GiB
    assert owners["wgpu_gpu_page_residency"]["address_space"] == "device"
    assert owners["stage"]["budget_bytes"] == 8 * GiB
    assert owners["retained_payload_refs"]["budget_bytes"] <= 512 * 1024**2
    assert result["configured_device_upper_bound_bytes"] == 512 * 1024**2


def test_pyqtgraph_raster_residency_counts_as_host_memory():
    common = {
        "input_bytes": 4 * GiB,
        "profile": "balanced",
        "render_cap_mb": 512,
        "source_mode": "lazy",
        "system": _system(),
    }
    gpu = run_audit(backend="wgpu", **common)
    cpu = run_audit(backend="pyqtgraph", **common)

    assert (
        cpu["configured_retained_upper_bound_bytes"] - gpu["configured_retained_upper_bound_bytes"]
        == 512 * 1024**2
    )
    assert cpu["configured_device_upper_bound_bytes"] == 0


def test_eager_source_adds_logical_input_to_retention_envelope():
    common = {
        "input_bytes": 4 * GiB,
        "profile": "balanced",
        "render_cap_mb": 512,
        "system": _system(),
    }
    lazy = run_audit(source_mode="lazy", **common)
    eager = run_audit(source_mode="eager", **common)

    assert (
        eager["configured_retained_upper_bound_bytes"]
        - lazy["configured_retained_upper_bound_bytes"]
        == 4 * GiB
    )
