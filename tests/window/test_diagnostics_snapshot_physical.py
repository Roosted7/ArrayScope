from types import SimpleNamespace

from arrayscope.window.diagnostics_snapshot import _presentation_diagnostics, _tile_identity_probe


def test_tile_identity_probe_merges_backend_draw_geometry():
    session = SimpleNamespace(
        diagnostic_tile_identity_rows=lambda **_kwargs: (
            {"tile": 4, "target_lod": {"level": 2}},
            {"tile": 5, "target_lod": {"level": 1}},
        )
    )
    physical = {
        4: {
            "physical_draw_world_bounds": (10.0, 20.0, 30.0, 40.0),
            "physical_expected_world_rect": (10.0, 20.0, 30.0, 40.0),
            "physical_draw_bounds_match_layout": True,
        }
    }
    window = SimpleNamespace(img_view=SimpleNamespace(tileTruthPhysicalRows=lambda: physical))

    rows = _tile_identity_probe(window, session)

    assert rows[0]["target_lod"] == {"level": 2}
    assert rows[0]["physical_draw_world_bounds"] == (10.0, 20.0, 30.0, 40.0)
    assert rows[0]["physical_draw_bounds_match_layout"] is True
    assert "physical_draw_world_bounds" not in rows[1]


def test_presentation_diagnostics_uses_shared_backend_contract_for_wgpu():
    window = SimpleNamespace(
        img_view=SimpleNamespace(
            presentation_diagnostics=lambda: {
                "wgpu_page_pools": ({"representation": "scalar_r32f"},),
                "wgpu_last_pool_exhaustion": "pool exhausted",
            },
        )
    )

    diagnostics = _presentation_diagnostics(window)

    assert diagnostics["wgpu_page_pools"] == ({"representation": "scalar_r32f"},)
    assert diagnostics["wgpu_last_pool_exhaustion"] == "pool exhausted"
