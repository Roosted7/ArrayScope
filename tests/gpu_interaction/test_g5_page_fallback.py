"""Real-Wayland G5 page-resolution acceptance gate (ADR 0056 / G5 §9).

This is deliberately a ring-4 test.  It presents canonical source-grid pages
through the real VisPy surface and samples the GL framebuffer after every
binding transition; offscreen metadata-only coverage cannot replace it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from arrayscope.tools.interaction_budget import (
    INTERACTION_SETTLE_HARD_LIMIT_MS,
    INTERACTION_SETTLE_HARD_LIMIT_S,
    bounded_interaction_settle_timeout_s,
    interaction_settle_timeout_ms,
)


pytestmark = pytest.mark.gpu_interaction


def _canonical_payloads():
    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.frame import DisplayTilePayload, PageBackedPresentation
    from arrayscope.display.pyramid import materialize_lod_page, plan_source_grid_pages

    yy, xx = np.mgrid[:64, :64]
    source = np.ascontiguousarray(
        0.2
        + 0.5 * (xx + yy) / 126.0
        + 0.1 * ((xx // 8 + yy // 8) % 2),
        dtype=np.float32,
    )
    route = {
        "content_key": ("g5-real-gl-never-black", ("document", 1), ("operation", "identity")),
        "valid_source_rect_yx": (0, 64, 0, 64),
        "stored_page_shape": (32, 32),
        "dtype": "float32",
        "representation": "scalar_r32f",
        "reducer": "mean",
    }
    fine_plan = plan_source_grid_pages(reduction_yx=(1, 1), **route)[0]
    coarse_plan = plan_source_grid_pages(reduction_yx=(2, 2), **route)[0]
    fine = materialize_lod_page(source, source_origin_yx=(0, 0), plan=fine_plan)
    coarse = materialize_lod_page(source, source_origin_yx=(0, 0), plan=coarse_plan)
    requested_lod = LodInfo(
        level=1,
        factor=2,
        source_shape=source.shape,
        texture_shape=fine_plan.stored_shape,
        gutter=0,
    )

    def payload(page, *, quality: str):
        return DisplayTilePayload(
            tile_number=0,
            source_index=0,
            image=page.values,
            histogram_data=None,
            source_id=(
                "g5-real-gl-never-black",
                "requested-l1",
                ("actual-page", page.key),
            ),
            texture_data=page.values,
            semantic_data=None,
            lod=requested_lod,
            quality=quality,
            page_backing=PageBackedPresentation(
                requested_plans=(fine_plan,),
                materialized_pages=(page,),
                source_coverage_yx=(0, 64, 0, 64),
                requested_lod=requested_lod,
            ),
        )

    return source, fine_plan, fine, coarse, payload(coarse, quality="preview"), payload(
        fine,
        quality="exact",
    )


def _geometry():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState

    return DisplayGeometry(
        view_state=ViewState.from_shape((64, 64, 1)).with_montage_axis(
            2,
            columns=1,
            indices=(0,),
            text=":",
        ),
        display_shape=(64, 64),
        montage=MontageGeometry(
            indices=(0,),
            tile_shape=(64, 64),
            columns=1,
            rows=1,
            gap=0,
        ),
        montage_tile_states=(MontageTileState.LOADED,),
    )


def _delta(payload, *, base: int, target: int):
    from arrayscope.display.model.frame import TilePresentationDelta

    return TilePresentationDelta(
        structure_revision=1,
        payload_revision=target,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        base_revision=base,
        target_revision=target,
        transaction_generation=1,
        upserts={0: payload},
        active_tiles=(0,),
        planned_tiles=(0,),
    )


def _save_frame(path: Path, frame: np.ndarray) -> None:
    from pyqtgraph.Qt import QtGui

    values = np.ascontiguousarray(frame, dtype=np.uint8)
    if values.shape[-1] == 4:
        fmt = QtGui.QImage.Format.Format_RGBA8888
    elif values.shape[-1] == 3:
        fmt = QtGui.QImage.Format.Format_RGB888
    else:
        raise AssertionError(f"unexpected framebuffer shape {values.shape}")
    image = QtGui.QImage(
        values.data,
        int(values.shape[1]),
        int(values.shape[0]),
        int(values.strides[0]),
        fmt,
    ).copy()
    assert image.save(str(path)), f"could not save framebuffer artifact {path}"


def _frame_has_tile_signal(frame: np.ndarray) -> bool:
    """Signal oracle whose actual-GL fault audit is exercised in the test."""

    rgb = np.asarray(frame)[..., :3]
    active = np.max(rgb, axis=-1) >= 16
    return bool(np.count_nonzero(active) >= max(64, int(active.size * 0.08)))


def _binding(layer):
    rows = layer.tile_truth_physical_rows()
    assert set(rows) == {0}, rows
    bindings = tuple(rows[0].get("physical_page_bindings", ()))
    assert len(bindings) == 1, rows[0]
    return bindings[0]


def test_g5_pinned_coarse_fine_arrival_and_eviction_never_black_real_gl(
    qt_app,
    qtbot,
    tmp_path,
):
    """G5 §9: resolution changes bindings, never pixels-to-black or uploads.

    Sequence: missing fine -> pinned coarse -> resident fine -> forced fine
    removal -> pinned coarse.  Every accepted state is read from the real GL
    framebuffer.  Hiding the actual tile visual proves that the pixel oracle
    rejects a non-vacuous black-frame fault.
    """

    if os.environ.get("QT_QPA_PLATFORM") != "wayland":
        pytest.skip("dedicated G5 framebuffer gate requires real Wayland")

    from arrayscope.app.settings_state import (
        AppSettingsState,
        ImageRenderingBackendChoice,
    )
    from arrayscope.display.image_view_factory import create_image_view
    from arrayscope.display.model.frame import TilePresentationState
    from arrayscope.display.viewport import ViewportPolicy

    source, fine_plan, fine, coarse, coarse_payload, fine_payload = (
        _canonical_payloads()
    )
    geometry = _geometry()
    view = create_image_view(
        AppSettingsState(
            image_rendering_backend=ImageRenderingBackendChoice.VISPY,
        )
    )
    qtbot.addWidget(view)
    view.resize(640, 560)
    view.show()
    qtbot.waitExposed(
        view,
        timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
    )
    assert type(view).__name__ == "VisPySurface"
    layer = view._vispy_gpu_montage_layer
    pool = layer._pool
    frames: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    def step(name, action, *, require_draw: bool):
        before_draw = int(getattr(view, "_vispy_tile_presentation_draw_count", 0))
        start = perf_counter()
        result = action()
        if require_draw:
            timeout = bounded_interaction_settle_timeout_s(
                INTERACTION_SETTLE_HARD_LIMIT_S
            )
            qtbot.waitUntil(
                lambda: (
                    not bool(view.presentationDrawPending())
                    and int(
                        getattr(view, "_vispy_tile_presentation_draw_count", 0)
                    )
                    > before_draw
                ),
                timeout=interaction_settle_timeout_ms(timeout),
            )
        elapsed = perf_counter() - start
        timings[name] = elapsed
        assert elapsed <= INTERACTION_SETTLE_HARD_LIMIT_S, (
            f"{name} took {elapsed:.3f}s > "
            f"{INTERACTION_SETTLE_HARD_LIMIT_S:.3f}s hard interaction limit"
        )
        return result

    def present(payload, *, base: int, target: int, fit: bool = False):
        delta = _delta(payload, base=base, target=target)
        return view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState({0: payload}),
            tile_delta=delta,
            histogramPlotData=source,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            viewport_policy=(
                ViewportPolicy.FIT_ONCE if fit else ViewportPolicy.PRESERVE
            ),
            rgb_already_windowed=False,
            tile_residency_budget_bytes=8 * 1024 * 1024,
        )

    try:
        coarse_report = step(
            "missing-fine-to-coarse",
            lambda: present(coarse_payload, base=0, target=1, fit=True),
            require_draw=True,
        )
        assert coarse_report.texture_uploads > 0
        coarse_binding = _binding(layer)
        assert coarse_binding["target_key"] == fine_plan.key
        assert coarse_binding["actual_key"] == coarse.key
        assert coarse_binding["quality"] == "fallback"
        coarse_generation = int(coarse_binding["binding_generation"])
        assert coarse_generation > 0
        coarse_owner = ("g5-real-gl-retained-coarse", 0)
        pool._page_table.replace_pin_set(coarse_owner, (coarse.key,))
        assert pool._page_table.is_pinned(coarse.key)

        frames["coarse"] = np.asarray(view._vispy_canvas.render()).copy()
        assert _frame_has_tile_signal(frames["coarse"])

        # Non-vacuity audit: break the real physical draw, observe black, then
        # restore it.  This proves the framebuffer oracle is capable of
        # catching the exact transition the gate forbids.
        visible = tuple(
            visual
            for visual in layer._visuals_by_page
            if bool(getattr(visual, "visible", False))
        )
        assert visible, "fault injection would be vacuous without a visible GL page"
        for visual in visible:
            visual.visible = False
        frames["fault-black"] = np.asarray(view._vispy_canvas.render()).copy()
        assert not _frame_has_tile_signal(frames["fault-black"]), (
            "framebuffer oracle failed to reject an injected black tile surface"
        )
        for visual in visible:
            visual.visible = True
        frames["fault-restored"] = np.asarray(view._vispy_canvas.render()).copy()
        assert _frame_has_tile_signal(frames["fault-restored"])

        warm_stats = step(
            "fine-arrival-upload",
            lambda: layer.warm_residency(
                payloads={0: fine_payload},
                geometry=geometry,
                rgb_already_windowed=False,
                tile_delta=_delta(fine_payload, base=1, target=2),
                tile_residency_budget_bytes=8 * 1024 * 1024,
            ),
            require_draw=False,
        )
        assert warm_stats.texture_uploads > 0
        fine_report = step(
            "fine-resolution-rebind",
            lambda: present(fine_payload, base=1, target=2),
            require_draw=True,
        )
        assert fine_report.texture_uploads == 0
        fine_binding = _binding(layer)
        assert fine_binding["target_key"] == fine_plan.key
        assert fine_binding["actual_key"] == fine.key
        assert fine_binding["quality"] == "exact"
        fine_generation = int(fine_binding["binding_generation"])
        assert fine_generation > coarse_generation
        assert pool._page_table.is_pinned(coarse.key)
        frames["fine"] = np.asarray(view._vispy_canvas.render()).copy()
        assert _frame_has_tile_signal(frames["fine"])

        # Simulate a physical-pressure transaction.  Release the tile's fine
        # pin and fine slot, then bind the already-pinned coarse ancestor in
        # the same GUI turn: the compositor never receives an intermediate
        # black presentation.
        def evict_fine_and_rebind_coarse():
            tile_owner = pool._tile_page_pin_owners[0]
            pool._page_table.replace_pin_set(tile_owner, ())
            slot = pool._page_table.lookup(fine.key)
            assert slot is not None
            page = pool.pages[int(slot.page_index)]
            page.slot_owners[int(slot.slot_index)] = None
            page._free_slots.append(int(slot.slot_index))
            pool._release_victim(fine.key, near_keys=set())
            assert pool._page_table.lookup(fine.key) is None
            return present(coarse_payload, base=2, target=3)

        fallback_report = step(
            "fine-eviction-to-pinned-coarse",
            evict_fine_and_rebind_coarse,
            require_draw=True,
        )
        assert fallback_report.texture_uploads == 0
        fallback_binding = _binding(layer)
        assert fallback_binding["target_key"] == fine_plan.key
        assert fallback_binding["actual_key"] == coarse.key
        assert fallback_binding["quality"] == "fallback"
        assert int(fallback_binding["binding_generation"]) == coarse_generation
        assert pool._page_table.is_pinned(coarse.key)
        frames["fallback-after-eviction"] = np.asarray(
            view._vispy_canvas.render()
        ).copy()
        assert _frame_has_tile_signal(frames["fallback-after-eviction"])

        for name, frame in frames.items():
            _save_frame(tmp_path / f"{name}.png", frame)
        evidence = {
            "timings_s": timings,
            "hard_limit_s": INTERACTION_SETTLE_HARD_LIMIT_S,
            "coarse_generation": coarse_generation,
            "fine_generation": fine_generation,
            "fallback_generation": int(fallback_binding["binding_generation"]),
            "fine_resolution_texture_uploads": fine_report.texture_uploads,
            "eviction_resolution_texture_uploads": fallback_report.texture_uploads,
            "frames": sorted(f"{name}.png" for name in frames),
        }
        (tmp_path / "evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    finally:
        view.close()
