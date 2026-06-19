import ast
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[2]


def test_managed_docks_do_not_use_qt_toggle_view_action():
    offenders = []
    for path in (ROOT / "arrayscope").rglob("*.py"):
        if "layout_controller.py" in str(path):
            continue
        text = path.read_text()
        if "toggleViewAction" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_managed_dock_visibility_only_in_layout_controller_and_dock_chrome():
    managed_names = {"profile_dock", "operation_dock", "inspection_dock"}
    forbidden = {"show", "hide", "setVisible", "close", "setFloating"}
    allowed = {
        Path("arrayscope/window/layout_controller.py"),
    }
    offenders = []
    for path in (ROOT / "arrayscope").rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel in allowed:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr not in forbidden:
                continue
            value = func.value
            if (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "self"
                and value.attr in managed_names
            ):
                offenders.append(f"{rel}:{node.lineno}:{value.attr}.{func.attr}")
    assert offenders == []


def test_square_fov_is_not_visible_production_ui():
    offenders = []
    for path in (ROOT / "arrayscope").rglob("*.py"):
        rel = path.relative_to(ROOT)
        text = path.read_text()
        if "Square FOV" in text or "square_fov" in text:
            offenders.append(str(rel))
    assert offenders == []


def test_visible_render_paths_do_not_compare_partial_document_keys():
    text = "\n".join(
        (ROOT / rel).read_text()
        for rel in (
            Path("arrayscope/window/render.py"),
            Path("arrayscope/window/normal_renderer.py"),
            Path("arrayscope/window/render_prefetch.py"),
        )
    )
    assert ".image_key(" in text
    assert "image_key(view_state, colormap_lut=colormap_lut)[1]" not in text
    assert "line_key(profile_state)[1]" not in text
    assert "scalar_key(view_state, index)[1]" not in text


def test_layout_controller_has_no_dock_event_filter_repair_machinery():
    text = (ROOT / "arrayscope" / "window" / "layout_controller.py").read_text()
    forbidden = (
        "_ManagedDockEventFilter",
        "_visible_snapshots",
        "_schedule_snapshot_restore",
        "_prepare_direct_dock_close",
    )
    for token in forbidden:
        assert token not in text


def test_standard_dock_widget_has_no_close_event_lifecycle_override():
    text = (ROOT / "arrayscope" / "ui" / "docks" / "common.py").read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "StandardDockWidget":
            assert all(not isinstance(child, ast.FunctionDef) or child.name != "closeEvent" for child in node.body)
            return
    raise AssertionError("StandardDockWidget class not found")


def test_detached_dialog_hide_takes_body_before_state_change():
    text = (ROOT / "arrayscope" / "window" / "panels.py").read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_hide_detached_from_dialog":
            segment = ast.get_source_segment(text, node) or ""
            assert "take_body" in segment or "_destroy_dialog_and_take_body" in segment
            return
    raise AssertionError("_hide_detached_from_dialog not found")


def test_managed_panel_code_does_not_use_native_set_floating():
    text = (ROOT / "arrayscope" / "window" / "panels.py").read_text()
    assert ".setFloating(" not in text


def test_layout_controller_preserves_canvas_without_set_geometry_or_clamping():
    text = (ROOT / "arrayscope" / "window" / "layout_controller.py").read_text()
    preserve_text = (ROOT / "arrayscope" / "window" / "canvas_preserve.py").read_text()
    assert ".setGeometry(" not in text
    assert "_clamp_to_available_screen" not in text
    assert ".resize(" in preserve_text
    assert "run_panel_transition_preserving_canvas" in text
    assert "CanvasPreserveController" in text


def test_canvas_preserve_controller_owns_strong_preserve_path():
    layout_text = (ROOT / "arrayscope" / "window" / "layout_controller.py").read_text()
    preserve_text = (ROOT / "arrayscope" / "window" / "canvas_preserve.py").read_text()
    assert "[ArrayScope preserve-canvas]" not in layout_text
    assert "print(" not in layout_text
    assert "_correct_canvas_size" not in layout_text
    assert "_apply_strong_preserve_constraints" not in layout_text
    assert "_release_strong_preserve_constraints" not in layout_text
    assert "CanvasPreserveController" in preserve_text
    assert "_correct_canvas_size" in preserve_text
    assert "_apply_strong_preserve_constraints" in preserve_text
    assert "commit_nudge" in preserve_text


def test_montage_renderer_uses_viewport_canvas_not_full_montage():
    render_text = (ROOT / "arrayscope" / "window" / "render.py").read_text()
    montage_text = (ROOT / "arrayscope" / "window" / "montage_renderer.py").read_text()
    assert "make_montage(" not in render_text
    assert "make_montage(" not in montage_text
    assert "make_montage_viewport_canvas(" in montage_text


def test_render_display_commits_go_through_display_committer():
    render_text = (ROOT / "arrayscope" / "window" / "render.py").read_text()
    presenter_text = (ROOT / "arrayscope" / "window" / "display_presenter.py").read_text()
    forbidden = (
        ".setImage(",
        ".updateImageDataFast(",
        ".setHistogramRange(",
    )
    for token in forbidden:
        assert token not in render_text
    assert "DisplayPresentationMixin" in render_text
    assert "DisplayCommitter" in presenter_text


def test_window_render_does_not_own_presentation_policy():
    render_text = (ROOT / "arrayscope" / "window" / "render.py").read_text()
    presenter_text = (ROOT / "arrayscope" / "window" / "display_presenter.py").read_text()
    forbidden = (
        "choose_window_levels",
        "choose_montage_presentation",
        "choose_normal_image_presentation",
        "display_data_bounds",
        "finite_bounds",
        "_sampled_display_bounds",
        "_raw_display_bounds",
        "_display_histogram_bounds",
    )
    for token in forbidden:
        assert token not in render_text
    assert "decide_presentation" in presenter_text


def test_display_presentation_boundary_modules_exist():
    for rel in (
        Path("arrayscope/display/model/frame.py"),
        Path("arrayscope/display/model/commit.py"),
        Path("arrayscope/display/planning.py"),
        Path("arrayscope/display/commit.py"),
        Path("arrayscope/display/backends/pyqtgraph/tiles.py"),
        Path("arrayscope/display/backends/vispy/raster.py"),
        Path("arrayscope/display/backends/vispy/tiles.py"),
        Path("arrayscope/window/montage_levels.py"),
        Path("arrayscope/window/montage_renderer.py"),
        Path("arrayscope/window/normal_renderer.py"),
        Path("arrayscope/window/viewport_bridge.py"),
        Path("arrayscope/window/display_presenter.py"),
    ):
        assert (ROOT / rel).exists()


def test_display_presenter_does_not_infer_windowed_rgb_from_array_rank():
    text = (ROOT / "arrayscope" / "window" / "display_presenter.py").read_text()
    assert "data.ndim == 3" not in text
    assert "rgb_already_windowed=display_image.data.ndim" not in text


def test_imageview2d_owns_internal_montage_tile_layer_path():
    text = (ROOT / "arrayscope" / "display" / "imageview2d.py").read_text()
    layer_text = (ROOT / "arrayscope" / "display" / "backends" / "pyqtgraph" / "tiles.py").read_text()
    assert "setMontageTileLayerPresentation" in text
    assert "MontageTileLayer" in text
    assert "TileLayerItemState" in layer_text
    assert "montageDisplayMode" in text


def test_imageview2d_display_ownership_helpers_are_split_out():
    text = (ROOT / "arrayscope" / "display" / "imageview2d.py").read_text()
    assert "class _MontageTileOverlayItem" not in text
    assert "class MontageTileOverlayItem" in (ROOT / "arrayscope" / "display" / "overlays.py").read_text()
    assert "def item_for_roi" in (ROOT / "arrayscope" / "display" / "roi_items.py").read_text()
    assert "class ProfileMarkerOwner" in (ROOT / "arrayscope" / "display" / "profile_marker.py").read_text()


def test_image_view_graphics_items_are_added_only_by_layer_owner():
    offenders = []
    allowed = {Path("arrayscope/display/layers.py")}
    for path in (ROOT / "arrayscope").rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel in allowed:
            continue
        text = path.read_text()
        if ".view.addItem(" in text or "self.view.addItem(" in text:
            offenders.append(str(rel))
    assert offenders == []


def test_image_view_z_order_is_centralized_in_layer_owner():
    offenders = []
    allowed = {Path("arrayscope/display/layers.py")}
    for path in (ROOT / "arrayscope").rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel in allowed:
            continue
        if ".setZValue(" in path.read_text():
            offenders.append(str(rel))
    assert offenders == []


def test_predictive_compute_modules_exist():
    for rel in (
        Path("arrayscope/core/compute_policy.py"),
        Path("arrayscope/window/stage_warmup.py"),
        Path("arrayscope/window/montage_prefetch.py"),
        Path("arrayscope/operations/chunked_stage.py"),
    ):
        assert (ROOT / rel).exists()


def test_display_semantics_live_in_display_package():
    canonical = (
        Path("arrayscope/display/model/frame.py"),
        Path("arrayscope/display/model/commit.py"),
        Path("arrayscope/display/planning.py"),
        Path("arrayscope/display/commit.py"),
        Path("arrayscope/display/backends/pyqtgraph/tiles.py"),
        Path("arrayscope/display/backends/vispy/raster.py"),
        Path("arrayscope/display/backends/vispy/tiles.py"),
    )
    legacy = (
        Path("arrayscope/window/display_frame.py"),
        Path("arrayscope/window/render_model.py"),
        Path("arrayscope/window/presentation.py"),
        Path("arrayscope/window/display_commit.py"),
        Path("arrayscope/display/montage_tile_layer.py"),
        Path("arrayscope/display/vispy_tiled_renderer.py"),
    )
    for rel in canonical:
        assert (ROOT / rel).exists()
    for rel in legacy:
        assert not (ROOT / rel).exists()


def test_histogram_imageitem_binding_is_centralized():
    text = (ROOT / "arrayscope" / "display" / "imageview2d.py").read_text()
    assert "def _bind_histogram_item" in text
    assert text.count(".setImageItem(") == 1
    assert "self.histogram.setImageItem(item)" in text


def test_montage_renderer_does_not_mutate_image_items_directly():
    text = (ROOT / "arrayscope" / "window" / "montage_renderer.py").read_text()
    forbidden = (".setImage(", ".setMontageTileLayerPresentation(", "ImageItem(")
    for token in forbidden:
        assert token not in text


def test_direct_numpy_fft_calls_are_confined_to_fft_backend_and_tests():
    offenders = []
    allowed = {
        Path("arrayscope/operations/fft_backend.py"),
    }
    for path in (ROOT / "arrayscope").rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel in allowed:
            continue
        text = path.read_text()
        if "np.fft.fft(" in text or "np.fft.ifft(" in text:
            offenders.append(str(rel))
    assert offenders == []


def test_operation_coordinator_dtype_estimates_delegate_to_cost_model():
    text = (ROOT / "arrayscope" / "operations" / "coordinator.py").read_text()
    assert "operation_output_dtype" in text
    assert "np.result_type" not in text
    assert "RootSumSquares" not in text


def test_operation_cost_module_is_qt_free():
    text = (ROOT / "arrayscope" / "operations" / "cost.py").read_text()
    assert "Qt" not in text
    assert "pyqtgraph" not in text


def test_operation_planner_contract_modules_are_qt_free():
    for rel in (
        Path("arrayscope/operations/capabilities.py"),
        Path("arrayscope/operations/regions.py"),
        Path("arrayscope/operations/planner.py"),
    ):
        text = (ROOT / rel).read_text()
        assert "Qt" not in text
        assert "pyqtgraph" not in text


def test_operation_optimizer_is_qt_free_and_not_ui_coupled():
    text = (ROOT / "arrayscope" / "operations" / "optimizer.py").read_text()
    assert "Qt" not in text
    assert "pyqtgraph" not in text
    assert "arrayscope.ui" not in text
    assert "arrayscope.window" not in text


def test_operation_simplification_does_not_mutate_document_steps():
    from arrayscope.operations.optimizer import optimize_operations
    from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT, CenteredIFFT

    data = np.zeros((2, 3, 4), dtype=np.float32)
    document = ArrayDocument(data, operations=(CenteredFFT(axis=2), CenteredIFFT(axis=2)))
    steps = document.steps

    optimize_operations(data.shape, data.dtype, document.enabled_operations)

    assert document.steps == steps
    assert [type(step.operation).__name__ for step in document.steps] == ["CenteredFFT", "CenteredIFFT"]


def test_window_render_does_not_contain_operation_simplification_type_checks():
    text = (ROOT / "arrayscope" / "window" / "render.py").read_text()
    assert "optimize_operations" not in text
    assert "CastDType" not in text


def test_operation_cost_uses_operation_declarations_not_registered_type_switches():
    text = (ROOT / "arrayscope" / "operations" / "cost.py").read_text()
    for token in (
        "CenteredFFT",
        "CenteredIFFT",
        "RootSumSquares",
        "CombineRealImagAxis",
        "SplitComplexAxis",
    ):
        assert token not in text


def test_stage_cache_is_qt_free_and_owned_by_operation_evaluator():
    stage_cache_text = (ROOT / "arrayscope" / "operations" / "stage_cache.py").read_text()
    assert "Qt" not in stage_cache_text
    assert "pyqtgraph" not in stage_cache_text
    assert "StageKey" in stage_cache_text

    evaluator_text = (ROOT / "arrayscope" / "operations" / "evaluator.py").read_text()
    assert "self._stage_cache = StageCache" in evaluator_text
    assert "stage_cache_budget_bytes" in evaluator_text

    render_text = (ROOT / "arrayscope" / "window" / "render.py").read_text()
    assert "StageCache(" not in render_text


def test_optional_disk_stage_cache_is_not_in_roadmap_or_runtime():
    roadmap = (ROOT / "docs" / "roadmap.md").read_text()
    assert "disk-backed cache" not in roadmap
    assert "memmap" not in roadmap
    assert not (ROOT / "arrayscope" / "operations" / "disk_stage_cache.py").exists()


def test_slabs_do_not_branch_on_registered_operation_types():
    text = (ROOT / "arrayscope" / "operations" / "slabs.py").read_text()
    for token in (
        "CenteredFFT",
        "CenteredIFFT",
        "Crop",
        "ReverseAxis",
        "FFTShift",
        "RootSumSquares",
        "CombineRealImagAxis",
        "SplitComplexAxis",
    ):
        assert token not in text


def test_registered_operations_define_region_contract_methods():
    from arrayscope.operations.registry import operation_entries

    for entry in operation_entries():
        assert hasattr(entry.operation_type, "required_input_region"), entry.id
        assert hasattr(entry.operation_type, "apply_to_region"), entry.id


def test_memory_policy_and_runtime_diagnostics_are_qt_free():
    for rel in (
        Path("arrayscope/core/memory_policy.py"),
        Path("arrayscope/core/runtime_diagnostics.py"),
    ):
        text = (ROOT / rel).read_text()
        assert "Qt" not in text
        assert "pyqtgraph" not in text


def test_diagnostics_qt_imports_stay_in_ui_module():
    text = (ROOT / "arrayscope" / "ui" / "diagnostics.py").read_text()
    assert "Qt" in text
    assert "pyqtgraph" in text
    for rel in (
        Path("arrayscope/core/runtime_diagnostics.py"),
        Path("arrayscope/core/memory_policy.py"),
    ):
        pure_text = (ROOT / rel).read_text()
        assert "pyqtgraph" not in pure_text


def test_window_render_uses_memory_policy_not_static_budget_constants():
    text = (ROOT / "arrayscope" / "window" / "render.py").read_text()
    forbidden = (
        "VISIBLE_RENDER_BUDGET_BYTES",
        "MONTAGE_BUDGET_BYTES",
        "PREFETCH_BUDGET_BYTES",
        "_select_visible_montage_tiles_by_budget",
    )
    for token in forbidden:
        assert token not in text


def test_operation_evaluator_owns_separate_display_caches():
    text = (ROOT / "arrayscope" / "operations" / "evaluator.py").read_text()
    assert "self._image_cache = BoundedArrayCache" in text
    assert "self._tile_cache = BoundedArrayCache" in text
    assert "self._profile_cache = BoundedArrayCache" in text


def test_scheduler_v2_pure_modules_are_qt_free():
    for rel in (
        Path("arrayscope/operations/render_plan.py"),
        Path("arrayscope/operations/chunked.py"),
    ):
        text = (ROOT / rel).read_text()
        assert "Qt" not in text
        assert "pyqtgraph" not in text


def test_montage_state_modules_are_qt_free():
    for rel in (
        Path("arrayscope/display/montage.py"),
        Path("arrayscope/display/geometry.py"),
        Path("arrayscope/window/montage_session.py"),
    ):
        text = (ROOT / rel).read_text()
        assert "pyqtgraph" not in text
        if rel != Path("arrayscope/window/montage_session.py"):
            assert "Qt" not in text


def test_update_montage_view_does_not_batch_missing_tiles():
    text = (ROOT / "arrayscope" / "window" / "montage_renderer.py").read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "update_montage_view":
            segment = ast.get_source_segment(text, node) or ""
            assert "tuple((tile, evaluate_image_snapshot" not in segment
            assert "for tile in missing_tiles)" not in segment
            assert "_schedule_montage_tiles" in segment
            return
    raise AssertionError("update_montage_view not found")


def test_stale_montage_callbacks_do_not_clear_current_overlay():
    text = (ROOT / "arrayscope" / "window" / "montage_renderer.py").read_text()
    for name in ("_on_montage_tile_done", "_on_montage_tile_error"):
        marker = f"def {name}"
        assert marker in text
        segment = text.split(marker, 1)[1].split("\n    def ", 1)[0]
        stale_prefix = segment.split("return", 1)[0]
        assert "setEvaluationOverlay(False)" not in stale_prefix
        assert "setImageStale(False)" not in stale_prefix


def test_normal_renderer_uses_render_decision_helper():
    text = (ROOT / "arrayscope" / "window" / "normal_renderer.py").read_text()
    assert "choose_visible_render_decision" in text
    assert "estimate_visible_render_context" in text


def test_visible_controller_remains_single_worker():
    text = (ROOT / "arrayscope" / "window" / "main.py").read_text()
    policy_text = (ROOT / "arrayscope" / "core" / "compute_policy.py").read_text()
    assert 'EvaluationController(self, max_workers=self.compute_policy.visible_workers, name="visible")' in text
    assert "visible_workers=1" in policy_text


def test_degraded_preview_is_not_stored_in_exact_image_cache():
    text = (ROOT / "arrayscope" / "window" / "normal_renderer.py").read_text()
    marker = "if decision.kind == RenderDecisionKind.DEGRADED_PREVIEW:"
    assert marker in text
    degraded_block = text.split(marker, 1)[1].split("def evaluate(token):", 1)[0]
    assert "store_image_result" not in degraded_block
