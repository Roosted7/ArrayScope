import ast
from pathlib import Path


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


def test_window_render_does_not_compare_partial_document_keys():
    text = (ROOT / "arrayscope" / "window" / "render.py").read_text()
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
    assert ".setGeometry(" not in text
    assert "_clamp_to_available_screen" not in text
    assert ".resize(" in text
    assert "run_panel_transition_preserving_canvas" in text
    assert "_correct_canvas_size" in text


def test_window_render_montage_view_does_not_call_make_montage():
    text = (ROOT / "arrayscope" / "window" / "render.py").read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "update_montage_view":
            segment = ast.get_source_segment(text, node) or ""
            assert "make_montage(" not in segment
            assert "make_montage_viewport_canvas(" in segment
            return
    raise AssertionError("update_montage_view not found")


def test_imageview2d_has_no_multi_imageitem_tile_display_path():
    text = (ROOT / "arrayscope" / "display" / "imageview2d.py").read_text()
    forbidden = ("setImageTiles", "clearTiles", "_tile_items", "_tile_histogram_sources", "_tile_mode")
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


def test_scheduler_v2_pure_modules_are_qt_free():
    for rel in (
        Path("arrayscope/operations/render_plan.py"),
        Path("arrayscope/operations/chunked.py"),
    ):
        text = (ROOT / rel).read_text()
        assert "Qt" not in text
        assert "pyqtgraph" not in text


def test_window_render_uses_render_decision_helper():
    text = (ROOT / "arrayscope" / "window" / "render.py").read_text()
    assert "choose_visible_render_decision" in text
    assert "estimate_visible_render_context" in text


def test_visible_controller_remains_single_worker():
    text = (ROOT / "arrayscope" / "window" / "main.py").read_text()
    assert 'EvaluationController(self, max_workers=1, name="visible")' in text


def test_degraded_preview_is_not_stored_in_exact_image_cache():
    text = (ROOT / "arrayscope" / "window" / "render.py").read_text()
    marker = "if decision.kind == RenderDecisionKind.DEGRADED_PREVIEW:"
    assert marker in text
    degraded_block = text.split(marker, 1)[1].split("def evaluate(token):", 1)[0]
    assert "store_image_result" not in degraded_block
