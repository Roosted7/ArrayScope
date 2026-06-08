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
