import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from arrayscope.tools.interaction_budget import (
    INTERACTION_SETTLE_HARD_LIMIT_MS,
    INTERACTION_SETTLE_HARD_LIMIT_S,
)


ROOT = Path(__file__).parents[2]


def _constant_number(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -float(node.operand.value)
    return None


_INTERACTION_BUDGET_UNITS = {
    "INTERACTION_SETTLE_HARD_LIMIT_MS": "ms",
    "INTERACTION_SETTLE_HARD_LIMIT_S": "s",
}
_INTERACTION_BUDGET_HELPER_UNITS = {
    "bounded_interaction_settle_timeout_s": "s",
    "interaction_settle_timeout_ms": "ms",
}


def _call_name(node):
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _contains_number(node):
    return any(_constant_number(child) is not None for child in ast.walk(node))


def _literal_number_expression(node):
    if _constant_number(node) is not None:
        return True
    return bool(
        isinstance(node, ast.Call)
        and _call_name(node) in {"float", "int"}
        and len(node.args) == 1
        and not node.keywords
        and _constant_number(node.args[0]) is not None
    )


def _has_direct_deadline_literal(node):
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and (
            _literal_number_expression(node.left)
            or _literal_number_expression(node.right)
        )
    )


def _canonical_interaction_budget_bindings(tree):
    """Return unshadowed names imported from the one budget owner."""

    imported_budgets = {}
    imported_helpers = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "arrayscope.tools.interaction_budget":
            continue
        for alias in node.names:
            bound = alias.asname or alias.name
            if alias.name in _INTERACTION_BUDGET_UNITS:
                imported_budgets[bound] = _INTERACTION_BUDGET_UNITS[alias.name]
            if alias.name in _INTERACTION_BUDGET_HELPER_UNITS:
                imported_helpers[bound] = _INTERACTION_BUDGET_HELPER_UNITS[alias.name]

    shadowed = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            shadowed.update(
                label
                for target in targets
                for label in _assigned_labels(target)
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            shadowed.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
                shadowed.update(argument.arg for argument in arguments)
                if node.args.vararg is not None:
                    shadowed.add(node.args.vararg.arg)
                if node.args.kwarg is not None:
                    shadowed.add(node.args.kwarg.arg)
    return (
        {name: unit for name, unit in imported_budgets.items() if name not in shadowed},
        {name: unit for name, unit in imported_helpers.items() if name not in shadowed},
    )


def _interaction_budget_unit(node, *, budgets, helpers):
    """Return the unit of a safe outer cap, or ``None`` when it is unsafe."""

    if isinstance(node, ast.Name):
        return budgets.get(node.id)
    if not isinstance(node, ast.Call):
        return None
    name = _call_name(node)
    if name in helpers:
        return helpers[name]
    if name == "min" and not node.keywords:
        units = {
            unit
            for argument in node.args
            if (unit := _interaction_budget_unit(
                argument,
                budgets=budgets,
                helpers=helpers,
            )) is not None
        }
        return next(iter(units)) if len(units) == 1 else None
    return None


def _is_global_capped_interaction_budget(
    node,
    *,
    budgets,
    helpers,
    expected_unit=None,
):
    """Accept only safe outer caps expressed in the call site's unit."""

    unit = _interaction_budget_unit(node, budgets=budgets, helpers=helpers)
    return unit is not None and (expected_unit is None or unit == expected_unit)


def _module_capped_budget_names(tree, *, budgets, helpers):
    """Resolve only immutable-looking module assignments from capped forms."""

    capped = dict(budgets)
    assignments = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        names = {
            label
            for target in targets
            for label in _assigned_labels(target)
            if label.isupper()
        }
        for name in names:
            assignments.setdefault(name, []).append(node.value)
    changed = True
    while changed:
        changed = False
        for name, values in assignments.items():
            if name in capped or len(values) != 1:
                continue
            unit = _interaction_budget_unit(
                values[0],
                budgets=capped,
                helpers=helpers,
            )
            if unit is not None:
                capped[name] = unit
                changed = True
    return capped


def _assigned_labels(node):
    if isinstance(node, ast.Name):
        return {node.id}
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return {node.slice.value}
    return set()


def _interaction_wait_name(function_name):
    if function_name in {
        "waitUntil",
        "waitExposed",
        "settle",
        "wait_settled",
        "_wait_until",
        "_process_until",
        "_drain_until",
        "_wait_idle",
    }:
        return True
    lowered = function_name.lower()
    if "settle" in lowered:
        return True
    return any(
        token in lowered
        for token in (
            "target_lod",
            "montage_complete",
            "profile_montage_workflow",
            "presentation_quiet",
            "vispy_tile_draw",
        )
    )


def _interaction_timeout_call(function_name, keyword_name):
    if function_name in {"waitUntil", "waitExposed", "settle", "wait_settled"}:
        return keyword_name == "timeout"
    if keyword_name not in {"budget_s", "timeout_s"}:
        return False
    return _interaction_wait_name(function_name)


def _timeout_unit(*, function_name="", argument_name=""):
    lowered = str(argument_name).lower()
    if lowered.endswith("_ms"):
        return "ms"
    if lowered.endswith("_s") or lowered == "budget_s":
        return "s"
    if function_name in {"waitUntil", "waitExposed"}:
        return "ms"
    # Project-owned settle wrappers consistently accept seconds. Generic
    # ``timeout`` must not make the seconds/milliseconds owners interchangeable.
    return "s"


def _interaction_timeout_offenders(tree, rel):
    offenders = []
    budgets, helpers = _canonical_interaction_budget_bindings(tree)
    budgets = _module_capped_budget_names(
        tree,
        budgets=budgets,
        helpers=helpers,
    )
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            value = node.value
            labels = set().union(*(_assigned_labels(target) for target in targets))
            for name in labels:
                upper = name.upper()
                timeout_label = "timeout" in name.lower()
                interaction_budget_label = (
                    ("INTERACTION" in upper or "SETTLE" in upper)
                    and ("BUDGET" in upper or "LIMIT" in upper)
                )
                if (
                    (timeout_label or interaction_budget_label)
                    and _contains_number(value)
                    and rel != Path("arrayscope/tools/interaction_budget.py")
                    and not _is_global_capped_interaction_budget(
                        value,
                        budgets=budgets,
                        helpers=helpers,
                        expected_unit=_timeout_unit(argument_name=name),
                    )
                ):
                    offenders.append(f"{rel}:{node.lineno}:uncapped {name}")
                if (
                    "deadline" in name.lower()
                    and _has_direct_deadline_literal(value)
                    and any(
                        isinstance(child, ast.Call)
                        and _call_name(child) in {"monotonic", "perf_counter"}
                        for child in ast.walk(value)
                    )
                    and not _is_global_capped_interaction_budget(
                        value,
                        budgets=budgets,
                        helpers=helpers,
                        expected_unit="s",
                    )
                ):
                    offenders.append(f"{rel}:{node.lineno}:uncapped {name}")
        if isinstance(node, ast.FunctionDef) and _interaction_wait_name(node.name):
            positional = (*node.args.posonlyargs, *node.args.args)
            defaults = (None,) * (len(positional) - len(node.args.defaults)) + tuple(node.args.defaults)
            for argument, default in zip(positional, defaults, strict=True):
                if (
                    default is not None
                    and argument.arg in {"timeout", "timeout_ms", "timeout_s", "budget_s"}
                    and _contains_number(default)
                    and not _is_global_capped_interaction_budget(
                        default,
                        budgets=budgets,
                        helpers=helpers,
                        expected_unit=_timeout_unit(
                            function_name=node.name,
                            argument_name=argument.arg,
                        ),
                    )
                ):
                    offenders.append(
                        f"{rel}:{node.lineno}:uncapped {node.name} default {argument.arg}"
                    )
            for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
                if (
                    default is not None
                    and argument.arg in {"timeout", "timeout_ms", "timeout_s", "budget_s"}
                    and _contains_number(default)
                    and not _is_global_capped_interaction_budget(
                        default,
                        budgets=budgets,
                        helpers=helpers,
                        expected_unit=_timeout_unit(
                            function_name=node.name,
                            argument_name=argument.arg,
                        ),
                    )
                ):
                    offenders.append(
                        f"{rel}:{node.lineno}:uncapped {node.name} default {argument.arg}"
                    )
        if not isinstance(node, ast.Call):
            continue
        function_name = _call_name(node)
        if function_name == "singleShot" and node.args:
            delay = _constant_number(node.args[0])
            callback_name = _call_name(node.args[-1])
            if (
                delay is not None
                and delay > INTERACTION_SETTLE_HARD_LIMIT_MS
                and "process_guard" not in callback_name.lower()
            ):
                offenders.append(f"{rel}:{node.lineno}:singleShot delay={delay:g}")
        for keyword in node.keywords:
            if not _interaction_timeout_call(function_name, keyword.arg):
                continue
            if not _is_global_capped_interaction_budget(
                keyword.value,
                budgets=budgets,
                helpers=helpers,
                expected_unit=_timeout_unit(
                    function_name=function_name,
                    argument_name=keyword.arg,
                ),
            ):
                offenders.append(
                    f"{rel}:{node.lineno}:{function_name} {keyword.arg} is not globally capped"
                )
        if function_name in {"waitUntil", "waitExposed"} and len(node.args) >= 2:
            if not _is_global_capped_interaction_budget(
                node.args[1],
                budgets=budgets,
                helpers=helpers,
                expected_unit="ms",
            ):
                offenders.append(
                    f"{rel}:{node.lineno}:{function_name} positional timeout is not globally capped"
                )
    return offenders


# Documented eventual-settlement/build budgets (progressive presentation
# contract, docs/architecture/rendering.md): correctness is EVENTUAL
# convergence; the global 5 s cap governs per-gesture probes only. Hard-
# capping these turned alive-but-slow runs into aborts that tested nothing
# (2026-07-17: 0.5 s draw probe aborted at 35/36 with 202 tiles acked
# after it; 5 s build cap made the churn net erroring-red). Perf latency
# is owned by the bars program, not by widening correctness gates.
_EVENTUAL_SETTLEMENT_BUDGET_ALLOWLIST = (
    ("arrayscope/tools/profile_montage_workflow.py", "_wait_for_vispy_tile_draw"),
    ("tests/stress/test_interaction_convergence.py", "_FILL_TIMEOUT_S"),
    ("tests/stress/test_interaction_convergence.py", "waitUntil"),
    # Build-time cold-fill wait; the per-gesture wait uses the capped owner.
    ("tests/ui/test_lod_demand_freshness.py", "_FILL_TIMEOUT_MS"),
    ("tests/ui/test_lod_demand_freshness.py", "waitUntil"),
)


def test_interaction_gates_have_one_bounded_timeout_owner():
    """No UI gate may turn a slow step green by widening its local timeout."""

    roots = (
        ROOT / "arrayscope" / "tools",
        ROOT / "tools",
        ROOT / "tests",
    )
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            rel = path.relative_to(ROOT)
            if rel == Path("tests/app/test_architecture_guards.py"):
                continue
            tree = ast.parse(path.read_text())
            offenders.extend(_interaction_timeout_offenders(tree, rel))
    offenders = [
        offender
        for offender in offenders
        if not any(
            path_part in offender and token in offender
            for path_part, token in _EVENTUAL_SETTLEMENT_BUDGET_ALLOWLIST
        )
    ]
    assert offenders == [], "\n".join(offenders)


def test_interaction_timeout_guard_rejects_ui_literals_but_not_thread_guards():
    tree = ast.parse(
        """
def probe(qtbot, finished, thread):
    qtbot.waitUntil(lambda: True, timeout=3000)
    finished.wait(timeout=30)
    thread.join(timeout=30)
"""
    )

    assert _interaction_timeout_offenders(tree, Path("tools/probe.py")) == [
        "tools/probe.py:3:waitUntil timeout is not globally capped"
    ]


def test_interaction_timeout_guard_accepts_shorter_global_cap():
    tree = ast.parse(
        """
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS

def probe(qtbot):
    qtbot.waitUntil(
        lambda: True,
        timeout=min(3000, INTERACTION_SETTLE_HARD_LIMIT_MS),
    )
"""
    )

    assert _interaction_timeout_offenders(tree, Path("tests/ui/test_probe.py")) == []


def test_interaction_timeout_guard_covers_root_tool_phase_machines():
    tree = ast.parse(
        """
PHASES = [("load", lambda: None, 60.0)]
deadline = monotonic() + 20.0
settle_checks = 0
if settle_checks > 120:
    raise RuntimeError
QtCore.QTimer.singleShot(22000, win, start_work)
QtCore.QTimer.singleShot(180000, win, process_guard_expired)

def settle(timeout=25.0):
    return timeout
"""
    )

    offenders = _interaction_timeout_offenders(tree, Path("tools/probe.py"))
    assert len(offenders) == 3
    assert any("uncapped deadline" in offender for offender in offenders)
    assert any("singleShot delay=22000" in offender for offender in offenders)
    assert any("uncapped settle default timeout" in offender for offender in offenders)
    assert all("180000" not in offender for offender in offenders)


@pytest.mark.parametrize(
    "expression",
    (
        "INTERACTION_SETTLE_HARD_LIMIT_MS * 100",
        "max(3000, INTERACTION_SETTLE_HARD_LIMIT_MS)",
        "interaction_settle_timeout_ms(3.0) + 100000",
    ),
)
def test_interaction_timeout_guard_rejects_caps_wrapped_by_widening_operations(
    expression,
):
    tree = ast.parse(
        f"""
from arrayscope.tools.interaction_budget import (
    INTERACTION_SETTLE_HARD_LIMIT_MS,
    interaction_settle_timeout_ms,
)

def probe(qtbot):
    qtbot.waitUntil(lambda: True, timeout={expression})
"""
    )

    offenders = _interaction_timeout_offenders(tree, Path("tests/ui/test_probe.py"))
    assert len(offenders) == 1
    assert "is not globally capped" in offenders[0]


def test_interaction_timeout_guard_rejects_unit_mismatch():
    tree = ast.parse(
        """
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS

def _wait_until(*, timeout_s):
    return timeout_s

_wait_until(timeout_s=INTERACTION_SETTLE_HARD_LIMIT_MS)
"""
    )

    offenders = _interaction_timeout_offenders(tree, Path("tools/probe.py"))
    assert offenders == [
        "tools/probe.py:7:_wait_until timeout_s is not globally capped"
    ]


def test_interaction_timeout_guard_covers_settled_wrappers_and_ui_watchdogs():
    tree = ast.parse(
        """
def _settled(predicate, timeout=60000):
    return predicate()

QtCore.QTimer.singleShot(180000, win, interaction_watchdog)
QtCore.QTimer.singleShot(180000, win, process_guard_expired)
"""
    )

    offenders = _interaction_timeout_offenders(tree, Path("tests/ui/test_probe.py"))
    assert len(offenders) == 2
    assert any("uncapped _settled default timeout" in item for item in offenders)
    assert any("singleShot delay=180000" in item for item in offenders)
    assert all("process_guard_expired" not in item for item in offenders)


def test_interaction_timeout_guard_rejects_shadowed_owner_and_indirect_literals():
    tree = ast.parse(
        """
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS

INTERACTION_SETTLE_HARD_LIMIT_MS = 60000
LOCAL_TIMEOUT = 60000
REASSIGNED_TIMEOUT = min(3000, INTERACTION_SETTLE_HARD_LIMIT_MS)
REASSIGNED_TIMEOUT = 60000

def probe(qtbot):
    timeout_s = 60
    deadline = monotonic() + timeout_s
    other_deadline = monotonic() + float(60)
    qtbot.waitUntil(lambda: True, timeout=LOCAL_TIMEOUT)
    qtbot.waitUntil(lambda: True, timeout=REASSIGNED_TIMEOUT)
"""
    )

    offenders = _interaction_timeout_offenders(tree, Path("tests/ui/test_probe.py"))
    assert any("uncapped INTERACTION_SETTLE_HARD_LIMIT_MS" in item for item in offenders)
    assert any("uncapped LOCAL_TIMEOUT" in item for item in offenders)
    assert any("uncapped timeout_s" in item for item in offenders)
    assert any("uncapped other_deadline" in item for item in offenders)
    assert sum("waitUntil timeout is not globally capped" in item for item in offenders) == 2


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
            Path("arrayscope/window/frame_controller.py"),
            Path("arrayscope/window/render_prefetch.py"),
        )
    )
    assert ".display_tile_key(" in text
    assert "display_tile_key(view_state, colormap_lut=colormap_lut)[1]" not in text
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


def test_montage_stall_probe_is_evidence_only():
    path = ROOT / "arrayscope" / "window" / "frame_runtime.py"
    text = path.read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_montage_watchdog_tick":
            source = ast.get_source_segment(text, node) or ""
            forbidden = (
                "_dispatch_montage_" + "work",
                "apply_montage_presentation",
                "_schedule_deferred_montage_planning",
                "refresh_lod_for_viewport",
                "requeue_orphaned_loading_tiles",
                "_montage_stall_" + "repairs",
                "STALL " + "WATCHDOG",
            )
            for token in forbidden:
                assert token not in source
            assert "_montage_stall_assertions" in source
            assert "release_idle_evaluation_claims" not in source
            assert 'emit_trace(\n            "stall"' in source
            assert "TRACE.dump" in source
            assert "show_status_message" in source
            return
    raise AssertionError("_montage_watchdog_tick not found")


def test_frame_renderer_stays_below_r2_line_count_gate():
    path = ROOT / "arrayscope" / "window" / "frame_controller.py"
    assert len(path.read_text().splitlines()) < 2000


def test_montage_commits_flow_through_pipeline_effects_and_shared_surface():
    frame_text = (ROOT / "arrayscope" / "window" / "frame_controller.py").read_text()
    commit_text = (ROOT / "arrayscope" / "window" / "frame_effects.py").read_text()
    assert "_commit_frame_session_tile_layer(" not in frame_text
    assert "class FramePipelineEffects" in commit_text
    assert ".commit_tiled_delta(" in commit_text


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
        Path("arrayscope/display/frame_planner.py"),
        Path("arrayscope/display/model/frame.py"),
        Path("arrayscope/display/model/commit.py"),
        Path("arrayscope/display/planning.py"),
        Path("arrayscope/display/commit.py"),
        Path("arrayscope/display/backends/pyqtgraph/tiles.py"),
        Path("arrayscope/display/backends/vispy/tiles.py"),
        Path("arrayscope/display/model/montage_levels.py"),
        Path("arrayscope/window/frame_controller.py"),
        Path("arrayscope/window/viewport_bridge.py"),
        Path("arrayscope/window/display_presenter.py"),
    ):
        assert (ROOT / rel).exists()
    assert not (ROOT / "arrayscope" / "window" / "normal_renderer.py").exists()
    assert not (ROOT / "arrayscope" / "window" / "montage_renderer.py").exists()


def test_display_presenter_has_one_commit_path_without_fallback_planning():
    # The frame session owns the plan and tile presentation. The presenter
    # must not re-plan frame semantics behind the session's back — the
    # 2026-07-15 window-shift diagnosis traced a dead divergent secondary
    # planner that masked the live flow never using it. Commits without the
    # session's plan/state fail loudly instead.
    text = (ROOT / "arrayscope" / "window" / "display_presenter.py").read_text()
    assert "_frame_plan_for_display" not in text
    assert "display commits require the session's frame_plan and tile_state" in text
    # The single-layout viewport retarget legitimately re-plans active/near
    # sets over the committed payloads; that planner stays.
    assert "FramePlanner" in text


def test_tiled_display_committer_does_not_require_montage_geometry():
    text = (ROOT / "arrayscope" / "display" / "commit.py").read_text()
    assert "tiled display presentation requires montage geometry" not in text


def test_display_presenter_does_not_infer_windowed_rgb_from_array_rank():
    text = (ROOT / "arrayscope" / "window" / "display_presenter.py").read_text()
    assert "data.ndim == 3" not in text
    assert "rgb_already_windowed=display_image.data.ndim" not in text


def test_lod_admission_has_no_effects_side_pending_maps_or_shared_floor_markers():
    source_roots = [
        ROOT / "arrayscope" / "render",
        ROOT / "arrayscope" / "window",
        ROOT / "arrayscope" / "presentation",
    ]
    forbidden = (
        "_shared_floor_tiles",
        "_shared_floor_inflight_marker",
        "_shared_floor_admitted_marker",
        "pending_lod_requests",
        "montage_lod",
    )
    for root in source_roots:
        for path in root.rglob("*.py"):
            text = path.read_text()
            for token in forbidden:
                assert token not in text, f"{token} found in {path.relative_to(ROOT)}"
            for token in ("_pending_previews", "_pending_materializations", "_pending_evaluations"):
                for prefix in ("self.", "effects.", "session."):
                    assert f"{prefix}{token}" not in text, f"{token} found in {path.relative_to(ROOT)}"


def test_tile_lifecycle_is_the_only_per_region_transaction_owner():
    assert not (ROOT / "arrayscope" / "presentation" / "tile_ledger.py").exists()
    for path in (ROOT / "arrayscope").rglob("*.py"):
        text = path.read_text()
        assert "tile_ledger" not in text, f"parallel tile ledger found in {path.relative_to(ROOT)}"


def test_frame_control_plane_has_no_legacy_module_shims():
    removed = (
        "frame_renderer.py",
        "montage_commit.py",
        "montage_runtime.py",
        "montage_session.py",
    )
    for name in removed:
        assert not (ROOT / "arrayscope" / "window" / name).exists()


def test_qtimers_are_explicitly_allowlisted_by_category():
    """R4 guard: new timers must be justified as UI cosmetic or anti-hang."""

    allowed = Counter(
        {
            ("arrayscope/display/histogram_controller.py", "HistogramLevelPreviewController.__init__", "QTimer", "UI cosmetic"): 1,
            ("arrayscope/display/histogram_controller.py", "HistogramDisplayController.__init__", "QTimer", "UI cosmetic"): 2,
            ("arrayscope/display/imageview2d.py", "ImageViewShell.eventFilter", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/display/rendering_benchmarks.py", "_measure_presented_action.PaintProbe.eventFilter", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/display/rendering_benchmarks.py", "_measure_presented_action", "QTimer", "UI cosmetic"): 1,
            ("arrayscope/display/vispy_imageview2d.py", "VisPyImageView2D.setupUI", "QTimer", "UI cosmetic"): 1,
            ("arrayscope/display/vispy_imageview2d.py", "VisPyImageView2D._on_vispy_draw", "singleShot", "anti-hang fallback"): 1,
            ("arrayscope/display/vispy_imageview2d.py", "VisPyImageView2D._request_vispy_camera_sync", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/kernel/eval_adapter.py", "KernelEvaluationController._submit", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/kernel/qt_bridge.py", "QtKernelBridge.__init__", "QTimer", "anti-hang fallback"): 1,
            ("arrayscope/sync/bus.py", "SyncBus._schedule_retry", "QTimer", "anti-hang fallback"): 1,
            ("arrayscope/sync/controller.py", "WindowSyncController.schedule_publish", "QTimer", "UI cosmetic"): 1,
            ("arrayscope/tools/profile_montage_workflow.py", "_EventLoopProbe.__init__", "QTimer", "UI cosmetic"): 1,
            ("arrayscope/tools/profile_montage_workflow.py", "_VisualTimelineProbe.__init__", "QTimer", "UI cosmetic"): 1,
            ("arrayscope/tools/profile_montage_workflow.py", "_PresentationContinuityProbe.__init__", "QTimer", "UI cosmetic"): 1,
            ("arrayscope/tools/profile_montage_workflow.py", "_glide_view_range", "QTimer", "UI cosmetic"): 1,
            ("arrayscope/tools/profile_montage_workflow.py", "_fast_scroll_60fps", "QTimer", "UI cosmetic"): 2,
            ("arrayscope/tools/profile_montage_workflow.py", "_wait_for_target_lod", "QTimer", "anti-hang fallback"): 2,
            ("arrayscope/tools/profile_scroll_input.py", "main", "QTimer", "UI cosmetic"): 1,
            ("arrayscope/tools/profile_scroll_input.py", "main.on_tick", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/ui/diagnostics.py", "DiagnosticsDialog.__init__", "QTimer", "UI cosmetic"): 1,
            ("arrayscope/ui/dimension_strip.py", "DimensionStrip._schedule_relayout", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/ui/display_controls.py", "DisplayControlBuildMixin._create_button_groups_and_profile_timer", "QTimer", "UI cosmetic"): 1,
            ("arrayscope/ui/menus.py", "WindowMenuMixin.closeEvent", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/ui/toasts.py", "show_status_message", "QTimer", "UI cosmetic"): 1,
            ("arrayscope/ui/toasts.py", "show_status_action", "QTimer", "UI cosmetic"): 1,
            ("arrayscope/window/canvas_preserve.py", "CanvasPreserveController._single_shot", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/window/display_presenter.py", "DisplayPresentationMixin._schedule_frame_viewport_update", "QTimer", "UI cosmetic"): 1,
            ("arrayscope/window/display_presenter.py", "DisplayPresentationMixin._schedule_interactive_montage_viewport_update", "QTimer", "UI cosmetic"): 1,
            ("arrayscope/window/file_view_session.py", "FileViewSessionMixin._schedule_viewport_continuity_shape_restore", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/window/file_view_session.py", "FileViewSessionMixin._restore_viewport_continuity_shape_step", "singleShot", "UI cosmetic"): 2,
            ("arrayscope/window/file_view_session.py", "FileViewSessionMixin._schedule_viewport_continuity_when_ready", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/window/file_view_session.py", "FileViewSessionMixin._schedule_viewport_continuity_retarget", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/window/layout_controller.py", "WindowLayoutManager.restore_window_settings", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/window/layout_controller.py", "WindowLayoutManager.reset_layout", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/window/layout_controller.py", "WindowLayoutManager.schedule_view_geometry_refresh", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/window/layout_controller.py", "WindowLayoutManager.set_dock_visible_later", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/window/layout_controller.py", "WindowLayoutManager.set_managed_dock_visible", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/window/main.py", "ArrayScopeWindow.__init__", "singleShot", "UI cosmetic"): 3,
            ("arrayscope/window/main.py", "ArrayScopeWindow._note_viewport_interaction", "QTimer", "UI cosmetic"): 1,
            ("arrayscope/window/frame_effects.py", "FramePipelineEffects.request_presentation", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/window/frame_runtime.py", "FrameRuntimeMixin.request_montage_replan", "singleShot", "UI cosmetic"): 1,
            ("arrayscope/window/frame_runtime.py", "FrameRuntimeMixin._ensure_montage_watchdog", "QTimer", "anti-hang fallback"): 1,
            ("arrayscope/window/render_coordinator.py", "RenderCoordinator.__init__", "QTimer", "UI cosmetic"): 2,
        }
    )
    found = Counter()

    class Visitor(ast.NodeVisitor):
        def __init__(self, rel: str):
            self.rel = rel
            self.stack: list[str] = []

        def visit_ClassDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            kind = None
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr == "QTimer":
                    kind = "QTimer"
                elif func.attr == "singleShot":
                    kind = "singleShot"
            elif isinstance(func, ast.Name) and func.id == "QTimer":
                kind = "QTimer"
            if kind is not None:
                found[(self.rel, ".".join(self.stack) or "<module>", kind)] += 1
            self.generic_visit(node)

    for path in (ROOT / "arrayscope").rglob("*.py"):
        Visitor(str(path.relative_to(ROOT))).visit(ast.parse(path.read_text()))

    allowed_without_category = Counter(
        (rel, qualname, kind) for (rel, qualname, kind, _category), count in allowed.items() for _ in range(count)
    )
    assert found == allowed_without_category
def test_vispy_warm_residency_has_no_backend_scheduling_timer():
    text = (ROOT / "arrayscope" / "display" / "vispy_imageview2d.py").read_text()
    assert "_vispy_warm_tile_timer" not in text
    assert "_process_vispy_warm_tile_residency" in text
    assert "_vispy_warm_tile_scheduler" in text


def test_image_view_shell_exposes_surface_contract():
    text = (ROOT / "arrayscope" / "display" / "imageview2d.py").read_text()
    backend_text = (ROOT / "arrayscope" / "display" / "backends" / "base.py").read_text()
    layer_text = (ROOT / "arrayscope" / "display" / "backends" / "pyqtgraph" / "tiles.py").read_text()
    assert "class ImageViewShell" in text
    assert "class ImageView2D(ImageViewShell)" in text
    assert "class ImageSurface" in backend_text
    assert "def surface_for_view" in backend_text
    for method in (
        "def apply_camera",
        "def map_scene_to_overlay",
        "def current_viewport_rect",
        "def presentation_diagnostics",
        "def interaction_event_owner",
        "def sync_interaction_state",
        "def reset_surface",
        "def teardown_surface",
    ):
        assert method in backend_text
    assert "setMontageTileLayerPresentation" not in text
    assert "def present_tiled" in text
    assert "MontageTileLayer" in text
    assert "TileLayerItemState" in layer_text
    assert "montageDisplayMode" in text


def test_vispy_view_inherits_shell_not_pyqtgraph_concrete_view():
    text = (ROOT / "arrayscope" / "display" / "vispy_imageview2d.py").read_text()
    assert "from arrayscope.display.imageview2d import ImageViewShell" in text
    assert "class VisPyImageView2D(ImageViewShell)" in text
    assert "class VisPyImageView2D(ImageView2D)" not in text


def test_builtin_backend_method_adapters_are_removed():
    backend_text = (ROOT / "arrayscope" / "display" / "backends" / "base.py").read_text()
    committer_text = (ROOT / "arrayscope" / "display" / "commit.py").read_text()
    presenter_text = (ROOT / "arrayscope" / "window" / "display_presenter.py").read_text()
    assert "ImageViewMethodBackendAdapter" not in backend_text
    assert "backend_adapter_for_view" not in committer_text
    assert "surface_for_view" in committer_text
    assert 'hasattr(self.img_view, "setTiledPresentation")' not in presenter_text


def test_backend_identity_uses_declared_surface_capabilities():
    offenders = []
    forbidden = ("rendering_backend_name", "supports_" + "direct_montage_" + "tile_payloads")
    for path in (ROOT / "arrayscope").rglob("*.py"):
        rel = path.relative_to(ROOT)
        text = path.read_text()
        for token in forbidden:
            if token in text:
                offenders.append(f"{rel}:{token}")
    assert offenders == []


def test_imageview2d_display_ownership_helpers_are_split_out():
    text = (ROOT / "arrayscope" / "display" / "imageview2d.py").read_text()
    assert "class _MontageTileOverlayItem" not in text
    assert "class MontageTileOverlayItem" in (ROOT / "arrayscope" / "display" / "overlays.py").read_text()
    assert "def item_for_roi" in (ROOT / "arrayscope" / "display" / "roi_items.py").read_text()
    assert "class ProfileMarkerOwner" in (ROOT / "arrayscope" / "display" / "profile_marker.py").read_text()


def test_production_lod_has_no_synchronous_pyramid_entrypoints():
    forbidden = {
        "build_tile_lod_pyramid",
        "apply_tile_gutter",
        "lod_info_for_texture",
        "_box_reduce_2x2",
        "_reduction_dtype",
    }
    offenders = []
    for path in (ROOT / "arrayscope").rglob("*.py"):
        rel = path.relative_to(ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in forbidden:
                offenders.append(f"{rel}:{node.lineno}:def {node.name}")
            if isinstance(node, ast.ImportFrom) and node.module == "arrayscope.display.lod":
                for alias in node.names:
                    if alias.name in forbidden:
                        offenders.append(f"{rel}:{node.lineno}:import {alias.name}")
    assert offenders == []


def test_r3_lod_ladder_deletes_legacy_montage_lod_path():
    assert not (ROOT / "arrayscope" / "window" / "montage_lod.py").exists()
    assert not (ROOT / "arrayscope" / "window" / "montage_level_stats.py").exists()
    forbidden = (
        "montage_lod",
        "_montage_lod_",
        "pending_lod_requests",
        "lod_preview_pyramid",
        "ARRAYSCOPE_SHARED_TRANSFORM_PREVIEW",
        "admit_ingest_reduction",
        "admit_preview_reduction",
    )
    offenders = []
    for path in (ROOT / "arrayscope").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{rel}:{token}")
    assert offenders == []


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
        Path("arrayscope/window/montage_prefetch.py"),
        Path("arrayscope/operations/chunked_stage.py"),
    ):
        assert (ROOT / rel).exists()


def test_render_orchestrator_uses_one_frame_session_staleness_guard_name():
    offenders = []
    stale_name = "_is_current_" + "montage_session"
    for path in (ROOT / "arrayscope" / "window").rglob("*.py"):
        if stale_name in path.read_text():
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_display_semantics_live_in_display_package():
    canonical = (
        Path("arrayscope/display/model/frame.py"),
        Path("arrayscope/display/model/commit.py"),
        Path("arrayscope/display/planning.py"),
        Path("arrayscope/display/commit.py"),
        Path("arrayscope/display/backends/pyqtgraph/tiles.py"),
        Path("arrayscope/display/backends/vispy/tiles.py"),
    )
    retired = (
        Path("arrayscope/display/backends/vispy/gpu_mapped_visual.py"),
        Path("arrayscope/window/display_frame.py"),
        Path("arrayscope/window/render_model.py"),
        Path("arrayscope/window/presentation.py"),
        Path("arrayscope/window/display_commit.py"),
        Path("arrayscope/window/montage_levels.py"),
        Path("arrayscope/display/montage_tile_layer.py"),
        Path("arrayscope/display/vispy_tiled_renderer.py"),
    )
    for rel in canonical:
        assert (ROOT / rel).exists()
    for rel in retired:
        assert not (ROOT / rel).exists()


def test_histogram_imageitem_binding_is_centralized():
    adapter = (ROOT / "arrayscope" / "display" / "backends" / "pyqtgraph" / "histogram_adapter.py").read_text()
    image_view = (ROOT / "arrayscope" / "display" / "imageview2d.py").read_text()

    assert "def _bind_histogram_item" in image_view
    assert "PyQtGraphHistogramAdapter" in image_view
    assert ".setImageItem(" not in image_view
    assert "sigImageChanged" not in image_view
    assert "_setImageLookupTable" not in image_view
    assert "regionChanged()" not in image_view
    assert adapter.count(".setImageItem(") == 1
    assert "sigImageChanged" in adapter
    assert "_setImageLookupTable" in adapter
    assert "regionChanged()" in adapter


def test_display_surfaces_do_not_resurrect_normal_image_commit_entry_points():
    """The legacy normal-image single-quad path is deleted (2026-07).

    Every live 2D commit flows DisplayCommitter.commit_tile_layer ->
    present_tiled -> setTiledPresentation. The surfaces must not grow the
    legacy entry points back.
    """

    for rel in (
        Path("arrayscope/display/imageview2d.py"),
        Path("arrayscope/display/vispy_imageview2d.py"),
    ):
        text = (ROOT / rel).read_text()
        assert "def setImage(" not in text, rel
        assert "def updateImageDataFast(" not in text, rel


def test_frame_renderer_does_not_mutate_image_items_directly():
    text = (ROOT / "arrayscope" / "window" / "frame_controller.py").read_text()
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
        Path("arrayscope/core/diagnostics_jsonl.py"),
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


def test_operation_evaluator_owns_unified_display_cache():
    text = (ROOT / "arrayscope" / "operations" / "evaluator.py").read_text()
    assert "self._display_cache = BoundedArrayCache" in text
    assert ("self._" + "image_" + "cache = BoundedArrayCache") not in text
    assert ("self._" + "tile_" + "cache = BoundedArrayCache") not in text
    assert "self._profile_cache = BoundedArrayCache" in text


def test_scheduler_v2_pure_modules_are_qt_free():
    for rel in (
        Path("arrayscope/operations/chunked.py"),
    ):
        text = (ROOT / rel).read_text()
        assert "Qt" not in text
        assert "pyqtgraph" not in text


def test_frame_state_modules_are_qt_free():
    for rel in (
        Path("arrayscope/display/montage.py"),
        Path("arrayscope/display/geometry.py"),
        Path("arrayscope/window/frame_session.py"),
    ):
        text = (ROOT / rel).read_text()
        assert "pyqtgraph" not in text
        if rel != Path("arrayscope/window/frame_session.py"):
            assert "Qt" not in text


def test_update_image_view_does_not_batch_missing_regions():
    text = (ROOT / "arrayscope" / "window" / "frame_controller.py").read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "update_image_view":
            segment = ast.get_source_segment(text, node) or ""
            assert "tuple((tile, evaluate_image_snapshot" not in segment
            assert "for tile in missing_tiles)" not in segment
            assert "retarget_frame_pipeline" in segment
            return
    raise AssertionError("update_image_view not found")


def test_legacy_montage_tile_callbacks_are_deleted():
    text = (ROOT / "arrayscope" / "window" / "frame_controller.py").read_text()
    for name in (
        "_on_montage_tile_done",
        "_on_montage_tile_error",
        "_flush_montage_tile_results",
        "_apply_montage_tile_result",
        "_schedule_montage_tile_result_flush",
    ):
        assert f"def {name}" not in text
    session_text = (ROOT / "arrayscope" / "window" / "frame_session.py").read_text()
    assert "def mark_loaded" not in session_text


def test_frame_renderer_does_not_use_legacy_normal_render_decision_helper():
    text = (ROOT / "arrayscope" / "window" / "frame_controller.py").read_text()
    assert "choose_visible_render_decision" not in text
    assert "estimate_visible_render_context" not in text


def test_kernel_adapters_do_not_own_window_lane_quotas():
    text = (ROOT / "arrayscope" / "window" / "main.py").read_text()
    assert "self.visible_evaluation_controller = KernelEvaluationController(" in text
    assert "self.montage_tile_evaluation_controller = KernelEvaluationController(" in text
    assert "self.kernel.set_lane_quota(lane, workers)" in text
    assert "def _kernel_lane_for_compute_lane" in text
    assert text.count("apply_lane_quota=False") == 8
    assert 'name="visible"' in text


def test_frame_renderer_has_no_legacy_normal_degraded_preview_branch():
    text = (ROOT / "arrayscope" / "window" / "frame_controller.py").read_text()
    assert "RenderDecisionKind.DEGRADED_PREVIEW" not in text
    assert "store_display_tile_result" not in text


def test_deferred_single_shot_callbacks_carry_receiver_context():
    """QTimer.singleShot must use the (interval, receiver, callable) overload.

    A receiver context ties the callback to a QObject lifetime so it cannot
    fire into a destroyed window; generation guards alone only reject *stale*
    callbacks, not *dead* receivers (ADR 0045, roadmap Y1).
    """

    offenders = []
    for path in (ROOT / "arrayscope").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "singleShot"
                and len(node.args) < 3
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == []


def test_render_staleness_vocabulary_is_defined_once():
    """render_contract owns staleness; orchestration modules only delegate.

    No orchestration module may compare montage work tokens or session
    identities inline — the predicates live in window/render_contract.py
    (roadmap Y1 exit gate).
    """

    orchestration = (
        "window/render.py",
        "window/frame_controller.py",
        "window/display_presenter.py",
        "window/render_prefetch.py",
        "window/render_resources.py",
        "window/montage_prefetch.py",
    )
    for rel in orchestration:
        text = (ROOT / "arrayscope" / rel).read_text()
        assert "!= _montage_work_token(" not in text, rel
        assert "== _montage_work_token(" not in text, rel
        assert "session_id) == int(" not in text, rel
        assert "class RenderGeneration" not in text, rel
    contract = (ROOT / "arrayscope" / "window" / "render_contract.py").read_text()
    for predicate in (
        "def generation_is_current",
        "def session_is_current",
        "def session_token_is_current",
        "def montage_work_token",
        "def montage_work_token_is_current",
    ):
        assert predicate in contract


def test_view_state_sync_goes_through_the_binder():
    """Sync entry points delegate to ViewStateBinder; they do not enumerate
    widgets (roadmap Y3: adding a control means registering one binding)."""

    text = (ROOT / "arrayscope" / "window" / "state_sync.py").read_text()
    for name in ("_sync_controls_from_view_state", "_sync_slice_controls_immediately"):
        marker = f"def {name}"
        assert marker in text
        body = text.split(marker, 1)[1].split("\n    def ", 1)[0]
        assert "state_binder" in body
        assert "blockSignals" not in body
        assert "setChecked" not in body
        assert "setValue" not in body


def test_bounded_caches_share_the_core_eviction_implementation():
    """One eviction/priority implementation (roadmap Y3): every bounded cache
    builds on core.bounded_cache instead of hand-rolling an eviction loop."""

    core = (ROOT / "arrayscope" / "core" / "bounded_cache.py").read_text()
    assert "def _evict" in core
    for rel in (
        "operations/cache.py",
        "operations/stage_cache.py",
        "window/montage_payload_cache.py",
    ):
        text = (ROOT / "arrayscope" / rel).read_text()
        assert "bounded_cache import BoundedCache" in text, rel
        assert "def _evict" not in text, rel
        assert "popitem(" not in text, rel


def test_live_lod_modules_cannot_import_legacy_whole_plane_ownership():
    """ADR 0056 G5: live LOD ownership is canonical DataChunkKey pages only."""

    legacy_names = {"PyramidLevelKey", "PyramidCache"}
    live_modules = (
        "render/lod.py",
        "render/effects.py",
        "window/frame_effects.py",
        "window/frame_session.py",
        "window/montage_prefetch.py",
        "presentation/tile_lifecycle.py",
        "display/backends/vispy/tiles.py",
        "display/backends/pyqtgraph/tiles.py",
    )
    offenders = []
    for rel in live_modules:
        path = ROOT / "arrayscope" / rel
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = {alias.name for alias in node.names}
                for name in sorted(imported.intersection(legacy_names)):
                    offenders.append(f"{rel}:{node.lineno}:{name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.rsplit(".", 1)[-1] in legacy_names:
                        offenders.append(f"{rel}:{node.lineno}:{alias.name}")
    assert offenders == []

    pyramid_tree = ast.parse((ROOT / "arrayscope" / "display" / "pyramid.py").read_text())
    definitions = {
        node.name
        for node in pyramid_tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert definitions.isdisjoint(legacy_names)
