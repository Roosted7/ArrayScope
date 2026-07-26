"""Structural resurrection guards for the ADR 0056 G5 cutover."""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_render_and_gate_paths_cannot_resurrect_frame_session_pending_queue():
    """Target lifecycle is the sole live scheduling-debt authority."""

    render_and_gate_paths = (
        ROOT / "arrayscope" / "window" / "frame_session.py",
        ROOT / "arrayscope" / "window" / "frame_controller.py",
        ROOT / "arrayscope" / "window" / "frame_effects.py",
        ROOT / "arrayscope" / "window" / "frame_runtime.py",
        ROOT / "arrayscope" / "window" / "main.py",
        ROOT / "arrayscope" / "window" / "montage_prefetch.py",
        ROOT / "arrayscope" / "window" / "render_resources.py",
        ROOT / "arrayscope" / "render" / "lod.py",
        ROOT / "arrayscope" / "tools" / "profile_montage_workflow.py",
        ROOT / "tools" / "probes" / "profile_cached_rebuild.py",
        ROOT / "tools" / "probes" / "verify_scrub_fastpath.py",
        ROOT / "tools" / "probes" / "verify_stale.py",
        ROOT / "tools" / "ui_gallery.py",
        ROOT / "tools" / "demo_recorder.py",
    )
    forbidden_helpers = {
        "next_tile",
        "pending_tile_numbers",
        "enqueue_pending_tile",
        "discard_pending_tile",
        "prune_pending_tiles",
        "requeue_orphaned_loading_tiles",
        "release_display_owned_pending",
        "enqueue_stage_dependent_tiles",
        "_classify_visible_montage_tiles",
        "_enqueue_session_pending_tile",
    }
    violations: list[str] = []
    for path in render_and_gate_paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "pending_tiles":
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: .pending_tiles")
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in forbidden_helpers
            ):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.name}")
            if (
                path.name == "frame_session.py"
                and isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "pending_tiles"
            ):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: pending_tiles field")
    assert not violations, "live duplicate scheduling owner resurrected:\n" + "\n".join(violations)
