"""Structural resurrection guards for the ADR 0056 G5 cutover."""

import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_vispy_cannot_infer_reduced_page_keys():
    """Reduced page identity belongs only to the source-grid planner."""

    # VisPy may still partition native L0 source planes for the shift-reuse
    # seam. A backend-local mean/reduction inference would resurrect the
    # deleted window-origin identity path.
    vispy_path = ROOT / "arrayscope" / "display" / "backends" / "vispy" / "tiles.py"
    vispy_text = vispy_path.read_text()
    vispy_tree = ast.parse(vispy_text)
    key_builders = [
        node
        for node in vispy_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_data_chunk_key"
    ]
    assert len(key_builders) == 1
    key_builder = key_builders[0]
    assert "lod" not in {argument.arg for argument in key_builder.args.kwonlyargs}
    assert "REDUCER_MEAN" not in {
        node.id for node in ast.walk(key_builder) if isinstance(node, ast.Name)
    }
    assert vispy_text.count("DataChunkKey(") == 1
