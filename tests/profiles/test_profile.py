import ast
from pathlib import Path

from arrayscope.core.view_state import ViewState
from arrayscope.profiles import model as profile


ROOT = Path(__file__).parents[2]


def state_for(shape, image_axes=(0, 1), line_axis=2, slices=None):
    return ViewState(
        ndim=len(shape),
        shape=tuple(shape),
        image_axes=image_axes,
        line_axis=line_axis,
        slice_indices=tuple(slices if slices is not None else (0,) * len(shape)),
        axis_flipped=(False,) * len(shape),
        axis_fftshifted=(False,) * len(shape),
    )


def test_profile_y_range_matches_image_window_only_when_requested():
    assert profile.profile_y_range("match_image", (1, 5)) == (1.0, 5.0)
    assert profile.profile_y_range("auto", (1, 5)) is None
    assert profile.profile_y_range("match_image", None) is None


def test_profile_module_has_no_qt_or_pyqtgraph_imports():
    tree = ast.parse((ROOT / "arrayscope" / "profiles" / "model.py").read_text())

    imported_roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.append(node.module.split(".")[0])

    assert "pyqtgraph" not in imported_roots
    assert not any(name.startswith(("PyQt", "PySide")) for name in imported_roots)

