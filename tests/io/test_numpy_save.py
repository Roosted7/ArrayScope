import ast
import subprocess
import sys
from pathlib import Path

import numpy as np

from arrayscope.io.numpy_save import default_numpy_filename, selected_numpy_data


def test_default_numpy_filename_uses_source_stem_and_nii_gz_suffix():
    assert default_numpy_filename(None) == "arrayscope.npy"
    assert default_numpy_filename("/tmp/source.npy") == "source.npy"
    assert default_numpy_filename("/tmp/scan.nii.gz") == "scan.npy"


def test_selected_numpy_data_applies_ranges_and_optional_squeeze():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    squeezed = selected_numpy_data(data, [(1, 2), (0, 2), (1, 4)], squeeze=True)
    unsqueezed = selected_numpy_data(data, [(1, 2), (0, 2), (1, 4)], squeeze=False)

    np.testing.assert_array_equal(squeezed, data[1:2, 0:2, 1:4].squeeze())
    assert squeezed.shape == (2, 3)
    np.testing.assert_array_equal(unsqueezed, data[1:2, 0:2, 1:4])
    assert unsqueezed.shape == (1, 2, 3)


def test_numpy_save_helpers_have_no_qt_or_pyqtgraph_imports():
    path = Path(__file__).parents[2] / "arrayscope" / "io" / "numpy_save.py"
    tree = ast.parse(path.read_text())

    imported_roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.append(node.module.split(".")[0])

    assert "pyqtgraph" not in imported_roots
    assert not any(name.startswith(("PyQt", "PySide")) for name in imported_roots)


def test_importing_numpy_save_helpers_does_not_load_pyqtgraph():
    code = "import arrayscope.io.numpy_save, sys; print('pyqtgraph' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True)

    assert result.stdout.strip() == "False"
