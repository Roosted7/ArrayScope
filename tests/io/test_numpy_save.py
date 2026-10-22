import ast
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from arrayscope.io.numpy_save import (
    default_numpy_filename,
    save_derived_array,
    selected_numpy_data,
)


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


def test_derived_array_export_writes_npy_sidecars(tmp_path):
    data = np.arange(6).reshape(2, 3)
    written = save_derived_array(
        tmp_path / "derived.npy",
        data,
        recipe_json='{"recipe": true}',
        view_recipe_json='{"view": true}',
    )

    assert [path.name for path in written] == [
        "derived.npy",
        "derived.recipe.json",
        "derived.view.json",
    ]
    np.testing.assert_array_equal(np.load(written[0]), data)
    assert written[1].read_text(encoding="utf-8").strip() == '{"recipe": true}'
    assert written[2].read_text(encoding="utf-8").strip() == '{"view": true}'


def test_derived_array_export_writes_npz_payload(tmp_path):
    data = np.arange(4)
    written = save_derived_array(
        tmp_path / "derived.npz", data, recipe_json="recipe", view_recipe_json="view"
    )

    with np.load(written[0]) as archive:
        np.testing.assert_array_equal(archive["array"], data)
        assert str(archive["recipe_json"]) == "recipe"
        assert str(archive["view_recipe_json"]) == "view"


@pytest.mark.parametrize("suffix", [".NPY", ".NPZ"])
def test_numpy_export_preserves_case_sensitive_destination(tmp_path, suffix):
    path = tmp_path / f"result{suffix}"
    data = np.arange(12).reshape(3, 4)

    assert save_derived_array(path, data) == (path,)

    assert set(tmp_path.iterdir()) == {path}
    if suffix == ".NPZ":
        with np.load(path) as archive:
            np.testing.assert_array_equal(archive["array"], data)
    else:
        np.testing.assert_array_equal(np.load(path), data)


@pytest.mark.parametrize(("suffix", "writer"), [(".npy", "save"), (".npz", "savez")])
def test_numpy_export_failure_preserves_previous_file_and_cleans_temporaries(
    tmp_path, monkeypatch, suffix, writer
):
    path = tmp_path / f"result{suffix}"
    path.write_bytes(b"previous export")

    def fail_after_partial_write(destination, *_args, **_kwargs):
        if hasattr(destination, "write"):
            destination.write(b"partial new export")
        else:
            Path(destination).write_bytes(b"partial new export")
        raise OSError("disk full")

    monkeypatch.setattr(np, writer, fail_after_partial_write)
    with pytest.raises(OSError, match="disk full"):
        save_derived_array(path, np.arange(10))

    assert path.read_bytes() == b"previous export"
    assert set(tmp_path.iterdir()) == {path}


def test_sidecar_preparation_failure_does_not_publish_new_array(tmp_path):
    path = tmp_path / "result.npy"
    recipe_path = tmp_path / "result.recipe.json"
    view_path = tmp_path / "result.view.json"
    np.save(path, np.arange(3))
    original = path.read_bytes()
    recipe_path.write_text("old recipe", encoding="utf-8")
    view_path.mkdir()

    with pytest.raises(IsADirectoryError):
        save_derived_array(path, np.arange(10), recipe_json="new recipe", view_recipe_json="view")

    assert path.read_bytes() == original
    assert recipe_path.read_text(encoding="utf-8") == "old recipe"
    assert set(tmp_path.iterdir()) == {path, recipe_path, view_path}


@pytest.mark.skipif(os.name == "nt", reason="Windows refuses replacing an open mapped file")
def test_numpy_export_can_replace_its_mapped_source_without_corrupting_it(tmp_path):
    # A direct np.save(path, mapped_view) truncates the file before reading the
    # mapping; depending on page size that either corrupts values or SIGBUSes.
    # Keep that dangerous regression in a child process.
    path = tmp_path / "source.npy"
    data = np.arange(100_000, dtype=np.int64)
    np.save(path, data)
    code = """
import sys
import numpy as np
from arrayscope.io.numpy_save import save_derived_array
path = sys.argv[1]
mapped = np.load(path, mmap_mode='r')
expected = mapped[::2].copy()
save_derived_array(path, mapped[::2])
np.testing.assert_array_equal(np.load(path), expected)
np.testing.assert_array_equal(mapped, np.arange(100_000, dtype=np.int64))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(np.load(path), data[::2])


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission modes and symlinks")
def test_numpy_export_follows_symlink_and_preserves_existing_mode(tmp_path):
    target = tmp_path / "actual.npy"
    link = tmp_path / "linked.npy"
    np.save(target, np.arange(3))
    target.chmod(0o640)
    link.symlink_to(target.name)
    data = np.arange(6)

    assert save_derived_array(link, data) == (link,)

    assert link.is_symlink()
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    np.testing.assert_array_equal(np.load(target), data)
    assert set(tmp_path.iterdir()) == {target, link}


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission modes")
def test_numpy_export_does_not_replace_read_only_file(tmp_path):
    path = tmp_path / "protected.npy"
    original = b"previous export"
    path.write_bytes(original)
    path.chmod(0o440)
    if os.access(path, os.W_OK):
        pytest.skip("Current user can write despite read-only mode")

    with pytest.raises(PermissionError):
        save_derived_array(path, np.arange(10))

    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
def test_numpy_export_rejects_sidecar_linked_to_the_array(tmp_path):
    path = tmp_path / "result.npy"
    recipe_path = path.with_suffix(".recipe.json")
    np.save(path, np.arange(3))
    original = path.read_bytes()
    recipe_path.symlink_to(path.name)

    with pytest.raises(ValueError, match="different files"):
        save_derived_array(path, np.arange(10), recipe_json="new recipe")

    assert path.read_bytes() == original
    assert recipe_path.is_symlink()


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
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, text=True, capture_output=True
    )

    assert result.stdout.strip() == "False"
