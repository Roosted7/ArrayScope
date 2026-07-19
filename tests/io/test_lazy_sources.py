import numpy as np
import pytest

from arrayscope.core.array_source import LazySourceArray, SourceReadRefused
from arrayscope.io.file_interpreters import load_file, load_path
from arrayscope.io.lazy_sources import (
    lazy_load_threshold_bytes,
    open_memmap_source,
    open_scaled_nifti_source,
    should_load_lazily,
    supports_lazy_source,
    supports_memmap_source,
)


def _write_npy(tmp_path, data, name="data.npy"):
    path = tmp_path / name
    np.save(path, data)
    return path


def _write_cfl(tmp_path, data, name="data"):
    """Write a BART .cfl/.hdr pair (complex64, Fortran order)."""

    cfl_path = tmp_path / f"{name}.cfl"
    hdr_path = tmp_path / f"{name}.hdr"
    dims = list(data.shape) + [1] * (5 - data.ndim)
    hdr_path.write_text("# Dimensions\n" + " ".join(str(d) for d in dims) + "\n")
    np.asfortranarray(data.astype(np.complex64)).T.tofile(cfl_path)
    return cfl_path


def test_supports_memmap_source_suffixes():
    assert supports_memmap_source(".npy")
    assert supports_memmap_source(".CFL")
    assert not supports_memmap_source(".nii.gz")
    assert not supports_memmap_source(".txt")


def test_supports_lazy_source_includes_uncompressed_nifti():
    assert supports_lazy_source(".nii")
    assert supports_lazy_source(".npy")
    # compressed NIfTI cannot be mapped, so it is not a lazy candidate
    assert not supports_lazy_source(".nii.gz")
    assert not supports_lazy_source(".txt")


def _write_scaled_nifti(tmp_path, voxels, slope, inter, name="scan.nii"):
    nib = pytest.importorskip("nibabel")
    image = nib.Nifti1Image(voxels, np.eye(4))
    image.header.set_data_dtype(voxels.dtype)
    image.header["scl_slope"] = slope
    image.header["scl_inter"] = inter
    path = tmp_path / name
    nib.save(image, path)
    return path


def test_open_scaled_nifti_source_maps_int16_and_rescales(tmp_path):
    voxels = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
    path = _write_scaled_nifti(tmp_path, voxels, slope=100.0, inter=7.0)

    source, _axes = open_scaled_nifti_source(path)

    assert isinstance(source, LazySourceArray)
    assert source.dtype == np.float32
    assert source.shape == voxels.shape
    # The backing store stays int16 (2 bytes); nbytes reports the float32 cost.
    assert source.nbytes == voxels.size * 4
    expected = voxels.astype(np.float64) * 100.0 + 7.0
    region = source.read_region((slice(1, 3), 2, slice(None)))
    assert region.dtype == np.float32
    np.testing.assert_allclose(region, expected[1:3, 2, :], rtol=1e-5)
    source.close()


def test_load_path_lazy_nifti_matches_eager_and_carries_axes(tmp_path):
    pytest.importorskip("nibabel")
    voxels = np.arange(3 * 4 * 5, dtype=np.int16).reshape(3, 4, 5)
    path = _write_scaled_nifti(tmp_path, voxels, slope=50.0, inter=0.0)

    lazy = load_path(path, lazy=True)
    eager = load_path(path, lazy=False)

    assert isinstance(lazy.data, LazySourceArray)
    assert lazy.metadata["lazy"] is True
    assert lazy.metadata["detected_format"] == "nifti"
    assert isinstance(eager.data, np.ndarray)
    assert eager.data.dtype == np.float32
    # Eager and lazy share the same scaled read path — values are identical.
    np.testing.assert_array_equal(np.asarray(lazy.data), eager.data)
    # Axis metadata survives the lazy path.
    assert lazy.axes is not None
    assert tuple(axis.size for axis in lazy.axes) == voxels.shape


def test_complex_nifti_keeps_imaginary_part_as_complex64(tmp_path):
    nib = pytest.importorskip("nibabel")
    voxels = (np.arange(24, dtype=np.float32) + 1j * np.arange(24, dtype=np.float32)[::-1]).reshape(2, 3, 4).astype(np.complex64)
    image = nib.Nifti1Image(voxels, np.eye(4))
    image.header.set_data_dtype(np.complex64)
    path = tmp_path / "complex.nii"
    nib.save(image, path)

    eager = load_path(path, lazy=False)
    lazy = load_path(path, lazy=True)

    assert eager.data.dtype == np.complex64
    np.testing.assert_array_equal(eager.data, voxels)
    assert lazy.data.dtype == np.complex64
    np.testing.assert_array_equal(np.asarray(lazy.data), voxels)


def test_compressed_nifti_falls_back_to_eager(tmp_path):
    nib = pytest.importorskip("nibabel")
    voxels = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    image = nib.Nifti1Image(voxels, np.eye(4))
    image.header.set_data_dtype(np.int16)
    path = tmp_path / "scan.nii.gz"
    nib.save(image, path)

    loaded = load_path(path, lazy="auto", lazy_threshold_bytes=1)

    # .nii.gz cannot be mapped; auto stays eager (and still narrows to float32).
    assert isinstance(loaded.data, np.ndarray)
    assert loaded.data.dtype == np.float32


def test_lazy_load_threshold_scales_with_available_memory():
    assert lazy_load_threshold_bytes(system_available_bytes=64 * 1024**3) == 16 * 1024**3
    # tiny systems still keep a sane floor
    assert lazy_load_threshold_bytes(system_available_bytes=0) == 64 * 1024 * 1024


def test_should_load_lazily_thresholds_and_overrides(tmp_path):
    path = _write_npy(tmp_path, np.zeros((32, 32), dtype=np.float32))

    assert should_load_lazily(path, lazy=True)
    assert not should_load_lazily(path, lazy=False)
    assert should_load_lazily(path, lazy="auto", threshold_bytes=1)
    assert not should_load_lazily(path, lazy="auto", threshold_bytes=10**9)
    with pytest.raises(ValueError):
        should_load_lazily(path, lazy="sometimes")


def test_open_memmap_source_npy_matches_eager(tmp_path):
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    source = open_memmap_source(_write_npy(tmp_path, data))

    assert isinstance(source, LazySourceArray)
    assert source.shape == data.shape
    assert source.dtype == data.dtype
    assert np.array_equal(source.read_region((slice(1, 3), 2, slice(None))), data[1:3, 2, :])
    assert np.array_equal(source.materialize(), data)
    source.close()


def test_open_memmap_source_cfl_matches_eager_loader(tmp_path):
    from arrayscope.io.file_interpreters import BartLoader

    data = (
        (np.arange(3 * 4 * 2) + 1j * np.arange(3 * 4 * 2)[::-1])
        .reshape(3, 4, 2)
        .astype(np.complex64)
    )
    cfl_path = _write_cfl(tmp_path, data)

    eager = BartLoader(cfl_path).load()
    source = open_memmap_source(cfl_path)

    assert source.shape == eager.shape
    assert source.dtype == np.complex64
    assert np.array_equal(source.materialize(), eager)


def test_open_memmap_source_rejects_unsupported_suffix(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("1 2 3\n")
    with pytest.raises(ValueError):
        open_memmap_source(path)


def test_open_memmap_source_rejects_object_arrays(tmp_path):
    path = tmp_path / "objects.npy"
    np.save(path, np.array([{"a": 1}, None], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        open_memmap_source(path)


def test_load_path_auto_returns_lazy_source_above_threshold(tmp_path):
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    path = _write_npy(tmp_path, data)

    loaded = load_path(path, lazy="auto", lazy_threshold_bytes=1)

    assert isinstance(loaded.data, LazySourceArray)
    assert loaded.metadata["lazy"] is True
    assert loaded.metadata["detected_format"] == "numpy"
    assert loaded.metadata["shape"] == data.shape
    assert loaded.metadata["dtype"] == str(data.dtype)
    assert np.array_equal(np.asarray(loaded.data), data)


def test_load_path_auto_stays_eager_below_threshold(tmp_path):
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    loaded = load_path(_write_npy(tmp_path, data), lazy="auto", lazy_threshold_bytes=10**9)

    assert isinstance(loaded.data, np.ndarray)
    assert "lazy" not in loaded.metadata
    assert np.array_equal(loaded.data, data)


def test_load_path_lazy_false_never_maps(tmp_path):
    data = np.zeros((8, 8), dtype=np.float32)
    loaded = load_path(_write_npy(tmp_path, data), lazy=False, lazy_threshold_bytes=1)

    assert isinstance(loaded.data, np.ndarray)


def test_load_path_rejects_pickled_npy_in_any_mode(tmp_path):
    """Object arrays are rejected both lazily (unmappable) and eagerly (allow_pickle=False)."""

    path = tmp_path / "objects.npy"
    np.save(path, np.array([{"a": 1}, {"b": 2}], dtype=object), allow_pickle=True)

    with pytest.raises(ValueError):
        load_path(path, lazy=True)
    with pytest.raises(ValueError):
        load_path(path, lazy="auto", lazy_threshold_bytes=1)


def test_load_path_lazy_cfl_metadata(tmp_path):
    data = np.ones((4, 4), dtype=np.complex64)
    cfl_path = _write_cfl(tmp_path, data)

    loaded = load_path(cfl_path, lazy=True)

    assert isinstance(loaded.data, LazySourceArray)
    assert loaded.metadata["detected_format"] == "cfl"
    assert loaded.metadata["lazy"] is True


def test_load_file_passes_lazy_through(tmp_path):
    data = np.zeros((8, 8), dtype=np.float32)
    path = _write_npy(tmp_path, data)

    assert isinstance(load_file(path, lazy=False), np.ndarray)
    assert isinstance(load_file(path, lazy=True), LazySourceArray)


def test_lazy_source_refuses_unbounded_materialization(tmp_path):
    data = np.zeros((64, 64), dtype=np.float64)
    source = open_memmap_source(_write_npy(tmp_path, data))
    source._materialize_budget_bytes = 128

    with pytest.raises(SourceReadRefused):
        np.asarray(source)
