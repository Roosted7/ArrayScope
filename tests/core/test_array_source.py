import numpy as np
import pytest

from arrayscope.core.array_source import (
    ArraySource,
    LazySourceArray,
    NdArraySource,
    SourceReadRefused,
    is_lazy_source_array,
    read_index_spec,
)


def _memmap(tmp_path, data):
    path = tmp_path / "backing.npy"
    np.save(path, data)
    return np.load(path, mmap_mode="r")


def test_read_index_spec_matches_numpy_indexing():
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)

    assert np.array_equal(read_index_spec(data, (1, slice(None), slice(0, 3))), data[1, :, 0:3])
    assert np.array_equal(read_index_spec(data, (slice(None), 2, slice(None))), data[:, 2, :])
    assert read_index_spec(data, (1, 2, 3)) == data[1, 2, 3]


def test_read_index_spec_handles_index_tuples_per_axis():
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)

    result = read_index_spec(data, ((3, 0, 2), 1, slice(1, 5, 2)))

    expected = np.take(data, (3, 0, 2), axis=0)[:, 1, 1:5:2]
    assert np.array_equal(result, expected)


def test_read_index_spec_rejects_wrong_length_and_unknown_items():
    data = np.zeros((2, 3))
    with pytest.raises(ValueError):
        read_index_spec(data, (0,))
    with pytest.raises(TypeError):
        read_index_spec(data, (0, object()))


def test_ndarray_source_exposes_shape_dtype_nbytes_label():
    data = np.arange(12, dtype=np.complex64).reshape(3, 4)
    source = NdArraySource(data, label="test-source", chunk_shape=(1, 4))

    assert isinstance(source, ArraySource)
    assert source.shape == (3, 4)
    assert source.dtype == np.complex64
    assert source.nbytes == data.nbytes
    assert source.chunk_shape == (1, 4)
    assert source.label == "test-source"


def test_ndarray_source_read_region_detaches_from_memmap(tmp_path):
    data = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    mapped = _memmap(tmp_path, data)
    source = NdArraySource(mapped)

    result = source.read_region((slice(1, 3), slice(None)))

    assert not isinstance(result, np.memmap)
    assert result.base is None or not isinstance(result.base, np.memmap)
    assert np.array_equal(result, data[1:3, :])


def test_ndarray_source_close_invokes_hook_once():
    closes = []
    source = NdArraySource(np.zeros((2, 2)), close=lambda: closes.append(1))
    source.close()
    source.close()
    assert closes == [1]


def test_lazy_source_array_mirrors_source_metadata():
    data = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    lazy = LazySourceArray(NdArraySource(data, label="lazy"))

    assert is_lazy_source_array(lazy)
    assert not is_lazy_source_array(data)
    assert lazy.shape == (2, 3, 4)
    assert np.shape(lazy) == (2, 3, 4)
    assert lazy.dtype == np.float64
    assert lazy.ndim == 3
    assert lazy.size == 24
    assert lazy.nbytes == data.nbytes
    assert len(lazy) == 2
    assert "lazy" in repr(lazy)


def test_lazy_source_array_read_region_delegates():
    data = np.arange(24, dtype=np.float32).reshape(4, 6)
    lazy = LazySourceArray(NdArraySource(data))

    assert np.array_equal(lazy.read_region((2, slice(None))), data[2, :])


def test_lazy_source_array_materialize_within_budget():
    data = np.arange(24, dtype=np.float32).reshape(4, 6)
    lazy = LazySourceArray(NdArraySource(data))

    assert np.array_equal(lazy.materialize(), data)
    assert np.array_equal(np.asarray(lazy), data)
    assert np.asarray(lazy, dtype=np.float64).dtype == np.float64


def test_lazy_source_array_refuses_materialization_over_budget():
    data = np.zeros((32, 32), dtype=np.float64)
    lazy = LazySourceArray(NdArraySource(data), materialize_budget_bytes=64)

    with pytest.raises(SourceReadRefused) as excinfo:
        np.asarray(lazy)

    assert excinfo.value.requested_nbytes == data.nbytes
    assert excinfo.value.budget_bytes == 64
    assert np.array_equal(lazy.materialize(budget_bytes=None), data)
