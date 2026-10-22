"""Full document evaluation reads lazy sources through the shared read owner."""

from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.array_source import LazySourceArray, NdArraySource, SourceReadRefused
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT, ReverseAxis


class _ExplicitLazy(LazySourceArray):
    def __array__(self, *args, **kwargs):
        pytest.fail("full evaluation bypassed the explicit budgeted source-read owner")


class _Source:
    def __init__(self, data, *, on_read=None):
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.nbytes = data.nbytes
        self.chunk_shape = None
        self.label = "counted source"
        self.reads = []
        self.on_read = on_read

    def read_region(self, selection, *, cancellation_token=None):
        self.reads.append(tuple(selection))
        result = np.array(self.data[selection], copy=True)
        if self.on_read is not None:
            self.on_read()
        return result

    def close(self):
        pass


@pytest.mark.parametrize("kind", ["no-operations", "reverse", "fft", "optimized-away"])
def test_materialization_returns_values_after_one_explicit_full_read(kind):
    data = np.arange(4 * 7, dtype=np.float64).reshape(4, 7) * (1 + 2j)
    operations = {
        "no-operations": (),
        "reverse": (ReverseAxis(0),),
        "fft": (CenteredFFT(1),),
        "optimized-away": (ReverseAxis(0), ReverseAxis(0)),
    }[kind]
    source = _Source(data)

    result = ArrayDocument(_ExplicitLazy(source), operations=operations).materialize()

    expected = (
        data[::-1]
        if kind == "reverse"
        else np.fft.ifftshift(
            np.fft.ifft(np.fft.fftshift(data, axes=1), axis=1, norm="ortho"), axes=1
        )
        if kind == "fft"
        else data
    )
    assert isinstance(result, np.ndarray)
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert source.reads == [(slice(None), slice(None))]


@pytest.mark.parametrize("operations", [(), (ReverseAxis(0),)], ids=["no-operations", "reverse"])
def test_huge_virtual_full_source_is_refused_before_any_payload_read(operations):
    class VirtualSource:
        shape = (20_000, 20_000)
        dtype = np.dtype(np.complex128)
        nbytes = 6_400_000_000
        chunk_shape = None
        label = "virtual huge source"

        def read_region(self, *args, **kwargs):
            pytest.fail("over-budget materialization read source values")

    document = ArrayDocument(_ExplicitLazy(VirtualSource()), operations=operations)

    with pytest.raises(SourceReadRefused, match="lazy source read"):
        document.materialize()


@pytest.mark.parametrize("shape", [(), (0, 3)], ids=["scalar", "empty"])
def test_no_operation_materialization_preserves_scalar_and_empty_shapes(shape):
    data = np.zeros(shape, dtype=np.float32)
    source = _Source(data)

    result = ArrayDocument(_ExplicitLazy(source)).materialize()

    assert isinstance(result, np.ndarray)
    assert result.shape == shape
    assert result.dtype == data.dtype
    np.testing.assert_array_equal(result, data)
    assert len(source.reads) == 1


def test_eager_no_operation_materialization_preserves_its_existing_identity():
    data = np.arange(12).reshape(3, 4)
    assert ArrayDocument(data).materialize() is data


def test_already_cancelled_materialization_does_not_read_the_source():
    source = _Source(np.ones((3, 4)))
    context = SimpleNamespace(cancellation_token=SimpleNamespace(cancelled=True))

    with pytest.raises(EvaluationCancelled):
        ArrayDocument(_ExplicitLazy(source)).materialize(evaluation_context=context)

    assert source.reads == []


def test_cancellation_arriving_during_a_read_is_observed_before_operations():
    token = SimpleNamespace(cancelled=False)
    source = _Source(np.ones((3, 4)), on_read=lambda: setattr(token, "cancelled", True))
    context = SimpleNamespace(cancellation_token=token)

    with pytest.raises(EvaluationCancelled):
        ArrayDocument(_ExplicitLazy(source), operations=(ReverseAxis(0),)).materialize(
            evaluation_context=context
        )

    assert len(source.reads) == 1


def test_cancellation_arriving_in_the_last_operation_is_not_returned_as_success():
    token = SimpleNamespace(cancelled=False)

    class CancelOperation:
        def output_shape(self, shape):
            return tuple(shape)

        def output_dtype(self, dtype):
            return dtype

        def apply(self, data):
            token.cancelled = True
            return data

    context = SimpleNamespace(cancellation_token=token)
    document = ArrayDocument(np.ones((3, 4)), operations=(CancelOperation(),))

    with pytest.raises(EvaluationCancelled):
        document.materialize(evaluation_context=context)


@pytest.mark.parametrize("format", ["npy", "hdf5"])
def test_materialized_file_values_survive_source_close_and_file_removal(tmp_path, format):
    data = np.arange(3 * 7, dtype=np.float64).reshape(3, 7) * (1 + 2j)
    if format == "npy":
        path = tmp_path / "source.npy"
        np.save(path, data)
        mapping = np.load(path, mmap_mode="r", allow_pickle=False)
        source = _ExplicitLazy(NdArraySource(mapping, close=mapping._mmap.close))
    else:
        h5py = pytest.importorskip("h5py")
        from arrayscope.io.hdf5_source import open_hdf5_source

        path = tmp_path / "source.h5"
        with h5py.File(path, "w") as file:
            file.create_dataset("signal", data=data, chunks=(1, 7))
        source = open_hdf5_source(path, "signal")
    try:
        result = ArrayDocument(source, operations=(ReverseAxis(0), CenteredFFT(1))).materialize()
        assert isinstance(result, np.ndarray)
    finally:
        source.close()
    path.unlink()

    expected = np.fft.ifftshift(
        np.fft.ifft(np.fft.fftshift(data[::-1], axes=1), axis=1, norm="ortho"), axes=1
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
