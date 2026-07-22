"""Correctness tests for :class:`CompositeArraySource` (Compare v1c: A - B).

Every read is checked against a plain NumPy oracle ``(A_np - B_np)[spec]``.
Spy sources prove the composite reads only the requested region and forwards
the cancellation token to both sub-reads.
"""

import numpy as np
import pytest

from arrayscope.core.array_source import (
    ArraySource,
    CompositeArraySource,
    NdArraySource,
)
from arrayscope.io.progressive import ProgressiveArraySource


class SpyArraySource(NdArraySource):
    """Record every ``index_spec`` and ``cancellation_token`` a read is asked for."""

    def __init__(self, array, *, label="spy"):
        super().__init__(array, label=label)
        self.read_specs = []
        self.read_tokens = []

    def read_region(self, index_spec, *, cancellation_token=None):
        self.read_specs.append(tuple(index_spec))
        self.read_tokens.append(cancellation_token)
        return super().read_region(index_spec, cancellation_token=cancellation_token)


def _oracle(a_np, b_np):
    return a_np - b_np


# --- shape / dtype / ndim ---------------------------------------------------


def test_is_an_array_source_with_broadcast_free_metadata():
    a = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    b = np.ones((4, 5), dtype=np.float32)
    composite = CompositeArraySource(a, b)

    assert isinstance(composite, ArraySource)
    assert composite.shape == (4, 5)
    assert composite.ndim == 2
    assert composite.dtype == np.float32
    assert composite.nbytes == 4 * 5 * np.dtype(np.float32).itemsize
    assert composite.op == "subtract"


def test_complex_stays_complex():
    a = (np.arange(6, dtype=np.complex64).reshape(2, 3) + 1j).astype(np.complex64)
    b = (np.ones((2, 3)) * (2 - 3j)).astype(np.complex64)
    composite = CompositeArraySource(a, b)

    assert composite.dtype == np.complex64


def test_dtype_promotes_like_numpy_result_type():
    a = np.ones((3, 3), dtype=np.float32)
    b = np.ones((3, 3), dtype=np.float64)
    composite = CompositeArraySource(a, b)

    assert composite.dtype == np.result_type(np.float32, np.float64)


# --- read_region equals the NumPy oracle exactly ----------------------------


def _index_specs():
    return [
        (slice(None), slice(None)),  # full
        (slice(1, 3), slice(0, 4)),  # sub-rectangle
        (2, slice(None)),  # single index drops an axis
        (slice(None), 3),
        (1, 2),  # scalar
        (slice(None, None, 2), slice(1, None, 2)),  # stepped
        (slice(None, None, -1), slice(None)),  # negative step
        ((3, 0, 2), slice(None)),  # fancy per-axis
    ]


@pytest.mark.parametrize("spec", _index_specs())
def test_read_region_matches_oracle_float(spec):
    a_np = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    b_np = (np.arange(4 * 5, dtype=np.float32).reshape(4, 5) * 0.5) + 3.0
    composite = CompositeArraySource(a_np, b_np)

    np.testing.assert_array_equal(composite.read_region(spec), _oracle(a_np, b_np)[spec])


@pytest.mark.parametrize("spec", _index_specs())
def test_read_region_matches_oracle_complex(spec):
    rng = np.random.default_rng(0)
    a_np = (rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))).astype(np.complex64)
    b_np = (rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))).astype(np.complex64)
    composite = CompositeArraySource(a_np, b_np)

    result = composite.read_region(spec)
    expected = _oracle(a_np, b_np)[spec]
    assert result.dtype == np.complex64
    np.testing.assert_array_equal(result.real, expected.real)
    np.testing.assert_array_equal(result.imag, expected.imag)


# --- region-only: the composite reads exactly the requested spec ------------


def test_reads_only_the_requested_region_from_both_sources():
    a_np = np.arange(6 * 7, dtype=np.float32).reshape(6, 7)
    b_np = np.ones((6, 7), dtype=np.float32)
    spy_a = SpyArraySource(a_np, label="A")
    spy_b = SpyArraySource(b_np, label="B")
    composite = CompositeArraySource(spy_a, spy_b)

    spec = (slice(2, 4), slice(1, 5))
    result = composite.read_region(spec)

    # Each sub-source was asked for exactly the requested spec, once, and never
    # for the whole array.
    assert spy_a.read_specs == [spec]
    assert spy_b.read_specs == [spec]
    np.testing.assert_array_equal(result, (a_np - b_np)[spec])


# --- cancellation token is forwarded to both sub-reads ----------------------


def test_cancellation_token_forwarded_to_both_sources():
    spy_a = SpyArraySource(np.zeros((3, 3), dtype=np.float32), label="A")
    spy_b = SpyArraySource(np.zeros((3, 3), dtype=np.float32), label="B")
    composite = CompositeArraySource(spy_a, spy_b)

    token = object()
    composite.read_region((slice(None), slice(None)), cancellation_token=token)

    assert spy_a.read_tokens == [token]
    assert spy_b.read_tokens == [token]


# --- construction guard fires on incompatible shapes ------------------------


def test_rejects_incompatible_shapes_with_both_shapes_in_message():
    a = np.zeros((4, 5), dtype=np.float32)
    b = np.zeros((4, 6), dtype=np.float32)

    with pytest.raises(ValueError) as excinfo:
        CompositeArraySource(a, b)

    message = str(excinfo.value)
    assert "(4, 5)" in message
    assert "(4, 6)" in message


def test_rejects_unknown_op():
    a = np.zeros((2, 2))
    with pytest.raises(ValueError):
        CompositeArraySource(a, a, op="frobnicate")


# --- close ownership --------------------------------------------------------


class ClosableSpySource(NdArraySource):
    """Record whether ``close`` was called on this input source."""

    def __init__(self, array, *, label="closable"):
        self.closed = False
        super().__init__(array, label=label, close=self._on_close)

    def _on_close(self):
        self.closed = True


def test_default_owns_and_closes_both_inputs():
    a = ClosableSpySource(np.zeros((2, 2), dtype=np.float32), label="A")
    b = ClosableSpySource(np.zeros((2, 2), dtype=np.float32), label="B")
    composite = CompositeArraySource(a, b)

    composite.close()

    assert a.closed
    assert b.closed


def test_own_inputs_false_does_not_close_shared_inputs():
    # The difference-window case: A and B stay live in their own windows, so
    # closing the derived source must NOT tear their sources down.
    a = ClosableSpySource(np.zeros((2, 2), dtype=np.float32), label="A")
    b = ClosableSpySource(np.zeros((2, 2), dtype=np.float32), label="B")
    composite = CompositeArraySource(a, b, own_inputs=False)

    composite.close()

    assert not a.closed
    assert not b.closed
    # A and B remain readable after the composite is closed.
    np.testing.assert_array_equal(a.read_region((slice(None), slice(None))), np.zeros((2, 2)))
    np.testing.assert_array_equal(b.read_region((slice(None), slice(None))), np.zeros((2, 2)))


# --- a progressive/lazy input streams through the composite -----------------


def test_progressive_input_streams_written_region():
    a_np = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    b_np = np.ones((4, 5), dtype=np.float32)

    # A starts zero-filled and is written region by region, like a mid-fill load.
    backing = np.zeros((4, 5), dtype=np.float32)
    progressive = ProgressiveArraySource(backing, label="A")
    composite = CompositeArraySource(progressive, NdArraySource(b_np))

    spec = (slice(1, 3), slice(None))

    # Before A's region is written it reads as zero -> 0 - B.
    np.testing.assert_array_equal(composite.read_region(spec), (np.zeros((4, 5)) - b_np)[spec])

    # Write the rows the region covers; the composite now reflects A - B there.
    with progressive.write_transaction() as arr:
        arr[1:3, :] = a_np[1:3, :]

    np.testing.assert_array_equal(composite.read_region(spec), (a_np - b_np)[spec])


def test_progressive_read_is_detached_from_backing():
    backing = np.arange(9, dtype=np.float32).reshape(3, 3)
    progressive = ProgressiveArraySource(backing, label="A")
    composite = CompositeArraySource(progressive, NdArraySource(np.zeros((3, 3), dtype=np.float32)))

    region = composite.read_region((slice(None), slice(None)))
    # The op materializes a fresh array; mutating the backing must not leak in.
    assert region.base is None
    backing[:] = 999.0
    np.testing.assert_array_equal(region, np.arange(9, dtype=np.float32).reshape(3, 3))
