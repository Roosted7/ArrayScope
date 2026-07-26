"""Measure plausible fused Numba kernels against Bundle A's NumPy references.

The default shape is the repository's representative MRI volume,
``336 x 336 x 272``.  Results separate the first Numba call (import/JIT plus
execution) from warmed medians; the script is evidence, not a production
accelerator module.

Example:

    conda run -n arrayscope python tools/benchmark_native_ops_numba.py \
        --repeats 5 --json /tmp/native-ops-numba.json
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _numba_log_magnitude(data, epsilon):
    source = data.reshape(-1)
    output = np.empty(source.size, dtype=np.float32)
    for index in prange(source.size):
        magnitude = abs(source[index])
        output[index] = np.log(max(magnitude, epsilon))
    return output.reshape(data.shape)


@njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _numba_soft_threshold(data, threshold):
    source = data.reshape(-1)
    output = np.empty_like(source)
    for index in prange(source.size):
        value = source[index]
        magnitude = abs(value)
        if magnitude > threshold:
            output[index] = value * ((magnitude - threshold) / magnitude)
        else:
            output[index] = 0
    return output.reshape(data.shape)


@njit(cache=True, nogil=True, parallel=True, fastmath=False)
def _numba_normalize_last_axis(data):
    rows = data.reshape((-1, data.shape[-1]))
    output = np.empty_like(rows)
    for row in prange(rows.shape[0]):
        norm_squared = np.float32(0.0)
        for column in range(rows.shape[1]):
            value = rows[row, column]
            magnitude = abs(value)
            norm_squared += magnitude * magnitude
        norm = np.sqrt(norm_squared)
        if norm == 0:
            for column in range(rows.shape[1]):
                output[row, column] = 0
        else:
            for column in range(rows.shape[1]):
                output[row, column] = rows[row, column] / norm
    return output.reshape(data.shape)


def _numpy_log_magnitude(data, epsilon):
    return np.log(np.maximum(np.abs(data), np.float32(epsilon))).astype(np.float32, copy=False)


def _numpy_soft_threshold(data, threshold):
    magnitude = np.abs(data)
    scale = np.zeros_like(magnitude)
    np.divide(
        np.maximum(magnitude - threshold, 0),
        magnitude,
        out=scale,
        where=magnitude != 0,
    )
    return (data * scale).astype(data.dtype, copy=False)


def _numpy_normalize_last_axis(data):
    norm = np.sqrt(np.sum(np.abs(data) ** 2, axis=-1, keepdims=True))
    output = np.zeros_like(data)
    np.divide(data, norm, out=output, where=norm != 0)
    return output


def _numpy_normalize_middle_axis(data):
    norm = np.sqrt(np.sum(np.abs(data) ** 2, axis=1, keepdims=True))
    output = np.zeros_like(data)
    np.divide(data, norm, out=output, where=norm != 0)
    return output


def _numba_normalize_middle_axis_with_copy(data):
    moved = np.ascontiguousarray(np.moveaxis(data, 1, -1))
    result = _numba_normalize_last_axis(moved)
    return np.moveaxis(result, -1, 1)


def _time_once(function, *args):
    gc.collect()
    started = time.perf_counter()
    result = function(*args)
    elapsed = time.perf_counter() - started
    return elapsed, result


def _measure_candidate(name, numpy_fn, numba_fn, args, *, repeats):
    numpy_times = []
    numpy_result = None
    for _ in range(repeats):
        elapsed, numpy_result = _time_once(numpy_fn, *args)
        numpy_times.append(elapsed)

    numba_first, numba_result = _time_once(numba_fn, *args)
    np.testing.assert_allclose(numba_result, numpy_result, rtol=2e-5, atol=2e-6)

    numba_times = []
    for _ in range(repeats):
        elapsed, numba_result = _time_once(numba_fn, *args)
        numba_times.append(elapsed)
    np.testing.assert_allclose(numba_result, numpy_result, rtol=2e-5, atol=2e-6)

    numpy_median = float(np.median(numpy_times))
    numba_median = float(np.median(numba_times))
    return {
        "candidate": name,
        "numpy_seconds": numpy_times,
        "numpy_median_seconds": numpy_median,
        "numba_first_seconds": float(numba_first),
        "numba_warm_seconds": numba_times,
        "numba_warm_median_seconds": numba_median,
        "warm_speedup": numpy_median / numba_median,
        "estimated_jit_overhead_seconds": max(0.0, float(numba_first) - numba_median),
    }


def _sample(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(shape, dtype=np.float32)
    if np.dtype(dtype).kind == "c":
        imag = rng.standard_normal(shape, dtype=np.float32)
        return (real + 1j * imag).astype(dtype)
    return real.astype(dtype, copy=False)


def run(shape, repeats):
    results = []
    for dtype in (np.float32, np.complex64):
        data = _sample(shape, dtype, seed=20260726)
        candidates = (
            (
                "log_magnitude",
                _numpy_log_magnitude,
                _numba_log_magnitude,
                (data, np.float32(1e-6)),
            ),
            (
                "soft_threshold",
                _numpy_soft_threshold,
                _numba_soft_threshold,
                (data, np.float32(0.3)),
            ),
            (
                "normalize_last_axis",
                _numpy_normalize_last_axis,
                _numba_normalize_last_axis,
                (data,),
            ),
            (
                "normalize_middle_axis_copy",
                _numpy_normalize_middle_axis,
                _numba_normalize_middle_axis_with_copy,
                (data,),
            ),
        )
        for name, numpy_fn, numba_fn, args in candidates:
            result = _measure_candidate(name, numpy_fn, numba_fn, args, repeats=repeats)
            result.update(dtype=np.dtype(dtype).name, shape=list(shape), repeats=repeats)
            results.append(result)
            print(
                f"{result['dtype']:>9} {name:<20} "
                f"numpy={result['numpy_median_seconds']:.4f}s "
                f"numba_warm={result['numba_warm_median_seconds']:.4f}s "
                f"speedup={result['warm_speedup']:.2f}x "
                f"first={result['numba_first_seconds']:.4f}s"
            )
        del data
        gc.collect()
    return results


def _parse_shape(value):
    shape = tuple(int(part.strip()) for part in value.split(","))
    if not shape or any(size < 2 for size in shape):
        raise argparse.ArgumentTypeError("shape must contain comma-separated sizes >= 2")
    return shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=_parse_shape, default=(336, 336, 272))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    payload = {
        "shape": list(args.shape),
        "repeats": args.repeats,
        "results": run(args.shape, args.repeats),
    }
    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
