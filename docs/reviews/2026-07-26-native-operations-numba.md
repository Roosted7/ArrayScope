# Native operation Numba verdict — 2026-07-26

## Question

Do fused Numba kernels justify production complexity for Bundle A's
multi-pass NumPy operations?

The benchmark is reproducible with:

```bash
conda run -n arrayscope python tools/benchmark_native_ops_numba.py \
  --repeats 5 --json /tmp/arrayscope-native-ops-numba.json
```

It uses a `336 x 336 x 272` array, both float32 and complex64, and checks every
Numba result against its NumPy reference. This run used NumPy 2.4.6, Numba
0.66.0, 16 Numba threads, and an 8-core/16-thread Intel i7-11850H.

## Results

Times are medians of five warmed executions. “First” includes JIT compilation
and the first representative-volume execution after the benchmark source cache
was invalidated. The middle-axis copy reuses the already compiled normalize
specialization, so its first time is not another JIT measurement.

| dtype | candidate | NumPy | Numba warm | speedup | first Numba |
|---|---:|---:|---:|---:|---:|
| float32 | log magnitude | 120.1 ms | 35.6 ms | 3.37x | 902.0 ms |
| complex64 | log magnitude | 90.7 ms | 45.6 ms | 1.99x | 277.3 ms |
| float32 | soft threshold | 169.8 ms | 14.5 ms | 11.74x | 366.4 ms |
| complex64 | soft threshold | 205.3 ms | 36.7 ms | 5.59x | 296.1 ms |
| float32 | normalize, last axis | 107.9 ms | 15.5 ms | 6.98x | 438.7 ms |
| complex64 | normalize, last axis | 241.4 ms | 37.1 ms | 6.50x | 610.4 ms |
| float32 | normalize, middle axis + copy | 101.2 ms | 44.4 ms | 2.28x | 48.0 ms* |
| complex64 | normalize, middle axis + copy | 347.8 ms | 162.0 ms | 2.15x | 149.0 ms* |

`*` Normalize specialization already compiled by the preceding last-axis case.

## Verdict

**Land normalize only.** Its 6.5–7.0x contiguous-last-axis win is decisive.
Even paying one explicit contiguous axis-move copy remains 2.1–2.3x faster on
this volume. The production path uses the shared lazy Numba runtime, returns to
NumPy until the kernel is ready, handles only float32/complex64, and refuses an
already-strided input rather than stacking another copy on an unknown layout.
The visible first call therefore executes the NumPy fallback while compilation
runs off-path.

**Do not land log-magnitude or soft-threshold, despite their warm timing wins.**
They are declared `ELEMENTWISE`, whose contract is exact:
`fn(whole)[region] == fn(whole[region])`. A contiguous whole array selected the
Numba path while a strided region selected NumPy; the two implementations
differed by 1–2 ULP (maximum observed differences `2.38e-7` for log-magnitude
and `2.43e-7` for complex soft-threshold), so the existing exact conformance
oracle went red. Weakening that oracle or copying every region would trade
rendering truth for a microbenchmark. Their measured performance win is real,
but the production verdict is **NO** until one execution owner can guarantee
the same implementation for whole arrays and all region layouts without
visible-path compilation or extra copies.

No blanket “Numba native ops” policy follows from this result. The benchmark is
kept as the retry gate; new kernels need the same value, dtype, layout,
first-call, and region-truth evidence.
