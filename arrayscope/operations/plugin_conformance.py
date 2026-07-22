"""Tier-2 region-conformance harness for plugin operations.

A Tier-1 plugin op is OPAQUE: it always materializes the whole array (see
:mod:`arrayscope.operations.plugins`).  A **Tier-2** op additionally *claims* it
is windowable -- that it commutes with sub-region reads:

    fn(whole)[region] == fn(whole[region])   for a window ``region`` on any axis.

When that holds the engine may run the op per-region and never materialize the
whole array -- a real performance win.  When it does *not* hold (a global
reduction, roll, normalization, or FFT dressed up to look elementwise), running
it per-region yields plausible-but-wrong pixels at interactive speed: exactly the
silent-corruption class ArrayScope guards hardest against.

So a region claim is never trusted on the author's word.  This module
property-tests it: build a deterministic array, compute the whole-array result
once, then for many sampled sub-regions compare the region-path result
(``fn(whole[region])``) against the oracle (``fn(whole)[region]``).  The registry
honors the claim only if every sampled region matches; a mismatch downgrades the
op to the OPAQUE whole-array path (see the gate in ``plugins.py``).

The harness takes a duck-typed ``spec`` (anything with ``id`` and
``resolve_fn(axis, params)``) so it does not import the plugin registry -- no
import cycle, and it is unit-testable against a bare fake spec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arrayscope.operations.regions import (
    RegionSpec,
    apply_region,
    region_from_index_spec,
    region_text,
)

Shape = tuple[int, ...]


@dataclass(frozen=True)
class ConformanceResult:
    """Verdict of a Tier-2 region-conformance check.

    ``honored`` is the gate signal.  On failure ``failing_region`` +
    ``max_abs_diff`` pin the *specific* counterexample, which is what makes the
    check non-vacuous: a rejection is a concrete disagreement, not a shrug.
    """

    honored: bool
    op_id: str
    samples_checked: int
    reason: str = ""
    failing_region: RegionSpec | None = None
    max_abs_diff: float | None = None

    def __bool__(self) -> bool:
        return self.honored


def verify_region_conformance(
    spec,
    sample_shape: Shape,
    dtype,
    *,
    rng: np.random.Generator,
    axis: int | None = None,
    params=None,
    samples: int = 12,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> ConformanceResult:
    """Property-test a plugin op's Tier-2 (windowable) claim.

    Builds a deterministic array from ``rng``, computes ``fn(whole)`` once, then
    for ``samples`` sampled sub-regions checks
    ``fn(whole[region]) == fn(whole)[region]``.  Returns the first mismatching
    region on failure.

    Equality is **exact by default** (``rtol == atol == 0`` -> ``array_equal``).
    A windowable op is per-element by nature, so a truthful float op is bit-exact
    whether or not the input was windowed -- exactness is both correct and the
    strongest possible discriminator.  ``rtol``/``atol`` are exposed only for a
    float op that legitimately reorders arithmetic (and for the non-vacuity
    proof: widening the tolerance to swallow a real mismatch must be a
    deliberate, visible choice, not the default).
    """

    op_id = str(getattr(spec, "id", "<anonymous>"))
    param_map = dict(params or {})
    fn = spec.resolve_fn(axis, param_map)

    whole = _deterministic_array(rng, sample_shape, dtype)
    whole_result = np.asarray(fn(whole))

    # A windowable op must be shape-preserving: windowing an axis only makes
    # sense if that axis still exists (and lines up) in the output.  A
    # shape-changing op cannot be Tier-2 in this contract.
    if whole_result.shape != whole.shape:
        return ConformanceResult(
            honored=False,
            op_id=op_id,
            samples_checked=0,
            reason=(
                f"region-capable op must be shape-preserving; output shape "
                f"{whole_result.shape} != input shape {whole.shape}"
            ),
        )

    regions = _sample_regions(rng, sample_shape, samples)
    for checked, region in enumerate(regions, start=1):
        sub_input = apply_region(whole, region)
        sub_result = np.asarray(fn(sub_input))
        expected = apply_region(whole_result, region)

        if sub_result.shape != expected.shape:
            return ConformanceResult(
                honored=False,
                op_id=op_id,
                samples_checked=checked,
                reason=(
                    f"region path shape {sub_result.shape} != windowed whole-array "
                    f"shape {expected.shape} at region {region_text(region)}"
                ),
                failing_region=region,
            )

        if not _values_equal(sub_result, expected, rtol=rtol, atol=atol):
            return ConformanceResult(
                honored=False,
                op_id=op_id,
                samples_checked=checked,
                reason=(
                    f"region path disagrees with whole-array result at region "
                    f"{region_text(region)} (op is not windowable)"
                ),
                failing_region=region,
                max_abs_diff=_max_abs_diff(sub_result, expected),
            )

    return ConformanceResult(
        honored=True,
        op_id=op_id,
        samples_checked=len(regions),
        reason=f"windowable claim verified on {len(regions)} sampled regions",
    )


def _values_equal(left: np.ndarray, right: np.ndarray, *, rtol: float, atol: float) -> bool:
    if rtol == 0.0 and atol == 0.0:
        return bool(np.array_equal(left, right))
    return bool(np.allclose(left, right, rtol=rtol, atol=atol))


def _max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.complex128) - right.astype(np.complex128))))


def _deterministic_array(rng: np.random.Generator, shape: Shape, dtype) -> np.ndarray:
    shape = tuple(int(size) for size in shape)
    dt = np.dtype(dtype)
    if dt.kind == "c":
        real = rng.standard_normal(shape)
        imag = rng.standard_normal(shape)
        return (real + 1j * imag).astype(dt)
    if dt.kind == "f":
        return rng.standard_normal(shape).astype(dt)
    if dt.kind == "b":
        return rng.integers(0, 2, size=shape).astype(dt)
    if dt.kind == "u":
        return rng.integers(0, 16, size=shape).astype(dt)
    # signed integers
    return rng.integers(-8, 8, size=shape).astype(dt)


def _sample_regions(rng: np.random.Generator, shape: Shape, samples: int) -> tuple[RegionSpec, ...]:
    """Deterministic mix of sub-region windows that expose non-commutation.

    Guarantees coverage that a global op cannot survive -- a partial slice and a
    single-index point on every axis -- then fills up to ``samples`` with random
    per-axis windows (full / point / slice).  A global roll, reduction, or
    normalization disagrees on any of the partial windows; a truthful elementwise
    op agrees on all of them.
    """

    shape = tuple(int(size) for size in shape)
    ndim = len(shape)
    specs: list[tuple] = []
    seen: set[str] = set()

    def add(spec: tuple) -> None:
        region = region_from_index_spec(shape, spec)
        key = region_text(region)
        if key not in seen:
            seen.add(key)
            specs.append(region)

    # Guaranteed coverage: per axis, a partial slice and a mid point.
    for axis, size in enumerate(shape):
        partial = [slice(None)] * ndim
        start = size // 3
        stop = max(start + 1, size - 1)
        partial[axis] = slice(start, stop)
        add(tuple(partial))

        point = [slice(None)] * ndim
        point[axis] = size // 2
        add(tuple(point))

    # Random windows to reach the requested sample count.
    attempts = 0
    while len(specs) < samples and attempts < samples * 8:
        attempts += 1
        spec = []
        for size in shape:
            choice = int(rng.integers(0, 3))
            if choice == 0 or size == 1:
                spec.append(slice(None))
            elif choice == 1:
                spec.append(int(rng.integers(0, size)))
            else:
                lo = int(rng.integers(0, size))
                hi = int(rng.integers(lo + 1, size + 1))
                spec.append(slice(lo, hi))
        add(tuple(spec))

    return tuple(specs)
