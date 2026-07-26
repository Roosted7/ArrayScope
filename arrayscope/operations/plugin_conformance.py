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

import logging
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor
from time import perf_counter_ns

import numpy as np

from arrayscope.operations.regions import (
    RegionSpec,
    apply_region,
    region_from_index_spec,
    region_text,
)

Shape = tuple[int, ...]

_LOGGER = logging.getLogger(__name__)
_CHARACTERIZATION_CACHE: dict[tuple, OperationCharacterization] = {}
_CHARACTERIZATION_STATS: dict[str, int] = {
    "characterized": 0,
    "cache_hits": 0,
    "predictable": 0,
    "unpredictable": 0,
    "invalidated": 0,
    "probe_calls": 0,
    "probe_elements": 0,
    "probe_ns": 0,
    "region_verified": 0,
    "region_honored": 0,
    "region_rejected": 0,
}
_MAX_EXACT_PROBE_ELEMENTS = 8192
_SHAPE_PROBE_SEED = 0xD15C0
_CHARACTERIZATION_REGION_SAMPLES = 12


class CharacterizationMismatch(RuntimeError):
    """A real result invalidated a fitted shape/dtype rule."""


class CharacterizationUnavailable(ValueError):
    """No safe prediction could be obtained with bounded probes."""


@dataclass(frozen=True)
class AxisShapeRule:
    source_axis: int | None
    mode: str
    value: int | Fraction

    def predict(self, input_shape: Shape) -> int:
        if self.source_axis is None:
            return int(self.value)
        size = int(input_shape[self.source_axis])
        if self.mode == "offset":
            return size + int(self.value)
        scaled = Fraction(self.value) * size
        if self.mode == "exact_scale":
            if scaled.denominator != 1:
                raise ValueError("exact scale produced a fractional axis length")
            return int(scaled)
        if self.mode == "floor_scale":
            return floor(scaled)
        if self.mode == "ceil_scale":
            return ceil(scaled)
        if self.mode == "round_half_up_scale":
            return floor(scaled + Fraction(1, 2))
        raise ValueError(f"unknown shape-rule mode: {self.mode}")


@dataclass(frozen=True)
class ShapeRule:
    kind: str
    axes: tuple[AxisShapeRule, ...]
    detail: str

    def predict(self, input_shape: Shape) -> Shape:
        result = tuple(int(axis.predict(tuple(input_shape))) for axis in self.axes)
        if any(size < 1 for size in result):
            raise ValueError(f"shape rule {self.detail} predicted invalid shape {result}")
        return result


@dataclass(frozen=True)
class OperationCharacterization:
    op_id: str
    input_shape: Shape
    input_dtype: np.dtype
    output_dtype: np.dtype
    shape_rule: ShapeRule
    predictable: bool
    region_honored: bool
    reason: str
    probe_calls: int
    probe_elements: int
    probe_ns: int
    cache_key: tuple

    @property
    def output_shape(self) -> Shape:
        return self.shape_rule.predict(self.input_shape)

    def predict_shape(self, input_shape: Shape) -> Shape:
        return self.shape_rule.predict(tuple(input_shape))


def characterization_stats() -> dict[str, int]:
    return dict(_CHARACTERIZATION_STATS)


def reset_characterization_cache() -> None:
    _CHARACTERIZATION_CACHE.clear()
    for name in _CHARACTERIZATION_STATS:
        _CHARACTERIZATION_STATS[name] = 0


def characterize_operation(
    spec,
    input_shape: Shape,
    dtype,
    *,
    axis: int | None = None,
    params=None,
) -> OperationCharacterization:
    """Jointly adjudicate shape, dtype, and windowability with one cache."""

    input_shape = tuple(int(size) for size in input_shape)
    input_dtype = np.dtype(dtype)
    param_map = dict(params or {})
    key = (
        str(getattr(spec, "id", "<anonymous>")),
        axis,
        _freeze(param_map),
        input_shape,
        input_dtype.str,
        _freeze(_source_identity(spec)),
    )
    cached = _CHARACTERIZATION_CACHE.get(key)
    if cached is not None:
        _CHARACTERIZATION_STATS["cache_hits"] += 1
        return cached

    fn = spec.resolve_fn(axis, param_map)
    probe_shapes = _probe_shapes(input_shape)
    started = perf_counter_ns()
    observations = []
    failure = None
    probe_elements = 0
    for index, probe_shape in enumerate(probe_shapes):
        probe_input = _deterministic_array(
            np.random.default_rng(_SHAPE_PROBE_SEED + index), probe_shape, input_dtype
        )
        try:
            probe_output = np.asarray(fn(probe_input))
        except Exception as exc:
            failure = exc
            break
        probe_elements += int(probe_input.size)
        observations.append(
            (probe_shape, tuple(probe_output.shape), probe_output.dtype, probe_input, probe_output)
        )

    rule = None if failure is not None else _fit_shape_rule(probe_shapes, observations, param_map)
    if failure is not None and not observations:
        raise failure
    predictable = rule is not None
    reason = ""
    if rule is None:
        exact = next(
            (
                (output_shape, output_dtype)
                for probe_shape, output_shape, output_dtype, _input, _output in observations
                if probe_shape == input_shape
            ),
            None,
        )
        if exact is None:
            if getattr(spec, "output_shape", None) is None:
                detail = f": {failure}" if failure is not None else ""
                raise CharacterizationUnavailable(
                    f"operation {getattr(spec, 'id', '<anonymous>')!r} has no "
                    f"conservative shape-rule fit within the bounded probe{detail}"
                )
            exact = (
                tuple(spec.resolve_output_shape(input_shape, axis, param_map)),
                np.dtype(spec.resolve_output_dtype(input_dtype)),
            )
            reason = "bounded discovery failed; declared adapter retained as exact opaque fallback"
        output_shape, output_dtype = exact
        rule = ShapeRule(
            "exact",
            tuple(AxisShapeRule(None, "fixed", int(size)) for size in output_shape),
            f"exact output {tuple(output_shape)} for input signature {input_shape}",
        )
        output_dtype = np.dtype(output_dtype)
        if not reason:
            reason = "observations did not fit an extrapolatable rule; exact whole-array shape only"
    else:
        output_dtypes = {np.dtype(observation[2]).str for observation in observations}
        output_dtype = np.dtype(observations[0][2])
        if len(output_dtypes) != 1:
            predictable = False
            reason = "output dtype changed across representative shapes; whole-array only"
        else:
            reason = rule.detail

    region_honored = False
    if bool(getattr(spec, "region_capable", False)):
        _CHARACTERIZATION_STATS["region_verified"] += 1
        conformance = _verify_region_from_observation(
            spec,
            observations[0] if observations else None,
            axis=axis,
            params=param_map,
            samples=_CHARACTERIZATION_REGION_SAMPLES,
        )
        region_honored = bool(conformance.honored and predictable and rule.kind == "identity")
        if region_honored:
            _CHARACTERIZATION_STATS["region_honored"] += 1
        else:
            _CHARACTERIZATION_STATS["region_rejected"] += 1
            _LOGGER.warning(
                "plugin operation %r FAILED conformance/characterization (%s); "
                "downgrading to OPAQUE whole-array with cache_stage",
                getattr(spec, "id", "<anonymous>"),
                conformance.reason or reason,
            )

    elapsed = perf_counter_ns() - started
    result = OperationCharacterization(
        op_id=str(getattr(spec, "id", "<anonymous>")),
        input_shape=input_shape,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        shape_rule=rule,
        predictable=bool(predictable),
        region_honored=region_honored,
        reason=reason,
        probe_calls=len(observations),
        probe_elements=probe_elements,
        probe_ns=elapsed,
        cache_key=key,
    )
    _CHARACTERIZATION_CACHE[key] = result
    _CHARACTERIZATION_STATS["characterized"] += 1
    _CHARACTERIZATION_STATS["predictable" if predictable else "unpredictable"] += 1
    _CHARACTERIZATION_STATS["probe_calls"] += len(observations)
    _CHARACTERIZATION_STATS["probe_elements"] += probe_elements
    _CHARACTERIZATION_STATS["probe_ns"] += elapsed
    return result


def record_runtime_mismatch(
    characterization: OperationCharacterization,
    *,
    actual_shape: Shape,
    actual_dtype,
) -> OperationCharacterization:
    """Invalidate a fitted relation before a mismatching result is returned."""

    _CHARACTERIZATION_CACHE.pop(characterization.cache_key, None)
    actual_shape = tuple(int(size) for size in actual_shape)
    exact = OperationCharacterization(
        op_id=characterization.op_id,
        input_shape=characterization.input_shape,
        input_dtype=characterization.input_dtype,
        output_dtype=np.dtype(actual_dtype),
        shape_rule=ShapeRule(
            "exact",
            tuple(AxisShapeRule(None, "fixed", size) for size in actual_shape),
            f"runtime mismatch forced exact output {actual_shape}",
        ),
        predictable=False,
        region_honored=False,
        reason="runtime output disagreed with fitted rule; cache invalidated and demoted",
        probe_calls=characterization.probe_calls,
        probe_elements=characterization.probe_elements,
        probe_ns=characterization.probe_ns,
        cache_key=characterization.cache_key,
    )
    _CHARACTERIZATION_CACHE[characterization.cache_key] = exact
    _CHARACTERIZATION_STATS["invalidated"] += 1
    if characterization.predictable and _CHARACTERIZATION_STATS["predictable"] > 0:
        _CHARACTERIZATION_STATS["predictable"] -= 1
    _CHARACTERIZATION_STATS["unpredictable"] += 1
    return exact


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


def _verify_region_from_observation(
    spec,
    observation,
    *,
    axis: int | None,
    params,
    samples: int,
) -> ConformanceResult:
    op_id = str(getattr(spec, "id", "<anonymous>"))
    if observation is None:
        return ConformanceResult(
            honored=False,
            op_id=op_id,
            samples_checked=0,
            reason="operation failed before a region-conformance sample was available",
        )
    sample_shape, output_shape, _dtype, whole, whole_result = observation
    if tuple(output_shape) != tuple(sample_shape):
        return ConformanceResult(
            honored=False,
            op_id=op_id,
            samples_checked=0,
            reason=(
                "region-capable op must be shape-preserving; output shape "
                f"{output_shape} != input shape {sample_shape}"
            ),
        )

    fn = spec.resolve_fn(axis, dict(params))
    rng = np.random.default_rng(_SHAPE_PROBE_SEED)
    regions = _sample_regions(rng, sample_shape, samples)
    for checked, region in enumerate(regions, start=1):
        sub_result = np.asarray(fn(apply_region(whole, region)))
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
        if not np.array_equal(sub_result, expected):
            return ConformanceResult(
                honored=False,
                op_id=op_id,
                samples_checked=checked,
                reason=(
                    "region path disagrees with whole-array result at region "
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


def _probe_shapes(input_shape: Shape) -> tuple[Shape, ...]:
    """Return a bounded base slab plus two independent variations per axis."""

    input_shape = tuple(int(size) for size in input_shape)
    if int(np.prod(input_shape, dtype=np.int64)) <= _MAX_EXACT_PROBE_ELEMENTS:
        base = input_shape
    else:
        compact_size = 5 if len(input_shape) <= 5 else 3 if len(input_shape) <= 6 else 2
        base = tuple(1 if size == 1 else compact_size for size in input_shape)

    shapes = [base]
    for axis, size in enumerate(base):
        if input_shape[axis] == 1:
            continue
        for varied_size in (max(5, size + 3), max(8, size + 6)):
            varied = list(base)
            varied[axis] = varied_size
            shapes.append(tuple(varied))
    return tuple(dict.fromkeys(shapes))


def _fit_shape_rule(probe_shapes, observations, params) -> ShapeRule | None:
    if not observations or len(observations) != len(probe_shapes):
        return None
    output_shapes = [tuple(observation[1]) for observation in observations]
    output_rank = len(output_shapes[0])
    if any(len(shape) != output_rank for shape in output_shapes):
        return None

    base_input = tuple(probe_shapes[0])
    base_output = output_shapes[0]
    axes = []
    for output_axis in range(output_rank):
        dependencies = set()
        for input_axis in range(len(base_input)):
            values = {
                output_shape[output_axis]
                for probe_shape, output_shape in zip(probe_shapes, output_shapes, strict=True)
                if all(
                    probe_shape[other] == base_input[other]
                    for other in range(len(base_input))
                    if other != input_axis
                )
            }
            if len(values) > 1:
                dependencies.add(input_axis)
        if len(dependencies) > 1:
            return None
        if not dependencies:
            if any(shape[output_axis] != base_output[output_axis] for shape in output_shapes):
                return None
            axes.append(AxisShapeRule(None, "fixed", int(base_output[output_axis])))
            continue
        source_axis = dependencies.pop()
        pairs = [
            (int(probe_shape[source_axis]), int(output_shape[output_axis]))
            for probe_shape, output_shape in zip(probe_shapes, output_shapes, strict=True)
        ]
        relation = _fit_axis_relation(pairs, params)
        if relation is None:
            return None
        axes.append(AxisShapeRule(source_axis, relation.mode, relation.value))

    mapped_axes = [axis.source_axis for axis in axes if axis.source_axis is not None]
    if len(mapped_axes) != len(set(mapped_axes)):
        return None
    rule = ShapeRule(_classify_shape_rule(tuple(axes), len(base_input)), tuple(axes), "")
    detail = _shape_rule_detail(rule, len(base_input))
    rule = ShapeRule(rule.kind, rule.axes, detail)
    if any(
        rule.predict(probe) != output
        for probe, output in zip(probe_shapes, output_shapes, strict=True)
    ):
        return None
    return rule


def _fit_axis_relation(pairs, params) -> AxisShapeRule | None:
    unique = tuple(dict.fromkeys((int(source), int(output)) for source, output in pairs))
    if not unique:
        return None
    offsets = {output - source for source, output in unique}
    if len(offsets) == 1:
        return AxisShapeRule(0, "offset", offsets.pop())

    candidates: list[tuple[str, Fraction]] = []
    if all(source != 0 for source, _output in unique):
        ratios = {Fraction(output, source) for source, output in unique}
        if len(ratios) == 1:
            candidates.append(("exact_scale", ratios.pop()))

    factors = {Fraction(1, divisor) for divisor in range(2, 17)}
    for value in dict(params).values():
        if isinstance(value, (int, float, np.integer, np.floating)) and float(value) > 0:
            factor = Fraction(str(float(value))).limit_denominator(1024)
            factors.add(factor)
            factors.add(1 / factor)
    for factor in sorted(factors):
        candidates.extend(
            (
                ("floor_scale", factor),
                ("ceil_scale", factor),
                ("round_half_up_scale", factor),
            )
        )

    fits = []
    for mode, factor in candidates:
        candidate = AxisShapeRule(0, mode, factor)
        if all(candidate.predict((source,)) == output for source, output in unique):
            fits.append(candidate)
    if not fits:
        return None
    # A small sample can make floor/ceil/round aliases look identical.  Only
    # extrapolate if every fitting candidate agrees on a substantially larger
    # holdout; otherwise the author supplied too little evidence for a rule.
    behaviors = {tuple(candidate.predict((size,)) for size in range(1, 65)) for candidate in fits}
    if len(behaviors) != 1:
        return None
    return fits[0]


def _classify_shape_rule(axes: tuple[AxisShapeRule, ...], input_rank: int) -> str:
    mapped = tuple(axis.source_axis for axis in axes)
    identity_axes = tuple(range(input_rank))
    if len(axes) == input_rank and mapped == identity_axes:
        if all(axis.mode == "offset" and int(axis.value) == 0 for axis in axes):
            return "identity"
        if all(axis.mode == "offset" for axis in axes):
            return "pad_crop"
        return "axis_scaled"
    if len(axes) == input_rank and set(mapped) == set(identity_axes):
        return "permutation"
    if not axes or all(axis.source_axis is None for axis in axes):
        return "fixed_size"
    if all(axis.source_axis is not None for axis in axes) and len(axes) < input_rank:
        return "axis_reduced"
    return "fixed_and_mapped"


def _shape_rule_detail(rule: ShapeRule, input_rank: int) -> str:
    descriptions = []
    for axis in rule.axes:
        if axis.source_axis is None:
            descriptions.append(f"fixed {int(axis.value)}")
        elif axis.mode == "offset":
            descriptions.append(f"input[{axis.source_axis}] {int(axis.value):+d}")
        else:
            descriptions.append(
                f"{axis.mode.removesuffix('_scale')}({axis.value} * input[{axis.source_axis}])"
            )
    return f"{rule.kind} rule ({', '.join(descriptions)}) from rank {input_rank}"


def _source_identity(spec):
    identity = getattr(spec, "source_identity", None)
    return identity() if callable(identity) else identity


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    if isinstance(value, np.generic):
        return value.item()
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


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
