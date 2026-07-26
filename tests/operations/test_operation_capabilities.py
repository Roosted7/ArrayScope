from dataclasses import dataclass

import numpy as np
import pytest

from arrayscope.operations.capabilities import OperationCapabilities, OperationKind
from arrayscope.operations.pipeline import (
    CenteredFFT,
    CenteredIFFT,
    Clip,
    CombineRealImagAxis,
    Conjugate,
    Crop,
    FFTShift,
    HardThreshold,
    ImaginaryPart,
    LogMagnitude,
    Magnitude,
    Maximum,
    Mean,
    Minimum,
    Offset,
    Phase,
    Power,
    RealPart,
    ReverseAxis,
    RootSumSquares,
    Scale,
    SoftThreshold,
    SplitComplexAxis,
    Sum,
)
from arrayscope.operations.registry import operation_entries


def test_every_registered_operation_declares_dtype_and_capabilities():
    shape = (4, 5, 6)
    dtype = np.dtype(np.float32)
    samples = {
        "crop": Crop(axis=1, start=1, stop=4),
        "reverse": ReverseAxis(axis=1),
        "conjugate": Conjugate(),
        "magnitude": Magnitude(),
        "phase": Phase(),
        "real": RealPart(),
        "imag": ImaginaryPart(),
        "log_magnitude": LogMagnitude(epsilon=1e-6),
        "scale": Scale(factor=1.0),
        "offset": Offset(value=0.0),
        "power": Power(exponent=2.0),
        "clip": Clip(minimum=-1.0, maximum=1.0),
        "soft_threshold": SoftThreshold(threshold=0.1),
        "hard_threshold": HardThreshold(threshold=0.1),
        "mean": Mean(axis=1),
        "rss": RootSumSquares(axis=1),
        "sum": Sum(axis=1),
        "max": Maximum(axis=1),
        "min": Minimum(axis=1),
        "centered_fft": CenteredFFT(axis=1),
        "centered_ifft": CenteredIFFT(axis=1),
        "fftshift": FFTShift(axis=1),
        "combine_real_imag": CombineRealImagAxis(axis=2),
        "split_complex": SplitComplexAxis(axis=2),
    }

    for entry in operation_entries():
        operation = samples[entry.id]
        op_shape = (
            (4, 5, 2)
            if entry.id == "combine_real_imag"
            else (4, 5, 1)
            if entry.id == "split_complex"
            else shape
        )
        op_dtype = np.dtype(np.complex64) if entry.id == "split_complex" else dtype
        assert operation.output_dtype(op_dtype) is None or isinstance(
            np.dtype(operation.output_dtype(op_dtype)), np.dtype
        )
        capabilities = operation.capabilities(op_shape, op_dtype)
        assert isinstance(capabilities.kind, OperationKind)


def test_fft_declares_blocking_expanded_axis_and_stage_cache():
    capabilities = CenteredFFT(axis=1).capabilities((4, 8, 16), np.float32)

    assert capabilities.kind == OperationKind.TRANSFORM
    assert capabilities.blocking_axes == (1,)
    assert capabilities.expands_request_axes == (1,)
    assert capabilities.cache_stage is True
    assert 1 not in capabilities.chunkable_axes
    assert CenteredIFFT(axis=1).capabilities((4, 8, 16), np.float32).cache_stage is True


@pytest.mark.parametrize(
    "operation",
    [Mean(axis=1), Sum(axis=1), Maximum(axis=1), Minimum(axis=1), RootSumSquares(axis=1)],
)
def test_reductions_declare_blocking_and_expanded_axis(operation):
    capabilities = operation.capabilities((4, 8, 16), np.float32)

    assert capabilities.kind == OperationKind.REDUCTION
    assert capabilities.blocking_axes == (1,)
    assert capabilities.expands_request_axes == (1,)


@pytest.mark.parametrize(
    "operation", [Crop(axis=1, start=1, stop=4), ReverseAxis(axis=1), FFTShift(axis=1)]
)
def test_view_operations_are_fusible_without_blocking_axes(operation):
    capabilities = operation.capabilities((4, 8, 16), np.float32)

    assert capabilities.kind == OperationKind.VIEW
    assert capabilities.blocking_axes == ()
    assert capabilities.cache_stage is False
    assert capabilities.can_fuse is True


def test_complex_conversion_dtype_declarations_match_current_behavior():
    assert CombineRealImagAxis(axis=0).output_dtype(np.float32) == np.dtype(np.complex64)
    assert SplitComplexAxis(axis=0).output_dtype(np.complex64) == np.dtype(np.float32)


def test_capabilities_validate_axes_through_existing_shape_rules():
    with pytest.raises(ValueError, match="out of bounds"):
        CenteredFFT(axis=4).capabilities((4, 8, 16), np.float32)


# --- Display-LOD commuting contract (ADR 0050) ---


@dataclass(frozen=True)
class _OpaqueMontageAxisOp:
    """A shape-preserving transform on one axis that declares no linearity.

    Stands in for the general case the predicate must keep refusing: an
    operation may touch only the montage axis and still be a nonlinear
    per-line map, which an average taken across lines does not survive.
    """

    axis: int
    real_linear: bool = False

    def apply(self, data):
        return np.abs(data)

    def output_shape(self, shape):
        return tuple(shape)

    def output_dtype(self, input_dtype):
        return None if input_dtype is None else np.dtype(input_dtype)

    def capabilities(self, input_shape, input_dtype=None) -> OperationCapabilities:
        return OperationCapabilities(
            kind=OperationKind.TRANSFORM,
            blocking_axes=(int(self.axis),),
            expands_request_axes=(int(self.axis),),
            real_linear=bool(self.real_linear),
        )


@dataclass(frozen=True)
class _LinearMontageAxisOp(_OpaqueMontageAxisOp):
    """The same op with linearity declared -- the only difference that matters."""

    real_linear: bool = True


@pytest.mark.parametrize(
    ("operation", "commutes"),
    [
        (Conjugate(), True),
        (CenteredFFT(axis=2), False),
        (CenteredIFFT(axis=2), False),
        (FFTShift(axis=2), False),
        (Crop(axis=1, start=1, stop=4), False),
        (ReverseAxis(axis=1), False),
        (Mean(axis=0), False),
        (Sum(axis=0), False),
        (Maximum(axis=0), False),
        (Minimum(axis=0), False),
        (RootSumSquares(axis=0), False),
        (CombineRealImagAxis(axis=0), False),
    ],
)
def test_lod_commuting_is_conservative_and_true_only_for_pointwise_maps(operation, commutes):
    # Conservative contract: only pointwise value maps commute with box-mean
    # display reduction.  FFT/domain transforms, geometry changes, and
    # reductions never do; the flag defaults to False.
    shape = (2, 8, 16)
    assert bool(operation.capabilities(shape, np.float32).lod_commuting) is commutes


def test_normalize_capabilities_preserves_lod_commuting():
    from arrayscope.operations.capabilities import OperationCapabilities, normalize_capabilities

    capabilities = OperationCapabilities(kind=OperationKind.ELEMENTWISE, lod_commuting=True)
    assert normalize_capabilities(capabilities, ndim=2).lod_commuting is True
    capabilities = OperationCapabilities(kind=OperationKind.ELEMENTWISE)
    assert normalize_capabilities(capabilities, ndim=2).lod_commuting is False


def test_pipeline_commutes_for_display_lod_requires_every_stage_to_commute():
    from arrayscope.operations.capabilities import pipeline_commutes_for_display_lod

    shape = (4, 8, 16)
    assert pipeline_commutes_for_display_lod((Conjugate(),), shape, np.complex64) is True
    assert (
        pipeline_commutes_for_display_lod((Conjugate(), Conjugate()), shape, np.complex64) is True
    )
    # One non-commuting stage poisons the whole pipeline.
    assert (
        pipeline_commutes_for_display_lod((Conjugate(), CenteredFFT(axis=2)), shape, np.complex64)
        is False
    )
    assert pipeline_commutes_for_display_lod((CenteredFFT(axis=2),), shape, np.float32) is False
    # Shape-changing steps are rejected even when otherwise cheap.
    assert (
        pipeline_commutes_for_display_lod((Crop(axis=1, start=1, stop=4),), shape, np.float32)
        is False
    )
    # Capability-less callables are conservatively non-commuting.
    assert pipeline_commutes_for_display_lod((object(),), shape, np.float32) is False
    assert pipeline_commutes_for_display_lod((), shape, np.float32) is True


def test_fft_off_the_display_axes_commutes_and_on_them_does_not():
    # The axis is half the question. An FFT along the montage axis is exactly
    # commuting with a box mean of the display axes; the same FFT along a
    # display axis is not, and unknown display axes keep the old answer.
    from arrayscope.operations.capabilities import pipeline_commutes_for_display_lod

    shape = (4, 8, 16)
    profile_pipeline = (CenteredFFT(axis=2), FFTShift(axis=2), CenteredIFFT(axis=2))

    assert (
        pipeline_commutes_for_display_lod(profile_pipeline, shape, np.float32, display_axes=(0, 1))
        is True
    )
    # Negative: the same chain on a display axis stays non-commuting, one
    # stage at a time and as a chain.
    for operation in (CenteredFFT(axis=1), FFTShift(axis=1), CenteredIFFT(axis=1)):
        assert (
            pipeline_commutes_for_display_lod((operation,), shape, np.float32, display_axes=(0, 1))
            is False
        )
    assert (
        pipeline_commutes_for_display_lod(
            (CenteredFFT(axis=1), CenteredIFFT(axis=2)), shape, np.float32, display_axes=(0, 1)
        )
        is False
    )
    # Display axes the caller did not supply leave only the axis-blind
    # licence, which is the pre-2026-07 answer.
    assert pipeline_commutes_for_display_lod(profile_pipeline, shape, np.float32) is False
    # A display axis outside the base shape is a caller mismatch.
    assert (
        pipeline_commutes_for_display_lod(profile_pipeline, shape, np.float32, display_axes=(0, 7))
        is False
    )
    # Axis-disjointness alone is not enough: a reduction on the montage axis
    # is linear, but its result depends on how many samples fed it.
    assert (
        pipeline_commutes_for_display_lod((Mean(axis=2),), shape, np.float32, display_axes=(0, 1))
        is False
    )
    # And a stage that declares no linearity is out however disjoint it is.
    assert (
        pipeline_commutes_for_display_lod(
            (_OpaqueMontageAxisOp(axis=2),), shape, np.float32, display_axes=(0, 1)
        )
        is False
    )
    assert (
        pipeline_commutes_for_display_lod(
            (_LinearMontageAxisOp(axis=2),), shape, np.float32, display_axes=(0, 1)
        )
        is True
    )


def test_montage_axis_fft_on_reduced_display_input_is_numerically_exact():
    """The reason gate 1 exists: reduce-then-FFT == FFT-then-reduce, exactly.

    Both sides use the production box mean (`reduce_array_display_axes`) and
    the production pipeline evaluator, so this pins the claim the predicate
    makes and not an idealised restatement of it.
    """
    from arrayscope.operations.pipeline import evaluate as evaluate_pipeline
    from arrayscope.render.effects import reduce_array_display_axes

    rng = np.random.default_rng(20260726)
    volume = rng.standard_normal((16, 24, 8)).astype(np.float32)
    display_axes = (0, 1)
    factor_xy = (4, 4)
    operations = (CenteredFFT(axis=2), FFTShift(axis=2), CenteredIFFT(axis=2))

    reduce_then_apply = evaluate_pipeline(
        reduce_array_display_axes(volume, display_axes, factor_xy), operations
    )
    apply_then_reduce = reduce_array_display_axes(
        evaluate_pipeline(volume, operations), display_axes, factor_xy
    )

    assert reduce_then_apply.shape == apply_then_reduce.shape == (4, 6, 8)
    # float32/complex64 round-off only; the two arrays are the same array.
    np.testing.assert_allclose(
        reduce_then_apply, apply_then_reduce, rtol=1e-5, atol=1e-5 * float(np.abs(volume).max())
    )

    # Negative control: the same reduction against an FFT along a DISPLAY axis
    # is a different array, so the predicate's refusal there is load-bearing.
    display_axis_operations = (CenteredFFT(axis=1),)
    assert not np.allclose(
        evaluate_pipeline(
            reduce_array_display_axes(volume, display_axes, factor_xy), display_axis_operations
        ),
        reduce_array_display_axes(
            evaluate_pipeline(volume, display_axis_operations), display_axes, factor_xy
        ),
    )


def test_fft_pipeline_is_reduced_input_suitable_without_lod_commuting():
    from arrayscope.operations.capabilities import (
        pipeline_commutes_for_display_lod,
        pipeline_supports_reduced_display_lod,
    )

    shape = (4, 8, 16)
    operations = (CenteredFFT(axis=1),)

    assert (
        pipeline_commutes_for_display_lod(operations, shape, np.float32, display_axes=(0, 1))
        is False
    )
    # Broader by design and deliberately axis-blind: an FFT along a display
    # axis is still a legitimate preview presentation.
    assert pipeline_supports_reduced_display_lod(operations, shape, np.float32) is True


def test_pipeline_windowable_display_axes_raw_and_elementwise_chains():
    from arrayscope.operations.capabilities import pipeline_windowable_display_axes

    shape = (4, 8, 16)
    # The v1 fast-path case: no operations -> every display axis windowable.
    assert pipeline_windowable_display_axes((), shape, np.float32, display_axes=(1, 2)) == (1, 2)
    # Pointwise value maps keep windows valid on every axis.
    assert pipeline_windowable_display_axes(
        (Conjugate(),), shape, np.complex64, display_axes=(1, 2)
    ) == (1, 2)


def test_pipeline_windowable_display_axes_fft_blocks_only_its_axis():
    from arrayscope.operations.capabilities import pipeline_windowable_display_axes

    shape = (4, 8, 16)
    # FFT along a displayed axis: a window shift there is new content
    # (the ADR 0055 canonical negative case). The other display axis
    # stays windowable.
    assert pipeline_windowable_display_axes(
        (CenteredFFT(axis=1),), shape, np.complex64, display_axes=(1, 2)
    ) == (2,)
    # FFT along a non-displayed axis leaves both display axes windowable.
    assert pipeline_windowable_display_axes(
        (CenteredFFT(axis=0),), shape, np.complex64, display_axes=(1, 2)
    ) == (1, 2)


def test_pipeline_windowable_display_axes_is_conservative():
    from arrayscope.operations.capabilities import pipeline_windowable_display_axes

    shape = (4, 8, 16)
    # Shape-changing steps disqualify everything (axis identity drifts).
    assert (
        pipeline_windowable_display_axes((Mean(axis=0),), shape, np.float32, display_axes=(1, 2))
        == ()
    )
    # Coordinate-remapping view steps on a display axis are excluded in v1
    # even though required_input_region could window them.
    assert pipeline_windowable_display_axes(
        (ReverseAxis(axis=1),), shape, np.float32, display_axes=(1, 2)
    ) == (2,)
    assert pipeline_windowable_display_axes(
        (FFTShift(axis=2),), shape, np.float32, display_axes=(1, 2)
    ) == (1,)
    # Capability-less objects poison the chain.
    assert (
        pipeline_windowable_display_axes((object(),), shape, np.float32, display_axes=(1, 2)) == ()
    )


def test_operation_execution_class_covers_the_endpoint_table():
    from arrayscope.operations.capabilities import OperationClass, operation_execution_class

    shape = (4, 8, 16)
    # Coordinate metadata: moving/relabelling samples, never copying values.
    assert (
        operation_execution_class(Crop(axis=1, start=1, stop=4), shape, np.float32)
        is OperationClass.COORDINATE_METADATA
    )
    assert (
        operation_execution_class(ReverseAxis(axis=0), shape, np.float32)
        is OperationClass.COORDINATE_METADATA
    )
    assert (
        operation_execution_class(FFTShift(axis=2), shape, np.float32)
        is OperationClass.COORDINATE_METADATA
    )
    assert (
        operation_execution_class(SplitComplexAxis(axis=0), (1, 8, 16), np.complex64)
        is OperationClass.COORDINATE_METADATA
    )
    # Cheap pointwise value maps sample-time work.
    assert (
        operation_execution_class(Conjugate(), shape, np.complex64) is OperationClass.SHADER_ON_READ
    )
    # Reductions return small results.
    assert operation_execution_class(Mean(axis=0), shape, np.float32) is OperationClass.REDUCTION
    assert (
        operation_execution_class(RootSumSquares(axis=0), shape, np.float32)
        is OperationClass.REDUCTION
    )
    # Whole-axis transforms are cost-model territory.
    assert (
        operation_execution_class(CenteredFFT(axis=1), shape, np.complex64)
        is OperationClass.GLOBAL_TRANSFORM
    )
    # Anything undeclared stays opaque CPU materialization.
    assert operation_execution_class(object(), shape, np.float32) is OperationClass.OPAQUE
