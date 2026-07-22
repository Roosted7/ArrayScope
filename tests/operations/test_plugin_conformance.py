"""Tier-2 region-conformance harness: honor a windowable claim only if true.

A Tier-2 plugin op *claims* it commutes with sub-region reads
(``fn(whole)[region] == fn(whole[region])``) so the engine may run it per-region
instead of materializing the whole array.  A FALSE claim yields
plausible-but-wrong pixels at interactive speed, so the claim is never trusted on
the author's word: :func:`verify_region_conformance` property-tests it, and the
registry gate in ``plugins.py`` honors it only if it passes -- otherwise the op is
downgraded to the OPAQUE whole-array path (correct, just not the fast path).

The exit gate (red-first): a deliberately mis-declared op (a global roll / global
normalization dressed up as elementwise) is REJECTED, while an honestly-declared
elementwise op (``x*2+1``, which genuinely commutes with windowing) is honored.
"""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.operations import plugins
from arrayscope.operations.capabilities import OperationClass, OperationKind
from arrayscope.operations.plugin_conformance import verify_region_conformance
from arrayscope.operations.plugins import PluginOperationSpec
from arrayscope.operations.regions import (
    AxisRegionKind,
    apply_region,
    region_axis_kinds,
    region_from_index_spec,
)

PROBE_SHAPE = (5, 4, 3)


# --- fake specs (duck-typed; no plugin install needed for the harness) -------


def honest_double() -> PluginOperationSpec:
    # Pure elementwise: x*2+1 is computed per element, so it is bit-exact whether
    # or not the input was windowed -- a truthful Tier-2 claim.
    return PluginOperationSpec(
        id="demo:double", label="Double", fn=lambda a: a * 2 + 1, region_capable=True
    )


def misdeclared_roll() -> PluginOperationSpec:
    # Shape-preserving but GLOBAL: rolling the whole array then windowing differs
    # from rolling a window.  Claims region_capable -> must be rejected.
    return PluginOperationSpec(
        id="demo:bad_roll",
        label="Bad roll",
        fn=lambda a: np.roll(a, 1, axis=0),
        region_capable=True,
    )


def misdeclared_normalize() -> PluginOperationSpec:
    # Looks elementwise (subtract a scalar) but the scalar is a GLOBAL statistic:
    # a window has a different mean, so the region path disagrees.
    return PluginOperationSpec(
        id="demo:bad_norm",
        label="Bad normalize",
        fn=lambda a: a - a.mean(),
        region_capable=True,
    )


def shape_changer() -> PluginOperationSpec:
    return PluginOperationSpec(
        id="demo:bad_shape",
        label="Drop first",
        fn=lambda a: a[1:],
        output_shape=lambda shape, axis, params: (int(shape[0]) - 1, *shape[1:]),
        changes_shape=True,
        region_capable=True,
    )


def tier1_opaque() -> PluginOperationSpec:
    # region_capable defaults False -> a Tier-1 OPAQUE op, never region-tested.
    return PluginOperationSpec(id="demo:opaque", label="Opaque", fn=lambda a: np.roll(a, 1, axis=0))


def _rng() -> np.random.Generator:
    return np.random.default_rng(1234)


# --- harness-level tests (the property test in isolation) --------------------


def test_honest_elementwise_op_passes_conformance():
    result = verify_region_conformance(
        honest_double(), PROBE_SHAPE, "float64", rng=_rng(), samples=16
    )
    assert result.honored is True
    assert bool(result) is True
    assert result.failing_region is None
    assert result.samples_checked >= 8  # at least the guaranteed per-axis coverage


def test_misdeclared_global_roll_is_rejected_with_a_concrete_counterexample():
    result = verify_region_conformance(
        misdeclared_roll(), PROBE_SHAPE, "float64", rng=_rng(), samples=16
    )
    assert result.honored is False
    assert result.failing_region is not None
    assert result.max_abs_diff is not None
    assert result.max_abs_diff > 0
    assert "not windowable" in result.reason


def test_misdeclared_global_normalization_is_rejected():
    result = verify_region_conformance(
        misdeclared_normalize(), PROBE_SHAPE, "float64", rng=_rng(), samples=16
    )
    assert result.honored is False
    assert result.failing_region is not None


def test_shape_changing_op_cannot_be_region_capable():
    result = verify_region_conformance(
        shape_changer(), PROBE_SHAPE, "float64", rng=_rng(), samples=8
    )
    assert result.honored is False
    assert "shape-preserving" in result.reason


def test_conformance_is_non_vacuous():
    # Law #5: prove the check has teeth.  Under the real (exact) comparison the
    # mis-declared roll is rejected on a CONCRETE region with a non-zero diff...
    strict = verify_region_conformance(
        misdeclared_roll(), PROBE_SHAPE, "float64", rng=_rng(), samples=16
    )
    assert strict.honored is False
    assert strict.max_abs_diff > 0

    # ...and if we WEAKEN the equality check to swallow any difference, the exact
    # same op is (wrongly) honored.  So it is the VALUE comparison -- not the
    # shape/plumbing -- that triggers rejection: the harness is not vacuous.
    weak = verify_region_conformance(
        misdeclared_roll(),
        PROBE_SHAPE,
        "float64",
        rng=_rng(),
        samples=16,
        rtol=1e18,
        atol=1e18,
    )
    assert weak.honored is True


# --- registry gate tests (honor-only-after-conformance) ----------------------


@pytest.fixture
def primed():
    """Prime the spec cache directly (the documented ``_reset_plugin_cache`` seam)
    so gate behavior can be exercised without entry-point plumbing."""

    plugins._reset_plugin_cache()
    for spec in (
        honest_double(),
        misdeclared_roll(),
        misdeclared_normalize(),
        tier1_opaque(),
    ):
        plugins._SPEC_CACHE[spec.id] = spec
    yield
    plugins._reset_plugin_cache()


def test_honest_claim_is_honored_and_observable(primed):
    op = plugins.create_plugin_operation("demo:double")

    assert plugins.is_region_honored("demo:double") is True
    assert op._region_honored() is True

    caps = op.capabilities(PROBE_SHAPE)
    assert caps.kind is OperationKind.ELEMENTWISE
    assert caps.blocking_axes == ()
    assert caps.expands_request_axes == ()
    assert op.execution_class is OperationClass.SHADER_ON_READ

    stats = plugins.region_conformance_stats()
    assert stats["verified"] >= 1
    assert stats["honored"] >= 1
    assert stats["rejected"] == 0


def test_honored_region_path_matches_the_whole_array_result(primed):
    op = plugins.create_plugin_operation("demo:double")
    data = np.arange(int(np.prod(PROBE_SHAPE))).reshape(PROBE_SHAPE).astype(float)
    region = region_from_index_spec(data.shape, (slice(1, 4), 2, slice(None)))

    # Honored -> identity input map: it needs exactly the output sub-region
    # (not the whole axis), unlike the OPAQUE path which asks for ALL everywhere.
    input_region = op.required_input_region(data.shape, region)
    assert input_region == region
    assert region_axis_kinds(input_region)[:2] == (
        AxisRegionKind.SLICE.value,
        AxisRegionKind.POINT.value,
    )

    sub = apply_region(data, input_region)
    got = op.apply_to_region(sub, input_region=input_region, output_region=region)
    np.testing.assert_array_equal(got, apply_region(op.apply(data), region))


def test_misdeclared_claim_is_downgraded_to_opaque_loudly(primed, caplog):
    with caplog.at_level("WARNING"):
        op = plugins.create_plugin_operation("demo:bad_roll")

    assert plugins.is_region_honored("demo:bad_roll") is False
    assert op._region_honored() is False

    # Downgraded to the Tier-1 OPAQUE whole-array shape.
    caps = op.capabilities(PROBE_SHAPE)
    assert caps.kind is OperationKind.TRANSFORM
    assert caps.blocking_axes == (0, 1, 2)
    assert caps.expands_request_axes == (0, 1, 2)
    assert op.execution_class is OperationClass.OPAQUE

    # The downgrade is observable: a loud warning + a rejected tally.
    assert "bad_roll" in caplog.text
    assert "FAILED conformance" in caplog.text
    assert plugins.region_conformance_stats()["rejected"] >= 1

    # And it still produces CORRECT pixels via the whole-array path: OPAQUE
    # required_input_region asks for ALL, apply_to_region gets the whole array.
    data = np.arange(int(np.prod(PROBE_SHAPE))).reshape(PROBE_SHAPE).astype(float)
    region = region_from_index_spec(data.shape, (slice(1, 4), 2, slice(None)))
    whole_region = op.required_input_region(data.shape, region)
    assert region_axis_kinds(whole_region) == (
        AxisRegionKind.ALL.value,
        AxisRegionKind.ALL.value,
        AxisRegionKind.ALL.value,
    )
    got = op.apply_to_region(data, input_region=whole_region, output_region=region)
    np.testing.assert_array_equal(got, apply_region(op.apply(data), region))


def test_tier1_opaque_op_is_never_subjected_to_region_claims(primed):
    op = plugins.create_plugin_operation("demo:opaque")

    assert op._region_honored() is False
    assert plugins.is_region_honored("demo:opaque") is False
    # region_capable=False -> the harness is never even invoked for it.
    assert plugins.region_conformance_stats()["verified"] == 0

    # Unchanged Tier-1: OPAQUE, whole-array.
    caps = op.capabilities(PROBE_SHAPE)
    assert caps.kind is OperationKind.TRANSFORM
    assert op.execution_class is OperationClass.OPAQUE


def test_honored_op_is_faithful_through_the_operation_evaluator(primed):
    # Drive the honored region op through the real slab/region engine and compare
    # against a materialized oracle -- the region fast path must be faithful.
    from arrayscope.core.view_state import ViewState
    from arrayscope.operations.evaluator import OperationEvaluator
    from arrayscope.operations.pipeline import ArrayDocument

    data = np.arange(int(np.prod(PROBE_SHAPE))).reshape(PROBE_SHAPE).astype(float)
    plugin_document = ArrayDocument(data).with_operation(
        plugins.create_plugin_operation("demo:double")
    )
    oracle_document = ArrayDocument(data * 2 + 1)
    state = ViewState.from_shape(plugin_document.current_shape)

    plugin_image = OperationEvaluator(plugin_document).image(state)
    oracle_image = OperationEvaluator(oracle_document).image(state)
    np.testing.assert_allclose(np.asarray(plugin_image.data), np.asarray(oracle_image.data))
