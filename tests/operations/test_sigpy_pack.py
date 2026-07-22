"""Tests for the optional in-process sigpy operation pack.

Covers: the ops apply correctly against a direct sigpy reference; they appear in
the registry / ``all_operations`` when sigpy is importable; the honesty gate (a
mis-declared region-capable FFT is downgraded by the conformance harness, while
the real ops ship OPAQUE); optionality (sigpy-absent contributes nothing and the
registry stays green); laziness (enumeration does not import sigpy); and an
FFT->IFFT round-trip through the recipe path.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

sigpy = pytest.importorskip("sigpy")

from arrayscope.operations import plugins, registry
from arrayscope.operations.capabilities import OperationClass
from arrayscope.operations.packs import sigpy_pack
from arrayscope.operations.plugin_conformance import verify_region_conformance
from arrayscope.operations.plugins import PluginOperationSpec
from arrayscope.operations.recipes import dumps_recipe, loads_recipe

PROBE_SHAPE = (5, 4, 3)


@pytest.fixture(autouse=True)
def _clean_pack_state():
    """Reset pack + plugin-spec caches around each test so registration and the
    Tier-2 gate verdicts start from a known state."""

    registry._reset_operation_packs()
    plugins._reset_plugin_cache()
    yield
    registry._reset_operation_packs()
    plugins._reset_plugin_cache()


# --- discovery / enumeration -------------------------------------------------


def test_sigpy_ops_appear_in_all_operations():
    ids = {entry.id for entry in registry.all_operations()}
    assert {"sigpy:fft", "sigpy:ifft"} <= ids

    fft_entry = registry.get_operation_entry("sigpy:fft")
    assert fft_entry.label == "Centered FFT (sigpy)"
    assert fft_entry.requires_axis is True
    assert fft_entry.changes_shape is False


def test_enumeration_does_not_import_sigpy(monkeypatch):
    """Laziness: listing ops (dock build) must not import sigpy -- only find_spec.

    Loading the pack module is fine (it is side-effect-free); actually importing
    sigpy is what must stay deferred until an op is applied.
    """

    monkeypatch.delitem(sys.modules, "sigpy", raising=False)
    registry.all_operations()
    assert "sigpy" not in sys.modules, "enumerating operations must not import sigpy"


# --- correctness against a direct sigpy reference ----------------------------


@pytest.mark.parametrize("dtype", ["float64", "float32", "complex64", "complex128", "int16"])
def test_fft_matches_sigpy_reference(dtype):
    x = (np.arange(60).reshape(PROBE_SHAPE)).astype(dtype)
    op = registry.create_operation("sigpy:fft", axis=1)
    got = np.asarray(op.apply(x))
    ref = sigpy.fft(x, axes=(1,))
    # A centered FFT is a linear transform: the op is a thin wrapper, so the
    # result is bit-exact with the direct sigpy call.
    assert np.array_equal(got, ref)
    assert np.dtype(op.output_dtype(np.dtype(dtype))) == got.dtype


@pytest.mark.parametrize("dtype", ["float64", "complex64", "complex128"])
def test_ifft_matches_sigpy_reference(dtype):
    x = (np.arange(60).reshape(PROBE_SHAPE)).astype(dtype)
    op = registry.create_operation("sigpy:ifft", axis=2)
    got = np.asarray(op.apply(x))
    ref = sigpy.ifft(x, axes=(2,))
    assert np.array_equal(got, ref)


# --- honest capability: the real ops are OPAQUE / Tier-1 ---------------------


def test_real_sigpy_ops_are_opaque_and_never_region_honored():
    for op_id in ("sigpy:fft", "sigpy:ifft"):
        op = registry.create_operation(op_id, axis=0)
        # region_capable defaults False -> the conformance harness is never even
        # invoked, and the op stays on the OPAQUE whole-array path.
        assert plugins.is_region_honored(op_id, op.axis, op.params) is False
        assert op.execution_class is OperationClass.OPAQUE
    # No claim was ever made, so nothing was verified by the gate.
    assert plugins.region_conformance_stats()["verified"] == 0


# --- honesty gate (red-first): a mis-declared FFT claim is DOWNGRADED ---------


def _bad_region_fft_spec() -> PluginOperationSpec:
    """A sigpy FFT that FALSELY claims to be windowable (region_capable=True).

    An FFT along axis 0 is global on that axis, so ``fft(whole)[region]`` differs
    from ``fft(whole[region])``.  It is shape-preserving (so it clears the shape
    check) but must fail the value comparison -> the harness must reject it.
    """

    return PluginOperationSpec(
        id="sigpy_demo:bad_fft",
        label="Bad FFT (falsely windowable)",
        build=lambda axis, params: (lambda data: sigpy.fft(np.asarray(data), axes=(0,))),
        region_capable=True,
    )


def test_misdeclared_sigpy_fft_is_rejected_by_the_harness():
    result = verify_region_conformance(
        _bad_region_fft_spec(),
        PROBE_SHAPE,
        "complex128",
        rng=np.random.default_rng(0xF17),
        samples=16,
    )
    assert result.honored is False
    assert result.failing_region is not None
    assert result.max_abs_diff is not None and result.max_abs_diff > 0
    assert "not windowable" in result.reason


def test_misdeclared_sigpy_fft_is_downgraded_by_the_registry_gate(caplog):
    # Prime the false spec straight into the spec cache (the documented seam) so
    # the gate can adjudicate it without entry-point plumbing.
    spec = _bad_region_fft_spec()
    plugins._SPEC_CACHE[spec.id] = spec

    with caplog.at_level("WARNING"):
        op = plugins.create_plugin_operation(spec.id)

    assert plugins.is_region_honored(spec.id, op.axis, op.params) is False
    assert op.execution_class is OperationClass.OPAQUE
    assert "bad_fft" in caplog.text
    assert "FAILED conformance" in caplog.text
    assert plugins.region_conformance_stats()["rejected"] >= 1


# --- optionality: sigpy-absent contributes nothing, registry stays green -----


def test_pack_contributes_nothing_when_sigpy_is_absent(monkeypatch):
    monkeypatch.setattr(sigpy_pack, "sigpy_available", lambda: False)
    registry._reset_operation_packs()

    # register() reports it added nothing...
    assert sigpy_pack.register() is False

    # ...and the registry enumerates cleanly with no sigpy ops present.
    ids = {entry.id for entry in registry.all_operations()}
    assert not any(op_id.startswith("sigpy:") for op_id in ids)
    # Built-ins are untouched -- the registry stays green.
    assert "centered_fft" in ids
    assert "crop" in ids


def test_pack_specs_exist_independently_of_installation():
    # The spec set is defined by the pack, not by whether sigpy is installed;
    # registration is what is gated.
    ids = {spec.id for spec in sigpy_pack.pack_specs()}
    assert ids == {"sigpy:fft", "sigpy:ifft"}


# --- round-trip through the recipe path --------------------------------------


def test_fft_then_ifft_round_trips_through_the_recipe_path():
    x = (np.arange(60).reshape(PROBE_SHAPE)).astype("float64")
    forward = registry.create_operation("sigpy:fft", axis=1)
    inverse = registry.create_operation("sigpy:ifft", axis=1)

    text = dumps_recipe([forward, inverse])
    reconstructed = loads_recipe(text, x.shape)
    assert [op.plugin_id for op in reconstructed] == ["sigpy:fft", "sigpy:ifft"]

    result = x
    for op in reconstructed:
        result = np.asarray(op.apply(result))
    # ortho-normalised centered FFT is unitary -> IFFT(FFT(x)) == x (complex64
    # single-precision round-trip tolerance).
    np.testing.assert_allclose(result, x.astype(result.dtype), atol=1e-4)


# --- dock/engine usability: drive an op through the real evaluator ------------


def test_sigpy_fft_is_faithful_through_the_operation_evaluator():
    from arrayscope.core.view_state import ViewState
    from arrayscope.operations.evaluator import OperationEvaluator
    from arrayscope.operations.pipeline import ArrayDocument

    x = (np.arange(60).reshape(PROBE_SHAPE)).astype("float64")
    op = registry.create_operation("sigpy:fft", axis=1)

    plugin_document = ArrayDocument(x).with_operation(op)
    oracle_document = ArrayDocument(sigpy.fft(x, axes=(1,)))
    state = ViewState.from_shape(plugin_document.current_shape)

    plugin_image = OperationEvaluator(plugin_document).image(state)
    oracle_image = OperationEvaluator(oracle_document).image(state)
    np.testing.assert_allclose(np.asarray(plugin_image.data), np.asarray(oracle_image.data))
