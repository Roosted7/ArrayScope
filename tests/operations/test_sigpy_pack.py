"""Tests for the optional in-process sigpy operation pack.

Covers: op correctness against a direct sigpy reference on *realistic complex*
data (a Shepp-Logan phantom and its k-space), the numeric-precision narrowing
(sigpy's always-complex128 threshold output narrowed back to the input dtype),
the **Tier-2 windowable claim** for the two threshold ops actually passing the
conformance harness (and being honored by the registry gate), the OPAQUE
shape-changing ``sigpy:resize`` (centered zero-pad / center-crop), recipe
round-trip of the float ``lamda`` parameter, and optionality (sigpy-absent -> the
pack registers nothing) + laziness (enumeration never imports sigpy).

The sigpy-dependent assertions ``pytest.importorskip`` cleanly when sigpy is not
installed, mirroring the zfpy/blosc2 codec skip precedents so CI without sigpy
stays green; where sigpy is installed they execute.
"""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.operations import plugins, registry
from arrayscope.operations.packs import sigpy_pack

PROBE_SHAPE = (6, 5, 4)


@pytest.fixture(autouse=True)
def _clean_pack_state():
    registry._reset_operation_packs()
    plugins._reset_plugin_cache()
    yield
    registry._reset_operation_packs()
    plugins._reset_plugin_cache()


def _complex_phantom(shape=(16, 16)) -> np.ndarray:
    """A realistic complex image: a Shepp-Logan phantom with a phase ramp."""

    sp = pytest.importorskip("sigpy")
    mag = np.real(np.asarray(sp.shepp_logan(shape))).astype(np.float64)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    phase = 0.15 * (xx - yy)
    return (mag * np.exp(1j * phase)).astype(np.complex64)


def _kspace(image: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image))).astype(np.complex64)


# --- discovery / optionality / laziness --------------------------------------


def test_sigpy_ops_appear_when_available():
    pytest.importorskip("sigpy")
    ids = {entry.id for entry in registry.all_operations()}
    assert {"sigpy:soft_thresh", "sigpy:hard_thresh", "sigpy:resize"} <= ids
    soft = registry.get_operation_entry("sigpy:soft_thresh")
    assert soft.label == "Soft threshold (sigpy)"
    assert soft.requires_axis is False
    assert soft.changes_shape is False
    resize = registry.get_operation_entry("sigpy:resize")
    assert resize.requires_axis is True
    assert resize.changes_shape is True


def test_pack_specs_exist_independently_of_installation():
    ids = {spec.id for spec in sigpy_pack.pack_specs()}
    assert ids == {"sigpy:soft_thresh", "sigpy:hard_thresh", "sigpy:resize"}


def test_pack_contributes_nothing_when_sigpy_absent(monkeypatch):
    monkeypatch.setattr(sigpy_pack, "sigpy_available", lambda: False)
    registry._reset_operation_packs()
    assert sigpy_pack.register() is False
    ids = {entry.id for entry in registry.all_operations()}
    assert not any(op_id.startswith("sigpy:") for op_id in ids)
    assert "centered_fft" in ids  # built-ins untouched


def test_enumeration_never_imports_sigpy(monkeypatch):
    """Laziness / import-health: listing ops must not import sigpy."""

    import builtins

    real_import = builtins.__import__

    def _guarded(name, *args, **kwargs):
        if name == "sigpy" or name.startswith("sigpy."):
            raise AssertionError("enumerating operations must not import sigpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded)
    # find_spec-based availability + enumeration must not import sigpy.
    assert sigpy_pack.sigpy_available() in (True, False)
    registry.all_operations()  # must not raise


# --- soft / hard threshold correctness + dtype narrowing ---------------------


@pytest.mark.parametrize(
    ("op_id", "kind"), [("sigpy:soft_thresh", "soft"), ("sigpy:hard_thresh", "hard")]
)
def test_threshold_matches_sigpy_reference_on_kspace(op_id, kind):
    sp = pytest.importorskip("sigpy")
    ksp = _kspace(_complex_phantom())
    lamda = 0.5 * float(np.max(np.abs(ksp)))
    op = registry.create_operation(op_id, parameters={"lamda": lamda})
    got = np.asarray(op.apply(ksp))

    thresh_fn = sp.soft_thresh if kind == "soft" else sp.hard_thresh
    ref = np.asarray(thresh_fn(lamda, ksp))  # complex128
    # Narrowed back to complex64, but numerically equal to the sigpy result.
    assert got.dtype == np.complex64
    np.testing.assert_allclose(got, ref.astype(np.complex64), rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    ("in_dtype", "out_dtype"),
    [
        ("complex64", "complex64"),
        ("complex128", "complex128"),
        ("float32", "float32"),
        ("float64", "float64"),
        ("int16", "float32"),
    ],
)
def test_threshold_narrows_dtype(in_dtype, out_dtype):
    pytest.importorskip("sigpy")
    rng = np.random.default_rng(0)
    dt = np.dtype(in_dtype)
    if dt.kind == "c":
        x = (rng.standard_normal(PROBE_SHAPE) + 1j * rng.standard_normal(PROBE_SHAPE)).astype(dt)
    else:
        x = (rng.standard_normal(PROBE_SHAPE) * 4).astype(dt)
    op = registry.create_operation("sigpy:soft_thresh", parameters={"lamda": 0.5})
    got = np.asarray(op.apply(x))
    assert got.dtype == np.dtype(out_dtype)
    # The declared output_dtype adapter agrees with the realized dtype.
    assert np.dtype(op.output_dtype(dt)) == np.dtype(out_dtype)


def test_soft_thresh_shrinks_magnitude():
    pytest.importorskip("sigpy")
    x = np.array([3 + 4j, 0.1 + 0j, -0.3j], dtype=np.complex64)  # |.| = 5, 0.1, 0.3
    op = registry.create_operation("sigpy:soft_thresh", parameters={"lamda": 1.0})
    got = np.asarray(op.apply(x))
    # 5 -> 4 (shrunk by 1), sub-threshold entries -> 0.
    np.testing.assert_allclose(np.abs(got[0]), 4.0, atol=1e-5)
    assert got[1] == 0
    assert got[2] == 0


# --- Tier-2: the threshold ops are windowable AND the harness honors it -------


@pytest.mark.parametrize("op_id", ["sigpy:soft_thresh", "sigpy:hard_thresh"])
def test_threshold_is_tier2_honored_by_conformance_gate(op_id):
    pytest.importorskip("sigpy")
    from arrayscope.operations.capabilities import OperationClass

    op = registry.create_operation(op_id, parameters={"lamda": 0.5})
    # The registry gate property-tested the windowable claim and honored it.
    assert plugins.is_region_honored(op_id, op.axis, op.params) is True
    assert op.execution_class is OperationClass.SHADER_ON_READ
    stats = plugins.region_conformance_stats()
    assert stats["honored"] >= 1
    assert stats["rejected"] == 0


def test_threshold_region_path_equals_whole_array_path():
    """The honored Tier-2 fast path must equal the whole-array result exactly."""

    pytest.importorskip("sigpy")
    from arrayscope.operations.regions import RegionSpec, region_from_index_spec

    ksp = _kspace(_complex_phantom())
    op = registry.create_operation("sigpy:soft_thresh", parameters={"lamda": 0.3})
    whole = np.asarray(op.apply(ksp))
    region = region_from_index_spec(ksp.shape, (slice(2, 10), slice(3, 9)))
    assert isinstance(region, RegionSpec)
    sub = np.asarray(op.apply_to_region(ksp[2:10, 3:9], input_region=region, output_region=region))
    np.testing.assert_array_equal(sub, whole[2:10, 3:9])


# --- sigpy:resize is OPAQUE, shape-changing, dtype-preserving ----------------


def test_resize_zero_pads_and_center_crops():
    pytest.importorskip("sigpy")
    # 1-D centered pad: [0,1,2,3] -> [0,0,0,1,2,3,0,0]; crop back keeps the center.
    x = np.arange(4, dtype=np.complex64)
    pad = registry.create_operation("sigpy:resize", axis=0, parameters={"size": 8})
    y = np.asarray(pad.apply(x))
    assert y.shape == (8,)
    assert y.dtype == np.complex64
    np.testing.assert_array_equal(y, np.array([0, 0, 0, 1, 2, 3, 0, 0], dtype=np.complex64))

    crop = registry.create_operation("sigpy:resize", axis=0, parameters={"size": 2})
    z = np.asarray(crop.apply(x))
    np.testing.assert_array_equal(z, np.array([1, 2], dtype=np.complex64))


def test_resize_output_shape_and_dtype_adapters():
    pytest.importorskip("sigpy")
    op = registry.create_operation("sigpy:resize", axis=1, parameters={"size": 12})
    assert op.output_shape(PROBE_SHAPE) == (6, 12, 4)
    # resize preserves dtype (allocates in the input dtype).
    assert np.dtype(op.output_dtype(np.dtype("complex64"))) == np.complex64
    assert np.dtype(op.output_dtype(np.dtype("float32"))) == np.float32


def test_resize_is_opaque_shape_changer():
    pytest.importorskip("sigpy")
    from arrayscope.operations.capabilities import OperationClass

    op = registry.create_operation("sigpy:resize", axis=0, parameters={"size": 10})
    # A centered resize re-indexes the whole axis and changes shape -> never a
    # Tier-2 windowable claim.
    assert op.execution_class is OperationClass.OPAQUE
    assert plugins.is_region_honored("sigpy:resize", op.axis, op.params) is False


# --- float parameter surface: coercion + recipe round-trip -------------------


def test_float_lamda_param_is_coerced_from_string():
    pytest.importorskip("sigpy")
    # kind="float" -> the create path coerces a string value to float.
    op = registry.create_operation("sigpy:soft_thresh", parameters={"lamda": "0.25"})
    assert op.params == (("lamda", 0.25),)
    assert isinstance(op.params[0][1], float)


def test_soft_thresh_recipe_round_trips_float_param():
    pytest.importorskip("sigpy")
    op = registry.create_operation("sigpy:soft_thresh", parameters={"lamda": 0.75})
    item = plugins.recipe_item_for_plugin_operation(op, enabled=True)
    assert item == {"id": "sigpy:soft_thresh", "parameters": {"lamda": 0.75}, "enabled": True}
    rebuilt = registry.create_operation(item["id"], parameters=item["parameters"])
    assert rebuilt == op  # identity (id + axis + params) compares equal


def test_describe_operation_reads_plugin_param_from_mapping():
    # A PluginOperation stores params in its opaque ``params`` mapping, not as
    # attributes; describe_operation must not AttributeError on them (the crash
    # that took down the whole op-dock refresh when a parameterized plugin op
    # was added).
    pytest.importorskip("sigpy")
    op = registry.create_operation("sigpy:soft_thresh", parameters={"lamda": 0.5})
    assert registry.operation_parameter_value(op, "lamda") == 0.5
    assert "lamda=0.5" in registry.describe_operation(op)
