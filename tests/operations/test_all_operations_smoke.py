"""Non-crash smoke harness over *every* registered operation (built-ins + packs).

The guarantee: any operation reachable from ``all_operations()`` -- a built-in, a
first-party pack op (sigpy/bart), or a future addition -- can be built from its
parameter form and round-tripped through ``output_shape`` / ``output_dtype`` /
``capabilities`` / ``apply`` without a hand-written per-op test. A new op that
declares a parameter it never handles, or whose ``apply`` crashes, or whose
predicted ``output_shape`` disagrees with what ``apply`` actually produces, fails
CI here -- and the failure names the offending op id.

Per-op context (shape + axis + which dtypes are legitimate) is carried by a
small, documented expectations map. Ops absent from it use the generic default;
only the genuinely dtype/shape-constrained ops (the complex real/imag pair) need
an entry.
"""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.operations import library, plugins, registry
from arrayscope.operations.parameter_forms import build_parameter_form

# Generic context used unless an op overrides it below.
_DEFAULT_SHAPE = (6, 8, 10)
_DEFAULT_AXIS = 1
_DEFAULT_DTYPES = (np.dtype(np.float32), np.dtype(np.complex64))


# Per-op expectations, kept deliberately small: only ops with a genuine
# shape/dtype constraint appear. Each value overrides shape / axis / dtypes.
#
# - combine_real_imag: rejects complex input and needs a size-2 axis (it folds a
#   real/imag pair into one complex sample) -> float32 only, size-2 axis.
# - split_complex: requires complex input and a size-1 axis (it unpacks one
#   complex sample into a real/imag pair) -> complex64 only, size-1 axis.
_EXPECTATIONS: dict[str, dict] = {
    "combine_real_imag": {
        "shape": (6, 2, 10),
        "axis": 1,
        "dtypes": (np.dtype(np.float32),),
    },
    "split_complex": {
        "shape": (6, 1, 10),
        "axis": 1,
        "dtypes": (np.dtype(np.complex64),),
    },
    "squeeze": {
        "shape": (6, 1, 10),
        "axis": 1,
    },
}


@pytest.fixture(autouse=True)
def _clean_pack_state():
    registry._reset_operation_packs()
    plugins._reset_plugin_cache()
    yield
    registry._reset_operation_packs()
    plugins._reset_plugin_cache()


@pytest.fixture
def shape_changing_user_operation(tmp_path, monkeypatch):
    operations_dir = tmp_path / "operations"
    monkeypatch.setattr(library, "user_operations_directory", lambda: str(operations_dir))
    source = tmp_path / "decimate.py"
    source.write_text("def decimate(data):\n    return data[..., ::2]\n")
    operation_id = library.import_custom_operation(str(source), "decimate", changes_shape=True)
    yield operation_id
    library.remove_user_operation(operation_id)


def _context_for(operation_id: str):
    override = _EXPECTATIONS.get(operation_id, {})
    shape = override.get("shape", _DEFAULT_SHAPE)
    axis = override.get("axis", _DEFAULT_AXIS)
    dtypes = override.get("dtypes", _DEFAULT_DTYPES)
    return shape, axis, dtypes


def _sample_array(shape, dtype) -> np.ndarray:
    rng = np.random.default_rng(0x5A1E)
    real = rng.standard_normal(shape)
    if np.dtype(dtype).kind == "c":
        return (real + 1j * rng.standard_normal(shape)).astype(dtype)
    return real.astype(dtype)


def test_demoted_numpy_wrapper_packs_only_expose_real_bart_examples():
    """NumPy-equivalent pack ops stay demoted; real BART examples are explicit."""

    entries = registry.all_operations()
    assert not any(entry.id.startswith("sigpy:") for entry in entries)
    bart_entries = {entry.id: entry for entry in entries if entry.id.startswith("bart:")}
    assert set(bart_entries) == {"bart:ecalib", "bart:walsh"}
    assert all(entry.unavailable_reason for entry in bart_entries.values())


def test_every_operation_builds_and_applies_without_crashing(shape_changing_user_operation):
    entries = registry.all_operations()
    assert entries, "no operations registered"
    assert shape_changing_user_operation in {entry.id for entry in entries}

    for entry in entries:
        if entry.unavailable_reason:
            continue
        shape, axis, dtypes = _context_for(entry.id)
        op_axis = axis if entry.requires_axis else None

        form = build_parameter_form(entry, shape=shape, axis=op_axis)
        parameters = form.values() if form is not None else {}
        if form is not None:
            assert form.validate() is None, f"{entry.id}: default form failed validation"

        try:
            operation = registry.create_operation(entry.id, axis=op_axis, parameters=parameters)
        except Exception as exc:  # pragma: no cover - failure path names the op
            raise AssertionError(f"{entry.id}: create_operation crashed: {exc!r}") from exc

        for dtype in dtypes:
            data = _sample_array(shape, dtype)
            try:
                predicted_shape = tuple(operation.output_shape(shape))
                predicted_dtype = operation.output_dtype(dtype)
                capabilities = operation.capabilities(shape, dtype)
                result = operation.apply(data)
            except Exception as exc:  # pragma: no cover - failure path names the op
                raise AssertionError(
                    f"{entry.id} ({np.dtype(dtype).name}): pipeline call crashed: {exc!r}"
                ) from exc

            assert capabilities is not None, f"{entry.id}: capabilities returned None"
            result = np.asarray(result)
            assert result.shape == predicted_shape, (
                f"{entry.id} ({np.dtype(dtype).name}): apply produced shape "
                f"{result.shape}, output_shape predicted {predicted_shape}"
            )
            if predicted_dtype is not None:
                # A prediction is a promise; the produced dtype must match it.
                assert result.dtype == np.dtype(predicted_dtype), (
                    f"{entry.id} ({np.dtype(dtype).name}): apply produced dtype "
                    f"{result.dtype}, output_dtype predicted {np.dtype(predicted_dtype)}"
                )
