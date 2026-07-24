"""Tests for the Qt-free operation parameter-form model.

Covers the metadata-driven default form (fields seeded from parameter defaults /
bounds), the crop provider's interdependence (editing one bound nudges the other
so start < stop always holds) and its derived output-length line, the resize
provider seeding ``size`` from the current axis length, and validation.
"""

from __future__ import annotations

import pytest

from arrayscope.operations import plugins, registry
from arrayscope.operations.parameter_forms import (
    DerivedValue,
    ParameterField,
    build_parameter_form,
)
from arrayscope.operations.registry import get_operation_entry


@pytest.fixture(autouse=True)
def _clean_pack_state():
    registry._reset_operation_packs()
    plugins._reset_plugin_cache()
    yield
    registry._reset_operation_packs()
    plugins._reset_plugin_cache()


# --- default form ------------------------------------------------------------


def test_parameterless_op_has_no_form():
    entry = get_operation_entry("mean")
    assert build_parameter_form(entry, shape=(4, 5, 6), axis=1) is None


def test_default_form_seeds_from_metadata():
    # sigpy:soft_thresh declares lamda default=0.0, minimum=0.0, step=0.01.
    pytest.importorskip("sigpy")
    registry.load_operation_packs()
    entry = get_operation_entry("sigpy:soft_thresh")
    form = build_parameter_form(entry, shape=(4, 5, 6), axis=None)
    assert form is not None
    field = form.field("lamda")
    assert isinstance(field, ParameterField)
    assert field.kind == "float"
    assert field.value == 0.0
    assert field.minimum == 0.0
    assert field.step == 0.01
    assert field.description
    # Values are ready to hand to create_operation.
    assert form.values() == {"lamda": 0.0}


def test_default_form_validate_bounds():
    pytest.importorskip("sigpy")
    registry.load_operation_packs()
    entry = get_operation_entry("sigpy:soft_thresh")
    form = build_parameter_form(entry, shape=(4, 5, 6), axis=None)
    assert form.validate() is None
    form.set_value("lamda", -1.0)
    message = form.validate()
    assert message is not None
    assert "at least" in message


# --- crop provider: interdependence + derived value --------------------------


def test_crop_form_defaults_span_full_axis():
    entry = get_operation_entry("crop")
    form = build_parameter_form(entry, shape=(4, 10, 6), axis=1)
    assert form.field("start").value == 0
    assert form.field("stop").value == 10
    assert form.field("start").maximum == 9
    assert form.field("stop").maximum == 10
    assert form.derived() == [DerivedValue("Output length", "10")]


def test_crop_editing_start_past_stop_nudges_stop():
    entry = get_operation_entry("crop")
    form = build_parameter_form(entry, shape=(4, 10, 6), axis=1)
    form.set_value("stop", 5)
    # Push start up to 5 -> stop must be nudged to keep start < stop.
    form.set_value("start", 5)
    assert form.field("start").value == 5
    assert form.field("stop").value == 6
    assert form.derived() == [DerivedValue("Output length", "1")]


def test_crop_editing_start_at_ceiling_pulls_start_back():
    entry = get_operation_entry("crop")
    form = build_parameter_form(entry, shape=(4, 10, 6), axis=1)
    # stop is already at its ceiling (10); driving start to 10 cannot push stop
    # higher, so start is pulled back to 9.
    form.set_value("start", 10)
    assert form.field("stop").value == 10
    assert form.field("start").value == 9
    assert form.derived() == [DerivedValue("Output length", "1")]


def test_crop_editing_stop_below_start_nudges_start():
    entry = get_operation_entry("crop")
    form = build_parameter_form(entry, shape=(4, 10, 6), axis=1)
    form.set_value("start", 4)
    form.set_value("stop", 4)
    # stop crossed start -> start pulled down to 3.
    assert form.field("stop").value == 4
    assert form.field("start").value == 3


def test_crop_without_context_still_builds():
    entry = get_operation_entry("crop")
    form = build_parameter_form(entry, shape=None, axis=None)
    assert form.field("start").value == 0
    assert form.field("stop").value == 1
    assert form.field("start").maximum is None


# --- resize provider: context default ----------------------------------------


def test_resize_form_defaults_to_axis_length():
    pytest.importorskip("sigpy")
    registry.load_operation_packs()
    entry = get_operation_entry("sigpy:resize")
    form = build_parameter_form(entry, shape=(4, 8, 6), axis=1)
    assert form.field("size").value == 8
    assert form.field("size").minimum == 1
    assert DerivedValue("Current length", "8") in form.derived()
    # Editing the target updates the derived output line.
    form.set_value("size", 16)
    assert DerivedValue("Output length", "16") in form.derived()


def test_resize_form_without_context_falls_back():
    pytest.importorskip("sigpy")
    registry.load_operation_packs()
    entry = get_operation_entry("sigpy:resize")
    form = build_parameter_form(entry, shape=None, axis=None)
    assert form.field("size").value == 1
    # No current-length line without context, but the output line is present.
    assert form.derived() == [DerivedValue("Output length", "1")]


def test_form_values_feed_create_operation():
    entry = get_operation_entry("crop")
    form = build_parameter_form(entry, shape=(4, 10, 6), axis=1)
    form.set_value("start", 2)
    form.set_value("stop", 7)
    operation = registry.create_operation("crop", axis=1, parameters=form.values())
    assert operation.output_shape((4, 10, 6)) == (4, 5, 6)
