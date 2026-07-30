"""Tests for the Qt-free operation parameter-form model.

Covers the metadata-driven default form (fields seeded from parameter defaults /
bounds), the crop provider's interdependence (editing one bound nudges the other
so start < stop always holds) and its derived output-length line, the resize
provider seeding ``size`` from the current axis length, and validation.
"""

from __future__ import annotations

import os

import pytest

from arrayscope.operations import plugins, registry
from arrayscope.operations.parameter_forms import (
    DerivedValue,
    ParameterField,
    build_parameter_form,
)
from arrayscope.operations.registry import (
    OperationEntry,
    OperationParameter,
    get_operation_entry,
)


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
    entry = get_operation_entry("soft_threshold")
    form = build_parameter_form(entry, shape=(4, 5, 6), axis=None)
    assert form is not None
    field = form.field("threshold")
    assert isinstance(field, ParameterField)
    assert field.kind == "float"
    assert field.value == 0.1
    assert field.minimum == 0.0
    assert field.step == 0.01
    assert field.description
    assert form.values() == {"threshold": 0.1}


def test_default_form_validate_bounds():
    entry = get_operation_entry("soft_threshold")
    form = build_parameter_form(entry, shape=(4, 5, 6), axis=None)
    assert form.validate() is None
    form.set_value("threshold", -1.0)
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


def test_pad_form_centres_to_next_power_of_two_and_tracks_asymmetry():
    entry = get_operation_entry("pad")
    form = build_parameter_form(entry, shape=(4, 10, 6), axis=1)

    assert form.values() == {"before": 3, "after": 3, "mode": 0}
    assert form.derived() == [
        DerivedValue("Current length", "10"),
        DerivedValue("Output length", "16"),
    ]
    form.set_value("before", 1)
    form.set_value("after", 4)
    assert form.derived()[-1] == DerivedValue("Output length", "15")


def test_resample_form_accepts_fractional_factor_and_derives_rounded_length():
    entry = get_operation_entry("resample")
    form = build_parameter_form(entry, shape=(4, 10, 6), axis=1)

    form.set_value("factor", 0.65)

    assert form.values()["factor"] == 0.65
    assert form.derived() == [
        DerivedValue("Current length", "10"),
        DerivedValue("Output length", "7"),
    ]


def test_transpose_form_bounds_and_defaults_partner_axis_from_context():
    entry = get_operation_entry("transpose")
    form = build_parameter_form(entry, shape=(4, 10, 6), axis=1)

    assert form.field("other_axis").value == 2
    assert form.field("other_axis").maximum == 2


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


def test_form_values_feed_create_operation():
    entry = get_operation_entry("crop")
    form = build_parameter_form(entry, shape=(4, 10, 6), axis=1)
    form.set_value("start", 2)
    form.set_value("stop", 7)
    operation = registry.create_operation("crop", axis=1, parameters=form.values())
    assert operation.output_shape((4, 10, 6)) == (4, 5, 6)


# --- non-numeric kinds -------------------------------------------------------


def _kinds_entry(*parameters: OperationParameter) -> OperationEntry:
    return OperationEntry(
        id="test:kinds",
        label="Kinds",
        operation_type=object,
        parameters=parameters,
        requires_axis=False,
    )


def test_non_numeric_fields_seed_from_their_kind():
    form = build_parameter_form(
        _kinds_entry(
            OperationParameter("mode", "Mode", kind="enum", choices=("reflect", "edge", "zero")),
            OperationParameter("wavelet", "Wavelet", kind="string"),
            OperationParameter("verbose", "Verbose", kind="bool"),
            OperationParameter("model", "Model", kind="path"),
        )
    )

    assert form.values() == {"mode": "reflect", "wavelet": "", "verbose": False, "model": ""}
    assert form.field("mode").choices == ("reflect", "edge", "zero")
    assert form.validate() is None


def test_enum_rejects_a_value_outside_its_choices_instead_of_clamping():
    form = build_parameter_form(
        _kinds_entry(
            OperationParameter(
                "mode", "Mode", kind="enum", choices=("reflect", "edge"), default="edge"
            )
        )
    )
    assert form.field("mode").value == "edge"

    with pytest.raises(ValueError, match="reflect, edge"):
        form.set_value("mode", "wrap")
    assert form.field("mode").value == "edge"


def test_bool_and_path_fields_coerce_their_serialized_forms():
    form = build_parameter_form(
        _kinds_entry(
            OperationParameter("verbose", "Verbose", kind="bool", default=True),
            OperationParameter("model", "Model", kind="path"),
        )
    )
    assert form.field("verbose").value is True

    # Recipes carry real booleans; a hand-written wrapper may carry the token.
    form.set_value("verbose", "false")
    assert form.field("verbose").value is False
    form.set_value("verbose", 1)
    assert form.field("verbose").value is True
    with pytest.raises(ValueError, match="not a boolean"):
        form.set_value("verbose", "maybe")

    # A path is expanded but never required to exist: availability owns that.
    form.set_value("model", "~/weights.pt")
    assert form.field("model").value == os.path.expanduser("~/weights.pt")


def test_bounds_and_choices_are_refused_where_they_mean_nothing():
    with pytest.raises(ValueError, match="must declare choices"):
        OperationParameter("mode", "Mode", kind="enum")
    with pytest.raises(ValueError, match="must not declare choices"):
        OperationParameter("count", "Count", kind="int", choices=("a",))
    with pytest.raises(ValueError, match="cannot declare minimum"):
        OperationParameter("name", "Name", kind="string", minimum=0)
    with pytest.raises(ValueError, match="invalid default"):
        OperationParameter("mode", "Mode", kind="enum", choices=("a",), default="b")
    with pytest.raises(ValueError, match="unsupported kind"):
        OperationParameter("tag", "Tag", kind="str")


# --- axis-set field (an operation's subject, not one of its parameters) -------


def test_an_axis_set_op_gets_a_form_even_with_no_parameters():
    """Parameterless is not selectionless: the axis set still needs choosing."""

    entry = get_operation_entry("centered_fftn")
    form = build_parameter_form(
        entry,
        shape=(4, 5, 6),
        axis_choices=(("d0", 0), ("d1 (Y)", 1), ("d2 (X)", 2)),
        display_axes=(1, 2),
    )

    assert form is not None
    assert form.fields == []
    assert form.axes_field is not None
    # Defaults to what the user is looking at, so a 2-slab op arrives pointing
    # at the displayed image rather than at axes 0 and 1 by accident.
    assert form.selected_axes() == (1, 2)
    assert [label for label, _axis in form.axes_field.choices] == ["d0", "d1 (Y)", "d2 (X)"]
    assert form.validate() is None


def test_an_axis_set_form_prefers_an_explicit_selection_over_the_display():
    entry = get_operation_entry("mean_axes")
    form = build_parameter_form(entry, shape=(4, 5, 6), axes=(2, 0), display_axes=(1, 2))

    assert form.selected_axes() == (0, 2)


def test_an_axis_set_form_falls_back_to_the_leading_axes_without_a_view():
    entry = get_operation_entry("mean_axes")
    form = build_parameter_form(entry, shape=(4, 5, 6))

    assert form.selected_axes() == (0, 1)
    assert [label for label, _axis in form.axes_field.choices] == ["d0", "d1", "d2"]


def test_an_axis_set_form_refuses_an_empty_selection():
    entry = get_operation_entry("centered_fftn")
    form = build_parameter_form(entry, shape=(4, 5, 6), display_axes=(1, 2))

    form.set_selected_axes(())
    message = form.validate()

    assert message is not None
    assert "at least 1" in message


def test_a_single_axis_op_has_no_axis_set_field():
    form = build_parameter_form(get_operation_entry("crop"), shape=(4, 5, 6), axis=1)

    assert form.axes_field is None
    assert form.selected_axes() == ()
    with pytest.raises(KeyError):
        form.set_selected_axes((0,))
