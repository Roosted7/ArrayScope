"""Tests for the operation add popup (stage 1 of the add flow)."""

from __future__ import annotations

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from arrayscope.ui.operation_add_popup import OperationAddPopup
from arrayscope.ui.operation_listing import build_operation_listing


def _make_popup(qtbot, *, fixed_axis=None, is_enabled=None, accepted=None, needs=None):
    accepted = accepted if accepted is not None else []
    needs = needs if needs is not None else []
    popup = OperationAddPopup(
        build_operation_listing(),
        axis_choices=[("dim 0 [4]", 0), ("dim 1 [5]", 1), ("dim 2 [6]", 2)],
        default_axis=1,
        fixed_axis=fixed_axis,
        is_enabled=is_enabled or (lambda entry: True),
        on_accept=lambda op_id, axis: accepted.append((op_id, axis)),
        on_needs_parameters=lambda op_id, axis: needs.append((op_id, axis)),
    )
    qtbot.addWidget(popup)
    return popup, accepted, needs


def test_popup_builds_with_common_and_group_sections(qtbot):
    popup, _accepted, _needs = _make_popup(qtbot)
    titles = popup.visible_section_titles()
    assert titles[0] == "COMMON"
    assert "TRANSFORM" in titles
    # Common ops (e.g. crop, mean) are listed exactly once, in Common.
    op_ids = popup.visible_operation_ids()
    assert op_ids.count("crop") == 1
    assert op_ids.count("mean") == 1


def test_axis_row_shown_for_axis_op_hidden_for_non_axis_op(qtbot):
    popup, _accepted, _needs = _make_popup(qtbot)
    assert popup.select_operation("mean")  # requires_axis=True
    assert popup._axis_row.isVisibleTo(popup)
    assert popup.select_operation("conjugate")  # requires_axis=False
    assert not popup._axis_row.isVisibleTo(popup)


def test_axis_row_hidden_in_fixed_axis_mode(qtbot):
    popup, _accepted, _needs = _make_popup(qtbot, fixed_axis=2)
    assert popup.select_operation("mean")
    assert not popup._axis_row.isVisibleTo(popup)


def test_parameterless_op_fires_on_accept_with_axis(qtbot):
    popup, accepted, needs = _make_popup(qtbot)
    assert popup.select_operation("mean")
    # default_axis=1 preselected in the axis combo.
    popup.activate_current()
    assert accepted == [("mean", 1)]
    assert needs == []


def test_fixed_axis_used_for_parameterless_activation(qtbot):
    popup, accepted, _needs = _make_popup(qtbot, fixed_axis=2)
    assert popup.select_operation("mean")
    popup.activate_current()
    assert accepted == [("mean", 2)]


def test_parameterized_op_requests_stage_two(qtbot):
    popup, accepted, needs = _make_popup(qtbot)
    assert popup.select_operation("crop")  # has start/stop parameters
    popup.activate_current()
    assert needs == [("crop", 1)]
    assert accepted == []


def test_non_axis_op_activates_with_none_axis(qtbot):
    popup, accepted, _needs = _make_popup(qtbot)
    assert popup.select_operation("conjugate")
    popup.activate_current()
    assert accepted == [("conjugate", None)]


def test_disabled_op_is_unselectable(qtbot):
    popup, _accepted, _needs = _make_popup(qtbot, is_enabled=lambda entry: entry.id != "mean")
    # The disabled item is present but cannot become the current selection.
    assert not popup.select_operation("mean")


def test_native_pointwise_section_is_in_everyday_collapsed_listing(qtbot):
    popup, _accepted, _needs = _make_popup(qtbot)
    assert "POINTWISE" in popup.visible_section_titles()
    assert "soft_threshold" in popup.visible_operation_ids()
    popup.set_expanded(True)
    assert "POINTWISE" in popup.visible_section_titles()
    assert "soft_threshold" in popup.visible_operation_ids()
    popup.set_expanded(False)
    assert "POINTWISE" in popup.visible_section_titles()
