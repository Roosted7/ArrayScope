"""Tests for the operation add popup (stage 1 of the add flow)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from arrayscope.ui.operation_add_popup import OperationAddPopup
from arrayscope.ui.operation_listing import build_operation_listing


def _make_popup(
    qtbot,
    *,
    fixed_axis=None,
    is_enabled=None,
    accepted=None,
    needs=None,
    searched=None,
):
    accepted = accepted if accepted is not None else []
    needs = needs if needs is not None else []
    searched = searched if searched is not None else []
    popup = OperationAddPopup(
        build_operation_listing(),
        axis_choices=[("dim 0 [4]", 0), ("dim 1 [5]", 1), ("dim 2 [6]", 2)],
        default_axis=1,
        fixed_axis=fixed_axis,
        is_enabled=is_enabled or (lambda entry: True),
        on_search=lambda: searched.append(True),
        on_accept=lambda op_id, axis: accepted.append((op_id, axis)),
        on_needs_parameters=lambda op_id, axis: needs.append((op_id, axis)),
    )
    qtbot.addWidget(popup)
    return popup, accepted, needs, searched


def test_browse_accordion_keeps_common_small_and_opens_one_category_at_a_time(qtbot):
    # The native toolbox is ~40 operations. A previous fold revealed all of
    # them as one long scroll; browsing must instead keep Common compact and
    # reveal at most one taxonomy group while the command palette owns search.
    popup, _accepted, _needs, searched = _make_popup(qtbot)

    collapsed_titles = popup.visible_section_titles()
    collapsed_height = popup._list.height()
    collapsed_popup_height = popup.height()
    collapsed_ids = popup.visible_operation_ids()
    assert collapsed_titles == ["COMMON"]
    assert popup.visible_category_titles() == []
    assert {"mean", "crop"} <= set(collapsed_ids)

    popup.set_expanded(True)
    assert popup.visible_section_titles() == ["COMMON", "BROWSE BY CATEGORY"]
    assert "Transform" in popup.visible_category_titles()
    assert "Complex" in popup.visible_category_titles()
    # Opening the category chooser alone does not flatten every operation into
    # the list.
    assert popup.visible_operation_ids() == collapsed_ids

    popup.set_expanded_category("Transform")
    assert "reverse" in popup.visible_operation_ids()
    assert "conjugate" not in popup.visible_operation_ids()
    assert popup._list.height() > collapsed_height

    # Switching categories must close the previous group. Keeping both open
    # recreates the long flat catalogue the accordion was introduced to avoid.
    popup.set_expanded_category("Complex")
    op_ids = popup.visible_operation_ids()
    assert "conjugate" in op_ids
    assert "reverse" not in op_ids
    # Common ops are listed exactly once, never repeated in their home group.
    assert op_ids.count("crop") == 1
    assert op_ids.count("mean") == 1

    # Collapsing round-trips to the compact opening state.
    popup.set_expanded(False)
    assert popup.visible_section_titles() == ["COMMON"]
    assert popup.visible_category_titles() == []
    assert popup._list.height() == collapsed_height
    assert popup.height() == collapsed_popup_height

    search_item = next(
        popup._list.item(row)
        for row in range(popup._list.count())
        if popup._list.item(row).text().startswith("Search all operations")
    )
    popup._list.itemActivated.emit(search_item)
    assert searched == [True]


def test_select_operation_reveals_a_folded_away_operation(qtbot):
    # Selecting by id must not depend on which side of the fold an operation
    # landed on -- "not found" for an operation that plainly exists would be a
    # trap for callers and for a future search box.
    popup, _accepted, _needs, _searched = _make_popup(qtbot)
    assert "conjugate" not in popup.visible_operation_ids()

    assert popup.select_operation("conjugate")

    assert popup._expanded
    assert popup._expanded_category == "Complex"
    assert "conjugate" in popup.visible_operation_ids()


@pytest.mark.parametrize(
    ("operation_id", "fixed_axis", "expected_axis", "axis_row_visible"),
    [
        pytest.param("mean", None, 1, True, id="chosen-axis"),
        pytest.param("mean", 2, 2, False, id="fixed-axis"),
        pytest.param("conjugate", None, None, False, id="no-axis"),
    ],
)
def test_parameterless_activation_uses_the_operation_axis_contract(
    qtbot,
    operation_id,
    fixed_axis,
    expected_axis,
    axis_row_visible,
):
    popup, accepted, needs, _searched = _make_popup(qtbot, fixed_axis=fixed_axis)
    if operation_id == "conjugate":
        # Exercise the visible -> hidden transition, not merely the initial
        # state: changing selection used to leave stale parameter UI behind.
        assert popup.select_operation("mean")
        assert popup._axis_row.isVisibleTo(popup)
    assert popup.select_operation(operation_id)
    assert popup._axis_row.isVisibleTo(popup) is axis_row_visible

    popup.activate_current()
    assert accepted == [(operation_id, expected_axis)]
    assert needs == []


def test_parameterized_op_requests_stage_two(qtbot):
    popup, accepted, needs, _searched = _make_popup(qtbot)
    assert popup.select_operation("crop")  # has start/stop parameters
    popup.activate_current()
    assert needs == [("crop", 1)]
    assert accepted == []


def test_disabled_op_is_unselectable(qtbot):
    popup, _accepted, _needs, _searched = _make_popup(
        qtbot,
        is_enabled=lambda entry: entry.id != "mean",
    )
    # The disabled item is present but cannot become the current selection.
    assert not popup.select_operation("mean")
