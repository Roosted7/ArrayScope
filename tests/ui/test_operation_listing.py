"""Tests for the grouped operation-listing model.

Two layers are pinned here:

1. The pure presentation split (explicit ``grouped`` / ``common_ids`` /
   ``more_groups`` arguments): a pinned "Common" section first (in
   ``common_ids`` order), Common ops excluded from their home group, emptied
   groups dropped, and the ``more_groups`` flagged for the collapsed "More"
   partition.
2. The library-backed default (no arguments): the listing reflects the
   operation library, so a hidden op disappears, a layout group override moves
   an op's section, and a ``common_ids`` override is honored.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from arrayscope.operations import library
from arrayscope.operations.registry import OperationEntry
from arrayscope.ui.operation_listing import ListingSection, build_operation_listing


def _entry(op_id, group="Other", parameters=()):
    return OperationEntry(
        id=op_id,
        label=op_id.title(),
        operation_type=object,
        parameters=tuple(parameters),
        group=group,
    )


def _grouped(*pairs):
    """Build an ordered ``[(group, [entries])]`` listing from (group, ids) pairs."""

    return [(group, [_entry(op_id, group=group) for op_id in ids]) for group, ids in pairs]


# ---------------------------------------------------------------------------
# pure presentation split (explicit arguments)
# ---------------------------------------------------------------------------


def test_common_section_is_first_and_in_declared_order():
    grouped = _grouped(("Reduce", ["mean", "sum"]), ("Transform", ["crop"]))
    sections = build_operation_listing(grouped, common_ids=("crop", "mean"), more_groups=())
    assert sections[0].title == "Common"
    # Common section follows common_ids order, not the grouped order.
    assert [entry.id for entry in sections[0].entries] == ["crop", "mean"]


def test_common_ops_not_repeated_in_home_group():
    grouped = _grouped(("Reduce", ["mean", "sum"]), ("Transform", ["crop", "reverse"]))
    sections = build_operation_listing(grouped, common_ids=("mean", "crop"), more_groups=())
    common_ids = {"mean", "crop"}
    for section in sections[1:]:
        assert not (common_ids & {entry.id for entry in section.entries})


def test_emptied_group_is_dropped():
    # A group whose only member is a Common op must not produce a section.
    grouped = _grouped(("Reduce", ["mean"]), ("Transform", ["crop"]))
    sections = build_operation_listing(grouped, common_ids=("mean", "crop"), more_groups=())
    assert [section.title for section in sections] == ["Common"]


def test_more_groups_flagged_and_ordered_last():
    grouped = _grouped(
        ("Transform", ["reverse"]),
        ("SigPy", ["plug:a"]),
        ("User", ["plug:b"]),
    )
    sections = build_operation_listing(grouped, common_ids=(), more_groups=("SigPy", "User"))
    more = [section for section in sections if section.is_more]
    not_more = [section for section in sections if not section.is_more]
    assert {section.title for section in more} == {"SigPy", "User"}
    # grouped order is preserved, so the everyday Transform section comes first.
    assert not_more[0].title == "Transform"
    first_more = min(i for i, s in enumerate(sections) if s.is_more)
    last_plain = max(i for i, s in enumerate(sections) if not s.is_more)
    assert first_more > last_plain


def test_grouped_order_is_preserved():
    grouped = _grouped(("Complex", ["conjugate"]), ("Reduce", ["mean"]))
    sections = build_operation_listing(grouped, common_ids=(), more_groups=())
    assert [section.title for section in sections] == ["Complex", "Reduce"]


def test_section_is_frozen_dataclass():
    section = ListingSection("X", (), is_more=True)
    assert section.title == "X"
    assert replace(section, is_more=False).is_more is False


# ---------------------------------------------------------------------------
# library-backed default (no arguments)
# ---------------------------------------------------------------------------


@pytest.fixture
def _isolated_library(tmp_path, monkeypatch):
    ops_dir = tmp_path / "operations"
    monkeypatch.setattr(library, "user_operations_directory", lambda: str(ops_dir))
    library.reset_layout()
    library.refresh_user_operations()
    yield
    library.reset_layout()
    library.refresh_user_operations()


def _listing_ids(sections):
    return {entry.id for section in sections for entry in section.entries}


def _section_of(sections, op_id):
    for section in sections:
        if any(entry.id == op_id for entry in section.entries):
            return section.title
    return None


@pytest.mark.usefixtures("_isolated_library")
def test_default_source_is_the_library():
    sections = build_operation_listing()
    assert sections[0].title == "Common"
    # Built-in ops flow through from the library with no explicit arguments.
    assert {"mean", "crop", "conjugate"} <= _listing_ids(sections)


@pytest.mark.usefixtures("_isolated_library")
def test_hidden_op_disappears_from_listing():
    assert "reverse" in _listing_ids(build_operation_listing())
    library.set_operation_hidden("reverse", True)
    try:
        assert "reverse" not in _listing_ids(build_operation_listing())
    finally:
        library.set_operation_hidden("reverse", False)


@pytest.mark.usefixtures("_isolated_library")
def test_layout_group_override_moves_op_section():
    # "reverse" declares group Transform by default.
    assert _section_of(build_operation_listing(), "reverse") == "Transform"
    library.apply_library_layout(op_groups={"reverse": "Reduce"})
    assert _section_of(build_operation_listing(), "reverse") == "Reduce"


@pytest.mark.usefixtures("_isolated_library")
def test_common_ids_override_respected():
    # By default "conjugate" is not a Common op; a layout override pins it.
    default_common = build_operation_listing()[0]
    assert "conjugate" not in {entry.id for entry in default_common.entries}

    library.apply_library_layout(common_ids=["conjugate"])
    common = build_operation_listing()[0]
    assert [entry.id for entry in common.entries] == ["conjugate"]
