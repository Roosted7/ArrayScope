"""Tests for the Qt-free grouped operation-listing model.

Pins the presentation contract both op menus render from: a pinned "Common"
section first (in COMMON_OPERATION_IDS order), one section per taxonomy group
with Common ops excluded from their home group, emptied groups dropped, and the
backend groups flagged for the collapsed "More" partition.
"""

from __future__ import annotations

from dataclasses import replace

from arrayscope.operations.registry import (
    COMMON_OPERATION_IDS,
    OperationEntry,
    operation_entries,
)
from arrayscope.ui.operation_listing import (
    MORE_GROUPS,
    ListingSection,
    build_operation_listing,
)


def _entry(op_id, group="Other", parameters=()):
    return OperationEntry(
        id=op_id,
        label=op_id.title(),
        operation_type=object,
        parameters=tuple(parameters),
        group=group,
    )


def test_common_section_is_first_and_in_declared_order():
    sections = build_operation_listing(operation_entries())
    assert sections[0].title == "Common"
    listed = [entry.id for entry in sections[0].entries]
    expected = [op_id for op_id in COMMON_OPERATION_IDS if op_id in listed]
    assert listed == expected


def test_common_ops_not_repeated_in_home_group():
    sections = build_operation_listing(operation_entries())
    common_ids = set(COMMON_OPERATION_IDS)
    for section in sections[1:]:
        assert not (common_ids & {entry.id for entry in section.entries})


def test_emptied_group_is_dropped():
    # A group whose only member is a Common op must not produce a section.
    entries = [
        _entry("mean", group="Reduce"),  # Common -> excluded from Reduce
        _entry("crop", group="Transform"),  # Common -> excluded from Transform
    ]
    sections = build_operation_listing(entries)
    titles = [section.title for section in sections]
    assert titles == ["Common"]


def test_more_groups_flagged_and_ordered_last():
    entries = [
        _entry("reverse", group="Transform"),
        _entry("plug:a", group="SigPy"),
        _entry("plug:b", group="User"),
    ]
    sections = build_operation_listing(entries)
    more = [section for section in sections if section.is_more]
    not_more = [section for section in sections if not section.is_more]
    assert {section.title for section in more} == {"SigPy", "User"}
    assert all(section.title in MORE_GROUPS for section in more)
    # Every more-section comes after every everyday section.
    first_more = min(i for i, s in enumerate(sections) if s.is_more)
    last_plain = max(i for i, s in enumerate(sections) if not s.is_more)
    assert first_more > last_plain
    assert next(s.title for s in not_more) == "Transform"


def test_unknown_group_files_under_other_more_section():
    entries = [_entry("weird:op", group="Nonexistent")]
    sections = build_operation_listing(entries)
    assert [s.title for s in sections] == ["Other"]
    assert sections[0].is_more is True


def test_section_is_frozen_dataclass():
    section = ListingSection("X", (), is_more=True)
    assert section.title == "X"
    assert replace(section, is_more=False).is_more is False
