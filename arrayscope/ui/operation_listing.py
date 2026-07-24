"""Grouped operation-listing model shared by the op menus and add popup.

Both the dimension-chip "+" menu and the dock's add popup present the same
catalogue of operations, grouped and ordered the same way: a pinned "Common"
section first, then one section per taxonomy group, with the less-used backend
groups (SigPy / BART / User / Other) tucked behind a trailing "More" partition.

This module is deliberately Qt-free -- it turns a flat sequence of
:class:`~arrayscope.operations.registry.OperationEntry` into an ordered list of
:class:`ListingSection` -- so the grouping logic is unit-testable without a
display and both UI surfaces render from one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from arrayscope.operations.registry import (
    COMMON_OPERATION_IDS,
    DEFAULT_GROUP_ORDER,
    OperationEntry,
    is_common,
)

# Groups whose sections live behind the trailing "More" fold-out. These are the
# optional / backend-specific families a first-time user rarely reaches for, so
# they stay collapsed until asked for while the everyday groups stay visible.
MORE_GROUPS: tuple[str, ...] = ("SigPy", "BART", "User", "Other")


@dataclass(frozen=True)
class ListingSection:
    """One titled block of operations in the listing.

    ``is_more`` flags a section that belongs to the collapsed "More" partition
    (a UI surface may hide these behind a fold-out); everyday sections carry
    ``is_more=False``.
    """

    title: str
    entries: tuple[OperationEntry, ...]
    is_more: bool = False


def _effective_group(entry: OperationEntry) -> str:
    """The taxonomy group an entry is filed under, mapping strays to "Other"."""

    return entry.group if entry.group in DEFAULT_GROUP_ORDER else "Other"


def build_operation_listing(entries) -> list[ListingSection]:
    """Group ``entries`` into ordered listing sections.

    Section 1 is "Common" (the entries in :data:`COMMON_OPERATION_IDS`, in that
    order). Then one section per group in :data:`DEFAULT_GROUP_ORDER`, each
    excluding its Common ops (so nothing is listed twice) and dropped entirely
    when that leaves it empty. Groups in :data:`MORE_GROUPS` are flagged
    ``is_more=True`` for the collapsed partition. An entry whose group is not in
    the taxonomy is filed under "Other".
    """

    entries = tuple(entries)
    by_id = {entry.id: entry for entry in entries}

    sections: list[ListingSection] = []

    # 1. The pinned "Common" section, in COMMON_OPERATION_IDS order.
    common = tuple(by_id[op_id] for op_id in COMMON_OPERATION_IDS if op_id in by_id)
    if common:
        sections.append(ListingSection("Common", common, is_more=False))

    # 2. One section per real group, excluding ops already pinned to Common.
    for group in DEFAULT_GROUP_ORDER:
        if group == "Common":
            # "Common" is a presentation concern, never a real group value.
            continue
        group_entries = tuple(
            entry
            for entry in entries
            if _effective_group(entry) == group and not is_common(entry.id)
        )
        if not group_entries:
            continue
        sections.append(ListingSection(group, group_entries, is_more=group in MORE_GROUPS))

    return sections
