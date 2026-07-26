"""Grouped operation-listing model shared by the op menus and add popup.

Both the dimension-chip "+" menu and the dock's add popup present the same
catalogue of operations, grouped and ordered the same way: a pinned "Common"
section first, then one section per taxonomy group. Groups in the trailing
"more" partition become a submenu in the chip menu and individually expandable
categories in the add popup.

The catalogue itself -- which ops exist, in what groups, in what order, and
which are hidden -- is owned by :mod:`arrayscope.operations.library`. This module
consumes the library's already-grouped listing (hidden ops excluded, layout
group/order overrides applied) and applies the pure presentation split: pull the
"Common" ops into a pinned first section and flag the "More" groups. By default
it reads the library directly, so both UI surfaces reflect the user's hidden-op
and layout choices; tests exercise the pure split by passing explicit arguments.

This module is deliberately Qt-free -- it turns the grouped listing into an
ordered list of :class:`ListingSection` -- so the grouping logic is
unit-testable without a display and both UI surfaces render from one source of
truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from arrayscope.operations.registry import OperationEntry


@dataclass(frozen=True)
class ListingSection:
    """One titled block of operations in the listing.

    ``is_more`` flags a section that belongs to the secondary browse partition
    (a UI surface may hide it behind a submenu or category chooser); immediately
    visible sections carry ``is_more=False``.
    """

    title: str
    entries: tuple[OperationEntry, ...]
    is_more: bool = False


def build_operation_listing(
    grouped=None,
    *,
    common_ids=None,
    more_groups=None,
) -> list[ListingSection]:
    """Split a grouped operation listing into ordered presentation sections.

    ``grouped`` is an ordered ``[(group, [entries])]`` listing; it defaults to
    :func:`arrayscope.operations.library.grouped_operations` (which excludes
    hidden ops and applies the persisted layout group/order overrides).
    ``common_ids`` (default :func:`library.effective_common_ids`) are the ids
    pinned to a leading "Common" section, in that order; ``more_groups``
    (default :func:`library.effective_more_groups`) are the groups flagged for
    the collapsed "More" partition.

    Section 1 is "Common" (the ``common_ids`` present anywhere in ``grouped``,
    in that order). Then one section per group in ``grouped`` order, each with
    its Common ops removed (so nothing is listed twice) and dropped entirely
    when that leaves it empty. Groups in ``more_groups`` carry ``is_more=True``.
    """

    if grouped is None or common_ids is None or more_groups is None:
        from arrayscope.operations import library

        if grouped is None:
            grouped = library.grouped_operations()
        if common_ids is None:
            common_ids = library.effective_common_ids()
        if more_groups is None:
            more_groups = library.effective_more_groups()

    grouped = [(group, tuple(entries)) for group, entries in grouped]
    common_ids = tuple(common_ids)
    common_set = set(common_ids)
    more_groups = frozenset(more_groups)

    by_id: dict[str, OperationEntry] = {}
    for _group, entries in grouped:
        for entry in entries:
            by_id[entry.id] = entry

    sections: list[ListingSection] = []

    # 1. The pinned "Common" section, in common_ids order.
    common = tuple(by_id[op_id] for op_id in common_ids if op_id in by_id)
    if common:
        sections.append(ListingSection("Common", common, is_more=False))

    # 2. One section per group, excluding ops already pinned to Common.
    for group, entries in grouped:
        group_entries = tuple(entry for entry in entries if entry.id not in common_set)
        if not group_entries:
            continue
        sections.append(ListingSection(group, group_entries, is_more=group in more_groups))

    return sections
