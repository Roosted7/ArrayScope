"""Compare launcher and linked complex cursor (queue item 6).

Two pieces live here:

- ``CompareLauncherMixin.open_compare_window`` opens a second, in-process
  ``ArrayScopeWindow`` over the same source array and pre-links it: dims,
  camera (pan/zoom), and levels are enabled on *both* windows' sync
  controllers so they move together immediately. The sibling is an ordinary
  window joined to the same local-socket sync group the "Sync" toggles use;
  no new transport is introduced.

- ``CompareCursorGroup`` is the in-process linked cursor. When the pointer
  resolves to a data coordinate in any member, the shared cursor is the full
  *source* array index (``ViewPointMapping.array_index`` — the same tuple the
  status bar reports as ``d{axis}=…``). Every member reads its OWN source
  array at that index and both HUDs show ``A`` and ``B`` (magnitude and phase
  for complex dtypes).

Why an in-process link rather than a sync-bus facet: exact value readout needs
the *sibling's* data, which the JSON sync envelopes deliberately never carry.
The in-process sibling is the target case, so each window computes its own
value locally by direct source indexing — exact by construction, matching a
plain NumPy ``array[index]``. No cursor position is ever sent over the bus.

No new scheduler/timer is introduced: the readout is driven entirely by the
existing hover refresh path (``RenderOrchestrator.getPixel``), which already
fires on ``sigMouseMoved`` and after every display commit.
"""

from __future__ import annotations

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.ui.toasts import show_status_message


def read_source_value(source, array_index):
    """Read the exact source value at ``array_index`` (a full N-d index).

    ``array_index`` is the source-array index produced by the display
    geometry, so this is a plain element read and matches ``array[index]``
    for a NumPy array. Lazy/derived sources (``LazySourceArray`` over a
    ``CompositeArraySource``, memmap-backed sources) have no ``__getitem__``
    but do expose ``read_region``; for those we read the single-element region
    at ``idx`` (one int per axis), which is exact and — for the A - B
    difference window — equals ``A[idx] - B[idx]`` by construction.
    """

    idx = tuple(int(value) for value in array_index)
    reader = getattr(source, "read_region", None)
    value = reader(idx) if callable(reader) else source[idx]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value[()]
        return value
    return value


def format_compare_value(value) -> str:
    """Format one member's value; complex renders as magnitude and phase."""

    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        flat = arr.reshape(-1)
        z = complex(flat[0]) if flat.size else 0j
        magnitude = float(np.abs(z))
        phase = float(np.angle(z))
        return f"|{magnitude:.4g}| ∠{phase:.4g} rad"
    try:
        scalar = arr.item()
    except (ValueError, AttributeError):
        return str(value)
    if isinstance(scalar, (bool, np.bool_)):
        return str(bool(scalar))
    if isinstance(scalar, (int, np.integer)):
        return str(int(scalar))
    return f"{float(scalar):.4g}"


class CompareCursorGroup:
    """In-process shared cursor across a set of linked compare windows.

    Membership holds strong references; windows are already retained for their
    lifetime by the app, and each removes itself on close.
    """

    def __init__(self):
        self._members: list = []

    # -- membership ----------------------------------------------------
    def add(self, window) -> None:
        if window in self._members:
            return
        window.compare_label = chr(ord("A") + len(self._members))
        self._members.append(window)

    def remove(self, window) -> None:
        try:
            self._members.remove(window)
        except ValueError:
            return
        # Renumber remaining members so labels stay dense (A, B, C…).
        for index, member in enumerate(self._members):
            member.compare_label = chr(ord("A") + index)

    def members(self) -> tuple:
        return tuple(self._members)

    def is_active(self) -> bool:
        return len(self._members) >= 2

    # -- cursor broadcast ---------------------------------------------
    def broadcast(self, source_window, array_index, view_point, coords_text) -> None:
        """Read every member's own value at ``array_index`` and update HUDs."""

        members = self.members()
        if len(members) < 2:
            return
        values: dict[str, object] = {}
        parts: list[str] = []
        for window in members:
            label = getattr(window, "compare_label", "?")
            try:
                value = read_source_value(window.base_data, array_index)
            except Exception:
                values[label] = None
                parts.append(f"{label} n/a")
                continue
            values[label] = value
            parts.append(f"{label} {format_compare_value(value)}")
        text = " · ".join(parts)
        if coords_text:
            text = f"{coords_text}  {text}"
        for window in members:
            pos = window._compare_scene_pos(view_point)
            window._show_compare_hud(text, values, array_index, pos)

    def clear_huds(self) -> None:
        for window in self.members():
            window._hide_compare_hud()


class CompareLauncherMixin:
    """Window-side "Compare with…" launcher and linked-cursor hooks."""

    _COMPARE_FACETS = ("dims", "camera", "levels")

    # -- launcher ------------------------------------------------------
    def open_compare_window(self, data=None):
        """Open a second window over ``data`` (defaults to this source) and
        pre-link dims + camera + levels on both windows.

        Returns the sibling ``ArrayScopeWindow`` (or ``None`` if it could not
        be created)."""

        from arrayscope.window import ArrayScopeWindow

        if data is None:
            data = self.base_data
        try:
            sibling = ArrayScopeWindow(data)
        except Exception as exc:  # pragma: no cover - defensive
            show_status_message(self, f"Could not open compare window: {exc}", timeout=4000)
            return None

        base_title = self.windowTitle() or "ArrayScope"
        sibling.setWindowTitle(f"{base_title} — compare")
        self.link_compare_window(sibling)
        self._retain_and_show_compare_window(sibling)
        show_status_message(self, "Opened linked compare window.", timeout=2500)
        return sibling

    def open_difference_window(self):
        """Open a third window over ``A - B`` (a derived ``CompositeArraySource``),
        linked into the *same* compare group as A and B.

        A and B are the first two members of this window's compare group. The
        derived source reads both inputs region-by-region (never materializing
        either), so the difference window renders progressively through the
        unchanged pipeline. Ownership is deliberately *not* propagated: the
        composite is built with ``own_inputs=False`` so closing the difference
        window never tears down A's or B's still-live sources.

        Returns the difference ``ArrayScopeWindow`` (or ``None`` if the group is
        not a linked pair yet, or the two inputs have mismatched shapes).
        """

        from arrayscope.core.array_source import CompositeArraySource, LazySourceArray
        from arrayscope.window import ArrayScopeWindow

        group = getattr(self, "_compare_group", None)
        members = group.members() if group is not None else ()
        if len(members) < 2:
            show_status_message(
                self,
                "Open a linked compare window first, then take the difference.",
                timeout=4000,
            )
            return None

        win_a, win_b = members[0], members[1]
        base_a = win_a.base_data
        base_b = win_b.base_data
        shape_a = tuple(int(size) for size in np.shape(base_a))
        shape_b = tuple(int(size) for size in np.shape(base_b))
        if shape_a != shape_b:
            show_status_message(
                self,
                f"Cannot difference windows of different shapes: {shape_a} vs {shape_b}.",
                timeout=5000,
            )
            return None

        try:
            composite = CompositeArraySource(base_a, base_b, op="subtract", own_inputs=False)
            diff_data = LazySourceArray(composite, materialize_budget_bytes=None)
            diff_window = ArrayScopeWindow(diff_data)
        except Exception as exc:  # pragma: no cover - defensive
            show_status_message(self, f"Could not open difference window: {exc}", timeout=4000)
            return None

        label_a = getattr(win_a, "compare_label", "A")
        label_b = getattr(win_b, "compare_label", "B")
        diff_window.setWindowTitle(f"ArrayScope — {label_a} − {label_b}")
        # Join the SAME group so dims + camera + levels stay linked across A, B
        # and the difference window.
        self.link_compare_window(diff_window)
        self._retain_and_show_compare_window(diff_window)
        show_status_message(
            self, f"Opened difference window ({label_a} − {label_b}).", timeout=2500
        )
        return diff_window

    def _retain_and_show_compare_window(self, window) -> None:
        """Register ``window`` in the app-global retention list and show it.

        Compare/difference siblings have no ``WA_DeleteOnClose``; the app-global
        list is what keeps them alive (and the test fixture drops that list to
        dispose them).
        """

        app = QtWidgets.QApplication.instance()
        if app is not None:
            from arrayscope.app.launch import _retain_window_reference

            _retain_window_reference(app, window)
        window.show()

    def link_compare_window(self, other) -> CompareCursorGroup:
        """Join ``self`` and ``other`` into one compare group and enable the
        shared sync facets (dims, camera, levels) on both."""

        group = getattr(self, "_compare_group", None)
        if group is None:
            group = CompareCursorGroup()
            group.add(self)
            self._compare_group = group
        group.add(other)
        other._compare_group = group
        for window in (self, other):
            controller = getattr(window, "sync_controller", None)
            if controller is None:
                continue
            for facet in self._COMPARE_FACETS:
                controller.set_facet_enabled(facet, True)
        return group

    def _teardown_compare(self) -> None:
        group = getattr(self, "_compare_group", None)
        if group is None:
            return
        group.remove(self)
        self._compare_group = None

    # -- linked cursor hooks (called from the hover refresh path) ------
    def _broadcast_compare_cursor(self, mapping) -> None:
        group = getattr(self, "_compare_group", None)
        if group is None or not group.is_active():
            return
        array_index = getattr(mapping, "array_index", None)
        if array_index is None:
            return
        coords_text = f"({int(mapping.local_x)}, {int(mapping.local_y)})"
        group.broadcast(
            self,
            tuple(array_index),
            (int(mapping.view_x), int(mapping.view_y)),
            coords_text,
        )

    def _clear_compare_cursor(self) -> None:
        group = getattr(self, "_compare_group", None)
        if group is not None:
            group.clear_huds()

    def _compare_scene_pos(self, view_point):
        view = self.img_view.getView()
        return view.mapViewToScene(
            QtCore.QPointF(float(view_point[0]) + 0.5, float(view_point[1]) + 0.5)
        )

    def _show_compare_hud(self, text, values, array_index, scene_pos) -> None:
        self._last_compare_hud_text = str(text)
        self._last_compare_values = dict(values)
        self._last_compare_array_index = tuple(array_index)
        view = getattr(self, "img_view", None)
        if view is None or scene_pos is None:
            return
        show_hud = getattr(view, "showHudText", None)
        if callable(show_hud):
            show_hud(text, scene_pos)

    def _hide_compare_hud(self) -> None:
        self._last_compare_hud_text = None
        self._last_compare_values = {}
        self._last_compare_array_index = None
        view = getattr(self, "img_view", None)
        hide_hud = getattr(view, "hideHud", None)
        if callable(hide_hud):
            hide_hud()
