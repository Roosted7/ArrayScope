"""Fresh-window handoff when a window's document changes rank.

The per-dimension controls -- the Y/X/P buttons, the slice spinboxes, the
dimension-strip chips, the flip/shift/complex indicator labels -- are allocated
*once*, from the rank the window was constructed at
(``display_controls._create_widget_registry``), and every sync loop indexes them
positionally. A rank DECREASE survives: the loops hide everything at ``i >=
ndim`` and never index past the arrays. A rank INCREASE cannot survive, because
there is nothing to index: ``self.widgets["labels"]["complex"][ndim - 1]`` is an
``IndexError``, and no in-place rebuild exists.

``file_reload`` has stated that and done the only correct thing since it grew a
shape guard: open a fresh window at the new rank and close this one. This module
is that handoff generalized from *"the file reloaded at a different ndim"* to
*"this document's ndim no longer fits this window"*, so the operation stack can
reach it too -- re-enabling an inherited rank-increasing step, or loading a
recipe that contains one.

**What the handoff carries.** The base data, the operation steps and the base
axis metadata (so the new window shows the same derived array through the same,
still editable, stack), the window title, and the file identity (path, dataset
key, selector class) so the new window keeps watching the same file.

**What it drops, deliberately.** The view state (image/line/montage axis choice,
slice positions, per-axis flips, per-axis ranges), the display settings (levels,
colormap, scale, aspect), ROIs, the dock layout, and compare-group membership.
Every one of those is indexed *by axis*, so carrying it across a rank change
needs exactly the migration the widgets cannot do; the fresh window mints them
from the new shape, the same way it would for a file opened cold. The reload
handoff dropped all of this before, silently -- this only says so.
"""

from __future__ import annotations

from pyqtgraph.Qt import QtWidgets


class WindowRankHandoffMixin:
    """Per-dimension widget capacity, and the fresh window when it is exceeded."""

    def per_dimension_widget_capacity(self):
        """How many axes the per-dimension widget arrays hold, or ``None``.

        ``None`` means the UI is not built yet, so no caller can ask the
        question. Note this is the CONSTRUCTION rank, not the current one: a
        document that shrank and grew back still fits.
        """

        widgets = getattr(self, "widgets", None)
        if not isinstance(widgets, dict):
            return None
        try:
            return len(widgets["labels"]["complex"])
        except (KeyError, TypeError):
            return None

    def document_rank_fits_widgets(self, ndim) -> bool:
        capacity = self.per_dimension_widget_capacity()
        return capacity is None or int(ndim) <= int(capacity)

    def open_window_at_new_rank(self, data, *, steps=None, axes=None, title=None):
        """Open a fresh window over ``data``/``steps`` and close this one.

        Returns the new window. See the module docstring for what survives.
        """

        window = type(self)(
            data,
            filepath=self._filepath,
            dataset_path=self._dataset_path,
            selector_class_name=self._selector_class_name,
            axes=axes,
            steps=steps,
        )
        window.setWindowTitle(self.windowTitle() if title is None else str(title))
        _retain_window(window)
        window.show()
        self.close()
        return window

    def _hand_off_document_rank_change(self, document):
        """Move ``document`` into a fresh window because it outgrew this one."""

        from arrayscope.ui.toasts import show_status_message

        window = self.open_window_at_new_rank(
            document.base_data,
            steps=document.steps,
            axes=document.base_axes,
            title=self.windowTitle(),
        )
        show_status_message(
            window,
            "The operation stack added a dimension — reopened at the new shape.",
            timeout=6000,
        )
        return window


def _retain_window(window) -> None:
    """Keep the new window alive; nothing else holds a reference to it."""

    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    from arrayscope.app.launch import _retain_window_reference

    _retain_window_reference(app, window)
