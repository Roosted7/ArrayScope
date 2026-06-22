"""User-facing status message helpers."""

from __future__ import annotations

from html import escape

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets


def show_status_message(window, message, timeout=4000):
    if hasattr(window, "statusBar"):
        window.statusBar().showMessage(str(message), int(timeout))


def show_status_action(window, message, action_text, on_action, timeout=5000):
    if not hasattr(window, "statusBar"):
        return None
    status_bar = window.statusBar()
    _clear_status_action(window)
    widget = QtWidgets.QLabel(status_bar)
    widget.setObjectName("ArrayScopeStatusActionLabel")
    widget.setTextFormat(Qt.QtCore.Qt.TextFormat.RichText)
    widget.setTextInteractionFlags(Qt.QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse)
    widget.setOpenExternalLinks(False)
    widget.setText(
        f"{escape(str(message))} <a href=\"action\">{escape(str(action_text))}</a>"
    )
    rect = status_bar.contentsRect()
    hint = widget.sizeHint()
    widget.move(rect.left(), rect.top())
    widget.resize(min(max(1, hint.width()), max(1, rect.width())), max(1, rect.height()))
    widget.show()
    widget.raise_()

    timer = Qt.QtCore.QTimer(widget)
    timer.setSingleShot(True)

    def clear():
        _clear_status_action(window, widget)

    def trigger():
        clear()
        on_action()

    timer.timeout.connect(clear)
    widget.linkActivated.connect(lambda _link: trigger())
    window._arrayscope_status_action_widget = widget
    window._arrayscope_status_action_timer = timer
    timer.start(max(1, int(timeout)))
    return widget


def show_revert_action(window, message, on_revert, *, timeout=5000):
    """Show a transient non-blocking status action with a shared revert affordance."""

    return show_status_action(
        window,
        message,
        "Revert",
        on_revert,
        timeout=timeout,
    )


def _clear_status_action(window, widget=None):
    existing = getattr(window, "_arrayscope_status_action_widget", None)
    if widget is not None and existing is not widget:
        return
    timer = getattr(window, "_arrayscope_status_action_timer", None)
    if timer is not None:
        timer.stop()
    if existing is not None:
        existing.hide()
        existing.deleteLater()
    window._arrayscope_status_action_widget = None
    window._arrayscope_status_action_timer = None
