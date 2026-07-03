"""User-facing status message helpers."""

from __future__ import annotations

from html import escape

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets


_STATUS_LANE_GAP = 12


class _StatusBarLayoutFilter(Qt.QtCore.QObject):
    def __init__(self, window, status_bar):
        super().__init__(status_bar)
        self._window = window

    def eventFilter(self, obj, event):
        if event.type() in {
            Qt.QtCore.QEvent.Type.Resize,
            Qt.QtCore.QEvent.Type.Show,
            Qt.QtCore.QEvent.Type.LayoutRequest,
        }:
            _layout_status_widgets(self._window)
        return super().eventFilter(obj, event)


def show_status_message(window, message, timeout=4000):
    if not hasattr(window, "statusBar"):
        return None
    status_bar = window.statusBar()
    status_bar.clearMessage()
    _ensure_status_layout_filter(window, status_bar)
    label = getattr(window, "_arrayscope_status_message_widget", None)
    if label is None:
        label = QtWidgets.QLabel(status_bar)
        label.setObjectName("ArrayScopeStatusMessageLabel")
        label.setTextFormat(Qt.QtCore.Qt.TextFormat.PlainText)
        label.setAlignment(
            Qt.QtCore.Qt.AlignmentFlag.AlignRight
            | Qt.QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        label.setAttribute(Qt.QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        window._arrayscope_status_message_widget = label
    label.setProperty("arrayscope_status_message_text", str(message))
    _set_status_message_text(label, None)
    label.show()

    timer = getattr(window, "_arrayscope_status_message_timer", None)
    if timer is not None:
        timer.stop()
    window._arrayscope_status_message_timer = None
    if int(timeout) > 0:
        timer = Qt.QtCore.QTimer(label)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: _clear_status_message(window, label))
        window._arrayscope_status_message_timer = timer
        timer.start(int(timeout))

    _layout_status_widgets(window)
    return label


def show_status_action(window, message, action_text, on_action, timeout=5000):
    if not hasattr(window, "statusBar"):
        return None
    status_bar = window.statusBar()
    status_bar.clearMessage()
    _ensure_status_layout_filter(window, status_bar)
    _clear_status_action(window)
    widget = QtWidgets.QLabel(status_bar)
    widget.setObjectName("ArrayScopeStatusActionLabel")
    widget.setTextFormat(Qt.QtCore.Qt.TextFormat.RichText)
    widget.setTextInteractionFlags(Qt.QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse)
    widget.setOpenExternalLinks(False)
    widget.setProperty("arrayscope_status_action_message", str(message))
    widget.setProperty("arrayscope_status_action_text", str(action_text))
    _set_status_action_text(widget)
    widget.show()

    # User-visible timeout. The widget owns the timer, and clicking or replacing
    # the status action clears it explicitly.
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
    _layout_status_widgets(window)
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
    _layout_status_widgets(window)


def _clear_status_message(window, widget=None):
    existing = getattr(window, "_arrayscope_status_message_widget", None)
    if widget is not None and existing is not widget:
        return
    timer = getattr(window, "_arrayscope_status_message_timer", None)
    if timer is not None:
        timer.stop()
    if existing is not None:
        existing.hide()
        existing.deleteLater()
    window._arrayscope_status_message_widget = None
    window._arrayscope_status_message_timer = None
    _layout_status_widgets(window)


def _ensure_status_layout_filter(window, status_bar):
    existing = getattr(window, "_arrayscope_status_layout_filter", None)
    if existing is not None:
        return
    event_filter = _StatusBarLayoutFilter(window, status_bar)
    status_bar.installEventFilter(event_filter)
    window._arrayscope_status_layout_filter = event_filter


def _layout_status_widgets(window) -> None:
    if not hasattr(window, "statusBar"):
        return
    status_bar = window.statusBar()
    rect = status_bar.contentsRect()
    if rect.width() <= 0 or rect.height() <= 0:
        return

    status_label = getattr(window, "_arrayscope_status_message_widget", None)
    action_label = getattr(window, "_arrayscope_status_action_widget", None)
    action_width = 0

    if _is_visible_widget(action_label):
        _set_status_action_text(action_label)
        action_width = min(max(1, action_label.sizeHint().width()), max(1, rect.width()))
        action_label.setGeometry(
            rect.left(),
            rect.top(),
            action_width,
            max(1, rect.height()),
        )
        action_label.raise_()

    if _is_visible_widget(status_label):
        available = rect.width()
        if action_width > 0:
            available = max(1, rect.width() - action_width - _STATUS_LANE_GAP)
        _set_status_message_text(status_label, available)
        status_width = min(max(1, status_label.sizeHint().width()), max(1, available))
        status_label.setAlignment(
            Qt.QtCore.Qt.AlignmentFlag.AlignRight
            | Qt.QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        status_label.setGeometry(
            rect.left() + rect.width() - status_width,
            rect.top(),
            status_width,
            max(1, rect.height()),
        )
        status_label.raise_()


def _is_visible_widget(widget) -> bool:
    if widget is None:
        return False
    try:
        return not widget.isHidden()
    except RuntimeError:
        return False


def _set_status_action_text(widget) -> None:
    message = str(widget.property("arrayscope_status_action_message") or "")
    action_text = str(widget.property("arrayscope_status_action_text") or "")
    widget.setText(
        f"{escape(message)} <a href=\"action\">{escape(action_text)}</a>"
    )


def _set_status_message_text(widget, available_width: int | None) -> None:
    message = str(widget.property("arrayscope_status_message_text") or "")
    visible_message = message
    if available_width is not None:
        visible_message = _elide_ascii_right(message, widget.fontMetrics(), int(available_width))
    widget.setText(visible_message)


def _elide_ascii_right(text: str, metrics, width: int) -> str:
    text = str(text)
    if metrics.horizontalAdvance(text) <= width:
        return text
    marker = "..."
    if width <= 0 or metrics.horizontalAdvance(marker) > width:
        return marker
    low = 0
    high = len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if metrics.horizontalAdvance(text[:mid] + marker) <= width:
            low = mid
        else:
            high = mid - 1
    return text[:low] + marker
