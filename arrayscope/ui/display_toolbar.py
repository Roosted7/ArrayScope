"""Compact display controls for the ArrayScope main window."""

from __future__ import annotations

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from arrayscope.ui.icons import set_action_icon
from arrayscope.ui.widgets import TOOL_BUTTON_STYLE, configure_tool_button


class DisplayToolbar(QtWidgets.QToolBar):
    channelChanged = Qt.QtCore.Signal(str)
    scaleChanged = Qt.QtCore.Signal(str)
    fitRequested = Qt.QtCore.Signal(bool)
    oneToOneRequested = Qt.QtCore.Signal()
    windowModeChanged = Qt.QtCore.Signal(str)
    autoWindowRequested = Qt.QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__("Display", parent)
        self.setObjectName("DisplayToolbar")
        self.setMovable(False)
        self.setIconSize(Qt.QtCore.QSize(16, 16))
        self.setStyleSheet(TOOL_BUTTON_STYLE)

        self.channel_combo = QtWidgets.QComboBox()
        for label, value in (("Complex", "complex"), ("Real", "real"), ("Abs", "abs"), ("Imag", "imag"), ("Phase", "angle")):
            self.channel_combo.addItem(label, value)
        self.channel_combo.currentIndexChanged.connect(self._channel_index_changed)
        self.addWidget(QtWidgets.QLabel("Channel "))
        self.addWidget(self.channel_combo)

        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItem("Linear", "linear")
        self.scale_combo.addItem("Symlog", "symlog")
        self.scale_combo.currentIndexChanged.connect(lambda _i: self.scaleChanged.emit(self.scale_combo.currentData()))
        self.addSeparator()
        self.addWidget(QtWidgets.QLabel("Scale "))
        self.addWidget(self.scale_combo)

        self.addSeparator()
        self.fit_action = self.addAction("Fit")
        self.fit_action.setCheckable(True)
        set_action_icon(self.fit_action, "fit_screen")
        self.fit_action.setToolTip("Fit image to viewport")
        self.fit_action.triggered.connect(lambda checked=False: self.fitRequested.emit(bool(checked)))
        self.one_to_one_action = self.addAction("1:1")
        set_action_icon(self.one_to_one_action, "aspect_ratio")
        self.one_to_one_action.setToolTip("Show image at one screen pixel per image pixel")
        self.one_to_one_action.triggered.connect(lambda _checked=False: self.oneToOneRequested.emit())
        for action in (self.fit_action, self.one_to_one_action):
            button = self.widgetForAction(action)
            if button is not None:
                configure_tool_button(button)

        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.addItem("Relative", "relative")
        self.window_combo.addItem("Absolute", "absolute")
        self.window_combo.currentIndexChanged.connect(lambda _i: self.windowModeChanged.emit(self.window_combo.currentData()))
        self.addSeparator()
        self.addWidget(QtWidgets.QLabel("Window "))
        self.addWidget(self.window_combo)

        self.auto_window_action = self.addAction("Auto")
        set_action_icon(self.auto_window_action, "tonality")
        self.auto_window_action.setToolTip("Auto window levels")
        self.auto_window_action.triggered.connect(self.autoWindowRequested)
        button = self.widgetForAction(self.auto_window_action)
        if button is not None:
            configure_tool_button(button)

    def set_channel_options(self, enabled_channels):
        for index in range(self.channel_combo.count()):
            value = self.channel_combo.itemData(index)
            item = self.channel_combo.model().item(index)
            if item is not None:
                item.setEnabled(bool(enabled_channels.get(value, False)))

    def _channel_index_changed(self, index):
        item = self.channel_combo.model().item(index)
        if item is not None and not item.isEnabled():
            return
        self.channelChanged.emit(self.channel_combo.currentData())

    def set_current(self, *, channel=None, scale=None, aspect=None, window_mode=None, live_profile=None):
        for combo, value in (
            (self.channel_combo, channel),
            (self.scale_combo, scale),
            (self.window_combo, window_mode),
        ):
            if value is None:
                continue
            index = combo.findData(value)
            if index >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)
        if aspect is not None:
            blocker = Qt.QtCore.QSignalBlocker(self.fit_action)
            try:
                self.fit_action.setChecked(aspect == "fit")
            finally:
                del blocker
        del live_profile
