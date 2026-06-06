"""Compact display controls for the ArrayScope main window."""

from __future__ import annotations

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from arrayscope.ui.icons import set_action_icon


class DisplayToolbar(QtWidgets.QToolBar):
    channelChanged = Qt.QtCore.Signal(str)
    scaleChanged = Qt.QtCore.Signal(str)
    aspectChanged = Qt.QtCore.Signal(str)
    windowModeChanged = Qt.QtCore.Signal(str)
    autoWindowRequested = Qt.QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__("Display", parent)
        self.setObjectName("DisplayToolbar")
        self.setMovable(False)
        self.setIconSize(Qt.QtCore.QSize(16, 16))

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

        self.aspect_combo = QtWidgets.QComboBox()
        self.aspect_combo.addItem("1:1", "square_pixels")
        self.aspect_combo.addItem("FOV", "square_fov")
        self.aspect_combo.addItem("Fit", "fit")
        self.aspect_combo.currentIndexChanged.connect(lambda _i: self.aspectChanged.emit(self.aspect_combo.currentData()))
        self.addSeparator()
        self.addWidget(QtWidgets.QLabel("Aspect "))
        self.addWidget(self.aspect_combo)

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
            (self.aspect_combo, aspect),
            (self.window_combo, window_mode),
        ):
            if value is None:
                continue
            index = combo.findData(value)
            if index >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)
        del live_profile
