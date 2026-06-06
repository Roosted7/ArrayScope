from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui, QtWidgets

from arrayscope.display.colormaps import gray_colormap, named_colormap, phase_colormap
from arrayscope.ui.dialogs import SaveRangeDialog
from arrayscope.operations.recipes import load_recipe, save_recipe
from arrayscope.operations.registry import operation_entries
from arrayscope.profiles.model import clamp_marker_position, image_hover_indices, profile_y_range
from arrayscope.display.slice_engine import apply_channel
from arrayscope.app.settings_state import AppSettingsState, settings_from_mapping, settings_to_mapping
from arrayscope.app.theme import ThemeChoice, apply_theme_to_qapplication
from arrayscope.export.video import VideoExportWorker, VideoExportDialog, VideoExportSettingsDialog
from arrayscope.core.view_state import ChannelMode, ScaleMode
from arrayscope.core.window_levels import choose_window_levels
from arrayscope.window.domain import Domain


class WindowMenuMixin:
    def _load_app_settings(self):
        return settings_from_mapping(
            {
                "theme": self._settings.value("theme", ThemeChoice.SYSTEM.value),
                "prefetch_nearby_slices": self._settings.value("prefetch_nearby_slices", False),
            }
        )

    def _save_app_settings(self):
        for key, value in settings_to_mapping(self.app_settings).items():
            self._settings.setValue(key, value)

    def _setup_menus(self):
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.operation_dock.toggleViewAction())
        view_menu.addAction(self.profile_dock.toggleViewAction())
        view_menu.addSeparator()
        reset_layout_action = QtGui.QAction("Reset layout", self)
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)

        theme_menu = self.menuBar().addMenu("Theme")
        self._theme_actions = {}
        self._theme_action_group = QtGui.QActionGroup(self)
        self._theme_action_group.setExclusive(True)
        for choice, label in (
            (ThemeChoice.SYSTEM, "System / Native"),
            (ThemeChoice.NATIVE, "Native"),
            (ThemeChoice.DARK, "Dark"),
            (ThemeChoice.LIGHT, "Light"),
        ):
            action = QtGui.QAction(label, self, checkable=True)
            self._theme_action_group.addAction(action)
            action.triggered.connect(lambda checked=False, choice=choice: self._apply_theme_choice(choice))
            theme_menu.addAction(action)
            self._theme_actions[choice] = action
        self._sync_theme_actions()

    def _sync_theme_actions(self):
        if not hasattr(self, "_theme_actions"):
            return
        for choice, action in self._theme_actions.items():
            action.blockSignals(True)
            action.setChecked(self.app_settings.theme == choice)
            action.blockSignals(False)

    def _apply_theme_choice(self, choice, persist=True):
        result = apply_theme_to_qapplication(QtWidgets.QApplication.instance(), choice)
        if result.warning:
            print(f"Theme warning: {result.warning}")
            if persist:
                QtWidgets.QMessageBox.warning(self, "Theme Warning", result.warning)
        theme_to_store = result.requested if result.applied == result.requested else result.applied
        self.applied_theme = result.applied
        self.theme_backend = result.backend
        self.app_settings = AppSettingsState(theme=theme_to_store, prefetch_nearby_slices=getattr(self, "app_settings", AppSettingsState()).prefetch_nearby_slices)
        if persist:
            self._save_app_settings()
        self._sync_theme_actions()

    def _set_prefetch_enabled(self, enabled):
        self.app_settings = AppSettingsState(theme=self.app_settings.theme, prefetch_nearby_slices=bool(enabled))
        self._save_app_settings()

    def _restore_window_settings(self):
        geometry = self._settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self._settings.value("window_state")
        if state is not None:
            self.restoreState(state)
        if not self.profile_dock.isVisible() and self.data.ndim == 1:
            self.profile_dock.show()
        Qt.QtCore.QTimer.singleShot(0, self._resize_default_docks)

    def reset_layout(self):
        self.profile_dock.setFloating(False)
        self.profile_dock.hide()
        if self.data.ndim == 1:
            self.profile_dock.show()
        self.operation_dock.setFloating(False)
        self.addDockWidget(Qt.QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.operation_dock)
        self.operation_dock.show()
        self.addDockWidget(Qt.QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.profile_dock)
        Qt.QtCore.QTimer.singleShot(0, self._resize_default_docks)

    def _resize_default_docks(self):
        try:
            if self.profile_dock.isVisible() and not self.profile_dock.isFloating():
                self.resizeDocks([self.profile_dock], [max(140, int(self.height() * 0.23))], Qt.QtCore.Qt.Orientation.Vertical)
            if self.operation_dock.isVisible() and not self.operation_dock.isFloating():
                self.resizeDocks([self.operation_dock], [max(220, int(self.width() * 0.24))], Qt.QtCore.Qt.Orientation.Horizontal)
        except Exception:
            pass

    def closeEvent(self, event):
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("window_state", self.saveState())
        self._save_app_settings()
        super().closeEvent(event)

