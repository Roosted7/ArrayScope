from __future__ import annotations

import os

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from arrayscope.operations.registry import operation_entries
from arrayscope.export.video import VideoExportWorker, VideoExportDialog, VideoExportSettingsDialog
from arrayscope.ui.file_dialogs import get_existing_directory, get_save_file_name
from arrayscope.ui.icons import set_action_icon


class ExportWorkflowMixin:
    def _setup_export_context_menus(self):
        for i in range(self.data.ndim):
            prim_btn = self.widgets['buttons']['primary'][i]
            sec_btn = self.widgets['buttons']['secondary'][i]
            
            prim_btn.setContextMenuPolicy(Qt.QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            sec_btn.setContextMenuPolicy(Qt.QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            
            prim_btn.customContextMenuRequested.connect(lambda pos, btn=prim_btn, dim=i: self._show_export_context_menu(pos, btn, dim))
            sec_btn.customContextMenuRequested.connect( lambda pos, btn=sec_btn,  dim=i: self._show_export_context_menu(pos, btn, dim))
    
    def _show_export_context_menu(self, pos, btn, dim):
        menu = QtWidgets.QMenu(self)

        operations_menu = menu.addMenu("Operations")
        for entry in operation_entries():
            action = operations_menu.addAction(entry.label)
            set_action_icon(action, "data_array")
            action.setEnabled(self._operation_entry_enabled(entry, dim))
            action.triggered.connect(lambda checked=False, operation_id=entry.id: self._append_operation(operation_id, dim))

        menu.addSeparator()
        
        export_action = menu.addAction("Export along this dimension...")
        set_action_icon(export_action, "download")
        export_action.triggered.connect(lambda: self._start_export(dim))
        
        # Show menu at cursor position
        menu.exec(btn.mapToGlobal(pos))
    
    def _start_export(self, export_dim):
        """Initiate video export workflow"""
        if self.singleton[export_dim]:
            QtWidgets.QMessageBox.warning(self, "Cannot Export", 
                f"Dimension {export_dim} has size 1 and cannot be exported.")
            return
        
        # Get export settings from dialog
        settings_dialog = VideoExportSettingsDialog(parent=self, export_dim=export_dim, data_shape=self.data.shape)
        if settings_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        
        settings = settings_dialog.get_settings()
        
        # Get file save path (for PNG frames we ask for a directory)
        file_path = None
        if settings['format'] == 'png':
            dir_path = get_existing_directory(
                self, "Export frames to directory", os.path.expanduser("~")
            )
            if not dir_path:
                return
            file_path = dir_path
        else:
            file_filter = f"{settings['format'].upper()} files (*.{settings['format']})"
            file_path, _ = get_save_file_name(
                self, f"Export Video as {settings['format'].upper()}", 
                f"export.{settings['format']}", file_filter
            )
            if not file_path:
                return
        
        # Capture display mode and widget aspect ratio
        display_mode = getattr(self.img_view, 'displayMode', 'square_pixels')
        view = self.img_view.getView()
        widget_ratio = view.size().width() / view.size().height() if view.size().height() != 0 else 1.0
        
        # Capture current display levels for consistent frame scaling
        levels = None
        try:
            levels = tuple(self.img_view.getLevels())
        except Exception:
            levels = None

        # Capture current color map LUT (256x3) to apply in export
        lut = None
        try:
            if hasattr(self.img_view, 'histogram') and hasattr(self.img_view.histogram, 'gradient'):
                cm = self.img_view.histogram.gradient.colorMap()
                if cm is not None:
                    lut = cm.getLookupTable(0.0, 1.0, 256, alpha=False)
        except Exception:
            lut = None
        
        # Create worker thread
        worker = VideoExportWorker(
            document=self.document,
            data_shape=self.data.shape,
            view_state=self.view_state,
            export_dim=export_dim,
            output_path=file_path,
            fps=settings['fps'],
            format_type=settings['format'],
            window_level_mode=settings.get('window_level', 'displayed'),
            levels=levels,
            pixel_ratio_mode=settings.get('pixel_ratio', 'square_pixels'),
            display_mode=display_mode,
            widget_ratio=widget_ratio,
            colormap_lut=lut,
            evaluator=self.operation_evaluator,
        )
        
        # Show progress dialog
        progress_dialog = VideoExportDialog(self)
        progress_dialog.start_export(worker, self.data.shape[export_dim])
