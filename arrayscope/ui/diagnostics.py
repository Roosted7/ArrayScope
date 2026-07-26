"""Developer diagnostics dialog."""

from __future__ import annotations

from arrayscope import __version__
from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui, QtWidgets

from arrayscope.core.memory_budget import format_bytes
from arrayscope.core.runtime_diagnostics import (
    format_runtime_diagnostics,
    format_runtime_diagnostics_sections,
    runtime_bottleneck_text,
)
from arrayscope.ui.diagnostics_logging import DiagnosticsJsonlLogger, default_diagnostics_log_path
from arrayscope.ui.file_dialogs import get_save_file_name


class _CompactUsageBar(QtWidgets.QProgressBar):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = str(label)
        self.setRange(0, 1000)
        self.setTextVisible(True)
        self.setMinimumHeight(16)
        self.setMaximumHeight(18)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)

    def set_usage(
        self, *, used: int, total: int, detail: str | None = None, color_mode: str = "usage"
    ) -> None:
        total = max(1, int(total))
        used = max(0, int(used))
        fraction = min(1.0, used / float(total))
        self.setValue(round(fraction * 1000))
        text = detail or f"{format_bytes(used)} / {format_bytes(total)}"
        self.setFormat(f"{self._label}: {text}")
        self.setStyleSheet(_compact_bar_style(fraction, color_mode=color_mode))


class _CompactSegmentBar(QtWidgets.QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = str(label)
        self._segments = ()
        self._summary = "n/a"
        self.setMinimumHeight(16)
        self.setMaximumHeight(18)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)

    def set_segments(self, segments, *, summary: str) -> None:
        self._segments = tuple(
            (str(label), int(value), str(color))
            for label, value, color in segments
            if int(value) > 0
        )
        self._summary = str(summary)
        self.setVisible(bool(self._segments))
        self.update()

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(QtGui.QPen(self.palette().mid().color(), 1))
        painter.setBrush(self.palette().base())
        painter.drawRoundedRect(rect, 2, 2)
        total = sum(value for _label, value, _color in self._segments)
        if total > 0:
            x = rect.x()
            remaining_width = rect.width()
            for index, (_label, value, color) in enumerate(self._segments):
                width = (
                    remaining_width
                    if index == len(self._segments) - 1
                    else round(rect.width() * value / total)
                )
                segment_rect = Qt.QtCore.QRect(x, rect.y(), max(1, width), rect.height())
                painter.setPen(Qt.QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(QtGui.QColor(color))
                painter.drawRect(segment_rect)
                x += width
                remaining_width = max(0, rect.right() - x + 1)
        painter.setPen(self.palette().text().color())
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(
            rect.adjusted(5, 0, -5, 0),
            Qt.QtCore.Qt.AlignmentFlag.AlignVCenter,
            f"{self._label}: {self._summary}",
        )


class _CompactCapacitySegmentBar(QtWidgets.QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = str(label)
        self._segments = ()
        self._summary = "n/a"
        self._total = 1
        self.setMinimumHeight(16)
        self.setMaximumHeight(18)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)

    def set_capacity_segments(self, segments, *, total: int, summary: str) -> None:
        self._segments = tuple(
            (str(label), max(0, int(value)), str(color))
            for label, value, color in segments
            if int(value) > 0
        )
        self._total = max(1, int(total))
        self._summary = str(summary)
        self.update()

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(QtGui.QPen(self.palette().mid().color(), 1))
        painter.setBrush(self.palette().base())
        painter.drawRoundedRect(rect, 2, 2)
        x = rect.x()
        consumed = 0
        remaining_width = rect.width()
        for index, (_label, value, color) in enumerate(self._segments):
            consumed += int(value)
            if index == len(self._segments) - 1:
                width = min(
                    remaining_width,
                    round(rect.width() * min(consumed, self._total) / self._total) - (x - rect.x()),
                )
            else:
                width = round(rect.width() * min(value, self._total) / self._total)
            if width > 0:
                segment_rect = Qt.QtCore.QRect(x, rect.y(), max(1, width), rect.height())
                painter.setPen(Qt.QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(QtGui.QColor(color))
                painter.drawRect(segment_rect)
                x += width
                remaining_width = max(0, rect.right() - x + 1)
        painter.setPen(self.palette().text().color())
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(
            rect.adjusted(5, 0, -5, 0),
            Qt.QtCore.Qt.AlignmentFlag.AlignCenter,
            f"{self._label}: {self._summary}",
        )


class _ElidedOverviewLabel(QtWidgets.QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__("", parent)
        self._text = str(text)
        self.setToolTip(self._text)
        self.setMinimumHeight(18)
        self.setMinimumWidth(0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)

    def setText(self, text: str) -> None:
        self._text = str(text)
        self.setToolTip(self._text)
        self.update()

    def text(self) -> str:
        return self._text

    def sizeHint(self):
        return Qt.QtCore.QSize(80, 18)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setPen(self.palette().text().color())
        metrics = painter.fontMetrics()
        text = metrics.elidedText(
            self._text, Qt.QtCore.Qt.TextElideMode.ElideRight, max(1, self.width() - 2)
        )
        painter.drawText(
            self.rect(),
            Qt.QtCore.Qt.AlignmentFlag.AlignVCenter | Qt.QtCore.Qt.AlignmentFlag.AlignLeft,
            text,
        )


class DiagnosticsDialog(QtWidgets.QDialog):
    def __init__(self, parent, snapshot_provider, *, interval_ms: int = 500):
        super().__init__(parent)
        self._snapshot_provider = snapshot_provider
        self._logger = None
        self.setWindowTitle("ArrayScope Diagnostics")
        self.setMinimumSize(480, 400)
        self.resize(520, 540)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        overview = QtWidgets.QFrame()
        overview.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        overview_layout = QtWidgets.QGridLayout(overview)
        overview_layout.setContentsMargins(8, 6, 8, 6)
        overview_layout.setHorizontalSpacing(12)
        overview_layout.setVerticalSpacing(3)
        self._overview_labels = {
            "status": _overview_label("Status"),
            "ops": _overview_label("Ops"),
            "tiles": _overview_label("Tiles"),
        }
        self._overview_labels["status"].setMinimumWidth(210)
        self._overview_labels["ops"].setMinimumWidth(210)
        self._overview_labels["tiles"].setMinimumWidth(210)
        self._compact_bars = {
            "resource": _CompactCapacitySegmentBar("CPU"),
            "stage": _CompactUsageBar("Stage"),
            "workers": _CompactSegmentBar("Workers"),
            "gpu": _CompactUsageBar("GPU"),
            "tile": _CompactUsageBar("Tiles"),
        }
        self._segment_bars = {
            "render": _CompactSegmentBar("Render"),
            "montage": _CompactSegmentBar("Montage"),
        }
        overview_layout.addWidget(self._overview_labels["status"], 0, 0)
        overview_layout.addWidget(self._compact_bars["resource"], 0, 1)
        overview_layout.addWidget(self._overview_labels["ops"], 1, 0)
        overview_layout.addWidget(self._compact_bars["stage"], 1, 1)
        overview_layout.addWidget(self._compact_bars["workers"], 1, 1)
        overview_layout.addWidget(self._overview_labels["tiles"], 2, 0)
        overview_layout.addWidget(self._compact_bars["gpu"], 2, 1)
        overview_layout.addWidget(self._compact_bars["tile"], 2, 1)
        overview_layout.addWidget(self._segment_bars["montage"], 3, 0)
        overview_layout.addWidget(self._segment_bars["render"], 3, 1)
        overview_layout.setColumnStretch(0, 1)
        overview_layout.setColumnStretch(1, 1)
        layout.addWidget(overview, 0)

        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        self.tabs = QtWidgets.QTabWidget()
        self._section_edits = {}
        self._section_titles = tuple(
            format_runtime_diagnostics_sections(self._snapshot_provider()).keys()
        )
        for title in (*self._section_titles, "All"):
            edit = QtWidgets.QPlainTextEdit()
            edit.setReadOnly(True)
            edit.setFont(font)
            edit.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
            edit.setHorizontalScrollBarPolicy(Qt.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._section_edits[title] = edit
            self.tabs.addTab(edit, title)
        self.text_edit = self._section_edits["All"]
        self.tabs.currentChanged.connect(lambda _index: self._refresh_current_text_tab())
        layout.addWidget(self.tabs, 1)

        buttons = QtWidgets.QDialogButtonBox()
        self.refresh_button = buttons.addButton(
            "Auto text", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )
        self.refresh_button.setCheckable(True)
        self.refresh_button.setChecked(True)
        self.log_button = buttons.addButton(
            "Log...", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )
        close_button = buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close)
        self.refresh_button.toggled.connect(
            lambda checked: self.refresh(force_text=True) if checked else None
        )
        self.log_button.clicked.connect(self._toggle_logging)
        close_button.clicked.connect(self.close)
        layout.addWidget(buttons)
        self.setLayout(layout)

        # Timer category: UI cosmetic. User-visible refresh timer. It runs only while the diagnostics dialog
        # is visible and is stopped on hide/close.
        self._timer = Qt.QtCore.QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self.refresh)
        self._last_snapshot = None
        self.refresh(force_text=True)

    def refresh(self, *, force_text: bool = False) -> None:
        try:
            snapshot = self._snapshot_provider()
            self._last_snapshot = snapshot
            self._update_overview(snapshot)
            if force_text or self.refresh_button.isChecked():
                self._refresh_current_text_tab(force=force_text)
            self._write_snapshot_log_record(snapshot)
            return
        except Exception as exc:
            text = f"Diagnostics unavailable: {exc}"
        self.current_text_edit().setPlainText(text)

    def current_text_edit(self):
        widget = self.tabs.currentWidget()
        return widget if isinstance(widget, QtWidgets.QPlainTextEdit) else self.text_edit

    def _refresh_current_text_tab(self, *, force: bool = False) -> None:
        if self._last_snapshot is None:
            return
        if not force and not self.refresh_button.isChecked():
            return
        title = self.tabs.tabText(self.tabs.currentIndex())
        edit = self._section_edits.get(title)
        if edit is None:
            return
        if title == "All":
            edit.setPlainText(format_runtime_diagnostics(self._last_snapshot))
            return
        sections = format_runtime_diagnostics_sections(self._last_snapshot)
        edit.setPlainText(sections.get(title, ""))

    def _toggle_logging(self) -> None:
        if self._logger is not None:
            self._stop_logging()
            return
        self._start_logging()

    def _start_logging(self) -> None:
        file_path, _selected_filter = get_save_file_name(
            self,
            "Save diagnostics log",
            str(default_diagnostics_log_path()),
            "JSON Lines (*.jsonl);;All files (*)",
        )
        if not file_path:
            return
        try:
            self.refresh(force_text=True)
            snapshot = self._last_snapshot
            if snapshot is None:
                return
            logger = DiagnosticsJsonlLogger(file_path)
            logger.start(snapshot, app_version=__version__, interval_ms=self._timer.interval())
            self._logger = logger
            self.log_button.setText("Stop")
        except Exception as exc:
            self._stop_logging()
            QtWidgets.QMessageBox.warning(
                self, "Diagnostics Log Error", f"Failed to start diagnostics logging:\n{exc}"
            )

    def _write_snapshot_log_record(self, snapshot) -> None:
        if self._logger is None:
            return
        try:
            self._logger.write_snapshot(snapshot)
        except Exception as exc:
            self._stop_logging()
            QtWidgets.QMessageBox.warning(
                self, "Diagnostics Log Error", f"Diagnostics logging stopped:\n{exc}"
            )

    def _stop_logging(self) -> None:
        logger = self._logger
        self._logger = None
        self.log_button.setText("Log...")
        if logger is not None:
            logger.close()

    def _update_overview(self, snapshot) -> None:
        self._overview_labels["status"].setText(
            f"{_overview_state(snapshot)} | {runtime_bottleneck_text(snapshot)} | {_feedback_summary(snapshot)}"
        )
        self._overview_labels["ops"].setText(_ops_overview(snapshot))
        self._overview_labels["tiles"].setText(_drawn_tiles_overview(snapshot))
        self._update_compact_bars(snapshot)
        self._update_segment_bars(snapshot)

    def _update_compact_bars(self, snapshot) -> None:
        policy = snapshot.memory_policy
        bars = self._compact_bars
        bars["resource"].set_capacity_segments(
            _resource_segments(snapshot),
            total=int(policy.system_total_bytes),
            summary=_resource_bar_summary(snapshot),
        )
        workers_active = bool(_worker_segments(snapshot.schedulers))
        gpu_used, gpu_total, gpu_detail = _gpu_bar_usage(snapshot)
        bars["stage"].set_usage(
            used=snapshot.stage_cache.bytes_used,
            total=snapshot.stage_cache.max_bytes,
            detail=_cache_tier_detail(snapshot.stage_cache),
        )
        bars["workers"].set_segments(
            _worker_segments(snapshot.schedulers), summary=_active_work_summary(snapshot.schedulers)
        )
        bars["stage"].setVisible(not workers_active)
        bars["workers"].setVisible(workers_active)
        bars["gpu"].set_usage(used=gpu_used, total=gpu_total, detail=gpu_detail)
        bars["tile"].set_usage(
            used=snapshot.display_cache.bytes_used,
            total=snapshot.display_cache.max_bytes,
            detail=_cache_tier_detail(snapshot.display_cache),
        )
        show_gpu = str(
            getattr(snapshot, "image_rendering_backend_actual", "")
        ) == "wgpu" and _gpu_available(snapshot)
        bars["gpu"].setVisible(show_gpu)
        bars["tile"].setVisible(not show_gpu)

    def _update_segment_bars(self, snapshot) -> None:
        self._segment_bars["render"].set_segments(
            _timing_segments(
                (
                    ("control", snapshot.render_timing.last_control_sync_ms, "#2563eb"),
                    ("planning", snapshot.render_timing.last_planning_ms, "#7c3aed"),
                    ("queue", snapshot.render_timing.last_worker_queue_wait_ms, "#ca8a04"),
                    ("eval", snapshot.render_timing.last_evaluation_ms, "#c2410c"),
                    ("commit", snapshot.render_timing.last_display_commit_ms, "#15803d"),
                    ("dock", snapshot.render_timing.last_operation_dock_ms, "#0891b2"),
                    ("inspect", snapshot.render_timing.last_inspection_refresh_ms, "#6b7280"),
                )
            ),
            summary=_timing_summary(
                "total",
                (
                    snapshot.render_timing.last_control_sync_ms,
                    snapshot.render_timing.last_planning_ms,
                    snapshot.render_timing.last_worker_queue_wait_ms,
                    snapshot.render_timing.last_evaluation_ms,
                    snapshot.render_timing.last_display_commit_ms,
                    snapshot.render_timing.last_operation_dock_ms,
                    snapshot.render_timing.last_inspection_refresh_ms,
                ),
            ),
        )
        self._segment_bars["montage"].set_segments(
            _timing_segments(
                (
                    ("tile", snapshot.montage_timing.last_tile_eval_ms, "#c2410c"),
                    ("cache", snapshot.montage_timing.last_display_cache_lookup_ms, "#2563eb"),
                    ("stage", snapshot.montage_timing.last_stage_cache_lookup_ms, "#0f766e"),
                    ("wait", snapshot.montage_timing.last_stage_attach_wait_ms, "#0d9488"),
                    ("levels", snapshot.montage_timing.last_level_stats_ms, "#a16207"),
                    ("upload", snapshot.montage_timing.last_visible_upload_ms, "#15803d"),
                    ("hist", snapshot.montage_timing.last_histogram_upload_ms, "#65a30d"),
                    ("rgb", snapshot.montage_timing.last_rgb_window_ms, "#db2777"),
                    ("tile upload", snapshot.montage_timing.last_tile_layer_upload_ms, "#7c3aed"),
                    ("set", snapshot.montage_timing.last_set_image_ms, "#15803d"),
                    ("commit", snapshot.montage_timing.last_tile_commit_ms, "#15803d"),
                    ("overlay", snapshot.montage_timing.last_overlay_update_ms, "#0891b2"),
                )
            ),
            summary=_timing_summary(
                "total",
                (
                    snapshot.montage_timing.last_tile_eval_ms,
                    snapshot.montage_timing.last_display_cache_lookup_ms,
                    snapshot.montage_timing.last_stage_cache_lookup_ms,
                    snapshot.montage_timing.last_stage_attach_wait_ms,
                    snapshot.montage_timing.last_level_stats_ms,
                    snapshot.montage_timing.last_visible_upload_ms,
                    snapshot.montage_timing.last_histogram_upload_ms,
                    snapshot.montage_timing.last_rgb_window_ms,
                    snapshot.montage_timing.last_tile_layer_upload_ms,
                    snapshot.montage_timing.last_set_image_ms,
                    snapshot.montage_timing.last_tile_commit_ms,
                    snapshot.montage_timing.last_overlay_update_ms,
                ),
            ),
        )

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh(force_text=True)
        self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        self._stop_logging()
        super().hideEvent(event)

    def closeEvent(self, event):
        self._timer.stop()
        self._stop_logging()
        super().closeEvent(event)


def _overview_label(name: str) -> QtWidgets.QLabel:
    return _ElidedOverviewLabel(f"{name}: n/a")


def _compact_bar_style(fraction: float, *, color_mode: str = "usage") -> str:
    if color_mode == "info":
        color = "#2563eb"
    elif fraction >= 0.85:
        color = "#c2410c"
    elif fraction >= 0.60:
        color = "#ca8a04"
    else:
        color = "#15803d"
    return (
        "QProgressBar {"
        " border: 1px solid palette(mid);"
        " border-radius: 2px;"
        " background: palette(base);"
        " color: palette(text);"
        " text-align: center;"
        " padding: 0;"
        " font-size: 8pt;"
        "}"
        f"QProgressBar::chunk {{ background-color: {color}; }}"
    )


def _timing_segments(items):
    segments = []
    for label, value, color in items:
        if value is None:
            continue
        scaled = max(1, round(float(value) * 1000.0))
        segments.append((label, scaled, color))
    return tuple(segments)


def _timing_summary(label: str, values) -> str:
    present = [float(value) for value in values if value is not None]
    if not present:
        return "n/a"
    return f"{label} {sum(present):.2f} ms"


def _short_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    value = float(max(0, int(value)))
    units = ("B", "K", "M", "G", "T")
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)}B"
    if value >= 100.0:
        return f"{value:.0f}{unit}"
    return f"{value:.1f}{unit}"


def _percent(used: int, total: int) -> str:
    return f"{(float(used) / max(1.0, float(total)) * 100.0):.0f}%"


def _cache_tier_detail(cache) -> str:
    """`used / total`, plus the host-cache compressed-tier status when engaged.

    Shared by the display and stage cache bars. The tier keeps the *same* byte
    budget while retaining more of the working set, so the raw ``used / total``
    figure barely moves when a codec is selected — the visible effect is the
    tier suffix (codec, compression ratio, retained keys, and decode
    recoveries). Without it a Host Cache Compression menu change leaves no trace
    in this dialog.
    """

    base = f"{_short_bytes(cache.bytes_used)} / {_short_bytes(cache.max_bytes)}"
    if not getattr(cache, "tier_engaged", False):
        return base
    codec = getattr(cache, "tier_codec", "") or "?"
    entries = int(getattr(cache, "tier_entries", 0) or 0)
    compressed = int(getattr(cache, "tier_compressed_bytes", 0) or 0)
    uncompressed = int(getattr(cache, "tier_resident_uncompressed_bytes", 0) or 0)
    recoveries = int(getattr(cache, "tier_recoveries", 0) or 0)
    parts = [base, f"{codec} tier"]
    if compressed > 0 and uncompressed > 0:
        parts.append(f"×{uncompressed / compressed:.1f}")
    parts.append(f"{entries} keys")
    if recoveries:
        parts.append(f"{recoveries} recov")
    return " · ".join(parts)


def _resource_segments(snapshot) -> tuple[tuple[str, int, str], ...]:
    policy = snapshot.memory_policy
    total = max(1, int(policy.system_total_bytes))
    process = max(0, min(total, int(policy.process_rss_bytes)))
    system_used = max(0, min(total, total - int(policy.system_available_bytes)))
    other_system = max(0, system_used - process)
    app_color = "#c2410c" if _memory_pressure_high(snapshot) else "#2563eb"
    return (
        ("system", other_system, "#15803d"),
        ("app", process, app_color),
    )


def _resource_bar_summary(snapshot) -> str:
    governor = snapshot.resource_governor
    if governor is None:
        return "CPU n/a load n/a"
    return (
        f"sys {_percent_value(getattr(governor, 'system_cpu_percent', None))} /"
        f"app {_percent_value(getattr(governor, 'process_cpu_percent', None))} |"
        f"load {_float_value(getattr(governor, 'load_average_1m', None))}"
    )


def _memory_pressure_high(snapshot) -> bool:
    governor = snapshot.resource_governor
    if governor is None:
        return False
    pressure = getattr(getattr(governor, "pressure", None), "memory_pressure", None)
    return str(getattr(pressure, "value", pressure)) == "high"


def _feedback_summary(snapshot) -> str:
    governor = snapshot.resource_governor
    if governor is None:
        return "feedback n/a"
    pressure = governor.pressure
    return (
        f"feedback ui {pressure.ui_pressure.value}, "
        f"mem {pressure.memory_pressure.value}, "
        f"cache {pressure.cache_pressure.value}"
    )


def _overview_state(snapshot) -> str:
    if snapshot.montage.active:
        return f"montage {snapshot.montage.display_mode}"
    decision = snapshot.render.last_decision_kind or "idle"
    return str(decision)


def _ops_overview(snapshot) -> str:
    total_ops = int(snapshot.operation_count)
    computed = int(snapshot.montage.loaded_tiles)
    total = max(0, int(snapshot.montage.visible_tiles))
    staged = int(snapshot.montage.tile_compute_stage_backed)
    optimized = (
        "n/a"
        if snapshot.optimized_operation_count is None
        else str(int(snapshot.optimized_operation_count))
    )
    return (
        f"Ops {optimized}/{total_ops}: "
        f"{staged}/{computed}/{total} ({_percent(computed, total or 1)})"
    )


def _drawn_tiles_overview(snapshot) -> str:
    drawn = int(
        snapshot.montage_timing.tile_layer_visible_items or snapshot.montage.presented_tiles
    )
    total = max(0, int(snapshot.montage.visible_tiles))
    return f"Drawn {drawn}/{total} ({_percent(drawn, total or 1)}) | Up: {format_bytes(_upload_total_bytes(snapshot))}"


def _upload_total_bytes(snapshot) -> int:
    timing = snapshot.montage_timing
    total = getattr(timing, "upload_total_bytes", None)
    if total is not None:
        return int(total)
    return (
        int(getattr(timing, "upload_visible_bytes", 0) or 0)
        + int(getattr(timing, "upload_histogram_bytes", 0) or 0)
        + int(getattr(timing, "tile_layer_texture_upload_bytes", 0) or 0)
    )


def _gpu_bar_usage(snapshot) -> tuple[int, int, str]:
    pools = tuple(getattr(snapshot.montage, "wgpu_page_pools", ()) or ())
    if pools:
        resident = sum(
            int(row.get("raw_resident_layers", 0) or 0)
            + int(row.get("codec_resident_layers", 0) or 0)
            for row in pools
        )
        pinned = sum(
            int(row.get("raw_pinned_layers", 0) or 0) + int(row.get("codec_pinned_layers", 0) or 0)
            for row in pools
        )
        allocated = sum(
            int(row.get("raw_allocated_layers", 0) or 0)
            + int(row.get("codec_allocated_layers", 0) or 0)
            for row in pools
        )
        budget_bytes = int(getattr(snapshot.montage_timing, "tile_layer_budget_bytes", 0) or 0)
        resident_bytes = int(getattr(snapshot.montage, "wgpu_active_resident_bytes", 0) or 0)
        if budget_bytes > 0:
            return (
                resident_bytes,
                budget_bytes,
                f"{_short_bytes(resident_bytes)} / {_short_bytes(budget_bytes)} "
                f"{_percent(resident_bytes, budget_bytes)} | "
                f"pages {resident}/{allocated} allocated | pinned {pinned} | "
                f"alloc {_short_bytes(snapshot.montage.wgpu_allocated_pool_bytes)}",
            )
        return (
            resident,
            max(1, allocated),
            f"pages {resident}/{allocated} {_percent(resident, allocated or 1)} | "
            f"pinned {pinned} | alloc "
            f"{_short_bytes(snapshot.montage.wgpu_allocated_pool_bytes)}",
        )
    timing = snapshot.montage_timing
    gpu_bytes = int(getattr(timing, "tile_layer_estimated_gpu_bytes", 0) or 0)
    budget_bytes = int(getattr(timing, "tile_layer_budget_bytes", 0) or 0)
    resident = int(getattr(timing, "tile_layer_resident_items", 0) or 0)
    capacity = int(getattr(timing, "tile_layer_storage_capacity", 0) or 0)
    slot_text = (
        "slots n/a"
        if capacity <= 0
        else f"slots {resident}/{capacity} {_percent(resident, capacity)}"
    )
    if budget_bytes > 0:
        return (
            gpu_bytes,
            budget_bytes,
            f"{_short_bytes(gpu_bytes)} / {_short_bytes(budget_bytes)} {_percent(gpu_bytes, budget_bytes)} | {slot_text}",
        )
    if capacity > 0:
        return resident, capacity, f"{slot_text} | {_short_bytes(gpu_bytes)}"
    return 0, 1, "n/a"


def _gpu_available(snapshot) -> bool:
    if tuple(getattr(snapshot.montage, "wgpu_page_pools", ()) or ()):
        return True
    timing = snapshot.montage_timing
    return bool(
        int(getattr(timing, "tile_layer_budget_bytes", 0) or 0) > 0
        or int(getattr(timing, "tile_layer_storage_capacity", 0) or 0) > 0
        or int(getattr(timing, "tile_layer_estimated_gpu_bytes", 0) or 0) > 0
    )


def _gpu_overview(snapshot) -> str:
    timing = snapshot.montage_timing
    gpu_bytes = int(getattr(timing, "tile_layer_estimated_gpu_bytes", 0) or 0)
    budget_bytes = int(getattr(timing, "tile_layer_budget_bytes", 0) or 0)
    resident = int(getattr(timing, "tile_layer_resident_items", 0) or 0)
    capacity = int(getattr(timing, "tile_layer_storage_capacity", 0) or 0)
    if budget_bytes > 0:
        return f"{_short_bytes(gpu_bytes)}/{_short_bytes(budget_bytes)} {_percent(gpu_bytes, budget_bytes)}"
    if capacity > 0:
        return f"slots {resident}/{capacity} {_percent(resident, capacity)}"
    return "n/a"


def _worker_segments(schedulers) -> tuple[tuple[str, int, str], ...]:
    segments = []
    colors = ("#2563eb", "#9333ea", "#0f766e", "#ca8a04", "#0891b2", "#64748b", "#dc2626")
    for scheduler, color in zip(schedulers, colors, strict=False):
        active = (
            int(getattr(scheduler, "pending", 0) or 0)
            + int(getattr(scheduler, "running", 0) or 0)
            + int(getattr(scheduler, "queued", 0) or 0)
        )
        if active:
            segments.append((scheduler.name, active, color))
    return tuple(segments)


def _active_work_summary(schedulers) -> str:
    parts = []
    for scheduler in schedulers:
        active = (
            int(getattr(scheduler, "pending", 0) or 0)
            + int(getattr(scheduler, "running", 0) or 0)
            + int(getattr(scheduler, "queued", 0) or 0)
        )
        if active:
            parts.append(f"{scheduler.name} {active}")
    return ", ".join(parts) if parts else "idle"


def _montage_overview(snapshot) -> str:
    if not snapshot.montage.active:
        return "inactive"
    return (
        f"{snapshot.montage.loaded_tiles}/{snapshot.montage.visible_tiles} loaded, "
        f"target unsettled {snapshot.montage.target_unsettled_tiles}, "
        f"resident {snapshot.montage_timing.tile_layer_resident_items}/"
        f"{snapshot.montage_timing.tile_layer_storage_capacity}, "
        f"updated {snapshot.montage_timing.tile_layer_items_updated}, "
        f"rgb tiles {snapshot.montage_timing.tile_layer_rgb_window_tiles}, "
        f"stage-backed {snapshot.montage.tile_compute_stage_backed}, "
        f"direct {snapshot.montage.tile_compute_direct}"
    )


def _ms_text(value) -> str:
    return "n/a" if value is None else f"{float(value):.2f} ms"


def _percent_value(value) -> str:
    return "n/a" if value is None else f"{float(value):.0f}%"


def _float_value(value) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"
