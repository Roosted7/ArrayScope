"""Developer diagnostics dialog."""

from __future__ import annotations

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui, QtWidgets

from arrayscope.core.memory_budget import format_bytes
from arrayscope.core.runtime_diagnostics import format_runtime_diagnostics, format_runtime_diagnostics_sections


class _CompactUsageBar(QtWidgets.QProgressBar):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = str(label)
        self.setRange(0, 1000)
        self.setTextVisible(True)
        self.setMinimumHeight(16)
        self.setMaximumHeight(18)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)

    def set_usage(self, *, used: int, total: int, detail: str | None = None, color_mode: str = "usage") -> None:
        total = max(1, int(total))
        used = max(0, int(used))
        fraction = min(1.0, used / float(total))
        self.setValue(int(round(fraction * 1000)))
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
        self._segments = tuple((str(label), int(value), str(color)) for label, value, color in segments if int(value) > 0)
        self._summary = str(summary)
        self.setVisible(bool(self._segments))
        self.update()

    def paintEvent(self, event):
        del event
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
                width = remaining_width if index == len(self._segments) - 1 else int(round(rect.width() * value / total))
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
        painter.drawText(rect.adjusted(5, 0, -5, 0), Qt.QtCore.Qt.AlignmentFlag.AlignVCenter, f"{self._label}: {self._summary}")


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

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setPen(self.palette().text().color())
        metrics = painter.fontMetrics()
        text = metrics.elidedText(self._text, Qt.QtCore.Qt.TextElideMode.ElideRight, max(1, self.width() - 2))
        painter.drawText(self.rect(), Qt.QtCore.Qt.AlignmentFlag.AlignVCenter | Qt.QtCore.Qt.AlignmentFlag.AlignLeft, text)


class DiagnosticsDialog(QtWidgets.QDialog):
    def __init__(self, parent, snapshot_provider, *, interval_ms: int = 500):
        super().__init__(parent)
        self._snapshot_provider = snapshot_provider
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
            "resources": _overview_label("Resources"),
            "work": _overview_label("Work"),
            "render": _overview_label("Render"),
            "montage": _overview_label("Montage"),
        }
        overview_layout.addWidget(self._overview_labels["status"], 0, 0, 1, 2)
        overview_layout.addWidget(self._overview_labels["resources"], 1, 0)
        overview_layout.addWidget(self._overview_labels["work"], 1, 1)
        overview_layout.addWidget(self._overview_labels["render"], 2, 0)
        overview_layout.addWidget(self._overview_labels["montage"], 2, 1)
        bar_row = QtWidgets.QWidget()
        bar_layout = QtWidgets.QHBoxLayout(bar_row)
        bar_layout.setContentsMargins(0, 2, 0, 0)
        bar_layout.setSpacing(6)
        self._compact_bars = {
            "system": _CompactUsageBar("System"),
            "rss": _CompactUsageBar("RSS"),
            "render": _CompactUsageBar("Render"),
            "stage": _CompactUsageBar("Stage"),
            "tile": _CompactUsageBar("Tiles"),
            "canvas": _CompactUsageBar("Canvas"),
        }
        for bar in self._compact_bars.values():
            bar_layout.addWidget(bar, 1)
        overview_layout.addWidget(bar_row, 3, 0, 1, 2)
        segment_row = QtWidgets.QWidget()
        segment_layout = QtWidgets.QHBoxLayout(segment_row)
        segment_layout.setContentsMargins(0, 0, 0, 0)
        segment_layout.setSpacing(6)
        self._segment_bars = {
            "work": _CompactSegmentBar("Work"),
            "render": _CompactSegmentBar("Render"),
            "montage": _CompactSegmentBar("Montage"),
        }
        for bar in self._segment_bars.values():
            segment_layout.addWidget(bar, 1)
        overview_layout.addWidget(segment_row, 4, 0, 1, 2)
        layout.addWidget(overview, 0)

        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        self.tabs = QtWidgets.QTabWidget()
        self._section_edits = {}
        self._section_titles = tuple(format_runtime_diagnostics_sections(self._snapshot_provider()).keys())
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
        self.refresh_button = buttons.addButton("Auto text", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        self.refresh_button.setCheckable(True)
        self.refresh_button.setChecked(True)
        close_button = buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close)
        self.refresh_button.toggled.connect(lambda checked: self.refresh(force_text=True) if checked else None)
        close_button.clicked.connect(self.close)
        layout.addWidget(buttons)
        self.setLayout(layout)

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

    def _update_overview(self, snapshot) -> None:
        policy = snapshot.memory_policy
        system_used = max(0, int(policy.system_total_bytes) - int(policy.system_available_bytes))
        governor = snapshot.resource_governor
        pressure = "n/a" if governor is None else (
            f"ui {governor.pressure.ui_pressure.value}, "
            f"cpu headroom {governor.pressure.cpu_headroom:.0%}, "
            f"memory {governor.pressure.memory_pressure.value}"
        )
        self._overview_labels["status"].setText(
            f"{_overview_bottleneck(snapshot)} | {pressure}"
        )
        cache_parts = _interesting_cache_parts(snapshot)
        self._overview_labels["resources"].setText(
            f"RSS {_short_bytes(policy.process_rss_bytes)} | "
            f"sys {_short_bytes(system_used)}/{_short_bytes(policy.system_total_bytes)}"
            + (", " + ", ".join(cache_parts) if cache_parts else "")
        )
        self._overview_labels["work"].setText(
            "Work " + _active_work_summary(snapshot.schedulers)
        )
        self._overview_labels["render"].setText(
            f"{snapshot.render.last_decision_kind or 'n/a'} | "
            f"sync {_ms_text(snapshot.render_timing.last_render_sync_ms)}, "
            f"eval {_ms_text(snapshot.render_timing.last_evaluation_ms)}, "
            f"commit {_ms_text(snapshot.render_timing.last_display_commit_ms)}"
        )
        self._overview_labels["montage"].setText(
            "Montage " + _montage_overview(snapshot)
        )
        self._update_compact_bars(snapshot, system_used)
        self._update_segment_bars(snapshot)

    def _update_compact_bars(self, snapshot, system_used: int) -> None:
        policy = snapshot.memory_policy
        bars = self._compact_bars
        bars["system"].set_usage(used=system_used, total=policy.system_total_bytes)
        bars["system"].setFormat(f"System {_percent(system_used, policy.system_total_bytes)}")
        bars["rss"].set_usage(used=policy.process_rss_bytes, total=policy.system_total_bytes)
        bars["rss"].setFormat(f"RSS {_short_bytes(policy.process_rss_bytes)}")
        render_used = snapshot.render.estimated_display_bytes
        render_budget = snapshot.render.render_budget_bytes or policy.visible_render_budget_bytes
        bars["render"].set_usage(
            used=0 if render_used is None else render_used,
            total=render_budget,
            detail="n/a" if render_used is None else f"{_short_bytes(render_used)} / {_short_bytes(render_budget)}",
        )
        bars["stage"].set_usage(
            used=snapshot.stage_cache.bytes_used,
            total=snapshot.stage_cache.max_bytes,
            detail=f"{_short_bytes(snapshot.stage_cache.bytes_used)} / {_short_bytes(snapshot.stage_cache.max_bytes)}",
        )
        bars["tile"].set_usage(
            used=snapshot.tile_cache.bytes_used,
            total=snapshot.tile_cache.max_bytes,
            detail=f"{_short_bytes(snapshot.tile_cache.bytes_used)} / {_short_bytes(snapshot.tile_cache.max_bytes)}",
        )
        canvas_used = 0 if snapshot.montage.canvas_bytes is None else int(snapshot.montage.canvas_bytes)
        bars["canvas"].set_usage(
            used=canvas_used,
            total=policy.montage_canvas_budget_bytes,
            detail="n/a" if snapshot.montage.canvas_bytes is None else f"{_short_bytes(canvas_used)} / {_short_bytes(policy.montage_canvas_budget_bytes)}",
        )
        bars["stage"].setVisible(bool(snapshot.stage_cache.bytes_used or snapshot.stage_cache.entries))
        bars["tile"].setVisible(bool(snapshot.tile_cache.bytes_used or snapshot.montage.active))
        bars["canvas"].setVisible(bool(snapshot.montage.active or canvas_used))

    def _update_segment_bars(self, snapshot) -> None:
        work_segments = []
        for scheduler, color in zip(snapshot.schedulers, ("#2563eb", "#9333ea", "#0f766e", "#ca8a04", "#0891b2", "#64748b", "#dc2626")):
            active = int(getattr(scheduler, "pending", 0) or 0) + int(getattr(scheduler, "running", 0) or 0) + int(getattr(scheduler, "queued", 0) or 0)
            if active:
                work_segments.append((scheduler.name, active, color))
        self._segment_bars["work"].set_segments(work_segments, summary=_active_work_summary(snapshot.schedulers))
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
                    ("cache", snapshot.montage_timing.last_tile_cache_lookup_ms, "#2563eb"),
                    ("stage", snapshot.montage_timing.last_stage_cache_lookup_ms, "#0f766e"),
                    ("wait", snapshot.montage_timing.last_stage_attach_wait_ms, "#0d9488"),
                    ("levels", snapshot.montage_timing.last_level_stats_ms, "#a16207"),
                    ("upload", snapshot.montage_timing.last_visible_upload_ms, "#15803d"),
                    ("hist", snapshot.montage_timing.last_histogram_upload_ms, "#65a30d"),
                    ("rgb", snapshot.montage_timing.last_rgb_window_ms, "#db2777"),
                    ("compose", snapshot.montage_timing.last_canvas_compose_ms, "#7c3aed"),
                    ("patch", snapshot.montage_timing.last_canvas_patch_ms, "#9333ea"),
                    ("set", snapshot.montage_timing.last_set_image_ms, "#15803d"),
                    ("commit", snapshot.montage_timing.last_canvas_commit_ms, "#15803d"),
                    ("overlay", snapshot.montage_timing.last_overlay_update_ms, "#0891b2"),
                )
            ),
            summary=_timing_summary(
                "total",
                (
                    snapshot.montage_timing.last_tile_eval_ms,
                    snapshot.montage_timing.last_tile_cache_lookup_ms,
                    snapshot.montage_timing.last_stage_cache_lookup_ms,
                    snapshot.montage_timing.last_stage_attach_wait_ms,
                    snapshot.montage_timing.last_level_stats_ms,
                    snapshot.montage_timing.last_visible_upload_ms,
                    snapshot.montage_timing.last_histogram_upload_ms,
                    snapshot.montage_timing.last_rgb_window_ms,
                    snapshot.montage_timing.last_canvas_compose_ms,
                    snapshot.montage_timing.last_canvas_patch_ms,
                    snapshot.montage_timing.last_set_image_ms,
                    snapshot.montage_timing.last_canvas_commit_ms,
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
        super().hideEvent(event)

    def closeEvent(self, event):
        self._timer.stop()
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
        scaled = max(1, int(round(float(value) * 1000.0)))
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


def _overview_bottleneck(snapshot) -> str:
    governor = snapshot.resource_governor
    if governor is not None and governor.pressure.ui_pressure.value in {"elevated", "high"}:
        return "UI fan-in"
    if governor is not None and governor.pressure.memory_pressure.value in {"elevated", "high"}:
        return "memory"
    if snapshot.montage.active and snapshot.montage.tile_compute_waiting_for_stage:
        return "stage compute"
    if snapshot.montage.pending_tiles:
        return "tile compute"
    if snapshot.montage_timing.tile_layer_rgb_window_tiles:
        return "RGB/window upload"
    return "idle"


def _interesting_cache_parts(snapshot) -> list[str]:
    parts = []
    for label, cache in (
        ("image", snapshot.image_cache),
        ("tiles", snapshot.tile_cache),
        ("stage", snapshot.stage_cache),
    ):
        used = int(getattr(cache, "bytes_used", 0) or 0)
        total = max(1, int(getattr(cache, "max_bytes", 1) or 1))
        fraction = used / float(total)
        if used and (fraction >= 0.5 or label == "stage"):
            parts.append(f"{label} {format_bytes(used)} / {format_bytes(total)}")
    return parts


def _active_work_summary(schedulers) -> str:
    parts = []
    for scheduler in schedulers:
        active = int(getattr(scheduler, "pending", 0) or 0) + int(getattr(scheduler, "running", 0) or 0) + int(getattr(scheduler, "queued", 0) or 0)
        if active:
            parts.append(f"{scheduler.name} {active}")
    return ", ".join(parts) if parts else "idle"


def _montage_overview(snapshot) -> str:
    if not snapshot.montage.active:
        return "inactive"
    return (
        f"{snapshot.montage.loaded_tiles}/{snapshot.montage.visible_tiles} loaded, "
        f"pending {snapshot.montage.pending_tiles}, "
        f"updated {snapshot.montage_timing.tile_layer_items_updated}, "
        f"rgb tiles {snapshot.montage_timing.tile_layer_rgb_window_tiles}, "
        f"stage-backed {snapshot.montage.tile_compute_stage_backed}, "
        f"direct {snapshot.montage.tile_compute_direct}"
    )


def _ms_text(value) -> str:
    return "n/a" if value is None else f"{float(value):.2f} ms"
