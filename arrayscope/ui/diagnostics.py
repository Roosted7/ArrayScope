"""Developer diagnostics dialog."""

from __future__ import annotations

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui, QtWidgets

from arrayscope.core.memory_budget import format_bytes
from arrayscope.core.runtime_diagnostics import format_runtime_diagnostics, format_runtime_diagnostics_sections


class _SegmentBar(QtWidgets.QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = str(title)
        self._segments = ()
        self._summary = "no activity"
        self.setMinimumHeight(26)

    def set_segments(self, segments, *, summary: str) -> None:
        self._segments = tuple((str(label), int(value), str(color)) for label, value, color in segments if int(value) > 0)
        self._summary = str(summary)
        self.update()

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(0, 2, -1, -2)
        painter.setPen(QtGui.QPen(self.palette().mid().color(), 1))
        painter.setBrush(self.palette().base())
        painter.drawRoundedRect(rect, 3, 3)

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
        font.setPointSize(max(8, font.pointSize()))
        painter.setFont(font)
        painter.drawText(rect.adjusted(8, 0, -8, 0), Qt.QtCore.Qt.AlignmentFlag.AlignVCenter, f"{self._title}: {self._summary}")


class _UsageBar(QtWidgets.QProgressBar):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = str(label)
        self.setRange(0, 1000)
        self.setTextVisible(True)
        self.setMinimumHeight(22)

    def set_usage(self, *, used: int, total: int, detail: str | None = None, color_mode: str = "usage") -> None:
        total = max(1, int(total))
        used = max(0, int(used))
        fraction = min(1.0, used / float(total))
        self.setValue(int(round(fraction * 1000)))
        text = detail or f"{format_bytes(used)} / {format_bytes(total)}"
        self.setFormat(f"{self._label}: {text}")
        self.setStyleSheet(_bar_style(fraction, color_mode=color_mode))


class DiagnosticsDialog(QtWidgets.QDialog):
    def __init__(self, parent, snapshot_provider, *, interval_ms: int = 500):
        super().__init__(parent)
        self._snapshot_provider = snapshot_provider
        self.setWindowTitle("ArrayScope Diagnostics")
        self.resize(840, 720)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        overview = QtWidgets.QWidget()
        overview_layout = QtWidgets.QVBoxLayout(overview)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        overview_layout.setSpacing(8)

        bars = QtWidgets.QGridLayout()
        bars.setHorizontalSpacing(10)
        bars.setVerticalSpacing(6)
        self._bars = {
            "system": _UsageBar("System used"),
            "rss": _UsageBar("Process RSS"),
            "image": _UsageBar("Image cache"),
            "tile": _UsageBar("Tile cache"),
            "profile": _UsageBar("Profile cache"),
            "stage": _UsageBar("Stage cache"),
            "render": _UsageBar("Last render"),
            "canvas": _UsageBar("Montage canvas"),
            "prefetch": _UsageBar("Prefetch"),
        }
        for row, key in enumerate(("system", "rss", "image", "tile", "profile", "stage", "render", "canvas", "prefetch")):
            bars.addWidget(self._bars[key], row // 2, row % 2)
        overview_layout.addLayout(bars)

        scheduler_layout = QtWidgets.QVBoxLayout()
        scheduler_layout.setSpacing(4)
        scheduler_grid = QtWidgets.QGridLayout()
        scheduler_grid.setHorizontalSpacing(8)
        scheduler_grid.setVerticalSpacing(4)
        self._scheduler_bars = {}
        for index, name in enumerate(("visible", "pixel", "profile", "roi", "prefetch")):
            bar = _SegmentBar(name)
            self._scheduler_bars[name] = bar
            scheduler_grid.addWidget(bar, index // 2, index % 2)
        scheduler_layout.addLayout(scheduler_grid)
        self._canvas_preserve_bar = _SegmentBar("Canvas preserve")
        scheduler_layout.addWidget(self._canvas_preserve_bar)
        self._render_timing_bar = _SegmentBar("Render timing")
        self._montage_timing_bar = _SegmentBar("Montage timing")
        scheduler_layout.addWidget(self._render_timing_bar)
        scheduler_layout.addWidget(self._montage_timing_bar)
        overview_layout.addLayout(scheduler_layout)
        layout.addWidget(overview, 0)

        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        self.tabs = QtWidgets.QTabWidget()
        self._section_edits = {}
        for title in ("Memory", "Caches", "Schedulers", "Render", "Canvas Preserve", "Montage", "FFT", "Operations", "All"):
            edit = QtWidgets.QPlainTextEdit()
            edit.setReadOnly(True)
            edit.setFont(font)
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
            self._update_bars(snapshot)
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

    def _update_bars(self, snapshot) -> None:
        policy = snapshot.memory_policy
        system_used = max(0, int(policy.system_total_bytes) - int(policy.system_available_bytes))
        self._bars["system"].set_usage(used=system_used, total=policy.system_total_bytes)
        self._bars["rss"].set_usage(used=policy.process_rss_bytes, total=policy.system_total_bytes)
        self._bars["image"].set_usage(
            used=snapshot.image_cache.bytes_used,
            total=snapshot.image_cache.max_bytes,
            detail=f"entries={snapshot.image_cache.entries}, {format_bytes(snapshot.image_cache.bytes_used)} / {format_bytes(snapshot.image_cache.max_bytes)}",
        )
        self._bars["tile"].set_usage(
            used=snapshot.tile_cache.bytes_used,
            total=snapshot.tile_cache.max_bytes,
            detail=f"entries={snapshot.tile_cache.entries}, {format_bytes(snapshot.tile_cache.bytes_used)} / {format_bytes(snapshot.tile_cache.max_bytes)}",
        )
        self._bars["profile"].set_usage(
            used=snapshot.profile_cache.bytes_used,
            total=snapshot.profile_cache.max_bytes,
            detail=f"entries={snapshot.profile_cache.entries}, {format_bytes(snapshot.profile_cache.bytes_used)} / {format_bytes(snapshot.profile_cache.max_bytes)}",
        )
        stage_hit_rate = "n/a" if snapshot.stage_cache.hit_rate is None else f"{snapshot.stage_cache.hit_rate:.0%}"
        self._bars["stage"].set_usage(
            used=snapshot.stage_cache.bytes_used,
            total=snapshot.stage_cache.max_bytes,
            detail=(
                f"entries={snapshot.stage_cache.entries}, "
                f"{format_bytes(snapshot.stage_cache.bytes_used)} / {format_bytes(snapshot.stage_cache.max_bytes)}, "
                f"hit-rate={stage_hit_rate}"
            ),
        )
        render_used = snapshot.render.estimated_display_bytes
        render_budget = snapshot.render.render_budget_bytes or policy.visible_render_budget_bytes
        self._bars["render"].set_usage(
            used=0 if render_used is None else render_used,
            total=render_budget,
            detail=(
                f"n/a / {format_bytes(render_budget)}"
                if render_used is None
                else f"{format_bytes(render_used)} / {format_bytes(render_budget)}"
            ),
        )
        canvas_used = snapshot.montage.canvas_bytes
        self._bars["canvas"].set_usage(
            used=0 if canvas_used is None else canvas_used,
            total=policy.montage_canvas_budget_bytes,
            detail=(
                f"n/a / {format_bytes(policy.montage_canvas_budget_bytes)}"
                if canvas_used is None
                else f"{format_bytes(canvas_used)} / {format_bytes(policy.montage_canvas_budget_bytes)}"
            ),
        )
        self._bars["prefetch"].set_usage(
            used=policy.prefetch_budget_bytes,
            total=max(1, policy.stage_cache_budget_bytes),
            detail=f"{format_bytes(policy.prefetch_budget_bytes)} of stage {format_bytes(policy.stage_cache_budget_bytes)}",
            color_mode="info",
        )
        for scheduler in snapshot.schedulers:
            bar = self._scheduler_bars.get(scheduler.name)
            if bar is None:
                continue
            planned = int(scheduler.pending) + int(scheduler.queued) + int(scheduler.running)
            completed = int(scheduler.completed)
            cancelled = int(scheduler.cancelled)
            stale = int(scheduler.stale)
            failed = int(scheduler.failed)
            summary = (
                f"done {completed}, planned {planned}, "
                f"cancelled {cancelled}, stale {stale}, failed {failed}"
            )
            bar.set_segments(
                (
                    ("completed", completed, "#15803d"),
                    ("planned", planned, "#2563eb"),
                    ("cancelled", cancelled, "#ca8a04"),
                    ("stale", stale, "#6b7280"),
                    ("failed", failed, "#c2410c"),
                ),
                summary=summary,
            )
        preserve = snapshot.canvas_preserve
        result = str(preserve.last_result or "none")
        color = _preserve_color(result, bool(preserve.active), bool(preserve.strong_used))
        summary = (
            f"{result}, mode {preserve.mode}, attempts {preserve.attempts_used}, "
            f"strong {'yes' if preserve.strong_used else 'no'}"
        )
        self._canvas_preserve_bar.set_segments((("state", 1, color),), summary=summary)
        self._render_timing_bar.set_segments(
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
        self._montage_timing_bar.set_segments(
            _timing_segments(
                (
                    ("tile", snapshot.montage_timing.last_tile_eval_ms, "#c2410c"),
                    ("compose", snapshot.montage_timing.last_canvas_compose_ms, "#7c3aed"),
                    ("commit", snapshot.montage_timing.last_canvas_commit_ms, "#15803d"),
                    ("overlay", snapshot.montage_timing.last_overlay_update_ms, "#0891b2"),
                )
            ),
            summary=(
                f"{_timing_summary('total', (snapshot.montage_timing.last_tile_eval_ms, snapshot.montage_timing.last_canvas_compose_ms, snapshot.montage_timing.last_canvas_commit_ms, snapshot.montage_timing.last_overlay_update_ms))}, "
                f"tiles cached {snapshot.montage_timing.cached_tiles_last_session}, missing {snapshot.montage_timing.missing_tiles_last_session}"
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


def _bar_style(fraction: float, *, color_mode: str = "usage") -> str:
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
        " border-radius: 3px;"
        " background: palette(base);"
        " color: palette(text);"
        " text-align: center;"
        " padding: 1px;"
        "}"
        f"QProgressBar::chunk {{ background-color: {color}; border-radius: 2px; }}"
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


def _preserve_color(result: str, active: bool, strong_used: bool) -> str:
    if active:
        return "#2563eb"
    if "unsettled" in result:
        return "#ca8a04"
    if result.startswith("skipped"):
        return "#6b7280"
    if strong_used:
        return "#c2410c"
    if result == "settled":
        return "#15803d"
    return "#6b7280"
