"""Window-side orchestration for ROI inspection workflows."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from arrayscope.core.compare import CompareDocument, compatible_roi_shape
from arrayscope.core.histograms import HistogramSpec, comparison_histograms
from arrayscope.core.roi import RoiKind, roi_statistics, roi_values


@dataclass(frozen=True)
class RoiInspectionSnapshot:
    key: tuple
    stats_by_roi: OrderedDict
    histograms: tuple


class InspectionWorkflowMixin:
    def _init_compare_document(self, data):
        self.compare_document = CompareDocument.from_base(data)

    def _add_compare_layer(self, data, label=None):
        if not hasattr(self, "compare_document"):
            self._init_compare_document(getattr(self, "base_data", data))
        self.compare_document = self.compare_document.with_layer(data, label=label)
        self._refresh_inspection_dock_now()
        return self.compare_document.layers[-1]

    def _on_inspection_tool_changed(self, tool):
        if hasattr(self, "img_view"):
            self.img_view.setInspectionTool(tool)
        if tool == "profile":
            self.widgets["buttons"]["display"]["live_profile"].setChecked(True)

    def _add_roi_for_tool(self, tool):
        return self._add_roi_for_tool_at(tool, None)

    def _add_roi_for_tool_at(self, tool, image_point):
        if not hasattr(self, "img_view"):
            return None
        if tool in {"roi_polyline", "roi_freehand"}:
            self.img_view.beginRoiDrawingOnce(tool)
            return None
        mapping = {
            "roi_line": RoiKind.LINE,
            "roi_rectangle": RoiKind.RECTANGLE,
            "roi_polyline": RoiKind.POLYLINE,
            "roi_freehand": RoiKind.FREEHAND_POLYGON,
        }
        kind = mapping.get(tool)
        if kind is None:
            return None
        kwargs = self._roi_kwargs_for_point(kind, image_point)
        selection = self.img_view.createRoi(kind, **kwargs)
        return selection

    def _on_roi_created(self, selection):
        self.roi_store = self.roi_store.upsert(selection)
        self._refresh_inspection_dock()

    def _on_roi_changed(self, _roi_id, _geometry):
        if hasattr(self, "img_view"):
            self.roi_store = self.roi_store.replace_all(self.img_view.roiSelections()).select(_roi_id)
        self._refresh_inspection_dock()

    def _on_roi_deleted(self, _roi_id):
        self.roi_store = self.roi_store.remove(_roi_id)
        self._refresh_inspection_dock()

    def _delete_roi(self, roi_id):
        if hasattr(self, "img_view"):
            self.img_view.removeRoi(roi_id)
        self._refresh_inspection_dock()

    def _clear_rois(self):
        if hasattr(self, "img_view"):
            self.img_view.clearRois()
        self.roi_store = self.roi_store.clear()
        self._refresh_inspection_dock()

    def _select_roi(self, roi_id):
        self.roi_store = self.roi_store.select(roi_id)
        if hasattr(self, "img_view"):
            self.img_view.highlightRoi(roi_id)

    def _show_inspection_dock(self):
        if not hasattr(self, "inspection_dock"):
            return
        self._inspection_dock_user_visible = True
        self.layout_manager.set_managed_dock_visible(self.inspection_dock, True, reason="show-inspection")

    def _refresh_inspection_dock(self):
        self._schedule_refresh_inspection_dock("refresh")

    def _schedule_refresh_inspection_dock(self, reason):
        if not hasattr(self, "inspection_dock") or not hasattr(self, "img_view"):
            return
        self.roi_store = self.roi_store.replace_all(self.img_view.roiSelections())
        selections = self.roi_store.selections
        self.inspection_dock.set_rois(selections)
        if not hasattr(self, "_roi_refresh_timer"):
            self._roi_refresh_timer = Qt.QtCore.QTimer(self)
            self._roi_refresh_timer.setSingleShot(True)
            self._roi_refresh_timer.setInterval(60)
            self._roi_refresh_timer.timeout.connect(self._refresh_inspection_dock_now)
        self._roi_refresh_reason = reason
        self._roi_refresh_timer.start()

    def _refresh_inspection_dock_now(self):
        if not hasattr(self, "inspection_dock") or not hasattr(self, "img_view"):
            return
        self.roi_store = self.roi_store.replace_all(self.img_view.roiSelections())
        selections = self.roi_store.selections
        self.inspection_dock.set_rois(selections)
        image = self._roi_source_image()
        layers = self._compatible_compare_layers(image) if image is not None else ()
        key = self._roi_inspection_key(image, selections, layers)
        self._roi_inspection_request_key = key
        work_size = 0 if image is None else int(np.size(image)) * max(1, sum(1 for selection in selections if selection.enabled))
        if work_size <= 250_000:
            self._apply_roi_inspection_snapshot_if_current(key, self._compute_roi_inspection_snapshot(key, image, selections, layers))
            return
        self.roi_evaluation_controller.start(
            lambda key=key, image=image, selections=selections, layers=layers: self._compute_roi_inspection_snapshot(key, image, selections, layers),
            on_done=lambda snapshot, key=key: self._apply_roi_inspection_snapshot_if_current(key, snapshot),
            on_error=lambda exc: None,
            slow_ms=0,
        )

    def _roi_inspection_key(self, image, selections, layers):
        image_key = None if image is None else (id(image), tuple(np.shape(image)), str(getattr(image, "dtype", None)))
        selection_key = tuple((selection.id, selection.enabled, selection.geometry) for selection in selections)
        layer_key = tuple((layer.label, id(layer.data), tuple(np.shape(layer.data)), str(getattr(layer.data, "dtype", None))) for layer in layers)
        return image_key, selection_key, layer_key

    def _compute_roi_inspection_snapshot(self, key, image, selections, layers):
        stats_by_roi = OrderedDict()
        hist_inputs = []
        if image is not None:
            for selection in selections:
                if not selection.enabled:
                    continue
                values = roi_values(image, selection.geometry)
                stats = roi_statistics(values)
                stats_by_roi[selection.id] = (selection, stats)
                finite = np.asarray(values).ravel()
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    hist_inputs.append((selection.label, finite))
                for layer in layers:
                    layer_values = roi_values(layer.data, selection.geometry)
                    layer_finite = np.asarray(layer_values).ravel()
                    layer_finite = layer_finite[np.isfinite(layer_finite)]
                    if layer_finite.size:
                        hist_inputs.append((f"{selection.label} / {layer.label}", layer_finite))
        return RoiInspectionSnapshot(key, stats_by_roi, comparison_histograms(hist_inputs, HistogramSpec(bins=96)))

    def _apply_roi_inspection_snapshot_if_current(self, key, snapshot):
        if key != getattr(self, "_roi_inspection_request_key", None):
            return False
        self.inspection_dock.set_statistics(snapshot.stats_by_roi)
        self.inspection_dock.set_histograms(snapshot.histograms)
        self._update_roi_info_overlay(snapshot.stats_by_roi)
        self._sync_progressive_docks()
        return True

    def _compatible_compare_layers(self, reference_image):
        if not hasattr(self, "compare_document"):
            return ()
        layers = []
        for layer in self.compare_document.layers[1:]:
            if not layer.visible:
                continue
            if compatible_roi_shape(layer.data, np.shape(reference_image)):
                layers.append(layer)
        return tuple(layers)

    def _roi_source_image(self):
        if not hasattr(self, "img_view"):
            return None
        source = getattr(self.img_view, "histogramSource", None)
        if source is None:
            source = getattr(self.img_view, "image", None)
        if source is None:
            return None
        return np.asarray(source)

    def _set_inspection_dock_visible_from_user(self, visible):
        self.layout_manager.set_inspection_dock_visible_from_user(visible)

    def _show_image_context_menu(self, global_pos, image_point=None):
        menu = QtWidgets.QMenu(self)
        live = self.widgets["buttons"]["display"]["live_profile"].isChecked()
        profile_action = menu.addAction("Live profile")
        profile_action.setCheckable(True)
        profile_action.setChecked(live)
        profile_action.triggered.connect(lambda checked=False: self._set_live_profile_from_context(bool(checked), image_point))
        menu.addSeparator()
        for label, tool in (
            ("Add line ROI", "roi_line"),
            ("Add rectangle ROI", "roi_rectangle"),
            ("Draw polyline ROI", "roi_polyline"),
            ("Draw freehand ROI", "roi_freehand"),
        ):
            action = menu.addAction(label)
            action.triggered.connect(lambda checked=False, tool=tool, image_point=image_point: self._add_roi_for_tool_at(tool, image_point))
        menu.addSeparator()
        show_inspection = menu.addAction("Show inspection dock")
        show_inspection.triggered.connect(self._show_inspection_dock)
        clear_rois = menu.addAction("Clear ROIs")
        clear_rois.setEnabled(hasattr(self, "img_view") and bool(self.img_view.roiSelections()))
        clear_rois.triggered.connect(self._clear_rois)
        menu.exec(global_pos)

    def _set_live_profile_from_context(self, enabled, image_point=None):
        self.widgets["buttons"]["display"]["live_profile"].setChecked(bool(enabled))
        if enabled and image_point is not None:
            self.img_view.setProfileMarker(image_point[0], image_point[1], visible=True)
            self._on_profile_marker_moved(image_point[0], image_point[1])

    def _roi_kwargs_for_point(self, kind, image_point):
        if image_point is None:
            return {}
        x, y = (float(image_point[0]), float(image_point[1]))
        if kind == RoiKind.LINE:
            return {"points": ((x - 12, y), (x + 12, y))}
        if kind == RoiKind.RECTANGLE:
            return {"rect": (x - 10, y - 10, 20, 20)}
        if kind == RoiKind.POLYLINE:
            return {"points": ((x - 10, y - 6), (x, y + 8), (x + 10, y - 6))}
        if kind == RoiKind.FREEHAND_POLYGON:
            return {"points": ((x - 10, y - 10), (x + 10, y - 10), (x + 10, y + 10), (x - 10, y + 10))}
        return {}

    def _update_roi_info_overlay(self, stats_by_roi):
        if not hasattr(self, "img_view"):
            return
        lines = []
        for _roi_id, (selection, stats) in list(stats_by_roi.items())[:6]:
            label = selection.label
            kind = selection.geometry.kind.value.replace("_", " ")
            mean = "" if stats.mean is None or not np.isfinite(stats.mean) else f" mean={stats.mean:.4g}"
            count = f" n={stats.finite_count}"
            lines.append(f"{label}: {kind}{count}{mean}")
        self.img_view.setRoiInfoText("\n".join(lines))
