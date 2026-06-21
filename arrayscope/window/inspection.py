"""Window-side orchestration for ROI inspection workflows."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from arrayscope.core.compare import CompareDocument, compatible_roi_shape
from arrayscope.core.histograms import HistogramSpec, comparison_histograms
from arrayscope.core.roi import RoiKind, RoiStatsAccumulator, roi_bounding_rect, roi_statistics, roi_values, roi_values_for_region
from arrayscope.operations.evaluator import _document_key
from arrayscope.operations.tile_regions import TileRegionRequest
from arrayscope.window.tile_data_provider import TileDataProvider
from arrayscope.window.evaluation_controller import EvalPriority
from arrayscope.window.interaction_mode import InteractionMode


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
        if tool == "profile":
            self.interaction_mode = InteractionMode.LIVE_PROFILE
        else:
            self.interaction_mode = InteractionMode(tool)
            if hasattr(self, "widgets"):
                self.widgets["buttons"]["display"]["live_profile"].setChecked(False)
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
        if getattr(self, "_inspection_stale", False):
            self._refresh_inspection_dock_now()

    def _refresh_inspection_dock(self):
        from time import perf_counter

        start = perf_counter()
        self._schedule_refresh_inspection_dock("refresh")
        self._last_inspection_refresh_ms = (perf_counter() - start) * 1000.0

    def _inspection_panel_is_visible(self) -> bool:
        if not hasattr(self, "inspection_dock"):
            return False
        panel_manager = getattr(self, "panel_manager", None)
        if panel_manager is not None:
            try:
                return bool(panel_manager.is_visible("inspection"))
            except Exception:
                pass
        return bool(self.inspection_dock.isVisible())

    def _schedule_refresh_inspection_dock(self, reason):
        if not hasattr(self, "inspection_dock") or not hasattr(self, "img_view"):
            return
        self.roi_store = self.roi_store.replace_all(self.img_view.roiSelections())
        selections = self.roi_store.selections
        self.inspection_dock.set_rois(selections)
        if not self._inspection_panel_is_visible():
            self._inspection_stale = True
            stats_by_roi = self._hidden_roi_statistics(selections)
            self._update_roi_info_overlay(stats_by_roi)
            return
        if not hasattr(self, "_roi_refresh_timer"):
            self._roi_refresh_timer = Qt.QtCore.QTimer(self)
            self._roi_refresh_timer.setSingleShot(True)
            self._roi_refresh_timer.setInterval(60)
            self._roi_refresh_timer.timeout.connect(self._refresh_inspection_dock_now)
        decision = getattr(self, "_ui_work_decision", lambda *args, **kwargs: None)("roi_refresh", interactive=False)
        if decision is not None:
            self._roi_refresh_timer.setInterval(max(1, int(decision.interval_ms)))
        self._roi_refresh_reason = reason
        self._roi_refresh_timer.start()

    def _refresh_inspection_dock_now(self):
        from time import perf_counter

        start = perf_counter()
        try:
            if not hasattr(self, "inspection_dock") or not hasattr(self, "img_view"):
                return
            if not self._inspection_panel_is_visible():
                self._inspection_stale = True
                return
            self._inspection_stale = False
            self.roi_store = self.roi_store.replace_all(self.img_view.roiSelections())
            selections = self.roi_store.selections
            self.inspection_dock.set_rois(selections)
            if not any(selection.enabled for selection in selections):
                key = ("empty-roi", tuple((selection.id, selection.enabled, selection.geometry) for selection in selections))
                self._roi_inspection_request_key = key
                self._roi_inspection_in_flight = False
                if key != getattr(self, "_roi_inspection_applied_key", None):
                    self.inspection_dock.set_statistics(OrderedDict())
                    self.inspection_dock.set_histograms(())
                    self._update_roi_info_overlay(OrderedDict())
                    self._roi_inspection_applied_key = key
                return
            image = self._roi_source_image()
            layers = self._compatible_compare_layers(image) if image is not None else ()
            key = self._roi_inspection_key(image, selections, layers)
            if key == getattr(self, "_roi_inspection_request_key", None) and (
                getattr(self, "_roi_inspection_in_flight", False) or key == getattr(self, "_roi_inspection_applied_key", None)
            ):
                return
            self._roi_inspection_request_key = key
            work_size = 0 if image is None else int(np.size(image)) * max(1, sum(1 for selection in selections if selection.enabled))
            if work_size <= 250_000 and not self._roi_uses_montage_demand(selections):
                self._apply_roi_inspection_snapshot_if_current(key, self._compute_roi_inspection_snapshot(key, image, selections, layers))
                return
            self._roi_inspection_in_flight = True
            self.roi_evaluation_controller.start_latest(
                lambda key=key, image=image, selections=selections, layers=layers: self._compute_roi_inspection_snapshot(key, image, selections, layers),
                key=key,
                priority=EvalPriority.SELECTED_ROI,
                replace_group="roi-inspection",
                on_done=lambda snapshot, key=key: self._apply_roi_inspection_snapshot_if_current(key, snapshot),
                on_error=lambda exc: self._finish_roi_inspection_error(),
                slow_ms=0,
            )
        finally:
            self._last_inspection_refresh_ms = (perf_counter() - start) * 1000.0
            if hasattr(self, "_record_ui_work"):
                self._record_ui_work("roi_refresh", self._last_inspection_refresh_ms)

    def _roi_inspection_key(self, image, selections, layers):
        if self._roi_uses_montage_demand(selections):
            geometry = getattr(self, "display_geometry", None)
            image_key = (
                "montage-demand",
                _document_key(self.document),
                self.view_state,
                None if geometry is None else getattr(geometry, "montage", None),
            )
        else:
            image_key = None if image is None else (id(image), tuple(np.shape(image)), str(getattr(image, "dtype", None)))
        selection_key = tuple((selection.id, selection.enabled, selection.geometry) for selection in selections)
        layer_key = tuple((layer.label, id(layer.data), tuple(np.shape(layer.data)), str(getattr(layer.data, "dtype", None))) for layer in layers)
        return image_key, selection_key, layer_key

    def _compute_roi_inspection_snapshot(self, key, image, selections, layers):
        if self._roi_uses_montage_demand(selections):
            return self._compute_montage_roi_inspection_snapshot(key, selections)
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
        self._roi_inspection_in_flight = False
        self._roi_inspection_applied_key = key
        self.inspection_dock.set_statistics(snapshot.stats_by_roi)
        self.inspection_dock.set_histograms(snapshot.histograms)
        self._update_roi_info_overlay(snapshot.stats_by_roi)
        self._sync_progressive_docks()
        return True

    def _finish_roi_inspection_error(self):
        self._roi_inspection_in_flight = False

    def _roi_uses_montage_demand(self, selections) -> bool:
        if not selections:
            return False
        geometry = getattr(self, "display_geometry", None)
        return bool(geometry is not None and getattr(geometry, "montage", None) is not None)

    def _compute_montage_roi_inspection_snapshot(self, key, selections):
        stats_by_roi = OrderedDict()
        hist_inputs = []
        provider = self._tile_data_provider()
        plan = getattr(self, "_current_montage_plan", None)
        if provider is None or plan is None:
            return RoiInspectionSnapshot(key, stats_by_roi, ())
        for selection in selections:
            if not selection.enabled:
                continue
            accumulator = RoiStatsAccumulator()
            exact_values = []
            for tile, region in self._roi_tile_regions(selection.geometry, plan):
                request = TileRegionRequest(
                    document_key=_document_key(self.document),
                    view_state=tile.view_state,
                    montage_axis=getattr(self.view_state, "montage_axis", None),
                    source_index=tile.source_index,
                    tile_number=tile.montage_index,
                    tile_local_region=region,
                    purpose="roi",
                )
                result = provider.request_tile_region(request, priority=EvalPriority.SELECTED_ROI)
                source = result.histogram_data if result.histogram_data is not None else result.image
                y_slice, x_slice = region
                offset = (tile.x0 + int(x_slice.start or 0), tile.y0 + int(y_slice.start or 0))
                values = roi_values_for_region(source, selection.geometry, offset=offset)
                accumulator.add_values(values)
                finite = np.asarray(values).ravel()
                finite = finite[np.isfinite(finite)]
                if finite.size and sum(value.size for value in exact_values) + finite.size <= 250_000:
                    exact_values.append(finite.copy())
            stats = accumulator.result()
            stats_by_roi[selection.id] = (selection, stats)
            if exact_values:
                hist_inputs.append((selection.label, np.concatenate(exact_values)))
        return RoiInspectionSnapshot(key, stats_by_roi, comparison_histograms(hist_inputs, HistogramSpec(bins=96)))

    def _tile_data_provider(self):
        if not hasattr(self, "operation_evaluator"):
            return None
        evaluation_context = None
        if hasattr(self, "_evaluation_context"):
            evaluation_context = self._evaluation_context("roi")
        return TileDataProvider(
            operation_evaluator=self.operation_evaluator,
            document=self.document,
            committed_frame=getattr(self, "_committed_display_frame", None),
            montage_plan=getattr(self, "_current_montage_plan", None),
            colormap_lut=self._roi_colormap_lut(),
            evaluation_context=evaluation_context,
        )

    def _roi_colormap_lut(self):
        try:
            if getattr(getattr(self.view_state, "channel", None), "value", getattr(self.view_state, "channel", None)) == "phase":
                return self._phase_colormap().getLookupTable(0.0, 1.0, 256, alpha=False)
        except Exception:
            return None
        return None

    def _roi_tile_regions(self, geometry, plan):
        bounds = roi_bounding_rect(geometry)
        if bounds is None:
            return ()
        x0, y0, x1, y1 = bounds
        regions = []
        for tile in plan.tiles:
            tx0 = int(tile.x0)
            ty0 = int(tile.y0)
            tx1 = tx0 + int(tile.width)
            ty1 = ty0 + int(tile.height)
            if tx1 <= x0 or tx0 >= x1 or ty1 <= y0 or ty0 >= y1:
                continue
            if geometry.kind == RoiKind.RECTANGLE:
                rx0 = max(tx0, int(np.floor(x0)))
                rx1 = min(tx1, int(np.ceil(x1)))
                ry0 = max(ty0, int(np.floor(y0)))
                ry1 = min(ty1, int(np.ceil(y1)))
                if rx1 <= rx0 or ry1 <= ry0:
                    continue
                region = (slice(ry0 - ty0, ry1 - ty0), slice(rx0 - tx0, rx1 - tx0))
            else:
                region = (slice(0, int(tile.height)), slice(0, int(tile.width)))
            regions.append((tile, region))
        return tuple(regions)

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
        image = getattr(self.img_view, "image", None)
        if image is None or tuple(np.shape(source)[:2]) != tuple(np.shape(image)[:2]):
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

    def _hidden_roi_statistics(self, selections):
        image = self._roi_source_image()
        stats_by_roi = OrderedDict()
        if image is not None:
            for selection in selections:
                if not selection.enabled:
                    continue
                try:
                    stats_by_roi[selection.id] = (selection, roi_statistics(roi_values(image, selection.geometry)))
                except Exception:
                    continue
        return stats_by_roi
