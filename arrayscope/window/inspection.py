"""Window-side orchestration for ROI inspection workflows."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from arrayscope.core.compare import CompareDocument
from arrayscope.core.frame_targets import FrameTarget
from arrayscope.core.histograms import HistogramSpec, comparison_histograms
from arrayscope.core.roi import (
    RoiKind,
    RoiStatsAccumulator,
    roi_bounding_rect,
    roi_values_for_region,
)
from arrayscope.kernel import Lane as WorkLane
from arrayscope.kernel import Priority, WorkItem
from arrayscope.operations.evaluator import _document_key
from arrayscope.operations.tile_regions import TileRegionRequest
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.interaction_mode import InteractionMode
from arrayscope.window.tile_data_provider import TileDataProvider


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
        self._refresh_inspection_dock()
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
            self.roi_store = self.roi_store.replace_all(self.img_view.roiSelections()).select(
                _roi_id
            )
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
        self._notify_sync("rois")

    def _select_roi(self, roi_id):
        self.roi_store = self.roi_store.select(roi_id)
        if hasattr(self, "img_view"):
            self.img_view.highlightRoi(roi_id)
        self._notify_sync("rois")

    def _show_inspection_dock(self):
        if not hasattr(self, "inspection_dock"):
            return
        self._inspection_dock_user_visible = True
        self.layout_manager.set_managed_dock_visible(
            self.inspection_dock, True, reason="show-inspection"
        )
        if getattr(self, "_inspection_stale", False):
            self._refresh_inspection_dock()

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

    #: How long commit-driven hidden refreshes wait before submitting, so a
    #: burst of frame commits collapses to one ROI evaluation.
    _HIDDEN_ROI_REFRESH_COALESCE_MS = 50

    #: Refresh reasons that originate from a display/montage frame commit. A
    #: live montage fires these continuously while refining; submitting one ROI
    #: job per commit pins a worker thread at ~100% (BUG 3), because the dedup
    #: key changes every commit and never holds. These are coalesced behind a
    #: single-shot timer. ROI edits and other reasons stay immediate so the
    #: on-image overlay reacts promptly to user changes.
    _COMMIT_DRIVEN_ROI_REFRESH_REASONS = frozenset(
        {"display-commit", "montage-semantic-commit", "montage-layout-reflow"}
    )

    def _schedule_refresh_inspection_dock(self, reason):
        if not hasattr(self, "inspection_dock") or not hasattr(self, "img_view"):
            return
        self.roi_store = self.roi_store.replace_all(self.img_view.roiSelections())
        selections = self.roi_store.selections
        self.inspection_dock.set_rois(selections)
        if not any(selection.enabled for selection in selections):
            self._apply_empty_inspection_state_if_needed(selections)
            return
        if not self._inspection_panel_is_visible():
            self._inspection_stale = True
            if self._roi_uses_montage_demand(selections) and self._montage_roi_values_pending():
                return
            if str(reason) in self._COMMIT_DRIVEN_ROI_REFRESH_REASONS:
                self._schedule_hidden_roi_inspection_refresh(reason)
                return
            self._roi_refresh_reason = reason
            self._queue_roi_inspection_refresh(selections, panel_visible=False)
            return
        self._roi_refresh_reason = reason
        self._queue_roi_inspection_refresh(selections, panel_visible=True)

    def _schedule_hidden_roi_inspection_refresh(self, reason):
        """Coalesce a commit-driven hidden refresh behind a single-shot timer."""
        # No consumer at all: the dock is user-closed and nothing on-image
        # needs keeping current. Do not wake the ROI worker lane.
        if getattr(self, "_inspection_dock_user_visible", None) is False:
            selections_fn = getattr(self.img_view, "roiSelections", None)
            selections = tuple(selections_fn()) if callable(selections_fn) else ()
            if not any(selection.enabled for selection in selections):
                return
        self._pending_hidden_roi_refresh_reason = str(reason)
        timer = getattr(self, "_hidden_roi_refresh_timer", None)
        if timer is None:
            # Timer category: UI cosmetic. Debounces a burst of frame commits
            # into a single hidden ROI refresh; the callback re-derives
            # everything from the current selections, so this is pure
            # rescheduling, not an ordering source for any frame semantics.
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._drain_hidden_roi_inspection_refresh)
            self._hidden_roi_refresh_timer = timer
        timer.start(int(self._HIDDEN_ROI_REFRESH_COALESCE_MS))

    @Qt.QtCore.Slot()
    def _drain_hidden_roi_inspection_refresh(self):
        if not hasattr(self, "inspection_dock") or not hasattr(self, "img_view"):
            return
        if self._inspection_panel_is_visible():
            return
        selections = tuple(self.img_view.roiSelections())
        if not any(selection.enabled for selection in selections):
            self._update_roi_info_overlay(OrderedDict())
            return
        if self._roi_uses_montage_demand(selections) and self._montage_roi_values_pending():
            self._inspection_stale = True
            return
        self._roi_refresh_reason = getattr(
            self, "_pending_hidden_roi_refresh_reason", "display-commit"
        )
        self._queue_roi_inspection_refresh(selections, panel_visible=False)

    def _quiesce_hidden_roi_refresh(self):
        """Stop all hidden-ROI background work when the dock is closed.

        A live/refining source would otherwise keep feeding the ROI worker
        lane after the panel that consumes the result is gone. Stops the
        coalescing timer and cancels any in-flight/queued ROI evaluation. A
        later ROI edit or a reopen wakes the pipeline again.
        """
        timer = getattr(self, "_hidden_roi_refresh_timer", None)
        if timer is not None:
            timer.stop()
        self._pending_hidden_roi_refresh_reason = None
        controller = getattr(self, "roi_evaluation_controller", None)
        if controller is not None:
            controller.clear_group("roi-inspection")
        self._roi_inspection_in_flight = False
        self._roi_inspection_request_key = None

    def _on_managed_panel_closed_by_user(self, name):
        """Panel-manager hook: the user closed a detached panel's dialog.

        Mirrors the docked-close path so a floating inspection dock that is
        closed also quiesces the ROI pipeline.
        """
        if str(name) == "inspection":
            self._inspection_dock_user_visible = False
            self._quiesce_hidden_roi_refresh()

    def _queue_roi_inspection_refresh(self, selections, *, panel_visible: bool) -> None:
        self._pending_roi_inspection_refresh = SimpleNamespace(
            selections=tuple(selections),
            panel_visible=bool(panel_visible),
        )
        if bool(getattr(self, "_roi_inspection_refresh_queued", False)):
            return
        self._roi_inspection_refresh_queued = True
        try:
            queued = Qt.QtCore.QMetaObject.invokeMethod(
                self,
                "_drain_pending_roi_inspection_refresh",
                Qt.QtCore.Qt.ConnectionType.QueuedConnection,
            )
        except Exception:
            queued = False
        if not queued:
            self._roi_inspection_refresh_queued = False
            self._drain_pending_roi_inspection_refresh()

    @Qt.QtCore.Slot()
    def _drain_pending_roi_inspection_refresh(self) -> None:
        self._roi_inspection_refresh_queued = False
        pending = getattr(self, "_pending_roi_inspection_refresh", None)
        self._pending_roi_inspection_refresh = None
        if pending is None:
            return
        self._submit_roi_inspection_refresh(
            tuple(getattr(pending, "selections", ()) or ()),
            panel_visible=bool(getattr(pending, "panel_visible", False)),
        )

    def _refresh_inspection_dock_now(self):
        if not hasattr(self, "inspection_dock") or not hasattr(self, "img_view"):
            return
        self.roi_store = self.roi_store.replace_all(self.img_view.roiSelections())
        selections = self.roi_store.selections
        self.inspection_dock.set_rois(selections)
        if not any(selection.enabled for selection in selections):
            self._apply_empty_inspection_state_if_needed(selections)
            return
        panel_visible = self._inspection_panel_is_visible()
        self._inspection_stale = not panel_visible
        if self._roi_uses_montage_demand(selections) and self._montage_roi_values_pending():
            self._inspection_stale = True
            return
        self._queue_roi_inspection_refresh(selections, panel_visible=panel_visible)

    def _submit_roi_inspection_refresh(self, selections, *, panel_visible: bool) -> None:
        from time import perf_counter

        start = perf_counter()
        try:
            controller = getattr(self, "roi_evaluation_controller", None)
            if controller is None:
                return
            destination = "visible" if panel_visible else "hidden"
            roi_key = self._roi_inspection_key(selections)
            request_key = (destination, roi_key)
            if request_key == getattr(self, "_roi_inspection_request_key", None) and (
                getattr(self, "_roi_inspection_in_flight", False)
                or request_key == getattr(self, "_roi_inspection_applied_key", None)
            ):
                return
            self._roi_inspection_request_key = request_key
            priority = self._roi_refresh_priority(selections, panel_visible=panel_visible)
            self._roi_inspection_priority = priority
            work_size = self._roi_inspection_work_size(selections)
            frame_target = FrameTarget(
                semantic_key=request_key,
                viewport_key=(
                    "roi",
                    tuple(selection.id for selection in selections if selection.enabled),
                ),
                presentation_key=("roi-inspection", destination),
                quality="exact-visible",
            )
            decision = getattr(self, "_gui_callback_budget_decision", lambda *args, **kwargs: None)(
                "roi_refresh", interactive=False
            )
            if decision is not None:
                controller.set_max_callback_dispatch_per_drain(decision.batch_limit)
                if hasattr(controller, "set_callback_budget_ms"):
                    controller.set_callback_budget_ms(decision.budget_ms)
            self._roi_inspection_in_flight = True
            controller.start_latest(
                lambda key=roi_key, selections=selections: self._compute_roi_inspection_snapshot(
                    key, selections
                ),
                key=request_key,
                priority=priority,
                replace_group="roi-inspection",
                frame_target=frame_target,
                supersession_key="roi-inspection",
                supersession_value=request_key,
                work_item=WorkItem(
                    key=("roi_inspection", request_key),
                    lane=WorkLane.PROFILE_ROI_HOVER,
                    frame_target=frame_target,
                    supersession_key="roi-inspection",
                    supersession_value=request_key,
                    estimated_cpu_ms=0.0,
                    estimated_bytes=int(work_size),
                    expected_value=2.0,
                    reusable_output=False,
                ),
                on_done=(
                    (
                        lambda snapshot, key=request_key: (
                            self._apply_roi_inspection_snapshot_if_current(key, snapshot)
                        )
                    )
                    if panel_visible
                    else (
                        lambda snapshot, key=request_key: (
                            self._apply_hidden_roi_overlay_snapshot_if_current(key, snapshot)
                        )
                    )
                ),
                on_stale=lambda: setattr(self, "_inspection_stale", True),
                on_error=lambda exc: self._finish_roi_inspection_error(),
                slow_ms=0,
            )
        finally:
            self._last_inspection_refresh_ms = (perf_counter() - start) * 1000.0
            if hasattr(self, "_record_ui_work"):
                self._record_ui_work("roi_refresh", self._last_inspection_refresh_ms)

    def _apply_empty_inspection_state_if_needed(self, selections) -> None:
        key = (
            "empty-roi",
            tuple(
                (selection.id, selection.enabled, selection.geometry) for selection in selections
            ),
        )
        self._roi_inspection_request_key = key
        self._roi_inspection_in_flight = False
        if key == getattr(self, "_roi_inspection_applied_key", None):
            return
        self.inspection_dock.set_statistics(OrderedDict())
        self.inspection_dock.set_histograms(())
        self._update_roi_info_overlay(OrderedDict())
        self._roi_inspection_applied_key = key

    def _roi_inspection_key(self, selections):
        if self._roi_uses_montage_demand(selections):
            geometry = getattr(self, "display_geometry", None)
            source_key = (
                "montage-demand",
                _document_key(self.document),
                self.view_state,
                None if geometry is None else getattr(geometry, "montage", None),
            )
        else:
            frame = self._committed_tiled_frame()
            scene = None if frame is None else getattr(frame, "scene", None)
            value_source = None if frame is None else getattr(frame, "value_source", None)
            payloads = (
                {} if value_source is None else dict(getattr(value_source, "payloads", {}) or {})
            )
            source_key = (
                "tiled-demand",
                None if frame is None else getattr(frame, "key", None),
                tuple(
                    (
                        int(region.region_id),
                        tuple(float(value) for value in region.bounds),
                        bool(region.resident),
                    )
                    for region in tuple(getattr(scene, "regions", ()) or ())
                ),
                tuple(
                    (int(key), getattr(payload, "source_id", None))
                    for key, payload in sorted(payloads.items(), key=lambda item: int(item[0]))
                ),
            )
        selection_key = tuple(
            (selection.id, selection.enabled, selection.geometry) for selection in selections
        )
        return source_key, selection_key

    def _roi_inspection_work_size(self, selections) -> int:
        total = 0
        for selection in selections:
            if not selection.enabled:
                continue
            bounds = roi_bounding_rect(selection.geometry)
            if bounds is None:
                continue
            x0, y0, x1, y1 = bounds
            total += max(1, int(np.ceil(x1) - np.floor(x0))) * max(
                1, int(np.ceil(y1) - np.floor(y0))
            )
        return int(total)

    def _compute_roi_inspection_snapshot(self, key, selections):
        if self._roi_uses_montage_demand(selections):
            return self._compute_montage_roi_inspection_snapshot(key, selections)
        return self._compute_tiled_roi_inspection_snapshot(key, selections)

    def _apply_roi_inspection_snapshot_if_current(self, key, snapshot):
        if key != getattr(self, "_roi_inspection_request_key", None):
            return False
        self._roi_inspection_in_flight = False
        self._roi_inspection_applied_key = key
        self._inspection_stale = False
        self.inspection_dock.set_statistics(snapshot.stats_by_roi)
        self.inspection_dock.set_histograms(snapshot.histograms)
        self._update_roi_info_overlay(snapshot.stats_by_roi)
        self._sync_progressive_docks()
        return True

    def _apply_hidden_roi_overlay_snapshot_if_current(self, key, snapshot):
        if key != getattr(self, "_roi_inspection_request_key", None):
            return False
        self._roi_inspection_in_flight = False
        self._roi_inspection_applied_key = key
        self._inspection_stale = True
        self._update_roi_info_overlay(snapshot.stats_by_roi)
        return True

    def _finish_roi_inspection_error(self):
        self._roi_inspection_in_flight = False

    def _roi_uses_montage_demand(self, selections) -> bool:
        if not selections:
            return False
        return getattr(getattr(self, "view_state", None), "montage_axis", None) is not None

    def _roi_refresh_priority(self, selections, *, panel_visible: bool) -> Priority:
        if not panel_visible:
            return Priority.HIDDEN_ROI
        reason = str(getattr(self, "_roi_refresh_reason", "") or "")
        if self._roi_uses_montage_demand(selections) and reason != "refresh":
            return Priority.VISIBLE_ROI
        return Priority.SELECTED_ROI

    def _montage_roi_values_pending(self) -> bool:
        renderer = getattr(self, "renderer", None)
        session = None if renderer is None else getattr(renderer, "_frame_session", None)
        if session is None:
            return False
        current_view_state = getattr(self, "view_state", None)
        session_view_state = getattr(session, "view_state", None)
        if current_view_state is not None and session_view_state != current_view_state:
            return True
        visible_plan_complete = getattr(session, "visible_plan_complete", None)
        if not callable(visible_plan_complete):
            raise RuntimeError("live frame session has no completion owner")
        return not bool(visible_plan_complete())

    def _roi_uses_tiled_demand(self, selections) -> bool:
        return bool(selections and self._committed_tiled_frame() is not None)

    def _current_committed_display_frame(self):
        frame = getattr(self, "_committed_display_frame", None)
        if frame is None:
            return None
        is_current = getattr(self, "_is_committed_display_frame_current", None)
        if callable(is_current) and not is_current(frame):
            return None
        if getattr(frame, "geometry", None) != getattr(self, "display_geometry", None):
            return None
        return frame

    def _committed_tiled_frame(self):
        frame = self._current_committed_display_frame()
        if frame is None or not getattr(frame, "is_tiled", False):
            return None
        return frame

    def _compute_tiled_roi_inspection_snapshot(self, key, selections):
        stats_by_roi, hist_inputs = self._committed_tiled_roi_values(
            selections, collect_histograms=True
        )
        return RoiInspectionSnapshot(
            key, stats_by_roi, comparison_histograms(hist_inputs, HistogramSpec(bins=96))
        )

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
                result = provider.request_tile_region(
                    request,
                    priority=getattr(self, "_roi_inspection_priority", Priority.HIDDEN_ROI),
                )
                source = (
                    result.histogram_data if result.histogram_data is not None else result.image
                )
                y_slice, x_slice = region
                offset = (tile.x0 + int(x_slice.start or 0), tile.y0 + int(y_slice.start or 0))
                values = roi_values_for_region(source, selection.geometry, offset=offset)
                accumulator.add_values(values)
                finite = np.asarray(values).ravel()
                finite = finite[np.isfinite(finite)]
                if (
                    finite.size
                    and sum(value.size for value in exact_values) + finite.size <= 250_000
                ):
                    exact_values.append(finite.copy())
            stats = accumulator.result()
            stats_by_roi[selection.id] = (selection, stats)
            if exact_values:
                hist_inputs.append((selection.label, np.concatenate(exact_values)))
        return RoiInspectionSnapshot(
            key, stats_by_roi, comparison_histograms(hist_inputs, HistogramSpec(bins=96))
        )

    def _tile_data_provider(self):
        if not hasattr(self, "operation_evaluator"):
            return None
        evaluation_context = None
        if hasattr(self, "_evaluation_context"):
            evaluation_context = self._evaluation_context("roi")
        return TileDataProvider(
            operation_evaluator=self.operation_evaluator,
            document=self.document,
            committed_frame=self._current_committed_display_frame(),
            montage_plan=getattr(self, "_current_montage_plan", None),
            colormap_lut=self._roi_colormap_lut(),
            evaluation_context=evaluation_context,
        )

    def _roi_colormap_lut(self):
        try:
            if (
                getattr(
                    getattr(self.view_state, "channel", None),
                    "value",
                    getattr(self.view_state, "channel", None),
                )
                == "phase"
            ):
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

    def _set_inspection_dock_visible_from_user(self, visible):
        self.layout_manager.set_inspection_dock_visible_from_user(visible)

    def _hud_context_rows(self):
        """Extra hover-HUD rows for the ROI or profile marker under the cursor."""
        view = getattr(self, "img_view", None)
        if view is None:
            return ()
        state = view.interaction_controller.state
        target = state.capture or state.hover
        if target is None:
            return ()
        rows = []
        if target.kind == "roi" and target.object_id:
            roi_id = str(target.object_id)
            selection = self.roi_store.get(roi_id)
            if selection is not None:
                kind = selection.geometry.kind.value.replace("_", " ")
                rows.append(("crop", f"{selection.label} · {kind}"))
                payload = getattr(self, "_hud_stats_by_roi", {}).get(roi_id)
                if payload is not None:
                    _selection, stats = payload
                    mean = getattr(stats, "mean", None)
                    if mean is not None and np.isfinite(mean):
                        rows.append(("functions", f"mean {mean:.4g} · n {stats.finite_count}"))
                    minimum = getattr(stats, "minimum", None)
                    maximum = getattr(stats, "maximum", None)
                    if minimum is not None and maximum is not None:
                        rows.append(("analytics", f"min {minimum:.4g} · max {maximum:.4g}"))
        elif target.kind == "profile":
            position = view.profileMarkerPosition()
            axes_text = ", ".join(
                f"d{axis}" for axis in tuple(getattr(self, "profile_axes", ()) or ())
            )
            if position is not None:
                rows.append(
                    ("show_chart", f"Profile {axes_text} @ ({position[0]:.0f}, {position[1]:.0f})")
                )
            else:
                rows.append(("show_chart", f"Profile {axes_text}"))
        return tuple(rows)

    def _roi_at_image_point(self, image_point):
        if image_point is None or not hasattr(self, "img_view"):
            return None
        try:
            tolerance = self.img_view._pointer_interaction._hit_tolerance()
        except Exception:
            tolerance = 3.0
        try:
            candidates = self.img_view.roiHitCandidates(
                (float(image_point[0]), float(image_point[1])), tolerance=float(tolerance)
            )
        except Exception:
            return None
        return candidates[-1] if candidates else None

    def _update_roi_selection(self, roi_id, **changes):
        import dataclasses

        roi_id = str(roi_id)
        selections = tuple(
            dataclasses.replace(selection, **changes) if str(selection.id) == roi_id else selection
            for selection in self.img_view.roiSelections()
        )
        self.img_view.setRoiSelections(selections, selected_id=roi_id)
        self.roi_store = self.roi_store.replace_all(self.img_view.roiSelections()).select(roi_id)
        self._refresh_inspection_dock()
        self._notify_sync("rois")

    def _rename_roi(self, roi_id, global_pos=None):
        from pyqtgraph.Qt import QtGui

        from arrayscope.ui.bubbles import LineEditBubble

        selection = self.roi_store.get(str(roi_id))
        if selection is None:
            return
        bubble = LineEditBubble(
            self,
            icon_name="edit",
            initial=selection.label,
            on_accept=lambda text, roi_id=str(roi_id): self._update_roi_selection(
                roi_id, label=text
            ),
        )
        bubble.open_at(global_pos or QtGui.QCursor.pos(), focus_widget=bubble.edit)

    def _change_roi_color(self, roi_id, global_pos=None):
        from pyqtgraph.Qt import QtGui

        from arrayscope.core.roi_store import DEFAULT_ROI_COLORS
        from arrayscope.ui.bubbles import ColorSwatchBubble

        selection = self.roi_store.get(str(roi_id))
        if selection is None:
            return
        bubble = ColorSwatchBubble(
            self,
            colors=DEFAULT_ROI_COLORS,
            current=selection.color,
            on_accept=lambda color, roi_id=str(roi_id): self._update_roi_selection(
                roi_id, color=color
            ),
        )
        bubble.open_at(global_pos or QtGui.QCursor.pos())

    def _show_image_context_menu(self, global_pos, image_point=None):
        from arrayscope.ui.icons import material_icon

        menu = QtWidgets.QMenu(self)
        roi_selection = self._roi_at_image_point(image_point)
        if roi_selection is not None:
            roi_id = str(roi_selection.id)
            header = menu.addAction(f"{roi_selection.label}")
            header.setEnabled(False)
            rename_action = menu.addAction(material_icon("edit"), "Rename ROI…")
            rename_action.triggered.connect(lambda _checked=False: self._rename_roi(roi_id))
            color_action = menu.addAction(material_icon("colorize"), "Change color…")
            color_action.triggered.connect(lambda _checked=False: self._change_roi_color(roi_id))
            delete_action = menu.addAction(material_icon("delete"), "Delete ROI")
            delete_action.triggered.connect(lambda _checked=False: self._delete_roi(roi_id))
            menu.addSeparator()
        # Truthful checked state: the button can be stale-checked while the
        # marker is gone (e.g. cleared by a clamp failure); showing that as
        # active forced a click to "disable" before one could enable.
        live_button = self.widgets["buttons"]["display"]["live_profile"]
        marker_visible = (
            hasattr(self, "img_view") and self.img_view.profileMarkerPosition() is not None
        )
        live = live_button.isChecked() and marker_visible
        profile_action = menu.addAction(material_icon("show_chart"), "Live profile")
        profile_action.setCheckable(True)
        profile_action.setChecked(live)
        profile_action.triggered.connect(
            lambda checked=False: self._set_live_profile_from_context(bool(checked), image_point)
        )
        menu.addSeparator()
        for label, icon_name, tool in (
            ("Add line ROI", "show_chart", "roi_line"),
            ("Add rectangle ROI", "crop", "roi_rectangle"),
            ("Draw polyline ROI", "waves", "roi_polyline"),
            ("Draw freehand ROI", "edit", "roi_freehand"),
        ):
            action = menu.addAction(material_icon(icon_name), label)
            action.triggered.connect(
                lambda checked=False, tool=tool, image_point=image_point: self._add_roi_for_tool_at(
                    tool, image_point
                )
            )
        menu.addSeparator()
        # Save-viewport entry: clicking the parent saves the default flavor
        # (viewport with overlays); hovering expands the specific options.
        save_menu = menu.addMenu(material_icon("save"), "Save viewport")
        default_action = save_menu.addAction(material_icon("save"), "Viewport with overlays")
        default_font = default_action.font()
        default_font.setBold(True)
        default_action.setFont(default_font)
        default_action.triggered.connect(
            lambda _checked=False: self._save_viewport_image("viewport-with-overlays")
        )
        without_action = save_menu.addAction(material_icon("crop"), "Viewport without overlays")
        without_action.triggered.connect(
            lambda _checked=False: self._save_viewport_image("viewport-without-overlays")
        )
        full_action = save_menu.addAction(material_icon("data_array"), "Full content")
        full_action.triggered.connect(
            lambda _checked=False: self._save_viewport_image("full-content")
        )
        save_menu.menuAction().triggered.connect(
            lambda _checked=False: self._save_viewport_image("viewport-with-overlays")
        )
        menu.addSeparator()
        show_inspection = menu.addAction(material_icon("analytics"), "Show inspection dock")
        show_inspection.triggered.connect(self._show_inspection_dock)
        clear_rois = menu.addAction(material_icon("delete_sweep"), "Clear ROIs")
        clear_rois.setEnabled(hasattr(self, "img_view") and bool(self.img_view.roiSelections()))
        clear_rois.triggered.connect(self._clear_rois)
        menu.exec(global_pos)

    def _save_viewport_image(self, flavor: str) -> None:
        """Export the current view as a PNG.

        - viewport-with-overlays: what you see (ROIs, crosshair, chips).
        - viewport-without-overlays: same crop, overlays hidden.
        - full-content: the displayed image at native resolution with the
          current LUT/levels applied.
        """
        from arrayscope.ui.file_dialogs import get_save_file_name

        file_path, _ = get_save_file_name(
            self,
            "Save viewport image",
            f"arrayscope-{flavor}.png",
            "PNG image (*.png)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".png"):
            file_path += ".png"
        if flavor == "full-content":
            image = self._full_content_qimage()
            saved = image is not None and not image.isNull() and image.save(file_path, "PNG")
        else:
            pixmap = self._grab_viewport_pixmap(include_overlays=flavor == "viewport-with-overlays")
            saved = pixmap is not None and not pixmap.isNull() and pixmap.save(file_path, "PNG")
        if saved:
            show_status_message(self, f"Saved {file_path}", timeout=3500)
        else:
            show_status_message(self, "Could not save viewport image.", timeout=3500)

    def _grab_viewport_pixmap(self, *, include_overlays: bool):
        view = getattr(self, "img_view", None)
        if view is None:
            return None
        hidden = []
        if not include_overlays:
            candidates = [
                getattr(view, "_hud_widget", None),
                getattr(view, "_evaluation_overlay", None),
                getattr(view, "_roi_info_panel", None),
                getattr(view, "_profile_vline", None),
                getattr(view, "_profile_hline", None),
                getattr(view, "_profile_handle", None),
            ]
            for item, _selection in getattr(view, "_roi_items", {}).values():
                candidates.append(item)
            for candidate in candidates:
                if candidate is not None and candidate.isVisible():
                    candidate.setVisible(False)
                    hidden.append(candidate)
        try:
            return view.graphicsView.grab()
        finally:
            for candidate in hidden:
                candidate.setVisible(True)

    def _full_content_qimage(self):
        view = getattr(self, "img_view", None)
        image_item = getattr(view, "imageItem", None)
        if image_item is None:
            return None
        try:
            image_item.render()
            qimage = getattr(image_item, "qimage", None)
            if qimage is not None:
                return qimage.copy()
        except Exception:
            pass
        # Fallback: render the presented array with the active LUT/levels
        # (tiled presentations don't populate the base image item).
        try:
            import pyqtgraph as pg

            data = getattr(view, "image", None)
            if data is None:
                return None
            data = np.asarray(data)
            lut = view.displayColorMapLookupTable() if data.ndim == 2 else None
            levels = view.getLevels() if data.ndim == 2 else None
            argb, alpha = pg.functions.makeARGB(data, lut=lut, levels=levels)
            return pg.functions.makeQImage(argb, alpha, transpose=False).copy()
        except Exception:
            return None

    def _set_live_profile_from_context(self, enabled, image_point=None):
        button = self.widgets["buttons"]["display"]["live_profile"]
        if bool(enabled) and button.isChecked():
            # setChecked would be a no-op on a stale-checked button; re-run
            # the activation path explicitly so one click always works.
            self._on_live_profile_toggled(True)
        else:
            button.setChecked(bool(enabled))
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
            return {
                "points": ((x - 10, y - 10), (x + 10, y - 10), (x + 10, y + 10), (x - 10, y + 10))
            }
        return {}

    def _update_roi_info_overlay(self, stats_by_roi):
        if not hasattr(self, "img_view"):
            return
        # Cached for the hover HUD (per-ROI rows on mouse-over).
        self._hud_stats_by_roi = dict(stats_by_roi)
        rows = []
        for _roi_id, (selection, stats) in list(stats_by_roi.items())[:6]:
            kind = selection.geometry.kind.value.replace("_", " ")
            mean = (
                "" if stats.mean is None or not np.isfinite(stats.mean) else f"µ={stats.mean:.4g}"
            )
            rows.append((selection.label, kind, f"n={stats.finite_count}", mean))
        self.img_view.setRoiInfoRows(rows)

    def _refresh_hidden_roi_overlay_from_committed_frame(self) -> None:
        if not hasattr(self, "img_view") or not hasattr(self, "inspection_dock"):
            return
        if self._inspection_panel_is_visible():
            return
        selections_fn = getattr(self.img_view, "roiSelections", None)
        if not callable(selections_fn):
            return
        selections = tuple(selections_fn())
        if not any(selection.enabled for selection in selections):
            self._update_roi_info_overlay(OrderedDict())
            return
        if self._roi_uses_montage_demand(selections) and self._montage_roi_values_pending():
            self._inspection_stale = True
            return
        self._schedule_refresh_inspection_dock("display-commit")

    def _hidden_roi_statistics(self, selections):
        tiled = self._committed_tiled_roi_values(selections, collect_histograms=False)
        if tiled is not None:
            return tiled[0]
        stats_by_roi = OrderedDict()
        return stats_by_roi

    def _committed_tiled_roi_values(self, selections, *, collect_histograms: bool):
        frame = self._committed_tiled_frame()
        if frame is None:
            return None
        value_source = getattr(frame, "value_source", None)
        scene = getattr(frame, "scene", None)
        regions = tuple(getattr(scene, "regions", ()) or ())
        if value_source is None or not regions:
            return None

        stats_by_roi = OrderedDict()
        hist_inputs = []
        for selection in selections:
            if not selection.enabled:
                continue
            accumulator = RoiStatsAccumulator()
            exact_values = []
            for region, local_region, offset in self._roi_scene_regions(
                selection.geometry, regions
            ):
                committed = value_source.tile_region(
                    SimpleNamespace(
                        region_id=int(region.region_id), tile_number=int(region.region_id)
                    ),
                    local_region,
                )
                if committed is None:
                    continue
                image, histogram_data, _source = committed
                source = histogram_data if histogram_data is not None else image
                values = roi_values_for_region(source, selection.geometry, offset=offset)
                accumulator.add_values(values)
                if collect_histograms:
                    finite = np.asarray(values).ravel()
                    finite = finite[np.isfinite(finite)]
                    if (
                        finite.size
                        and sum(value.size for value in exact_values) + finite.size <= 250_000
                    ):
                        exact_values.append(finite.copy())
            stats = accumulator.result()
            stats_by_roi[selection.id] = (selection, stats)
            if collect_histograms and exact_values:
                hist_inputs.append((selection.label, np.concatenate(exact_values)))
        return stats_by_roi, tuple(hist_inputs)

    def _roi_scene_regions(self, geometry, regions):
        bounds = roi_bounding_rect(geometry)
        if bounds is None:
            return ()
        x0, y0, x1, y1 = bounds
        selected = []
        for region in regions:
            if not getattr(region, "resident", False):
                continue
            rx0, ry0, rx1, ry1 = tuple(float(value) for value in region.bounds)
            if rx1 < x0 or rx0 > x1 or ry1 < y0 or ry0 > y1:
                continue
            if geometry.kind == RoiKind.RECTANGLE:
                sx0 = max(rx0, float(np.floor(x0)))
                sx1 = min(rx1 + 1.0, float(np.ceil(x1)))
                sy0 = max(ry0, float(np.floor(y0)))
                sy1 = min(ry1 + 1.0, float(np.ceil(y1)))
                if sx1 <= sx0 or sy1 <= sy0:
                    continue
                local_region = (
                    slice(int(sy0 - ry0), int(sy1 - ry0)),
                    slice(int(sx0 - rx0), int(sx1 - rx0)),
                )
                offset = (sx0, sy0)
            else:
                local_region = (slice(0, int(region.height)), slice(0, int(region.width)))
                offset = (rx0, ry0)
            selected.append((region, local_region, offset))
        return tuple(selected)
