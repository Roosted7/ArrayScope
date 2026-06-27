"""Idle prefetch orchestration for ArrayScope windows."""

from __future__ import annotations

import pyqtgraph.Qt as Qt

from arrayscope.core.compute_policy import ComputeLane
from arrayscope.core.scheduler import FrameTarget
from arrayscope.core.work_graph import WorkItem, WorkLane
from arrayscope.operations.cost import estimate_pipeline_cost
from arrayscope.operations.evaluator import stage_document_key
from arrayscope.operations.render_plan import MAX_IDLE_PREFETCH_SLICES
from arrayscope.operations.slabs import plan_slab, request_for_image


class RenderPrefetchMixin:
    def _schedule_prefetch_nearby_slices(self, view_state, colormap_lut):
        if not getattr(self.app_settings, "prefetch_nearby_slices", False):
            self.operation_evaluator.note_prefetch_skipped()
            return
        self._pending_prefetch_request = (view_state, colormap_lut)
        revision = int(getattr(self, "_prefetch_dispatch_revision", 0) or 0) + 1
        self._prefetch_dispatch_revision = revision
        if bool(getattr(self, "_prefetch_dispatch_queued", False)):
            return
        self._prefetch_dispatch_queued = True
        # Qt event-turn barrier. The queued callback collapses rapid slice
        # changes to the latest request; WorkGraph/resource gates below decide
        # whether speculation is actually allowed.
        Qt.QtCore.QTimer.singleShot(0, lambda revision=revision: self._run_pending_prefetch(revision))

    def _run_pending_prefetch(self, revision: int | None = None):
        self._prefetch_dispatch_queued = False
        if revision is not None and int(revision) != int(getattr(self, "_prefetch_dispatch_revision", 0) or 0):
            if getattr(self, "_pending_prefetch_request", None) is not None:
                self._prefetch_dispatch_queued = True
                latest = int(getattr(self, "_prefetch_dispatch_revision", 0) or 0)
                Qt.QtCore.QTimer.singleShot(0, lambda revision=latest: self._run_pending_prefetch(revision))
            return
        request = getattr(self, "_pending_prefetch_request", None)
        self._pending_prefetch_request = None
        if request is None:
            return
        view_state, colormap_lut = request
        self._refresh_memory_policy(active_render=self.visible_evaluation_controller.is_busy())
        if self.visible_evaluation_controller.is_busy():
            self.prefetch_evaluation_controller.start_prefetch(lambda: None, blocked_reason="visible_busy")
            self.operation_evaluator.note_prefetch_skipped()
            return
        self._prefetch_nearby_slices(view_state, colormap_lut)

    def _prefetch_nearby_slices(self, view_state, colormap_lut):
        if not getattr(self.app_settings, "prefetch_nearby_slices", False):
            self.operation_evaluator.note_prefetch_skipped()
            return
        if view_state.montage_axis is not None:
            self.operation_evaluator.note_prefetch_skipped()
            return
        if self.visible_evaluation_controller.is_busy():
            self.prefetch_evaluation_controller.start_prefetch(lambda: None, blocked_reason="visible_busy")
            self.operation_evaluator.note_prefetch_skipped()
            return
        policy = self._memory_policy()
        if self.operation_evaluator._display_cache.bytes_used > int(self.operation_evaluator._display_cache.max_bytes * policy.cache_prefetch_skip_fraction):
            self.operation_evaluator.note_prefetch_skipped()
            return
        if self._estimated_image_display_bytes(view_state) > policy.prefetch_budget_bytes:
            self.operation_evaluator.note_prefetch_skipped()
            return
        if not self._prefetch_cost_allowed(view_state):
            self.prefetch_evaluation_controller.start_prefetch(lambda: None, blocked_reason="cost")
            self.operation_evaluator.note_prefetch_skipped()
            return
        axis = getattr(self, "_active_slice_axis", None)
        if axis is None or view_state.image_axes is None or axis in view_state.image_axes:
            return
        document = self.document
        size = view_state.shape[axis]
        current = view_state.slice_indices[axis]
        last = getattr(self, "_last_prefetch_slice_index", None)
        direction = 0 if last is None else (1 if current >= last else -1)
        self._last_prefetch_slice_index = current
        deltas = self._prefetch_deltas(direction, max_radius=min(2, max(1, size - 1)))
        scheduled = 0
        for delta in deltas:
            limit = MAX_IDLE_PREFETCH_SLICES if direction != 0 else 1
            if scheduled >= limit:
                break
            index = current + delta
            if 0 <= index < size:
                prefetch_state = view_state.with_slice(axis, index)
                prefetch_key = self.operation_evaluator.display_tile_key(
                    prefetch_state,
                    colormap_lut=colormap_lut,
                    document=document,
                )
                started = self.prefetch_evaluation_controller.start_prefetch(
                    lambda prefetch_state=prefetch_state, document=document: self.operation_evaluator.prefetch_display_tile_snapshot(
                        document,
                        prefetch_state,
                        colormap_lut=colormap_lut,
                        evaluation_context=self._evaluation_context(ComputeLane.PREFETCH, None),
                    ),
                    on_done=lambda result, prefetch_state=prefetch_state, document=document, prefetch_key=prefetch_key: self._store_prefetch_display_tile_if_current(
                        document,
                        prefetch_key,
                        prefetch_state,
                        colormap_lut,
                        result,
                    ),
                    key=prefetch_key,
                    memory_budget_bytes=policy.prefetch_budget_bytes,
                    work_item=WorkItem(
                        key=("prefetch_display_tile", prefetch_key),
                        lane=WorkLane.SPECULATIVE_RESIDENCY,
                        frame_target=FrameTarget(
                            semantic_key=prefetch_key,
                            viewport_key=("slice", axis, index),
                            presentation_key=("prefetch",),
                            quality="retained",
                        ),
                        supersession_key=("prefetch-image", axis, index),
                        supersession_value=prefetch_key,
                        estimated_bytes=int(policy.prefetch_budget_bytes),
                        expected_value=1.0,
                        reusable_output=True,
                    ),
                )
                self._note_prefetch_start(started)
                if started.scheduled:
                    scheduled += 1

    def _prefetch_cost_allowed(self, view_state):
        operations = tuple(self.document.enabled_operations)
        if not operations:
            return True
        image_axes = set(view_state.image_axes or ())
        for operation in operations:
            if type(operation).__name__ in {"Mean", "Sum", "Maximum", "Minimum", "RootSumSquares"} and int(operation.axis) in image_axes:
                return False
        cost = estimate_pipeline_cost(self.base_data.shape, getattr(self.base_data, "dtype", None), operations)
        peak = cost.estimated_peak_bytes or 0
        policy = self._memory_policy()
        if peak > policy.operation_prefetch_peak_budget_bytes:
            return False
        has_fft = any(type(operation).__name__ in {"CenteredFFT", "CenteredIFFT"} for operation in operations)
        if has_fft and peak > policy.fft_prefetch_peak_budget_bytes and not self._stage_cached_or_in_flight_for_prefetch(view_state):
            return False
        return True

    def _stage_cached_or_in_flight_for_prefetch(self, view_state) -> bool:
        try:
            request = request_for_image(view_state)
            plan = plan_slab(self.document, request)
        except Exception:
            return False
        candidates = tuple(candidate for candidate in getattr(plan.region_plan, "cache_candidates", ()) if getattr(candidate, "retain", True))
        if not candidates:
            return False
        candidate = candidates[-1]
        if candidate.estimated_nbytes is not None and int(candidate.estimated_nbytes) > int(self._memory_policy().stage_cache_budget_bytes):
            return False
        key = self.operation_evaluator.stage_materializer.key_for_candidate(stage_document_key(self.document), candidate)
        cache = self.operation_evaluator.stage_cache
        if (cache.get_containing(key) if hasattr(cache, "get_containing") else cache.get(key)) is not None:
            return True
        return key in getattr(self.operation_evaluator.stage_materializer, "_in_flight", {})

    def _prefetch_deltas(self, direction, *, max_radius):
        radii = range(1, int(max_radius) + 1)
        if direction > 0:
            return tuple(delta for radius in radii for delta in (radius, -radius))
        if direction < 0:
            return tuple(delta for radius in radii for delta in (-radius, radius))
        return tuple(delta for radius in radii for delta in (-radius, radius))

    def _prefetch_profiles_near_marker(self, view_state, image_x, image_y, *, line_axis=None):
        if view_state.image_axes is None or line_axis is None:
            return
        document = self.document
        primary_axis, secondary_axis = view_state.image_axes
        cx = int(round(image_x))
        cy = int(round(image_y))
        max_radius = 4
        scheduled = 0
        request_key_cache = {}
        for radius in range(0, max_radius + 1):
            points = []
            if radius == 0:
                points.append((cx, cy))
            else:
                for dx in (-radius, radius):
                    points.append((cx + dx, cy))
                for dy in (-radius, radius):
                    points.append((cx, cy + dy))
            for x, y in points:
                if scheduled >= 24:
                    return
                if not (0 <= x < view_state.shape[secondary_axis] and 0 <= y < view_state.shape[primary_axis]):
                    continue
                profile_state = self.profile_coordinator.state_from_marker(view_state, x, y, line_axis=line_axis)
                if profile_state is None:
                    continue
                request_key_cache[profile_state] = self.operation_evaluator.line_key(profile_state, document=document)
                started = self.prefetch_evaluation_controller.start_prefetch(
                    lambda profile_state=profile_state, document=document: self.operation_evaluator.prefetch_line_snapshot(
                        document,
                        profile_state,
                        evaluation_context=self._evaluation_context(ComputeLane.PREFETCH, None),
                    ),
                    on_done=lambda result, profile_state=profile_state, document=document, key=request_key_cache[profile_state]: self._store_prefetch_profile_if_current(
                        document,
                        key,
                        profile_state,
                        result,
                    ),
                    key=request_key_cache[profile_state],
                    memory_budget_bytes=self._prefetch_budget_bytes(),
                    work_item=WorkItem(
                        key=("prefetch_profile", request_key_cache[profile_state]),
                        lane=WorkLane.PROFILE_ROI_HOVER,
                        frame_target=FrameTarget(
                            semantic_key=request_key_cache[profile_state],
                            viewport_key=("marker", int(x), int(y)),
                            presentation_key=("prefetch-profile",),
                            quality="retained",
                        ),
                        supersession_key=("prefetch-profile", profile_state),
                        supersession_value=request_key_cache[profile_state],
                        estimated_bytes=int(self._prefetch_budget_bytes()),
                        expected_value=0.5,
                        reusable_output=True,
                    ),
                )
                self._note_prefetch_start(started)
                if started.scheduled:
                    scheduled += 1

    def _store_prefetch_profile_if_current(self, document, request_key, profile_state, result):
        if request_key != self.operation_evaluator.line_key(profile_state):
            self.operation_evaluator.note_prefetch_stale()
            return False
        return self.operation_evaluator.store_prefetch_line_result(document, profile_state, result)

    def _store_prefetch_display_tile_if_current(self, document, request_key, view_state, colormap_lut, result):
        current_key = self.operation_evaluator.display_tile_key(view_state, colormap_lut=colormap_lut)
        if request_key != current_key:
            self.operation_evaluator.note_prefetch_stale()
            return False
        return self.operation_evaluator.store_prefetch_display_tile_result(document, view_state, colormap_lut, result)

    def _note_prefetch_start(self, started):
        if started.scheduled:
            self.operation_evaluator.note_prefetch_scheduled()
        elif started.reason == "deduped":
            self.operation_evaluator.note_prefetch_deduped()
        elif started.reason == "limited":
            self.operation_evaluator.note_prefetch_limited()
