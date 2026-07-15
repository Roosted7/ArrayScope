"""Idle prefetch orchestration for ArrayScope windows."""

from __future__ import annotations

from time import monotonic

from arrayscope.core.compute_policy import ComputeLane
from arrayscope.core.prefetch_policy import SliceScrubMomentum
from arrayscope.core.frame_targets import FrameTarget
from arrayscope.kernel import Lane as WorkLane, Priority, WorkItem
from arrayscope.operations.cost import estimate_pipeline_cost
from arrayscope.operations.evaluator import stage_document_key
from arrayscope.operations.slabs import plan_slab, request_for_image


class RenderPrefetchMixin:
    def _schedule_prefetch_nearby_slices(self, view_state, colormap_lut):
        if not getattr(self.win.app_settings, "prefetch_nearby_slices", False):
            self.win.operation_evaluator.note_prefetch_skipped()
            return
        self._pending_prefetch_request = (view_state, colormap_lut)
        if bool(getattr(self, "_prefetch_dispatch_queued", False)):
            return
        self._prefetch_dispatch_queued = True
        # Speculative kernel admission replaces the old event-turn timer. The
        # callback reads the latest pending request, so rapid slice changes
        # still collapse while visible work and quotas decide when it may run.
        kernel = getattr(self.win, "kernel", None)
        if kernel is None:
            self._run_pending_prefetch()
            return
        handle = kernel.submit_speculative_batch(
            kind="slice-prefetch-dispatch",
            scope="slice-prefetch",
            generation=id(self),
            key=("slice-prefetch-dispatch", id(self)),
            fn=lambda: True,
            on_done=lambda _value=None: self._run_pending_prefetch(),
            on_stale=lambda: setattr(self, "_prefetch_dispatch_queued", False),
            lane=WorkLane.SPECULATIVE_RESIDENCY,
            priority=Priority.PREFETCH,
            max_items=1,
        )
        if handle is None:
            self._prefetch_dispatch_queued = False

    def _run_pending_prefetch(self):
        self._prefetch_dispatch_queued = False
        request = getattr(self, "_pending_prefetch_request", None)
        self._pending_prefetch_request = None
        if request is None:
            return
        view_state, colormap_lut = request
        self._refresh_memory_policy(active_render=self.win.visible_evaluation_controller.is_busy())
        if self.win.visible_evaluation_controller.is_busy():
            self.win.prefetch_evaluation_controller.start_prefetch(lambda: None, blocked_reason="visible_busy")
            self.win.operation_evaluator.note_prefetch_skipped()
            return
        self._prefetch_nearby_slices(view_state, colormap_lut)

    def _prefetch_nearby_slices(self, view_state, colormap_lut):
        if not getattr(self.win.app_settings, "prefetch_nearby_slices", False):
            self.win.operation_evaluator.note_prefetch_skipped()
            return
        if view_state.montage_axis is not None:
            self.win.operation_evaluator.note_prefetch_skipped()
            return
        if self.win.visible_evaluation_controller.is_busy():
            self.win.prefetch_evaluation_controller.start_prefetch(lambda: None, blocked_reason="visible_busy")
            self.win.operation_evaluator.note_prefetch_skipped()
            return
        policy = self._memory_policy()
        if self.win.operation_evaluator._display_cache.bytes_used > int(self.win.operation_evaluator._display_cache.max_bytes * policy.cache_prefetch_skip_fraction):
            self.win.operation_evaluator.note_prefetch_skipped()
            return
        if self._estimated_image_display_bytes(view_state) > policy.prefetch_budget_bytes:
            self.win.operation_evaluator.note_prefetch_skipped()
            return
        if not self._prefetch_cost_allowed(view_state):
            self.win.prefetch_evaluation_controller.start_prefetch(lambda: None, blocked_reason="cost")
            self.win.operation_evaluator.note_prefetch_skipped()
            return
        axis = getattr(self.win, "_active_slice_axis", None)
        if axis is None or view_state.image_axes is None or axis in view_state.image_axes:
            return
        document = self.win.document
        size = view_state.shape[axis]
        current = view_state.slice_indices[axis]
        momentum = getattr(self, "_prefetch_momentum", None)
        if momentum is None:
            momentum = SliceScrubMomentum()
            self._prefetch_momentum = momentum
        momentum.observe(current, now=monotonic())
        plan = momentum.plan(size=size)
        scheduled = 0
        for delta in plan.deltas:
            if scheduled >= plan.depth:
                break
            index = current + delta
            if 0 <= index < size:
                prefetch_state = view_state.with_slice(axis, index)
                prefetch_key = self.win.operation_evaluator.display_tile_key(
                    prefetch_state,
                    colormap_lut=colormap_lut,
                    document=document,
                )
                started = self.win.prefetch_evaluation_controller.start_prefetch(
                    lambda prefetch_state=prefetch_state, document=document: self.win.operation_evaluator.prefetch_display_tile_snapshot(
                        document,
                        prefetch_state,
                        colormap_lut=colormap_lut,
                        evaluation_context=self.win._evaluation_context(ComputeLane.PREFETCH, None),
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
        operations = tuple(self.win.document.enabled_operations)
        if not operations:
            return True
        image_axes = set(view_state.image_axes or ())
        for operation in operations:
            if type(operation).__name__ in {"Mean", "Sum", "Maximum", "Minimum", "RootSumSquares"} and int(operation.axis) in image_axes:
                return False
        cost = estimate_pipeline_cost(self.win.base_data.shape, getattr(self.win.base_data, "dtype", None), operations)
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
            plan = plan_slab(self.win.document, request)
        except Exception:
            return False
        candidates = tuple(candidate for candidate in getattr(plan.region_plan, "cache_candidates", ()) if getattr(candidate, "retain", True))
        if not candidates:
            return False
        candidate = candidates[-1]
        if candidate.estimated_nbytes is not None and int(candidate.estimated_nbytes) > int(self._memory_policy().stage_cache_budget_bytes):
            return False
        key = self.win.operation_evaluator.stage_materializer.key_for_candidate(stage_document_key(self.win.document), candidate)
        cache = self.win.operation_evaluator.stage_cache
        if (cache.get_containing(key) if hasattr(cache, "get_containing") else cache.get(key)) is not None:
            return True
        return key in getattr(self.win.operation_evaluator.stage_materializer, "_in_flight", {})

    def _prefetch_profiles_near_marker(self, view_state, image_x, image_y, *, line_axis=None):
        if view_state.image_axes is None or line_axis is None:
            return
        document = self.win.document
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
                profile_state = self.win.profile_coordinator.state_from_marker(view_state, x, y, line_axis=line_axis)
                if profile_state is None:
                    continue
                request_key_cache[profile_state] = self.win.operation_evaluator.line_key(profile_state, document=document)
                started = self.win.prefetch_evaluation_controller.start_prefetch(
                    lambda profile_state=profile_state, document=document: self.win.operation_evaluator.prefetch_line_snapshot(
                        document,
                        profile_state,
                        evaluation_context=self.win._evaluation_context(ComputeLane.PREFETCH, None),
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
        if request_key != self.win.operation_evaluator.line_key(profile_state):
            self.win.operation_evaluator.note_prefetch_stale()
            return False
        return self.win.operation_evaluator.store_prefetch_line_result(document, profile_state, result)

    def _store_prefetch_display_tile_if_current(self, document, request_key, view_state, colormap_lut, result):
        current_key = self.win.operation_evaluator.display_tile_key(view_state, colormap_lut=colormap_lut)
        if request_key != current_key:
            self.win.operation_evaluator.note_prefetch_stale()
            return False
        return self.win.operation_evaluator.store_prefetch_display_tile_result(document, view_state, colormap_lut, result)

    def _note_prefetch_start(self, started):
        if started.scheduled:
            self.win.operation_evaluator.note_prefetch_scheduled()
        elif started.reason == "deduped":
            self.win.operation_evaluator.note_prefetch_deduped()
        elif started.reason == "limited":
            self.win.operation_evaluator.note_prefetch_limited()
