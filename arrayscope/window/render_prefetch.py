"""Idle prefetch orchestration for ArrayScope windows."""

from __future__ import annotations

from time import monotonic

from arrayscope.core.compute_policy import ComputeLane
from arrayscope.core.frame_targets import FrameTarget
from arrayscope.core.prefetch_policy import SliceScrubMomentum
from arrayscope.kernel import Lane as WorkLane
from arrayscope.kernel import Priority, WorkItem
from arrayscope.operations.cost import estimate_pipeline_cost
from arrayscope.operations.evaluator import stage_document_key
from arrayscope.operations.slabs import plan_slab, request_for_image


class RenderPrefetchMixin:
    def _schedule_prefetch_nearby_slices(self, view_state, colormap_lut):
        if not getattr(self.win.app_settings, "prefetch_nearby_slices", False):
            self.win.operation_evaluator.note_prefetch_skipped()
            return
        # Direction data accrues at SCHEDULE time: during a fast scrub the
        # dispatch below is gated (visible busy) on almost every step, and
        # that is exactly when the momentum model needs the step stream so
        # the first ungated run speculates in the right direction.
        self._observe_prefetch_momentum(view_state)
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
        if request is None:
            return
        view_state, colormap_lut = request
        self._refresh_memory_policy(active_render=self.win.visible_evaluation_controller.is_busy())
        if self.win.visible_evaluation_controller.is_busy():
            # The speculative dispatch lands milliseconds after the slice
            # change that armed it — while that same change's visible eval is
            # still in flight, so this gate closes on ~every scrub step.
            # Dropping the request here left the prefetcher deterministically
            # dead. Keep it pending (latest-wins: newer slice changes simply
            # overwrite it) and retry when the visible work drains.
            self.win.prefetch_evaluation_controller.start_prefetch(
                lambda: None, blocked_reason="visible_busy"
            )
            self.win.operation_evaluator.note_prefetch_skipped()
            self._arm_prefetch_visible_drain_retry()
            return
        self._pending_prefetch_request = None
        self._prefetch_nearby_slices(view_state, colormap_lut)

    def _arm_prefetch_visible_drain_retry(self) -> None:
        """Retry the retained prefetch request after the next completion drain.

        ADR 0053: no new timers. The kernel bridge's one-shot capacity waiter
        fires on the GUI thread right after a completion batch is dispatched —
        the visible evaluation that closed the gate is itself such a
        completion, so its drain is the wake-up. If the gate is still closed
        (another visible frame started), the retry re-arms; the waiter key
        keeps at most one continuation outstanding per window.
        """

        notify = getattr(self.win.visible_evaluation_controller, "notify_when_capacity", None)
        if not callable(notify):
            return
        notify(
            ("slice-prefetch-visible-drain", id(self)),
            self._retry_prefetch_after_visible_drain,
        )

    def _retry_prefetch_after_visible_drain(self) -> None:
        if getattr(self.win, "_closing", False):
            return
        if getattr(self, "_pending_prefetch_request", None) is None:
            return
        if bool(getattr(self, "_prefetch_dispatch_queued", False)):
            # A fresh speculative dispatch is already queued; it reads the
            # latest pending request itself.
            return
        self._run_pending_prefetch()

    def _observe_prefetch_momentum(self, view_state) -> None:
        axis = getattr(self.win, "_active_slice_axis", None)
        if axis is None or view_state.image_axes is None or axis in view_state.image_axes:
            return
        if not (0 <= int(axis) < len(view_state.slice_indices)):
            return
        momentum = getattr(self, "_prefetch_momentum", None)
        if momentum is None:
            momentum = SliceScrubMomentum()
            self._prefetch_momentum = momentum
        momentum.observe(int(view_state.slice_indices[int(axis)]), now=monotonic())

    def _observe_montage_prefetch_momentum(self, previous_state, view_state) -> None:
        """Observe an index-window shift for montage speculation ordering.

        Montage evaluation and GPU warming keep using the standing prefetch
        scheduler.  This owner records only the input direction so that the
        next idle admission can prefer already-eligible tiles ahead of the
        scroll without creating future-window lifecycle state.
        """

        axis = getattr(view_state, "montage_axis", None)
        indices = tuple(getattr(view_state, "montage_indices", ()) or ())
        previous_axis = getattr(previous_state, "montage_axis", None)
        previous_indices = tuple(getattr(previous_state, "montage_indices", ()) or ())
        if axis is None or not indices:
            return
        key = (int(axis), len(indices))
        if (
            previous_axis != axis
            or len(previous_indices) != len(indices)
            or getattr(self, "_montage_prefetch_momentum_key", None) != key
        ):
            self._montage_prefetch_momentum_key = key
            self._montage_prefetch_momentum = SliceScrubMomentum()
        momentum = getattr(self, "_montage_prefetch_momentum", None)
        if momentum is None:
            momentum = SliceScrubMomentum()
            self._montage_prefetch_momentum = momentum
        momentum.observe(int(indices[0]), now=monotonic())

    def _prefetch_nearby_slices(self, view_state, colormap_lut):
        if not getattr(self.win.app_settings, "prefetch_nearby_slices", False):
            self.win.operation_evaluator.note_prefetch_skipped()
            return
        if view_state.montage_axis is not None:
            self.win.operation_evaluator.note_prefetch_skipped()
            return
        if self.win.visible_evaluation_controller.is_busy():
            self.win.prefetch_evaluation_controller.start_prefetch(
                lambda: None, blocked_reason="visible_busy"
            )
            self.win.operation_evaluator.note_prefetch_skipped()
            if getattr(self, "_pending_prefetch_request", None) is None:
                self._pending_prefetch_request = (view_state, colormap_lut)
            self._arm_prefetch_visible_drain_retry()
            return
        policy = self._memory_policy()
        if self.win.operation_evaluator._display_cache.bytes_used > int(
            self.win.operation_evaluator._display_cache.max_bytes
            * policy.cache_prefetch_skip_fraction
        ):
            self.win.operation_evaluator.note_prefetch_skipped()
            return
        if self._estimated_image_display_bytes(view_state) > policy.prefetch_budget_bytes:
            self.win.operation_evaluator.note_prefetch_skipped()
            return
        if not self._prefetch_cost_allowed(view_state):
            self.win.prefetch_evaluation_controller.start_prefetch(
                lambda: None, blocked_reason="cost"
            )
            self.win.operation_evaluator.note_prefetch_skipped()
            return
        axis = getattr(self.win, "_active_slice_axis", None)
        if axis is None or view_state.image_axes is None or axis in view_state.image_axes:
            return
        document = self.win.document
        # Evaluate exactly what the visible flow would evaluate for this
        # backend: shader-display construction keeps scale/LUT out of the
        # data plane, so the CPU-cache key matches the visible key AND the
        # GPU warm texels (ADR 0055 G4c) are byte-compatible with the plane
        # the later visible commit uploads/reuses.
        shader_display = self._prefetch_shader_display()
        size = view_state.shape[axis]
        current = view_state.slice_indices[axis]
        # observe() happens at schedule time (_observe_prefetch_momentum);
        # this ungated run only reads the accumulated direction.
        momentum = getattr(self, "_prefetch_momentum", None)
        if momentum is None:
            momentum = SliceScrubMomentum()
            self._prefetch_momentum = momentum
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
                    shader_display=shader_display,
                )
                started = self.win.prefetch_evaluation_controller.start_prefetch(
                    lambda prefetch_state=prefetch_state, document=document, shader_display=shader_display: (
                        self.win.operation_evaluator.prefetch_display_tile_snapshot(
                            document,
                            prefetch_state,
                            colormap_lut=colormap_lut,
                            evaluation_context=self.win._evaluation_context(
                                ComputeLane.PREFETCH, None
                            ),
                            shader_display=shader_display,
                        )
                    ),
                    on_done=lambda result, prefetch_state=prefetch_state, document=document, prefetch_key=prefetch_key, shader_display=shader_display: (
                        self._store_prefetch_display_tile_if_current(
                            document,
                            prefetch_key,
                            prefetch_state,
                            colormap_lut,
                            result,
                            shader_display=shader_display,
                        )
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
            if (
                type(operation).__name__ in {"Mean", "Sum", "Maximum", "Minimum", "RootSumSquares"}
                and int(operation.axis) in image_axes
            ):
                return False
        cost = estimate_pipeline_cost(
            self.win.base_data.shape, getattr(self.win.base_data, "dtype", None), operations
        )
        peak = cost.estimated_peak_bytes or 0
        policy = self._memory_policy()
        if peak > policy.operation_prefetch_peak_budget_bytes:
            return False
        has_fft = any(
            type(operation).__name__ in {"CenteredFFT", "CenteredIFFT"} for operation in operations
        )
        return not (
            has_fft
            and peak > policy.fft_prefetch_peak_budget_bytes
            and not self._stage_cached_or_in_flight_for_prefetch(view_state)
        )

    def _stage_cached_or_in_flight_for_prefetch(self, view_state) -> bool:
        try:
            request = request_for_image(view_state)
            plan = plan_slab(self.win.document, request)
        except Exception:
            return False
        candidates = tuple(
            candidate
            for candidate in getattr(plan.region_plan, "cache_candidates", ())
            if getattr(candidate, "retain", True)
        )
        if not candidates:
            return False
        candidate = candidates[-1]
        if candidate.estimated_nbytes is not None and int(candidate.estimated_nbytes) > int(
            self._memory_policy().stage_cache_budget_bytes
        ):
            return False
        key = self.win.operation_evaluator.stage_materializer.key_for_candidate(
            stage_document_key(self.win.document), candidate
        )
        cache = self.win.operation_evaluator.stage_cache
        if (
            cache.get_containing(key) if hasattr(cache, "get_containing") else cache.get(key)
        ) is not None:
            return True
        return key in getattr(self.win.operation_evaluator.stage_materializer, "_in_flight", {})

    def _prefetch_profiles_near_marker(self, view_state, image_x, image_y, *, line_axis=None):
        if view_state.image_axes is None or line_axis is None:
            return
        document = self.win.document
        primary_axis, secondary_axis = view_state.image_axes
        cx = round(image_x)
        cy = round(image_y)
        max_radius = 4
        scheduled = 0
        request_key_cache = {}
        for radius in range(max_radius + 1):
            points = []
            if radius == 0:
                points.append((cx, cy))
            else:
                points.extend((cx + dx, cy) for dx in (-radius, radius))
                points.extend((cx, cy + dy) for dy in (-radius, radius))
            for x, y in points:
                if scheduled >= 24:
                    return
                if not (
                    0 <= x < view_state.shape[secondary_axis]
                    and 0 <= y < view_state.shape[primary_axis]
                ):
                    continue
                profile_state = self.win.profile_coordinator.state_from_marker(
                    view_state, x, y, line_axis=line_axis
                )
                if profile_state is None:
                    continue
                request_key_cache[profile_state] = self.win.operation_evaluator.line_key(
                    profile_state, document=document
                )
                started = self.win.prefetch_evaluation_controller.start_prefetch(
                    lambda profile_state=profile_state, document=document: (
                        self.win.operation_evaluator.prefetch_line_snapshot(
                            document,
                            profile_state,
                            evaluation_context=self.win._evaluation_context(
                                ComputeLane.PREFETCH, None
                            ),
                        )
                    ),
                    on_done=lambda result, profile_state=profile_state, document=document, key=request_key_cache[profile_state]: (
                        self._store_prefetch_profile_if_current(
                            document,
                            key,
                            profile_state,
                            result,
                        )
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
        return self.win.operation_evaluator.store_prefetch_line_result(
            document, profile_state, result
        )

    def _prefetch_shader_display(self) -> bool:
        """Whether this window's backend evaluates shader-display images."""

        from arrayscope.display.backend_contract import image_view_backend_capabilities

        return bool(
            getattr(
                image_view_backend_capabilities(self.win.img_view),
                "shader_windowing",
                False,
            )
        )

    def _store_prefetch_display_tile_if_current(
        self,
        document,
        request_key,
        view_state,
        colormap_lut,
        result,
        *,
        shader_display: bool = False,
    ):
        current_key = self.win.operation_evaluator.display_tile_key(
            view_state,
            colormap_lut=colormap_lut,
            shader_display=shader_display,
        )
        if request_key != current_key:
            self.win.operation_evaluator.note_prefetch_stale()
            return False
        stored = self.win.operation_evaluator.store_prefetch_display_tile_result(
            document,
            view_state,
            colormap_lut,
            result,
            shader_display=shader_display,
        )
        if stored:
            # ADR 0055 G4c: the plane is now CPU-cached; also push it into
            # GPU chunk residency so scroll-forward commits upload-free.
            self._warm_prefetched_plane_residency(document, view_state, result)
        return stored

    def _warm_prefetched_plane_residency(self, document, view_state, result) -> bool:
        """Warm a prefetched adjacent plane into GPU atlas chunk residency.

        Runs on the GUI thread (kernel-bridge callback dispatch) directly
        after the CPU display cache accepted the prefetch result — the
        adjacent-plane evaluation itself stayed on PREFETCH-lane workers.
        Currency: the display-tile key was just re-validated against the
        live document (stale documents never reach this), and the current
        window state must still be the same non-montage 2D view the plane
        was prefetched for; a superseded result is dropped, not warmed.
        Warm chunks are content-keyed, so even a wasted warm can never
        present wrong pixels. Bails silently on non-gpu_atlas backends and
        on pool capacity/budget denial.
        """

        if result is None or getattr(result, "value", None) is None:
            return False
        view = getattr(self.win, "img_view", None)
        warm = getattr(view, "warmPlaneResidency", None)
        if not callable(warm):
            return False
        from arrayscope.display.backend_contract import image_view_backend_capabilities

        capabilities = image_view_backend_capabilities(view)
        if getattr(capabilities, "tile_residency_kind", None) != "gpu_atlas":
            return False
        current_state = getattr(self.win, "view_state", None)
        if current_state is None or getattr(current_state, "montage_axis", None) is not None:
            return False
        if tuple(getattr(current_state, "image_axes", None) or ()) != tuple(
            getattr(view_state, "image_axes", None) or ()
        ):
            return False
        payload = self._prefetched_plane_payload(
            document,
            view_state,
            result,
            shader_display=bool(getattr(capabilities, "shader_windowing", False)),
            canonical_orientation=bool(getattr(capabilities, "display_axis_transpose", False)),
        )
        if payload is None:
            return False
        return bool(warm(payload))

    def _prefetched_plane_payload(
        self,
        document,
        view_state,
        result,
        *,
        shader_display: bool,
        canonical_orientation: bool,
    ):
        """Build the anchored exact payload of one prefetched plane.

        Mirrors the frame session's construction for the pieces that decide
        chunk identity (``_payload_source_anchor`` + ``_payload_chunk_plan``):
        the anchoring content key from ``source_anchoring_for_view``, the
        native source rect from the anchored starts plus the plane shape, the
        texture source/kind via ``texture_source_for_rendered``, and a native
        (factor-1, gutter-free) LOD identity. The visible commit later reads
        the SAME cached ``DisplayImage``, so warm texels are byte-identical.
        """

        import numpy as np

        value = result.value
        image_axes = getattr(view_state, "image_axes", None)
        if (
            getattr(view_state, "montage_axis", None) is not None
            or image_axes is None
            or len(tuple(image_axes)) != 2
        ):
            return None
        lod = getattr(value, "lod", None)
        if lod is not None and (
            int(getattr(lod, "factor", 1) or 1) != 1 or int(getattr(lod, "gutter", 0) or 0) != 0
        ):
            # Reduced/gutter previews never take the chunked path; warming
            # them would key residency the visible commit cannot reuse.
            return None
        image = np.asarray(getattr(value, "data", None))
        if image.ndim < 2:
            return None
        from types import SimpleNamespace

        from arrayscope.display.lod import LodInfo
        from arrayscope.display.model.frame import DisplayTilePayload, PayloadSourceAnchor
        from arrayscope.display.source_anchoring import source_anchoring_for_view
        from arrayscope.render.lod import texture_source_for_rendered

        anchoring = source_anchoring_for_view(
            document,
            view_state,
            canonical_orientation=canonical_orientation,
        )
        if anchoring is None:
            return None
        texture, _histogram, texture_kind = texture_source_for_rendered(
            SimpleNamespace(
                image=image,
                texture_kind=getattr(value, "texture_kind", None),
                semantic_data=getattr(value, "semantic_data", None),
                histogram_data=None,
            ),
            shader_display=shader_display,
        )
        height, width = (int(image.shape[0]), int(image.shape[1]))
        starts = tuple(getattr(anchoring, "source_starts_yx", (None, None)))
        y_start = int(starts[0] or 0)
        x_start = int(starts[1] or 0)
        source_axes = (
            tuple(sorted(int(axis) for axis in image_axes))
            if canonical_orientation
            else tuple(int(axis) for axis in image_axes)
        )
        plane_shape = tuple(int(view_state.shape[axis]) for axis in source_axes)
        source_anchor = PayloadSourceAnchor(
            content_key=anchoring.content_key,
            source_rect=(y_start, y_start + height, x_start, x_start + width),
            plane_shape=plane_shape,
        )
        texture_shape = (int(texture.shape[0]), int(texture.shape[1]))
        try:
            return DisplayTilePayload(
                tile_number=0,
                source_index=0,
                image=image,
                histogram_data=None,
                source_id=("prefetch-warm-plane", anchoring.content_key, source_anchor.source_rect),
                texture_data=texture,
                texture_kind=texture_kind,
                semantic_data=getattr(value, "semantic_data", None),
                shader_mapping=getattr(value, "shader_mapping", None),
                lod=LodInfo(
                    level=0,
                    factor=1,
                    source_shape=texture_shape,
                    texture_shape=texture_shape,
                    gutter=0,
                ),
                quality="exact",
                source_anchor=source_anchor,
            )
        except (TypeError, ValueError):
            return None

    def _note_prefetch_start(self, started):
        if started.scheduled:
            self.win.operation_evaluator.note_prefetch_scheduled()
        elif started.reason == "deduped":
            self.win.operation_evaluator.note_prefetch_deduped()
        elif started.reason == "limited":
            self.win.operation_evaluator.note_prefetch_limited()
