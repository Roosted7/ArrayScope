"""Frame render orchestration for ArrayScope windows.

The frame path owns the single visible image surface for every image view.  A
single slice and a multi-region montage both become one semantic frame with
region payloads; backends differ only in physical presentation mechanics.
"""

from __future__ import annotations


from dataclasses import replace
from time import perf_counter, thread_time

import numpy as np
import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception
from arrayscope.core.cache_status import CacheStatus, CacheStatusSnapshot
from arrayscope.core.memory_budget import estimate_display_image_bytes, format_bytes
from arrayscope.core.frame_targets import FrameTarget
from arrayscope.core.trace import emit_trace
from arrayscope.core.view_state import ChannelMode
from arrayscope.kernel import Lane as WorkLane, WorkItem, complete_inline_work as _complete_inline_work
from arrayscope.display.frame_planner import FramePlanner
from arrayscope.display.montage import (
    MontageTileState,
    RenderedTile,
    make_montage_plan,
)
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.backends.base import surface_for_view, tiled_presentation_visible
from arrayscope.operations.evaluator import _document_key
from arrayscope.render import effects as render_effects
from arrayscope.render.stages import RenderIntent
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.display_presenter import tile_residency_budget_bytes
from arrayscope.display.model.montage_levels import (
    MontageLevelStats,
)
from arrayscope.display.lod import LOD_POLICY_RESIDENT, factor_xy_for_level, select_lod_demand
from arrayscope.display.pyramid import preview_level_for_tile_shape
from arrayscope.window.montage_payload_cache import (
    payload_lod_matches as _payload_lod_matches,
    payload_compatible_with_tile as _payload_compatible_with_tile,
    previous_tiled_payloads_by_base_source as _previous_tiled_payloads_by_base_source,
    RetainedTiledPayloadStore,
)
from arrayscope.window import frame_effects as montage_commit
from arrayscope.window.frame_effects import FramePipelineEffects
from arrayscope.render.level_stats import LevelStatsService
from arrayscope.window.montage_prefetch import schedule_near_viewport_montage_prefetch
from arrayscope.window.frame_runtime import (
    FrameRuntimeMixin,
    _montage_autofit_scope_grew,
    _montage_autofit_signature,
    _should_auto_fit_montage_view,
    _viewport_controller_auto_active_for_range,
)
from arrayscope.window.montage_viewport import (
    MontageViewportPlan,
    effective_montage_columns,
    frame_session_key,
    montage_priority_focus as _montage_priority_focus,
    montage_tile_semantic_key,
    montage_viewport_intent,
    montage_viewport_retarget_policy,
    plan_full_view_range,
    remap_montage_roi_selections,
    retarget_montage_viewport_plan,
    square_montage_fit_view_range,
)
from arrayscope.render import lod as render_lod
from arrayscope.window.frame_session import FrameSession, plan_presentation_transition
from arrayscope.window.render_contract import (
    session_token_is_current as _session_token_is_current,
)
from arrayscope.display.planning import fallback_level_source, normalize_bounds


MONTAGE_VERY_SLOW_UPLOAD_MS = 100.0
class FrameControllerMixin(FrameRuntimeMixin, LevelStatsService):
    def _interactive_frame_cache_hit(self) -> bool:
        view_state = getattr(self.win, "view_state", None)
        if view_state is None or view_state.image_axes is None:
            return False
        evaluator = getattr(self.win, "operation_evaluator", None)
        if evaluator is None:
            return False
        shader_display = bool(image_view_backend_capabilities(self.win.img_view).shader_windowing)
        colormap_lut = self._evaluation_colormap_lut(view_state, shader_display=shader_display)
        viewport_plan = self._montage_viewport_plan(view_state)
        expected_key = frame_session_key(
            _document_key(self.win.document),
            view_state,
            viewport_plan,
            colormap_lut,
        )
        frame = getattr(self.win, "_committed_display_frame", None)
        frame_key = None if frame is None else getattr(frame, "key", None)
        if getattr(frame_key, "request_key", None) != expected_key:
            return False
        for tile in viewport_plan.candidate_tiles(margin_tiles=0):
            if (
                evaluator.cached_montage_tile(
                    tile.view_state,
                    montage_axis=viewport_plan.axis,
                    source_index=tile.source_index,
                    colormap_lut=colormap_lut,
                    document=self.win.document,
                    shader_display=shader_display,
                )
                is None
            ):
                return False
        return True

    def _interactive_render_supersedes_presentation(self, *, reason: str) -> bool:
        view_state = getattr(self.win, "view_state", None)
        if view_state is None or view_state.image_axes is None:
            return False
        frame = getattr(self.win, "_committed_display_frame", None)
        frame_key = None if frame is None else getattr(frame, "key", None)
        capabilities = image_view_backend_capabilities(self.win.img_view)
        shader_display = bool(capabilities.shader_windowing)
        colormap_lut = self._evaluation_colormap_lut(view_state, shader_display=shader_display)
        viewport_plan = self._montage_viewport_plan(view_state)
        semantic_key = RenderIntent.semantic_key_for_montage(
            _document_key(self.win.document),
            view_state,
            viewport_plan,
            colormap_lut,
        )
        return getattr(frame_key, "request_key", None) != semantic_key

    def _montage_frame_planner(self) -> FramePlanner:
        provider = getattr(self, "_frame_planner", None)
        if provider is not None:
            return provider()
        planner = getattr(self, "_montage_frame_planner_instance", None)
        if planner is None:
            planner = FramePlanner()
            self._montage_frame_planner_instance = planner
        return planner

    def _montage_tile_layer_policy(self, geometry, data) -> bool:
        # Every montage display commits through the tile layer; the historical
        # per-size/per-backend chooser collapsed to this predicate.
        return getattr(geometry, "montage", None) is not None

    def _retained_tiled_payload_store(self) -> RetainedTiledPayloadStore:
        store = getattr(self, "_montage_retained_tiled_payloads", None)
        if not isinstance(store, RetainedTiledPayloadStore):
            store = RetainedTiledPayloadStore()
            self._montage_retained_tiled_payloads = store
        return store

    def _session_source_anchoring(self, document, view_state, axis):
        """ADR 0055 G3: window-invariant payload anchoring for the session.

        Non-montage sessions on atlas-resident backends stamp exact payloads
        with a window-free content identity so the backend can keep chunked
        residency across display-window shifts. Montage sessions already
        anchor per source index; CPU-item backends re-window anyway.
        """

        if axis is not None:
            return None
        capabilities = image_view_backend_capabilities(self.win.img_view)
        if getattr(capabilities, "tile_residency_kind", None) != "gpu_atlas":
            return None
        from arrayscope.display.source_anchoring import source_anchoring_for_view

        return source_anchoring_for_view(document, view_state)

    def _montage_viewport_plan(self, view_state, *, view_range=None) -> MontageViewportPlan:
        axis = view_state.montage_axis
        all_indices = (0,) if axis is None else tuple(view_state.montage_indices or tuple(range(int(view_state.shape[axis]))))
        viewport_size = self.win.img_view.graphicsView.viewport().size()
        viewport_shape = (max(1, viewport_size.height()), max(1, viewport_size.width()))
        tile_shape = self._montage_tile_shape(view_state)
        pending_restore_range = None
        pending_restore_columns = None
        if view_range is None:
            pending_restore = getattr(self.win, "_pending_viewport_continuity_range", None)
            pending_restore_range = pending_restore() if callable(pending_restore) else None
            if pending_restore_range is None:
                # A continuity restore that has been applied but whose window
                # shape has not settled yet still owns the target range; the
                # live camera is transitional and would misanchor everything
                # derived from this plan (priority focus, tile bands).
                active_restore = getattr(self.win, "_active_viewport_continuity_range", None)
                pending_restore_range = active_restore() if callable(active_restore) else None
            pending_columns = getattr(self.win, "_pending_viewport_continuity_columns", None)
            pending_restore_columns = pending_columns() if callable(pending_columns) else None
        # The live Qt camera range is only trustworthy once this montage owns
        # the viewport. Entering montage mode from a single plane (or a montage
        # on a different axis), and the whole first fill before the initial
        # commit rescales the camera, leave the camera on the *previous* view —
        # a tiny single-plane range that reads as native scale (desired_level
        # 0). Seeding the plan from that stale range makes the first montage
        # render evaluate native tiles instead of the cheap reduced fit LOD.
        # When the camera is not yet on this montage, fall through to the fit
        # extent (`_initial_montage_planning_view_range`) below.
        current_session = getattr(self, "_frame_session", None)
        camera_on_montage = bool(
            current_session is not None
            and getattr(current_session, "montage_axis", None) == axis
            and getattr(current_session, "display_committed", False)
        )
        current_range = view_range if view_range is not None else (
            pending_restore_range
            if pending_restore_range is not None
            else (
                self._current_montage_global_view_range()
                if getattr(self.win.img_view, "image", None) is not None and camera_on_montage
                else None
            )
        )
        columns = self._effective_montage_columns(
            view_state,
            all_indices=all_indices,
            tile_shape=tile_shape,
            viewport_shape=viewport_shape,
            view_range=current_range,
            restored_columns=pending_restore_columns,
        )
        plan = make_montage_plan(
            view_state,
            axis=axis,
            indices=all_indices,
            tile_shape=tile_shape,
            columns=columns,
            viewport_shape=viewport_shape,
        )
        viewport_controller = getattr(self.win.img_view, "viewport_controller", None)
        planning_intent = montage_viewport_intent(viewport_controller, current_range)
        if (
            view_range is None
            and pending_restore_range is None
            and planning_intent.auto_like
        ):
            # AUTO/FIT describes the intended successor camera, even while the
            # acknowledged predecessor camera must remain onscreen. Plan the
            # successor's full visible obligation from that intent so a CPU
            # backend cannot commit only the predecessor-sized subset and then
            # expose holes when acknowledgement finally applies the new fit.
            current_range = _initial_montage_planning_view_range(
                plan,
                viewport_shape,
                viewport_controller,
            )
        elif current_range is None:
            current_range = _initial_montage_planning_view_range(
                plan,
                viewport_shape,
                viewport_controller,
            )
        priority_focus = _montage_priority_focus(self, current_range)
        capabilities = image_view_backend_capabilities(self.win.img_view)
        return MontageViewportPlan(
            axis=None if axis is None else int(axis),
            all_indices=all_indices,
            viewport_shape=viewport_shape,
            tile_shape=tile_shape,
            plan=plan,
            view_range=current_range,
            shader_display=bool(capabilities.shader_windowing),
            persistent_tile_residency=bool(capabilities.persistent_tile_residency),
            priority_focus=priority_focus,
        )

    def _effective_montage_columns(
        self,
        view_state,
        *,
        all_indices,
        tile_shape,
        viewport_shape,
        view_range,
        restored_columns=None,
    ) -> int | None:
        if not all_indices:
            return None
        viewport_controller = getattr(getattr(self.win, "img_view", None), "viewport_controller", None)
        requested_columns = getattr(view_state, "montage_columns", None)
        if requested_columns is None and restored_columns is not None:
            try:
                requested_columns = max(1, int(restored_columns))
            except (TypeError, ValueError):
                requested_columns = None
        intent = montage_viewport_intent(viewport_controller, view_range)
        return effective_montage_columns(
            len(all_indices),
            tile_shape,
            viewport_shape,
            requested_columns=requested_columns,
            fit_locked=intent.fit_locked,
            auto_active=intent.auto_active,
        )

    def _on_image_viewport_resized(self, *, previous_viewport_size=None, base_view_range=None, resize_focus=None) -> None:
        if getattr(self.win, "_closing", False):
            return
        continuity_shape = getattr(self.win, "_viewport_continuity_shape_target", lambda: None)()
        if continuity_shape is not None:
            # Child layouts can resize after an outer-window continuity resize
            # has looked settled (the VisPy native canvas is one example).
            # A genuine top-level user resize releases the transaction in the
            # window's resizeEvent before this callback, so an extant target is
            # still authoritative and must be restored instead of reflowed.
            restore_viewport_shape = getattr(self.win, "_restore_viewport_continuity_shape_after_layout", None)
            if callable(restore_viewport_shape):
                restore_viewport_shape()
            return
        active_continuity_range = getattr(self.win, "_active_viewport_continuity_range", lambda: None)()
        if active_continuity_range is not None:
            restore_viewport_shape = getattr(self.win, "_restore_viewport_continuity_shape_after_layout", None)
            if callable(restore_viewport_shape):
                restore_viewport_shape()
            self._set_montage_view_range(active_continuity_range)
            self.retarget_montage_viewport()
            return
        self._montage_live_layout_reflow = True
        self.win._montage_viewport_update_pending = False
        viewport_plan = self._retarget_montage_resize_camera(
            previous_viewport_size=_montage_viewport_shape_from_qt_size_tuple(previous_viewport_size),
            base_view_range=base_view_range,
            resize_focus=resize_focus,
        )
        if viewport_plan is not None:
            self._retarget_montage_resize_payloads(viewport_plan)
        self.retarget_montage_viewport()

    def _current_montage_resize_focus(self, view_range) -> tuple[float, float] | None:
        return _montage_priority_focus(self, view_range)

    def _retarget_montage_resize_camera(
        self,
        *,
        previous_viewport_size=None,
        base_view_range=None,
        resize_focus=None,
    ) -> MontageViewportPlan | None:
        session = getattr(self, "_frame_session", None)
        if not self._frame_session_is_current(session):
            return None
        view_state = self.win.view_state
        if view_state.image_axes is None:
            # The live view state no longer presents an image (e.g. a reduction
            # switched to line-plot mode) while the montage session from the
            # previous state is still attached. There is nothing to retarget;
            # the pending render replaces the session.
            return None
        capabilities = image_view_backend_capabilities(self.win.img_view)
        viewport_plan = self._montage_viewport_plan(view_state, view_range=base_view_range)
        colormap_lut = self._evaluation_colormap_lut(
            view_state,
            shader_display=bool(capabilities.shader_windowing),
        )
        expected_key = frame_session_key(
            _document_key(self.win.document),
            view_state,
            viewport_plan,
            colormap_lut,
        )
        # Not the shared currency predicate: this compares the session against
        # the key re-derived from the *live* view state (semantic-match check).
        if session.key != expected_key:
            return None
        previous_plan = getattr(session, "plan", None)
        viewport_plan = self._retargeted_montage_viewport_plan(
            session,
            viewport_plan,
            previous_viewport_shape=previous_viewport_size,
            focus=resize_focus,
        )
        self._remap_montage_rois_for_layout_reflow(previous_plan, viewport_plan.plan)
        return viewport_plan

    def _retarget_montage_resize_payloads(self, viewport_plan: MontageViewportPlan) -> bool:
        session = getattr(self, "_frame_session", None)
        if not self._frame_session_is_current(session):
            return False
        capabilities = image_view_backend_capabilities(self.win.img_view)
        try:
            display_mode = str(self.win.img_view.montageDisplayMode())
        except Exception:
            display_mode = ""
        if not montage_viewport_retarget_policy(capabilities, display_mode).enabled:
            return False
        cancel_speculative = getattr(self.win.img_view, "_cancel_vispy_speculative_work", None)
        if callable(cancel_speculative):
            cancel_speculative()
        view_state = self.win.view_state
        _additions, presentation_changed = session.retarget_viewport(
            view_range=viewport_plan.view_range,
            viewport_shape=viewport_plan.viewport_shape,
            plan=viewport_plan.plan,
            coverage_margin_tiles=0,
            near_margin_tiles=0,
            priority_focus=viewport_plan.priority_focus,
            priority_retarget_limit=1,
        )
        memory_policy = self._memory_policy() if hasattr(self, "_memory_policy") else None
        session.frame_plan = self._montage_frame_planner().plan(
            target=FrameTarget(
                semantic_key=session.key,
                viewport_key=viewport_plan.view_range,
                presentation_key=(
                    str(session.window_mode),
                    normalize_bounds(getattr(session, "user_levels_override", None)),
                    bool(getattr(session, "force_auto", False)),
                ),
                quality="exact-visible",
            ),
            view_state=view_state,
            display_shape=viewport_plan.plan.display_shape,
            backend_capabilities=capabilities,
            viewport_shape=viewport_plan.viewport_shape,
            view_range=viewport_plan.view_range,
            memory_policy=memory_policy,
            montage_plan=viewport_plan.plan,
        )
        if memory_policy is not None:
            session.tile_residency_budget_bytes = tile_residency_budget_bytes(memory_policy)
        lod_swap_ready = session.mark_ladder_swaps_for_viewport()
        self.retarget_frame_pipeline(session)
        if presentation_changed or lod_swap_ready:
            self._commit_montage_resize_presentation_retarget(session)
        return True

    def _commit_montage_resize_presentation_retarget(self, session) -> None:
        self.apply_montage_presentation(session)

    def _publish_first_frame_content_extent(self, plan) -> bool:
        """Publish planned camera coordinates only when no frame can be retained."""

        if getattr(self.win, "_committed_display_frame", None) is not None:
            return False
        self._publish_montage_content_extent(plan)
        return True

    def update_image_view(self, *, force_autolevel: bool = False, defer_side_panels: bool = False):
        for attribute in (
            "_last_montage_viewport_plan_ms",
            "_last_montage_cache_resolve_ms",
            "_last_montage_stage_plan_ms",
            "_last_frame_session_setup_ms",
            "_last_montage_initial_commit_ms",
        ):
            setattr(self, attribute, None)
        axis = self.win.view_state.montage_axis
        if self.win.view_state.image_axes is None or axis in self.win.view_state.image_axes:
            return
        plan_start = perf_counter()
        policy = self._refresh_memory_policy(active_render=self._montage_render_active())
        if policy is None:
            policy = self._memory_policy()
        user_levels = self._pending_display_levels_for_render()
        if force_autolevel and user_levels is not None:
            self._queue_display_levels(None)
            user_levels = None
        force_auto = bool(
            force_autolevel
            or (getattr(self.win, '_force_autolevel', False) and user_levels is None)
        )
        if getattr(self.win, '_force_autolevel', False):
            self.win._force_autolevel = False
        window_mode = self._current_window_mode()
        previous_frame = self._previous_display_frame_for_policy(force_auto=force_auto)
        pending_auto_level_source = getattr(self, "_pending_auto_level_source", None) if force_auto else None
        carried_user_level_source = None
        if not force_auto:
            candidate_user_source = getattr(self, "_explicit_user_level_source", None)
            if normalize_bounds(getattr(candidate_user_source, "levels", None)) is not None:
                carried_user_level_source = candidate_user_source

        view_state = self.win.view_state
        shader_display = bool(image_view_backend_capabilities(self.win.img_view).shader_windowing)
        colormap_lut = self._evaluation_colormap_lut(view_state, shader_display=shader_display)
        document = self.win.document
        viewport_plan = self._montage_viewport_plan(view_state)
        all_indices = viewport_plan.all_indices
        viewport_shape = viewport_plan.viewport_shape
        tile_shape = viewport_plan.tile_shape
        plan = viewport_plan.plan
        if self._maybe_auto_fit_montage_tiles(plan):
            viewport_plan = self._montage_viewport_plan(
                view_state,
                view_range=self._current_montage_global_view_range(),
            )
            viewport_shape = viewport_plan.viewport_shape
            tile_shape = viewport_plan.tile_shape
            plan = viewport_plan.plan
        # The committed frame owns coordinate semantics. On a first frame there
        # is no prior surface to preserve, so publishing the planned extent is
        # useful for startup fit. A replacement keeps the acknowledged extent
        # until its pixels commit; moving the camera to uncommitted geometry
        # makes compatible retained tiles appear to go black.
        self._publish_first_frame_content_extent(plan)
        self._montage_live_layout_reflow = False
        previous_session_plan = getattr(getattr(self, "_frame_session", None), "plan", None)
        self._remap_montage_rois_for_layout_reflow(previous_session_plan, plan)
        current_range = viewport_plan.view_range
        display_tiles = viewport_plan.candidate_tiles(margin_tiles=0)
        candidate_tiles = viewport_plan.candidate_tiles(
            margin_tiles=1 if viewport_plan.persistent_tile_residency else 0,
            prioritize=True,
        )
        shader_display = viewport_plan.shader_display
        output_dtype = getattr(
            getattr(self.win, "data", None),
            "dtype",
            getattr(document.base_data, "dtype", np.dtype(float)),
        )
        single_estimate = estimate_display_image_bytes(
            tile_shape,
            output_dtype,
            rgb=view_state.channel == ChannelMode.COMPLEX,
            histogram=True,
        )
        if single_estimate > policy.single_tile_budget_bytes:
            compute_tiles = ()
            skipped_tiles = tuple(candidate_tiles)
            skipped_count = len(skipped_tiles)
        else:
            compute_tiles = tuple(candidate_tiles)
            skipped_tiles = ()
            skipped_count = 0
        if not compute_tiles and not skipped_tiles:
            show_status_message(
                self.win,
                f"Montage tile would allocate {format_bytes(single_estimate)}. Zoom out less or reduce tile size/range.",
                timeout=6000,
            )
            return
        if skipped_count:
            self._warn_montage_tiles_skipped(
                skipped_count=skipped_count,
                tile_bytes=single_estimate,
                budget_bytes=policy.single_tile_budget_bytes,
                tile_shape=tile_shape,
            )
        self._last_montage_viewport_plan_ms = (perf_counter() - plan_start) * 1000.0
        cache_start = perf_counter()
        cached_tiles, missing_tiles = self._resolve_montage_tiles_from_cache(
            compute_tiles,
            document=document,
            axis=axis,
            colormap_lut=colormap_lut,
            shader_display=shader_display,
        )
        self._last_montage_cache_resolve_ms = (perf_counter() - cache_start) * 1000.0
        self._montage_cached_tiles_last_session = len(cached_tiles)
        self._montage_missing_tiles_last_session = len(missing_tiles)
        if self._maybe_retarget_frame_session(
            getattr(self, "_frame_session", None),
            document=document,
            axis=axis,
            view_state=view_state,
            viewport_plan=viewport_plan,
            plan=plan,
            policy=policy,
            colormap_lut=colormap_lut,
            window_mode=window_mode,
            force_auto=force_auto,
            user_levels=user_levels,
            output_dtype=output_dtype,
            shader_display=shader_display,
            cached_tiles=cached_tiles,
            missing_tiles=missing_tiles,
            skipped_tiles=skipped_tiles,
            all_indices=all_indices,
            display_tiles=display_tiles,
            current_range=current_range,
            viewport_shape=viewport_shape,
        ):
            return
        render_generation = self._capture_render_generation()
        lod_policy_mode = self._montage_quality_policy_mode()
        initial_demand = select_lod_demand(
            current_range,
            viewport_shape,
            plan.tile_shape,
        )
        # Interaction fast path: during a scrub/pan burst, supersedable stage
        # planning runs only for the step the user lands on. Native policy and
        # first-ever montage builds still plan inline because they have no
        # retained screen to carry.
        previous_session = getattr(self, "_frame_session", None)
        defer_stage_planning = bool(
            missing_tiles
            and _viewport_interaction_active(self)
            and self._montage_quality_policy_mode() == LOD_POLICY_RESIDENT
            and previous_session is not None
            # Only a committed predecessor on the same axis can carry the
            # screen; a first fill or axis change is not a scrub step.
            and bool(getattr(previous_session, "display_committed", False))
            and getattr(previous_session, "montage_axis", None) == axis
        )
        queue_native_missing_tiles = bool(
            not defer_stage_planning
            and render_lod.native_missing_tile_queue_required(
                lod_policy_mode,
                initial_demand,
            )
        )
        stage_plan_start = perf_counter()
        if missing_tiles and not queue_native_missing_tiles and not defer_stage_planning:
            stage_plan = montage_commit.deferred_stage_fan_in_plan()
        elif defer_stage_planning:
            stage_plan = montage_commit.hot_cached_stage_fan_in_plan(
                self,
                document,
                missing_tiles,
            )
            if not montage_commit.stage_fan_in_plan_has_existing_sources(stage_plan):
                stage_plan = montage_commit.deferred_stage_fan_in_plan()
            self._montage_stage_plans_deferred = (
                int(getattr(self, "_montage_stage_plans_deferred", 0) or 0) + 1
            )
        else:
            stage_plan = montage_commit.build_stage_fan_in_plan(self, document, missing_tiles)
        self._last_montage_stage_plan_ms = (perf_counter() - stage_plan_start) * 1000.0
        session_setup_start = perf_counter()
        pending_tiles = list(missing_tiles) if queue_native_missing_tiles else []
        session_key = frame_session_key(_document_key(document), view_state, viewport_plan, colormap_lut)
        level_key = self._montage_level_key(document, view_state, all_indices, colormap_lut)
        frame_plan = self._montage_frame_planner().plan(
            target=FrameTarget(
                semantic_key=session_key,
                viewport_key=current_range,
                presentation_key=(str(window_mode), normalize_bounds(user_levels), bool(force_auto)),
                quality="exact-visible",
            ),
            view_state=view_state,
            display_shape=plan.display_shape,
            backend_capabilities=image_view_backend_capabilities(self.win.img_view),
            viewport_shape=viewport_shape,
            view_range=current_range,
            memory_policy=policy,
            montage_plan=plan,
        )
        session_id = int(getattr(self, "_frame_session_id", 0)) + 1
        self._frame_session_id = session_id
        lod_preview_level = (
            preview_level_for_tile_shape(plan.tile_shape, min_level=render_lod.PREVIEW_FLOOR_MIN_LEVEL)
            if lod_policy_mode == LOD_POLICY_RESIDENT
            else 0
        )
        session = FrameSession(
            session_id=session_id,
            key=session_key,
            semantic_key=montage_tile_semantic_key(
                _document_key(document), view_state, viewport_plan, colormap_lut
            ),
            render_generation=render_generation,
            level_key=level_key,
            level_expected_indices=tuple(int(index) for index in all_indices),
            frame_plan=frame_plan,
            plan=plan,
            view_state=view_state,
            document=document,
            montage_axis=axis,
            source_anchoring=self._session_source_anchoring(document, view_state, axis),
            colormap_lut=colormap_lut,
            viewport_shape=viewport_shape,
            view_range=current_range,
            output_dtype=np.dtype(output_dtype),
            rgb=view_state.channel == ChannelMode.COMPLEX,
            window_mode=window_mode,
            force_auto=force_auto,
            visible_tiles=tuple(display_tiles),
            rendered_tiles={int(rendered.tile.montage_index): rendered for rendered in cached_tiles},
            loading_tiles=set(),
            skipped_tiles={int(tile.montage_index) for tile in skipped_tiles},
            pending_tiles=list(pending_tiles),
            shader_display=bool(shader_display),
            stage_fan_in=montage_commit.stage_fan_in_state(stage_plan),
            defer_side_panels=bool(defer_side_panels),
            applied_level_source=(
                pending_auto_level_source
                if pending_auto_level_source is not None
                else (
                    carried_user_level_source
                    if carried_user_level_source is not None
                    else (None if previous_frame is None else fallback_level_source(previous_frame))
                )
            ),
            user_levels_override=user_levels,
            tile_compute_cache_hits=len(cached_tiles),
            tile_compute_waiting_for_stage=len(stage_plan["waiting_indices"]),
            lead_direct_tiles=stage_plan["lead_direct_tiles"],
            stage_backed_tiles_pending=len(stage_plan["waiting_indices"]),
            retained_stage_index=stage_plan["retained_stage_index"],
            retained_stage_decision=stage_plan["retained_stage_decision"],
            repeated_expensive_stage_per_tile=stage_plan["repeated_expensive_stage_per_tile"],
            priority_focus=viewport_plan.priority_focus,
            lod_policy_mode=lod_policy_mode,
            lod_native_reason=render_lod.native_policy_reason_for_renderer(self),
            lod_preview_level=lod_preview_level,
            lod_preview_min_level=lod_preview_level,
            tile_residency_budget_bytes=tile_residency_budget_bytes(policy),
            lod_page_cache=(
                self._lod_page_cache() if lod_policy_mode == LOD_POLICY_RESIDENT else None
            ),
        )
        stage_planning_deferred = bool(
            defer_stage_planning
            or (missing_tiles and not queue_native_missing_tiles)
        )
        session.stage_planning_deferred = stage_planning_deferred
        session.stage_planning_async = False
        session.deferred_missing_tiles = tuple(missing_tiles) if stage_planning_deferred else ()
        # The dying session's planned-but-undrained LOD requests hold
        # singleflight claims in the shared pyramid; scrubbing back to the
        # same slice would find those levels permanently claimed (stale
        # wrong-LOD tiles).  Balance them before the replacement takes over.
        dying_session = getattr(self, "_frame_session", None)
        render_lod.release_session_claims(dying_session)
        # Backend slots outlive sessions (persistent tile residency), so the
        # identity ground truth from the last report stays valid — but a fresh
        # lifecycle starts without backend slot truth, blind to inherited
        # stale slots until its own first report.  Inherit the backend snapshot
        # into the new lifecycle; tiles absent from the new plan fall out
        # naturally (no current payload → mismatch scan skips them).
        if dying_session is not None:
            inherited = getattr(dying_session.lifecycle, "backend_presented_identities", None)
            if inherited:
                session.lifecycle.backend_presented_snapshot(inherited)
        # Activating a new target is an atomic visibility boundary.  Backend
        # residency outlives a FrameSession, but inherited slot mappings do
        # not: until the new session explicitly rebinds and acknowledges them,
        # they are not evidence for the new target (including viewport-only
        # retargets whose semantic key intentionally remains stable).
        #
        # "Pixels stay visible" is a separate axis from "mappings are not
        # evidence" (ADR 0051).  A slice-index-only rebirth keeps DRAWING the
        # predecessor's plane — a stale-but-honest preview, exactly like a
        # video player between frames — while all session bookkeeping below
        # still starts cold: no acknowledgement inheritance, no seeded
        # evidence (payload seeding requires exact source identities), and
        # the successor's first commit swaps the drawn pixels atomically.
        # The transition planner keeps the predecessor as independently named
        # physical truth while compatible successor semantics are prepared.
        # It rejects a different staged document or surface/backend contract;
        # derived representation, view state, and auto layout belong to the
        # complete successor transaction and cannot justify a black flash.
        surface = surface_for_view(self.win.img_view)
        transition = plan_presentation_transition(
            dying_session,
            session,
            predecessor_visible=tiled_presentation_visible(surface),
        )
        retain_stale_pixels = bool(transition.retain_pixels)
        session.atomic_successor_pending = bool(transition.atomic_successor)
        emit_trace(
            "presentation_transition_retention",
            session_id=int(session.session_id),
            predecessor_session_id=int(
                getattr(dying_session, "session_id", 0) or 0
            ),
            retained=bool(retain_stale_pixels),
            reason=str(transition.reason),
            detail=str(transition.detail),
            force_auto=bool(session.force_auto),
            montage_axis=getattr(session, "montage_axis", None),
            atomic_successor_pending=bool(session.atomic_successor_pending),
        )
        self._frame_session_transition_retained_pixels = bool(retain_stale_pixels)
        if retain_stale_pixels:
            self._frame_session_transitions_retained = (
                int(getattr(self, "_frame_session_transitions_retained", 0) or 0) + 1
            )
            self._slice_retention_started_at = perf_counter()
            self._slice_retention_session_id = int(session.session_id)
            emit_trace(
                "slice_retention_started",
                session_id=int(session.session_id),
                predecessor_session_id=int(getattr(dying_session, "session_id", 0) or 0),
                transition_count=int(self._frame_session_transitions_retained),
            )
        else:
            self._slice_retention_started_at = None
            self._slice_retention_session_id = None
        surface.invalidate_tiled_presentation(
            "frame-session-transition",
            hide_pixels=not retain_stale_pixels,
        )
        self._frame_session = session
        # A live session can receive cached/refined level evidence as soon as
        # its level work is queued below.  Attach its one effects owner first
        # so an immediate kernel completion cannot observe a current session
        # without the presentation gate needed to publish that evidence.
        self._frame_pipeline_for_session(session)
        # A viewport-update token armed for the dying session would make every
        # later apply_montage_viewport_retarget bail as stale — a dead
        # continuation (lost-wakeup class).  The new session's construction
        # subsumes any pending retarget; clear the token so future runs (timer
        # or explicit) act on the current session.
        self._montage_viewport_update_token = None
        self._ensure_montage_watchdog()
        apply_restored_viewport = getattr(self.win, "_apply_viewport_continuity_when_ready", None)
        if callable(apply_restored_viewport):
            apply_restored_viewport()
        _complete_inline_work(
            self,
            WorkItem(
                key=("montage_visible_planning", session.key, int(session.session_id)),
                lane=WorkLane.VISIBLE_PLANNING,
                frame_target=session.frame_plan.target,
                supersession_key=("montage-visible", session.key),
                supersession_value=int(session.session_id),
                estimated_cpu_ms=float(self._last_montage_viewport_plan_ms or 0.0)
                + float(self._last_montage_cache_resolve_ms or 0.0)
                + float(self._last_montage_stage_plan_ms or 0.0),
                estimated_bytes=int(single_estimate) * max(1, len(tuple(display_tiles or ()))),
            ),
        )
        if pending_auto_level_source is not None:
            self._pending_auto_level_source = None
        self._ensure_montage_level_stats(level_key, expected_indices=all_indices)
        self._queue_montage_cached_level_stats(session, cached_tiles, seed_if_empty=True)
        self._last_frame_session_setup_ms = (perf_counter() - session_setup_start) * 1000.0
        initial_commit_start = perf_counter()
        try:
            self.commit_frame_session_presentation(session)
        except MemoryError as exc:
            show_status_message(self.win, str(exc), timeout=6000)
            return
        finally:
            self._last_montage_initial_commit_ms = (perf_counter() - initial_commit_start) * 1000.0
        if session.is_complete():
            self._finish_frame_session_if_complete(session)
            if defer_side_panels or _viewport_interaction_active(self):
                self.win._deferred_side_panel_refresh_pending = True
            else:
                self.win._update_operation_dock()
            self._schedule_montage_cached_level_stats(session)
            return
        visible_complete = self._settle_montage_visible_plan_if_complete(session)
        self.win.prefetch_evaluation_controller.cancel_prefetch()
        if not visible_complete:
            self.win.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.COMPUTING, "Evaluating image frame")
        if defer_side_panels or _viewport_interaction_active(self):
            self.win._deferred_side_panel_refresh_pending = True
        else:
            self.win._update_operation_dock()
        if not visible_complete:
            self.show_frame_session_slow_overlay(session)
        self._schedule_montage_cached_level_stats(session)
        if defer_stage_planning:
            montage_commit.submit_deferred_stage_fan_in_plan(self, session, missing_tiles)
            self.retarget_frame_pipeline(session)
        else:
            montage_commit.submit_stage_tasks(self, session, stage_plan["stage_requests"])
            self.retarget_frame_pipeline(session)

    def _maybe_retarget_frame_session(
        self,
        previous_session,
        *,
        document,
        axis,
        view_state,
        viewport_plan,
        plan,
        policy,
        colormap_lut,
        window_mode,
        force_auto,
        user_levels,
        output_dtype,
        shader_display,
        cached_tiles,
        missing_tiles,
        skipped_tiles,
        all_indices,
        display_tiles,
        current_range,
        viewport_shape,
    ) -> bool:
        """Retarget the live session to a new index window instead of a rebirth.

        ADR 0051 P2 (session-rebirth cost): index-window scrubs and
        camera-only viewport changes reuse the session object. Backend
        acknowledgement state and drawn payloads survive; missing stage
        planning is either attached from retained source data immediately or
        submitted as kernel-owned supersedable work. Returns True when the
        retarget handled the step.
        """

        def _reject(reason: str) -> bool:
            self._frame_session_retarget_last_reject = reason
            rejects = getattr(self, "_frame_session_retarget_rejects", None)
            if rejects is None:
                rejects = {}
                self._frame_session_retarget_rejects = rejects
            rejects[reason] = int(rejects.get(reason, 0)) + 1
            return False

        session = previous_session
        if session is None or session is not getattr(self, "_frame_session", None):
            return _reject("no-session")
        if axis is None or getattr(session, "montage_axis", None) is None:
            # Normal sliced images share this tiled path with axis=None; a
            # slice change is new semantic content behind an unchanged
            # layout, not an index-window move.  Only true montage sessions
            # retarget.
            return _reject("no-axis")
        if not bool(getattr(session, "display_committed", False)):
            return _reject("uncommitted")
        if getattr(session, "montage_axis", None) != axis:
            return _reject("axis")
        if bool(force_auto):
            return _reject("force-auto")
        if bool(skipped_tiles) or bool(getattr(session, "skipped_tiles", None)):
            return _reject("skipped-tiles")
        if _document_key(session.document) != _document_key(document):
            return _reject("document")
        if session.lod_policy_mode != self._montage_quality_policy_mode():
            return _reject("lod-policy")
        if session.window_mode != window_mode:
            return _reject("window-mode")
        if session.user_levels_override != user_levels:
            return _reject("user-levels")
        if session.colormap_lut is not colormap_lut:
            return _reject("colormap")
        if bool(getattr(session, "shader_display", False)) != bool(shader_display):
            return _reject("shader-display")
        if session.output_dtype != np.dtype(output_dtype):
            return _reject("dtype")
        if bool(session.rgb) != bool(view_state.channel == ChannelMode.COMPLEX):
            return _reject("rgb")
        # Level convergence no longer forces rebirth: level upserts are
        # machine-gated, and retarget_index_window resets per-window evidence
        # while kept tiles continue converging.
        previous_geometry = getattr(session.plan, "geometry", None)
        geometry = getattr(plan, "geometry", None)
        if previous_geometry is None or geometry is None:
            return _reject("geometry-missing")
        if (
            tuple(previous_geometry.tile_shape) != tuple(geometry.tile_shape)
            or int(previous_geometry.columns) != int(geometry.columns)
            or int(previous_geometry.rows) != int(geometry.rows)
            or int(previous_geometry.gap) != int(geometry.gap)
            or len(previous_geometry.indices) != len(geometry.indices)
        ):
            return _reject("geometry")
        session_key = frame_session_key(
            _document_key(document), view_state, viewport_plan, colormap_lut
        )
        viewport_changed = (
            tuple(session.viewport_shape) != tuple(viewport_shape)
            or session.view_range != current_range
        )
        if session_key == session.key and viewport_changed:
            if self._try_update_montage_viewport_only():
                self._frame_session_reuses = (
                    int(getattr(self, "_frame_session_reuses", 0) or 0) + 1
                )
                return True
            return _reject("viewport-retarget")
        if session_key == session.key:
            # Same identity: refresh the generation stamp, commit genuine
            # dirt, and let the standing machinery converge without rebirth.
            self._frame_session_reuses = (
                int(getattr(self, "_frame_session_reuses", 0) or 0) + 1
            )
            session.render_generation = self._capture_render_generation()
            self._montage_viewport_update_token = None
            self._ensure_montage_watchdog()
            reuse_commit_start = perf_counter()
            try:
                # Only genuinely pending presentation work commits here; the
                # flush/level continuations own their own pacing, and a
                # settled re-render must stay a true no-op.
                if (
                    session.dirty_payloads
                    or session.pending_removals
                    or session.pending_payload_upserts
                    or not session.display_committed
                ):
                    self.commit_frame_session_presentation(session)
            finally:
                self._last_montage_initial_commit_ms = (
                    perf_counter() - reuse_commit_start
                ) * 1000.0
            self._last_montage_stage_plan_ms = 0.0
            self._last_frame_session_setup_ms = 0.0
            if session.defer_side_panels or _viewport_interaction_active(self):
                self.win._deferred_side_panel_refresh_pending = True
            else:
                self.win._update_operation_dock()
            return True
        # Change detection uses the same memoized semantic key batch that the
        # lazy tile_source_ids fill uses, so unchanged tiles are exact hits.
        source_ids_start = perf_counter()
        try:
            tile_key_for = self.win.operation_evaluator.montage_tile_key_batch(
                colormap_lut=colormap_lut,
                document=document,
                shader_display=bool(shader_display),
            )
            new_source_ids = {
                int(tile.montage_index): tile_key_for(tile.view_state)
                for tile in tuple(getattr(plan, "tiles", ()) or ())
            }
        except Exception:
            return False
        self._last_montage_retarget_source_ids_ms = (perf_counter() - source_ids_start) * 1000.0
        semantic_key = montage_tile_semantic_key(
            _document_key(document), view_state, viewport_plan, colormap_lut
        )
        level_key = self._montage_level_key(document, view_state, all_indices, colormap_lut)
        frame_plan_start = perf_counter()
        frame_plan = self._montage_frame_planner().plan(
            target=FrameTarget(
                semantic_key=session_key,
                viewport_key=current_range,
                presentation_key=(str(window_mode), normalize_bounds(user_levels), bool(force_auto)),
                quality="exact-visible",
            ),
            view_state=view_state,
            display_shape=plan.display_shape,
            backend_capabilities=image_view_backend_capabilities(self.win.img_view),
            viewport_shape=viewport_shape,
            view_range=current_range,
            memory_policy=policy,
            montage_plan=plan,
        )
        self._last_montage_retarget_frame_plan_ms = (perf_counter() - frame_plan_start) * 1000.0
        render_generation = self._capture_render_generation()
        # Undrained pyramid claims from the old window are balanced exactly as
        # on a rebirth; source identities are window-agnostic, so re-plans
        # re-claim cheaply.
        release_start = perf_counter()
        render_lod.release_session_claims(session)
        self._last_montage_retarget_release_ms = (perf_counter() - release_start) * 1000.0
        session_id = int(getattr(self, "_frame_session_id", 0)) + 1
        self._frame_session_id = session_id
        setup_start = perf_counter()
        model_start = perf_counter()
        stats = session.retarget_index_window(
            session_id=session_id,
            key=session_key,
            semantic_key=semantic_key,
            level_key=level_key,
            render_generation=render_generation,
            view_state=view_state,
            plan=plan,
            frame_plan=frame_plan,
            all_indices=all_indices,
            new_source_ids=new_source_ids,
            cached_tiles={
                int(rendered.tile.montage_index): rendered for rendered in cached_tiles
            },
            visible_tiles=tuple(display_tiles),
        )
        self._last_montage_retarget_model_ms = (perf_counter() - model_start) * 1000.0
        session.tile_residency_budget_bytes = tile_residency_budget_bytes(policy)
        session.force_auto = bool(force_auto)
        session.user_levels_override = user_levels
        stage_start = perf_counter()
        stage_cpu_start = thread_time()
        stage_plan_deferred = False
        if missing_tiles:
            hot_call_cpu_start = thread_time()
            hot_stage_plan = montage_commit.hot_cached_stage_fan_in_plan(
                self,
                session.document,
                missing_tiles,
            )
            self._last_montage_retarget_hot_call_cpu_ms = (thread_time() - hot_call_cpu_start) * 1000.0
            hot_predicate_cpu_start = thread_time()
            has_existing_stage = montage_commit.stage_fan_in_plan_has_existing_sources(hot_stage_plan)
            self._last_montage_retarget_hot_predicate_cpu_ms = (
                thread_time() - hot_predicate_cpu_start
            ) * 1000.0
            if not has_existing_stage:
                deferred_cpu_start = thread_time()
                hot_stage_plan = montage_commit.deferred_stage_fan_in_plan()
                stage_plan_deferred = True
                self._last_montage_retarget_hot_deferred_cpu_ms = (
                    thread_time() - deferred_cpu_start
                ) * 1000.0
        else:
            hot_stage_plan = montage_commit.deferred_stage_fan_in_plan()
        self._last_montage_retarget_hot_stage_ms = (perf_counter() - stage_start) * 1000.0
        self._last_montage_retarget_hot_stage_cpu_ms = (thread_time() - stage_cpu_start) * 1000.0
        attach_start = perf_counter()
        session.attach_stage_fan_in(montage_commit.stage_fan_in_state(hot_stage_plan))
        self._last_montage_retarget_attach_ms = (perf_counter() - attach_start) * 1000.0
        session.stage_planning_deferred = bool(missing_tiles and stage_plan_deferred)
        session.stage_planning_async = False
        session.deferred_missing_tiles = tuple(missing_tiles)
        session.tile_compute_cache_hits = int(stats["hits"])
        session.tile_compute_waiting_for_stage = len(hot_stage_plan["waiting_indices"])
        session.stage_backed_tiles_pending = len(hot_stage_plan["waiting_indices"])
        session.lead_direct_tiles = 0
        session.retained_stage_index = hot_stage_plan["retained_stage_index"]
        session.retained_stage_decision = (
            hot_stage_plan["retained_stage_decision"] or "deferred-retarget"
        )
        session.repeated_expensive_stage_per_tile = bool(hot_stage_plan["repeated_expensive_stage_per_tile"])
        if session.stage_planning_deferred:
            self._montage_stage_plans_deferred = (
                int(getattr(self, "_montage_stage_plans_deferred", 0) or 0) + 1
            )
        self._frame_session_retargets = (
            int(getattr(self, "_frame_session_retargets", 0) or 0) + 1
        )
        self._montage_cached_tiles_last_session = int(stats["hits"])
        self._montage_missing_tiles_last_session = len(missing_tiles)
        self._montage_viewport_update_token = None
        self._ensure_montage_watchdog()
        level_setup_start = perf_counter()
        self._ensure_montage_level_stats(level_key, expected_indices=all_indices)
        self._queue_montage_cached_level_stats(session, tuple(cached_tiles), seed_if_empty=True)
        self._last_montage_retarget_level_setup_ms = (perf_counter() - level_setup_start) * 1000.0
        self._last_montage_stage_plan_ms = 0.0
        self._last_frame_session_setup_ms = (perf_counter() - setup_start) * 1000.0
        _complete_inline_work(
            self,
            WorkItem(
                key=("montage_visible_planning", session.key, int(session.session_id)),
                lane=WorkLane.VISIBLE_PLANNING,
                frame_target=session.frame_plan.target,
                supersession_key=("montage-visible", session.key),
                supersession_value=int(session.session_id),
                estimated_cpu_ms=float(self._last_montage_viewport_plan_ms or 0.0)
                + float(self._last_montage_cache_resolve_ms or 0.0)
                + float(self._last_frame_session_setup_ms or 0.0),
                estimated_bytes=0,
            ),
        )
        initial_commit_start = perf_counter()
        try:
            self.commit_frame_session_presentation(session)
        finally:
            self._last_montage_initial_commit_ms = (
                perf_counter() - initial_commit_start
            ) * 1000.0
        if session.defer_side_panels or _viewport_interaction_active(self):
            self.win._deferred_side_panel_refresh_pending = True
        else:
            self.win._update_operation_dock()
        if missing_tiles:
            if session.stage_planning_deferred:
                montage_commit.submit_deferred_stage_fan_in_plan(self, session, missing_tiles)
            else:
                montage_commit.submit_stage_tasks(self, session, hot_stage_plan["stage_requests"])
        # Retained/floor cache hits can be valid first pixels without satisfying
        # the successor target.  They produce no ``missing_tiles`` entry, so
        # index-window retargeting must always give the ladder one chance to
        # admit the exact/desired follow-up for any unsettled target.
        self.retarget_frame_pipeline(session)
        return True

    def _resolve_montage_tiles_from_cache(
        self,
        tiles,
        *,
        document,
        axis: int,
        colormap_lut,
        shader_display: bool,
    ) -> tuple[list[RenderedTile], list[object]]:
        """Resolve only the supplied tiles from semantic CPU caches."""

        selected_lod_factor = 1
        # ADR 0050: under the resident policy a retained payload at any LOD
        # level is reusable evidence of the tile's native content (its exact
        # image/semantic planes are always native).  Rejecting reduced-level
        # payloads here forced pan/scroll re-evaluation of tiles that were
        # already on screen, which is what read as black/placeholder flashes
        # during LOD transitions.  The session re-selects the presented level
        # from live demand and streams the swap.
        reuse_any_lod = self._montage_quality_policy_mode() == LOD_POLICY_RESIDENT
        retained_store = self._retained_tiled_payload_store()
        previous_payloads = retained_store.payloads_by_base_source(
            lod_factor=None if reuse_any_lod else selected_lod_factor
        )
        previous_payloads.update(
            {
                key: payload
                for key, payload in _previous_tiled_payloads_by_base_source(
                    getattr(self.win, "_committed_display_frame", None)
                ).items()
                if reuse_any_lod or _payload_lod_matches(payload, selected_lod_factor)
            }
        )
        cached_tiles: list[RenderedTile] = []
        missing_tiles: list[object] = []
        total_lookup_ms = 0.0
        last_hit = False
        tile_tuple = tuple(tiles)
        tile_key_for = self.win.operation_evaluator.montage_tile_key_batch(
            colormap_lut=colormap_lut,
            document=document,
            shader_display=shader_display,
        )
        for tile in tile_tuple:
            display_cache_start = perf_counter()
            tile_key = tile_key_for(tile.view_state)
            cached = self.win.operation_evaluator.cached_montage_tile_by_key(tile_key)
            total_lookup_ms += (perf_counter() - display_cache_start) * 1000.0
            last_hit = cached is not None
            if cached is None:
                previous_payload = previous_payloads.get(tile_key)
                if previous_payload is None:
                    previous_payload = retained_store.resolve(
                        tile_key,
                        None if reuse_any_lod else selected_lod_factor,
                        tile.view_state,
                        shader_display=shader_display,
                    )
                if previous_payload is None or not _payload_compatible_with_tile(
                    previous_payload,
                    tile.view_state,
                    shader_display=shader_display,
                ):
                    missing_tiles.append(tile)
                else:
                    cached_tiles.append(_rendered_tile_from_previous_payload(tile, previous_payload))
            else:
                cached_tiles.append(_rendered_tile_from_cached_display(tile, cached))
        self._last_montage_display_cache_lookup_ms = total_lookup_ms
        self._last_montage_display_cache_hit = last_hit if tile_tuple else False
        if cached_tiles and reuse_any_lod:
            # Each cached resolve under the resident LOD policy is a pipeline
            # evaluation a display-LOD-driven rebuild did not have to run.
            self._montage_quality_pipeline_reruns_avoided = int(
                getattr(self, "_montage_quality_pipeline_reruns_avoided", 0) or 0
            ) + len(cached_tiles)
        return cached_tiles, missing_tiles

    def _try_update_montage_viewport_only(self) -> bool:
        """Retarget a persistent tiled session without restarting evaluation."""

        session = getattr(self, "_frame_session", None)
        if not self._frame_session_is_current(session):
            return False
        capabilities = image_view_backend_capabilities(self.win.img_view)
        try:
            display_mode = str(self.win.img_view.montageDisplayMode())
        except Exception:
            display_mode = ""
        retarget_policy = montage_viewport_retarget_policy(capabilities, display_mode)
        if not retarget_policy.enabled:
            return False

        view_state = self.win.view_state
        if view_state.montage_axis is None:
            return False
        viewport_plan = self._montage_viewport_plan(view_state)
        colormap_lut = self._evaluation_colormap_lut(
            view_state,
            shader_display=bool(capabilities.shader_windowing),
        )
        expected_key = frame_session_key(
            _document_key(self.win.document),
            view_state,
            viewport_plan,
            colormap_lut,
        )
        # Not the shared currency predicate: this compares the session against
        # the key re-derived from the *live* view state (semantic-match check).
        if session.key != expected_key:
            self._montage_live_layout_reflow = False
            return False
        self._montage_live_layout_reflow = False
        previous_plan = getattr(session, "plan", None)
        viewport_plan = self._retargeted_montage_viewport_plan(session, viewport_plan)
        self._remap_montage_rois_for_layout_reflow(previous_plan, viewport_plan.plan)

        # Hidden residency is keyed to the viewport that selected its near
        # payloads. Programmatic camera changes do not pass through the render
        # coordinator's interaction cancellation path, so cancel the queued
        # backend batches at the canonical viewport-retarget boundary before
        # a freshly installed scheduler can accidentally adopt them.
        cancel_speculative = getattr(self.win.img_view, "_cancel_vispy_speculative_work", None)
        if callable(cancel_speculative):
            cancel_speculative()

        additions, presentation_changed = session.retarget_viewport(
            view_range=viewport_plan.view_range,
            viewport_shape=viewport_plan.viewport_shape,
            plan=viewport_plan.plan,
            coverage_margin_tiles=retarget_policy.coverage_margin_tiles,
            near_margin_tiles=retarget_policy.near_margin_tiles,
            priority_focus=viewport_plan.priority_focus,
            priority_retarget_limit=max(1, len(tuple(getattr(session, "pending_tiles", ()) or ())) + len(tuple(viewport_plan.plan.tiles))),
        )
        memory_policy = self._memory_policy() if hasattr(self, "_memory_policy") else None
        session.frame_plan = self._montage_frame_planner().plan(
            target=FrameTarget(
                semantic_key=session.key,
                viewport_key=viewport_plan.view_range,
                presentation_key=(str(session.window_mode), normalize_bounds(getattr(session, "user_levels_override", None)), bool(getattr(session, "force_auto", False))),
                quality="exact-visible",
            ),
            view_state=view_state,
            display_shape=viewport_plan.plan.display_shape,
            backend_capabilities=capabilities,
            viewport_shape=viewport_plan.viewport_shape,
            view_range=viewport_plan.view_range,
            memory_policy=memory_policy,
            montage_plan=viewport_plan.plan,
        )
        if memory_policy is not None:
            session.tile_residency_budget_bytes = tile_residency_budget_bytes(memory_policy)
        # Camera-only changes must retarget the LOD decision immediately:
        # black/too-coarse tiles improve immediately, while already-presented
        # finer tiles stay put unless visible residency pressure requires a
        # quality demotion. Demand math only; reduction stays on worker lanes.
        lod_swap_ready = session.mark_ladder_swaps_for_viewport()
        self.retarget_frame_pipeline(session)
        required_tile_numbers = frozenset(session.required_tile_numbers())
        # ``pending_tiles`` no longer drives production scheduling: the frame
        # pipeline derives visible work from the lifecycle's required scope.
        # Keep the viewport-only cache/evaluation path on that same scope.
        # Coverage-margin misses are speculative and are owned by
        # ``schedule_near_viewport_montage_prefetch`` once visible work is
        # quiet; admitting them here leaves a dormant queue entry with no
        # pipeline consumer and later suppresses the cache lookup when the
        # tile actually becomes visible.
        additions = tuple(
            tile
            for tile in viewport_plan.prioritize_tiles(additions)
            if int(tile.montage_index) in required_tile_numbers
        )
        self._prune_stale_montage_tile_work(session)
        if not additions:
            if presentation_changed or lod_swap_ready:
                self.apply_montage_presentation(session)
            if session.pending_tiles and not _viewport_interaction_active(self):
                self.retarget_frame_pipeline(session)
                return True
            if session.pending_tiles:
                return True
            else:
                self._finish_frame_session_if_complete(session)
                schedule_near_viewport_montage_prefetch(self, session)
            return True
        additions_to_process = tuple(additions)
        budget = self._montage_callback_budget(
            "montage_viewport_update",
            interactive=_interactive_active(self),
            work_class="viewport_cached_additions",
            item_cap=max(1, len(additions_to_process)),
        )
        cached_tiles = []
        missing_tiles = []
        processed_additions = 0
        for tile in additions_to_process:
            display_cached, display_missing = self._resolve_montage_tiles_from_cache(
                (tile,),
                document=session.document,
                axis=session.montage_axis,
                colormap_lut=session.colormap_lut,
                shader_display=bool(getattr(session, "shader_display", False)),
            )
            cached_tiles.extend(display_cached)
            missing_tiles.extend(display_missing)
            processed_additions += 1
            byte_count = sum(montage_commit.rendered_tile_nbytes(rendered) for rendered in display_cached)
            budget.record_item(byte_count=byte_count)
            if budget.should_yield():
                break
        self._record_gui_budget(budget)
        remaining_additions = max(0, len(additions_to_process) - int(processed_additions))
        self._last_montage_viewport_deferred_additions = int(remaining_additions)
        queue_native_additions = render_lod.native_missing_tile_queue_required(
            str(getattr(session, "lod_policy_mode", "")),
            getattr(getattr(session, "lod_policy_decision", None), "demand", None),
        )
        if remaining_additions and queue_native_additions:
            self.win._montage_viewport_update_pending = True
            self._montage_viewport_continue_immediately = True
        session.tile_compute_cache_hits += len(cached_tiles)
        for rendered in cached_tiles:
            session.mark_materialized(rendered)
        self._queue_montage_cached_level_stats(session, cached_tiles, seed_if_empty=False)

        self._montage_cached_tiles_last_session = len(cached_tiles)
        self._montage_missing_tiles_last_session = len(missing_tiles)
        if presentation_changed or cached_tiles or lod_swap_ready:
            self.apply_montage_presentation(session)
        if cached_tiles:
            self._schedule_montage_cached_level_stats(session)
        if not missing_tiles:
            self._finish_frame_session_if_complete(session)
            schedule_near_viewport_montage_prefetch(self, session)
            return True
        missing_tiles = viewport_plan.prioritize_tiles(missing_tiles)
        if not queue_native_additions:
            self.win._montage_viewport_update_pending = False
            self._montage_viewport_continue_immediately = False
            session.stage_planning_deferred = bool(missing_tiles)
            session.stage_planning_async = False
            session.deferred_missing_tiles = tuple(missing_tiles)
            montage_commit.submit_deferred_stage_fan_in_plan(self, session, missing_tiles)
            self.retarget_frame_pipeline(session)
            return True

        if _viewport_interaction_active(self):
            stage_plan = montage_commit.hot_cached_stage_fan_in_plan(self, session.document, missing_tiles)
            if montage_commit.stage_fan_in_plan_has_existing_sources(stage_plan):
                montage_commit.merge_stage_fan_in_plan(session, stage_plan)
            queued = set(session.pending_tile_numbers())
            for tile in missing_tiles:
                index = int(tile.montage_index)
                if index not in queued:
                    _enqueue_session_pending_tile(session, tile)
                    queued.add(index)
            session.stage_planning_deferred = True
            session.stage_planning_async = False
            session.deferred_missing_tiles = tuple(missing_tiles)
            montage_commit.submit_deferred_stage_fan_in_plan(self, session, missing_tiles)
            self.retarget_frame_pipeline(session)
            return True

        stage_plan = montage_commit.build_stage_fan_in_plan(self, session.document, missing_tiles)
        montage_commit.merge_stage_fan_in_plan(session, stage_plan)
        queued = set(session.pending_tile_numbers())
        for tile in missing_tiles:
            index = int(tile.montage_index)
            if index not in queued:
                _enqueue_session_pending_tile(session, tile)
                queued.add(index)

        self.win.prefetch_evaluation_controller.cancel_prefetch()
        self.win.operation_evaluator.last_status = CacheStatusSnapshot(
            CacheStatus.COMPUTING,
            "Extending montage viewport",
        )
        self.show_frame_session_slow_overlay(session)
        montage_commit.submit_stage_tasks(self, session, stage_plan["stage_requests"])
        self.retarget_frame_pipeline(session)
        return True

    def _prune_stale_montage_tile_work(self, session: FrameSession) -> None:
        if not _viewport_interaction_active(self):
            return
        keep = {
            int(tile.montage_index)
            for tile in session.plan.tiles_intersecting(session.view_range, margin_tiles=2)
        }
        if not keep:
            return
        pruned_pending = session.prune_pending_tiles(keep)
        stale = (set(session.loading_tiles) - set(session.active_tile_requests)) - keep
        if stale:
            for index in sorted(stale):
                session.loading_tiles.discard(int(index))
                if 0 <= int(index) < len(session.tile_states):
                    session.tile_states[int(index)] = MontageTileState.UNLOADED
                    session.invalidate_tile_states()
        pruned = pruned_pending + len(stale)
        if pruned > 0:
            self._last_montage_pruned_tile_work = int(getattr(self, "_last_montage_pruned_tile_work", 0) or 0) + int(pruned)

    def _warn_montage_tiles_skipped(self, *, skipped_count: int, tile_bytes: int, budget_bytes: int, tile_shape) -> None:
        message = (
            f"Montage skipped {int(skipped_count)} tile(s) because each tile would allocate "
            f"{format_bytes(int(tile_bytes))}, over the visible render budget of {format_bytes(int(budget_bytes))}. "
            f"Tile shape is {tuple(int(size) for size in tile_shape)}. Zoom in, crop/range the image axes, "
            "or increase Performance > Render Memory Budget."
        )
        show_status_message(self.win, message, timeout=8000)
        warning_key = (int(skipped_count), int(tile_bytes), int(budget_bytes), tuple(int(size) for size in tile_shape))
        if getattr(self, "_last_montage_skip_warning_key", None) == warning_key:
            return
        self._last_montage_skip_warning_key = warning_key
        try:
            Qt.QtWidgets.QMessageBox.warning(self.win, "Montage tiles skipped", message)
        except Exception as exc:
            handle_ui_exception("montage skipped warning", exc)

    def _retargeted_montage_viewport_plan(
        self,
        session,
        viewport_plan: MontageViewportPlan,
        *,
        previous_viewport_shape=None,
        focus=None,
    ) -> MontageViewportPlan:
        continuity = getattr(self.win, "_viewport_continuity_transaction", lambda: None)()
        skip_remap = bool(
            continuity is not None
            and not continuity.released
            and continuity.range_applied
            and continuity.view_range is not None
        )

        viewport_controller = getattr(self.win.img_view, "viewport_controller", None)
        current_range = viewport_plan.view_range
        intent = montage_viewport_intent(viewport_controller, current_range)
        camera_focus = focus if focus is not None else (None if current_range is None else _montage_priority_focus(self, current_range))
        reflow = retarget_montage_viewport_plan(
            getattr(session, "plan", None),
            viewport_plan,
            previous_viewport_shape or getattr(session, "viewport_shape", viewport_plan.viewport_shape),
            fit_locked=intent.fit_locked,
            auto_active=intent.auto_active,
            skip_remap=skip_remap,
            focus=camera_focus,
        )
        if viewport_controller is not None and reflow.last_auto_view_range is not None:
            viewport_controller.last_auto_view_range = reflow.last_auto_view_range
        if reflow.view_range_to_apply is not None:
            self._set_montage_view_range(reflow.view_range_to_apply)
        if skip_remap:
            complete_continuity = getattr(self.win, "_complete_viewport_continuity_if_settled", None)
            if callable(complete_continuity):
                complete_continuity()
        view_range = reflow.viewport_plan.view_range
        priority_focus = focus if focus is not None else (None if view_range is None else _montage_priority_focus(self, view_range))
        return replace(
            reflow.viewport_plan,
            priority_focus=priority_focus,
        )

    def _remap_montage_rois_for_layout_reflow(self, previous_plan, next_plan) -> None:
        if previous_plan is None or next_plan is None:
            return
        if getattr(previous_plan, "geometry", None) == getattr(next_plan, "geometry", None):
            return
        img_view = getattr(self.win, "img_view", None)
        selections_fn = getattr(img_view, "roiSelections", None)
        if not callable(selections_fn):
            return
        selections = tuple(selections_fn() or ())
        remapped_selections = remap_montage_roi_selections(previous_plan, next_plan, selections)
        updates = [
            (previous, current.geometry)
            for previous, current in zip(selections, remapped_selections)
            if current.geometry != previous.geometry
        ]
        if not updates:
            return
        set_geometry = getattr(img_view, "_set_roi_geometry", None)
        if callable(set_geometry):
            for selection, geometry in updates:
                set_geometry(selection.id, geometry, emit=True, sync_item=True)
        else:
            selected_id = getattr(getattr(self.win, "roi_store", None), "selected_id", None)
            setter = getattr(img_view, "setRoiSelections", None)
            if callable(setter):
                setter(remapped_selections, selected_id=selected_id)
        roi_store = getattr(self.win, "roi_store", None)
        if roi_store is not None:
            selected_id = getattr(roi_store, "selected_id", None)
            self.win.roi_store = roi_store.replace_all(selections_fn()).select(selected_id)
            dock = getattr(self.win, "inspection_dock", None)
            set_rois = getattr(dock, "set_rois", None)
            if callable(set_rois):
                set_rois(self.win.roi_store.selections)
        schedule_refresh = getattr(self.win, "_schedule_refresh_inspection_dock", None)
        if callable(schedule_refresh):
            schedule_refresh("montage-layout-reflow")

    def apply_montage_presentation(self, session) -> None:
        if not self._frame_session_is_current(session):
            return
        pipeline = getattr(session, "pipeline", None)
        if pipeline is None:
            # Commit-only test/teardown sessions do not own a live pipeline.
            self.commit_frame_session_presentation(session)
        else:
            pipeline.effects.request_presentation()
        if (
            getattr(session, "show_loading_overlays", False)
            and not session.visible_plan_complete()
            and (session.pending_tiles or session.loading_tiles or session.active_tile_requests or session.stage_fan_in.attached_requests)
        ):
            self.win.img_view.setImageStale(True)
            self.win.img_view.setEvaluationOverlay(True, "Updating image frame...")

    def commit_frame_session_presentation(self, session) -> None:
        if not self._frame_session_is_current(session):
            return
        # One effects instance per session: it owns the in-flight rung guards
        # and the presentation gate flag. A throwaway instance would fork
        # that state (exactly the parallel-bookkeeping defect ADR 0051 bans).
        # Commit-only callers without a live pipeline (tests, teardown) get a
        # transient instance — they never submit rungs, so no guard state is
        # forked.
        pipeline = getattr(session, "pipeline", None)
        effects = pipeline.effects if pipeline is not None else FramePipelineEffects(self, session)
        effects.commit_pending_session()

    def apply_ready_montage_display(self, session) -> None:
        if not self._frame_session_is_current(session):
            return
        has_commit_work = (
            getattr(session, "dirty_rects", None)
            or getattr(session, "dirty_tiles", None)
            or getattr(session, "dirty_payloads", None)
            or getattr(session, "pending_payload_upserts", None)
            or getattr(session, "pending_removals", None)
            or (
                session.has_pending_level_update()
                and session.has_stale_level_presentations()
            )
        )
        if not has_commit_work:
            return
        session.final_commit_pending = True
        session.flush_pending = True
        self.apply_montage_presentation(session)

    def _finish_frame_session_if_complete(self, session) -> bool:
        if not self._frame_session_is_current(session):
            return False
        if not session.is_complete():
            return False
        self._settle_montage_visible_plan_if_complete(session)
        # Montage level phasing: rough -> hold -> refined. Refined evidence is
        # queued after visible settlement and admitted through the kernel.
        if bool(getattr(session, "shader_display", False)):
            # Payload evidence has already closed the rough pass. The
            # statistics-only semantic owner now refines the full population;
            # target display payloads must not enter a second rough/refinement
            # owner.
            self._schedule_semantic_level_evidence(session)
        else:
            self._queue_montage_current_level_evidence(session)
            self._queue_montage_final_level_refinements(session)
            self._schedule_montage_refined_level_stats(session)
        return True

    def _settle_montage_visible_plan_if_complete(self, session) -> bool:
        if not self._frame_session_is_current(session):
            return False
        if not session.visible_plan_complete():
            return False
        session.show_loading_overlays = False
        self._stop_frame_session_slow_overlay()
        self.win.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.READY, "Image frame ready")
        self.win.img_view.setImageStale(False)
        self.win.img_view.setEvaluationOverlay(False)
        if hasattr(self.win.img_view, "clearMontageTileOverlays"):
            self.win.img_view.clearMontageTileOverlays()
        return True

    def _sync_committed_montage_geometry(self, geometry, *, semantic_commit: bool = True) -> None:
        self.display_geometry = geometry
        frame = getattr(self.win, "_committed_display_frame", None)
        if (
            bool(semantic_commit)
            and frame is not None
            and frame.geometry != geometry
        ):
            self._set_committed_display_frame(replace(frame, geometry=geometry, scene=None))
        refresh_hover = getattr(self, "_refresh_hover_after_display_commit", None)
        if callable(refresh_hover):
            refresh_hover()
        if bool(semantic_commit):
            schedule_refresh = getattr(self.win, "_schedule_refresh_inspection_dock", None)
            if callable(schedule_refresh):
                schedule_refresh("montage-semantic-commit")

    def _notify_file_session_montage_committed(self) -> None:
        restore = getattr(self.win, "_schedule_viewport_continuity_when_ready", None)
        if callable(restore):
            restore()
        if not bool(getattr(self.win, "_file_session_roi_refresh_pending", False)):
            return
        schedule_roi_refresh = getattr(self.win, "_schedule_file_session_roi_refresh", None)
        if callable(schedule_roi_refresh):
            schedule_roi_refresh("montage-semantic-commit")

    def _should_publish_montage_level_metadata(self, session, stats: MontageLevelStats) -> bool:
        # Histogram metadata is independent from whether display levels are
        # allowed to move.  Publishing better semantic stats lets absolute mode
        # update the histogram while preserving numeric levels, and lets
        # relative mode remap through WindowLevelController.
        #
        # Reduced-input display montages (opaque/FFT and reduced-LOD scalar
        # fits) present entirely through preview payloads and never populate
        # `rendered_tiles`, so gating on it alone left them stranded at default
        # (0, 1) levels with an empty histogram.  Their rough sampled stats are
        # the correct first-pass source; refinement improves them later.
        if not session.rendered_tiles and not getattr(session, "display_tile_payloads", None):
            return False
        bounds = stats.bounds
        bounds = normalize_bounds(bounds)
        if bounds is None:
            return False
        applied = getattr(session, "applied_level_source", None)
        same_semantic = getattr(applied, "semantic_key", None) == session.level_key
        if not same_semantic:
            return True
        applied_rank = int(getattr(applied, "rank", 0) or 0)
        if int(stats.rank) > applied_rank:
            return True
        if int(getattr(stats, "evidence_quality", 0) or 0) > int(getattr(applied, "evidence_quality", 0) or 0):
            return True
        if (
            bool(getattr(session, "shader_display", False))
            and not bool(getattr(session, "first_pass_histogram_published", False))
            and len(getattr(stats, "source_indices", ()) or ())
            > int(getattr(applied, "source_count", 0) or 0)
        ):
            # First-pass shader levels intentionally track bounded coverage
            # growth. Histogram plotting remains coalesced until the physical
            # pass is complete.
            return True
        # Coverage growth inside one rank is intentionally coalesced. A
        # 60-source fill otherwise rebuilds the histogram plot after every
        # small evidence batch, starving paints/input while showing a sequence
        # of transient partial distributions. Rank/quality transitions still
        # publish immediately (first rough pixels, complete population,
        # refined evidence), so correctness and useful progress stay visible.
        return False

    def _note_montage_level_source_applied(self, session, source, *, explicit: bool) -> None:
        if source is None:
            return
        # Store partial as well as complete semantic sources.  The presentation
        # controller expands same-key sources monotonically and protects explicit
        # user locks, so storing partial coverage is safe and prevents fallback
        # to stale tiny placeholder ranges.
        session.applied_level_source = source

    def _is_current_frame_session(self, session_id, key) -> bool:
        return _session_token_is_current(getattr(self, "_frame_session", None), session_id, key)

    def _frame_session_is_current(self, session) -> bool:
        """Shared staleness predicate for a montage session object.

        Delegates to :meth:`_is_current_frame_session` so tests that stub the
        (session_id, key) predicate keep controlling both forms. Callbacks that
        captured a raw ``(session_id, key)`` pair call the canonical predicate
        directly.
        """
        if session is None:
            return False
        return self._is_current_frame_session(session.session_id, session.key)

    def _montage_tile_shape(self, view_state):
        primary_axis, secondary_axis = view_state.image_axes
        primary_indices = view_state.axis_range_indices[primary_axis]
        secondary_indices = view_state.axis_range_indices[secondary_axis]
        return (
            len(primary_indices) if primary_indices is not None else int(view_state.shape[primary_axis]),
            len(secondary_indices) if secondary_indices is not None else int(view_state.shape[secondary_axis]),
        )

    def _current_montage_global_view_range(self):
        try:
            view_range = self.win.img_view.getView().viewRange()
        except Exception:
            return None
        return (
            (float(view_range[0][0]), float(view_range[0][1])),
            (float(view_range[1][0]), float(view_range[1][1])),
        )


def _copy_view_range(view_range):
    return (
        (float(view_range[0][0]), float(view_range[0][1])),
        (float(view_range[1][0]), float(view_range[1][1])),
    )


def _montage_viewport_shape_from_qt_size_tuple(size):
    if size is None:
        return None
    try:
        width, height = size
        return (max(1, int(round(float(height)))), max(1, int(round(float(width)))))
    except Exception:
        return None


def _montage_full_view_range(montage):
    height = int(montage.rows) * int(montage.tile_height) + max(0, int(montage.rows) - 1) * int(montage.gap)
    width = int(montage.columns) * int(montage.tile_width) + max(0, int(montage.columns) - 1) * int(montage.gap)
    return ((0.0, float(max(1, width))), (0.0, float(max(1, height))))


def _initial_montage_planning_view_range(plan, viewport_shape, viewport_controller):
    """Measure startup LOD from layout intent before a backend image exists."""

    if plan is None:
        return None
    intent = montage_viewport_intent(viewport_controller, None)
    if intent.fit_locked:
        return plan_full_view_range(plan)
    return square_montage_fit_view_range(plan, viewport_shape)


def _montage_tile_layer_placeholder(session) -> np.ndarray:
    height, width = (max(1, int(value)) for value in session.plan.display_shape)
    if bool(getattr(session, "rgb", False)):
        base = np.zeros((1, 1, 3), dtype=np.uint8)
        return np.broadcast_to(base, (height, width, 3))
    base = np.zeros((1, 1), dtype=np.float32)
    return np.broadcast_to(base, (height, width))


def _enqueue_session_pending_tile(session, tile) -> None:
    enqueue = getattr(session, "enqueue_pending_tile", None)
    if callable(enqueue):
        enqueue(tile)
        return
    session.pending_tiles.append(tile)


def _rendered_tile_from_previous_payload(tile, payload) -> RenderedTile:
    semantic = None if payload.semantic_data is None else np.asarray(payload.semantic_data)
    image = semantic if semantic is not None else np.asarray(payload.image)
    semantic_histogram = (
        None
        if getattr(payload, "semantic_histogram_data", None) is None
        else np.asarray(payload.semantic_histogram_data)
    )
    histogram = semantic_histogram if semantic_histogram is not None else (
        None if payload.histogram_data is None else np.asarray(payload.histogram_data)
    )
    slab_shape = tuple(getattr(payload, "source_shape", None) or image.shape)
    return RenderedTile(
        tile=tile,
        image=image,
        histogram_data=histogram,
        eval_ms=0.0,
        slab_shape=slab_shape,
        slab_nbytes=int(payload.nbytes),
        shader_mapping=getattr(payload, "shader_mapping", None),
        texture_kind=getattr(payload, "texture_kind", None),
        semantic_data=semantic,
        semantic_histogram_data=semantic_histogram,
        lod_source_data=getattr(payload, "lod_source_data", None),
        lod=getattr(payload, "lod", None),
        level_data=getattr(payload, "level_data", None),
        level_stats=getattr(payload, "level_stats", None),
        quality=str(getattr(payload, "quality", "exact") or "exact"),
    )


def _rendered_tile_from_cached_display(tile, cached) -> RenderedTile:
    if hasattr(cached, "bind"):
        return cached.bind(tile)
    if hasattr(cached, "payload"):
        return cached.payload().bind(tile)
    image = np.asarray(cached.data)
    histogram = None if getattr(cached, "histogram_data", None) is None else np.asarray(cached.histogram_data)
    return RenderedTile(
        tile=tile,
        image=image,
        histogram_data=histogram,
        eval_ms=0.0,
        slab_shape=tuple(image.shape),
        slab_nbytes=int(image.nbytes),
        shader_mapping=getattr(cached, "shader_mapping", None),
        texture_kind=getattr(cached, "texture_kind", None),
        semantic_data=getattr(cached, "semantic_data", None),
        semantic_histogram_data=getattr(cached, "semantic_histogram_data", None),
        lod_source_data=getattr(cached, "lod_source_data", None),
        lod=getattr(cached, "lod", None),
        level_data=getattr(cached, "level_data", None),
        level_stats=getattr(cached, "level_stats", None),
        quality=str(getattr(cached, "quality", "exact") or "exact"),
    )


def _rendered_tile_from_evaluation_result(tile, result) -> RenderedTile:
    value = result.value
    return RenderedTile(
        tile=tile,
        image=value.data,
        histogram_data=value.histogram_data,
        eval_ms=float(getattr(result, "eval_ms", 0.0) or 0.0),
        slab_shape=tuple(getattr(result, "slab_shape", np.shape(value.data))),
        slab_nbytes=getattr(result, "slab_nbytes", None),
        shader_mapping=getattr(value, "shader_mapping", None),
        texture_kind=getattr(value, "texture_kind", None),
        semantic_data=getattr(value, "semantic_data", None),
        semantic_histogram_data=getattr(value, "semantic_histogram_data", None),
        lod_source_data=getattr(value, "lod_source_data", None),
        lod=getattr(value, "lod", None),
        level_data=getattr(value, "level_data", None),
        level_stats=getattr(value, "level_stats", None),
        quality=str(getattr(value, "quality", "exact") or "exact"),
    )


def _shared_preview_batch_key(session, tile, demand) -> tuple:
    preview_level = render_effects.preview_evaluation_level(session, demand)
    view_state = tile.view_state
    image_axes = tuple(int(axis) for axis in view_state.image_axes)
    display_ranges = tuple(
        (
            axis,
            None
            if view_state.axis_range_indices[axis] is None
            else tuple(int(index) for index in view_state.axis_range_indices[axis]),
        )
        for axis in image_axes
    )
    montage_axis = getattr(session, "montage_axis", None)
    non_display_slices = tuple(
        (axis, int(index))
        for axis, index in enumerate(tuple(view_state.slice_indices))
        if axis not in image_axes and (montage_axis is None or axis != int(montage_axis))
    )
    return (
        int(preview_level),
        tuple(int(value) for value in factor_xy_for_level(demand, int(preview_level))),
        image_axes,
        display_ranges,
        non_display_slices,
    )


def _shared_preview_candidate_tiles(session):
    for candidate in tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ()):
        tile_number = int(candidate.montage_index)
        if tile_number in getattr(session, "rendered_tiles", {}):
            continue
        existing = getattr(session, "display_tile_payloads", {}).get(tile_number)
        if existing is not None:
            continue
        yield candidate


def _preview_tile_shape(session, demand) -> tuple[int, int]:
    factor_x, factor_y = factor_xy_for_level(demand, render_effects.preview_evaluation_level(session, demand))
    tile_h, tile_w = (max(1, int(value)) for value in tuple(session.plan.tile_shape)[:2])
    return (max(1, int(np.ceil(tile_h / max(1, factor_y)))), max(1, int(np.ceil(tile_w / max(1, factor_x)))))


def _preview_floor_blocks_exact_submission(session, tile) -> bool:
    if tile is None:
        return False
    scope = set(int(value) for value in getattr(session, "lod_preview_floor_scope", set()) or set())
    tile_number = int(tile.montage_index)
    if tile_number not in scope:
        return False
    payloads = dict(getattr(getattr(session, "tile_presentation_state", None), "payloads", {}) or {})
    payload = payloads.get(tile_number)
    return str(getattr(payload, "quality", "")) not in {"preview", "exact"}


def _visible_cpu_tile_layer_backlog_pending(window, session) -> bool:
    capabilities = image_view_backend_capabilities(getattr(window.win, "img_view", None))
    if bool(capabilities.shader_windowing):
        return False
    return bool(
        getattr(session, "dirty_payloads", None)
        or getattr(session, "pending_payload_upserts", None)
        or getattr(session, "pending_removals", None)
        or (session.has_pending_level_update() and session.has_stale_level_presentations())
    )


def _latency_feedback(window):
    return getattr(window.win, "latency_feedback", None)


def _interactive_active(window) -> bool:
    coordinator = getattr(window.win, "render_coordinator", None)
    return bool(
        coordinator is not None and getattr(coordinator, "interactive_active", False)
        or _viewport_interaction_active(window)
    )


def _should_defer_montage_side_panels(window, session) -> bool:
    return bool(getattr(session, "defer_side_panels", False) or _viewport_interaction_active(window))


def _viewport_interaction_active(window) -> bool:
    return bool(getattr(window.win, "_viewport_interaction_active", False))
