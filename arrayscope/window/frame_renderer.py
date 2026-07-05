"""Frame render orchestration for ArrayScope windows.

The frame path owns the single visible image surface for every image view.  A
single slice and a multi-region montage both become one semantic frame with
region payloads; backends differ only in physical presentation mechanics.
"""

from __future__ import annotations

import os
import sys

from collections import deque
from dataclasses import replace
from time import monotonic, perf_counter
from types import SimpleNamespace

import numpy as np
import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception
from arrayscope.core.cache_status import CacheStatus, CacheStatusSnapshot
from arrayscope.core.compute_policy import ComputeLane
from arrayscope.core.gui_callback_budget import GuiCallbackBudget, should_yield_after_item
from arrayscope.core.memory_budget import estimate_display_image_bytes, format_bytes
from arrayscope.core.scheduler import FrameTarget
from arrayscope.core.view_state import ChannelMode
from arrayscope.core.work_graph import WorkItem, WorkLane, complete_inline_work as _complete_inline_work
from arrayscope.display.frame_planner import FramePlanner
from arrayscope.display.geometry import DisplayGeometry, display_geometry_coordinates_equal
from arrayscope.display.imageview2d import MontageTileOverlay
from arrayscope.display.montage import (
    MontageTileState,
    RenderedTile,
    make_montage_plan,
    montage_rect_for_viewport,
)
from arrayscope.display.slice_engine import DisplayImage, make_image_from_slab, make_shader_image_from_slab
from arrayscope.display.shader_mapping import apply_scale as apply_shader_scale, extract_component
from arrayscope.display.viewport import ViewportMode, ViewportPolicy, view_ranges_near
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.operations.evaluator import EvaluationResult, _document_key, evaluate_image_snapshot, stage_document_key
from arrayscope.operations.chunked_stage import materialize_stage_candidate_chunked, stage_materialization_allowed_chunk_axes
from arrayscope.operations.stage_fanin import StageFanInState
from arrayscope.operations.slabs import (
    evaluate_slab_from_stage,
    plan_slab,
    request_for_image,
)
from arrayscope.ui.toasts import show_revert_action, show_status_message
from arrayscope.window.evaluation_controller import EvalPriority
from arrayscope.window.montage_backend import choose_montage_backend
from arrayscope.display.model.montage_levels import (
    MontageLevelStats,
    MontageLevelTracker,
    montage_level_key,
    provisional_tile_level_stats,
    sample_tile_level_stats,
)
from arrayscope.display.lod import LOD_POLICY_RESIDENT
from arrayscope.presentation.dispatch import derive_montage_dispatch
from arrayscope.display.pyramid import PyramidCache, preview_level_for_tile_shape
from arrayscope.window.montage_payload_cache import (
    payload_lod_matches as _payload_lod_matches,
    payload_compatible_with_tile as _payload_compatible_with_tile,
    previous_tiled_payloads as _previous_tiled_payloads,
    previous_tiled_payloads_by_base_source as _previous_tiled_payloads_by_base_source,
    RetainedTiledPayloadStore,
)
from arrayscope.window.montage_prefetch import schedule_near_viewport_montage_prefetch
from arrayscope.window.montage_viewport import (
    MontageViewportPlan,
    effective_montage_columns,
    montage_session_key,
    montage_tile_semantic_key,
    montage_viewport_intent,
    montage_viewport_retarget_policy,
    montage_viewport_update_delay_ms as _montage_viewport_update_delay_ms,
    prioritize_montage_tiles,
    remap_montage_roi_selections,
    retarget_montage_viewport_plan,
    square_montage_fit_view_range,
)
from arrayscope.window import montage_lod
from arrayscope.window.montage_lod import (
    admit_ingest_reduction as _admit_ingest_reduction,
    admit_preview_reduction as _admit_preview_reduction,
)
from arrayscope.window.montage_session import MontageRenderSession
from arrayscope.window.render_contract import (
    montage_work_token as _montage_work_token,
    montage_work_token_is_current as _montage_work_token_is_current,
    session_token_is_current as _session_token_is_current,
)
from arrayscope.display.planning import LevelSourceRank, decide_presentation, fallback_level_source, normalize_bounds
from arrayscope.display.model.commit import CommitKind, DisplayPayload, PresentationInput
from arrayscope.display.model.frame import TiledValueSource
from arrayscope.window.display_presenter import tile_residency_budget_bytes


MONTAGE_VERY_SLOW_UPLOAD_MS = 100.0
MONTAGE_AUTOFIT_VISIBLE_FRACTION = 0.80
MONTAGE_LEVEL_STATS_COMMIT_BATCH = 8
MONTAGE_LEVEL_STATS_BACKGROUND_BATCH = 4
MONTAGE_LEVEL_STATS_BACKGROUND_BUDGET_MS = 4.0


class FrameRenderMixin:
    def _interactive_frame_cache_hit(self) -> bool:
        view_state = getattr(self.win, "view_state", None)
        if view_state is None or view_state.image_axes is None:
            return False
        if view_state.montage_axis is None:
            return False
        evaluator = getattr(self.win, "operation_evaluator", None)
        if evaluator is None:
            return False
        shader_display = bool(image_view_backend_capabilities(self.win.img_view).shader_windowing)
        colormap_lut = self._evaluation_colormap_lut(view_state, shader_display=shader_display)
        viewport_plan = self._montage_viewport_plan(view_state)
        expected_key = montage_session_key(
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
        if view_state is None or view_state.image_axes is None or view_state.montage_axis is None:
            return False
        frame = getattr(self.win, "_committed_display_frame", None)
        frame_key = None if frame is None else getattr(frame, "key", None)
        return getattr(frame_key, "request_key", None) != self._montage_session_key_for_view(view_state)

    def _montage_session_key_for_view(self, view_state):
        capabilities = image_view_backend_capabilities(self.win.img_view)
        shader_display = bool(capabilities.shader_windowing)
        colormap_lut = self._evaluation_colormap_lut(view_state, shader_display=shader_display)
        viewport_plan = self._montage_viewport_plan(view_state)
        return montage_session_key(
            _document_key(self.win.document),
            view_state,
            viewport_plan,
            colormap_lut,
        )

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
        return self._montage_backend_policy(geometry, data).backend == "tile_layer"

    def _retained_tiled_payload_store(self) -> RetainedTiledPayloadStore:
        store = getattr(self, "_montage_retained_tiled_payloads", None)
        if not isinstance(store, RetainedTiledPayloadStore):
            store = RetainedTiledPayloadStore()
            self._montage_retained_tiled_payloads = store
        return store

    def _montage_backend_policy(self, geometry, data):
        return choose_montage_backend(
            geometry,
            data,
            renderer_backend=image_view_backend_capabilities(self.win.img_view).name,
        )

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
        current_range = view_range if view_range is not None else (
            pending_restore_range
            if pending_restore_range is not None
            else (
                self._current_montage_global_view_range()
                if getattr(self.win.img_view, "image", None) is not None
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
        active_continuity_range = getattr(self.win, "_active_viewport_continuity_range", lambda: None)()
        if active_continuity_range is not None:
            restore_viewport_shape = getattr(self.win, "_restore_viewport_continuity_shape_after_layout", None)
            if callable(restore_viewport_shape):
                restore_viewport_shape()
            self._set_montage_view_range(active_continuity_range)
            self._schedule_montage_viewport_update(delay_ms=0)
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
        self._schedule_montage_viewport_update(delay_ms=0)

    def _current_montage_resize_focus(self, view_range) -> tuple[float, float] | None:
        return _montage_priority_focus(self, view_range)

    def _retarget_montage_resize_camera(
        self,
        *,
        previous_viewport_size=None,
        base_view_range=None,
        resize_focus=None,
    ) -> MontageViewportPlan | None:
        session = getattr(self, "_montage_session", None)
        if not self._montage_session_is_current(session):
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
        expected_key = montage_session_key(
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
        session = getattr(self, "_montage_session", None)
        if not self._montage_session_is_current(session):
            return False
        capabilities = image_view_backend_capabilities(self.win.img_view)
        try:
            display_mode = str(self.win.img_view.montageDisplayMode())
        except Exception:
            display_mode = ""
        if not montage_viewport_retarget_policy(capabilities, display_mode).enabled:
            return False
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
            memory_policy=self._memory_policy() if hasattr(self, "_memory_policy") else None,
            montage_plan=viewport_plan.plan,
        )
        lod_swap_ready = session.refresh_lod_for_viewport()
        if getattr(session, "pending_lod_requests", None):
            self._schedule_montage_lod_materializations(session)
        if presentation_changed or lod_swap_ready:
            self._commit_montage_resize_presentation_retarget(session)
        return True

    def _commit_montage_resize_presentation_retarget(self, session) -> None:
        if bool(getattr(self, "_montage_presentation_commit_active", False)):
            self._schedule_montage_presentation_commit(session, force=True)
            return
        self._commit_montage_session_presentation(session, force=True)

    def update_image_view(self, *, force_autolevel: bool = False, defer_side_panels: bool = False):
        for attribute in (
            "_last_montage_viewport_plan_ms",
            "_last_montage_cache_resolve_ms",
            "_last_montage_stage_plan_ms",
            "_last_montage_session_setup_ms",
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
            viewport_plan = self._montage_viewport_plan(view_state)
            viewport_shape = viewport_plan.viewport_shape
            tile_shape = viewport_plan.tile_shape
            plan = viewport_plan.plan
        self._montage_live_layout_reflow = False
        previous_session_plan = getattr(getattr(self, "_montage_session", None), "plan", None)
        self._remap_montage_rois_for_layout_reflow(previous_session_plan, plan)
        current_range = viewport_plan.view_range
        canvas_rect = montage_rect_for_viewport(plan, view_range=current_range, viewport_shape=viewport_shape)
        display_tiles = viewport_plan.candidate_tiles(margin_tiles=0)
        candidate_tiles = viewport_plan.candidate_tiles(
            margin_tiles=1 if viewport_plan.persistent_tile_residency else 0,
            prioritize=True,
        )
        shader_display = viewport_plan.shader_display
        output_dtype = np.uint8 if view_state.channel == ChannelMode.COMPLEX and not shader_display else getattr(document.base_data, "dtype", np.dtype(float))
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
        if self._maybe_retarget_montage_session(
            getattr(self, "_montage_session", None),
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
        stage_plan_start = perf_counter()
        # Interaction fast path (ADR 0051 rule 4 at the architecture level):
        # during a scrub/pan burst, stage planning is *deferred, superseded
        # work* — mid-burst steps present pyramid floors and cached tiles
        # only, and the full plan (plan_slab per tile + stage materializer
        # claims, ~30 ms synchronous) runs once, for the step the user lands
        # on.  Superseded steps never touch the stage materializer at all,
        # so there are no claims to repair.  Native policy has no floors to
        # carry the screen, and the first session of a document has a user
        # actively waiting: both plan inline.
        previous_session = getattr(self, "_montage_session", None)
        defer_stage_planning = bool(
            missing_tiles
            and _viewport_interaction_active(self)
            and self._montage_lod_policy_mode() == LOD_POLICY_RESIDENT
            and previous_session is not None
            # Only a predecessor that actually committed montage content can
            # carry the screen through the burst (floors/retained payloads).
            # A first-ever montage build has a user staring at nothing —
            # plan inline.  Same axis: an axis change is a new montage, not a
            # scrub step.
            and bool(getattr(previous_session, "display_committed", False))
            and getattr(previous_session, "montage_axis", None) == axis
            and not os.environ.get("ARRAYSCOPE_DISABLE_SCRUB_FASTPATH")
        )
        if defer_stage_planning:
            stage_plan = _deferred_stage_plan_stub()
            self._montage_stage_plans_deferred = (
                int(getattr(self, "_montage_stage_plans_deferred", 0) or 0) + 1
            )
        else:
            stage_plan = self._plan_montage_stages(document, missing_tiles)
        self._last_montage_stage_plan_ms = (perf_counter() - stage_plan_start) * 1000.0
        session_setup_start = perf_counter()
        pending_tiles = (
            []
            if defer_stage_planning
            else [tile for tile in missing_tiles if int(tile.montage_index) not in stage_plan["waiting_indices"]]
        )
        session_key = montage_session_key(_document_key(document), view_state, viewport_plan, colormap_lut)
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
        session_id = int(getattr(self, "_montage_session_id", 0)) + 1
        self._montage_session_id = session_id
        lod_policy_mode = self._montage_lod_policy_mode()
        lod_preview_level = (
            preview_level_for_tile_shape(plan.tile_shape) if lod_policy_mode == LOD_POLICY_RESIDENT else 0
        )
        session = MontageRenderSession(
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
            colormap_lut=colormap_lut,
            viewport_shape=viewport_shape,
            view_range=current_range,
            output_dtype=np.dtype(output_dtype),
            rgb=view_state.channel == ChannelMode.COMPLEX,
            window_mode=window_mode,
            force_auto=force_auto,
            visible_tiles=tuple(display_tiles),
            rendered_tiles={int(rendered.tile.montage_index): rendered for rendered in cached_tiles},
            loading_tiles={int(tile.montage_index) for tile in missing_tiles},
            skipped_tiles={int(tile.montage_index) for tile in skipped_tiles},
            pending_tiles=list(pending_tiles),
            stage_fan_in=StageFanInState(
                tile_stage_keys=stage_plan["tile_stage_keys"],
                tile_stage_plans=stage_plan["tile_stage_plans"],
                tile_stage_candidates=stage_plan["tile_stage_candidates"],
                waiting_tiles=stage_plan["stage_waiting_tiles"],
                attached_requests=stage_plan["attached_stage_keys"],
                values=stage_plan["stage_values"],
                lead_warmups=stage_plan["lead_stage_warmups"],
            ),
            defer_side_panels=bool(defer_side_panels),
            applied_level_source=(
                pending_auto_level_source
                if pending_auto_level_source is not None
                else (None if previous_frame is None else fallback_level_source(previous_frame))
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
            lod_native_reason=montage_lod.native_policy_reason_for_renderer(self),
            lod_preview_pyramid=(self._montage_lod_preview_pyramid() if lod_preview_level else None),
            lod_preview_level=lod_preview_level,
            lod_pyramid=(
                self._montage_lod_pyramid() if lod_policy_mode == LOD_POLICY_RESIDENT else None
            ),
        )
        session.shader_display = bool(shader_display)
        session.stage_planning_deferred = bool(defer_stage_planning)
        session.deferred_missing_tiles = tuple(missing_tiles) if defer_stage_planning else ()
        # The dying session's planned-but-undrained LOD requests hold
        # singleflight claims in the shared pyramid; scrubbing back to the
        # same slice would find those levels permanently claimed (stale
        # wrong-LOD tiles).  Balance them before the replacement takes over.
        montage_lod.release_session_claims(getattr(self, "_montage_session", None))
        # Backend slots outlive sessions (persistent tile residency), so the
        # identity ground truth from the last report stays valid — but a fresh
        # session started with last_presented_identities EMPTY, blind to
        # inherited stale slots until its own first report, whose repairs only
        # ran on the commit AFTER that (field defect 2026-07-05 #3: sid 68
        # rebuilt on top of 29 stale slots and settled without healing them).
        # Inherit the map; tiles absent from the new plan fall out naturally
        # (no current payload → mismatch scan skips them).
        dying_session = getattr(self, "_montage_session", None)
        if dying_session is not None:
            inherited = getattr(dying_session, "last_presented_identities", None)
            if inherited:
                session.last_presented_identities = dict(inherited)
        self._montage_session = session
        # A viewport-update token armed for the dying session would make every
        # later _run_montage_viewport_update bail as stale — a dead
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
        self._last_montage_session_setup_ms = (perf_counter() - session_setup_start) * 1000.0
        initial_commit_start = perf_counter()
        try:
            self._commit_montage_session_presentation(session, force=True)
        except MemoryError as exc:
            show_status_message(self.win, str(exc), timeout=6000)
            return
        finally:
            self._last_montage_initial_commit_ms = (perf_counter() - initial_commit_start) * 1000.0
        if session.is_complete():
            self._finish_montage_session_if_complete(session)
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
            self._schedule_montage_session_slow_overlay(session)
        self._schedule_montage_cached_level_stats(session)
        if defer_stage_planning:
            self._schedule_deferred_montage_planning(session)
        else:
            self._schedule_montage_stage_jobs(session, stage_plan["stage_requests"])
            self._schedule_montage_attached_stage_waits(session)
            self._schedule_montage_tiles(session)

    def _maybe_retarget_montage_session(
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

        ADR 0051 P2 (session-rebirth cost): an index-window scrub step with
        identical layout geometry, viewport, and presentation inputs reuses
        the session object — the backend acknowledgement state and drawn
        payloads survive, and the budgeted flush machinery converges the
        content.  Stage planning for missing tiles always goes through the
        deferred-planning continuation (it runs on the next event-loop turn
        outside a burst).  Returns True when the retarget handled the step.

        Kill switch: ``ARRAYSCOPE_DISABLE_SESSION_RETARGET``.
        """

        def _reject(reason: str) -> bool:
            self._montage_session_retarget_last_reject = reason
            rejects = getattr(self, "_montage_session_retarget_rejects", None)
            if rejects is None:
                rejects = {}
                self._montage_session_retarget_rejects = rejects
            rejects[reason] = int(rejects.get(reason, 0)) + 1
            return False

        session = previous_session
        if session is None or session is not getattr(self, "_montage_session", None):
            return _reject("no-session")
        if axis is None or getattr(session, "montage_axis", None) is None:
            # Normal sliced images share this tiled path with axis=None; a
            # slice change is new semantic content behind an unchanged
            # layout, not an index-window move.  Only true montage sessions
            # retarget.
            return _reject("no-axis")
        if os.environ.get("ARRAYSCOPE_DISABLE_SESSION_RETARGET"):
            return _reject("kill-switch")
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
        if session.lod_policy_mode != self._montage_lod_policy_mode():
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
        if session.has_pending_level_update() and session.has_stale_level_presentations():
            # A pending level refinement keeps re-upserting stale tiles; the
            # rebirth path resets that bookkeeping (pinned behavior).  Reuse
            # only settled sessions until level convergence is owned by the
            # machine (ADR 0051 P2 remaining).
            return _reject("level-pending")
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
        if tuple(session.viewport_shape) != tuple(viewport_shape):
            return _reject("viewport-shape")
        if session.view_range != current_range:
            return _reject("view-range")
        session_key = montage_session_key(
            _document_key(document), view_state, viewport_plan, colormap_lut
        )
        if session_key == session.key:
            # Same montage identity with every presentation input equal: the
            # live session already represents this render request.  A rebirth
            # here re-resolves caches, re-seeds payloads, and force-commits
            # the whole scene for nothing — this same-key rebirth on every
            # re-render of a converged montage was the dominant share of the
            # "cached scrub ~50 ms/step" cost (session-rebirth class, ADR
            # 0051 P2).  Refresh the generation stamp so in-flight
            # completions stay current, commit anything dirty, and let the
            # standing machinery converge.
            self._montage_session_reuses = (
                int(getattr(self, "_montage_session_reuses", 0) or 0) + 1
            )
            session.render_generation = self._capture_render_generation()
            self._montage_viewport_update_token = None
            self._ensure_montage_watchdog()
            reuse_commit_start = perf_counter()
            try:
                # Only genuinely pending presentation work commits here; the
                # flush/level continuations own their own pacing, and a
                # settled re-render must stay a true no-op (zero item
                # updates) exactly like a rebirth that reseeds identical
                # payloads.
                # Only genuinely pending presentation work commits here; the
                # flush/level continuations own their own pacing, and a
                # settled re-render must stay a true no-op.
                if (
                    session.dirty_payloads
                    or session.pending_removals
                    or session.pending_payload_upserts
                    or not session.display_committed
                ):
                    self._commit_montage_session_presentation(session, force=True)
            finally:
                self._last_montage_initial_commit_ms = (
                    perf_counter() - reuse_commit_start
                ) * 1000.0
            self._last_montage_stage_plan_ms = 0.0
            self._last_montage_session_setup_ms = 0.0
            if session.defer_side_panels or _viewport_interaction_active(self):
                self.win._deferred_side_panel_refresh_pending = True
            else:
                self.win._update_operation_dock()
            return True
        # Change detection uses the same memoized semantic key batch that the
        # lazy tile_source_ids fill uses, so unchanged tiles are exact hits.
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
        semantic_key = montage_tile_semantic_key(
            _document_key(document), view_state, viewport_plan, colormap_lut
        )
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
        render_generation = self._capture_render_generation()
        # Undrained pyramid claims from the old window are balanced exactly as
        # on a rebirth; source identities are window-agnostic, so re-plans
        # re-claim cheaply.
        montage_lod.release_session_claims(session)
        session_id = int(getattr(self, "_montage_session_id", 0)) + 1
        self._montage_session_id = session_id
        setup_start = perf_counter()
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
        for tile, result in stats["stale_completions"]:
            self._store_reusable_montage_tile_result(
                tile,
                result,
                document=document,
                montage_axis=axis,
                colormap_lut=colormap_lut,
                shader_display=shader_display,
            )
        session.force_auto = bool(force_auto)
        session.user_levels_override = user_levels
        session.attach_stage_fan_in(StageFanInState())
        session.stage_planning_deferred = bool(missing_tiles)
        session.deferred_missing_tiles = tuple(missing_tiles)
        session.tile_compute_cache_hits = int(stats["hits"])
        session.tile_compute_waiting_for_stage = 0
        session.stage_backed_tiles_pending = 0
        session.lead_direct_tiles = 0
        session.retained_stage_index = None
        session.retained_stage_decision = "deferred-retarget"
        session.repeated_expensive_stage_per_tile = False
        if missing_tiles:
            self._montage_stage_plans_deferred = (
                int(getattr(self, "_montage_stage_plans_deferred", 0) or 0) + 1
            )
        self._montage_session_retargets = (
            int(getattr(self, "_montage_session_retargets", 0) or 0) + 1
        )
        self._montage_cached_tiles_last_session = int(stats["hits"])
        self._montage_missing_tiles_last_session = len(missing_tiles)
        self._montage_viewport_update_token = None
        self._ensure_montage_watchdog()
        self._ensure_montage_level_stats(level_key, expected_indices=all_indices)
        self._queue_montage_cached_level_stats(session, tuple(cached_tiles), seed_if_empty=True)
        self._last_montage_stage_plan_ms = 0.0
        self._last_montage_session_setup_ms = (perf_counter() - setup_start) * 1000.0
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
                + float(self._last_montage_session_setup_ms or 0.0),
                estimated_bytes=0,
            ),
        )
        initial_commit_start = perf_counter()
        try:
            self._commit_montage_session_presentation(session, force=True)
        finally:
            self._last_montage_initial_commit_ms = (
                perf_counter() - initial_commit_start
            ) * 1000.0
        if session.defer_side_panels or _viewport_interaction_active(self):
            self.win._deferred_side_panel_refresh_pending = True
        else:
            self.win._update_operation_dock()
        if session.stage_planning_deferred:
            if (
                _viewport_interaction_active(self)
                and self._montage_lod_policy_mode() == LOD_POLICY_RESIDENT
                and not os.environ.get("ARRAYSCOPE_DISABLE_SCRUB_FASTPATH")
            ):
                # Mid-burst under the resident policy: floors carry the
                # screen; the landing step plans (deferred continuation).
                self._schedule_deferred_montage_planning(session)
            else:
                # Single step, or a policy without floors: plan inline so
                # evaluations start within this call, like a rebirth.
                self._plan_deferred_montage_stages_now(session)
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
        reuse_any_lod = self._montage_lod_policy_mode() == LOD_POLICY_RESIDENT
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
            self._montage_lod_pipeline_reruns_avoided = int(
                getattr(self, "_montage_lod_pipeline_reruns_avoided", 0) or 0
            ) + len(cached_tiles)
        return cached_tiles, missing_tiles

    def _merge_montage_stage_plan(self, session: MontageRenderSession, stage_plan) -> None:
        session.stage_fan_in.merge_plan(stage_plan)
        session.ensure_stage_waiting_priority_queues()
        session.tile_compute_waiting_for_stage += len(stage_plan["waiting_indices"])
        session.stage_backed_tiles_pending += len(stage_plan["waiting_indices"])
        session.lead_direct_tiles += int(stage_plan["lead_direct_tiles"])
        if stage_plan["retained_stage_index"] is not None:
            session.retained_stage_index = stage_plan["retained_stage_index"]
        if stage_plan["retained_stage_decision"]:
            session.retained_stage_decision = stage_plan["retained_stage_decision"]
        session.repeated_expensive_stage_per_tile = bool(
            session.repeated_expensive_stage_per_tile
            or stage_plan["repeated_expensive_stage_per_tile"]
        )

    def _try_update_montage_viewport_only(self) -> bool:
        """Retarget a persistent tiled session without restarting evaluation."""

        session = getattr(self, "_montage_session", None)
        if not self._montage_session_is_current(session):
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
        expected_key = montage_session_key(
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

        additions, presentation_changed = session.retarget_viewport(
            view_range=viewport_plan.view_range,
            viewport_shape=viewport_plan.viewport_shape,
            plan=viewport_plan.plan,
            coverage_margin_tiles=retarget_policy.coverage_margin_tiles,
            near_margin_tiles=retarget_policy.near_margin_tiles,
            priority_focus=viewport_plan.priority_focus,
            priority_retarget_limit=_montage_priority_retarget_batch_limit(self, interactive=_interactive_active(self)),
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
        # Camera-only changes must retarget the LOD decision immediately:
        # already-resident levels swap on the next commit and missing levels
        # are scheduled now, superseded by the new viewport revision (ADR
        # 0050).  Demand math only; reduction stays on worker lanes.
        lod_swap_ready = session.refresh_lod_for_viewport()
        if getattr(session, "pending_lod_requests", None):
            self._schedule_montage_lod_materializations(session)
        additions = viewport_plan.prioritize_tiles(additions)
        self._prune_stale_montage_tile_work(session)
        if not additions:
            if presentation_changed or lod_swap_ready:
                self._schedule_montage_presentation_commit(session, force=False)
            if session.pending_tiles and not _viewport_interaction_active(self):
                self._schedule_montage_tiles(session)
                return True
            if session.pending_tiles:
                return True
            else:
                self._finish_montage_session_if_complete(session)
                schedule_near_viewport_montage_prefetch(self, session)
            return True
        budget = self._montage_callback_budget(
            "montage_viewport_update",
            interactive=_interactive_active(self),
            work_class="viewport_cached_additions",
            item_cap=_montage_viewport_addition_batch_limit(self, interactive=_interactive_active(self)),
        )
        cached_tiles = []
        missing_tiles = []
        additions_to_process = tuple(additions)
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
            byte_count = sum(_rendered_tile_nbytes(rendered) for rendered in display_cached)
            budget.record_item(byte_count=byte_count)
            if budget.should_yield():
                break
        self._record_gui_budget(budget)
        remaining_additions = max(0, len(additions_to_process) - int(processed_additions))
        self._last_montage_viewport_deferred_additions = int(remaining_additions)
        if remaining_additions:
            self.win._montage_viewport_update_pending = True
            self._montage_viewport_continue_immediately = True
        session.tile_compute_cache_hits += len(cached_tiles)
        for rendered in cached_tiles:
            session.mark_loaded(rendered)
        self._queue_montage_cached_level_stats(session, cached_tiles, seed_if_empty=False)

        self._montage_cached_tiles_last_session = len(cached_tiles)
        self._montage_missing_tiles_last_session = len(missing_tiles)
        if presentation_changed or cached_tiles or lod_swap_ready:
            self._schedule_montage_presentation_commit(session, force=False)
        if cached_tiles:
            self._schedule_montage_cached_level_stats(session)
        if not missing_tiles:
            self._finish_montage_session_if_complete(session)
            schedule_near_viewport_montage_prefetch(self, session)
            return True
        missing_tiles = viewport_plan.prioritize_tiles(missing_tiles)

        if _viewport_interaction_active(self):
            queued = set(session.pending_tile_numbers())
            for tile in missing_tiles:
                index = int(tile.montage_index)
                if index not in queued:
                    _enqueue_session_pending_tile(session, tile)
                    queued.add(index)
            self.win._montage_viewport_update_pending = True
            return True

        for tile in missing_tiles:
            session.mark_loading(tile)

        stage_plan = self._plan_montage_stages(session.document, missing_tiles)
        self._merge_montage_stage_plan(session, stage_plan)
        waiting = {int(index) for index in stage_plan["waiting_indices"]}
        queued = set(session.pending_tile_numbers())
        for tile in missing_tiles:
            index = int(tile.montage_index)
            if index not in waiting and index not in queued:
                _enqueue_session_pending_tile(session, tile)
                queued.add(index)

        self.win.prefetch_evaluation_controller.cancel_prefetch()
        self.win.operation_evaluator.last_status = CacheStatusSnapshot(
            CacheStatus.COMPUTING,
            "Extending montage viewport",
        )
        self._schedule_montage_session_slow_overlay(session)
        self._schedule_montage_stage_jobs(session, stage_plan["stage_requests"])
        self._dispatch_montage_work(session)
        return True

    def _prune_stale_montage_tile_work(self, session: MontageRenderSession) -> None:
        if not _viewport_interaction_active(self):
            return
        keep = {
            int(tile.montage_index)
            for tile in session.plan.tiles_intersecting(session.view_range, margin_tiles=2)
        }
        if not keep:
            return
        pending_before = len(session.pending_tiles)
        pruned_pending = session.prune_pending_tiles(keep)
        stale = (set(session.loading_tiles) - set(session.active_tile_requests)) - keep
        if stale:
            for index in sorted(stale):
                session.loading_tiles.discard(int(index))
                if 0 <= int(index) < len(session.tile_states):
                    session.tile_states[int(index)] = MontageTileState.UNLOADED
                    session.invalidate_tile_states()
        for key, waiting in list(session.stage_fan_in.waiting_tiles.items()):
            kept = [tile for tile in waiting if int(tile.montage_index) in keep]
            if kept:
                if hasattr(waiting, "clear") and hasattr(waiting, "extend"):
                    waiting.clear()
                    waiting.extend(kept)
                    session.stage_fan_in.waiting_tiles[key] = waiting
                else:
                    session.stage_fan_in.waiting_tiles[key] = kept
            else:
                session.stage_fan_in.waiting_tiles.pop(key, None)
        pruned = pruned_pending + len(stale)
        if pruned > 0:
            self._last_montage_pruned_tile_work = int(getattr(self, "_last_montage_pruned_tile_work", 0) or 0) + int(pruned)

    def _montage_level_key(self, document, view_state, all_indices, colormap_lut):
        return montage_level_key(
            _document_key(document),
            view_state,
            all_indices,
            colormap_lut,
        )

    def _montage_level_expected_indices(self, session) -> tuple[int, ...]:
        expected = tuple(int(index) for index in getattr(session, "level_expected_indices", ()) or ())
        if expected:
            return expected
        return tuple(int(tile.source_index) for tile in getattr(session.plan, "tiles", ()))

    def _empty_montage_level_stats(self, expected_indices) -> MontageLevelStats:
        tracker = self._montage_level_tracker()
        key = ("empty", tuple(int(index) for index in expected_indices))
        return tracker.ensure(key, expected_indices)

    def _ensure_montage_level_stats(self, level_key, *, expected_indices) -> MontageLevelStats:
        return self._montage_level_tracker().ensure(level_key, expected_indices)

    def _montage_coverage_rank(self, source_indices, expected_indices) -> int:
        stats = self._montage_level_tracker().ensure(("rank", tuple(expected_indices)), expected_indices)
        rank = self._montage_level_tracker()._rank_for(source_indices, stats.expected_indices)
        if rank == LevelSourceRank.NONE:
            return 0
        if rank == LevelSourceRank.MONTAGE_COMPLETE:
            return 2
        return 1

    def _update_montage_level_bounds_from_rendered(self, level_key, rendered, *, expected_indices=None, refined: bool = False) -> None:
        if expected_indices is None:
            previous_stats = self._montage_level_tracker().stats_for(level_key)
            expected_indices = () if previous_stats is None else previous_stats.expected_indices
        tracker = self._montage_level_tracker()
        tracker.ensure_expected(level_key, expected_indices)
        source_index = int(rendered.tile.source_index)
        level_stats = getattr(rendered, "level_stats", None)
        existing_refined = tracker.has_source(level_key, source_index, refined=True)
        existing_any = existing_refined or tracker.has_source(level_key, source_index)
        if existing_refined or (
            existing_any
            and not bool(refined)
            and (level_stats is None or not bool(getattr(level_stats, "refined", False)))
        ):
            return
        if level_stats is not None and not refined:
            tracker.update_from_stats(level_key, level_stats, aggregate=False)
            return
        level_data = getattr(rendered, "level_data", None)
        if level_data is not None and not refined:
            stats = provisional_tile_level_stats(level_data, source_index)
            if stats is not None:
                tracker.update_from_stats(level_key, stats, aggregate=False)
                return
        stats = sample_tile_level_stats(
            _montage_refined_level_values(rendered),
            source_index,
            refined=bool(refined),
        )
        if stats is not None:
            tracker.update_from_stats(level_key, stats, aggregate=False)
        elif refined:
            # Nothing finite to sample: record that as refined evidence, or
            # level convergence re-queues this source forever and an
            # explicit-auto flush parked on the rank can never re-commit.
            tracker.record_vacuous_source(level_key, source_index)

    def _update_montage_level_bounds_from_prepared(self, level_key, rendered, *, expected_indices=None, require_refined: bool = False) -> bool:
        """Merge already-prepared level evidence without sampling source pixels."""

        if expected_indices is None:
            previous_stats = self._montage_level_tracker().stats_for(level_key)
            expected_indices = () if previous_stats is None else previous_stats.expected_indices
        tracker = self._montage_level_tracker()
        tracker.ensure_expected(level_key, expected_indices)
        source_index = int(rendered.tile.source_index)
        if tracker.has_source(level_key, source_index, refined=bool(require_refined)):
            return True
        if require_refined and tracker.has_source(level_key, source_index):
            return False
        level_stats = getattr(rendered, "level_stats", None)
        if level_stats is not None:
            if require_refined and not bool(getattr(level_stats, "refined", False)):
                return False
            tracker.update_from_stats(level_key, level_stats, aggregate=False)
            return True
        if require_refined:
            return False
        level_data = getattr(rendered, "level_data", None)
        if level_data is not None:
            stats = provisional_tile_level_stats(level_data, source_index)
            if stats is not None:
                tracker.update_from_stats(level_key, stats, aggregate=False)
                return True
        return False

    def _queue_montage_level_refinement(self, session, rendered) -> None:
        tracker = self._montage_level_tracker()
        source_index = int(rendered.tile.source_index)
        if tracker.has_source(session.level_key, source_index, refined=True):
            return
        pending = getattr(session, "pending_refined_level_tiles", None)
        if pending is None:
            pending = deque()
            session.pending_refined_level_tiles = pending
        pending_sources = getattr(session, "pending_refined_level_sources", None)
        if pending_sources is None:
            pending_sources = {int(item.tile.source_index) for item in pending}
            session.pending_refined_level_sources = pending_sources
        if source_index in pending_sources:
            return
        pending.append(rendered)
        pending_sources.add(source_index)

    def _montage_level_stats_for_session(self, session) -> MontageLevelStats:
        expected = self._montage_level_expected_indices(session)
        self._montage_level_tracker().ensure_expected(session.level_key, expected)
        stats = self._montage_level_tracker().summary_for(session.level_key)
        if stats is None:
            return self._ensure_montage_level_stats(session.level_key, expected_indices=expected)
        return stats

    def _montage_level_bounds_for_session(self, session, *, allow_partial: bool = False):
        source = self._montage_level_source_for_session(session, allow_partial=allow_partial)
        return None if source is None else source.histogram_range

    def _montage_level_source_for_session(self, session, *, allow_partial: bool = False):
        # Partial semantic tile coverage is a valid provisional level source.
        # It must not be confused with viewport pixels; the level key is semantic
        # and excludes zoom/pan.  WindowLevelController keeps updates monotonic.
        tracker = self._montage_level_tracker()
        stats = tracker.summary_for(session.level_key)
        if stats is None:
            return None
        if not allow_partial and stats.rank not in {LevelSourceRank.MONTAGE_COMPLETE, LevelSourceRank.MONTAGE_SAMPLED_FULL}:
            return None
        return tracker.source_for_stats(session.level_key, stats)

    def _montage_histogram_plot_data_for_session(self, session, *, allow_partial: bool = False):
        tracker = self._montage_level_tracker()
        stats = tracker.stats_for(session.level_key)
        if stats is None:
            return None
        if not allow_partial and stats.rank not in {LevelSourceRank.MONTAGE_COMPLETE, LevelSourceRank.MONTAGE_SAMPLED_FULL}:
            return None
        return tracker.histogram_data_for_stats(stats)

    def _montage_level_tracker(self) -> MontageLevelTracker:
        tracker = getattr(self, "_montage_level_tracker_instance", None)
        if tracker is None:
            tracker = MontageLevelTracker()
            self._montage_level_tracker_instance = tracker
        return tracker

    def _schedule_montage_cached_level_stats(self, session) -> None:
        if (
            not getattr(session, "pending_level_tiles", None)
            and int(getattr(session, "level_scan_remaining_tiles", 0) or 0) <= 0
        ):
            return
        timer = getattr(self, "_montage_level_stats_timer", None)
        if timer is None:
            # Bounded continuation. Cached level stats are secondary UI work
            # and each slice is budgeted by `_process_montage_cached_level_stats`;
            # remove when histogram refinement is fully WorkGraph-owned.
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._process_montage_cached_level_stats)
            self._montage_level_stats_timer = timer
        if not timer.isActive():
            timer.start(0)

    def _pending_montage_level_sources(self, session):
        pending = getattr(session, "pending_level_tiles", None)
        if pending is None:
            pending = deque()
            session.pending_level_tiles = pending
        queued_sources = getattr(session, "pending_level_sources", None)
        if queued_sources is None:
            queued_sources = {int(item.tile.source_index) for item in pending}
            session.pending_level_sources = queued_sources
        return pending, queued_sources

    def _mark_montage_level_scan_pending(self, session) -> None:
        # Restart a FULL pass even mid-scan: a tile that materializes after
        # the cursor already passed its position would otherwise fall through
        # a completed pass and no continuation would ever sample it (level
        # rank then never completes for exactly that source).  Passes are
        # cheap — already-merged sources are skip-checked — and arrivals are
        # bounded by the plan, so restarts terminate.
        tile_count = len(getattr(getattr(session, "plan", None), "tiles", ()) or ())
        if tile_count <= 0:
            return
        session.level_scan_remaining_tiles = tile_count

    def _queue_montage_cached_level_stats(self, session, rendered_tiles, *, seed_if_empty: bool) -> None:
        """Admit cached-payload level work without making commit latency scale with tile count.

        Prepared per-tile evidence can be merged immediately for a small,
        bounded batch.  Anything requiring source-pixel sampling is queued for
        the same timer-driven maintenance path used by later cached residents.
        """

        tracker = self._montage_level_tracker()
        expected = self._montage_level_expected_indices(session)
        tracker.ensure_expected(session.level_key, expected)
        pending, queued_sources = self._pending_montage_level_sources(session)
        require_refined = _montage_level_evidence_requires_refined(self, session)
        summary = tracker.summary_for(session.level_key)
        inspected = 0
        seeded = bool(summary is not None and summary.source_indices)
        for rendered in rendered_tiles or ():
            if inspected >= MONTAGE_LEVEL_STATS_COMMIT_BATCH:
                self._mark_montage_level_scan_pending(session)
                break
            inspected += 1
            source_index = int(rendered.tile.source_index)
            if source_index in queued_sources or tracker.has_source(session.level_key, source_index, refined=require_refined):
                continue
            if self._update_montage_level_bounds_from_prepared(
                session.level_key,
                rendered,
                expected_indices=expected,
                require_refined=require_refined,
            ):
                seeded = True
                self._queue_montage_level_refinement(session, rendered)
                continue
            pending.append(rendered)
            queued_sources.add(source_index)
            if seed_if_empty and not seeded:
                seeded = True
        self._montage_pending_level_tiles_last_session = len(pending or ())

    def _queue_montage_level_stats_for_payloads(self, session, payloads) -> int:
        """Request level evidence for a presentation delta without scanning it inline."""

        tracker = self._montage_level_tracker()
        expected = self._montage_level_expected_indices(session)
        tracker.ensure_expected(session.level_key, expected)
        stats_start = perf_counter()
        merged = 0
        inspected = 0
        pending, queued_sources = self._pending_montage_level_sources(session)
        require_refined = _montage_level_evidence_requires_refined(self, session)
        for tile_number in payloads or ():
            if inspected >= MONTAGE_LEVEL_STATS_COMMIT_BATCH:
                self._mark_montage_level_scan_pending(session)
                break
            inspected += 1
            rendered = getattr(session, "rendered_tiles", {}).get(int(tile_number))
            if rendered is None:
                continue
            source_index = int(rendered.tile.source_index)
            if tracker.has_source(session.level_key, source_index, refined=require_refined):
                continue
            if self._update_montage_level_bounds_from_prepared(
                session.level_key,
                rendered,
                expected_indices=expected,
                require_refined=require_refined,
            ):
                self._queue_montage_level_refinement(session, rendered)
                queued_sources.discard(source_index)
                merged += 1
            elif source_index not in queued_sources:
                pending.append(rendered)
                queued_sources.add(source_index)
        self._last_montage_level_stats_ms = (perf_counter() - stats_start) * 1000.0
        self._montage_level_sources_added_last_commit = int(merged)
        self._montage_pending_level_tiles_last_session = len(getattr(session, "pending_level_tiles", ()) or ())
        self._schedule_montage_cached_level_stats(session)
        return int(merged)

    def _scan_montage_level_stats_from_session(self, session, *, expected, stats_start: float, processed: int, budget: GuiCallbackBudget) -> int:
        tile_count = len(getattr(getattr(session, "plan", None), "tiles", ()) or ())
        remaining = int(getattr(session, "level_scan_remaining_tiles", 0) or 0)
        if tile_count <= 0 or remaining <= 0:
            session.level_scan_remaining_tiles = 0
            return int(processed)
        pending, queued_sources = self._pending_montage_level_sources(session)
        tracker = self._montage_level_tracker()
        require_refined = _montage_level_evidence_requires_refined(self, session)
        cursor = int(getattr(session, "level_scan_cursor", 0) or 0) % tile_count
        inspected = 0
        while remaining > 0 and inspected < int(budget.item_cap):
            rendered = getattr(session, "rendered_tiles", {}).get(cursor)
            cursor = (cursor + 1) % tile_count
            remaining -= 1
            inspected += 1
            if rendered is None:
                continue
            source_index = int(rendered.tile.source_index)
            if source_index in queued_sources or tracker.has_source(session.level_key, source_index, refined=require_refined):
                continue
            if self._update_montage_level_bounds_from_prepared(
                session.level_key,
                rendered,
                expected_indices=expected,
                require_refined=require_refined,
            ):
                self._queue_montage_level_refinement(session, rendered)
                processed += 1
            else:
                pending.append(rendered)
                queued_sources.add(source_index)
            budget.record_item(byte_count=_rendered_tile_nbytes(rendered))
            if budget.should_yield():
                break
        session.level_scan_cursor = int(cursor)
        session.level_scan_remaining_tiles = max(0, int(remaining))
        return int(processed)

    def _process_montage_cached_level_stats(self) -> None:
        session = getattr(self, "_montage_session", None)
        if not self._montage_session_is_current(session):
            return
        pending = getattr(session, "pending_level_tiles", None)
        if not pending and int(getattr(session, "level_scan_remaining_tiles", 0) or 0) <= 0:
            return
        budget = self._montage_callback_budget(
            "montage_level_evidence",
            interactive=_interactive_active(self),
            work_class="semantic_level_evidence",
            item_cap=MONTAGE_LEVEL_STATS_BACKGROUND_BATCH,
            target_ms=MONTAGE_LEVEL_STATS_BACKGROUND_BUDGET_MS,
        )
        stats_start = perf_counter()
        expected = self._montage_level_expected_indices(session)
        require_refined = _montage_level_evidence_requires_refined(self, session)
        processed = 0
        while pending and processed < int(budget.item_cap):
            rendered = pending.popleft()
            source_index = int(rendered.tile.source_index)
            pending_sources = getattr(session, "pending_level_sources", None)
            if pending_sources is not None:
                pending_sources.discard(source_index)
            if not self._montage_level_tracker().has_source(session.level_key, source_index, refined=require_refined):
                self._update_montage_level_bounds_from_rendered(
                    session.level_key,
                    rendered,
                    expected_indices=expected,
                    refined=require_refined,
                )
            self._queue_montage_level_refinement(session, rendered)
            processed += 1
            budget.record_item(byte_count=_rendered_tile_nbytes(rendered))
            if budget.should_yield():
                break
        if not pending and not budget.should_yield():
            processed = self._scan_montage_level_stats_from_session(
                session,
                expected=expected,
                stats_start=stats_start,
                processed=processed,
                budget=budget,
            )
        self._last_montage_level_stats_ms = (perf_counter() - stats_start) * 1000.0
        self._montage_pending_level_tiles_last_session = len(pending or ())
        self._record_gui_budget(budget)
        if processed:
            _complete_inline_work(
                self,
                WorkItem(
                    key=(
                        "montage_level_evidence",
                        session.key,
                        int(session.session_id),
                        int(getattr(session, "level_revision", 0) or 0),
                        int(processed),
                    ),
                    lane=WorkLane.HISTOGRAM_REFINEMENT,
                    quality="retained",
                    supersession_key=("montage-level-evidence", session.key),
                    supersession_value=int(session.session_id),
                    estimated_cpu_ms=float(self._last_montage_level_stats_ms or 0.0),
                    estimated_bytes=int(budget.processed_bytes),
                ),
            )
        # A histogram/level refinement is presentation metadata.  It must not
        # force a full tiled-payload refresh or replay stale removals after a
        # viewport change.  Normal display commits will publish richer sources;
        # when there is no visible upload backlog, a non-forced commit can
        # update uniforms/histogram without invalidating residency.
        # Evidence-drain pacing (wedge cost fix 2026-07-05): while a parked
        # explicit-auto flush waits on level evidence, committing after EVERY
        # budget slice re-runs the full payload build per handful of tiles
        # (~68 no-op commits for a 272-tile scene).  Commit when the evidence
        # queue actually drained — the parked flush re-checks the rank then —
        # or when nothing is parked (metadata refresh for a settled session).
        evidence_remaining = bool(pending) or int(getattr(session, "level_scan_remaining_tiles", 0) or 0) > 0
        flush_parked = bool(getattr(session, "flush_pending", False) or getattr(session, "final_commit_pending", False))
        if (
            processed
            and not (evidence_remaining and flush_parked)
            and not getattr(session, "dirty_payloads", ())
            and not getattr(session, "pending_removals", ())
        ):
            self._schedule_montage_presentation_commit(session, force=flush_parked)
        self._schedule_montage_cached_level_stats(session)

    def _schedule_montage_refined_level_stats(self, session) -> None:
        if not self._montage_session_is_current(session):
            return
        pending = getattr(session, "pending_refined_level_tiles", None)
        if not pending:
            return
        controller = getattr(self.win, "histogram_evaluation_controller", None)
        if controller is None:
            return
        scheduled = 0
        while pending and scheduled < 4:
            rendered = pending.popleft()
            source_index = int(rendered.tile.source_index)
            if self._montage_level_tracker().has_source(session.level_key, source_index, refined=True):
                pending_sources = getattr(session, "pending_refined_level_sources", None)
                if pending_sources is not None:
                    pending_sources.discard(source_index)
                continue
            source = _montage_refined_level_values(rendered)
            key = ("montage_refined_level_stats", session.level_key, source_index)

            def evaluate(source=source, source_index=source_index):
                return sample_tile_level_stats(source, int(source_index), refined=True)

            def done(
                stats,
                session_id=session.session_id,
                session_key=session.key,
                level_key=session.level_key,
                source_index=source_index,
            ):
                self._on_montage_refined_level_stats_done(session_id, session_key, level_key, source_index, stats)

            started = controller.start_latest(
                evaluate,
                on_done=done,
                key=key,
                priority=EvalPriority.HISTOGRAM,
                replace_group=f"montage_level_refinement:{source_index}",
                memory_budget_bytes=self._memory_policy().display_cache_budget_bytes,
            )
            if started is None:
                pending.appendleft(rendered)
                break
            scheduled += 1

    def _on_montage_refined_level_stats_done(self, session_id, session_key, level_key, source_index, stats) -> None:
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session_id, session_key):
            return
        pending_sources = getattr(session, "pending_refined_level_sources", None)
        if pending_sources is not None:
            pending_sources.discard(int(source_index))
        if stats is not None:
            self._montage_level_tracker().update_from_stats(level_key, stats, aggregate=False)
            summary = self._montage_level_tracker().summary_for(level_key)
            if (
                summary is not None
                and session.display_committed
                and not getattr(session, "dirty_payloads", ())
                and not getattr(session, "pending_removals", ())
                and getattr(session, "user_levels_override", None) is None
                and self._should_publish_montage_level_metadata(session, summary)
            ):
                self._schedule_montage_presentation_commit(session, force=False)
        self._schedule_montage_refined_level_stats(session)

    def _plan_montage_stages(self, document, missing_tiles):
        document_key = stage_document_key(document)
        groups = {}
        tile_candidates = {}
        tile_stage_plans = {}
        tile_stage_candidates = {}
        for tile in tuple(missing_tiles):
            try:
                request = request_for_image(tile.view_state)
                plan = plan_slab(document, request)
            except Exception as exc:
                handle_ui_exception("montage stage planning", exc)
                continue
            candidates = tuple(getattr(plan.region_plan, "cache_candidates", ()))
            retained = tuple(candidate for candidate in candidates if getattr(candidate, "retain", True))
            if not retained:
                continue
            candidate = retained[-1]
            key = self.win.operation_evaluator.stage_materializer.key_for_candidate(document_key, candidate)
            groups.setdefault(key, {"candidate": candidate, "tiles": [], "plan": plan, "request": request})
            groups[key]["tiles"].append(tile)
            tile_candidates[int(tile.montage_index)] = key
            tile_stage_plans[int(tile.montage_index)] = plan
            tile_stage_candidates[int(tile.montage_index)] = candidate

        tile_stage_keys = {}
        stage_waiting_tiles = {}
        stage_values = {}
        lead_stage_warmups = {}
        stage_requests = []
        attached_stage_keys = set()
        waiting_indices = set()
        lead_direct_tile_count = 0
        retained_stage_index = None
        retained_stage_decision = ""
        repeated_expensive_stage_per_tile = False
        for key, group in groups.items():
            tiles = tuple(group["tiles"])
            candidate = group["candidate"]
            retained_stage_index = int(getattr(candidate, "stage_index", -1) or -1)
            estimated = int(getattr(candidate, "estimated_nbytes", 0) or 0)
            if len(tiles) < 2 and estimated < 16 * 1024 * 1024:
                continue
            result = self.win.operation_evaluator.stage_materializer.request_stage(document_key, candidate)
            retained_stage_decision = result.decision
            if result.decision == "hit":
                stage_values[key] = result.value
                for tile in tiles:
                    tile_stage_keys[int(tile.montage_index)] = key
                    tile_stage_plans[int(tile.montage_index)] = tile_stage_plans.get(int(tile.montage_index), group["plan"])
                    tile_stage_candidates[int(tile.montage_index)] = candidate
                continue
            if result.decision == "scheduled":
                # Compute the shared stage as a stage-lane job and keep every
                # tile behind the fan-in. A montage-tile "lead" computing the
                # stage inline runs it with the tile lane's single FFT worker
                # while the whole montage waits; the stage lane gets the
                # multi-worker FFT context. The stage cache's in-flight claim
                # keeps any concurrent direct evaluation from duplicating it.
                stage_waiting_tiles[key] = list(tiles)
                for tile in tiles:
                    tile_stage_keys[int(tile.montage_index)] = key
                    tile_stage_plans[int(tile.montage_index)] = tile_stage_plans.get(int(tile.montage_index), group["plan"])
                    tile_stage_candidates[int(tile.montage_index)] = candidate
                    waiting_indices.add(int(tile.montage_index))
                stage_requests.append((result.request, group["plan"]))
                continue
            if result.decision == "attached":
                stage_waiting_tiles[key] = list(tiles)
                attached_stage_keys.add(key)
                for tile in tiles:
                    tile_stage_keys[int(tile.montage_index)] = key
                    tile_stage_plans[int(tile.montage_index)] = tile_stage_plans.get(int(tile.montage_index), group["plan"])
                    tile_stage_candidates[int(tile.montage_index)] = candidate
                    waiting_indices.add(int(tile.montage_index))
                if result.request is not None:
                    stage_requests.append((result.request, group["plan"]))
                continue
            for tile in tiles:
                tile_stage_keys.pop(int(tile.montage_index), None)
            if len(tiles) > 1:
                repeated_expensive_stage_per_tile = True
        return {
            "tile_stage_keys": tile_stage_keys,
            "tile_stage_plans": tile_stage_plans,
            "tile_stage_candidates": tile_stage_candidates,
            "stage_waiting_tiles": stage_waiting_tiles,
            "stage_values": stage_values,
            "lead_stage_warmups": lead_stage_warmups,
            "stage_requests": stage_requests,
            "attached_stage_keys": attached_stage_keys,
            "waiting_indices": waiting_indices,
            "lead_direct_tiles": int(lead_direct_tile_count),
            "retained_stage_index": retained_stage_index,
            "retained_stage_decision": retained_stage_decision,
            "repeated_expensive_stage_per_tile": bool(repeated_expensive_stage_per_tile),
        }

    def _schedule_deferred_montage_planning(self, session, *, delay_ms: int = 0) -> None:
        """Arm the deferred planning continuation for an interaction-burst session.

        Receiver-scoped single shot (architecture guard: 3-arg form) so a
        closed window cancels the chain; supersession is checked in the
        completion, and a superseded session simply never plans — it took no
        stage claims, so there is nothing to balance.
        """

        Qt.QtCore.QTimer.singleShot(
            int(delay_ms),
            self.win,
            lambda session=session: self._complete_deferred_montage_planning(session),
        )

    def _complete_deferred_montage_planning(self, session) -> None:
        if not self._montage_session_is_current(session):
            return
        if not bool(getattr(session, "stage_planning_deferred", False)):
            return
        if _viewport_interaction_active(self):
            # Still mid-burst: floors are carrying the screen; re-arm and let
            # the next scrub step supersede this session instead.
            self._schedule_deferred_montage_planning(session, delay_ms=80)
            return
        self._plan_deferred_montage_stages_now(session)

    def _plan_deferred_montage_stages_now(self, session) -> None:
        """Run the deferred stage planning immediately (no interaction gate)."""

        missing_tiles = tuple(getattr(session, "deferred_missing_tiles", ()) or ())
        session.stage_planning_deferred = False
        session.deferred_missing_tiles = ()
        stage_plan_start = perf_counter()
        stage_plan = self._plan_montage_stages(session.document, missing_tiles)
        self._last_montage_stage_plan_ms = (perf_counter() - stage_plan_start) * 1000.0
        session.attach_stage_fan_in(StageFanInState(
            tile_stage_keys=stage_plan["tile_stage_keys"],
            tile_stage_plans=stage_plan["tile_stage_plans"],
            tile_stage_candidates=stage_plan["tile_stage_candidates"],
            waiting_tiles=stage_plan["stage_waiting_tiles"],
            attached_requests=stage_plan["attached_stage_keys"],
            values=stage_plan["stage_values"],
            lead_warmups=stage_plan["lead_stage_warmups"],
        ))
        for tile in missing_tiles:
            if int(tile.montage_index) not in stage_plan["waiting_indices"]:
                session.enqueue_pending_tile(tile)
        session.tile_compute_waiting_for_stage = len(stage_plan["waiting_indices"])
        session.stage_backed_tiles_pending = len(stage_plan["waiting_indices"])
        session.lead_direct_tiles = stage_plan["lead_direct_tiles"]
        session.retained_stage_index = stage_plan["retained_stage_index"]
        session.retained_stage_decision = stage_plan["retained_stage_decision"]
        session.repeated_expensive_stage_per_tile = stage_plan["repeated_expensive_stage_per_tile"]
        self._schedule_montage_stage_jobs(session, stage_plan["stage_requests"])
        self._dispatch_montage_work(session)

    def _schedule_montage_stage_jobs(self, session, stage_requests) -> None:
        if not self._montage_session_is_current(session):
            return
        controller = getattr(self.win, "stage_evaluation_controller", self.win.visible_evaluation_controller)
        for request, plan in tuple(stage_requests):
            if request is None or request.key in session.stage_fan_in.active_requests:
                continue
            session.stage_fan_in.active_requests.add(request.key)

            def evaluate(token, request=request, plan=plan):
                context = self.win._evaluation_context(ComputeLane.STAGE, token)
                return materialize_stage_candidate_chunked(
                    session.document,
                    plan.region_plan,
                    request.candidate,
                    stage_cache=self.win.operation_evaluator.stage_cache,
                    document_key=request.document_key,
                    cancellation_token=token,
                    evaluation_context=context,
                    memory_policy=context.memory_policy,
                    allowed_chunk_axes=stage_materialization_allowed_chunk_axes(request.candidate.shape),
                )

            started = controller.start_latest(
                evaluate,
                key=("stage", request.key),
                priority=EvalPriority.VISIBLE_IMAGE,
                replace_group=f"montage-stage:{int(session.session_id)}:{hash(request.key)}",
                frame_target=session.frame_plan.target,
                supersession_key=("montage-stage", request.key),
                supersession_value=(session.key, int(session.session_id)),
                work_item=WorkItem(
                    key=("montage_stage_materialization", request.key, int(session.session_id)),
                    lane=WorkLane.STAGE_MATERIALIZATION,
                    frame_target=session.frame_plan.target,
                    quality=session.frame_plan.target.quality,
                    supersession_key=("montage-stage", request.key),
                    supersession_value=(session.key, int(session.session_id)),
                    estimated_bytes=int(getattr(request.candidate, "estimated_nbytes", 0) or 0),
                    reusable_output=True,
                ),
                on_done=lambda value, session_id=session.session_id, key=request.key: self._on_montage_stage_done(session_id, key, value),
                on_error=lambda exc, session_id=session.session_id, key=request.key: self._on_montage_stage_error(session_id, key, exc),
                on_stale=lambda key=request.key: self._on_montage_stage_stale(key),
                on_slow=lambda: self._on_montage_tile_slow(session.session_id),
                slow_ms=100,
                pass_token=True,
            )
            if started is None:
                # Admission declined: the key was optimistically marked
                # active, and leaving it there makes the planner's dedup
                # skip this stage forever while every stage-backed tile
                # waits on it (observed: 64 tiles wedged behind one lost
                # ~490 MB stage after a session-restore burst). Roll back
                # so the next planning pass re-requests it.
                session.stage_fan_in.active_requests.discard(request.key)
                self._montage_stage_admission_declined = (
                    int(getattr(self, "_montage_stage_admission_declined", 0) or 0) + 1
                )

    def _on_montage_stage_stale(self, key) -> None:
        self.win.operation_evaluator.stage_materializer.cancel(key)
        session = getattr(self, "_montage_session", None)
        if session is not None:
            session.stage_fan_in.active_requests.discard(key)
            # A superseded stage can strand its waiting tiles: re-derive
            # (the stage-waits pump releases them to direct evaluation).
            self._dispatch_montage_work(session)

    def _on_montage_stage_done(self, session_id, key, value) -> None:
        session = getattr(self, "_montage_session", None)
        self.win.operation_evaluator.stage_materializer.complete(key, value)
        if session is None or not self._is_current_montage_session(session_id, session.key):
            return
        if not self._is_current_render_generation(session.render_generation):
            return
        # Activation batches pop waiting tiles in priority order; make sure
        # that order reflects the live viewport, not the pre-commit range.
        self._refresh_montage_priority_targets(session)
        budget = self._montage_callback_budget(
            "montage_stage_wait",
            interactive=_interactive_active(self),
            work_class="stage_wait_activation",
        )
        self._activate_montage_stage_value(session, key, value, budget=budget)
        self._record_gui_budget(budget)
        self._dispatch_montage_work(session)

    def _activate_montage_stage_value(self, session, key, value, *, budget: GuiCallbackBudget | None = None) -> None:
        max_items = None if budget is None else max(0, int(budget.item_cap) - int(budget.processed_items))
        if max_items == 0:
            return
        batch = session.stage_fan_in.activate_value(key, value, max_items=max_items)
        processed = 0
        for tile in batch.tiles:
            index = int(tile.montage_index)
            if index not in session.rendered_tiles and index not in session.skipped_tiles:
                _enqueue_session_pending_tile(session, tile)
                session.mark_loading(tile)
            processed += 1
        if budget is not None and processed:
            budget.record_item(item_count=processed)

    def _release_stage_waiting_tiles_to_direct(self, session, key, *, budget: GuiCallbackBudget | None = None) -> None:
        max_items = None if budget is None else max(0, int(budget.item_cap) - int(budget.processed_items))
        if max_items == 0:
            return
        batch = session.stage_fan_in.release_missing(key, max_items=max_items)
        processed = 0
        for tile in batch.tiles:
            index = int(tile.montage_index)
            if index not in session.rendered_tiles and index not in session.skipped_tiles:
                _enqueue_session_pending_tile(session, tile)
                session.mark_loading(tile)
            processed += 1
        if budget is not None and processed:
            budget.record_item(item_count=processed)

    def _schedule_montage_attached_stage_waits(self, session) -> None:
        if not self._stage_wait_has_actionable_work(session, release_missing=True):
            return
        self._montage_attached_stage_token = _montage_work_token(session, "stage_wait")
        timer = getattr(self, "_montage_attached_stage_timer", None)
        if timer is None:
            # Bounded continuation guarded by `_montage_attached_stage_token`.
            # It releases stage waiters in GUI-budgeted batches instead of
            # draining a completed stage in one callback.
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._process_montage_attached_stage_waits)
            self._montage_attached_stage_timer = timer
        if not timer.isActive():
            timer.start(0)

    def _process_montage_attached_stage_waits(self) -> None:
        session = getattr(self, "_montage_session", None)
        if not self._montage_session_is_current(session):
            return
        token = getattr(self, "_montage_attached_stage_token", None)
        if not _montage_work_token_is_current(session, token, "stage_wait"):
            return
        pending_keys = tuple(
            dict.fromkeys(
                tuple(session.stage_fan_in.attached_requests)
                + tuple(session.stage_fan_in.waiting_tiles)
            )
        )
        if not pending_keys:
            return
        wait_start = perf_counter()
        budget = self._montage_callback_budget(
            "montage_stage_wait",
            interactive=_interactive_active(self),
            work_class="stage_wait_release",
        )
        for key in pending_keys:
            self._activate_or_release_waiting_stage(session, key, release_missing=True, budget=budget)
            if budget.should_yield():
                break
        self._last_montage_stage_attach_wait_ms = (perf_counter() - wait_start) * 1000.0
        if budget.processed_items:
            _complete_inline_work(
                self,
                WorkItem(
                    key=("montage_stage_wait_fan_in", session.key, int(session.session_id), int(session.stage_fan_in.has_waiting())),
                    lane=WorkLane.GUI_FAN_IN,
                    frame_target=session.frame_plan.target,
                    supersession_key=("montage-stage-wait", session.key),
                    supersession_value=int(session.session_id),
                    estimated_cpu_ms=float(self._last_montage_stage_attach_wait_ms),
                    estimated_bytes=int(budget.processed_bytes),
                ),
            )
        self._record_gui_budget(budget)
        self._dispatch_montage_work(session)

    def _activate_cached_waiting_stages(self, session, *, release_missing: bool = False) -> None:
        if not self._stage_wait_has_actionable_work(session, release_missing=release_missing):
            return
        budget = self._montage_callback_budget(
            "montage_stage_wait",
            interactive=_interactive_active(self),
            work_class="stage_wait_cached_activation",
        )
        for key in tuple(session.stage_fan_in.waiting_tiles):
            self._activate_or_release_waiting_stage(session, key, release_missing=release_missing, budget=budget)
            if budget.should_yield():
                break
        if budget.processed_items:
            _complete_inline_work(
                self,
                WorkItem(
                    key=("montage_cached_stage_fan_in", session.key, int(session.session_id), bool(release_missing)),
                    lane=WorkLane.GUI_FAN_IN,
                    frame_target=session.frame_plan.target,
                    supersession_key=("montage-stage-wait", session.key),
                    supersession_value=int(session.session_id),
                    estimated_bytes=int(budget.processed_bytes),
                ),
            )
        self._record_gui_budget(budget)
        if self._stage_wait_has_actionable_work(session, release_missing=release_missing):
            self._schedule_montage_attached_stage_waits(session)

    def _activate_or_release_waiting_stage(self, session, key, *, release_missing: bool, budget: GuiCallbackBudget | None = None) -> None:
        if key in session.stage_fan_in.values:
            self._activate_montage_stage_value(session, key, session.stage_fan_in.values[key], budget=budget)
            return
        cache = self.win.operation_evaluator.stage_cache
        value = cache.get_containing(key) if hasattr(cache, "get_containing") else cache.get(key)
        if value is not None:
            self.win.operation_evaluator.stage_materializer.cancel(key)
            self._activate_montage_stage_value(session, key, value, budget=budget)
            return
        in_flight = getattr(self.win.operation_evaluator.stage_materializer, "_in_flight", {})
        if release_missing and key not in in_flight:
            self._release_stage_waiting_tiles_to_direct(session, key, budget=budget)

    def _stage_wait_has_actionable_work(self, session, *, release_missing: bool) -> bool:
        waiting_by_key = session.stage_fan_in.waiting_tiles
        if not waiting_by_key:
            return False
        cache = self.win.operation_evaluator.stage_cache
        in_flight = getattr(self.win.operation_evaluator.stage_materializer, "_in_flight", {})
        for key, waiting in dict(waiting_by_key).items():
            if not waiting:
                continue
            if key in session.stage_fan_in.values:
                return True
            value = cache.get_containing(key) if hasattr(cache, "get_containing") else cache.get(key)
            if value is not None:
                return True
            if release_missing and key not in in_flight:
                return True
        return False

    def _on_montage_stage_error(self, session_id, key, exc) -> None:
        session = getattr(self, "_montage_session", None)
        self.win.operation_evaluator.stage_materializer.fail(key, exc)
        if session is None or not self._is_current_montage_session(session_id, session.key):
            return
        waiting = list(session.stage_fan_in.fail(key))
        for tile in waiting:
            session.mark_skipped(tile)
        show_status_message(self.win, f"Montage stage update failed: {exc}", timeout=4000)
        self._dispatch_montage_work(session, force=True)

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

    def _schedule_montage_tiles(self, session: MontageRenderSession) -> None:
        if not self._montage_session_is_current(session):
            return
        if hasattr(self.win, "_apply_resource_governor_decisions"):
            self.win._apply_resource_governor_decisions()
        controller = getattr(self.win, "montage_tile_evaluation_controller", self.win.visible_evaluation_controller)
        max_workers = max(1, int(controller.pool.maxThreadCount()))
        while len(session.active_tile_requests) < max_workers:
            scheduled = self._schedule_next_montage_tile(session)
            if not scheduled:
                break

    def _schedule_next_montage_tile(self, session: MontageRenderSession) -> bool:
        if not self._montage_session_is_current(session):
            return False
        tile = session.next_tile()
        if tile is None:
            if session.active_tile_requests or session.loading_tiles or session.pending_completed_tiles:
                return False
            if session.stage_fan_in.active_requests or session.stage_fan_in.waiting_tiles:
                return False
            self._schedule_montage_presentation_commit(session, force=True)
            if session.pending_tiles:
                return self._schedule_next_montage_tile(session)
            if self._finish_montage_session_if_complete(session):
                if _should_defer_montage_side_panels(self, session):
                    self.win._deferred_side_panel_refresh_pending = True
                else:
                    self.win._update_operation_dock()
            return False

        # Reduce-at-ingest (ADR 0050): capture the demand as an immutable
        # snapshot now; the worker reduces the finished tile to that level so
        # its first presentation never uploads a native texture.  A demand
        # change while the tile is in flight is corrected by the ordinary
        # streaming materialization path, not by special cases here.
        ingest_demand = session.ingest_lod_demand()
        ingest_pyramid = getattr(session, "lod_pyramid", None) if ingest_demand is not None else None
        # Semantic identity captured at schedule time: the worker must key the
        # pyramid without touching the live session (ADR 0050 key contract).
        ingest_semantic_id = (
            session.tile_semantic_source_id(tile.source_index) if ingest_pyramid is not None else None
        )
        ingest_state = {"admitted": False}
        preview_pyramid = getattr(session, "lod_preview_pyramid", None)
        preview_level = int(getattr(session, "lod_preview_level", 0) or 0)
        preview_semantic_id = (
            session.tile_semantic_source_id(tile.source_index)
            if preview_pyramid is not None and preview_level > 0 and ingest_semantic_id is None
            else ingest_semantic_id
        )

        def evaluate(token):
            result = self._evaluate_montage_tile_snapshot(session, tile, token)
            if getattr(result, "value", None) is None:
                return result
            rendered_for_lod = None
            reduced = None
            if ingest_pyramid is not None:
                rendered_for_lod = _rendered_tile_from_evaluation_result(tile, result)
                reduced = _admit_ingest_reduction(
                    ingest_pyramid,
                    ingest_demand,
                    rendered_for_lod,
                    semantic_source_id=ingest_semantic_id,
                )
                ingest_state["admitted"] = reduced is not None
            if preview_pyramid is not None and preview_level > 0:
                # Retained preview level (ADR 0050): every evaluated tile
                # leaves a coarse copy in the pinned preview cache, so any
                # index ever computed re-presents instantly through the
                # floor for the lifetime of the dataset view.
                if rendered_for_lod is None:
                    rendered_for_lod = _rendered_tile_from_evaluation_result(tile, result)
                _admit_preview_reduction(
                    preview_pyramid,
                    rendered_for_lod,
                    semantic_source_id=preview_semantic_id,
                    preview_level=preview_level,
                    reduced=reduced,
                    reduced_level=None if ingest_demand is None else int(ingest_demand.desired_level),
                )
            return result

        session_id = int(session.session_id)
        montage_axis = session.montage_axis
        document = session.document
        colormap_lut = session.colormap_lut
        shader_display = bool(getattr(session, "shader_display", False))

        def done(result):
            if ingest_state["admitted"]:
                self._montage_lod_ingest_reductions = (
                    int(getattr(self, "_montage_lod_ingest_reductions", 0) or 0) + 1
                )
                if str(getattr(result, "compute_path", "direct") or "direct") == "stage_backed":
                    # A cached/shared stage output served this tile's reduced
                    # display payload: the expensive pipeline stage was not
                    # re-run for a display-LOD demand (ADR 0050).
                    self._montage_lod_stage_hits_serving_derivations = (
                        int(getattr(self, "_montage_lod_stage_hits_serving_derivations", 0) or 0) + 1
                    )
            self._on_montage_tile_done(
                session_id,
                tile,
                result,
                document=document,
                montage_axis=montage_axis,
                colormap_lut=colormap_lut,
                shader_display=shader_display,
            )

        def error(exc):
            self._on_montage_tile_error(session_id, tile, exc)

        controller = getattr(self.win, "montage_tile_evaluation_controller", self.win.visible_evaluation_controller)
        started = controller.start_latest(
            evaluate,
            key=("montage_tile", session.key, int(tile.montage_index)),
            priority=EvalPriority.VISIBLE_IMAGE,
            replace_group=f"montage-tile:{session_id}:{int(tile.montage_index)}",
            frame_target=session.frame_plan.target,
            supersession_key=("montage-tile", int(tile.montage_index)),
            supersession_value=(session.key, session_id),
            work_item=WorkItem(
                key=("montage_tile_materialization", session.key, session_id, int(tile.montage_index)),
                lane=WorkLane.VISIBLE_MATERIALIZATION,
                frame_target=session.frame_plan.target,
                quality=session.frame_plan.target.quality,
                supersession_key=("montage-tile", int(tile.montage_index)),
                supersession_value=(session.key, session_id),
                estimated_bytes=int(estimate_display_image_bytes(session.frame_plan.tile_shape, dtype=session.output_dtype, rgb=session.rgb)),
                reusable_output=True,
            ),
            on_done=done,
            on_error=error,
            on_stale=lambda: None,
            on_slow=lambda: self._on_montage_tile_slow(session_id),
            slow_ms=100,
            pass_token=True,
        )
        if started is None:
            # Admission declined (visible-lane backpressure): the tile was
            # already dequeued and marked loading, and dropping it here left
            # it "loading" forever with no pending work — the visible plan
            # could then never complete and auto-levels retried commits in a
            # timer loop at idle.  Revert the bookkeeping, requeue, and stop
            # the drain — but NEVER without a wakeup: if this was the last
            # in-flight work for the session, no completion of ours will
            # ever call the pump again (the 2026-07-05 dead-pump field
            # freeze: pending=35, loading=60, active=0).  The capacity
            # waiter re-derives dispatch when any tracked work finishes.
            session.active_tile_requests.discard(int(tile.montage_index))
            session.loading_tiles.discard(int(tile.montage_index))
            _enqueue_session_pending_tile(session, tile)
            self._montage_tile_admission_declined = (
                int(getattr(self, "_montage_tile_admission_declined", 0) or 0) + 1
            )
            controller.notify_when_capacity(
                ("montage-tiles", session.key),
                lambda: self._dispatch_montage_work(getattr(self, "_montage_session", None)),
            )
            return False
        return True

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

    def _evaluate_montage_tile_snapshot(self, session, tile, token=None):
        start = perf_counter()
        context = self.win._evaluation_context(ComputeLane.MONTAGE_TILE, token)
        try:
            stage_key = session.stage_fan_in.tile_stage_keys.get(int(tile.montage_index))
            stage_value = None if stage_key is None else session.stage_fan_in.values.get(stage_key)
            if stage_value is not None:
                request = request_for_image(tile.view_state)
                plan = session.stage_fan_in.tile_stage_plans.get(int(tile.montage_index))
                candidate = session.stage_fan_in.tile_stage_candidates.get(int(tile.montage_index))
                if plan is None or candidate is None:
                    plan = plan_slab(session.document, request)
                    candidates = tuple(getattr(plan.region_plan, "cache_candidates", ()))
                    candidate = next(
                        (
                            candidate
                            for candidate in candidates
                            if self.win.operation_evaluator.stage_materializer.key_for_candidate(stage_document_key(session.document), candidate) == stage_key
                        ),
                        None,
                    )
                if candidate is not None:
                    slab = evaluate_slab_from_stage(
                        session.document,
                        request,
                        plan,
                        stage_value,
                        candidate,
                        cancellation_token=token,
                        evaluation_context=context,
                    )
                    if bool(getattr(session, "shader_display", False)):
                        display_image = make_shader_image_from_slab(
                            slab,
                            request,
                            colormap_lut=session.colormap_lut,
                            provisional_histogram=True,
                        )
                    else:
                        display_image = make_image_from_slab(slab, request, colormap_lut=session.colormap_lut)
                    display_image = _attach_montage_tile_level_stats(
                        display_image,
                        tile,
                        refined=not bool(getattr(session, "shader_display", False)),
                    )
                    return EvaluationResult(
                        value=display_image,
                        eval_ms=(perf_counter() - start) * 1000.0,
                        slab_shape=tuple(np.shape(slab)),
                        slab_nbytes=int(getattr(slab, "nbytes", 0)),
                        region_plan=plan.region_plan,
                        compute_path="stage_backed",
                    )

            result = evaluate_image_snapshot(
                session.document,
                tile.view_state,
                colormap_lut=session.colormap_lut,
                cancellation_token=token,
                shader_display=bool(getattr(session, "shader_display", False)),
                provisional_histogram=bool(getattr(session, "shader_display", False)),
                stage_cache=self.win.operation_evaluator.stage_cache,
                stage_document_key=stage_document_key(session.document),
                evaluation_context=context,
            )
            result = replace(
                result,
                value=_attach_montage_tile_level_stats(
                    result.value,
                    tile,
                    refined=not bool(getattr(session, "shader_display", False)),
                ),
            )
            return result
        finally:
            self._last_montage_tile_eval_ms = (perf_counter() - start) * 1000.0

    def _on_montage_tile_slow(self, session_id):
        session = getattr(self, "_montage_session", None)
        # Not the shared predicate: on_slow callbacks capture only the session
        # id, so this is intentionally an id-only currency check.
        if session is None or int(session.session_id) != int(session_id):
            return
        self._show_montage_session_loading_overlay(session)

    def _schedule_montage_session_slow_overlay(self, session):
        timer = self._ensure_montage_session_slow_timer()
        self._montage_session_slow_key = (int(session.session_id), session.key)
        timer.start(100)

    def _ensure_montage_session_slow_timer(self):
        timer = getattr(self, "_montage_session_slow_timer", None)
        if timer is None:
            # User-visible timeout. The session key check in the callback
            # prevents stale overlays; remove only if progress UI becomes tied
            # to explicit scheduler deadline events.
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._on_montage_session_slow_timer)
            self._montage_session_slow_timer = timer
        return timer

    def _stop_montage_session_slow_overlay(self):
        timer = getattr(self, "_montage_session_slow_timer", None)
        if timer is not None:
            timer.stop()
        self._montage_session_slow_key = None

    def _on_montage_session_slow_timer(self):
        key = getattr(self, "_montage_session_slow_key", None)
        if key is None:
            return
        session_id, session_key = key
        self._show_montage_session_loading_overlay_if_current(session_id, session_key)

    def _show_montage_session_loading_overlay_if_current(self, session_id, key):
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session_id, key):
            return
        if session.visible_plan_complete():
            return
        if not session.pending_tiles and not session.loading_tiles and not session.stage_fan_in.attached_requests:
            return
        self._show_montage_session_loading_overlay(session)

    def _show_montage_session_loading_overlay(self, session):
        if not self._montage_session_is_current(session):
            return
        if session.visible_plan_complete():
            return
        session.show_loading_overlays = True
        self._schedule_montage_presentation_commit(session, force=True)
        self.win.img_view.setImageStale(True)
        self.win.img_view.setEvaluationOverlay(True, "Updating image frame...")
        self.win.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.COMPUTING, "Evaluating image frame")
        if getattr(session, "defer_side_panels", False) or _viewport_interaction_active(self):
            self.win._deferred_side_panel_refresh_pending = True
        else:
            self.win._update_operation_dock()

    def _on_montage_tile_done(self, session_id, tile, result, *, document, montage_axis: int | None, colormap_lut, shader_display: bool) -> None:
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session_id, session.key):
            self._store_reusable_montage_tile_result(
                tile,
                result,
                document=document,
                montage_axis=montage_axis,
                colormap_lut=colormap_lut,
                shader_display=shader_display,
            )
            return
        if not self._is_current_render_generation(session.render_generation):
            self._store_reusable_montage_tile_result(
                tile,
                result,
                document=document,
                montage_axis=montage_axis,
                colormap_lut=colormap_lut,
                shader_display=shader_display,
            )
            return
        session.pending_completed_tiles.append((tile, result))
        self._schedule_montage_tile_result_flush(session)

    def _store_reusable_montage_tile_result(self, tile, result, *, document, montage_axis: int | None, colormap_lut, shader_display: bool):
        if _document_key(document) != _document_key(self.win.document):
            return None
        stored = self.win.operation_evaluator.store_montage_tile_result(
            tile,
            montage_axis=montage_axis,
            colormap_lut=colormap_lut,
            result=result,
            document=document,
            shader_display=shader_display,
        )
        controller = getattr(self.win, "montage_tile_evaluation_controller", None)
        if stored is not None and controller is not None and hasattr(controller, "note_stale_reused"):
            controller.note_stale_reused()
        return stored

    def _schedule_montage_tile_result_flush(self, session) -> None:
        if not self._montage_session_is_current(session):
            return
        self._montage_tile_result_key = (int(session.session_id), session.key)
        self._montage_tile_result_token = _montage_work_token(session, "tile_result")
        if _persistent_tile_layer_fast_drain_enabled(self, session):
            if not bool(getattr(self, "_montage_tile_result_flush_queued", False)):
                self._montage_tile_result_flush_queued = True
                try:
                    queued = Qt.QtCore.QMetaObject.invokeMethod(
                        self,
                        "_flush_montage_tile_results",
                        Qt.QtCore.Qt.ConnectionType.QueuedConnection,
                    )
                except Exception:
                    queued = False
                if queued:
                    return
                self._montage_tile_result_flush_queued = False
        timer = getattr(self, "_montage_tile_result_timer", None)
        if timer is None:
            # Bounded continuation guarded by `_montage_tile_result_token`.
            # Ready worker bursts are fan-in work, not a license to mutate the
            # whole scene in one GUI callback.
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._flush_montage_tile_results)
            self._montage_tile_result_timer = timer
        if not timer.isActive():
            timer.start(0)

    @Qt.QtCore.Slot()
    def _flush_montage_tile_results(self) -> None:
        self._montage_tile_result_flush_queued = False
        key = getattr(self, "_montage_tile_result_key", None)
        session = getattr(self, "_montage_session", None)
        if session is None or key is None or not self._is_current_montage_session(key[0], key[1]):
            return
        token = getattr(self, "_montage_tile_result_token", None)
        if not _montage_work_token_is_current(session, token, "tile_result"):
            return
        interactive = _interactive_active(self)
        budget = self._montage_callback_budget(
            "montage_tile_result",
            interactive=interactive,
            work_class="ready_tile_fan_in",
            item_cap=_montage_tile_result_batch_limit(self, interactive=interactive),
        )
        flush_start = perf_counter()
        processed = 0
        processed_tiles = []
        first_vispy_display = bool(
            _persistent_tile_residency_backend(self, session)
            and not getattr(session, "display_committed", False)
        )
        expected_indices = self._montage_level_expected_indices(session)
        while session.pending_completed_tiles:
            tile, result = session.pending_completed_tiles.popleft()
            byte_count = self._apply_montage_tile_result(session, tile, result, expected_indices=expected_indices)
            processed_tiles.append(tile)
            processed += 1
            budget.record_item(byte_count=byte_count)
            if budget.should_yield():
                break
        if processed:
            elapsed_ms = (perf_counter() - flush_start) * 1000.0
            self._last_montage_tile_result_flush_ms = elapsed_ms
            self._last_montage_tile_result_flush_count = int(processed)
            _complete_inline_work(
                self,
                WorkItem(
                    key=("montage_tile_result_fan_in", session.key, int(session.session_id), tuple(int(tile.montage_index) for tile in processed_tiles)),
                    lane=WorkLane.GUI_FAN_IN,
                    frame_target=session.frame_plan.target,
                    supersession_key=("montage-fan-in", session.key),
                    supersession_value=int(session.session_id),
                    estimated_cpu_ms=float(elapsed_ms),
                    estimated_bytes=int(budget.processed_bytes),
                ),
            )
            self._record_gui_budget(budget)
            first_visible_committed = False
            if first_vispy_display and (getattr(session, "dirty_tiles", None) or getattr(session, "dirty_payloads", None)):
                self._commit_montage_session_presentation(session, force=False)
                first_visible_committed = bool(getattr(session, "display_committed", False))
            self._activate_cached_waiting_stages(session, release_missing=True)
            self._schedule_montage_cached_level_stats(session)
            if session.pending_tiles:
                self._schedule_montage_tiles(session)
            force = not session.pending_tiles and not session.active_tile_requests and not session.pending_completed_tiles
            if first_visible_committed:
                pass
            elif force:
                self._schedule_montage_presentation_commit(session, force=True)
            else:
                self._schedule_montage_ready_display_commit(session)
        if session.pending_completed_tiles:
            self._schedule_montage_tile_result_flush(session)
        # Machine-derived dispatch (ADR 0051 P2): re-derive everything the
        # records imply after this fan-in edge — nothing above may be the
        # only holder of a pending record.
        self._dispatch_montage_work(session)

    def _apply_montage_tile_result(self, session, tile, result, *, expected_indices=None) -> int:
        if not self._montage_session_is_current(session):
            return 0
        if not self._is_current_render_generation(session.render_generation):
            return 0
        # The semantic display cache is the reuse point for every later
        # demand on this tile (session rebuilds, viewport re-entry, display
        # LOD changes).  The fast-drain path used to skip this store, so
        # tiles evaluated during a settled VisPy drain were reachable only as
        # backend-acknowledged payloads; once those were superseded or
        # evicted, a display-LOD-driven request re-ran the full pipeline
        # (observed as occasional per-tile FFT re-runs).  Level stats are
        # attached worker-side, so the store is a cache put, cheap enough for
        # every drain mode (ADR 0050).
        rendered = self.win.operation_evaluator.store_montage_tile_result(
            tile,
            montage_axis=session.montage_axis,
            colormap_lut=session.colormap_lut,
            result=result,
            shader_display=bool(getattr(session, "shader_display", False)),
        )
        compute_path = str(getattr(result, "compute_path", "direct") or "direct")
        eval_ms = max(0.0, float(getattr(result, "eval_ms", 0.0) or 0.0))
        if compute_path == "stage_backed":
            session.tile_compute_stage_backed += 1
            session.tile_compute_stage_backed_ms += eval_ms
            session.tile_compute_stage_backed_max_ms = max(float(session.tile_compute_stage_backed_max_ms), eval_ms)
        else:
            session.tile_compute_direct += 1
            session.tile_compute_direct_ms += eval_ms
            session.tile_compute_direct_max_ms = max(float(session.tile_compute_direct_max_ms), eval_ms)
        self._update_montage_level_bounds_from_rendered(
            session.level_key,
            rendered,
            expected_indices=self._montage_level_expected_indices(session) if expected_indices is None else expected_indices,
        )
        self._queue_montage_level_refinement(session, rendered)
        session.mark_loaded(rendered)
        patch_start = perf_counter()
        session.dirty_tiles.append(int(tile.montage_index))
        return _rendered_tile_nbytes(rendered)

    def _on_montage_tile_error(self, session_id, tile, exc) -> None:
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session_id, session.key):
            return
        if not self._is_current_render_generation(session.render_generation):
            return
        session.mark_skipped(tile)
        show_status_message(self.win, f"Montage tile update failed: {exc}", timeout=4000)
        self._dispatch_montage_work(session, force=True)

    def _montage_lod_policy_mode(self) -> str:
        return montage_lod.policy_mode_for_renderer(self)

    def _montage_lod_pyramid(self) -> PyramidCache:
        return montage_lod.shared_pyramid(self)

    def _montage_lod_preview_pyramid(self) -> PyramidCache:
        return montage_lod.preview_pyramid(self)

    def _schedule_montage_lod_materializations(self, session) -> None:
        return montage_lod.schedule_materializations(self, session)

    # -- machine-derived dispatch (ADR 0051 P2) -------------------------------

    def _dispatch_montage_work(self, session, *, force: bool = False) -> None:
        """The single montage pump: schedule everything the records imply.

        Every montage event edge (tile done/error, stage done/stale/error,
        LOD level ready, result flush, admission-decline wakeup, watchdog
        assertion) ends here.  `derive_montage_dispatch` is the one decision
        site; the schedulers below are idempotent and coalesced, so redundant
        dispatch is cheap.  A state mutation that does NOT end in dispatch is
        the ADR 0051 lost-wakeup defect — the watchdog assertion reports it.
        """

        if session is None or not self._montage_session_is_current(session):
            return
        plan = derive_montage_dispatch(session)
        if plan.requeue_orphans:
            requeued = session.requeue_orphaned_loading_tiles()
            if requeued:
                self._montage_orphaned_tiles_repaired = (
                    int(getattr(self, "_montage_orphaned_tiles_repaired", 0) or 0) + int(requeued)
                )
                plan = derive_montage_dispatch(session)
        if plan.deferred_planning:
            self._schedule_deferred_montage_planning(session)
        if plan.schedule_tiles:
            self._schedule_montage_tiles(session)
        if plan.flush_results:
            self._schedule_montage_tile_result_flush(session)
        if plan.stage_waits:
            self._schedule_montage_attached_stage_waits(session)
        if plan.lod_materializations:
            self._schedule_montage_lod_materializations(session)
        if plan.level_evidence:
            self._schedule_montage_cached_level_stats(session)
        if plan.commit or force:
            self._schedule_montage_presentation_commit(
                session, force=bool(plan.force_commit or force)
            )
        if plan.unsettled:
            self._ensure_montage_watchdog()

    # -- stall watchdog (ADR 0051): an ASSERTION, not a repair -----------------
    # Machine-derived dispatch above makes a dead pump impossible by
    # construction: every event edge re-derives all scheduled work from the
    # session records, and a declined admission arms a capacity waiter on the
    # controller.  The watchdog remains armed while a session is unsettled
    # purely to catch violations of that construction: a zero-progress tick
    # increments `_montage_stall_repairs` (JSONL: stall_repairs — every count
    # is a lost-wakeup bug report, asserted zero in the GPU harness) and then
    # rescues via the ordinary dispatch, never a bespoke repair path.

    def _ensure_montage_watchdog(self) -> None:
        timer = getattr(self, "_montage_watchdog_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setInterval(1000)
            timer.timeout.connect(self._montage_watchdog_tick)
            self._montage_watchdog_timer = timer
            self._montage_watchdog_state = None
        if not timer.isActive():
            self._montage_watchdog_state = None
            timer.start()

    def _montage_watchdog_stop(self) -> None:
        timer = getattr(self, "_montage_watchdog_timer", None)
        if timer is not None:
            timer.stop()
        self._montage_watchdog_state = None

    @Qt.QtCore.Slot()
    def _montage_watchdog_tick(self) -> None:
        session = getattr(self, "_montage_session", None)
        if session is None or not self._montage_session_is_current(session):
            self._montage_watchdog_stop()
            return
        pending = len(session.pending_tiles)
        evaluating = len(session.lifecycle.evaluating_tiles)
        active = len(session.active_tile_requests)
        dirty = len(session.dirty_payloads)
        upserts = len(session.pending_payload_upserts)
        lod_pending = len(getattr(session, "pending_lod_requests", ()) or ())
        planning_deferred = bool(getattr(session, "stage_planning_deferred", False))
        level_evidence = len(getattr(session, "pending_level_tiles", ()) or ()) + int(
            getattr(session, "level_scan_remaining_tiles", 0) or 0
        )
        # Level-value convergence drains stale tiles through budgeted commits
        # without touching dirty/upsert queues; its progress must be part of
        # the stall signature or a long drain reads as a frozen session.
        level_stale = (
            int(session.level_presentation_snapshot().stale_count)
            if session.has_pending_level_update()
            else 0
        )
        unsettled = bool(
            pending
            or evaluating
            or active
            or dirty
            or upserts
            or lod_pending
            or planning_deferred
            or level_evidence
            or level_stale
            or session.flush_pending
            or session.final_commit_pending
        )
        if not unsettled:
            self._montage_watchdog_stop()
            return
        if planning_deferred:
            # Deferred planning is scheduled work, not a stall: its
            # continuation chain re-arms itself while interaction lasts.
            # Re-kick it (idempotent — the completion no-ops once the flag
            # clears) instead of counting a repair.
            self._schedule_deferred_montage_planning(session)
            return
        level_timer = getattr(self, "_montage_level_stats_timer", None)
        if level_evidence and level_timer is not None and level_timer.isActive():
            # Level evidence with its continuation timer armed is scheduled
            # work; a busy event loop can hold the drain across watchdog
            # ticks (pyqtgraph fft commits block for hundreds of ms).  A
            # lost wakeup is the timer NOT being armed while records imply
            # level work — that still fires below.
            return
        signature = (
            int(session.session_id),
            pending,
            evaluating,
            active,
            dirty,
            upserts,
            lod_pending,
            level_evidence,
            level_stale,
            len(session.presented_tiles),
            len(session.rendered_tiles),
        )
        previous = getattr(self, "_montage_watchdog_state", None)
        self._montage_watchdog_state = signature
        if previous != signature:
            return  # work is progressing; stay armed.
        self._montage_stall_repairs = int(getattr(self, "_montage_stall_repairs", 0) or 0) + 1
        # ASSERTION FIRED: a state mutation escaped the dispatch construction.
        # Keep the frozen signature visible for root-causing and rescue via
        # the ordinary dispatch — if dispatch cannot rescue it, the defect is
        # in the derivation, which is the bug report we want.
        self._montage_watchdog_last_stall = signature
        print(
            "[arrayscope] STALL WATCHDOG FIRED (lost wakeup, ADR 0051): "
            f"signature={signature} "
            f"stage_active={len(session.stage_fan_in.active_requests)} "
            f"stage_attached={len(session.stage_fan_in.attached_requests)} "
            f"stage_waiting={len(session.stage_fan_in.waiting_tiles)} "
            f"loading={len(session.loading_tiles)} "
            f"flush_pending={session.flush_pending} final={session.final_commit_pending}",
            file=sys.stderr,
            flush=True,
        )
        if session.refresh_lod_for_viewport():
            self._schedule_montage_presentation_commit(session, force=False)
        self._dispatch_montage_work(session, force=bool(dirty or upserts))

    def _on_montage_lod_level_ready(self, session_id, session_key, tile_number) -> None:
        return montage_lod.on_level_ready(self, session_id, session_key, tile_number)

    def _schedule_montage_presentation_commit(self, session, *, force=False) -> None:
        if not self._montage_session_is_current(session):
            return
        self._montage_commit_token = _montage_work_token(session, "commit")
        interval_ms = self._montage_commit_interval_ms(session, force=force)
        elapsed_ms = (monotonic() - float(session.last_commit_monotonic or 0.0)) * 1000.0
        needs_initial_commit = not bool(getattr(session, "display_committed", False))
        needs_final_dirty_commit = bool(
            force
            and not session.pending_tiles
            and not session.active_tile_requests
            and not session.pending_completed_tiles
            and (
                getattr(session, "dirty_payloads", None)
                or getattr(session, "pending_payload_upserts", None)
                or getattr(session, "pending_removals", None)
                or (session.has_pending_level_update() and session.has_stale_level_presentations())
            )
        )
        if _viewport_interaction_active(self) and not needs_initial_commit:
            session.final_commit_pending = True
            session.flush_pending = True
            self._start_montage_commit_timer(max(1, int(interval_ms)))
            return
        if needs_initial_commit or needs_final_dirty_commit or force and not session.flush_pending or elapsed_ms >= interval_ms:
            self._commit_montage_session_presentation(session, force=force)
            return
        session.final_commit_pending = True
        session.flush_pending = True
        self._montage_coalesced_commits = int(getattr(self, "_montage_coalesced_commits", 0) or 0) + 1
        self._start_montage_commit_timer(max(1, int(interval_ms - elapsed_ms)))

    def _start_montage_commit_timer(self, interval_ms: int) -> None:
        timer = getattr(self, "_montage_commit_timer", None)
        if timer is None:
            # Bounded continuation guarded by `_montage_commit_token`.
            # Commit spacing comes from feedback/resource policy; remove when
            # backend commits are scheduled directly as WorkGraph callbacks.
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._flush_montage_presentation_commit)
            self._montage_commit_timer = timer
        if not timer.isActive():
            timer.start(max(1, int(interval_ms)))

    def _montage_commit_interval_ms(self, session, *, force: bool) -> int:
        if force:
            if not session.pending_tiles and not session.loading_tiles and not session.active_tile_requests:
                return _montage_commit_interval_ms(self, force=True)
        interval = _montage_commit_interval_ms(self, force=False)
        if _persistent_tile_layer_fast_drain_enabled(self, session):
            return min(int(interval), 8)
        return interval

    @Qt.QtCore.Slot()
    def _flush_montage_presentation_commit(self):
        self._montage_presentation_commit_flush_queued = False
        session = getattr(self, "_montage_session", None)
        if session is None or not session.final_commit_pending:
            return
        token = getattr(self, "_montage_commit_token", None)
        if not _montage_work_token_is_current(session, token, "commit"):
            return
        self._commit_montage_session_presentation(session, force=False)
        repaired = session.requeue_orphaned_loading_tiles()
        if repaired:
            self._montage_orphaned_tiles_repaired = (
                int(getattr(self, "_montage_orphaned_tiles_repaired", 0) or 0) + int(repaired)
            )
            self._schedule_montage_tiles(session)
        if session.stage_fan_in.waiting_tiles and not session.stage_fan_in.active_requests:
            # Tiles are waiting on a stage no scheduler knows about (lost to
            # declined admission or supersession): replan and reschedule the
            # stage work, mirroring the orphaned-tile repair.
            waiting_tiles = [
                tile
                for tiles in session.stage_fan_in.waiting_tiles.values()
                for tile in tuple(tiles)
            ]
            if waiting_tiles:
                stage_plan = self._plan_montage_stages(session.document, waiting_tiles)
                self._schedule_montage_stage_jobs(session, stage_plan["stage_requests"])
                self._montage_orphaned_stages_repaired = (
                    int(getattr(self, "_montage_orphaned_stages_repaired", 0) or 0) + 1
                )
        if (
            not session.pending_tiles
            and not session.loading_tiles
            and not session.active_tile_requests
            and not session.stage_fan_in.attached_requests
        ):
            # Settle repair (field defect 2026-07-05, JSONL 110937 sid=80):
            # materializations for the demanded level die to supersession
            # during zoom bounces, and refresh_lod_for_viewport only runs on
            # camera events — at idle nothing re-requested the missing level,
            # so tiles wedged on a coarser resident level until the next pan.
            # Re-evaluate demand as the last work drains; singleflight claims
            # make this idempotent, and this flush only runs when new work
            # completed, so a repeatedly blocked admission cannot loop.
            if session.refresh_lod_for_viewport():
                self._schedule_montage_presentation_commit(session, force=False)
            if getattr(session, "pending_lod_requests", None):
                self._schedule_montage_lod_materializations(session)
        if (
            getattr(session, "show_loading_overlays", False)
            and not session.visible_plan_complete()
            and (session.pending_tiles or session.loading_tiles or session.active_tile_requests or session.stage_fan_in.attached_requests)
        ):
            self.win.img_view.setImageStale(True)
            self.win.img_view.setEvaluationOverlay(True, "Updating image frame...")

    def _commit_montage_session_presentation(self, session, *, force=False) -> None:
        if not self._montage_session_is_current(session):
            return
        commit_start = perf_counter()
        self._classify_visible_montage_tiles(session)
        direct_presentation = self._direct_montage_tile_layer_presentation(session)
        if direct_presentation is None:
            raise RuntimeError("montage presentation could not be built")
        self._commit_montage_session_tile_layer(session, direct_presentation, commit_start=commit_start)

    def _direct_montage_tile_layer_presentation(self, session):
        tile_states = session.ensure_tile_states()
        placeholder = _montage_tile_layer_placeholder(session)
        geometry = DisplayGeometry(
            view_state=session.view_state,
            display_shape=tuple(placeholder.shape[:2]),
            montage=session.plan.geometry,
            montage_origin_x=0,
            montage_origin_y=0,
            montage_tile_states=tile_states,
        )
        decision = self._montage_backend_policy(geometry, placeholder)
        if decision.backend != "tile_layer":
            return None
        return DisplayImage(data=placeholder, histogram_data=None, rgb_already_windowed=False), geometry

    def _commit_montage_session_tile_layer(self, session, direct_presentation, *, commit_start: float) -> None:
        display_image, rendered_geometry = direct_presentation
        dirty_tiles = session.consume_dirty_tiles()
        tile_source_ids = self._montage_tile_source_ids(session)
        self._montage_committed_tile_upserts_last_flush = len(dirty_tiles)
        self._current_montage_geometry = session.plan.geometry
        self._current_montage_plan = session.plan
        self._next_viewport_policy = ViewportPolicy.PRESERVE
        self._montage_presentation_commit_active = True
        try:
            payload_start = perf_counter()
            selected_lod_factor = int(session._selected_lod_factor())
            # ADR 0050: seeding a fresh session must keep whatever level the
            # previous session presented on screen; the resident policy swaps
            # levels afterwards through ordinary payload-identity commits.
            reuse_any_lod = bool(getattr(session, "_resident_lod_active", lambda: False)())
            # The previous-frame and retained-store scans exist only to seed
            # a session that has never presented; for every later flush (and
            # for retargeted sessions, whose payloads persist in-session)
            # they were O(committed frame + retained store) per commit for
            # nothing.
            if not getattr(session, "presented_tiles", None):
                previous_payloads = {
                    int(tile): payload
                    for tile, payload in _previous_tiled_payloads(getattr(self.win, "_committed_display_frame", None)).items()
                    if reuse_any_lod or _payload_lod_matches(payload, selected_lod_factor)
                }
                retained_payloads = self._retained_tiled_payload_store().payloads_by_base_source(
                    lod_factor=None if reuse_any_lod else selected_lod_factor
                )
                if retained_payloads:
                    previous_payloads.update(
                        {
                            int(tile): payload
                            for tile, payload in enumerate(retained_payloads.values())
                        }
                    )
                if previous_payloads:
                    session.seed_display_tile_payloads(previous_payloads, tile_source_ids)
                    if reuse_any_lod:
                        # Converge seeded stale-level payloads to the live
                        # demand: mismatched tiles become dirty and rebuild
                        # below; missing levels are queued and drained after
                        # this commit.
                        session.refresh_lod_for_viewport()
            base_tile_state = session.tile_presentation_state
            fast_drain = _persistent_tile_layer_fast_drain_enabled(self, session)
            self._persistent_tile_layer_fast_drain_last_enabled = bool(fast_drain)
            self._persistent_tile_layer_fast_drain_enabled_count = int(
                getattr(self, "_persistent_tile_layer_fast_drain_enabled_count", 0) or 0
            ) + int(bool(fast_drain))
            tile_layer_limits = _tile_layer_upsert_limits(self, session)
            tile_state, tile_delta = session.build_tile_presentation(
                tile_source_ids,
                cold_deadline_ms=(
                    _montage_commit_budget_ms(self)
                    if tile_layer_limits
                    else None
                ),
                **tile_layer_limits,
            )
            if _persistent_tile_residency_backend(self, session):
                dirty_tiles = tuple(int(tile) for tile in dirty_tiles if int(tile) in set(tile_delta.upserts))
            active_payloads = tile_state.active_payloads(tile_delta)
            first_display_commit = not bool(session.display_committed)
            requested_levels = _session_requested_levels(session)
            explicit_auto = bool(getattr(session, "force_auto", False) and requested_levels is None)
            if (
                not first_display_commit
                and not explicit_auto
                and not getattr(tile_delta, "force_refresh", False)
                and not getattr(tile_delta, "upserts", None)
                and not getattr(tile_delta, "removals", None)
                and not bool(getattr(session, "presentation_geometry_changed", False))
                and not (session.has_pending_level_update() and session.has_stale_level_presentations())
            ):
                session.final_commit_pending = False
                session.flush_pending = False
                self._last_montage_tile_payload_build_ms = (perf_counter() - payload_start) * 1000.0
                session.note_committed()
                self._settle_montage_visible_plan_if_complete(session)
                self._finish_montage_session_if_complete(session)
                if not (getattr(session, "dirty_payloads", None) or getattr(session, "pending_removals", None)):
                    schedule_near_viewport_montage_prefetch(self, session)
                self._schedule_montage_lod_materializations(session)
                self._retry_live_profile_after_montage_tile()
                return
            level_payloads = active_payloads if first_display_commit else dict(tile_delta.upserts)
            self._queue_montage_level_stats_for_payloads(session, level_payloads)
            rendered_geometry = replace(
                rendered_geometry,
                montage_tile_states=session.ensure_tile_states(),
            )
            self._last_montage_tile_payload_build_ms = (perf_counter() - payload_start) * 1000.0
            level_stats = self._montage_level_stats_for_session(session)
            semantic_commit = bool(active_payloads)
            decision_force_auto = bool(explicit_auto and semantic_commit)
            if _tile_layer_auto_levels_wait_for_complete_source(self, session, decision_force_auto, level_stats):
                session.final_commit_pending = True
                session.flush_pending = True
                # Rule 6: this park waits on level evidence, so ARM the
                # producer.  When no level work is queued (stats absent for a
                # fresh level key whose upserts were all consumed already),
                # mark the session scan pending — otherwise the continuation
                # below is a no-op and the parked flush can never re-commit
                # (the pyqtgraph+resident auto-levels wedge).
                if (
                    not getattr(session, "pending_level_tiles", None)
                    and int(getattr(session, "level_scan_remaining_tiles", 0) or 0) <= 0
                ):
                    self._mark_montage_level_scan_pending(session)
                self._schedule_montage_cached_level_stats(session)
                return
            publish_histogram_plot = _should_publish_montage_histogram_plot(
                first_display_commit,
                explicit_auto,
                level_stats,
                requires_semantic_plot=_tiled_payloads_require_semantic_histogram_plot(active_payloads),
            )
            publish_metadata = (
                bool(explicit_auto)
                or publish_histogram_plot
                or self._should_publish_montage_level_metadata(session, level_stats)
            )
            semantic_source = self._montage_level_source_for_session(session, allow_partial=publish_metadata)
            histogram_plot_data = (
                self._montage_histogram_plot_data_for_session(session, allow_partial=publish_metadata)
                if publish_histogram_plot
                else None
            )
            if first_display_commit:
                self._apply_full_display_image(
                    display_image,
                    geometry=rendered_geometry,
                    window_mode=session.window_mode,
                    previous_frame=getattr(self.win, "_committed_display_frame", None),
                    force_auto=decision_force_auto,
                    defer_side_panels=getattr(session, "defer_side_panels", False),
                    semantic_source=semantic_source,
                    applied_level_source=session.applied_level_source,
                    histogram_plot_data=histogram_plot_data,
                    commit_kind=CommitKind.EXPLICIT_AUTO_WINDOW if decision_force_auto else CommitKind.FULL_FRAME_INITIAL,
                    document_key=_document_key(session.document),
                    request_key=session.key,
                    render_generation=session.render_generation,
                    montage_level_key=session.level_key,
                    tile_state=tile_state,
                    base_tile_state=base_tile_state,
                    tile_delta=tile_delta,
                    frame_plan=session.frame_plan,
                    user_levels=requested_levels,
                    semantic_commit=semantic_commit,
                )
            elif fast_drain and self._commit_vispy_montage_tile_delta_direct(
                session,
                display_image,
                rendered_geometry,
                tile_state=tile_state,
                base_tile_state=base_tile_state,
                tile_delta=tile_delta,
                semantic_source=semantic_source,
                applied_level_source=session.applied_level_source,
                histogram_plot_data=histogram_plot_data,
                explicit_auto=explicit_auto,
                semantic_commit=semantic_commit,
            ):
                pass
            else:
                self._apply_progressive_display_image(
                    display_image,
                    geometry=rendered_geometry,
                    window_mode=session.window_mode,
                    previous_frame=getattr(self.win, "_committed_display_frame", None),
                    force_auto=False,
                    viewport_policy=ViewportPolicy.PRESERVE,
                    semantic_source=semantic_source,
                    applied_level_source=session.applied_level_source,
                    histogram_plot_data=histogram_plot_data,
                    commit_kind=CommitKind.EXPLICIT_AUTO_WINDOW if explicit_auto else CommitKind.PROGRESSIVE_FRAME_PATCH,
                    document_key=_document_key(session.document),
                    request_key=session.key,
                    render_generation=session.render_generation,
                    montage_level_key=session.level_key,
                    tile_state=tile_state,
                    base_tile_state=base_tile_state,
                    tile_delta=tile_delta,
                    frame_plan=session.frame_plan,
                    user_levels=requested_levels,
                    semantic_commit=semantic_commit,
                )
            report = getattr(self._display_committer(), "last_tile_commit_report", None)
            acknowledged = session.acknowledge_tile_presentation(tile_delta, report, levels=normalize_bounds(self.win.img_view.getLevels()))
            self._retained_tiled_payload_store().remember_acknowledged(acknowledged.payloads)
            if not session.has_stale_level_presentations():
                session.set_level_update_pending(False)
            presented_tiles = active_payloads if report is None else getattr(report, "presented_tiles", active_payloads)
            session.mark_presented(presented_tiles)
            session.display_committed = bool(session.presented_tiles)
            rendered_geometry = replace(
                rendered_geometry,
                montage_tile_states=session.ensure_tile_states(),
            )
            self._sync_committed_montage_geometry(rendered_geometry, semantic_commit=bool(semantic_commit))
            if first_display_commit:
                # The first commit rescales the viewport to the montage; the
                # queues were prioritized against the pre-montage range.
                self._refresh_montage_priority_targets(session)
            overlay_start = perf_counter()
            rect = montage_rect_for_viewport(session.plan, view_range=session.view_range, viewport_shape=session.viewport_shape)
            self._update_montage_tile_overlays_for_plan(session.plan, tuple(session.tile_states), rect)
            self._last_montage_overlay_update_ms = (perf_counter() - overlay_start) * 1000.0
        finally:
            self._montage_presentation_commit_active = False
        self._schedule_montage_lod_materializations(session)
        # Convergence must be event-driven, not luck-driven (field defect
        # 2026-07-05 #3): the report acknowledged above is the FIRST evidence
        # of drawn-slot identities, and the reconciliation that consumes it
        # runs inside the NEXT commit.  A settled session (dirty/upserts
        # empty) got no next commit, freezing backend_stale_identities
        # nonzero until a pan happened to schedule one.  Bounded by the
        # resigned-pair and attempt limits inside the query, so a backend
        # that cannot converge stops re-scheduling after the retry budget.
        if session.backend_identity_mismatch_tiles():
            self._montage_identity_repair_commits = (
                int(getattr(self, "_montage_identity_repair_commits", 0) or 0) + 1
            )
            self._schedule_montage_presentation_commit(session, force=False)
        self._last_montage_tile_commit_ms = (perf_counter() - commit_start) * 1000.0
        report = getattr(self._display_committer(), "last_tile_commit_report", None)
        _complete_inline_work(
            self,
            WorkItem(
                key=("montage_backend_commit", session.key, int(session.session_id), int(tile_state.revision), "tile_layer"),
                lane=WorkLane.BACKEND_COMMIT,
                frame_target=session.frame_plan.target,
                supersession_key=("montage-backend-commit", session.key),
                supersession_value=int(session.session_id),
                estimated_cpu_ms=float(self._last_montage_tile_commit_ms),
                estimated_bytes=int(getattr(report, "texture_upload_bytes", 0) or 0),
            ),
        )
        cold_count = int(getattr(report, "cold_count", 0) or 0)
        warm_count = int(getattr(report, "existing_items_shown", 0) or 0) + int(getattr(report, "relocated_tiles", 0) or 0)
        processed_count = max(1, cold_count + warm_count)
        texture_upload_bytes = int(getattr(report, "texture_upload_bytes", 0) or 0)
        storage_rebuilds = int(getattr(report, "storage_rebuilds", 0) or 0)
        backend_name = image_view_backend_capabilities(self.win.img_view).name
        feedback = _latency_feedback(self)
        if feedback is not None:
            cold_ms = 0.0
            if cold_count > 0 and storage_rebuilds <= 0:
                cold_ms = float(getattr(report, "cold_work_ms", 0.0) or 0.0) or self._last_montage_tile_commit_ms
                if hasattr(self.win, "_record_ui_work"):
                    self.win._record_ui_work(
                        "montage_cold_commit",
                        cold_ms,
                        count=cold_count,
                        byte_count=texture_upload_bytes,
                        work_class="texture_upload",
                        backend=backend_name,
                    )
                else:
                    feedback.observe(
                        "montage_cold_commit",
                        cold_ms,
                        count=cold_count,
                        byte_count=texture_upload_bytes,
                    )
            commit_feedback_ms = self._last_montage_tile_commit_ms
            commit_feedback_bytes = texture_upload_bytes
            vispy_backend = backend_name.lower() == "vispy"
            vispy_uniform_only = bool(
                vispy_backend
                and cold_count == 0
                and warm_count == 0
                and texture_upload_bytes == 0
            )
            if vispy_backend and storage_rebuilds > 0 and cold_count > 0:
                if hasattr(self.win, "_record_ui_work"):
                    self.win._record_ui_work(
                        "montage_layout_commit",
                        self._last_montage_tile_commit_ms,
                        count=processed_count,
                        byte_count=texture_upload_bytes,
                        work_class="presentation_layout",
                        backend=backend_name,
                    )
                else:
                    feedback.observe(
                        "montage_layout_commit",
                        self._last_montage_tile_commit_ms,
                        count=processed_count,
                        byte_count=texture_upload_bytes,
                    )
                commit_feedback_ms = 0.0
                commit_feedback_bytes = 0
            elif vispy_backend and (cold_count > 0 or vispy_uniform_only):
                if hasattr(self.win, "_record_ui_work"):
                    self.win._record_ui_work(
                        "montage_present_total",
                        self._last_montage_tile_commit_ms,
                        count=processed_count,
                        byte_count=texture_upload_bytes,
                        work_class="presentation_upsert",
                        backend=backend_name,
                    )
                else:
                    feedback.observe(
                        "montage_present_total",
                        self._last_montage_tile_commit_ms,
                        count=processed_count,
                        byte_count=texture_upload_bytes,
                    )
                commit_feedback_ms = max(0.0, self._last_montage_tile_commit_ms - cold_ms)
                commit_feedback_bytes = 0
                if vispy_uniform_only:
                    commit_feedback_ms = 0.0
            if hasattr(self.win, "_record_ui_work"):
                self.win._record_ui_work(
                    "montage_commit",
                    commit_feedback_ms,
                    count=processed_count,
                    byte_count=commit_feedback_bytes,
                    work_class="presentation_upsert",
                    backend=backend_name,
                )
            else:
                feedback.observe(
                    "montage_commit",
                    commit_feedback_ms,
                    count=processed_count,
                    byte_count=commit_feedback_bytes,
                )
        upload_backlog = bool(
            getattr(session, "dirty_payloads", None)
            or getattr(session, "pending_removals", None)
            or getattr(session, "pending_payload_upserts", None)
            or (session.has_pending_level_update() and session.has_stale_level_presentations())
        )
        session.note_committed()
        self._notify_file_session_montage_committed()
        if upload_backlog:
            self._schedule_montage_ready_display_commit(session)
        self._settle_montage_visible_plan_if_complete(session)
        self._finish_montage_session_if_complete(session)
        if not upload_backlog:
            schedule_near_viewport_montage_prefetch(self, session)
        self._retry_live_profile_after_montage_tile()

    def _commit_vispy_montage_tile_delta_direct(
        self,
        session,
        display_image,
        geometry,
        *,
        tile_state,
        base_tile_state,
        tile_delta,
        semantic_source,
        applied_level_source,
        histogram_plot_data,
        explicit_auto: bool,
        semantic_commit: bool,
    ) -> bool:
        if not _persistent_tile_residency_backend(self, session):
            return False
        previous_frame = getattr(self.win, "_committed_display_frame", None)
        if previous_frame is None or not isinstance(getattr(previous_frame, "value_source", None), TiledValueSource):
            return False
        previous_geometry = getattr(previous_frame, "geometry", None)
        if not _safe_tiled_payload_geometry_retarget(previous_geometry, geometry):
            return False
        context = self._render_request_context(
            document_key=_document_key(session.document),
            request_key=session.key,
            render_generation=session.render_generation,
            semantic_key=session.level_key,
        )
        decision = decide_presentation(
            PresentationInput(
                payload=DisplayPayload(
                    image=display_image,
                    geometry=geometry,
                    viewport_policy=ViewportPolicy.PRESERVE,
                    frame_plan=session.frame_plan,
                    rgb_already_windowed=bool(getattr(display_image, "rgb_already_windowed", False)),
                    histogram_plot_data=histogram_plot_data,
                    tile_state=tile_state,
                    base_tile_state=base_tile_state,
                    tile_delta=tile_delta,
                    tile_residency_budget_bytes=tile_residency_budget_bytes(self._memory_policy()),
                ),
                context=context,
                previous_frame=previous_frame,
                window_mode=session.window_mode,
                force_auto=False,
                commit_kind=CommitKind.EXPLICIT_AUTO_WINDOW if explicit_auto else CommitKind.PROGRESSIVE_FRAME_PATCH,
                semantic_source=semantic_source,
                applied_level_source=applied_level_source,
                user_levels=_session_requested_levels(session),
            )
        )
        set_image_start = perf_counter()
        backend_decision = self._montage_backend_decision_for_display(geometry, display_image.data)
        if backend_decision.backend != "tile_layer":
            return False
        committer = self._display_committer()
        committer.commit_tiled_delta(decision.display_presentation)
        self._record_montage_backend_commit(backend_decision, "tile_layer")
        self._last_set_image_ms = (perf_counter() - set_image_start) * 1000.0
        self.display_geometry = geometry

        report = getattr(committer, "last_tile_commit_report", None)
        semantic_frame_commit = bool(
            semantic_commit
            and bool(getattr(report, "presented_tiles", ()))
        )
        if semantic_frame_commit:
            committed_state = getattr(committer, "last_tile_committed_state", None)
            payloads = getattr(committed_state, "payloads", None)
            if payloads is not None:
                frame = replace(
                    previous_frame,
                    key=context.frame_key,
                    geometry=geometry,
                    levels=decision.levels,
                    histogram_range=decision.histogram_range,
                    value_source=TiledValueSource(payloads),
                    scene=None,
                )
                self._set_committed_display_frame(frame)
                self._consume_pending_display_levels(session.user_levels_override)
                self._note_display_level_source(decision)
                apply_restored_viewport = getattr(self.win, "_apply_viewport_continuity_when_ready", None)
                if callable(apply_restored_viewport):
                    apply_restored_viewport()
                show_pending_montage_revert = getattr(self, "_show_pending_montage_view_revert", None)
                if callable(show_pending_montage_revert):
                    show_pending_montage_revert()
                refresh_hover = getattr(self, "_refresh_hover_after_display_commit", None)
                if callable(refresh_hover):
                    refresh_hover()
        self.win.apply_axis_flips()
        self.win.img_view.setImageStale(False)
        return True

    def _schedule_montage_ready_display_commit(self, session) -> None:
        if not self._montage_session_is_current(session):
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
        self._montage_commit_token = _montage_work_token(session, "commit")
        if bool(getattr(session, "display_committed", False)):
            if _persistent_tile_layer_fast_drain_enabled(self, session):
                session.final_commit_pending = True
                session.flush_pending = True
                if self._queue_montage_presentation_commit_flush():
                    return
            self._schedule_montage_presentation_commit(session, force=False)
            return
        session.final_commit_pending = True
        session.flush_pending = True
        if _persistent_tile_layer_fast_drain_enabled(self, session) and self._queue_montage_presentation_commit_flush():
            return
        try:
            # Qt event-turn barrier guarded by `_montage_commit_token`; the
            # fallback timer exists only for bindings that reject the
            # receiver-context singleShot overload.
            Qt.QtCore.QTimer.singleShot(0, self, self._flush_montage_presentation_commit)
        except Exception:
            self._start_montage_commit_timer(1)

    def _queue_montage_presentation_commit_flush(self) -> bool:
        if bool(getattr(self, "_montage_presentation_commit_flush_queued", False)):
            return True
        self._montage_presentation_commit_flush_queued = True
        try:
            queued = Qt.QtCore.QMetaObject.invokeMethod(
                self,
                "_flush_montage_presentation_commit",
                Qt.QtCore.Qt.ConnectionType.QueuedConnection,
            )
        except Exception:
            queued = False
        if queued:
            return True
        self._montage_presentation_commit_flush_queued = False
        return False

    def _finish_montage_session_if_complete(self, session) -> bool:
        if not self._montage_session_is_current(session):
            return False
        if not session.is_complete():
            return False
        self._settle_montage_visible_plan_if_complete(session)
        self._schedule_montage_refined_level_stats(session)
        return True

    def _settle_montage_visible_plan_if_complete(self, session) -> bool:
        if not self._montage_session_is_current(session):
            return False
        if not session.visible_plan_complete():
            return False
        session.show_loading_overlays = False
        self._stop_montage_session_slow_overlay()
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

    def _maybe_auto_fit_montage_tiles(self, plan_or_geometry) -> bool:
        if bool(getattr(self, "_montage_live_layout_reflow", False)):
            return False
        pending_continuity = getattr(self.win, "_pending_viewport_continuity_range", None)
        active_continuity = getattr(self.win, "_active_viewport_continuity_range", None)
        if callable(pending_continuity) and pending_continuity() is not None:
            return False
        if callable(active_continuity) and active_continuity() is not None:
            return False
        plan = plan_or_geometry if hasattr(plan_or_geometry, "geometry") else None
        geometry = getattr(plan_or_geometry, "geometry", plan_or_geometry)
        montage = getattr(geometry, "montage", geometry)
        if montage is None or not getattr(montage, "indices", ()):
            self._last_montage_autofit_signature = None
            return False
        tile_count = len(tuple(montage.indices))
        fallback_range = _montage_full_view_range(montage)
        if plan is not None:
            viewport_size = self.win.img_view.graphicsView.viewport().size()
            auto_range = square_montage_fit_view_range(
                plan,
                (max(1, viewport_size.height()), max(1, viewport_size.width())),
            )
        else:
            auto_range = fallback_range
        signature = _montage_autofit_signature(montage)
        revert_previous_signature = getattr(self, "_last_montage_revert_signature", None)
        self._last_montage_revert_signature = signature
        scope_grew_for_revert = bool(
            revert_previous_signature is not None
            and _montage_autofit_scope_grew(revert_previous_signature, signature)
        )
        previous_signature = getattr(self, "_last_montage_autofit_signature", None)
        self._last_montage_autofit_signature = signature
        if previous_signature is not None and not _montage_autofit_scope_grew(previous_signature, signature):
            return False
        viewport_controller = getattr(self.win.img_view, "viewport_controller", None)
        if viewport_controller is not None and viewport_controller.is_fit_locked():
            return False
        view = self.win.img_view.getView()
        before_range = _copy_view_range(view.viewRange())
        auto_like = _viewport_controller_auto_active_for_range(viewport_controller, before_range)
        visible_count = _visible_montage_tile_count(montage, before_range)
        can_auto_adjust = _should_auto_fit_montage_view(
            before_range,
            auto_range,
            viewport_controller=viewport_controller,
            visible_count=visible_count,
            tile_count=tile_count,
        )
        if (
            can_auto_adjust
            and scope_grew_for_revert
            and not auto_like
            and tile_count > 0
            and visible_count / float(tile_count) > MONTAGE_AUTOFIT_VISIBLE_FRACTION
            and not _view_range_contains_near(before_range, fallback_range)
            and viewport_controller is not None
        ):
            if not bool(getattr(self.win, "_suppress_montage_autofit_revert_message", False)):
                previous_mode = viewport_controller.mode
                self._pending_montage_view_revert = (
                    before_range,
                    previous_mode,
                    "Adjusted montage view.",
                )
            return False
        if not can_auto_adjust:
            return False
        if not auto_like and (tile_count <= 0 or visible_count / float(tile_count) > MONTAGE_AUTOFIT_VISIBLE_FRACTION):
            return False
        if _view_range_contains_near(before_range, fallback_range):
            return False
        previous_mode = None if viewport_controller is None else viewport_controller.mode
        self._set_montage_view_range(auto_range)
        if viewport_controller is not None:
            viewport_controller.mode = ViewportMode.AUTO_UNTOUCHED
            viewport_controller.last_auto_view_range = auto_range

        def undo():
            self._set_montage_view_range(before_range)
            if viewport_controller is not None and previous_mode is not None:
                viewport_controller.mode = previous_mode

        if not bool(getattr(self.win, "_suppress_montage_autofit_revert_message", False)):
            show_revert_action(
                self.win,
                "Fitted montage to show all tiles.",
                undo,
                timeout=5000,
            )
        return True

    def _show_pending_montage_view_revert(self) -> None:
        pending = getattr(self, "_pending_montage_view_revert", None)
        if pending is None:
            return
        self._pending_montage_view_revert = None
        before_range, previous_mode, message = pending
        try:
            current_range = _copy_view_range(self.win.img_view.getView().viewRange())
        except Exception:
            return
        if view_ranges_near(current_range, before_range, tolerance_fraction=0.005):
            return
        viewport_controller = getattr(self.win.img_view, "viewport_controller", None)

        def undo():
            self._set_montage_view_range(before_range)
            if viewport_controller is not None and previous_mode is not None:
                viewport_controller.mode = previous_mode

        show_revert_action(
            self.win,
            message,
            undo,
            timeout=5000,
        )

    def _set_montage_view_range(self, view_range) -> None:
        view = self.win.img_view.getView()
        was_applying = bool(getattr(self.win.img_view, "_viewport_applying", False))
        self.win.img_view._viewport_applying = True
        try:
            view.setRange(
                xRange=(float(view_range[0][0]), float(view_range[0][1])),
                yRange=(float(view_range[1][0]), float(view_range[1][1])),
                padding=0,
            )
        finally:
            self.win.img_view._viewport_applying = was_applying

    def _montage_tile_source_ids(self, session) -> dict[int, object]:
        source_ids = getattr(session, "tile_source_ids", None)
        if source_ids is None:
            source_ids = {}
            session.tile_source_ids = source_ids
        plan_tiles = {
            int(tile.montage_index): tile
            for tile in tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ())
        }
        for stale in tuple(source_ids):
            if int(stale) not in plan_tiles:
                source_ids.pop(int(stale), None)
        tile_key_for = self.win.operation_evaluator.montage_tile_key_batch(
            colormap_lut=session.colormap_lut,
            document=session.document,
            shader_display=bool(getattr(session, "shader_display", False)),
        )
        for tile_number, tile in sorted(plan_tiles.items()):
            if int(tile_number) in source_ids:
                continue
            try:
                source_ids[tile_number] = tile_key_for(tile.view_state)
            except Exception:
                rendered = getattr(session, "rendered_tiles", {}).get(int(tile_number))
                if rendered is None:
                    source_ids[tile_number] = (
                        "planned_tile",
                        int(getattr(tile, "montage_index", tile_number)),
                        int(getattr(tile, "source_index", tile_number)),
                        id(getattr(tile, "view_state", None)),
                    )
                else:
                    image = getattr(rendered, "image", None)
                    histogram = getattr(rendered, "histogram_data", None)
                    source_ids[tile_number] = (
                        id(image),
                        tuple(np.shape(image)),
                        None if image is None else str(np.asarray(image).dtype),
                        id(histogram),
                        None if histogram is None else tuple(np.shape(histogram)),
                        None if histogram is None else str(np.asarray(histogram).dtype),
                    )
        return dict(source_ids)

    def _should_publish_montage_level_metadata(self, session, stats: MontageLevelStats) -> bool:
        # Histogram metadata is independent from whether display levels are
        # allowed to move.  Publishing better semantic stats lets absolute mode
        # update the histogram while preserving numeric levels, and lets
        # relative mode remap through WindowLevelController.
        if not session.rendered_tiles:
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
        applied_count = int(getattr(applied, "source_count", 0) or 0)
        if int(stats.rank) > applied_rank:
            return True
        if len(stats.source_indices) > applied_count:
            return True
        applied_bounds = normalize_bounds(getattr(applied, "histogram_range", None))
        if applied_bounds is None:
            return True
        return bounds[0] < applied_bounds[0] or bounds[1] > applied_bounds[1]

    def _note_montage_level_source_applied(self, session, source, *, explicit: bool) -> None:
        if source is None:
            return
        # Store partial as well as complete semantic sources.  The presentation
        # controller expands same-key sources monotonically and protects explicit
        # user locks, so storing partial coverage is safe and prevents fallback
        # to stale tiny placeholder ranges.
        session.applied_level_source = source

    def _classify_visible_montage_tiles(self, session) -> None:
        rect = montage_rect_for_viewport(session.plan, view_range=session.view_range, viewport_shape=session.viewport_shape)
        pending = set(session.pending_tile_numbers())
        newly_pending = []
        for tile in session.plan.tiles:
            index = int(tile.montage_index)
            intersects = tile.x0 < rect[2] and tile.x0 + tile.width > rect[0] and tile.y0 < rect[3] and tile.y0 + tile.height > rect[1]
            if not intersects:
                continue
            if index in session.rendered_tiles or index in session.loading_tiles or index in session.skipped_tiles:
                continue
            if index in pending:
                session.mark_loading(tile)
            else:
                newly_pending.append(tile)
                pending.add(index)
        for tile in prioritize_montage_tiles(
            newly_pending,
            view_range=((rect[0], rect[2]), (rect[1], rect[3])),
            focus=_montage_priority_focus(self, session.view_range),
        ):
            _enqueue_session_pending_tile(session, tile)
            session.mark_loading(tile)

    def _update_montage_tile_overlays_for_plan(self, plan, tile_states, viewport_rect) -> None:
        if not hasattr(self.win.img_view, "setMontageTileOverlays"):
            return
        session = getattr(self, "_montage_session", None)
        show_loading = bool(getattr(session, "show_loading_overlays", False))
        candidate_numbers: tuple[int, ...] | None = None
        tile_state_revision = None
        if session is not None and getattr(session, "plan", None) is plan:
            tile_state_revision = int(getattr(session, "tile_state_revision", 0) or 0)
            candidates = set(int(tile) for tile in getattr(session, "skipped_tiles", ()) or ())
            if show_loading:
                candidates.update(int(tile) for tile in getattr(session, "loading_tiles", ()) or ())
                candidates.update(
                    int(tile)
                    for tile in set(getattr(session, "rendered_tiles", {}) or {})
                    - set(getattr(session, "presented_tiles", ()) or ())
                )
            candidate_numbers = tuple(sorted(candidates))
            key = (
                id(plan),
                tuple(int(value) for value in viewport_rect),
                tile_state_revision,
                show_loading,
                candidate_numbers,
            )
            if key == getattr(self, "_last_montage_overlay_update_key", None):
                return
            self._last_montage_overlay_update_key = key
            if not candidate_numbers:
                if int(getattr(self.win.img_view, "montageTileOverlayCount", lambda: 0)() or 0) != 0:
                    self.win.img_view.setMontageTileOverlays(())
                return
        overlays = []
        viewport_x0, viewport_y0, viewport_x1, viewport_y1 = (int(value) for value in viewport_rect)
        if candidate_numbers is None:
            tiles = tuple(plan.tiles)
        else:
            plan_tiles = tuple(plan.tiles)
            tiles = tuple(
                plan_tiles[int(index)]
                for index in candidate_numbers
                if 0 <= int(index) < len(plan_tiles)
            )
        for tile in tiles:
            state = tile_states[int(tile.montage_index)] if int(tile.montage_index) < len(tile_states) else MontageTileState.UNLOADED
            if state == MontageTileState.LOADING and not show_loading:
                continue
            if state not in {MontageTileState.LOADING, MontageTileState.SKIPPED}:
                continue
            tile_x0 = int(tile.x0)
            tile_y0 = int(tile.y0)
            tile_x1 = tile_x0 + int(tile.width)
            tile_y1 = tile_y0 + int(tile.height)
            x = max(tile_x0, viewport_x0)
            y = max(tile_y0, viewport_y0)
            x1 = min(tile_x1, viewport_x1)
            y1 = min(tile_y1, viewport_y1)
            if x1 <= x or y1 <= y:
                continue
            overlays.append(
                MontageTileOverlay(
                    x=x,
                    y=y,
                    width=max(1, x1 - x),
                    height=max(1, y1 - y),
                    state=state.value,
                    text="Skipped" if state == MontageTileState.SKIPPED else "Loading",
                )
            )
        self.win.img_view.setMontageTileOverlays(tuple(overlays))

    def _is_current_montage_session(self, session_id, key) -> bool:
        return _session_token_is_current(getattr(self, "_montage_session", None), session_id, key)

    def _montage_session_is_current(self, session) -> bool:
        """Shared staleness predicate for a montage session object.

        Delegates to :meth:`_is_current_montage_session` so tests that stub the
        (session_id, key) predicate keep controlling both forms. Callbacks that
        captured a raw ``(session_id, key)`` pair call the canonical predicate
        directly.
        """
        if session is None:
            return False
        return self._is_current_montage_session(session.session_id, session.key)

    def _retry_live_profile_after_montage_tile(self) -> None:
        try:
            if not self.win.widgets['buttons']['display']['live_profile'].isChecked():
                return
            position = self.win.img_view.profileMarkerPosition()
            if position is None:
                return
            self._on_profile_marker_moved(*position)
            self._update_live_profile_from_pending_pos()
        except Exception as exc:
            handle_ui_exception("montage live profile retry", exc)

    def _montage_callback_budget(
        self,
        channel: str,
        *,
        interactive: bool,
        work_class: str,
        item_cap: int | None = None,
        byte_cap: int | None = None,
        target_ms: float | None = None,
    ) -> GuiCallbackBudget:
        decision = getattr(self.win, "_ui_work_decision", lambda *args, **kwargs: None)(
            channel,
            interactive=interactive,
        )
        budget = GuiCallbackBudget.for_decision(
            channel,
            decision,
            interactive=interactive,
            work_class=work_class,
            backend=image_view_backend_capabilities(self.win.img_view).name,
            item_cap=item_cap,
            byte_cap=byte_cap,
        )
        if target_ms is not None:
            budget.target_ms = max(0.0, float(target_ms))
        return budget

    def _record_gui_budget(self, budget: GuiCallbackBudget) -> None:
        observation = budget.observation()
        if observation.processed_items <= 0 and observation.elapsed_ms < observation.warning_ms:
            return
        recorder = getattr(getattr(self.win, "resource_governor", None), "record_gui_callback_observation", None)
        if callable(recorder):
            recorder(observation)
            return
        if hasattr(self.win, "_record_ui_work"):
            self.win._record_ui_work(
                observation.channel,
                observation.elapsed_ms,
                count=max(1, observation.processed_items),
                byte_count=observation.processed_bytes,
                work_class=observation.work_class,
                backend=observation.backend,
            )

    def _schedule_loading_montage_profile_retry(self, x, y) -> None:
        self._pending_montage_profile_retry = (float(x), float(y))
        if self.win.profile_dock.isVisible():
            self._retry_loading_montage_profile()

    def _retry_loading_montage_profile(self) -> None:
        point = getattr(self, "_pending_montage_profile_retry", None)
        self._pending_montage_profile_retry = None
        if point is None or self.win.view_state.montage_axis is None:
            return
        if not self.win.widgets['buttons']['display']['live_profile'].isChecked():
            return
        if not self.win.profile_dock.isVisible():
            self._pending_montage_profile_retry = (float(point[0]), float(point[1]))
            return
        self.win._pending_profile_point = (float(point[0]), float(point[1]))
        self.win._pending_profile_pos = None
        self._update_live_profile_from_pending_pos()

    def _schedule_montage_viewport_update(self, *, delay_ms: int | None = None) -> None:
        if getattr(self, "_montage_viewport_update_running", False):
            self.win._montage_viewport_update_pending = True
            return
        session = getattr(self, "_montage_session", None)
        self._montage_viewport_update_token = None if session is None else _montage_work_token(session, "viewport_update")
        timer = getattr(self, "_montage_viewport_update_timer", None)
        if timer is None:
            # Bounded continuation. The callback only retargets the current
            # montage session/viewport token; it does not establish semantic
            # order between sessions or payload revisions.
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._run_montage_viewport_update)
            self._montage_viewport_update_timer = timer
        interval = _montage_viewport_update_delay_ms(self) if delay_ms is None else max(0, int(delay_ms))
        timer.start(interval)

    def _schedule_montage_priority_retarget_from_hover(self) -> None:
        if getattr(self.win, "_closing", False):
            return
        if getattr(self.win.view_state, "montage_axis", None) is None:
            return
        session = getattr(self, "_montage_session", None)
        if not self._montage_session_is_current(session):
            return
        if not (session.pending_tiles or session.stage_fan_in.waiting_tiles):
            return
        self._montage_priority_retarget_token = _montage_work_token(session, "priority_retarget")
        timer = getattr(self, "_montage_priority_retarget_timer", None)
        if timer is None:
            # Bounded continuation guarded by `_montage_priority_retarget_token`.
            # Hover retargeting updates queue metadata in batches; remove when
            # tile priority updates are admitted as scheduler work items.
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._run_montage_priority_retarget)
            self._montage_priority_retarget_timer = timer
        self._montage_priority_retarget_pending = True
        if not timer.isActive():
            timer.start(_montage_priority_retarget_delay_ms(self))

    def _refresh_montage_priority_targets(self, session) -> int:
        """Rebuild tile-queue priorities from the live viewport.

        The session's priority context is captured before the first montage
        commit rescales the viewport, so the distance-from-focus ordering can
        point at the stale pre-montage view range — one corner of the montage
        — for the entire fill. Retarget every queued tile against the live
        range at the moments that decide fill order: the first display commit
        and a shared-stage activation. The range is passed as a priority-only
        override; ``session.view_range`` is viewport bookkeeping shared with
        level/commit scoping and stays untouched.
        """
        if not (session.pending_tiles or session.stage_fan_in.waiting_tiles):
            return 0
        # While a viewport-continuity restore (reload, layout change) is in
        # flight, the live camera range is transitional; prioritizing against
        # it anchors the fill order at an arbitrary point. Use the restore's
        # target range instead.
        continuity_range = getattr(self.win, "_active_viewport_continuity_range", lambda: None)()
        if continuity_range is not None:
            view_range = continuity_range
            focus = _montage_priority_focus(self, view_range)
        else:
            try:
                viewport_plan = self._montage_viewport_plan(self.win.view_state)
            except Exception:
                return 0
            if viewport_plan.view_range is None:
                return 0
            view_range = viewport_plan.view_range
            focus = viewport_plan.priority_focus
        total = len(session.pending_tiles) + sum(
            len(waiting) for waiting in session.stage_fan_in.waiting_tiles.values()
        )
        return session.retarget_tile_priority(
            focus=focus,
            max_items=max(1, int(total)),
            view_range=view_range,
        )

    def _run_montage_priority_retarget(self) -> None:
        self._montage_priority_retarget_pending = False
        if getattr(self.win, "_closing", False):
            return
        session = getattr(self, "_montage_session", None)
        if not self._montage_session_is_current(session):
            return
        token = getattr(self, "_montage_priority_retarget_token", None)
        if not _montage_work_token_is_current(session, token, "priority_retarget"):
            return
        if not (session.pending_tiles or session.stage_fan_in.waiting_tiles):
            return
        budget = self._montage_callback_budget(
            "montage_priority_retarget",
            interactive=True,
            work_class="queue_metadata",
            item_cap=_montage_priority_retarget_batch_limit(self, interactive=True),
        )
        viewport_plan = self._montage_viewport_plan(self.win.view_state)
        session.priority_focus = viewport_plan.priority_focus
        processed = session.retarget_tile_priority(
            focus=viewport_plan.priority_focus,
            max_items=budget.item_cap,
        )
        if processed:
            budget.record_item(item_count=processed)
            _complete_inline_work(
                self,
                WorkItem(
                    key=(
                        "montage_priority_retarget",
                        session.key,
                        int(session.session_id),
                        int(getattr(session, "viewport_revision", 0) or 0),
                    ),
                    lane=WorkLane.VISIBLE_PLANNING,
                    quality="retained",
                    supersession_key=("montage-priority-retarget", session.key),
                    supersession_value=int(session.session_id),
                ),
            )
        self._last_montage_priority_retarget_count = int(processed)
        self._last_montage_priority_retarget_pending = len(session.pending_tiles)
        self._record_gui_budget(budget)
        if session.pending_tiles:
            self._schedule_montage_tiles(session)

    def _run_montage_viewport_update(self) -> None:
        if getattr(self.win, "_closing", False):
            return
        session = getattr(self, "_montage_session", None)
        token = getattr(self, "_montage_viewport_update_token", None)
        if token is not None and (session is None or not _montage_work_token_is_current(session, token, "viewport_update")):
            return
        if getattr(self, "_montage_viewport_update_running", False):
            self.win._montage_viewport_update_pending = True
            return
        self._montage_viewport_update_running = True
        self.win._montage_viewport_update_pending = False
        try:
            if self._try_update_montage_viewport_only():
                retargeted = getattr(self, "_montage_session", None)
                if retargeted is not None:
                    _complete_inline_work(
                        self,
                        WorkItem(
                            key=(
                                "montage_viewport_retarget",
                                retargeted.key,
                                int(retargeted.session_id),
                                int(getattr(retargeted, "viewport_revision", 0) or 0),
                            ),
                            lane=WorkLane.DISPLAY_PREPARATION,
                            quality="retained",
                            supersession_key=("montage-viewport-retarget", retargeted.key),
                            supersession_value=int(retargeted.session_id),
                        ),
                    )
            else:
                self.update_image_view()
        finally:
            self._montage_viewport_update_running = False
        if getattr(self.win, "_montage_viewport_update_pending", False):
            self.win._montage_viewport_update_pending = False
            delay = 0 if getattr(self, "_montage_viewport_continue_immediately", False) else _montage_viewport_chunk_delay_ms(self)
            self._montage_viewport_continue_immediately = False
            self._schedule_montage_viewport_update(delay_ms=delay)

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


def _visible_montage_tile_count(montage, view_range) -> int:
    x0, x1 = sorted((float(view_range[0][0]), float(view_range[0][1])))
    y0, y1 = sorted((float(view_range[1][0]), float(view_range[1][1])))
    tile_width = int(montage.tile_width)
    tile_height = int(montage.tile_height)
    columns = max(1, int(montage.columns))
    gap = max(0, int(montage.gap))
    visible = 0
    for tile_number, _source_index in enumerate(tuple(montage.indices)):
        row = tile_number // columns
        col = tile_number % columns
        tx0 = col * (tile_width + gap)
        ty0 = row * (tile_height + gap)
        tx1 = tx0 + tile_width
        ty1 = ty0 + tile_height
        if tx1 > x0 and tx0 < x1 and ty1 > y0 and ty0 < y1:
            visible += 1
    return visible


def _view_range_contains(view_range, target_range) -> bool:
    x0, x1 = sorted((float(view_range[0][0]), float(view_range[0][1])))
    y0, y1 = sorted((float(view_range[1][0]), float(view_range[1][1])))
    tx0, tx1 = sorted((float(target_range[0][0]), float(target_range[0][1])))
    ty0, ty1 = sorted((float(target_range[1][0]), float(target_range[1][1])))
    return x0 <= tx0 and x1 >= tx1 and y0 <= ty0 and y1 >= ty1


def _view_range_contains_near(view_range, target_range, *, tolerance_fraction: float = 0.02) -> bool:
    x0, x1 = sorted((float(view_range[0][0]), float(view_range[0][1])))
    y0, y1 = sorted((float(view_range[1][0]), float(view_range[1][1])))
    tx0, tx1 = sorted((float(target_range[0][0]), float(target_range[0][1])))
    ty0, ty1 = sorted((float(target_range[1][0]), float(target_range[1][1])))
    tolerance_fraction = max(0.0, float(tolerance_fraction))
    x_tolerance = max(abs(tx1 - tx0), 1.0) * tolerance_fraction
    y_tolerance = max(abs(ty1 - ty0), 1.0) * tolerance_fraction
    return x0 <= tx0 + x_tolerance and x1 >= tx1 - x_tolerance and y0 <= ty0 + y_tolerance and y1 >= ty1 - y_tolerance


def _should_auto_fit_montage_view(
    view_range,
    full_range,
    *,
    viewport_controller,
    visible_count: int,
    tile_count: int,
) -> bool:
    if int(tile_count) <= 0:
        return False
    if _viewport_controller_auto_active_for_range(viewport_controller, view_range):
        return True
    if int(visible_count) <= 0:
        return bool(viewport_controller is None)
    if view_ranges_near(view_range, full_range):
        return True
    return False


def _viewport_controller_auto_active_for_range(viewport_controller, view_range) -> bool:
    if viewport_controller is None:
        return False
    active = getattr(viewport_controller, "is_auto_active", None)
    if callable(active) and bool(active()):
        return True
    return False


def _montage_autofit_signature(montage) -> tuple[tuple[int, ...], int, int, int]:
    return (
        tuple(int(index) for index in tuple(getattr(montage, "indices", ()) or ())),
        int(getattr(montage, "tile_width", 0) or 0),
        int(getattr(montage, "tile_height", 0) or 0),
        int(getattr(montage, "gap", 0) or 0),
    )


def _montage_autofit_scope_grew(previous, current) -> bool:
    try:
        previous_indices, previous_width, previous_height, previous_gap = previous
        current_indices, current_width, current_height, current_gap = current
    except Exception:
        return True
    previous_set = set(int(index) for index in tuple(previous_indices))
    current_set = set(int(index) for index in tuple(current_indices))
    return (
        len(current_set) > len(previous_set)
        or current_set > previous_set
        or int(current_width) > int(previous_width)
        or int(current_height) > int(previous_height)
        or int(current_gap) > int(previous_gap)
    )


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


def _montage_tile_result_batch_limit(window, *, interactive: bool) -> int:
    configured = getattr(window.win, "_montage_tile_result_batch_size", None)
    if configured is not None:
        return max(1, int(configured))
    decision = getattr(window.win, "_ui_work_decision", lambda *args, **kwargs: None)("montage_tile_result", interactive=interactive)
    if decision is not None:
        return max(1, int(decision.batch_limit))
    feedback = _latency_feedback(window)
    if feedback is None:
        return 4 if interactive else 8
    return int(feedback.batch_limit("montage_tile_result", interactive=interactive))


def _montage_viewport_addition_batch_limit(window, *, interactive: bool) -> int:
    configured = getattr(window.win, "_montage_viewport_addition_batch_size", None)
    if configured is not None:
        return max(1, int(configured))
    decision = getattr(window.win, "_ui_work_decision", lambda *args, **kwargs: None)("montage_viewport_update", interactive=interactive)
    if decision is not None:
        return max(1, min(32, int(decision.batch_limit)))
    return 8 if interactive else 16


def _montage_viewport_chunk_delay_ms(window) -> int:
    decision = getattr(window.win, "_ui_work_decision", lambda *args, **kwargs: None)(
        "montage_viewport_update",
        interactive=_interactive_active(window),
    )
    if decision is not None:
        return max(1, min(16, int(decision.interval_ms)))
    return 1


def _montage_priority_retarget_delay_ms(window) -> int:
    decision = getattr(window.win, "_ui_work_decision", lambda *args, **kwargs: None)(
        "montage_priority_retarget",
        interactive=True,
    )
    if decision is not None:
        return max(1, min(64, int(decision.interval_ms)))
    return 32


def _montage_priority_retarget_batch_limit(window, *, interactive: bool) -> int:
    decision = getattr(window.win, "_ui_work_decision", lambda *args, **kwargs: None)(
        "montage_priority_retarget",
        interactive=interactive,
    )
    if decision is not None:
        return max(1, min(128, int(decision.batch_limit)))
    return 64 if interactive else 128


def _rendered_tile_from_previous_payload(tile, payload) -> RenderedTile:
    image = np.asarray(payload.image)
    histogram = None if payload.histogram_data is None else np.asarray(payload.histogram_data)
    semantic = None if payload.semantic_data is None else np.asarray(payload.semantic_data)
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
        lod=getattr(payload, "lod", None),
        level_data=getattr(payload, "level_data", None),
        level_stats=getattr(payload, "level_stats", None),
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
        lod=getattr(cached, "lod", None),
        level_data=getattr(cached, "level_data", None),
        level_stats=getattr(cached, "level_stats", None),
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
        lod=getattr(value, "lod", None),
        level_data=getattr(value, "level_data", None),
        level_stats=getattr(value, "level_stats", None),
    )


def _rendered_tile_nbytes(rendered) -> int:
    total = 0
    for name in ("image", "histogram_data", "semantic_data", "level_data"):
        value = getattr(rendered, name, None)
        if value is not None:
            total += int(getattr(np.asarray(value), "nbytes", 0) or 0)
    return int(total)


def _montage_refined_level_values(rendered) -> np.ndarray:
    histogram = getattr(rendered, "histogram_data", None)
    if histogram is not None:
        return np.asarray(histogram)
    mapping = getattr(rendered, "shader_mapping", None)
    semantic = getattr(rendered, "semantic_data", None)
    if mapping is not None and semantic is not None:
        values = extract_component(np.asarray(semantic), getattr(mapping, "component", "real"))
        return apply_shader_scale(
            values,
            getattr(mapping, "scale", "linear"),
            symlog_constant=float(getattr(mapping, "symlog_constant", 0.0) or 0.0),
        )
    image = getattr(rendered, "image", None)
    if image is None:
        return np.asarray((), dtype=np.float32)
    image = np.asarray(image)
    if np.iscomplexobj(image):
        return np.abs(image).astype(np.float32, copy=False)
    return image


def _attach_montage_tile_level_stats(display_image, tile, *, refined: bool = False):
    if getattr(display_image, "level_stats", None) is not None:
        return display_image
    level_data = getattr(display_image, "level_data", None)
    if level_data is not None:
        stats = (
            sample_tile_level_stats(level_data, int(tile.source_index), refined=True)
            if refined
            else provisional_tile_level_stats(level_data, int(tile.source_index))
        )
        if stats is not None:
            return replace(display_image, level_stats=stats)
    values = _montage_refined_level_values(
        SimpleNamespace(
            image=getattr(display_image, "data", None),
            histogram_data=getattr(display_image, "histogram_data", None),
            shader_mapping=getattr(display_image, "shader_mapping", None),
            semantic_data=getattr(display_image, "semantic_data", None),
        )
    )
    stats = (
        sample_tile_level_stats(values, int(tile.source_index), refined=True)
        if refined
        else provisional_tile_level_stats(values, int(tile.source_index))
    )
    if stats is not None:
        return replace(display_image, level_stats=stats)
    return display_image


def _session_requested_levels(session) -> tuple[float, float] | None:
    """Return the exact level target for the current presentation session.

    ``user_levels_override`` describes persistence/window-mode semantics.
    ``level_generation.target_levels`` is the latest presentation command and remains
    authoritative while progressive PyQtGraph CPU redraws converge.  Keeping
    these concepts separate prevents an in-flight automatic commit from
    restoring older levels after a user or auto-window command.
    """

    return (
        normalize_bounds(getattr(session.level_generation, "target_levels", None))
        or normalize_bounds(getattr(session, "user_levels_override", None))
    )


def _should_publish_montage_histogram_plot(
    first_display_commit: bool,
    explicit_auto: bool,
    stats: MontageLevelStats,
    *,
    requires_semantic_plot: bool = False,
) -> bool:
    if bool(first_display_commit) or bool(explicit_auto) or bool(requires_semantic_plot):
        return True
    return stats.rank in {LevelSourceRank.MONTAGE_COMPLETE, LevelSourceRank.MONTAGE_SAMPLED_FULL}


def _tiled_payloads_require_semantic_histogram_plot(payloads) -> bool:
    return any(getattr(payload, "histogram_data", None) is None for payload in dict(payloads or {}).values())


def _montage_commit_budget_ms(window) -> float:
    interactive = _interactive_active(window)
    decision = getattr(window.win, "_ui_work_decision", lambda *args, **kwargs: None)(
        "montage_commit",
        interactive=interactive,
    )
    if decision is not None:
        return max(1.0, float(decision.budget_ms))
    feedback = _latency_feedback(window)
    if feedback is None:
        return 4.0 if interactive else 8.0
    return max(1.0, float(feedback.work_budget_ms("montage_commit", interactive=interactive)))


def _persistent_tile_layer_fast_drain_enabled(window, session) -> bool:
    if _viewport_interaction_active(window):
        return False
    if not bool(getattr(session, "display_committed", False)):
        return False
    return _persistent_gpu_tile_residency_backend(window, session)


def _persistent_tile_residency_backend(window, session) -> bool:
    capabilities = image_view_backend_capabilities(getattr(window.win, "img_view", None))
    return bool(capabilities.persistent_tile_residency)


def _persistent_gpu_tile_residency_backend(window, session) -> bool:
    capabilities = image_view_backend_capabilities(getattr(window.win, "img_view", None))
    kind = str(getattr(capabilities, "tile_residency_kind", "none") or "none")
    return bool(
        capabilities.persistent_tile_residency
        and capabilities.shader_windowing
        and kind in {"gpu_atlas", "none"}
    )


def _persistent_tile_upsert_limits(window, session) -> dict[str, int]:
    if not _persistent_gpu_tile_residency_backend(window, session):
        return {}
    interactive = _interactive_active(window)
    decision = getattr(window.win, "_ui_work_decision", lambda *args, **kwargs: None)(
        "montage_present_total",
        interactive=interactive,
    )
    upload_decision = getattr(window.win, "_ui_work_decision", lambda *args, **kwargs: None)(
        "montage_cold_commit",
        interactive=interactive,
    )
    batch_limit = int(getattr(decision, "batch_limit", 0) or 0)
    byte_cap = max(
        int(getattr(decision, "byte_cap", 0) or 0),
        int(getattr(upload_decision, "byte_cap", 0) or 0),
    )
    if batch_limit <= 0:
        feedback = _latency_feedback(window)
        batch_limit = 4 if feedback is None else int(feedback.batch_limit("montage_present_total", interactive=interactive))
    if byte_cap <= 0:
        byte_cap = 8 * 1024 * 1024 if interactive else 32 * 1024 * 1024
    limits = {
        "max_upserts": max(1, int(batch_limit)),
        "max_upsert_bytes": max(1024, int(byte_cap)),
        "upsert_cost_fn": _vispy_payload_upload_nbytes,
    }
    return limits


def _vispy_payload_upload_nbytes(payload) -> int:
    texture = getattr(payload, "texture_data", None)
    if texture is None:
        texture = getattr(payload, "image", None)
    total = 0 if texture is None else int(getattr(np.asarray(texture), "nbytes", 0) or 0)
    histogram = getattr(payload, "histogram_data", None)
    if histogram is not None and histogram is not texture:
        total += int(getattr(np.asarray(histogram), "nbytes", 0) or 0)
    return max(1, int(total))


def _pyqtgraph_payload_upload_nbytes(payload) -> int:
    image = getattr(payload, "image", None)
    return max(1, 0 if image is None else int(getattr(np.asarray(image), "nbytes", 0) or 0))


def _tile_layer_upsert_limits(window, session) -> dict[str, int]:
    if _persistent_gpu_tile_residency_backend(window, session):
        return _persistent_tile_upsert_limits(window, session)
    capabilities = image_view_backend_capabilities(getattr(window.win, "img_view", None))
    if not (
        not capabilities.shader_windowing
        and session.has_pending_level_update()
        and session.has_stale_level_presentations()
    ):
        return {}
    interactive = _interactive_active(window)
    decision = getattr(window.win, "_ui_work_decision", lambda *args, **kwargs: None)(
        "tile_layer_commit",
        interactive=interactive,
    )
    batch_limit = int(getattr(decision, "batch_limit", 0) or 0)
    byte_cap = int(getattr(decision, "byte_cap", 0) or 0)
    if batch_limit <= 0:
        feedback = _latency_feedback(window)
        batch_limit = 8 if feedback is None else int(feedback.batch_limit("tile_layer_commit", interactive=interactive))
    if byte_cap <= 0:
        byte_cap = 8 * 1024 * 1024 if interactive else 32 * 1024 * 1024
    return {
        "max_upserts": max(1, int(batch_limit)),
        "max_upsert_bytes": max(1024, int(byte_cap)),
        "upsert_cost_fn": _pyqtgraph_payload_upload_nbytes,
    }


def _montage_commit_interval_ms(window, *, force: bool) -> int:
    decision = getattr(window.win, "_ui_work_decision", lambda *args, **kwargs: None)("montage_commit", interactive=_interactive_active(window))
    if decision is not None:
        return int(8 if force else max(1, decision.interval_ms))
    feedback = _latency_feedback(window)
    if feedback is None:
        return 8 if force else 16
    return int(feedback.commit_interval_ms("montage_commit", force=force, interactive=_interactive_active(window)))


def _safe_tiled_payload_geometry_retarget(previous_geometry, geometry) -> bool:
    if display_geometry_coordinates_equal(previous_geometry, geometry):
        return True
    if previous_geometry is None or geometry is None:
        return False
    previous_montage = getattr(previous_geometry, "montage", None)
    montage = getattr(geometry, "montage", None)
    if previous_montage is None or montage is None:
        return False
    return (
        previous_geometry.view_state == geometry.view_state
        and tuple(previous_montage.indices) == tuple(montage.indices)
        and tuple(previous_montage.tile_shape) == tuple(montage.tile_shape)
        and int(previous_montage.gap) == int(montage.gap)
        and int(previous_geometry.montage_origin_x) == int(geometry.montage_origin_x)
        and int(previous_geometry.montage_origin_y) == int(geometry.montage_origin_y)
    )


def _latency_feedback(window):
    return getattr(window.win, "latency_feedback", None)


def _view_range_center(view_range) -> tuple[float, float] | None:
    try:
        x_range, y_range = view_range
        return (
            (float(x_range[0]) + float(x_range[1])) * 0.5,
            (float(y_range[0]) + float(y_range[1])) * 0.5,
        )
    except Exception:
        return None


def _montage_priority_focus(window, view_range) -> tuple[float, float] | None:
    """Return the user-attention point for scheduling visible montage tiles."""

    focus = getattr(window, "_last_image_hover_focus", None)
    if focus is not None:
        try:
            frame = getattr(window.win, "_committed_display_frame", None)
            if getattr(window, "_last_image_hover_focus_frame_key", None) != getattr(frame, "key", None):
                raise ValueError("stored hover focus belongs to an older committed frame")
            x = float(focus[0])
            y = float(focus[1])
            x_range, y_range = view_range
            x0, x1 = sorted((float(x_range[0]), float(x_range[1])))
            y0, y1 = sorted((float(y_range[0]), float(y_range[1])))
            if x < x0 or x > x1 or y < y0 or y > y1:
                raise ValueError("stored hover focus is outside the current viewport")
            return (x, y)
        except Exception:
            pass
    try:
        plan = getattr(getattr(window, "_montage_session", None), "plan", None)
        if plan is not None:
            focus = _nearest_montage_tile_center(plan, view_range)
            if focus is not None:
                return focus
        return _view_range_center(view_range)
    except Exception:
        return None


def _nearest_montage_tile_center(plan, view_range) -> tuple[float, float] | None:
    center = _view_range_center(view_range)
    if center is None:
        return None
    tiles = getattr(plan, "tiles", ())
    if not tiles:
        return None
    try:
        tile_height, tile_width = (int(value) for value in plan.tile_shape[:2])
        gap = max(0, int(plan.gap))
        columns = max(1, int(plan.columns))
        rows = max(1, int(plan.rows))
        count = len(tiles)
        stride_x = max(1, tile_width + gap)
        stride_y = max(1, tile_height + gap)
        col = int(round((float(center[0]) - float(tile_width) * 0.5) / float(stride_x)))
        row = int(round((float(center[1]) - float(tile_height) * 0.5) / float(stride_y)))
        row = max(0, min(rows - 1, row))
        max_col = min(columns - 1, count - row * columns - 1)
        if max_col < 0:
            row = max(0, min((count - 1) // columns, rows - 1))
            max_col = min(columns - 1, count - row * columns - 1)
        col = max(0, min(max_col, col))
        tile = tiles[row * columns + col]
        return (
            float(tile.x0) + float(tile.width) * 0.5,
            float(tile.y0) + float(tile.height) * 0.5,
        )
    except Exception:
        return None


def _interactive_active(window) -> bool:
    coordinator = getattr(window.win, "render_coordinator", None)
    return bool(
        coordinator is not None and getattr(coordinator, "interactive_active", False)
        or _viewport_interaction_active(window)
    )


def _should_defer_montage_side_panels(window, session) -> bool:
    return bool(getattr(session, "defer_side_panels", False) or _viewport_interaction_active(window))


def _tile_layer_auto_levels_wait_for_complete_source(window, session, decision_force_auto: bool, level_stats) -> bool:
    """Should this explicit-auto commit park until better level evidence exists?

    ADR 0051 rule 6 (wedge fixed 2026-07-05): a True here parks the commit
    with ``flush_pending``/``final_commit_pending`` set, so the CALLER must
    guarantee an evidence producer is armed (level scan / pending level
    tiles); parking on evidence that nothing is scheduled to produce is the
    dispatch-construction violation the watchdog assertion reports.  The
    ``loading_tiles`` read is a machine view now, so a tile whose payload the
    backend confirmed cannot hold this wait open (the pyqtgraph+resident
    auto-levels wedge signature: presented==rendered, queues 0, flush+final
    true, loading==tile_count).
    """

    if not bool(decision_force_auto):
        return False
    if bool(image_view_backend_capabilities(window.win.img_view).shader_windowing):
        return False
    if level_stats is None:
        return True
    if level_stats.rank == LevelSourceRank.MONTAGE_SAMPLED_FULL:
        return False
    return bool(
        getattr(session, "pending_tiles", None)
        or getattr(session, "loading_tiles", None)
        or getattr(session, "active_tile_requests", None)
        or getattr(session, "pending_completed_tiles", None)
        or getattr(session, "pending_level_tiles", None)
        or int(getattr(session, "level_scan_remaining_tiles", 0) or 0) > 0
    )


def _montage_level_evidence_requires_refined(window, session) -> bool:
    level_generation = getattr(session, "level_generation", None)
    requested_levels = (
        normalize_bounds(getattr(level_generation, "target_levels", None))
        or normalize_bounds(getattr(session, "user_levels_override", None))
    )
    return bool(
        requested_levels is None
        and getattr(session, "force_auto", False)
        and not image_view_backend_capabilities(window.win.img_view).shader_windowing
    )


def _viewport_interaction_active(window) -> bool:
    return bool(getattr(window.win, "_viewport_interaction_active", False))


def _deferred_stage_plan_stub() -> dict:
    """Empty stage plan for interaction-burst sessions (planning deferred).

    Shape-compatible with ``_plan_montage_stages`` so session construction is
    identical; ``retained_stage_decision`` documents the deferral in
    diagnostics.
    """

    return {
        "tile_stage_keys": {},
        "tile_stage_plans": {},
        "tile_stage_candidates": {},
        "stage_waiting_tiles": {},
        "stage_values": {},
        "lead_stage_warmups": {},
        "stage_requests": [],
        "attached_stage_keys": set(),
        "waiting_indices": set(),
        "lead_direct_tiles": 0,
        "retained_stage_index": None,
        "retained_stage_decision": "deferred-interaction",
        "repeated_expensive_stage_per_tile": False,
    }
