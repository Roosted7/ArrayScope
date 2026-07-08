"""Frame render orchestration for ArrayScope windows.

The frame path owns the single visible image surface for every image view.  A
single slice and a multi-region montage both become one semantic frame with
region payloads; backends differ only in physical presentation mechanics.
"""

from __future__ import annotations


from dataclasses import replace
from time import perf_counter

import numpy as np
import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception
from arrayscope.core.cache_status import CacheStatus, CacheStatusSnapshot
from arrayscope.core.memory_budget import estimate_display_image_bytes, format_bytes
from arrayscope.core.scheduler import FrameTarget
from arrayscope.core.view_state import ChannelMode
from arrayscope.kernel import Lane as WorkLane, WorkItem, complete_inline_work as _complete_inline_work
from arrayscope.display.frame_planner import FramePlanner
from arrayscope.display.montage import (
    MontageTileState,
    RenderedTile,
    make_montage_plan,
)
from arrayscope.display.viewport import view_ranges_near
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.operations.evaluator import _document_key
from arrayscope.render import effects as render_effects
from arrayscope.render.stages import RenderIntent
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.montage_backend import choose_montage_backend
from arrayscope.display.model.montage_levels import (
    MontageLevelStats,
)
from arrayscope.display.lod import LOD_POLICY_RESIDENT, factor_xy_for_level
from arrayscope.presentation import ClaimOwner
from arrayscope.display.pyramid import preview_level_for_tile_shape
from arrayscope.window.montage_payload_cache import (
    payload_lod_matches as _payload_lod_matches,
    payload_compatible_with_tile as _payload_compatible_with_tile,
    previous_tiled_payloads_by_base_source as _previous_tiled_payloads_by_base_source,
    RetainedTiledPayloadStore,
)
from arrayscope.window import montage_commit
from arrayscope.window.montage_commit import MontagePipelineEffects
from arrayscope.render.level_stats import LevelStatsService
from arrayscope.window.montage_prefetch import schedule_near_viewport_montage_prefetch
from arrayscope.window.montage_runtime import MontageRuntimeMixin
from arrayscope.window.montage_viewport import (
    MontageViewportPlan,
    effective_montage_columns,
    montage_session_key,
    montage_tile_semantic_key,
    montage_viewport_intent,
    montage_viewport_retarget_policy,
    remap_montage_roi_selections,
    retarget_montage_viewport_plan,
)
from arrayscope.render import lod as render_lod
from arrayscope.window.montage_session import MontageRenderSession
from arrayscope.window.render_contract import (
    session_token_is_current as _session_token_is_current,
)
from arrayscope.display.planning import LevelSourceRank, fallback_level_source, normalize_bounds


MONTAGE_VERY_SLOW_UPLOAD_MS = 100.0
MONTAGE_AUTOFIT_VISIBLE_FRACTION = 0.80
class FrameRenderMixin(MontageRuntimeMixin, LevelStatsService):
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
        lod_swap_ready = session.mark_ladder_swaps_for_viewport()
        self.retarget_montage_pipeline(session)
        if presentation_changed or lod_swap_ready:
            self._commit_montage_resize_presentation_retarget(session)
        return True

    def _commit_montage_resize_presentation_retarget(self, session) -> None:
        if bool(getattr(self, "_montage_presentation_commit_active", False)):
            self.apply_montage_presentation(session)
            return
        self.commit_montage_session_presentation(session)

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
            viewport_plan = self._montage_viewport_plan(
                view_state,
                view_range=self._current_montage_global_view_range(),
            )
            viewport_shape = viewport_plan.viewport_shape
            tile_shape = viewport_plan.tile_shape
            plan = viewport_plan.plan
        self._publish_montage_content_extent(plan)
        self._montage_live_layout_reflow = False
        previous_session_plan = getattr(getattr(self, "_montage_session", None), "plan", None)
        self._remap_montage_rois_for_layout_reflow(previous_session_plan, plan)
        current_range = viewport_plan.view_range
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
            and self._montage_quality_policy_mode() == LOD_POLICY_RESIDENT
            and previous_session is not None
            # Only a predecessor that actually committed montage content can
            # carry the screen through the burst (floors/retained payloads).
            # A first-ever montage build has a user staring at nothing —
            # plan inline.  Same axis: an axis change is a new montage, not a
            # scrub step.
            and bool(getattr(previous_session, "display_committed", False))
            and getattr(previous_session, "montage_axis", None) == axis
        )
        if defer_stage_planning:
            stage_plan = montage_commit.deferred_stage_fan_in_plan()
            self._montage_stage_plans_deferred = (
                int(getattr(self, "_montage_stage_plans_deferred", 0) or 0) + 1
            )
        else:
            stage_plan = montage_commit.build_stage_fan_in_plan(self, document, missing_tiles)
        self._last_montage_stage_plan_ms = (perf_counter() - stage_plan_start) * 1000.0
        session_setup_start = perf_counter()
        pending_tiles = [] if defer_stage_planning else list(missing_tiles)
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
        lod_policy_mode = self._montage_quality_policy_mode()
        lod_preview_level = (
            preview_level_for_tile_shape(plan.tile_shape, min_level=render_lod.PREVIEW_FLOOR_MIN_LEVEL)
            if lod_policy_mode == LOD_POLICY_RESIDENT
            else 0
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
            stage_fan_in=montage_commit.stage_fan_in_state(stage_plan),
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
            lod_native_reason=render_lod.native_policy_reason_for_renderer(self),
            lod_preview_level=lod_preview_level,
            pyramid_cache=(
                self._montage_pyramid_cache() if lod_policy_mode == LOD_POLICY_RESIDENT else None
            ),
        )
        session.shader_display = bool(shader_display)
        session.stage_planning_deferred = bool(defer_stage_planning)
        session.deferred_missing_tiles = tuple(missing_tiles) if defer_stage_planning else ()
        # The dying session's planned-but-undrained LOD requests hold
        # singleflight claims in the shared pyramid; scrubbing back to the
        # same slice would find those levels permanently claimed (stale
        # wrong-LOD tiles).  Balance them before the replacement takes over.
        render_lod.release_session_claims(getattr(self, "_montage_session", None))
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
        self._last_montage_session_setup_ms = (perf_counter() - session_setup_start) * 1000.0
        initial_commit_start = perf_counter()
        try:
            self.commit_montage_session_presentation(session)
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
            self.show_montage_session_slow_overlay(session)
        self._schedule_montage_cached_level_stats(session)
        if defer_stage_planning:
            self.retarget_montage_pipeline(session)
        else:
            montage_commit.submit_stage_tasks(self, session, stage_plan["stage_requests"])
            self.retarget_montage_pipeline(session)

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
        # Level convergence no longer forces a rebirth (ADR 0051 P2, landed
        # 2026-07-05): upserts are machine-gated (emit-once + identity-aware
        # acknowledgement with bounded resignation), stale-level tiles drain
        # through prioritized budgeted commits, and the dispatch derivation +
        # watchdog signature track level evidence and stale-count progress —
        # the blind re-upsert loop the old "level-pending" reject guarded
        # against cannot form.  retarget_index_window resets the per-window
        # evidence queues and forgets demoted tiles; kept tiles keep their
        # applied values and keep converging through the standing machinery.
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
                    self.commit_montage_session_presentation(session)
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
        render_lod.release_session_claims(session)
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
        session.force_auto = bool(force_auto)
        session.user_levels_override = user_levels
        session.attach_stage_fan_in(montage_commit.stage_fan_in_state(montage_commit.deferred_stage_fan_in_plan()))
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
            self.commit_montage_session_presentation(session)
        finally:
            self._last_montage_initial_commit_ms = (
                perf_counter() - initial_commit_start
            ) * 1000.0
        if session.defer_side_panels or _viewport_interaction_active(self):
            self.win._deferred_side_panel_refresh_pending = True
        else:
            self.win._update_operation_dock()
        if session.stage_planning_deferred:
            self.retarget_montage_pipeline(session)
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
        # Camera-only changes must retarget the LOD decision immediately:
        # already-resident levels swap on the next commit and missing levels
        # are scheduled now, superseded by the new viewport revision (ADR
        # 0050).  Demand math only; reduction stays on worker lanes.
        lod_swap_ready = session.mark_ladder_swaps_for_viewport()
        self.retarget_montage_pipeline(session)
        additions = viewport_plan.prioritize_tiles(additions)
        self._prune_stale_montage_tile_work(session)
        if not additions:
            if presentation_changed or lod_swap_ready:
                self.apply_montage_presentation(session)
            if session.pending_tiles and not _viewport_interaction_active(self):
                self.retarget_montage_pipeline(session)
                return True
            if session.pending_tiles:
                return True
            else:
                self._finish_montage_session_if_complete(session)
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
        if remaining_additions:
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
        self.show_montage_session_slow_overlay(session)
        montage_commit.submit_stage_tasks(self, session, stage_plan["stage_requests"])
        self.retarget_montage_pipeline(session)
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
        if not self._montage_session_is_current(session):
            return
        self.commit_montage_session_presentation(session)
        if (
            getattr(session, "show_loading_overlays", False)
            and not session.visible_plan_complete()
            and (session.pending_tiles or session.loading_tiles or session.active_tile_requests or session.stage_fan_in.attached_requests)
        ):
            self.win.img_view.setImageStale(True)
            self.win.img_view.setEvaluationOverlay(True, "Updating image frame...")

    def commit_montage_session_presentation(self, session) -> None:
        if not self._montage_session_is_current(session):
            return
        # One effects instance per session: it owns the in-flight rung guards
        # and the presentation gate flag. A throwaway instance would fork
        # that state (exactly the parallel-bookkeeping defect ADR 0051 bans).
        # Commit-only callers without a live pipeline (tests, teardown) get a
        # transient instance — they never submit rungs, so no guard state is
        # forked.
        pipeline = getattr(session, "pipeline", None)
        effects = pipeline.effects if pipeline is not None else MontagePipelineEffects(self, session)
        effects.commit_pending_session()

    def apply_ready_montage_display(self, session) -> None:
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
        session.final_commit_pending = True
        session.flush_pending = True
        self.commit_montage_session_presentation(session)

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
        semantic_histogram_data=getattr(cached, "semantic_histogram_data", None),
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
        semantic_histogram_data=getattr(value, "semantic_histogram_data", None),
        lod=getattr(value, "lod", None),
        level_data=getattr(value, "level_data", None),
        level_stats=getattr(value, "level_stats", None),
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


def _claim_preview_floor(session, tile_number: int, key) -> bool:
    cache = getattr(session, "pyramid_cache", None)
    if cache is None:
        return False
    tile_number = int(tile_number)
    if cache.peek(key) is not None:
        session.lifecycle.level_claimed(tile_number, key, ClaimOwner.PREVIEW, request=("preview-floor", key))
        session.lifecycle.level_resident(tile_number, key)
        return True
    if not cache.begin_pending(key):
        return False
    session.lifecycle.level_claimed(tile_number, key, ClaimOwner.PREVIEW, request=("preview-floor", key))
    session.lifecycle.level_materializing(tile_number, key)
    return True


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
    if level_stats.rank in {LevelSourceRank.MONTAGE_COMPLETE, LevelSourceRank.MONTAGE_SAMPLED_FULL}:
        return False
    return bool(
        getattr(session, "pending_tiles", None)
        or getattr(session, "loading_tiles", None)
        or getattr(session, "active_tile_requests", None)
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
