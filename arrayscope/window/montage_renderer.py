"""Progressive montage render orchestration for ArrayScope windows.

The montage path is large enough to have explicit ownership separate from
hover/profile/preference UI code.  It manages sessions, visible-tile planning,
stage-first tile scheduling, progressive canvas commits, and montage-specific
level source tracking.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from time import monotonic, perf_counter

import numpy as np
import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception
from arrayscope.core.cache_status import CacheStatus, CacheStatusSnapshot
from arrayscope.core.compute_policy import ComputeLane
from arrayscope.core.gui_callback_budget import GuiCallbackBudget, should_yield_after_item
from arrayscope.core.memory_budget import estimate_display_image_bytes, format_bytes
from arrayscope.core.view_state import ChannelMode
from arrayscope.display.geometry import DisplayGeometry, display_geometry_coordinates_equal
from arrayscope.display.imageview2d import MontageTileOverlay
from arrayscope.display.montage import (
    MontageTileState,
    RenderedTile,
    make_montage_plan,
    make_montage_viewport_canvas,
    montage_rect_for_viewport,
    optimal_montage_columns,
)
from arrayscope.display.slice_engine import DisplayImage, make_image_from_slab, make_shader_image_from_slab
from arrayscope.display.shader_mapping import apply_scale as apply_shader_scale, extract_component
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.operations.evaluator import EvaluationResult, _document_key, evaluate_image_snapshot, stage_document_key
from arrayscope.operations.chunked_stage import materialize_stage_candidate_chunked, stage_materialization_allowed_chunk_axes
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
from arrayscope.window.montage_payload_cache import (
    base_tile_source_id as _base_tile_source_id,
    limited_payload_cache as _limited_payload_cache,
    payload_lod_matches as _payload_lod_matches,
    payload_compatible_with_tile as _payload_compatible_with_tile,
    previous_tiled_payloads as _previous_tiled_payloads,
    previous_tiled_payloads_by_base_source as _previous_tiled_payloads_by_base_source,
)
from arrayscope.window.montage_prefetch import schedule_near_viewport_montage_prefetch
from arrayscope.window.montage_viewport import (
    MontageViewportPlan,
    montage_session_key,
    montage_viewport_update_delay_ms as _montage_viewport_update_delay_ms,
    prioritize_montage_tiles,
)
from arrayscope.window.montage_session import MontageRenderSession
from arrayscope.display.planning import LevelSourceRank, decide_presentation, fallback_level_source, normalize_bounds
from arrayscope.display.model.commit import CommitKind, DisplayPayload, PresentationInput
from arrayscope.display.model.frame import TiledValueSource
from arrayscope.window.display_presenter import tile_residency_budget_bytes


MONTAGE_VERY_SLOW_UPLOAD_MS = 100.0
MONTAGE_AUTOFIT_VISIBLE_FRACTION = 0.80


class MontageRenderMixin:
    def _montage_tile_layer_policy(self, geometry, data) -> bool:
        return self._montage_backend_policy(geometry, data).backend == "tile_layer"

    def _montage_backend_policy(self, geometry, data):
        return choose_montage_backend(
            geometry,
            data,
            setting=getattr(getattr(self, "app_settings", None), "montage_display_backend", "auto"),
            previous_upload_ms=float(getattr(self, "_last_set_image_ms", 0.0) or 0.0),
            patched_tiles=int(getattr(self, "_montage_patched_tiles_last_flush", 0) or 0),
            current_mode=str(getattr(self.img_view, "montageDisplayMode", lambda: "canvas")()),
            renderer_backend=getattr(self.img_view, "rendering_backend_name", "pyqtgraph"),
            renderer_capabilities=image_view_backend_capabilities(self.img_view),
            very_slow_upload_ms=MONTAGE_VERY_SLOW_UPLOAD_MS,
        )

    def _montage_viewport_plan(self, view_state) -> MontageViewportPlan:
        axis = view_state.montage_axis
        if axis is None:
            raise ValueError("montage viewport planning requires an active montage axis")
        all_indices = tuple(view_state.montage_indices or tuple(range(int(view_state.shape[axis]))))
        viewport_size = self.img_view.graphicsView.viewport().size()
        viewport_shape = (max(1, viewport_size.height()), max(1, viewport_size.width()))
        tile_shape = self._montage_tile_shape(view_state)
        columns = view_state.montage_columns
        if columns is None and all_indices:
            columns = optimal_montage_columns(len(all_indices), tile_shape, viewport_shape)
        plan = make_montage_plan(
            view_state,
            axis=axis,
            indices=all_indices,
            tile_shape=tile_shape,
            columns=columns,
            viewport_shape=viewport_shape,
        )
        current_range = (
            self._current_montage_global_view_range()
            if getattr(self.img_view, "image", None) is not None
            else None
        )
        priority_focus = _montage_priority_focus(self, current_range)
        capabilities = image_view_backend_capabilities(self.img_view)
        return MontageViewportPlan(
            axis=int(axis),
            all_indices=all_indices,
            viewport_shape=viewport_shape,
            tile_shape=tile_shape,
            plan=plan,
            view_range=current_range,
            shader_display=bool(capabilities.shader_windowing),
            persistent_tile_residency=bool(capabilities.persistent_tile_residency),
            priority_focus=priority_focus,
        )

    def update_montage_view(self, *, force_autolevel: bool = False, defer_side_panels: bool = False):
        for attribute in (
            "_last_montage_viewport_plan_ms",
            "_last_montage_cache_resolve_ms",
            "_last_montage_stage_plan_ms",
            "_last_montage_session_setup_ms",
            "_last_montage_initial_commit_ms",
        ):
            setattr(self, attribute, None)
        axis = self.view_state.montage_axis
        if axis is None or self.view_state.image_axes is None or axis in self.view_state.image_axes:
            return
        plan_start = perf_counter()
        policy = self._refresh_memory_policy(active_render=self._montage_render_active())
        user_levels = self._pending_display_levels_for_render()
        if force_autolevel and user_levels is not None:
            self._queue_display_levels(None)
            user_levels = None
        force_auto = bool(
            force_autolevel
            or (getattr(self, '_force_autolevel', False) and user_levels is None)
        )
        if getattr(self, '_force_autolevel', False):
            self._force_autolevel = False
        window_mode = self._current_window_mode()
        previous_frame = self._previous_display_frame_for_policy(force_auto=force_auto)
        pending_auto_level_source = getattr(self, "_pending_auto_level_source", None) if force_auto else None

        view_state = self.view_state
        shader_display = bool(image_view_backend_capabilities(self.img_view).shader_windowing)
        colormap_lut = self._evaluation_colormap_lut(view_state, shader_display=shader_display)
        document = self.document
        viewport_plan = self._montage_viewport_plan(view_state)
        all_indices = viewport_plan.all_indices
        viewport_shape = viewport_plan.viewport_shape
        tile_shape = viewport_plan.tile_shape
        plan = viewport_plan.plan
        if self._maybe_auto_fit_montage_tiles(plan.geometry):
            viewport_plan = self._montage_viewport_plan(view_state)
            viewport_shape = viewport_plan.viewport_shape
            tile_shape = viewport_plan.tile_shape
            plan = viewport_plan.plan
        current_range = viewport_plan.view_range
        canvas_rect = montage_rect_for_viewport(plan, view_range=current_range, viewport_shape=viewport_shape)
        display_tiles = viewport_plan.candidate_tiles(margin_tiles=0)
        candidate_tiles = viewport_plan.candidate_tiles(
            margin_tiles=1 if viewport_plan.persistent_tile_residency else 0,
            prioritize=True,
        )
        shader_display = viewport_plan.shader_display
        output_dtype = np.uint8 if view_state.channel == ChannelMode.COMPLEX and not shader_display else getattr(document.base_data, "dtype", np.dtype(float))
        canvas_estimate = estimate_display_image_bytes(
            (max(1, int(canvas_rect[3]) - int(canvas_rect[1])), max(1, int(canvas_rect[2]) - int(canvas_rect[0]))),
            output_dtype,
            rgb=view_state.channel == ChannelMode.COMPLEX,
            histogram=True,
        )
        canvas_warning_geometry = DisplayGeometry(
            view_state=view_state,
            display_shape=(max(1, int(canvas_rect[3]) - int(canvas_rect[1])), max(1, int(canvas_rect[2]) - int(canvas_rect[0]))),
            montage=plan.geometry,
        )
        if view_state.channel == ChannelMode.COMPLEX and not shader_display:
            backend_probe = np.zeros((1, 1, 3), dtype=np.uint8)
        else:
            backend_probe = np.zeros((1, 1), dtype=np.dtype(output_dtype))
        backend_decision = self._montage_backend_policy(canvas_warning_geometry, backend_probe)
        if backend_decision.backend == "canvas" and canvas_estimate > policy.montage_canvas_budget_bytes:
            show_status_message(
                self,
                f"Montage viewport canvas would allocate {format_bytes(canvas_estimate)} over budget {format_bytes(policy.montage_canvas_budget_bytes)}. Zoom in or increase Performance > Render Memory Budget.",
                timeout=6000,
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
                self,
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
        render_generation = self._capture_render_generation()
        stage_plan_start = perf_counter()
        stage_plan = self._plan_montage_stages(document, missing_tiles)
        self._last_montage_stage_plan_ms = (perf_counter() - stage_plan_start) * 1000.0
        session_setup_start = perf_counter()
        pending_tiles = [tile for tile in missing_tiles if int(tile.montage_index) not in stage_plan["waiting_indices"]]
        session_key = montage_session_key(_document_key(document), view_state, viewport_plan, colormap_lut)
        level_key = self._montage_level_key(document, view_state, all_indices, colormap_lut)
        session_id = int(getattr(self, "_montage_session_id", 0)) + 1
        self._montage_session_id = session_id
        session = MontageRenderSession(
            session_id=session_id,
            key=session_key,
            render_generation=render_generation,
            level_key=level_key,
            level_expected_indices=tuple(int(index) for index in all_indices),
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
            tile_stage_keys=stage_plan["tile_stage_keys"],
            tile_stage_plans=stage_plan["tile_stage_plans"],
            tile_stage_candidates=stage_plan["tile_stage_candidates"],
            stage_waiting_tiles=stage_plan["stage_waiting_tiles"],
            attached_stage_requests=stage_plan["attached_stage_keys"],
            stage_values=stage_plan["stage_values"],
            lead_stage_warmups=stage_plan["lead_stage_warmups"],
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
        )
        session.shader_display = bool(shader_display)
        self._montage_session = session
        if pending_auto_level_source is not None:
            self._pending_auto_level_source = None
        self._ensure_montage_level_stats(level_key, expected_indices=all_indices)
        self._queue_montage_cached_level_stats(session, cached_tiles, seed_if_empty=True)
        self._last_montage_session_setup_ms = (perf_counter() - session_setup_start) * 1000.0
        initial_commit_start = perf_counter()
        try:
            self._commit_montage_session_canvas(session, force=True)
        except MemoryError as exc:
            show_status_message(self, str(exc), timeout=6000)
            return
        finally:
            self._last_montage_initial_commit_ms = (perf_counter() - initial_commit_start) * 1000.0
        if session.is_complete():
            self._finish_montage_session_if_complete(session)
            if defer_side_panels:
                self._deferred_side_panel_refresh_pending = True
            else:
                self._update_operation_dock()
            self._schedule_montage_cached_level_stats(session)
            return
        self.prefetch_evaluation_controller.cancel_prefetch()
        self.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.COMPUTING, "Evaluating montage view")
        if defer_side_panels:
            self._deferred_side_panel_refresh_pending = True
        else:
            self._update_operation_dock()
        self._schedule_montage_session_slow_overlay(session)
        self._schedule_montage_cached_level_stats(session)
        self._schedule_montage_stage_jobs(session, stage_plan["stage_requests"])
        self._schedule_montage_attached_stage_waits(session)
        self._schedule_montage_tiles(session)

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
        previous_payloads = {
            key: payload
            for key, payload in dict(getattr(self, "_montage_recent_tile_payloads_by_base_source", {}) or {}).items()
            if _payload_lod_matches(payload, selected_lod_factor)
        }
        previous_payloads.update(
            {
                key: payload
                for key, payload in _previous_tiled_payloads_by_base_source(
                    getattr(self, "_committed_display_frame", None)
                ).items()
                if _payload_lod_matches(payload, selected_lod_factor)
            }
        )
        cached_tiles: list[RenderedTile] = []
        missing_tiles: list[object] = []
        total_lookup_ms = 0.0
        last_hit = False
        tile_tuple = tuple(tiles)
        for tile in tile_tuple:
            tile_cache_start = perf_counter()
            cached = self.operation_evaluator.cached_montage_tile(
                tile.view_state,
                montage_axis=axis,
                source_index=tile.source_index,
                colormap_lut=colormap_lut,
                shader_display=shader_display,
            )
            total_lookup_ms += (perf_counter() - tile_cache_start) * 1000.0
            last_hit = cached is not None
            if cached is None:
                tile_key = self.operation_evaluator.montage_tile_key(
                    tile.view_state,
                    montage_axis=axis,
                    source_index=tile.source_index,
                    colormap_lut=colormap_lut,
                    document=document,
                    shader_display=shader_display,
                )
                previous_payload = previous_payloads.get(tile_key)
                if previous_payload is None or not _payload_compatible_with_tile(
                    previous_payload,
                    tile.view_state,
                    shader_display=shader_display,
                ):
                    missing_tiles.append(tile)
                else:
                    cached_tiles.append(_rendered_tile_from_previous_payload(tile, previous_payload))
            else:
                cached_tiles.append(cached.bind(tile) if hasattr(cached, "bind") else cached.payload().bind(tile))
        self._last_montage_tile_cache_lookup_ms = total_lookup_ms
        self._last_montage_tile_cache_hit = last_hit if tile_tuple else False
        return cached_tiles, missing_tiles

    def _merge_montage_stage_plan(self, session: MontageRenderSession, stage_plan) -> None:
        session.tile_stage_keys.update(stage_plan["tile_stage_keys"])
        session.tile_stage_plans.update(stage_plan.get("tile_stage_plans", {}))
        session.tile_stage_candidates.update(stage_plan.get("tile_stage_candidates", {}))
        for key, waiting in dict(stage_plan["stage_waiting_tiles"]).items():
            existing = session.stage_waiting_tiles.get(key, ())
            existing_numbers = {int(tile.montage_index) for tile in existing}
            session.append_stage_waiting_tiles(
                key,
                tuple(tile for tile in waiting if int(tile.montage_index) not in existing_numbers),
            )
        session.attached_stage_requests.update(stage_plan["attached_stage_keys"])
        session.stage_values.update(stage_plan["stage_values"])
        session.lead_stage_warmups.update(stage_plan.get("lead_stage_warmups", {}))
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
        if session is None or not self._is_current_montage_session(session.session_id, session.key):
            return False
        capabilities = image_view_backend_capabilities(self.img_view)
        try:
            display_mode = str(self.img_view.montageDisplayMode())
        except Exception:
            display_mode = ""
        if not bool(capabilities.persistent_tile_residency) or "tile_layer" not in display_mode:
            return False

        view_state = self.view_state
        if view_state.montage_axis is None:
            return False
        viewport_plan = self._montage_viewport_plan(view_state)
        colormap_lut = self._evaluation_colormap_lut(
            view_state,
            shader_display=bool(capabilities.shader_windowing),
        )
        expected_key = montage_session_key(
            _document_key(self.document),
            view_state,
            viewport_plan,
            colormap_lut,
        )
        if session.key != expected_key:
            return False

        additions, presentation_changed = session.retarget_viewport(
            view_range=viewport_plan.view_range,
            viewport_shape=viewport_plan.viewport_shape,
            coverage_margin_tiles=1,
            near_margin_tiles=2,
            priority_focus=viewport_plan.priority_focus,
            priority_retarget_limit=_montage_priority_retarget_batch_limit(self, interactive=_interactive_active(self)),
        )
        additions = viewport_plan.prioritize_tiles(additions)
        self._prune_stale_montage_tile_work(session)
        if not additions:
            if presentation_changed:
                self._schedule_montage_canvas_commit(session, force=False)
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
            tile_cached, tile_missing = self._resolve_montage_tiles_from_cache(
                (tile,),
                document=session.document,
                axis=session.montage_axis,
                colormap_lut=session.colormap_lut,
                shader_display=bool(getattr(session, "shader_display", False)),
            )
            cached_tiles.extend(tile_cached)
            missing_tiles.extend(tile_missing)
            processed_additions += 1
            byte_count = sum(_rendered_tile_nbytes(rendered) for rendered in tile_cached)
            budget.record_item(byte_count=byte_count)
            if budget.should_yield():
                break
        self._record_gui_budget(budget)
        remaining_additions = max(0, len(additions_to_process) - int(processed_additions))
        self._last_montage_viewport_deferred_additions = int(remaining_additions)
        if remaining_additions:
            self._montage_viewport_update_pending = True
            self._montage_viewport_continue_immediately = True
        session.tile_compute_cache_hits += len(cached_tiles)
        for rendered in cached_tiles:
            session.mark_loaded(rendered)
        self._queue_montage_cached_level_stats(session, cached_tiles, seed_if_empty=False)

        self._montage_cached_tiles_last_session = len(cached_tiles)
        self._montage_missing_tiles_last_session = len(missing_tiles)
        if presentation_changed or cached_tiles:
            self._schedule_montage_canvas_commit(session, force=False)
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
            self._montage_viewport_update_pending = True
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

        self.prefetch_evaluation_controller.cancel_prefetch()
        self.operation_evaluator.last_status = CacheStatusSnapshot(
            CacheStatus.COMPUTING,
            "Extending montage viewport",
        )
        self._schedule_montage_session_slow_overlay(session)
        self._schedule_montage_stage_jobs(session, stage_plan["stage_requests"])
        self._schedule_montage_attached_stage_waits(session)
        self._schedule_montage_tiles(session)
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
        for key, waiting in list(session.stage_waiting_tiles.items()):
            kept = [tile for tile in waiting if int(tile.montage_index) in keep]
            if kept:
                if hasattr(waiting, "clear") and hasattr(waiting, "extend"):
                    waiting.clear()
                    waiting.extend(kept)
                    session.stage_waiting_tiles[key] = waiting
                else:
                    session.stage_waiting_tiles[key] = kept
            else:
                session.stage_waiting_tiles.pop(key, None)
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
        level_stats = getattr(rendered, "level_stats", None)
        if level_stats is not None and not refined:
            tracker.update_from_stats(level_key, level_stats, aggregate=False)
            return
        level_data = getattr(rendered, "level_data", None)
        if level_data is not None and not refined:
            stats = provisional_tile_level_stats(level_data, int(rendered.tile.source_index))
            if stats is not None:
                tracker.update_from_stats(level_key, stats, aggregate=False)
                return
        stats = sample_tile_level_stats(
            _montage_refined_level_values(rendered),
            int(rendered.tile.source_index),
            refined=bool(refined),
        )
        if stats is not None:
            tracker.update_from_stats(level_key, stats, aggregate=False)

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
        if not getattr(session, "pending_level_tiles", None):
            return
        timer = getattr(self, "_montage_level_stats_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._process_montage_cached_level_stats)
            self._montage_level_stats_timer = timer
        if not timer.isActive():
            timer.start(0)

    def _queue_montage_cached_level_stats(self, session, rendered_tiles, *, seed_if_empty: bool) -> None:
        """Queue cached-payload statistics without scanning every tile inline.

        Cache hits must make pixels available immediately.  Histogram sampling
        is secondary UI work, so only one tile is allowed to seed a brand-new
        semantic level scope before the first commit.  Remaining unseen sources
        are processed in bounded timer slices.
        """

        tracker = self._montage_level_tracker()
        expected = self._montage_level_expected_indices(session)
        tracker.ensure_expected(session.level_key, expected)
        pending = getattr(session, "pending_level_tiles", None)
        if pending is None:
            pending = deque()
            session.pending_level_tiles = pending
        queued_sources = getattr(session, "pending_level_sources", None)
        if queued_sources is None:
            queued_sources = {int(item.tile.source_index) for item in pending}
            session.pending_level_sources = queued_sources
        unseen = []
        for rendered in tuple(rendered_tiles or ()):
            source_index = int(rendered.tile.source_index)
            if source_index in queued_sources or tracker.has_source(session.level_key, source_index):
                continue
            unseen.append(rendered)
            queued_sources.add(source_index)

        summary = tracker.summary_for(session.level_key)
        if seed_if_empty and unseen and (summary is None or not summary.source_indices):
            seed = unseen.pop(0)
            self._update_montage_level_bounds_from_rendered(
                session.level_key,
                seed,
                expected_indices=expected,
            )
            self._queue_montage_level_refinement(session, seed)
        pending.extend(unseen)
        self._montage_pending_level_tiles_last_session = len(pending or ())

    def _ensure_montage_level_stats_for_payloads(self, session, payloads) -> int:
        """Merge stats for every tile that is about to be visible.

        Detailed histogram curves may lag, but automatic window/level state must
        already cover every payload in the same presentation commit.  This keeps
        bright or high-dynamic-range tiles from being drawn with levels derived
        from an earlier subset.
        """

        tracker = self._montage_level_tracker()
        expected = self._montage_level_expected_indices(session)
        tracker.ensure_expected(session.level_key, expected)
        stats_start = perf_counter()
        added = 0
        for tile_number in tuple(dict(payloads or {})):
            rendered = getattr(session, "rendered_tiles", {}).get(int(tile_number))
            if rendered is None:
                continue
            source_index = int(rendered.tile.source_index)
            if tracker.has_source(session.level_key, source_index):
                continue
            self._update_montage_level_bounds_from_rendered(
                session.level_key,
                rendered,
                expected_indices=expected,
            )
            self._queue_montage_level_refinement(session, rendered)
            added += 1
            pending_sources = getattr(session, "pending_level_sources", None)
            if pending_sources is not None:
                pending_sources.discard(source_index)
        self._last_montage_level_stats_ms = (perf_counter() - stats_start) * 1000.0
        self._montage_level_sources_added_last_commit = int(added)
        self._montage_pending_level_tiles_last_session = len(getattr(session, "pending_level_tiles", ()) or ())
        return int(added)

    def _process_montage_cached_level_stats(self) -> None:
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session.session_id, session.key):
            return
        pending = getattr(session, "pending_level_tiles", None)
        if not pending:
            return
        stats_start = perf_counter()
        expected = self._montage_level_expected_indices(session)
        processed = 0
        while pending and processed < 4:
            rendered = pending.popleft()
            source_index = int(rendered.tile.source_index)
            pending_sources = getattr(session, "pending_level_sources", None)
            if pending_sources is not None:
                pending_sources.discard(source_index)
            if not self._montage_level_tracker().has_source(session.level_key, source_index):
                self._update_montage_level_bounds_from_rendered(session.level_key, rendered, expected_indices=expected)
            self._queue_montage_level_refinement(session, rendered)
            processed += 1
            if processed >= 1 and (perf_counter() - stats_start) * 1000.0 >= 4.0:
                break
        self._last_montage_level_stats_ms = (perf_counter() - stats_start) * 1000.0
        self._montage_pending_level_tiles_last_session = len(pending or ())
        # A histogram/level refinement is presentation metadata.  It must not
        # force a full tiled-payload refresh or replay stale removals after a
        # viewport change.  Normal display commits will publish richer sources;
        # when there is no visible upload backlog, a non-forced commit can
        # update uniforms/histogram without invalidating residency.
        if (
            processed
            and session.display_committed
            and not getattr(session, "dirty_payloads", ())
            and not getattr(session, "pending_removals", ())
            and getattr(session, "user_levels_override", None) is None
        ):
            self._schedule_montage_canvas_commit(session, force=False)
        self._schedule_montage_cached_level_stats(session)

    def _schedule_montage_refined_level_stats(self, session) -> None:
        if not self._is_current_montage_session(session.session_id, session.key):
            return
        pending = getattr(session, "pending_refined_level_tiles", None)
        if not pending:
            return
        controller = getattr(self, "prefetch_evaluation_controller", None)
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

            started = controller.start_prefetch(
                evaluate,
                on_done=done,
                key=key,
                memory_budget_bytes=self._memory_policy().tile_cache_budget_bytes,
            )
            if not getattr(started, "scheduled", False):
                if str(getattr(started, "reason", "")) == "deduped":
                    scheduled += 1
                    continue
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
                self._schedule_montage_canvas_commit(session, force=False)
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
            key = self.operation_evaluator.stage_materializer.key_for_candidate(document_key, candidate)
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
            result = self.operation_evaluator.stage_materializer.request_stage(document_key, candidate)
            retained_stage_decision = result.decision
            if result.decision == "hit":
                stage_values[key] = result.value
                for tile in tiles:
                    tile_stage_keys[int(tile.montage_index)] = key
                    tile_stage_plans[int(tile.montage_index)] = tile_stage_plans.get(int(tile.montage_index), group["plan"])
                    tile_stage_candidates[int(tile.montage_index)] = candidate
                continue
            if result.decision == "scheduled":
                _direct_tiles, waiting_tiles = _lead_direct_tiles(tiles)
                lead_direct_tile_count += len(_direct_tiles)
                if waiting_tiles:
                    stage_waiting_tiles[key] = list(waiting_tiles)
                for tile in waiting_tiles:
                    tile_stage_keys[int(tile.montage_index)] = key
                    tile_stage_plans[int(tile.montage_index)] = tile_stage_plans.get(int(tile.montage_index), group["plan"])
                    tile_stage_candidates[int(tile.montage_index)] = candidate
                    waiting_indices.add(int(tile.montage_index))
                if _direct_tiles and _stage_fits_cache(candidate, self._memory_policy()):
                    for tile in _direct_tiles:
                        lead_stage_warmups[int(tile.montage_index)] = key
                else:
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

    def _schedule_montage_stage_jobs(self, session, stage_requests) -> None:
        if not self._is_current_montage_session(session.session_id, session.key):
            return
        controller = getattr(self, "stage_evaluation_controller", self.visible_evaluation_controller)
        for request, plan in tuple(stage_requests):
            if request is None or request.key in session.active_stage_requests:
                continue
            session.active_stage_requests.add(request.key)

            def evaluate(token, request=request, plan=plan):
                context = self._evaluation_context(ComputeLane.STAGE, token)
                return materialize_stage_candidate_chunked(
                    session.document,
                    plan.region_plan,
                    request.candidate,
                    stage_cache=self.operation_evaluator.stage_cache,
                    document_key=request.document_key,
                    cancellation_token=token,
                    evaluation_context=context,
                    memory_policy=context.memory_policy,
                    allowed_chunk_axes=stage_materialization_allowed_chunk_axes(request.candidate.shape),
                )

            controller.start_latest(
                evaluate,
                key=("stage", request.key),
                priority=EvalPriority.VISIBLE_IMAGE,
                replace_group=f"montage-stage:{int(session.session_id)}:{hash(request.key)}",
                on_done=lambda value, session_id=session.session_id, key=request.key: self._on_montage_stage_done(session_id, key, value),
                on_error=lambda exc, session_id=session.session_id, key=request.key: self._on_montage_stage_error(session_id, key, exc),
                on_stale=lambda key=request.key: self.operation_evaluator.stage_materializer.cancel(key),
                on_slow=lambda: self._on_montage_tile_slow(session.session_id),
                slow_ms=100,
                pass_token=True,
            )

    def _on_montage_stage_done(self, session_id, key, value) -> None:
        session = getattr(self, "_montage_session", None)
        self.operation_evaluator.stage_materializer.complete(key, value)
        if session is None or not self._is_current_montage_session(session_id, session.key):
            return
        if not self._is_current_render_generation(session.render_generation):
            return
        budget = self._montage_callback_budget(
            "montage_stage_wait",
            interactive=_interactive_active(self),
            work_class="stage_wait_activation",
        )
        self._activate_montage_stage_value(session, key, value, budget=budget)
        self._record_gui_budget(budget)
        self._schedule_montage_tiles(session)
        if getattr(session, "stage_waiting_tiles", None):
            self._schedule_montage_attached_stage_waits(session)

    def _activate_montage_stage_value(self, session, key, value, *, budget: GuiCallbackBudget | None = None) -> None:
        session.active_stage_requests.discard(key)
        session.attached_stage_requests.discard(key)
        session.stage_values[key] = value
        waiting = session.stage_waiting_tiles.get(key)
        if not waiting:
            session.stage_waiting_tiles.pop(key, None)
            return
        while waiting:
            tile = waiting.pop(0)
            index = int(tile.montage_index)
            if index not in session.rendered_tiles and index not in session.skipped_tiles:
                _enqueue_session_pending_tile(session, tile)
                session.mark_loading(tile)
            if budget is not None and should_yield_after_item(budget):
                break
        if waiting:
            session.stage_waiting_tiles[key] = waiting
        else:
            session.stage_waiting_tiles.pop(key, None)

    def _schedule_montage_attached_stage_waits(self, session) -> None:
        if not self._stage_wait_has_actionable_work(session, release_missing=True):
            return
        timer = getattr(self, "_montage_attached_stage_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._process_montage_attached_stage_waits)
            self._montage_attached_stage_timer = timer
        if not timer.isActive():
            timer.start(0)

    def _process_montage_attached_stage_waits(self) -> None:
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session.session_id, session.key):
            return
        pending_keys = tuple(
            dict.fromkeys(
                tuple(getattr(session, "attached_stage_requests", ()))
                + tuple(getattr(session, "stage_waiting_tiles", {}) or {})
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
        self._record_gui_budget(budget)
        if session.pending_tiles:
            self._schedule_montage_tiles(session)
        if self._stage_wait_has_actionable_work(session, release_missing=True):
            self._schedule_montage_attached_stage_waits(session)

    def _release_stage_waiting_tiles_to_direct(self, session, key, *, budget: GuiCallbackBudget | None = None) -> None:
        session.active_stage_requests.discard(key)
        session.attached_stage_requests.discard(key)
        waiting = session.stage_waiting_tiles.get(key)
        if not waiting:
            session.stage_waiting_tiles.pop(key, None)
            return
        while waiting:
            tile = waiting.pop(0)
            index = int(tile.montage_index)
            session.tile_stage_keys.pop(index, None)
            if index not in session.rendered_tiles and index not in session.skipped_tiles:
                _enqueue_session_pending_tile(session, tile)
                session.mark_loading(tile)
            if budget is not None and should_yield_after_item(budget):
                break
        if waiting:
            session.stage_waiting_tiles[key] = waiting
        else:
            session.stage_waiting_tiles.pop(key, None)

    def _activate_cached_waiting_stages(self, session, *, release_missing: bool = False) -> None:
        if not self._stage_wait_has_actionable_work(session, release_missing=release_missing):
            return
        budget = self._montage_callback_budget(
            "montage_stage_wait",
            interactive=_interactive_active(self),
            work_class="stage_wait_cached_activation",
        )
        for key in tuple(getattr(session, "stage_waiting_tiles", {})):
            self._activate_or_release_waiting_stage(session, key, release_missing=release_missing, budget=budget)
            if budget.should_yield():
                break
        self._record_gui_budget(budget)
        if self._stage_wait_has_actionable_work(session, release_missing=release_missing):
            self._schedule_montage_attached_stage_waits(session)

    def _activate_or_release_waiting_stage(self, session, key, *, release_missing: bool, budget: GuiCallbackBudget | None = None) -> None:
        if key in getattr(session, "stage_values", {}):
            self._activate_montage_stage_value(session, key, session.stage_values[key], budget=budget)
            return
        cache = self.operation_evaluator.stage_cache
        value = cache.get_containing(key) if hasattr(cache, "get_containing") else cache.get(key)
        if value is not None:
            self.operation_evaluator.stage_materializer.cancel(key)
            self._activate_montage_stage_value(session, key, value, budget=budget)
            return
        in_flight = getattr(self.operation_evaluator.stage_materializer, "_in_flight", {})
        if release_missing and key not in in_flight:
            self._release_stage_waiting_tiles_to_direct(session, key, budget=budget)

    def _stage_wait_has_actionable_work(self, session, *, release_missing: bool) -> bool:
        waiting_by_key = getattr(session, "stage_waiting_tiles", None)
        if not waiting_by_key:
            return False
        cache = self.operation_evaluator.stage_cache
        in_flight = getattr(self.operation_evaluator.stage_materializer, "_in_flight", {})
        for key, waiting in dict(waiting_by_key).items():
            if not waiting:
                continue
            if key in getattr(session, "stage_values", {}):
                return True
            value = cache.get_containing(key) if hasattr(cache, "get_containing") else cache.get(key)
            if value is not None:
                return True
            if release_missing and key not in in_flight:
                return True
        return False

    def _on_montage_stage_error(self, session_id, key, exc) -> None:
        session = getattr(self, "_montage_session", None)
        self.operation_evaluator.stage_materializer.fail(key, exc)
        if session is None or not self._is_current_montage_session(session_id, session.key):
            return
        session.active_stage_requests.discard(key)
        waiting = list(session.stage_waiting_tiles.pop(key, ()))
        for tile in waiting:
            session.mark_skipped(tile)
        show_status_message(self, f"Montage stage update failed: {exc}", timeout=4000)
        self._schedule_montage_canvas_commit(session, force=True)
        self._schedule_montage_tiles(session)

    def _warn_montage_tiles_skipped(self, *, skipped_count: int, tile_bytes: int, budget_bytes: int, tile_shape) -> None:
        message = (
            f"Montage skipped {int(skipped_count)} tile(s) because each tile would allocate "
            f"{format_bytes(int(tile_bytes))}, over the visible render budget of {format_bytes(int(budget_bytes))}. "
            f"Tile shape is {tuple(int(size) for size in tile_shape)}. Zoom in, crop/range the image axes, "
            "or increase Performance > Render Memory Budget."
        )
        show_status_message(self, message, timeout=8000)
        warning_key = (int(skipped_count), int(tile_bytes), int(budget_bytes), tuple(int(size) for size in tile_shape))
        if getattr(self, "_last_montage_skip_warning_key", None) == warning_key:
            return
        self._last_montage_skip_warning_key = warning_key
        try:
            Qt.QtWidgets.QMessageBox.warning(self, "Montage tiles skipped", message)
        except Exception as exc:
            handle_ui_exception("montage skipped warning", exc)

    def _schedule_montage_tiles(self, session: MontageRenderSession) -> None:
        if not self._is_current_montage_session(session.session_id, session.key):
            return
        if hasattr(self, "_apply_resource_governor_decisions"):
            self._apply_resource_governor_decisions()
        controller = getattr(self, "montage_tile_evaluation_controller", self.visible_evaluation_controller)
        max_workers = max(1, int(controller.pool.maxThreadCount()))
        while len(session.active_tile_requests) < max_workers:
            scheduled = self._schedule_next_montage_tile(session)
            if not scheduled:
                break

    def _schedule_next_montage_tile(self, session: MontageRenderSession) -> bool:
        if not self._is_current_montage_session(session.session_id, session.key):
            return False
        tile = session.next_tile()
        if tile is None:
            if session.active_tile_requests or session.loading_tiles or session.pending_completed_tiles:
                return False
            if session.active_stage_requests or session.stage_waiting_tiles:
                return False
            self._schedule_montage_canvas_commit(session, force=True)
            if session.pending_tiles:
                return self._schedule_next_montage_tile(session)
            if self._finish_montage_session_if_complete(session):
                if getattr(session, "defer_side_panels", False):
                    self._deferred_side_panel_refresh_pending = True
                else:
                    self._update_operation_dock()
            return False

        def evaluate(token):
            return self._evaluate_montage_tile_snapshot(session, tile, token)

        session_id = int(session.session_id)
        montage_axis = int(session.montage_axis)
        document = session.document
        colormap_lut = session.colormap_lut
        shader_display = bool(getattr(session, "shader_display", False))

        def done(result):
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

        controller = getattr(self, "montage_tile_evaluation_controller", self.visible_evaluation_controller)
        controller.start_latest(
            evaluate,
            key=("montage_tile", session.key, int(tile.montage_index)),
            priority=EvalPriority.VISIBLE_IMAGE,
            replace_group=f"montage-tile:{session_id}:{int(tile.montage_index)}",
            on_done=done,
            on_error=error,
            on_stale=lambda: None,
            on_slow=lambda: self._on_montage_tile_slow(session_id),
            slow_ms=100,
            pass_token=True,
        )
        return True

    def _evaluate_montage_tile_snapshot(self, session, tile, token=None):
        start = perf_counter()
        context = self._evaluation_context(ComputeLane.MONTAGE_TILE, token)
        try:
            stage_key = getattr(session, "tile_stage_keys", {}).get(int(tile.montage_index))
            stage_value = None if stage_key is None else getattr(session, "stage_values", {}).get(stage_key)
            if stage_value is not None:
                request = request_for_image(tile.view_state)
                plan = getattr(session, "tile_stage_plans", {}).get(int(tile.montage_index))
                candidate = getattr(session, "tile_stage_candidates", {}).get(int(tile.montage_index))
                if plan is None or candidate is None:
                    plan = plan_slab(session.document, request)
                    candidates = tuple(getattr(plan.region_plan, "cache_candidates", ()))
                    candidate = next(
                        (
                            candidate
                            for candidate in candidates
                            if self.operation_evaluator.stage_materializer.key_for_candidate(stage_document_key(session.document), candidate) == stage_key
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
                    display_image = _attach_montage_tile_level_stats(display_image, tile)
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
                stage_cache=self.operation_evaluator.stage_cache,
                stage_document_key=stage_document_key(session.document),
                evaluation_context=context,
            )
            result = replace(result, value=_attach_montage_tile_level_stats(result.value, tile))
            return result
        finally:
            self._last_montage_tile_eval_ms = (perf_counter() - start) * 1000.0

    def _on_montage_tile_slow(self, session_id):
        session = getattr(self, "_montage_session", None)
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
        if not session.pending_tiles and not session.loading_tiles and not getattr(session, "attached_stage_requests", None):
            return
        self._show_montage_session_loading_overlay(session)

    def _show_montage_session_loading_overlay(self, session):
        if not self._is_current_montage_session(session.session_id, session.key):
            return
        session.show_loading_overlays = True
        self._schedule_montage_canvas_commit(session, force=True)
        self.img_view.setImageStale(True)
        self.img_view.setEvaluationOverlay(True, "Updating montage...")
        self.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.COMPUTING, "Evaluating montage view")
        if getattr(session, "defer_side_panels", False) or _viewport_interaction_active(self):
            self._deferred_side_panel_refresh_pending = True
        else:
            self._update_operation_dock()

    def _on_montage_tile_done(self, session_id, tile, result, *, document, montage_axis: int, colormap_lut, shader_display: bool) -> None:
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

    def _store_reusable_montage_tile_result(self, tile, result, *, document, montage_axis: int, colormap_lut, shader_display: bool):
        if _document_key(document) != _document_key(self.document):
            return None
        stored = self.operation_evaluator.store_montage_tile_result(
            tile,
            montage_axis=montage_axis,
            colormap_lut=colormap_lut,
            result=result,
            shader_display=shader_display,
        )
        controller = getattr(self, "montage_tile_evaluation_controller", None)
        if stored is not None and controller is not None and hasattr(controller, "note_stale_reused"):
            controller.note_stale_reused()
        return stored

    def _schedule_montage_tile_result_flush(self, session) -> None:
        if not self._is_current_montage_session(session.session_id, session.key):
            return
        self._montage_tile_result_key = (int(session.session_id), session.key)
        if _vispy_tile_layer_fast_drain_enabled(self, session):
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
            _vispy_direct_tile_layer_backend(self, session)
            and not getattr(session, "display_committed", False)
        )
        first_batch_limit = 6
        expected_indices = self._montage_level_expected_indices(session)
        while session.pending_completed_tiles:
            tile, result = session.pending_completed_tiles.popleft()
            byte_count = self._apply_montage_tile_result(session, tile, result, expected_indices=expected_indices)
            processed_tiles.append(tile)
            processed += 1
            budget.record_item(byte_count=byte_count)
            if first_vispy_display and processed >= first_batch_limit:
                break
            if budget.should_yield():
                break
        if processed:
            elapsed_ms = (perf_counter() - flush_start) * 1000.0
            self._last_montage_tile_result_flush_ms = elapsed_ms
            self._last_montage_tile_result_flush_count = int(processed)
            self._record_gui_budget(budget)
            first_visible_committed = False
            if first_vispy_display and (getattr(session, "dirty_tiles", None) or getattr(session, "dirty_payloads", None)):
                self._commit_montage_session_canvas(session, force=False)
                first_visible_committed = bool(getattr(session, "display_committed", False))
            for tile in processed_tiles:
                self._resolve_lead_stage_warmup(session, tile)
            self._activate_cached_waiting_stages(session, release_missing=True)
            self._schedule_montage_cached_level_stats(session)
            if session.pending_tiles:
                self._schedule_montage_tiles(session)
            force = not session.pending_tiles and not session.active_tile_requests and not session.pending_completed_tiles
            if first_visible_committed:
                pass
            elif force:
                self._schedule_montage_canvas_commit(session, force=True)
            else:
                self._schedule_montage_ready_display_commit(session)
        if session.pending_completed_tiles:
            self._schedule_montage_tile_result_flush(session)

    def _apply_montage_tile_result(self, session, tile, result, *, expected_indices=None) -> int:
        if not self._is_current_montage_session(session.session_id, session.key):
            return 0
        if not self._is_current_render_generation(session.render_generation):
            return 0
        if _vispy_tile_layer_fast_drain_enabled(self, session):
            rendered = _rendered_tile_from_evaluation_result(tile, result)
        else:
            rendered = self.operation_evaluator.store_montage_tile_result(
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
        if session.has_canvas() and not _vispy_direct_tile_layer_backend(self, session):
            session.patch_rendered_tile(rendered)
            self._last_montage_canvas_patch_ms = (perf_counter() - patch_start) * 1000.0
        else:
            session.dirty_tiles.append(int(tile.montage_index))
            self._last_montage_canvas_patch_ms = 0.0
        return _rendered_tile_nbytes(rendered)

    def _resolve_lead_stage_warmup(self, session, tile) -> None:
        key = getattr(session, "lead_stage_warmups", {}).pop(int(tile.montage_index), None)
        if key is None:
            return
        cache = self.operation_evaluator.stage_cache
        value = cache.get_containing(key) if hasattr(cache, "get_containing") else cache.get(key)
        if value is not None:
            self.operation_evaluator.stage_materializer.complete(key, value)
            budget = self._montage_callback_budget(
                "montage_stage_wait",
                interactive=_interactive_active(self),
                work_class="stage_wait_activation",
            )
            self._activate_montage_stage_value(session, key, value, budget=budget)
            self._record_gui_budget(budget)
            return
        # The lead direct tile was expected to seed this reusable stage.  If it
        # did not, drop the fake in-flight marker so the existing release path
        # can fall back to direct work and still converge.
        self.operation_evaluator.stage_materializer.cancel(key)

    def _on_montage_tile_error(self, session_id, tile, exc) -> None:
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session_id, session.key):
            return
        if not self._is_current_render_generation(session.render_generation):
            return
        key = getattr(session, "lead_stage_warmups", {}).pop(int(tile.montage_index), None)
        if key is not None:
            self.operation_evaluator.stage_materializer.cancel(key)
        session.mark_skipped(tile)
        show_status_message(self, f"Montage tile update failed: {exc}", timeout=4000)
        self._schedule_montage_canvas_commit(session, force=True)
        self._schedule_montage_tiles(session)

    def _schedule_montage_canvas_commit(self, session, *, force=False) -> None:
        if not self._is_current_montage_session(session.session_id, session.key):
            return
        interval_ms = self._montage_commit_interval_ms(session, force=force)
        elapsed_ms = (monotonic() - float(session.last_commit_monotonic or 0.0)) * 1000.0
        needs_initial_commit = session.canvas is None and not bool(getattr(session, "display_committed", False))
        needs_final_dirty_commit = bool(
            force
            and not session.pending_tiles
            and not session.active_tile_requests
            and not session.pending_completed_tiles
            and (
                getattr(session, "dirty_payloads", None)
                or getattr(session, "pending_payload_upserts", None)
                or getattr(session, "pending_removals", None)
                or (bool(getattr(session, "pending_level_update", False)) and session.has_stale_level_presentations())
            )
        )
        if _viewport_interaction_active(self) and not needs_initial_commit:
            session.final_commit_pending = True
            session.flush_pending = True
            self._start_montage_commit_timer(max(1, int(interval_ms)))
            return
        if needs_initial_commit or needs_final_dirty_commit or force and not session.flush_pending or elapsed_ms >= interval_ms:
            self._commit_montage_session_canvas(session, force=force)
            return
        session.final_commit_pending = True
        session.flush_pending = True
        self._montage_coalesced_commits = int(getattr(self, "_montage_coalesced_commits", 0) or 0) + 1
        self._start_montage_commit_timer(max(1, int(interval_ms - elapsed_ms)))

    def _start_montage_commit_timer(self, interval_ms: int) -> None:
        timer = getattr(self, "_montage_commit_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._flush_montage_canvas_commit)
            self._montage_commit_timer = timer
        if not timer.isActive():
            timer.start(max(1, int(interval_ms)))

    def _montage_commit_interval_ms(self, session, *, force: bool) -> int:
        if force:
            if not session.pending_tiles and not session.loading_tiles and not session.active_tile_requests:
                return _montage_commit_interval_ms(self, force=True)
        interval = _montage_commit_interval_ms(self, force=False)
        if _vispy_tile_layer_fast_drain_enabled(self, session):
            return min(int(interval), 8)
        return interval

    @Qt.QtCore.Slot()
    def _flush_montage_canvas_commit(self):
        self._montage_canvas_commit_flush_queued = False
        session = getattr(self, "_montage_session", None)
        if session is None or not session.final_commit_pending:
            return
        self._commit_montage_session_canvas(session, force=False)
        if (
            getattr(session, "show_loading_overlays", False)
            and (session.pending_tiles or session.loading_tiles or session.active_tile_requests or getattr(session, "attached_stage_requests", None))
        ):
            self.img_view.setImageStale(True)
            self.img_view.setEvaluationOverlay(True, "Updating montage...")

    def _commit_montage_session_canvas(self, session, *, force=False) -> None:
        if not self._is_current_montage_session(session.session_id, session.key):
            return
        commit_start = perf_counter()
        self._classify_canvas_tiles(session)
        direct_presentation = self._direct_montage_tile_layer_presentation(session)
        if direct_presentation is not None:
            self._commit_montage_session_tile_layer(session, direct_presentation, commit_start=commit_start)
            return
        previous_canvas = getattr(self, "_current_montage_canvas", None)
        previous_global_range = self._current_montage_global_view_range()
        newly_composed = session.canvas is None
        if newly_composed:
            compose_start = perf_counter()
            canvas = make_montage_viewport_canvas(
                session.plan,
                session.rendered_tuple(),
                view_range=session.view_range,
                viewport_shape=session.viewport_shape,
                budget_bytes=self._montage_canvas_budget_bytes(),
                dtype=session.output_dtype,
                rgb=session.rgb,
                include_histogram=True,
                loading_tiles=session.loading_tile_tuple(),
                skipped_tiles=session.skipped_tile_tuple(),
            )
            self._last_montage_canvas_compose_ms = (perf_counter() - compose_start) * 1000.0
            session.initialize_canvas(canvas)
        else:
            canvas = session.current_canvas()
            object.__setattr__(canvas, "tile_states", tuple(session.tile_states))
            self._last_montage_canvas_compose_ms = 0.0
        dirty_rects = session.consume_dirty_rects()
        dirty_tiles = session.consume_dirty_tiles()
        tile_source_ids = self._montage_tile_source_ids(session)
        self._montage_patched_tiles_last_flush = len(dirty_rects)
        rendered_geometry = DisplayGeometry(
            view_state=session.view_state,
            display_shape=canvas.data.shape[:2],
            montage=session.plan.geometry,
            montage_origin_x=canvas.origin_x,
            montage_origin_y=canvas.origin_y,
            montage_tile_states=canvas.tile_states,
        )
        self._current_montage_geometry = session.plan.geometry
        self._current_montage_plan = session.plan
        self._current_montage_canvas = canvas
        self._next_viewport_policy = ViewportPolicy.PRESERVE
        self._montage_canvas_commit_active = True
        try:
            display_image = DisplayImage(data=canvas.data, histogram_data=canvas.histogram_data)
            level_stats = self._montage_level_stats_for_session(session)
            requested_levels = _session_requested_levels(session)
            explicit_auto = bool(getattr(session, "force_auto", False) and requested_levels is None)
            semantic_commit = bool(session.rendered_tiles)
            decision_force_auto = bool(explicit_auto and semantic_commit)
            first_display_commit = not bool(session.display_committed)
            publish_metadata = bool(explicit_auto) or self._should_publish_montage_level_metadata(session, level_stats)
            semantic_source = self._montage_level_source_for_session(session, allow_partial=publish_metadata)
            histogram_plot_data = (
                self._montage_histogram_plot_data_for_session(session, allow_partial=publish_metadata)
                if _should_publish_montage_histogram_plot(first_display_commit, explicit_auto, level_stats)
                else None
            )
            if newly_composed or first_display_commit:
                self._apply_full_display_image(
                    display_image,
                    geometry=rendered_geometry,
                    window_mode=session.window_mode,
                    previous_frame=getattr(self, "_committed_display_frame", None),
                    force_auto=decision_force_auto,
                    defer_side_panels=getattr(session, "defer_side_panels", False),
                    semantic_source=semantic_source,
                    applied_level_source=session.applied_level_source,
                    histogram_plot_data=histogram_plot_data,
                    commit_kind=CommitKind.EXPLICIT_AUTO_WINDOW if decision_force_auto else CommitKind.FULL_MONTAGE_INITIAL,
                    document_key=_document_key(session.document),
                    request_key=session.key,
                    render_generation=session.render_generation,
                    montage_level_key=session.level_key,
                    montage_dirty_tiles=dirty_tiles,
                    montage_tile_source_ids=tile_source_ids,
                    user_levels=requested_levels,
                    semantic_commit=semantic_commit,
                )
                session.display_committed = bool(session.rendered_tiles)
            else:
                self._apply_progressive_display_image(
                    display_image,
                    geometry=rendered_geometry,
                    window_mode=session.window_mode,
                    previous_frame=getattr(self, "_committed_display_frame", None),
                    force_auto=False,
                    viewport_policy=ViewportPolicy.PRESERVE,
                    semantic_source=semantic_source,
                    applied_level_source=session.applied_level_source,
                    histogram_plot_data=histogram_plot_data,
                    commit_kind=CommitKind.EXPLICIT_AUTO_WINDOW if explicit_auto else CommitKind.PROGRESSIVE_MONTAGE_PATCH,
                    document_key=_document_key(session.document),
                    request_key=session.key,
                    render_generation=session.render_generation,
                    montage_level_key=session.level_key,
                    montage_dirty_tiles=dirty_tiles,
                    montage_tile_source_ids=tile_source_ids,
                    user_levels=requested_levels,
                    semantic_commit=semantic_commit,
                )
            session.mark_presented(session.rendered_tiles.keys())
            session.display_committed = bool(session.presented_tiles)
            if session.canvas is not None:
                object.__setattr__(session.canvas, "tile_states", tuple(session.ensure_tile_states()))
            overlay_start = perf_counter()
            self._update_montage_tile_overlays(canvas)
            self._last_montage_overlay_update_ms = (perf_counter() - overlay_start) * 1000.0
        finally:
            self._montage_canvas_commit_active = False
        self._last_montage_canvas_commit_ms = (perf_counter() - commit_start) * 1000.0
        feedback = _latency_feedback(self)
        if feedback is not None:
            if hasattr(self, "_record_ui_work"):
                self._record_ui_work("montage_commit", self._last_montage_canvas_commit_ms)
            else:
                feedback.observe("montage_commit", self._last_montage_canvas_commit_ms)
        session.note_committed()
        self._finish_montage_session_if_complete(session)
        schedule_near_viewport_montage_prefetch(self, session)
        self._retry_live_profile_after_montage_tile()

    def _direct_montage_tile_layer_presentation(self, session):
        capabilities = image_view_backend_capabilities(self.img_view)
        if not capabilities.direct_montage_tile_payloads:
            return None
        if not hasattr(self.img_view, "setMontageTileLayerPresentation"):
            return None
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
        self._montage_patched_tiles_last_flush = len(dirty_tiles)
        self._last_montage_canvas_compose_ms = 0.0
        self._current_montage_geometry = session.plan.geometry
        self._current_montage_plan = session.plan
        self._current_montage_canvas = None
        self._next_viewport_policy = ViewportPolicy.PRESERVE
        self._montage_canvas_commit_active = True
        try:
            payload_start = perf_counter()
            previous_payloads = _previous_tiled_payloads(getattr(self, "_committed_display_frame", None))
            if previous_payloads and not getattr(session, "presented_tiles", None):
                session.seed_display_tile_payloads(previous_payloads, tile_source_ids)
            base_tile_state = session.tile_presentation_state
            fast_drain = _vispy_tile_layer_fast_drain_enabled(self, session)
            self._vispy_tile_layer_fast_drain_last_enabled = bool(fast_drain)
            self._vispy_tile_layer_fast_drain_enabled_count = int(
                getattr(self, "_vispy_tile_layer_fast_drain_enabled_count", 0) or 0
            ) + int(bool(fast_drain))
            tile_layer_limits = _tile_layer_upsert_limits(self, session, first_display_commit=not bool(session.display_committed))
            tile_state, tile_delta = session.build_tile_presentation(
                tile_source_ids,
                cold_deadline_ms=(
                    _montage_commit_budget_ms(self)
                    if tile_layer_limits
                    else None
                ),
                **tile_layer_limits,
            )
            if _vispy_direct_tile_layer_backend(self, session):
                dirty_tiles = tuple(int(tile) for tile in dirty_tiles if int(tile) in set(tile_delta.upserts))
            active_payloads = tile_state.active_payloads(tile_delta)
            first_display_commit = not bool(session.display_committed)
            if first_display_commit and not active_payloads:
                session.final_commit_pending = False
                session.flush_pending = False
                self._last_montage_tile_payload_build_ms = (perf_counter() - payload_start) * 1000.0
                if session.pending_tiles:
                    self._schedule_montage_tiles(session)
                return
            requested_levels = _session_requested_levels(session)
            explicit_auto = bool(getattr(session, "force_auto", False) and requested_levels is None)
            if (
                not first_display_commit
                and not explicit_auto
                and not getattr(tile_delta, "force_refresh", False)
                and not getattr(tile_delta, "upserts", None)
                and not getattr(tile_delta, "removals", None)
                and not (bool(getattr(session, "pending_level_update", False)) and session.has_stale_level_presentations())
            ):
                session.final_commit_pending = False
                session.flush_pending = False
                self._last_montage_tile_payload_build_ms = (perf_counter() - payload_start) * 1000.0
                session.note_committed()
                self._finish_montage_session_if_complete(session)
                if not (getattr(session, "dirty_payloads", None) or getattr(session, "pending_removals", None)):
                    schedule_near_viewport_montage_prefetch(self, session)
                self._retry_live_profile_after_montage_tile()
                return
            level_payloads = active_payloads if first_display_commit else dict(tile_delta.upserts)
            self._ensure_montage_level_stats_for_payloads(session, level_payloads)
            rendered_geometry = replace(
                rendered_geometry,
                montage_tile_states=session.ensure_tile_states(),
            )
            self._montage_recent_tile_payloads_by_base_source = _limited_payload_cache(
                getattr(self, "_montage_recent_tile_payloads_by_base_source", None),
                tile_state.payloads,
            )
            self._last_montage_tile_payload_build_ms = (perf_counter() - payload_start) * 1000.0
            level_stats = self._montage_level_stats_for_session(session)
            semantic_commit = bool(active_payloads)
            decision_force_auto = bool(explicit_auto and semantic_commit)
            publish_metadata = bool(explicit_auto) or self._should_publish_montage_level_metadata(session, level_stats)
            semantic_source = self._montage_level_source_for_session(session, allow_partial=publish_metadata)
            histogram_plot_data = (
                self._montage_histogram_plot_data_for_session(session, allow_partial=publish_metadata)
                if _should_publish_montage_histogram_plot(first_display_commit, explicit_auto, level_stats)
                else None
            )
            if first_display_commit:
                self._apply_full_display_image(
                    display_image,
                    geometry=rendered_geometry,
                    window_mode=session.window_mode,
                    previous_frame=getattr(self, "_committed_display_frame", None),
                    force_auto=decision_force_auto,
                    defer_side_panels=getattr(session, "defer_side_panels", False),
                    semantic_source=semantic_source,
                    applied_level_source=session.applied_level_source,
                    histogram_plot_data=histogram_plot_data,
                    commit_kind=CommitKind.EXPLICIT_AUTO_WINDOW if decision_force_auto else CommitKind.FULL_MONTAGE_INITIAL,
                    document_key=_document_key(session.document),
                    request_key=session.key,
                    render_generation=session.render_generation,
                    montage_level_key=session.level_key,
                    tile_state=tile_state,
                    base_tile_state=base_tile_state,
                    tile_delta=tile_delta,
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
                    previous_frame=getattr(self, "_committed_display_frame", None),
                    force_auto=False,
                    viewport_policy=ViewportPolicy.PRESERVE,
                    semantic_source=semantic_source,
                    applied_level_source=session.applied_level_source,
                    histogram_plot_data=histogram_plot_data,
                    commit_kind=CommitKind.EXPLICIT_AUTO_WINDOW if explicit_auto else CommitKind.PROGRESSIVE_MONTAGE_PATCH,
                    document_key=_document_key(session.document),
                    request_key=session.key,
                    render_generation=session.render_generation,
                    montage_level_key=session.level_key,
                    tile_state=tile_state,
                    base_tile_state=base_tile_state,
                    tile_delta=tile_delta,
                    user_levels=requested_levels,
                    semantic_commit=semantic_commit,
                )
            report = getattr(self._display_committer(), "last_tile_commit_report", None)
            session.acknowledge_tile_presentation(tile_delta, report, levels=normalize_bounds(self.img_view.getLevels()))
            if not session.has_stale_level_presentations():
                session.pending_level_update = False
            presented_tiles = active_payloads if report is None else getattr(report, "presented_tiles", active_payloads)
            session.mark_presented(presented_tiles)
            session.display_committed = bool(session.presented_tiles)
            rendered_geometry = replace(
                rendered_geometry,
                montage_tile_states=session.ensure_tile_states(),
            )
            self._sync_committed_montage_geometry(rendered_geometry)
            overlay_start = perf_counter()
            rect = montage_rect_for_viewport(session.plan, view_range=session.view_range, viewport_shape=session.viewport_shape)
            self._update_montage_tile_overlays_for_plan(session.plan, tuple(session.tile_states), rect)
            self._last_montage_overlay_update_ms = (perf_counter() - overlay_start) * 1000.0
        finally:
            self._montage_canvas_commit_active = False
        self._last_montage_canvas_commit_ms = (perf_counter() - commit_start) * 1000.0
        report = getattr(self._display_committer(), "last_tile_commit_report", None)
        cold_count = int(getattr(report, "cold_count", 0) or 0)
        warm_count = int(getattr(report, "existing_items_shown", 0) or 0) + int(getattr(report, "relocated_tiles", 0) or 0)
        processed_count = max(1, cold_count + warm_count)
        texture_upload_bytes = int(getattr(report, "texture_upload_bytes", 0) or 0)
        storage_rebuilds = int(getattr(report, "storage_rebuilds", 0) or 0)
        backend_name = str(getattr(self.img_view, "rendering_backend_name", "pyqtgraph"))
        feedback = _latency_feedback(self)
        if feedback is not None:
            cold_ms = 0.0
            if cold_count > 0 and storage_rebuilds <= 0:
                cold_ms = float(getattr(report, "cold_work_ms", 0.0) or 0.0) or self._last_montage_canvas_commit_ms
                if hasattr(self, "_record_ui_work"):
                    self._record_ui_work(
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
            commit_feedback_ms = self._last_montage_canvas_commit_ms
            commit_feedback_bytes = texture_upload_bytes
            vispy_backend = backend_name.lower() == "vispy"
            vispy_uniform_only = bool(
                vispy_backend
                and cold_count == 0
                and warm_count == 0
                and texture_upload_bytes == 0
            )
            if vispy_backend and storage_rebuilds > 0 and cold_count > 0:
                if hasattr(self, "_record_ui_work"):
                    self._record_ui_work(
                        "montage_layout_commit",
                        self._last_montage_canvas_commit_ms,
                        count=processed_count,
                        byte_count=texture_upload_bytes,
                        work_class="presentation_layout",
                        backend=backend_name,
                    )
                else:
                    feedback.observe(
                        "montage_layout_commit",
                        self._last_montage_canvas_commit_ms,
                        count=processed_count,
                        byte_count=texture_upload_bytes,
                    )
                commit_feedback_ms = 0.0
                commit_feedback_bytes = 0
            elif vispy_backend and (cold_count > 0 or vispy_uniform_only):
                if hasattr(self, "_record_ui_work"):
                    self._record_ui_work(
                        "montage_present_total",
                        self._last_montage_canvas_commit_ms,
                        count=processed_count,
                        byte_count=texture_upload_bytes,
                        work_class="presentation_upsert",
                        backend=backend_name,
                    )
                else:
                    feedback.observe(
                        "montage_present_total",
                        self._last_montage_canvas_commit_ms,
                        count=processed_count,
                        byte_count=texture_upload_bytes,
                    )
                commit_feedback_ms = max(0.0, self._last_montage_canvas_commit_ms - cold_ms)
                commit_feedback_bytes = 0
                if vispy_uniform_only:
                    commit_feedback_ms = 0.0
            if hasattr(self, "_record_ui_work"):
                self._record_ui_work(
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
            or (bool(getattr(session, "pending_level_update", False)) and session.has_stale_level_presentations())
        )
        session.note_committed()
        if upload_backlog:
            self._schedule_montage_ready_display_commit(session)
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
        if not _vispy_direct_tile_layer_backend(self, session):
            return False
        previous_frame = getattr(self, "_committed_display_frame", None)
        if previous_frame is None or not isinstance(getattr(previous_frame, "value_source", None), TiledValueSource):
            return False
        if not display_geometry_coordinates_equal(getattr(previous_frame, "geometry", None), geometry):
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
                commit_kind=CommitKind.EXPLICIT_AUTO_WINDOW if explicit_auto else CommitKind.PROGRESSIVE_MONTAGE_PATCH,
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
                    levels=decision.levels,
                    histogram_range=decision.histogram_range,
                    value_source=TiledValueSource(payloads),
                )
                self._set_committed_display_frame(frame)
                self._consume_pending_display_levels(session.user_levels_override)
                self._note_display_level_source(decision)
                refresh_hover = getattr(self, "_refresh_hover_after_display_commit", None)
                if callable(refresh_hover):
                    refresh_hover()
        self.apply_axis_flips()
        self.img_view.setImageStale(False)
        return True

    def _schedule_montage_ready_display_commit(self, session) -> None:
        if not self._is_current_montage_session(session.session_id, session.key):
            return
        if not (
            getattr(session, "dirty_rects", None)
            or getattr(session, "dirty_tiles", None)
            or getattr(session, "dirty_payloads", None)
            or getattr(session, "pending_payload_upserts", None)
            or getattr(session, "pending_removals", None)
            or (bool(getattr(session, "pending_level_update", False)) and session.has_stale_level_presentations())
        ):
            return
        if bool(getattr(session, "display_committed", False)):
            if _vispy_tile_layer_fast_drain_enabled(self, session):
                session.final_commit_pending = True
                session.flush_pending = True
                if self._queue_montage_canvas_commit_flush():
                    return
            self._schedule_montage_canvas_commit(session, force=False)
            return
        session.final_commit_pending = True
        session.flush_pending = True
        if _vispy_tile_layer_fast_drain_enabled(self, session) and self._queue_montage_canvas_commit_flush():
            return
        try:
            Qt.QtCore.QTimer.singleShot(0, self._flush_montage_canvas_commit)
        except Exception:
            self._start_montage_commit_timer(1)

    def _queue_montage_canvas_commit_flush(self) -> bool:
        if bool(getattr(self, "_montage_canvas_commit_flush_queued", False)):
            return True
        self._montage_canvas_commit_flush_queued = True
        try:
            queued = Qt.QtCore.QMetaObject.invokeMethod(
                self,
                "_flush_montage_canvas_commit",
                Qt.QtCore.Qt.ConnectionType.QueuedConnection,
            )
        except Exception:
            queued = False
        if queued:
            return True
        self._montage_canvas_commit_flush_queued = False
        return False

    def _finish_montage_session_if_complete(self, session) -> bool:
        if not self._is_current_montage_session(session.session_id, session.key):
            return False
        if not session.is_complete():
            return False
        session.show_loading_overlays = False
        self._stop_montage_session_slow_overlay()
        self.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.READY, "Montage view ready")
        self.img_view.setImageStale(False)
        self.img_view.setEvaluationOverlay(False)
        if hasattr(self.img_view, "clearMontageTileOverlays"):
            self.img_view.clearMontageTileOverlays()
        self._schedule_montage_refined_level_stats(session)
        return True

    def _sync_committed_montage_geometry(self, geometry) -> None:
        self.display_geometry = geometry
        frame = getattr(self, "_committed_display_frame", None)
        if (
            frame is not None
            and frame.geometry != geometry
            and not (
                isinstance(getattr(frame, "value_source", None), TiledValueSource)
                and display_geometry_coordinates_equal(frame.geometry, geometry)
            )
        ):
            self._set_committed_display_frame(replace(frame, geometry=geometry, scene=None))
        refresh_hover = getattr(self, "_refresh_hover_after_display_commit", None)
        if callable(refresh_hover):
            refresh_hover()

    def _maybe_auto_fit_montage_tiles(self, geometry) -> bool:
        montage = getattr(geometry, "montage", geometry)
        if montage is None or not getattr(montage, "indices", ()):
            self._last_montage_autofit_signature = None
            return False
        tile_count = len(tuple(montage.indices))
        full_range = _montage_full_view_range(montage)
        signature = _montage_autofit_signature(montage, full_range)
        previous_signature = getattr(self, "_last_montage_autofit_signature", None)
        self._last_montage_autofit_signature = signature
        if previous_signature is not None and not _montage_autofit_scope_grew(previous_signature, signature):
            return False
        viewport_controller = getattr(self.img_view, "viewport_controller", None)
        if viewport_controller is not None and viewport_controller.is_fit_locked():
            return False
        view = self.img_view.getView()
        before_range = _copy_view_range(view.viewRange())
        visible_count = _visible_montage_tile_count(montage, before_range)
        if tile_count <= 0 or visible_count / float(tile_count) > MONTAGE_AUTOFIT_VISIBLE_FRACTION:
            return False
        if _view_range_contains(before_range, full_range):
            return False
        previous_mode = None if viewport_controller is None else viewport_controller.mode
        self._set_montage_view_range(full_range)

        def undo():
            self._set_montage_view_range(before_range)
            if viewport_controller is not None and previous_mode is not None:
                viewport_controller.mode = previous_mode

        show_revert_action(
            self,
            "Fitted montage to show all tiles.",
            undo,
            timeout=5000,
        )
        return True

    def _set_montage_view_range(self, view_range) -> None:
        view = self.img_view.getView()
        was_applying = bool(getattr(self.img_view, "_viewport_applying", False))
        self.img_view._viewport_applying = True
        try:
            view.setRange(
                xRange=(float(view_range[0][0]), float(view_range[0][1])),
                yRange=(float(view_range[1][0]), float(view_range[1][1])),
                padding=0,
            )
        finally:
            self.img_view._viewport_applying = was_applying

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
        for tile_number, tile in sorted(plan_tiles.items()):
            if int(tile_number) in source_ids:
                continue
            try:
                source_ids[tile_number] = self.operation_evaluator.montage_tile_key(
                    tile.view_state,
                    montage_axis=session.montage_axis,
                    source_index=tile.source_index,
                    colormap_lut=session.colormap_lut,
                    document=session.document,
                    shader_display=bool(getattr(session, "shader_display", False)),
                )
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

    def _classify_canvas_tiles(self, session) -> None:
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

    def _update_montage_tile_overlays(self, canvas) -> None:
        self._update_montage_tile_overlays_for_plan(
            canvas.full_plan,
            canvas.tile_states,
            canvas.canvas_rect,
        )

    def _update_montage_tile_overlays_for_plan(self, plan, tile_states, canvas_rect) -> None:
        if not hasattr(self.img_view, "setMontageTileOverlays"):
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
                tuple(int(value) for value in canvas_rect),
                tile_state_revision,
                show_loading,
                candidate_numbers,
            )
            if key == getattr(self, "_last_montage_overlay_update_key", None):
                return
            self._last_montage_overlay_update_key = key
            if not candidate_numbers:
                if int(getattr(self.img_view, "montageTileOverlayCount", lambda: 0)() or 0) != 0:
                    self.img_view.setMontageTileOverlays(())
                return
        overlays = []
        canvas_x0, canvas_y0, canvas_x1, canvas_y1 = (int(value) for value in canvas_rect)
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
            x = max(tile_x0, canvas_x0)
            y = max(tile_y0, canvas_y0)
            x1 = min(tile_x1, canvas_x1)
            y1 = min(tile_y1, canvas_y1)
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
        self.img_view.setMontageTileOverlays(tuple(overlays))

    def _is_current_montage_session(self, session_id, key) -> bool:
        session = getattr(self, "_montage_session", None)
        if session is None:
            return False
        return int(session.session_id) == int(session_id) and session.key == key and self.view_state.montage_axis is not None

    def _retry_live_profile_after_montage_tile(self) -> None:
        try:
            if not self.widgets['buttons']['display']['live_profile'].isChecked():
                return
            position = self.img_view.profileMarkerPosition()
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
        decision = getattr(self, "_ui_work_decision", lambda *args, **kwargs: None)(
            channel,
            interactive=interactive,
        )
        budget = GuiCallbackBudget.for_decision(
            channel,
            decision,
            interactive=interactive,
            work_class=work_class,
            backend=str(getattr(self.img_view, "rendering_backend_name", "pyqtgraph")),
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
        recorder = getattr(getattr(self, "resource_governor", None), "record_gui_callback_observation", None)
        if callable(recorder):
            recorder(observation)
            return
        if hasattr(self, "_record_ui_work"):
            self._record_ui_work(
                observation.channel,
                observation.elapsed_ms,
                count=max(1, observation.processed_items),
                byte_count=observation.processed_bytes,
                work_class=observation.work_class,
                backend=observation.backend,
            )

    def _schedule_loading_montage_profile_retry(self, x, y) -> None:
        timer = getattr(self, "_montage_profile_retry_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._retry_loading_montage_profile)
            self._montage_profile_retry_timer = timer
        self._pending_montage_profile_retry = (float(x), float(y))
        if not timer.isActive():
            timer.start(80)

    def _retry_loading_montage_profile(self) -> None:
        point = getattr(self, "_pending_montage_profile_retry", None)
        self._pending_montage_profile_retry = None
        if point is None or self.view_state.montage_axis is None:
            return
        if not self.widgets['buttons']['display']['live_profile'].isChecked():
            return
        if not self.profile_dock.isVisible():
            self._schedule_loading_montage_profile_retry(float(point[0]), float(point[1]))
            return
        self._pending_profile_point = (float(point[0]), float(point[1]))
        self._pending_profile_pos = None
        self._update_live_profile_from_pending_pos()

    def _schedule_montage_viewport_update(self, *, delay_ms: int | None = None) -> None:
        if getattr(self, "_montage_viewport_update_running", False):
            self._montage_viewport_update_pending = True
            return
        timer = getattr(self, "_montage_viewport_update_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._run_montage_viewport_update)
            self._montage_viewport_update_timer = timer
        interval = _montage_viewport_update_delay_ms(self) if delay_ms is None else max(0, int(delay_ms))
        timer.start(interval)

    def _schedule_montage_priority_retarget_from_hover(self) -> None:
        if getattr(self, "_closing", False):
            return
        if getattr(self.view_state, "montage_axis", None) is None:
            return
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session.session_id, session.key):
            return
        if not (session.pending_tiles or getattr(session, "stage_waiting_tiles", None)):
            return
        timer = getattr(self, "_montage_priority_retarget_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._run_montage_priority_retarget)
            self._montage_priority_retarget_timer = timer
        self._montage_priority_retarget_pending = True
        if not timer.isActive():
            timer.start(_montage_priority_retarget_delay_ms(self))

    def _run_montage_priority_retarget(self) -> None:
        self._montage_priority_retarget_pending = False
        if getattr(self, "_closing", False):
            return
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session.session_id, session.key):
            return
        if not (session.pending_tiles or getattr(session, "stage_waiting_tiles", None)):
            return
        budget = self._montage_callback_budget(
            "montage_priority_retarget",
            interactive=True,
            work_class="queue_metadata",
            item_cap=_montage_priority_retarget_batch_limit(self, interactive=True),
        )
        viewport_plan = self._montage_viewport_plan(self.view_state)
        session.priority_focus = viewport_plan.priority_focus
        processed = session.retarget_tile_priority(
            focus=viewport_plan.priority_focus,
            max_items=budget.item_cap,
        )
        if processed:
            budget.record_item(item_count=processed)
        self._last_montage_priority_retarget_count = int(processed)
        self._last_montage_priority_retarget_pending = len(session.pending_tiles)
        self._record_gui_budget(budget)
        if session.pending_tiles:
            self._schedule_montage_tiles(session)

    def _run_montage_viewport_update(self) -> None:
        if getattr(self, "_closing", False):
            return
        if self.view_state.montage_axis is None:
            return
        if getattr(self, "_montage_viewport_update_running", False):
            self._montage_viewport_update_pending = True
            return
        self._montage_viewport_update_running = True
        self._montage_viewport_update_pending = False
        try:
            if not self._try_update_montage_viewport_only():
                self.update_montage_view()
        finally:
            self._montage_viewport_update_running = False
        if getattr(self, "_montage_viewport_update_pending", False) and self.view_state.montage_axis is not None:
            self._montage_viewport_update_pending = False
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
            view_range = self.img_view.getView().viewRange()
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


def _montage_autofit_signature(montage, full_range) -> tuple[int, float, float, int, int]:
    width = max(0.0, float(full_range[0][1]) - float(full_range[0][0]))
    height = max(0.0, float(full_range[1][1]) - float(full_range[1][0]))
    return (
        len(tuple(getattr(montage, "indices", ()) or ())),
        width,
        height,
        int(getattr(montage, "columns", 0) or 0),
        int(getattr(montage, "rows", 0) or 0),
    )


def _montage_autofit_scope_grew(previous, current) -> bool:
    try:
        previous_count, previous_width, previous_height, _previous_columns, _previous_rows = previous
        current_count, current_width, current_height, _current_columns, _current_rows = current
    except Exception:
        return True
    return (
        int(current_count) > int(previous_count)
        or float(current_width) > float(previous_width)
        or float(current_height) > float(previous_height)
    )


def _montage_tile_layer_placeholder(session) -> np.ndarray:
    height, width = (max(1, int(value)) for value in session.plan.display_shape)
    if bool(getattr(session, "rgb", False)):
        base = np.zeros((1, 1, 3), dtype=np.uint8)
        return np.broadcast_to(base, (height, width, 3))
    base = np.zeros((1, 1), dtype=np.float32)
    return np.broadcast_to(base, (height, width))


def _lead_direct_tiles(tiles, *, count: int = 1):
    tiles = tuple(tiles)
    count = max(0, min(int(count), len(tiles)))
    return tiles[:count], tiles[count:]


def _enqueue_session_pending_tile(session, tile) -> None:
    enqueue = getattr(session, "enqueue_pending_tile", None)
    if callable(enqueue):
        enqueue(tile)
        return
    session.pending_tiles.append(tile)


def _stage_fits_cache(candidate, memory_policy) -> bool:
    estimated = getattr(candidate, "estimated_nbytes", None)
    if estimated is None:
        return False
    return int(estimated) <= int(getattr(memory_policy, "stage_cache_budget_bytes", 0) or 0)


def _montage_tile_result_batch_limit(window, *, interactive: bool) -> int:
    configured = getattr(window, "_montage_tile_result_batch_size", None)
    if configured is not None:
        return max(1, int(configured))
    decision = getattr(window, "_ui_work_decision", lambda *args, **kwargs: None)("montage_tile_result", interactive=interactive)
    if decision is not None:
        return max(1, int(decision.batch_limit))
    feedback = _latency_feedback(window)
    if feedback is None:
        return 4 if interactive else 8
    return int(feedback.batch_limit("montage_tile_result", interactive=interactive))


def _montage_viewport_addition_batch_limit(window, *, interactive: bool) -> int:
    configured = getattr(window, "_montage_viewport_addition_batch_size", None)
    if configured is not None:
        return max(1, int(configured))
    decision = getattr(window, "_ui_work_decision", lambda *args, **kwargs: None)("montage_viewport_update", interactive=interactive)
    if decision is not None:
        return max(1, min(32, int(decision.batch_limit)))
    return 8 if interactive else 16


def _montage_viewport_chunk_delay_ms(window) -> int:
    decision = getattr(window, "_ui_work_decision", lambda *args, **kwargs: None)(
        "montage_viewport_update",
        interactive=_interactive_active(window),
    )
    if decision is not None:
        return max(1, min(16, int(decision.interval_ms)))
    return 1


def _montage_priority_retarget_delay_ms(window) -> int:
    decision = getattr(window, "_ui_work_decision", lambda *args, **kwargs: None)(
        "montage_priority_retarget",
        interactive=True,
    )
    if decision is not None:
        return max(1, min(64, int(decision.interval_ms)))
    return 32


def _montage_priority_retarget_batch_limit(window, *, interactive: bool) -> int:
    decision = getattr(window, "_ui_work_decision", lambda *args, **kwargs: None)(
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


def _attach_montage_tile_level_stats(display_image, tile):
    level_data = getattr(display_image, "level_data", None)
    if level_data is not None:
        stats = provisional_tile_level_stats(level_data, int(tile.source_index))
        if stats is not None:
            return replace(display_image, level_stats=stats)
    return display_image


def _session_requested_levels(session) -> tuple[float, float] | None:
    """Return the exact level target for the current presentation session.

    ``user_levels_override`` describes persistence/window-mode semantics.
    ``desired_level_values`` is the latest presentation command and remains
    authoritative while progressive PyQtGraph CPU redraws converge.  Keeping
    these concepts separate prevents an in-flight automatic commit from
    restoring older levels after a user or auto-window command.
    """

    return (
        normalize_bounds(getattr(session, "desired_level_values", None))
        or normalize_bounds(getattr(session, "user_levels_override", None))
    )


def _should_publish_montage_histogram_plot(first_display_commit: bool, explicit_auto: bool, stats: MontageLevelStats) -> bool:
    if bool(first_display_commit) or bool(explicit_auto):
        return True
    return stats.rank in {LevelSourceRank.MONTAGE_COMPLETE, LevelSourceRank.MONTAGE_SAMPLED_FULL}


def _montage_commit_budget_ms(window) -> float:
    interactive = _interactive_active(window)
    decision = getattr(window, "_ui_work_decision", lambda *args, **kwargs: None)(
        "montage_commit",
        interactive=interactive,
    )
    if decision is not None:
        return max(1.0, float(decision.budget_ms))
    feedback = _latency_feedback(window)
    if feedback is None:
        return 4.0 if interactive else 8.0
    return max(1.0, float(feedback.work_budget_ms("montage_commit", interactive=interactive)))


def _vispy_tile_layer_fast_drain_enabled(window, session) -> bool:
    if _viewport_interaction_active(window):
        return False
    if not bool(getattr(session, "display_committed", False)):
        return False
    return _vispy_direct_tile_layer_backend(window, session)


def _vispy_direct_tile_layer_backend(window, session) -> bool:
    if str(getattr(getattr(window, "img_view", None), "rendering_backend_name", "")).lower() != "vispy":
        return False
    capabilities = image_view_backend_capabilities(getattr(window, "img_view", None))
    return bool(capabilities.direct_montage_tile_payloads and capabilities.persistent_tile_residency)


def _vispy_tile_upsert_limits(window, session, *, first_display_commit: bool) -> dict[str, int]:
    if not _vispy_direct_tile_layer_backend(window, session):
        return {}
    decision = getattr(window, "_ui_work_decision", lambda *args, **kwargs: None)(
        "montage_present_total",
        interactive=_interactive_active(window),
    )
    upload_decision = getattr(window, "_ui_work_decision", lambda *args, **kwargs: None)(
        "montage_cold_commit",
        interactive=_interactive_active(window),
    )
    batch_limit = int(getattr(decision, "batch_limit", 0) or 0)
    byte_cap = max(
        int(getattr(decision, "byte_cap", 0) or 0),
        int(getattr(upload_decision, "byte_cap", 0) or 0),
    )
    if batch_limit <= 0:
        feedback = _latency_feedback(window)
        batch_limit = 4 if feedback is None else int(feedback.batch_limit("montage_present_total", interactive=_interactive_active(window)))
    if byte_cap <= 0:
        byte_cap = 8 * 1024 * 1024 if _interactive_active(window) else 32 * 1024 * 1024
    max_upserts = min(6, int(batch_limit)) if bool(first_display_commit) else int(batch_limit)
    limits = {
        "max_upserts": max(1, int(max_upserts)),
        "max_upsert_bytes": max(1024, int(byte_cap)),
    }
    return limits


def _tile_layer_upsert_limits(window, session, *, first_display_commit: bool) -> dict[str, int]:
    if _vispy_direct_tile_layer_backend(window, session):
        return _vispy_tile_upsert_limits(window, session, first_display_commit=first_display_commit)
    capabilities = image_view_backend_capabilities(getattr(window, "img_view", None))
    if not (
        capabilities.direct_montage_tile_payloads
        and not capabilities.shader_windowing
        and bool(getattr(session, "pending_level_update", False))
        and session.has_stale_level_presentations()
    ):
        return {}
    decision = getattr(window, "_ui_work_decision", lambda *args, **kwargs: None)(
        "tile_layer_commit",
        interactive=_interactive_active(window),
    )
    batch_limit = int(getattr(decision, "batch_limit", 0) or 0)
    byte_cap = int(getattr(decision, "byte_cap", 0) or 0)
    if batch_limit <= 0:
        feedback = _latency_feedback(window)
        batch_limit = 8 if feedback is None else int(feedback.batch_limit("tile_layer_commit", interactive=_interactive_active(window)))
    if byte_cap <= 0:
        byte_cap = 8 * 1024 * 1024 if _interactive_active(window) else 32 * 1024 * 1024
    return {
        "max_upserts": max(1, int(batch_limit)),
        "max_upsert_bytes": max(1024, int(byte_cap)),
    }


def _montage_commit_interval_ms(window, *, force: bool) -> int:
    decision = getattr(window, "_ui_work_decision", lambda *args, **kwargs: None)("montage_commit", interactive=_interactive_active(window))
    if decision is not None:
        return int(8 if force else max(1, decision.interval_ms))
    feedback = _latency_feedback(window)
    if feedback is None:
        return 8 if force else 16
    return int(feedback.commit_interval_ms("montage_commit", force=force, interactive=_interactive_active(window)))


def _latency_feedback(window):
    return getattr(window, "latency_feedback", None)


def _viewport_gesture_active() -> bool:
    try:
        buttons = Qt.QtWidgets.QApplication.mouseButtons()
        return bool(buttons & Qt.QtCore.Qt.MouseButton.LeftButton)
    except Exception:
        return False


def _montage_priority_focus(window, view_range) -> tuple[float, float] | None:
    """Return the user-attention point for scheduling visible montage tiles."""

    pos = getattr(window, "_last_image_mouse_scene_pos", None)
    if pos is not None:
        try:
            graphics_view = window.img_view.graphicsView
            viewport_point = graphics_view.mapFromScene(pos)
            if not graphics_view.viewport().rect().contains(viewport_point):
                raise ValueError("stored hover point is outside the image viewport")
            point = window.img_view.getView().mapSceneToView(pos)
            x = float(point.x())
            y = float(point.y())
            geometry = getattr(window, "display_geometry", None)
            status = None if geometry is None else geometry.view_point_to_tile_point(x, y, require_loaded=False)
            if status is not None and status.kind not in {"outside", "gap"}:
                return (x, y)
        except Exception:
            pass
    try:
        x_range, y_range = view_range
        return (
            (float(x_range[0]) + float(x_range[1])) * 0.5,
            (float(y_range[0]) + float(y_range[1])) * 0.5,
        )
    except Exception:
        return None


def _interactive_active(window) -> bool:
    coordinator = getattr(window, "render_coordinator", None)
    return bool(
        coordinator is not None and getattr(coordinator, "interactive_active", False)
        or _viewport_interaction_active(window)
    )


def _viewport_interaction_active(window) -> bool:
    return bool(getattr(window, "_viewport_interaction_active", False) or _viewport_gesture_active())
