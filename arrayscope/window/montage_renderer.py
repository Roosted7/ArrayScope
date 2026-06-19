"""Progressive montage render orchestration for ArrayScope windows.

The montage path is large enough to have explicit ownership separate from
hover/profile/preference UI code.  It manages sessions, visible-tile planning,
stage-first tile scheduling, progressive canvas commits, and montage-specific
level source tracking.
"""

from __future__ import annotations

from time import monotonic, perf_counter

import numpy as np
import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception
from arrayscope.core.cache_status import CacheStatus, CacheStatusSnapshot
from arrayscope.core.compute_policy import ComputeLane
from arrayscope.core.memory_budget import estimate_display_image_bytes, format_bytes
from arrayscope.core.view_state import ChannelMode
from arrayscope.display.geometry import DisplayGeometry
from arrayscope.display.imageview2d import MontageTileOverlay
from arrayscope.display.lod import select_lod_factor
from arrayscope.display.montage import (
    MontageTileState,
    RenderedTile,
    make_montage_plan,
    make_montage_viewport_canvas,
    montage_rect_for_viewport,
    optimal_montage_columns,
)
from arrayscope.display.slice_engine import DisplayImage, make_image_from_slab, make_shader_image_from_slab
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.operations.evaluator import EvaluationResult, _document_key, evaluate_image_snapshot, stage_document_key
from arrayscope.operations.chunked_stage import materialize_stage_candidate_chunked, stage_materialization_allowed_chunk_axes
from arrayscope.operations.slabs import (
    evaluate_slab_from_stage,
    plan_slab,
    request_for_image,
)
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.evaluation_controller import EvalPriority
from arrayscope.window.montage_backend import choose_montage_backend
from arrayscope.window.montage_levels import MontageLevelStats, MontageLevelTracker
from arrayscope.window.montage_prefetch import schedule_near_viewport_montage_prefetch
from arrayscope.window.montage_session import MontageRenderSession
from arrayscope.window.presentation import LevelSourceRank, fallback_level_source
from arrayscope.window.render_model import CommitKind


MONTAGE_VERY_SLOW_UPLOAD_MS = 100.0


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

    def update_montage_view(self, *, force_autolevel: bool = False, defer_side_panels: bool = False):
        axis = self.view_state.montage_axis
        if axis is None or self.view_state.image_axes is None or axis in self.view_state.image_axes:
            return
        policy = self._refresh_memory_policy(active_render=self._montage_render_active())
        force_auto = force_autolevel or getattr(self, '_force_autolevel', False)
        if getattr(self, '_force_autolevel', False):
            self._force_autolevel = False
        window_mode = self._current_window_mode()
        previous_frame = self._previous_display_frame_for_policy(force_auto=force_auto)

        colormap_lut = None
        if self.view_state.channel == ChannelMode.COMPLEX:
            colormap_lut = self._phase_colormap().getLookupTable(0.0, 1.0, 256, alpha=False)
        view_state = self.view_state
        document = self.document
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
        current_range = self._current_montage_global_view_range() if getattr(self.img_view, "image", None) is not None else None
        canvas_rect = montage_rect_for_viewport(plan, view_range=current_range, viewport_shape=viewport_shape)
        candidate_tiles = plan.tiles_intersecting(((canvas_rect[0], canvas_rect[2]), (canvas_rect[1], canvas_rect[3])), margin_tiles=0)
        shader_display = bool(image_view_backend_capabilities(self.img_view).shader_windowing)
        output_dtype = np.uint8 if view_state.channel == ChannelMode.COMPLEX and not shader_display else getattr(document.base_data, "dtype", np.dtype(float))
        canvas_estimate = estimate_display_image_bytes(
            (max(1, int(canvas_rect[3]) - int(canvas_rect[1])), max(1, int(canvas_rect[2]) - int(canvas_rect[0]))),
            output_dtype,
            rgb=view_state.channel == ChannelMode.COMPLEX,
            histogram=True,
        )
        if canvas_estimate > policy.montage_canvas_budget_bytes:
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
            visible_tiles = ()
            skipped_tiles = tuple(candidate_tiles)
            skipped_count = len(skipped_tiles)
        else:
            visible_tiles = tuple(candidate_tiles)
            skipped_tiles = ()
            skipped_count = 0
        if not visible_tiles and not skipped_tiles:
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
        cached_tiles = []
        missing_tiles = []
        selected_lod_factor = select_lod_factor(current_range, viewport_shape, plan.tile_shape)
        previous_payloads = {
            key: payload
            for key, payload in dict(getattr(self, "_montage_recent_tile_payloads_by_base_source", {}) or {}).items()
            if _payload_lod_matches(payload, selected_lod_factor)
        }
        previous_payloads.update(
            {
                key: payload
                for key, payload in _previous_tiled_payloads_by_base_source(getattr(self, "_committed_display_frame", None)).items()
                if _payload_lod_matches(payload, selected_lod_factor)
            }
        )
        for tile in visible_tiles:
            tile_cache_start = perf_counter()
            cached = self.operation_evaluator.cached_montage_tile(
                tile.view_state,
                montage_axis=axis,
                source_index=tile.source_index,
                colormap_lut=colormap_lut,
                shader_display=shader_display,
            )
            self._last_montage_tile_cache_lookup_ms = (perf_counter() - tile_cache_start) * 1000.0
            self._last_montage_tile_cache_hit = cached is not None
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
                if previous_payload is None:
                    missing_tiles.append(tile)
                else:
                    cached_tiles.append(_rendered_tile_from_previous_payload(tile, previous_payload))
            else:
                cached_tiles.append(cached.bind(tile) if hasattr(cached, "bind") else cached.payload().bind(tile))
        self._montage_cached_tiles_last_session = len(cached_tiles)
        self._montage_missing_tiles_last_session = len(missing_tiles)
        render_generation = self._capture_render_generation()
        stage_plan = self._plan_montage_stages(document, missing_tiles)
        pending_tiles = [tile for tile in missing_tiles if int(tile.montage_index) not in stage_plan["waiting_indices"]]
        session_key = (
            "montage_tiles",
            _document_key(document),
            view_state,
            tuple(tile.source_index for tile in candidate_tiles),
            colormap_lut.tobytes() if colormap_lut is not None else None,
            viewport_shape if view_state.montage_columns is None else None,
            bool(shader_display),
        )
        level_key = self._montage_level_key(document, view_state, all_indices, colormap_lut)
        session_id = int(getattr(self, "_montage_session_id", 0)) + 1
        self._montage_session_id = session_id
        session = MontageRenderSession(
            session_id=session_id,
            key=session_key,
            render_generation=render_generation,
            level_key=level_key,
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
            visible_tiles=tuple(visible_tiles),
            rendered_tiles={int(rendered.tile.montage_index): rendered for rendered in cached_tiles},
            loading_tiles={int(tile.montage_index) for tile in missing_tiles},
            skipped_tiles={int(tile.montage_index) for tile in skipped_tiles},
            pending_tiles=list(pending_tiles),
            tile_stage_keys=stage_plan["tile_stage_keys"],
            stage_waiting_tiles=stage_plan["stage_waiting_tiles"],
            attached_stage_requests=stage_plan["attached_stage_keys"],
            stage_values=stage_plan["stage_values"],
            defer_side_panels=bool(defer_side_panels),
            applied_level_source=None if previous_frame is None else fallback_level_source(previous_frame),
            tile_compute_cache_hits=len(cached_tiles),
            tile_compute_waiting_for_stage=len(stage_plan["waiting_indices"]),
            lead_direct_tiles=stage_plan["lead_direct_tiles"],
            stage_backed_tiles_pending=len(stage_plan["waiting_indices"]),
            retained_stage_index=stage_plan["retained_stage_index"],
            retained_stage_decision=stage_plan["retained_stage_decision"],
            repeated_expensive_stage_per_tile=stage_plan["repeated_expensive_stage_per_tile"],
        )
        session.shader_display = bool(shader_display)
        self._montage_session = session
        self._ensure_montage_level_stats(level_key, expected_indices=all_indices)
        immediate_level_tiles = tuple(cached_tiles[:4])
        deferred_level_tiles = list(cached_tiles[4:])
        session.pending_level_tiles = deferred_level_tiles
        self._montage_pending_level_tiles_last_session = len(deferred_level_tiles)
        for rendered in immediate_level_tiles:
            self._update_montage_level_bounds_from_rendered(level_key, rendered, expected_indices=all_indices)
        try:
            self._commit_montage_session_canvas(session, force=True)
        except MemoryError as exc:
            show_status_message(self, str(exc), timeout=6000)
            return
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

    def _montage_level_key(self, document, view_state, all_indices, colormap_lut):
        scope_state = view_state.with_montage_axis(
            view_state.montage_axis,
            columns=view_state.montage_columns,
            indices=None,
            text=None,
        )
        return (
            "montage_levels",
            _document_key(document),
            scope_state,
            int(view_state.montage_axis),
            None if colormap_lut is None else colormap_lut.tobytes(),
        )

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
        self._montage_level_tracker().ensure(level_key, expected_indices)
        self._montage_level_tracker().update_from_tile(
            level_key,
            int(rendered.tile.source_index),
            rendered.histogram_data,
            rendered.image,
            refined=bool(refined),
        )

    def _montage_level_stats_for_session(self, session) -> MontageLevelStats:
        expected = tuple(tile.source_index for tile in session.plan.tiles)
        return self._ensure_montage_level_stats(session.level_key, expected_indices=expected)

    def _montage_level_bounds_for_session(self, session, *, allow_partial: bool = False):
        source = self._montage_level_source_for_session(session, allow_partial=allow_partial)
        return None if source is None else source.histogram_range

    def _montage_level_source_for_session(self, session, *, allow_partial: bool = False):
        # Partial semantic tile coverage is a valid provisional level source.
        # It must not be confused with viewport pixels; the level key is semantic
        # and excludes zoom/pan.  WindowLevelController keeps updates monotonic.
        tracker = self._montage_level_tracker()
        stats = tracker.stats_for(session.level_key)
        return None if stats is None else tracker.source_for_stats(session.level_key, stats)

    def _montage_histogram_plot_data_for_session(self, session):
        tracker = self._montage_level_tracker()
        stats = tracker.stats_for(session.level_key)
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

    def _process_montage_cached_level_stats(self) -> None:
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session.session_id, session.key):
            return
        pending = getattr(session, "pending_level_tiles", None)
        if not pending:
            return
        stats_start = perf_counter()
        expected = tuple(tile.source_index for tile in session.plan.tiles)
        processed = 0
        while pending and processed < 4:
            rendered = pending.pop(0)
            self._update_montage_level_bounds_from_rendered(session.level_key, rendered, expected_indices=expected)
            processed += 1
            if processed >= 1 and (perf_counter() - stats_start) * 1000.0 >= 4.0:
                break
        self._last_montage_level_stats_ms = (perf_counter() - stats_start) * 1000.0
        self._montage_pending_level_tiles_last_session = len(pending)
        if session.display_committed and not pending:
            self._schedule_montage_canvas_commit(session, force=True)
        self._schedule_montage_cached_level_stats(session)

    def _plan_montage_stages(self, document, missing_tiles):
        document_key = stage_document_key(document)
        groups = {}
        tile_candidates = {}
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

        tile_stage_keys = {}
        stage_waiting_tiles = {}
        stage_values = {}
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
                continue
            if result.decision == "scheduled":
                _direct_tiles, waiting_tiles = _lead_direct_tiles(tiles)
                lead_direct_tile_count += len(_direct_tiles)
                if waiting_tiles:
                    stage_waiting_tiles[key] = list(waiting_tiles)
                for tile in waiting_tiles:
                    tile_stage_keys[int(tile.montage_index)] = key
                    waiting_indices.add(int(tile.montage_index))
                if _direct_tiles and _stage_fits_cache(candidate, self._memory_policy()):
                    self.operation_evaluator.stage_materializer.cancel(key)
                else:
                    stage_requests.append((result.request, group["plan"]))
                continue
            if result.decision == "attached":
                stage_waiting_tiles[key] = list(tiles)
                attached_stage_keys.add(key)
                for tile in tiles:
                    tile_stage_keys[int(tile.montage_index)] = key
                    waiting_indices.add(int(tile.montage_index))
                continue
            for tile in tiles:
                tile_stage_keys.pop(int(tile.montage_index), None)
            if len(tiles) > 1:
                repeated_expensive_stage_per_tile = True
        return {
            "tile_stage_keys": tile_stage_keys,
            "stage_waiting_tiles": stage_waiting_tiles,
            "stage_values": stage_values,
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
        self._activate_montage_stage_value(session, key, value)
        self._schedule_montage_tiles(session)

    def _activate_montage_stage_value(self, session, key, value) -> None:
        session.active_stage_requests.discard(key)
        session.attached_stage_requests.discard(key)
        session.stage_values[key] = value
        waiting = list(session.stage_waiting_tiles.pop(key, ()))
        for tile in waiting:
            index = int(tile.montage_index)
            if index not in session.rendered_tiles and index not in session.skipped_tiles:
                session.pending_tiles.append(tile)
                session.mark_loading(tile)

    def _schedule_montage_attached_stage_waits(self, session) -> None:
        if not getattr(session, "attached_stage_requests", None):
            return
        timer = getattr(self, "_montage_attached_stage_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._process_montage_attached_stage_waits)
            self._montage_attached_stage_timer = timer
        if not timer.isActive():
            timer.start(25)

    def _process_montage_attached_stage_waits(self) -> None:
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session.session_id, session.key):
            return
        pending_keys = tuple(getattr(session, "attached_stage_requests", ()))
        if not pending_keys:
            return
        wait_start = perf_counter()
        for key in pending_keys:
            self._activate_or_release_waiting_stage(session, key, release_missing=True)
        self._last_montage_stage_attach_wait_ms = (perf_counter() - wait_start) * 1000.0
        if session.pending_tiles:
            self._schedule_montage_tiles(session)
        if getattr(session, "attached_stage_requests", None):
            self._schedule_montage_attached_stage_waits(session)

    def _release_stage_waiting_tiles_to_direct(self, session, key) -> None:
        session.active_stage_requests.discard(key)
        session.attached_stage_requests.discard(key)
        waiting = list(session.stage_waiting_tiles.pop(key, ()))
        for tile in waiting:
            index = int(tile.montage_index)
            session.tile_stage_keys.pop(index, None)
            if index not in session.rendered_tiles and index not in session.skipped_tiles:
                session.pending_tiles.append(tile)
                session.mark_loading(tile)

    def _activate_cached_waiting_stages(self, session, *, release_missing: bool = False) -> None:
        for key in tuple(getattr(session, "stage_waiting_tiles", {})):
            self._activate_or_release_waiting_stage(session, key, release_missing=release_missing)

    def _activate_or_release_waiting_stage(self, session, key, *, release_missing: bool) -> None:
        cache = self.operation_evaluator.stage_cache
        value = cache.get_containing(key) if hasattr(cache, "get_containing") else cache.get(key)
        if value is not None:
            self.operation_evaluator.stage_materializer.cancel(key)
            self._activate_montage_stage_value(session, key, value)
            return
        in_flight = getattr(self.operation_evaluator.stage_materializer, "_in_flight", {})
        if release_missing and key not in in_flight:
            self._release_stage_waiting_tiles_to_direct(session, key)

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
            if not self._schedule_next_montage_tile(session):
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

        controller = getattr(self, "montage_tile_evaluation_controller", self.visible_evaluation_controller)
        controller.start_latest(
            evaluate,
            key=("montage_tile", session.key, int(tile.montage_index)),
            priority=EvalPriority.VISIBLE_IMAGE,
            replace_group=f"montage-tile:{int(session.session_id)}:{int(tile.montage_index)}",
            on_done=lambda result, session_id=session.session_id, tile=tile: self._on_montage_tile_done(session_id, tile, result),
            on_error=lambda exc, session_id=session.session_id, tile=tile: self._on_montage_tile_error(session_id, tile, exc),
            on_stale=lambda: None,
            on_slow=lambda: self._on_montage_tile_slow(session.session_id),
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
                    maker = make_shader_image_from_slab if bool(getattr(session, "shader_display", False)) else make_image_from_slab
                    display_image = maker(slab, request, colormap_lut=session.colormap_lut)
                    return EvaluationResult(
                        value=display_image,
                        eval_ms=(perf_counter() - start) * 1000.0,
                        slab_shape=tuple(np.shape(slab)),
                        slab_nbytes=int(getattr(slab, "nbytes", 0)),
                        region_plan=plan.region_plan,
                        compute_path="stage_backed",
                    )
            return evaluate_image_snapshot(
                session.document,
                tile.view_state,
                colormap_lut=session.colormap_lut,
                cancellation_token=token,
                shader_display=bool(getattr(session, "shader_display", False)),
                stage_cache=self.operation_evaluator.stage_cache,
                stage_document_key=stage_document_key(session.document),
                evaluation_context=context,
            )
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
        if getattr(session, "defer_side_panels", False):
            self._deferred_side_panel_refresh_pending = True
        else:
            self._update_operation_dock()

    def _on_montage_tile_done(self, session_id, tile, result) -> None:
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session_id, session.key):
            return
        if not self._is_current_render_generation(session.render_generation):
            return
        session.pending_completed_tiles.append((tile, result))
        self._schedule_montage_tile_result_flush(session)

    def _schedule_montage_tile_result_flush(self, session) -> None:
        if not self._is_current_montage_session(session.session_id, session.key):
            return
        timer = getattr(self, "_montage_tile_result_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._flush_montage_tile_results)
            self._montage_tile_result_timer = timer
        self._montage_tile_result_key = (int(session.session_id), session.key)
        if not timer.isActive():
            timer.start(0)

    def _flush_montage_tile_results(self) -> None:
        key = getattr(self, "_montage_tile_result_key", None)
        session = getattr(self, "_montage_session", None)
        if session is None or key is None or not self._is_current_montage_session(key[0], key[1]):
            return
        feedback = _latency_feedback(self)
        interactive = _interactive_active(self)
        max_batch = _montage_tile_result_batch_limit(self, interactive=interactive)
        budget_ms = _montage_tile_result_budget_ms(self, interactive=interactive)
        flush_start = perf_counter()
        processed = 0
        while session.pending_completed_tiles and processed < max_batch:
            tile, result = session.pending_completed_tiles.pop(0)
            self._apply_montage_tile_result(session, tile, result)
            processed += 1
            if processed >= 1 and (perf_counter() - flush_start) * 1000.0 >= budget_ms:
                break
        if processed:
            elapsed_ms = (perf_counter() - flush_start) * 1000.0
            self._last_montage_tile_result_flush_ms = elapsed_ms
            self._last_montage_tile_result_flush_count = int(processed)
            if feedback is not None:
                if hasattr(self, "_record_ui_work"):
                    self._record_ui_work("montage_tile_result", elapsed_ms, count=processed)
                else:
                    feedback.observe("montage_tile_result", elapsed_ms, count=processed)
            self._activate_cached_waiting_stages(session, release_missing=True)
            if session.pending_tiles:
                self._schedule_montage_tiles(session)
            force = not session.pending_tiles and not session.active_tile_requests and not session.pending_completed_tiles
            self._schedule_montage_canvas_commit(session, force=force)
        if session.pending_completed_tiles:
            self._schedule_montage_tile_result_flush(session)

    def _apply_montage_tile_result(self, session, tile, result) -> None:
        if not self._is_current_montage_session(session.session_id, session.key):
            return
        if not self._is_current_render_generation(session.render_generation):
            return
        rendered = self.operation_evaluator.store_montage_tile_result(
            tile,
            montage_axis=session.montage_axis,
            colormap_lut=session.colormap_lut,
            result=result,
            shader_display=bool(getattr(session, "shader_display", False)),
        )
        compute_path = str(getattr(result, "compute_path", "direct") or "direct")
        if compute_path == "stage_backed":
            session.tile_compute_stage_backed += 1
        else:
            session.tile_compute_direct += 1
        self._update_montage_level_bounds_from_rendered(
            session.level_key,
            rendered,
            expected_indices=tuple(tile.source_index for tile in session.plan.tiles),
        )
        session.mark_loaded(rendered)
        patch_start = perf_counter()
        if session.has_canvas():
            session.patch_rendered_tile(rendered)
            self._last_montage_canvas_patch_ms = (perf_counter() - patch_start) * 1000.0
        else:
            session.dirty_tiles.append(int(tile.montage_index))
            self._last_montage_canvas_patch_ms = 0.0

    def _on_montage_tile_error(self, session_id, tile, exc) -> None:
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session_id, session.key):
            return
        if not self._is_current_render_generation(session.render_generation):
            return
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
        if needs_initial_commit or force and not session.flush_pending or elapsed_ms >= interval_ms:
            self._commit_montage_session_canvas(session, force=force)
            return
        session.final_commit_pending = True
        session.flush_pending = True
        self._montage_coalesced_commits = int(getattr(self, "_montage_coalesced_commits", 0) or 0) + 1
        timer = getattr(self, "_montage_commit_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._flush_montage_canvas_commit)
            self._montage_commit_timer = timer
        if not timer.isActive():
            timer.start(max(1, int(interval_ms - elapsed_ms)))

    def _montage_commit_interval_ms(self, session, *, force: bool) -> int:
        if force:
            if not session.pending_tiles and not session.loading_tiles and not session.active_tile_requests:
                return _montage_commit_interval_ms(self, force=True)
        return _montage_commit_interval_ms(self, force=False)

    def _flush_montage_canvas_commit(self):
        session = getattr(self, "_montage_session", None)
        if session is None or not session.final_commit_pending:
            return
        self._commit_montage_session_canvas(session, force=False)

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
        if not session.rendered_tiles:
            self._last_montage_canvas_commit_ms = (perf_counter() - commit_start) * 1000.0
            return
        self._next_viewport_policy = ViewportPolicy.PRESERVE
        self._montage_canvas_commit_active = True
        try:
            display_image = DisplayImage(data=canvas.data, histogram_data=canvas.histogram_data)
            level_stats = self._montage_level_stats_for_session(session)
            complete = session.is_complete()
            histogram_plot_data = self._montage_histogram_plot_data_for_session(session)
            explicit_auto = bool(getattr(session, "force_auto", False))
            semantic_source = self._montage_level_source_for_session(session, allow_partial=explicit_auto)
            first_display_commit = not bool(session.display_committed)
            if newly_composed or first_display_commit:
                self._apply_full_display_image(
                    display_image,
                    geometry=rendered_geometry,
                    window_mode=session.window_mode,
                    previous_frame=getattr(self, "_committed_display_frame", None),
                    force_auto=explicit_auto,
                    defer_side_panels=getattr(session, "defer_side_panels", False),
                    semantic_source=semantic_source,
                    applied_level_source=session.applied_level_source,
                    histogram_plot_data=histogram_plot_data,
                    commit_kind=CommitKind.EXPLICIT_AUTO_WINDOW if explicit_auto else CommitKind.FULL_MONTAGE_INITIAL,
                    document_key=_document_key(session.document),
                    request_key=session.key,
                    render_generation=session.render_generation,
                    montage_level_key=session.level_key,
                    montage_dirty_tiles=dirty_tiles,
                    montage_tile_source_ids=tile_source_ids,
                )
                session.display_committed = True
            else:
                implicit_level_update = self._should_autolevel_progressive_montage(session, level_stats, complete=complete)
                semantic_source = self._montage_level_source_for_session(session, allow_partial=implicit_level_update)
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
                )
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
        if not session.rendered_tiles:
            self._last_montage_canvas_commit_ms = (perf_counter() - commit_start) * 1000.0
            return
        self._next_viewport_policy = ViewportPolicy.PRESERVE
        self._montage_canvas_commit_active = True
        try:
            level_stats = self._montage_level_stats_for_session(session)
            complete = session.is_complete()
            histogram_plot_data = self._montage_histogram_plot_data_for_session(session)
            explicit_auto = bool(getattr(session, "force_auto", False))
            semantic_source = self._montage_level_source_for_session(session, allow_partial=explicit_auto)
            first_display_commit = not bool(session.display_committed)
            payload_start = perf_counter()
            previous_payloads = _previous_tiled_payloads(getattr(self, "_committed_display_frame", None))
            if previous_payloads:
                session.seed_display_tile_payloads(previous_payloads, tile_source_ids)
            tile_state, tile_delta = session.build_tile_presentation(
                tile_source_ids,
                source_ids_trusted=bool(getattr(session, "tile_source_ids_trusted", True)),
            )
            self._montage_recent_tile_payloads_by_base_source = _limited_payload_cache(
                getattr(self, "_montage_recent_tile_payloads_by_base_source", None),
                tile_state.payloads,
            )
            self._last_montage_tile_payload_build_ms = (perf_counter() - payload_start) * 1000.0
            if first_display_commit:
                self._apply_full_display_image(
                    display_image,
                    geometry=rendered_geometry,
                    window_mode=session.window_mode,
                    previous_frame=getattr(self, "_committed_display_frame", None),
                    force_auto=explicit_auto,
                    defer_side_panels=getattr(session, "defer_side_panels", False),
                    semantic_source=semantic_source,
                    applied_level_source=session.applied_level_source,
                    histogram_plot_data=histogram_plot_data,
                    commit_kind=CommitKind.EXPLICIT_AUTO_WINDOW if explicit_auto else CommitKind.FULL_MONTAGE_INITIAL,
                    document_key=_document_key(session.document),
                    request_key=session.key,
                    render_generation=session.render_generation,
                    montage_level_key=session.level_key,
                    tile_state=tile_state,
                    tile_delta=tile_delta,
                )
                session.display_committed = True
            else:
                implicit_level_update = self._should_autolevel_progressive_montage(session, level_stats, complete=complete)
                semantic_source = self._montage_level_source_for_session(session, allow_partial=implicit_level_update)
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
                    tile_delta=tile_delta,
                )
            overlay_start = perf_counter()
            rect = montage_rect_for_viewport(session.plan, view_range=session.view_range, viewport_shape=session.viewport_shape)
            self._update_montage_tile_overlays_for_plan(session.plan, tuple(session.tile_states), rect)
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
        return True

    def _montage_tile_source_ids(self, session) -> dict[int, object]:
        source_ids = getattr(session, "tile_source_ids", None)
        if source_ids is None:
            source_ids = {}
            session.tile_source_ids = source_ids
        trusted = True
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
                trusted = False
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
        session.tile_source_ids_trusted = bool(trusted)
        return dict(source_ids)

    def _should_autolevel_progressive_montage(self, session, stats: MontageLevelStats, *, complete: bool) -> bool:
        # Automatic montage levels are semantic and monotonic: when new tiles for
        # the same montage source arrive, the available source may improve/expand
        # levels.  It must not wait for full completion, and it must not depend on
        # the current zoomed viewport.
        if session.window_mode == "absolute" or not session.rendered_tiles:
            return False
        bounds = stats.bounds
        if bounds is None:
            return False
        low, high = bounds
        if float(high) <= float(low):
            return False
        applied = getattr(session, "applied_level_source", None)
        applied_rank = int(getattr(applied, "rank", 0) or 0)
        applied_count = int(getattr(applied, "source_count", 0) or 0)
        if int(stats.rank) < applied_rank:
            return False
        if int(stats.rank) == applied_rank and len(stats.source_indices) <= applied_count:
            return False
        if not complete and not _should_apply_progressive_level_milestone(
            len(stats.source_indices),
            applied_count,
            len(stats.expected_indices),
        ):
            return False
        return True

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
        pending = {int(tile.montage_index) for tile in session.pending_tiles}
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
                session.pending_tiles.append(tile)
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
        overlays = []
        canvas_x0, canvas_y0, canvas_x1, canvas_y1 = (int(value) for value in canvas_rect)
        for tile in plan.tiles:
            state = tile_states[int(tile.montage_index)] if int(tile.montage_index) < len(tile_states) else MontageTileState.UNLOADED
            if state == MontageTileState.LOADING and not bool(getattr(getattr(self, "_montage_session", None), "show_loading_overlays", False)):
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

    def _schedule_montage_viewport_update(self) -> None:
        if getattr(self, "_montage_viewport_update_running", False):
            self._montage_viewport_update_pending = True
            return
        timer = getattr(self, "_montage_viewport_update_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._run_montage_viewport_update)
            self._montage_viewport_update_timer = timer
        timer.start(_montage_viewport_update_delay_ms(self))

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
            self.update_montage_view()
        finally:
            self._montage_viewport_update_running = False
        if getattr(self, "_montage_viewport_update_pending", False) and self.view_state.montage_axis is not None:
            self._montage_viewport_update_pending = False
            self._schedule_montage_viewport_update()

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


def _stage_fits_cache(candidate, memory_policy) -> bool:
    estimated = getattr(candidate, "estimated_nbytes", None)
    if estimated is None:
        return False
    return int(estimated) <= int(getattr(memory_policy, "stage_cache_budget_bytes", 0) or 0)


def _should_apply_progressive_level_milestone(source_count: int, applied_count: int, expected_count: int) -> bool:
    source_count = max(0, int(source_count))
    applied_count = max(0, int(applied_count))
    expected_count = max(0, int(expected_count))
    if source_count <= applied_count:
        return False
    if expected_count and source_count >= expected_count:
        return True
    first = min(4, expected_count) if expected_count else 4
    if applied_count <= 0:
        return source_count >= first
    return source_count >= max(applied_count + 8, applied_count * 2)


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


def _montage_tile_result_budget_ms(window, *, interactive: bool) -> float:
    decision = getattr(window, "_ui_work_decision", lambda *args, **kwargs: None)("montage_tile_result", interactive=interactive)
    if decision is not None:
        return max(1.0, float(decision.budget_ms))
    feedback = _latency_feedback(window)
    if feedback is None:
        return 4.0 if interactive else 8.0
    return float(feedback.work_budget_ms("montage_tile_result", interactive=interactive))


def _previous_tiled_payloads(frame) -> dict[int, object]:
    source = None if frame is None else getattr(frame, "value_source", None)
    payloads = getattr(source, "payloads", None)
    return {} if payloads is None else dict(payloads)


def _previous_tiled_payloads_by_base_source(frame) -> dict[object, object]:
    return {
        _base_tile_source_id(payload.source_id): payload
        for payload in _previous_tiled_payloads(frame).values()
        if _base_tile_source_id(payload.source_id) is not None
    }


def _limited_payload_cache(existing, payloads, *, limit: int = 4096) -> dict[object, object]:
    cache = dict(existing or {})
    for payload in dict(payloads or {}).values():
        key = _base_tile_source_id(payload.source_id)
        if key is not None:
            cache[key] = payload
    if len(cache) <= int(limit):
        return cache
    items = tuple(cache.items())[-int(limit) :]
    return dict(items)


def _base_tile_source_id(source_id) -> object | None:
    if isinstance(source_id, tuple) and len(source_id) >= 3 and source_id[1] == "texture_kind":
        return source_id[0]
    return source_id


def _payload_lod_matches(payload, factor: int) -> bool:
    lod = getattr(payload, "lod", None)
    payload_factor = int(getattr(lod, "factor", 1) or 1)
    return payload_factor == max(1, int(factor))


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
    )


def _montage_viewport_update_delay_ms(window) -> int:
    try:
        capabilities = image_view_backend_capabilities(window.img_view)
    except Exception:
        capabilities = None
    mode = ""
    try:
        mode = str(window.img_view.montageDisplayMode())
    except Exception:
        mode = ""
    if bool(getattr(capabilities, "persistent_tile_residency", False)) and "tile_layer" in mode:
        return 16
    return 120


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


def _interactive_active(window) -> bool:
    coordinator = getattr(window, "render_coordinator", None)
    return bool(coordinator is not None and getattr(coordinator, "interactive_active", False))
