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
from arrayscope.display.montage import (
    MontageTileState,
    make_montage_plan,
    make_montage_viewport_canvas,
    montage_rect_for_viewport,
    optimal_montage_columns,
)
from arrayscope.display.slice_engine import DisplayImage, make_image_from_slab
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.operations.evaluator import EvaluationResult, _document_key, evaluate_image_snapshot, stage_document_key
from arrayscope.operations.chunked_stage import materialize_stage_candidate_chunked
from arrayscope.operations.slabs import (
    evaluate_slab_from_stage,
    plan_slab,
    request_for_image,
)
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.evaluation_controller import EvalPriority
from arrayscope.window.montage_levels import MontageLevelStats, MontageLevelTracker
from arrayscope.window.montage_prefetch import schedule_near_viewport_montage_prefetch
from arrayscope.window.montage_session import MontageRenderSession
from arrayscope.window.presentation import LevelSourceRank, fallback_level_source
from arrayscope.window.render_model import CommitKind


MONTAGE_COMMIT_INTERVAL_MS = 16
MONTAGE_SLOW_UPLOAD_MS = 50.0
MONTAGE_VERY_SLOW_UPLOAD_MS = 100.0
MONTAGE_EXTREME_UPLOAD_MS = 250.0


class MontageRenderMixin:
    def _montage_tile_layer_policy(self, geometry, data) -> bool:
        if getattr(geometry, "montage", None) is None:
            return False
        pixels = int(np.prod(np.shape(data)[:2]))
        if pixels > 2_000_000:
            return True
        if float(getattr(self, "_last_set_image_ms", 0.0) or 0.0) > MONTAGE_VERY_SLOW_UPLOAD_MS:
            return True
        if int(getattr(self, "_montage_patched_tiles_last_flush", 0) or 0) > 8:
            return True
        mode = getattr(self.img_view, "montageDisplayMode", lambda: "canvas")()
        return str(mode) == "tile_layer"

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
        output_dtype = np.uint8 if view_state.channel == ChannelMode.COMPLEX else getattr(document.base_data, "dtype", np.dtype(float))
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
        for tile in visible_tiles:
            tile_cache_start = perf_counter()
            cached = self.operation_evaluator.cached_montage_tile(
                tile.view_state,
                montage_axis=axis,
                source_index=tile.source_index,
                colormap_lut=colormap_lut,
            )
            self._last_montage_tile_cache_lookup_ms = (perf_counter() - tile_cache_start) * 1000.0
            self._last_montage_tile_cache_hit = cached is not None
            if cached is None:
                missing_tiles.append(tile)
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
        )
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
        if not session.pending_tiles and not session.loading_tiles and not session.attached_stage_requests:
            self._stop_montage_session_slow_overlay()
            self.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.READY, "Montage view ready")
            self.img_view.setImageStale(False)
            self.img_view.setEvaluationOverlay(False)
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
        for _ in range(min(8, len(pending))):
            rendered = pending.pop(0)
            self._update_montage_level_bounds_from_rendered(session.level_key, rendered, expected_indices=expected)
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
        for key, group in groups.items():
            tiles = tuple(group["tiles"])
            candidate = group["candidate"]
            estimated = int(getattr(candidate, "estimated_nbytes", 0) or 0)
            if len(tiles) < 2 and estimated < 16 * 1024 * 1024:
                continue
            result = self.operation_evaluator.stage_materializer.request_stage(document_key, candidate)
            if result.decision == "hit":
                stage_values[key] = result.value
                for tile in tiles:
                    tile_stage_keys[int(tile.montage_index)] = key
                continue
            if result.decision == "scheduled":
                stage_waiting_tiles[key] = list(tiles)
                for tile in tiles:
                    tile_stage_keys[int(tile.montage_index)] = key
                    waiting_indices.add(int(tile.montage_index))
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
        return {
            "tile_stage_keys": tile_stage_keys,
            "stage_waiting_tiles": stage_waiting_tiles,
            "stage_values": stage_values,
            "stage_requests": stage_requests,
            "attached_stage_keys": attached_stage_keys,
            "waiting_indices": waiting_indices,
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
                    allowed_chunk_axes=_montage_stage_chunk_axes(session.view_state),
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
        cache = self.operation_evaluator.stage_cache
        for key in pending_keys:
            value = cache.get_containing(key) if hasattr(cache, "get_containing") else cache.get(key)
            if value is not None:
                self._activate_montage_stage_value(session, key, value)
        self._last_montage_stage_attach_wait_ms = (perf_counter() - wait_start) * 1000.0
        if session.pending_tiles:
            self._schedule_montage_tiles(session)
        if getattr(session, "attached_stage_requests", None):
            self._schedule_montage_attached_stage_waits(session)

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
            self._schedule_montage_canvas_commit(session, force=True)
            if session.active_stage_requests or session.stage_waiting_tiles:
                return False
            if session.pending_tiles:
                return self._schedule_next_montage_tile(session)
            if self._is_current_montage_session(session.session_id, session.key):
                self._stop_montage_session_slow_overlay()
                self.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.READY, "Montage view ready")
                self.img_view.setImageStale(False)
                self.img_view.setEvaluationOverlay(False)
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
                    display_image = make_image_from_slab(slab, request, colormap_lut=session.colormap_lut)
                    return EvaluationResult(
                        value=display_image,
                        eval_ms=(perf_counter() - start) * 1000.0,
                        slab_shape=tuple(np.shape(slab)),
                        slab_nbytes=int(getattr(slab, "nbytes", 0)),
                        region_plan=plan.region_plan,
                    )
            return evaluate_image_snapshot(
                session.document,
                tile.view_state,
                colormap_lut=session.colormap_lut,
                cancellation_token=token,
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
        rendered = self.operation_evaluator.store_montage_tile_result(
            tile,
            montage_axis=session.montage_axis,
            colormap_lut=session.colormap_lut,
            result=result,
        )
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
        self._schedule_montage_canvas_commit(session, force=not session.pending_tiles)
        self._schedule_montage_tiles(session)

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
        if session.canvas is None or force and not session.flush_pending or elapsed_ms >= interval_ms:
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
                return MONTAGE_COMMIT_INTERVAL_MS
        upload_ms = float(getattr(self, "_last_set_image_ms", 0.0) or 0.0)
        if upload_ms > MONTAGE_EXTREME_UPLOAD_MS:
            return 250
        if upload_ms > MONTAGE_VERY_SLOW_UPLOAD_MS:
            return 200
        if upload_ms > MONTAGE_SLOW_UPLOAD_MS:
            return 100
        return MONTAGE_COMMIT_INTERVAL_MS

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
            complete = (
                not session.pending_tiles
                and not session.loading_tiles
                and not session.active_stage_requests
                and not session.attached_stage_requests
                and not session.stage_waiting_tiles
            )
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
        session.note_committed()
        schedule_near_viewport_montage_prefetch(self, session)
        self._retry_live_profile_after_montage_tile()

    def _montage_tile_source_ids(self, session) -> dict[int, object]:
        source_ids = {}
        for tile_number, rendered in getattr(session, "rendered_tiles", {}).items():
            try:
                source_ids[int(tile_number)] = self.operation_evaluator.montage_tile_key(
                    rendered.tile.view_state,
                    montage_axis=session.montage_axis,
                    source_index=rendered.tile.source_index,
                    colormap_lut=session.colormap_lut,
                    document=session.document,
                )
            except Exception:
                image = getattr(rendered, "image", None)
                histogram = getattr(rendered, "histogram_data", None)
                source_ids[int(tile_number)] = (
                    id(image),
                    tuple(np.shape(image)),
                    None if image is None else str(np.asarray(image).dtype),
                    id(histogram),
                    None if histogram is None else tuple(np.shape(histogram)),
                    None if histogram is None else str(np.asarray(histogram).dtype),
                )
        return source_ids

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
        if not hasattr(self.img_view, "setMontageTileOverlays"):
            return
        overlays = []
        for tile in canvas.full_plan.tiles:
            state = canvas.tile_states[int(tile.montage_index)] if int(tile.montage_index) < len(canvas.tile_states) else MontageTileState.UNLOADED
            if state == MontageTileState.LOADING and not bool(getattr(getattr(self, "_montage_session", None), "show_loading_overlays", False)):
                continue
            if state not in {MontageTileState.LOADING, MontageTileState.SKIPPED}:
                continue
            canvas_x0 = int(canvas.origin_x)
            canvas_y0 = int(canvas.origin_y)
            canvas_x1 = canvas_x0 + int(canvas.display_shape[1])
            canvas_y1 = canvas_y0 + int(canvas.display_shape[0])
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
        timer = getattr(self, "_montage_viewport_update_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._run_montage_viewport_update)
            self._montage_viewport_update_timer = timer
        timer.start(60)

    def _run_montage_viewport_update(self) -> None:
        if getattr(self, "_closing", False):
            return
        if self.view_state.montage_axis is None:
            return
        self.update_montage_view()

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


def _montage_stage_chunk_axes(view_state) -> tuple[int, ...]:
    image_axes = set(() if view_state.image_axes is None else tuple(int(axis) for axis in view_state.image_axes))
    return tuple(axis for axis in range(len(tuple(view_state.shape))) if axis not in image_axes)
