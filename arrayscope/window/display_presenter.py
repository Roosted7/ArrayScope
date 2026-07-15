"""Display presentation commit mixin for ArrayScope windows.

This isolates the semantic presentation/Qt commit boundary from the large render
orchestrator.  Render code should build display payloads; this mixin decides and
applies presentation through DisplayCommitter.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from arrayscope.app.errors import handle_ui_exception
from arrayscope.core.memory_policy import MiB, MemoryPolicy
from arrayscope.core.frame_targets import FrameTarget
from arrayscope.kernel import Lane as WorkLane, WorkItem, complete_inline_work as _complete_inline_work
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.frame_planner import FramePlanner
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.operations.evaluator import _document_key
from arrayscope.ui.toasts import show_status_message
from arrayscope.display.commit import DisplayCommitter
from arrayscope.display.model.frame import (
    CommittedDisplayFrame,
    DisplayFrameKey,
    DisplayTilePayload,
    TilePresentationDelta,
    TilePresentationState,
    TiledValueSource,
)
from arrayscope.display.planning import LevelSource, LevelSourceRank, decide_presentation, normalize_bounds
from arrayscope.display.model.commit import CommitKind, DisplayPayload, DisplayTiledPresentation, PresentationInput, RenderRequestContext
from arrayscope.window.viewport_bridge import ViewportBridge


class DisplayPresentationMixin:
    def _apply_full_display_image(
        self,
        display_image,
        *,
        geometry,
        window_mode,
        previous_frame,
        force_auto,
        defer_side_panels: bool = False,
        level_bounds=None,
        semantic_source=None,
        applied_level_source=None,
        histogram_plot_data=None,
        commit_kind=None,
        document_key=None,
        request_key=None,
        render_generation=None,
        montage_level_key=None,
        montage_dirty_tiles=None,
        montage_tile_source_ids=None,
        tile_state=None,
        base_tile_state=None,
        tile_delta=None,
        frame_plan=None,
        user_levels=None,
        semantic_commit: bool = True,
    ):
        commit_start = perf_counter()
        try:
            viewport_policy = self._viewport_policy_for_display_shape(display_image.data.shape[:2])
            levels_start = perf_counter()
            if commit_kind is None:
                commit_kind = CommitKind.FULL_FRAME_INITIAL
            context = self._render_request_context(
                document_key=document_key,
                request_key=request_key,
                render_generation=render_generation,
                semantic_key=montage_level_key,
            )
            # One commit path: the frame session owns the plan and the tile
            # presentation. A commit without them is a programming error, not
            # a case to paper over with a divergent secondary plan (the
            # 2026-07-15 window-shift diagnosis found exactly such dead
            # fallbacks masking that the live flow never used them).
            if frame_plan is None or tile_state is None:
                raise ValueError("display commits require the session's frame_plan and tile_state")
            decision = decide_presentation(
                PresentationInput(
                    payload=DisplayPayload(
                        image=display_image,
                        geometry=frame_plan.geometry,
                        viewport_policy=viewport_policy,
                        frame_plan=frame_plan,
                        rgb_already_windowed=bool(getattr(display_image, "rgb_already_windowed", False)),
                        histogram_plot_data=histogram_plot_data,
                        montage_dirty_tiles=montage_dirty_tiles,
                        montage_tile_source_ids=montage_tile_source_ids,
                        tile_state=tile_state,
                        base_tile_state=base_tile_state,
                        tile_delta=tile_delta,
                        tile_residency_budget_bytes=tile_residency_budget_bytes(self._memory_policy()),
                    ),
                    context=context,
                    previous_frame=previous_frame,
                    window_mode=window_mode,
                    force_auto=force_auto,
                    commit_kind=commit_kind,
                    semantic_source=semantic_source,
                    applied_level_source=applied_level_source,
                    level_bounds=normalize_bounds(level_bounds),
                    user_levels=normalize_bounds(user_levels),
                )
            )
            self._last_levels_histogram_ms = (perf_counter() - levels_start) * 1000.0

            set_image_start = perf_counter()
            frame = self._display_committer().commit_tile_layer(decision.display_presentation, context.frame_key)
            self._last_set_image_ms = (perf_counter() - set_image_start) * 1000.0
            self.display_geometry = frame.geometry
            report = getattr(self._display_committer(), "last_tile_commit_report", None)
            semantic_frame_commit = bool(
                semantic_commit
                and bool(getattr(report, "presented_tiles", ()))
            )
            if semantic_frame_commit:
                self._set_committed_display_frame(frame)
                self._consume_pending_display_levels(user_levels)
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
            elif bool(getattr(report, "presented_tiles", ())):
                self._note_display_level_source(decision)
            if defer_side_panels:
                self.win._deferred_side_panel_refresh_pending = True
            elif semantic_frame_commit:
                self.win._update_operation_dock()
        
            # Apply axis flips after setting the image
            self.win.apply_axis_flips()
            self.win.img_view.setImageStale(False)
            if semantic_frame_commit or getattr(geometry, "montage", None) is None:
                self.win.img_view.setEvaluationOverlay(False)
            if defer_side_panels:
                self.win._deferred_side_panel_refresh_pending = True
            elif semantic_frame_commit:
                self.win._refresh_inspection_dock()
        
        except Exception as e:
            handle_ui_exception("image update", e)
            show_status_message(self.win, f"Image update failed: {e}")
        finally:
            self._last_display_commit_ms = (perf_counter() - commit_start) * 1000.0

    def _apply_progressive_display_image(
        self,
        display_image,
        *,
        geometry,
        window_mode,
        previous_frame,
        force_auto,
        viewport_policy,
        level_bounds=None,
        semantic_source=None,
        applied_level_source=None,
        histogram_plot_data=None,
        commit_kind=CommitKind.PROGRESSIVE_FRAME_PATCH,
        document_key=None,
        request_key=None,
        render_generation=None,
        montage_level_key=None,
        montage_dirty_tiles=None,
        montage_tile_source_ids=None,
        tile_state=None,
        base_tile_state=None,
        tile_delta=None,
        frame_plan=None,
        user_levels=None,
        semantic_commit: bool = True,
    ):
        commit_start = perf_counter()
        try:
            levels_start = perf_counter()
            context = self._render_request_context(
                document_key=document_key,
                request_key=request_key,
                render_generation=render_generation,
                semantic_key=montage_level_key,
            )
            # One commit path: the frame session owns the plan and the tile
            # presentation. A commit without them is a programming error, not
            # a case to paper over with a divergent secondary plan (the
            # 2026-07-15 window-shift diagnosis found exactly such dead
            # fallbacks masking that the live flow never used them).
            if frame_plan is None or tile_state is None:
                raise ValueError("display commits require the session's frame_plan and tile_state")
            decision = decide_presentation(
                PresentationInput(
                    payload=DisplayPayload(
                        image=display_image,
                        geometry=frame_plan.geometry,
                        viewport_policy=viewport_policy,
                        frame_plan=frame_plan,
                        rgb_already_windowed=bool(getattr(display_image, "rgb_already_windowed", False)),
                        histogram_plot_data=histogram_plot_data,
                        montage_dirty_tiles=montage_dirty_tiles,
                        montage_tile_source_ids=montage_tile_source_ids,
                        tile_state=tile_state,
                        base_tile_state=base_tile_state,
                        tile_delta=tile_delta,
                        tile_residency_budget_bytes=tile_residency_budget_bytes(self._memory_policy()),
                    ),
                    context=context,
                    previous_frame=previous_frame,
                    window_mode=window_mode,
                    force_auto=force_auto,
                    commit_kind=commit_kind,
                    semantic_source=semantic_source,
                    applied_level_source=applied_level_source,
                    level_bounds=normalize_bounds(level_bounds),
                    user_levels=normalize_bounds(user_levels),
                )
            )
            self._last_levels_histogram_ms = (perf_counter() - levels_start) * 1000.0
            set_image_start = perf_counter()
            frame = self._display_committer().commit_tile_layer(decision.display_presentation, context.frame_key)
            self._last_set_image_ms = (perf_counter() - set_image_start) * 1000.0
            self.display_geometry = frame.geometry
            report = getattr(self._display_committer(), "last_tile_commit_report", None)
            semantic_frame_commit = bool(
                semantic_commit
                and bool(getattr(report, "presented_tiles", ()))
            )
            if semantic_frame_commit:
                self._set_committed_display_frame(frame)
                self._consume_pending_display_levels(user_levels)
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
            elif bool(getattr(report, "presented_tiles", ())):
                self._note_display_level_source(decision)
            self.win.apply_axis_flips()
            self.win.img_view.setImageStale(False)
        except Exception as e:
            handle_ui_exception("progressive image update", e)
            show_status_message(self.win, f"Image update failed: {e}")
        finally:
            self._last_progressive_commit_ms = (perf_counter() - commit_start) * 1000.0
            self._last_display_commit_ms = self._last_progressive_commit_ms

    def _should_use_montage_tile_layer_for_display(self, geometry, data) -> bool:
        policy = getattr(self, "_montage_tile_layer_policy", None)
        if policy is None:
            return False
        return bool(policy(geometry, data))

    def _display_committer(self) -> DisplayCommitter:
        committer = getattr(self, "_display_committer_instance", None)
        if committer is None or getattr(committer, "image_view", None) is not self.win.img_view:
            committer = DisplayCommitter(self.win.img_view)
            self._display_committer_instance = committer
        return committer

    def _frame_planner(self) -> FramePlanner:
        planner = getattr(self, "_frame_planner_instance", None)
        if planner is None:
            planner = FramePlanner()
            self._frame_planner_instance = planner
        return planner

    def _previous_display_frame_for_policy(self, *, force_auto: bool) -> CommittedDisplayFrame | None:
        if force_auto:
            return None
        frame = getattr(self.win, "_committed_display_frame", None)
        if frame is None:
            return None
        return frame if self._is_level_history_frame_usable(frame) else None

    def _is_level_history_frame_usable(self, frame: CommittedDisplayFrame | None) -> bool:
        if frame is None or getattr(self.win, "_closing", False):
            return False
        if frame.key.document_key != _document_key(self.win.document):
            return False
        if normalize_bounds(frame.levels) is None:
            return False
        if normalize_bounds(frame.histogram_range) is None:
            return False
        geometry = getattr(frame, "geometry", None)
        if geometry is None:
            return False
        try:
            display_shape = tuple(int(size) for size in geometry.display_shape)
        except Exception:
            return False
        if len(display_shape) != 2 or display_shape[0] < 1 or display_shape[1] < 1:
            return False
        if frame.data is None:
            if not isinstance(frame.value_source, TiledValueSource):
                return False
        elif tuple(np.shape(frame.data)[:2]) != display_shape:
            return False
        if frame.histogram_data is not None and tuple(np.shape(frame.histogram_data)[:2]) != display_shape:
            return False
        return True

    def _render_request_context(self, *, document_key=None, request_key=None, render_generation=None, semantic_key=None) -> RenderRequestContext:
        if document_key is None:
            document_key = _document_key(self.win.document)
        if request_key is None:
            request_key = ("display", document_key, self.win.view_state)
        if render_generation is None:
            render_generation = self._capture_render_generation()
        return RenderRequestContext(
            document_key=document_key,
            request_key=request_key,
            render_generation=int(render_generation),
            semantic_key=semantic_key,
        )

    def _note_display_level_source(self, decision) -> None:
        session = getattr(self, "_frame_session", None)
        if session is None:
            return
        source = getattr(decision, "applied_level_source", None)
        if source is None:
            return
        if getattr(source, "semantic_key", None) != getattr(session, "level_key", None):
            return
        session.applied_level_source = source

    def _viewport_bridge(self) -> ViewportBridge:
        bridge = getattr(self, "_viewport_bridge_instance", None)
        if bridge is None:
            bridge = ViewportBridge(self)
            self._viewport_bridge_instance = bridge
        return bridge

    def _schedule_frame_viewport_update(self, *, delay_ms: int | None = None) -> None:
        timer = getattr(self, "_frame_viewport_update_timer", None)
        if timer is None:
            from pyqtgraph.Qt import QtCore

            # Timer category: UI cosmetic. Qt event-turn barrier. Restarting the single timer coalesces
            # bursts to one retarget, and the callback re-derives everything
            # from the committed frame, so this is pure rescheduling — not an
            # ordering source for frame semantics.
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._run_frame_viewport_update)
            self._frame_viewport_update_timer = timer
        delay_ms = 0 if delay_ms is None else max(0, int(delay_ms))
        timer.start(delay_ms)

    def _schedule_interactive_montage_viewport_update(self) -> None:
        timer = getattr(self, "_interactive_montage_viewport_timer", None)
        if timer is None:
            from pyqtgraph.Qt import QtCore

            # Timer category: UI cosmetic. Camera motion is already visible;
            # this bounds only the derived LOD/visibility replan cadence.
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._run_interactive_montage_viewport_update)
            self._interactive_montage_viewport_timer = timer
        if timer.isActive():
            return
        timer.start(16)

    def _run_interactive_montage_viewport_update(self) -> None:
        scheduler = getattr(self, "retarget_montage_viewport", None)
        if callable(scheduler):
            scheduler()

    def _run_frame_viewport_update(self) -> None:
        if getattr(self.win.view_state, "montage_axis", None) is not None:
            scheduler = getattr(self, "retarget_montage_viewport", None)
            if callable(scheduler):
                scheduler()
            return
        frame = getattr(self.win, "_committed_display_frame", None)
        scene = getattr(frame, "scene", None)
        if frame is None or scene is None:
            return
        if getattr(getattr(scene, "layout", None), "value", getattr(scene, "layout", None)) != "single":
            return
        value_source = getattr(frame, "value_source", None)
        if not isinstance(value_source, TiledValueSource):
            return
        view_range = _current_view_range(self)
        context = self._render_request_context(
            document_key=frame.key.document_key,
            request_key=frame.key.request_key,
            render_generation=frame.key.render_generation,
            semantic_key=frame.key.semantic_key,
        )
        frame_plan = self._frame_planner().plan(
            target=FrameTarget(
                semantic_key=context.semantic_key or context.request_key,
                viewport_key=view_range,
                presentation_key=None,
                quality="exact-visible",
            ),
            view_state=frame.geometry.view_state,
            display_shape=frame.geometry.display_shape,
            backend_capabilities=image_view_backend_capabilities(self.win.img_view),
            view_range=view_range,
            memory_policy=self._memory_policy(),
        )
        payloads = dict(value_source.payloads)
        revision = max(0, int(getattr(getattr(scene, "tile_state", None), "revision", 0) or 0))
        state = TilePresentationState(payloads, revision=revision)
        delta = TilePresentationDelta(
            structure_revision=1,
            payload_revision=1,
            visibility_revision=1,
            level_revision=1,
            histogram_revision=1,
            viewport_revision=1,
            base_revision=revision,
            target_revision=revision,
            upserts={},
            active_tiles=frame_plan.active_region_ids,
            planned_tiles=frame_plan.planned_region_ids,
            near_tiles=frame_plan.near_region_ids,
        )
        presentation = DisplayTiledPresentation(
            geometry=frame_plan.geometry,
            levels=frame.levels,
            histogram_range=frame.histogram_range,
            viewport_policy=ViewportPolicy.PRESERVE,
            tile_state=state,
            base_tile_state=state,
            tile_delta=delta,
            tile_residency_budget_bytes=tile_residency_budget_bytes(self._memory_policy()),
            frame_plan=frame_plan,
        )
        committed = self._display_committer().commit_tile_layer(presentation, frame.key)
        self.display_geometry = committed.geometry
        self._set_committed_display_frame(committed)
        self.win.apply_axis_flips()
        _complete_inline_work(
            self,
            WorkItem(
                key=("frame_viewport_retarget", frame.key.request_key, view_range),
                lane=WorkLane.DISPLAY_PREPARATION,
                quality="retained",
                supersession_key=("frame-viewport-retarget", frame.key.request_key),
                supersession_value=view_range,
            ),
        )

    def _display_frame_key(self, *, document_key=None, request_key=None, render_generation=None, semantic_key=None) -> DisplayFrameKey:
        if document_key is None:
            document_key = _document_key(self.win.document)
        if request_key is None:
            request_key = ("display", document_key, self.win.view_state)
        if render_generation is None:
            render_generation = self._capture_render_generation()
        return DisplayFrameKey(
            document_key=document_key,
            request_key=request_key,
            render_generation=int(render_generation),
            semantic_key=semantic_key,
        )

    def _set_committed_display_frame(self, frame: CommittedDisplayFrame) -> None:
        if not getattr(frame, "is_tiled", False):
            raise RuntimeError("Committed display frames must be tiled.")
        self._committed_display_request_key = frame.key.request_key
        self.win._committed_display_frame = frame
        refresh_hidden_roi_overlay = getattr(self.win, "_refresh_hidden_roi_overlay_from_committed_frame", None)
        if callable(refresh_hidden_roi_overlay):
            refresh_hidden_roi_overlay()

    def _queue_display_levels(self, levels) -> tuple[float, float] | None:
        """Queue exact levels for the next successful semantic presentation.

        Recipe restoration can precede asynchronous evaluation.  Writing the
        histogram widget immediately races that evaluation and produces a brief
        old/new/automatic-level flash.  A queued override instead travels with
        the render request and is consumed only after a frame is committed.
        """

        normalized = normalize_bounds(levels)
        self._pending_display_levels = normalized
        return normalized

    def _pending_display_levels_for_render(self) -> tuple[float, float] | None:
        return normalize_bounds(getattr(self, "_pending_display_levels", None))

    def _consume_pending_display_levels(self, applied_levels) -> None:
        applied = normalize_bounds(applied_levels)
        pending = self._pending_display_levels_for_render()
        if applied is not None and pending == applied:
            self._pending_display_levels = None

    def _apply_display_level_override(
        self,
        levels,
        *,
        histogram_range=None,
        emit_user: bool = False,
        source_rank=LevelSourceRank.PREVIOUS_COMMITTED,
        semantic_key=None,
    ) -> LevelSource | None:
        """Apply a level override consistently to Qt, frame, and montage state.

        The source override is exposed while the widget applies the levels so
        tiled presentation callbacks can distinguish an automatic/programmatic
        transition from a histogram gesture.
        """

        levels = normalize_bounds(levels)
        if levels is None:
            return None
        frame = getattr(self.win, "_committed_display_frame", None)
        session = getattr(self, "_frame_session", None)
        if semantic_key is None and frame is not None:
            semantic_key = frame.key.semantic_key
        if semantic_key is None and session is not None:
            semantic_key = getattr(session, "level_key", None)
        histogram_range = (
            normalize_bounds(histogram_range)
            or normalize_bounds(self.win.img_view.getHistogramDataBounds())
            or levels
        )
        source = LevelSource(
            levels=levels,
            histogram_range=histogram_range,
            rank=source_rank,
            source_count=0,
            expected_count=0,
            semantic_key=semantic_key,
            mode=self._current_window_mode(),
        )

        previous_override = getattr(self, "_level_presentation_source_override", None)
        self._level_presentation_source_override = source
        try:
            apply_levels = getattr(self.win.img_view, "_apply_display_levels", None)
            if callable(apply_levels):
                apply_levels(levels[0], levels[1], emit_user=bool(emit_user))
            else:
                self.win.img_view.setLevels(levels[0], levels[1])
        finally:
            self._level_presentation_source_override = previous_override

        if source.rank != LevelSourceRank.EXPLICIT_USER:
            self._explicit_user_level_source = None
            if session is not None:
                session.user_levels_override = None

        frame = getattr(self.win, "_committed_display_frame", None)
        if frame is not None:
            self.win._committed_display_frame = replace(
                frame,
                levels=levels,
                histogram_range=histogram_range,
            )
        if session is not None:
            session.applied_level_source = source
        if emit_user:
            controller = getattr(self.win, "sync_controller", None)
            publish_now = getattr(controller, "_publish_now", None)
            if controller is not None and "levels" in getattr(controller, "_pending_requests", set()):
                controller._pending_requests.discard("levels")
                controller._ignore_join_state.add("levels")
            if not callable(publish_now) or not publish_now("levels", force=True):
                self.win._notify_sync("levels")
        return source

    def _on_display_levels_changed(self) -> None:
        try:
            levels = normalize_bounds(self.win.img_view.getLevels())
        except Exception:
            levels = None
        if levels is None:
            return
        try:
            histogram_range = normalize_bounds(self.win.img_view.getHistogramDataBounds())
        except Exception:
            histogram_range = None
        mode = self._current_window_mode()
        source = LevelSource(
            levels=levels,
            histogram_range=histogram_range or levels,
            rank=LevelSourceRank.EXPLICIT_USER if mode == "absolute" else LevelSourceRank.PREVIOUS_COMMITTED,
            source_count=0,
            expected_count=0,
            semantic_key=getattr(getattr(self, "_frame_session", None), "level_key", None),
            mode=mode,
        )
        self._explicit_user_level_source = source
        session = getattr(self, "_frame_session", None)
        if session is not None:
            session.applied_level_source = source

        frame = getattr(self.win, "_committed_display_frame", None)
        if frame is not None and self._is_level_history_frame_usable(frame):
            self.win._committed_display_frame = replace(frame, levels=levels, histogram_range=histogram_range or frame.histogram_range)
        self.win._notify_sync("levels")

    def _on_level_presentation_changed(self, levels, *, final: bool = False) -> bool:
        levels = normalize_bounds(levels)
        if levels is None:
            return False
        session = getattr(self, "_frame_session", None)
        if session is None or not bool(getattr(session, "display_committed", False)):
            return False
        if str(getattr(self.win.img_view, "montageDisplayMode", lambda: "none")()) not in {"tile_layer", "vispy_tile_layer"}:
            return False

        histogram_range = normalize_bounds(getattr(self.win.img_view, "getHistogramDataBounds", lambda: None)()) or levels
        mode = self._current_window_mode()
        source_override = getattr(self, "_level_presentation_source_override", None)
        if source_override is None:
            source = LevelSource(
                levels=levels,
                histogram_range=histogram_range,
                rank=LevelSourceRank.EXPLICIT_USER if mode == "absolute" else LevelSourceRank.PREVIOUS_COMMITTED,
                source_count=0,
                expected_count=0,
                semantic_key=getattr(session, "level_key", None),
                mode=mode,
            )
        else:
            source = replace(
                source_override,
                levels=levels,
                histogram_range=histogram_range,
                semantic_key=getattr(session, "level_key", None),
                mode=mode,
            )
        if source.rank == LevelSourceRank.EXPLICIT_USER:
            self._explicit_user_level_source = source
            session.user_levels_override = levels
        else:
            self._explicit_user_level_source = None
            session.user_levels_override = None
        session.applied_level_source = source
        # A concrete presentation command supersedes any automatic level pass
        # that was attached to the still-draining montage session.  Otherwise
        # queued initial commits can repeatedly restore their older auto range.
        session.force_auto = False
        needs_level_work = bool(session.begin_level_presentation_update(levels))

        if image_view_backend_capabilities(self.win.img_view).shader_windowing:
            session.acknowledge_uniform_level_presentation(levels)
            needs_level_work = False

        frame = getattr(self.win, "_committed_display_frame", None)
        if frame is not None and self._is_level_history_frame_usable(frame):
            self.win._committed_display_frame = replace(frame, levels=levels, histogram_range=histogram_range)
        if source.rank == LevelSourceRank.EXPLICIT_USER:
            self.win._notify_sync("levels")

        if not needs_level_work:
            return True

        if bool(final):
            committer = getattr(self, "commit_frame_session_presentation", None)
            if callable(committer) and not bool(getattr(self, "_montage_presentation_commit_active", False)):
                committer(session)
                return True

        scheduler = getattr(self, "apply_montage_presentation", None)
        if callable(scheduler):
            scheduler(session)
        return True


def tile_residency_budget_bytes(policy: MemoryPolicy) -> int:
    return int(
        min(
            int(getattr(policy, "visible_render_budget_bytes", 0) or 0),
            max(64 * MiB, int(getattr(policy, "display_cache_budget_bytes", 0) or 0) // 2),
            int(getattr(policy, "user_render_cap_bytes", 0) or 0),
        )
    )


def _current_view_range(window):
    try:
        view_range = window.win.img_view.getView().viewRange()
        return (
            (float(view_range[0][0]), float(view_range[0][1])),
            (float(view_range[1][0]), float(view_range[1][1])),
        )
    except Exception:
        return None


