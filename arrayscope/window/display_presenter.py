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
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.operations.evaluator import _document_key
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.display_commit import DisplayCommitter
from arrayscope.window.display_frame import CommittedDisplayFrame, DisplayFrameKey, TiledValueSource
from arrayscope.window.montage_backend import MontageBackendDecision, backend_warning_for_actual_commit
from arrayscope.window.presentation import LevelSource, LevelSourceRank, decide_presentation, normalize_bounds
from arrayscope.window.render_model import CommitKind, DisplayPayload, PresentationInput, RenderRequestContext
from arrayscope.window.viewport_bridge import ViewportBridge


class DisplayPresentationMixin:
    def _apply_display_image(
        self,
        display_image,
        *,
        geometry,
        window_mode,
        previous_frame,
        force_auto,
        defer_side_panels: bool = False,
        document_key=None,
        request_key=None,
        render_generation=None,
        montage_level_key=None,
        montage_dirty_tiles=None,
        montage_tile_source_ids=None,
        montage_tile_payloads=None,
    ):
        self._apply_full_display_image(
            display_image,
            geometry=geometry,
            window_mode=window_mode,
            previous_frame=previous_frame,
            force_auto=force_auto,
            defer_side_panels=defer_side_panels,
            document_key=document_key,
            request_key=request_key,
            render_generation=render_generation,
            montage_level_key=montage_level_key,
            montage_dirty_tiles=montage_dirty_tiles,
            montage_tile_source_ids=montage_tile_source_ids,
            montage_tile_payloads=montage_tile_payloads,
        )

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
        montage_tile_payloads=None,
    ):
        commit_start = perf_counter()
        try:
            viewport_policy = self._viewport_policy_for_display_shape(display_image.data.shape[:2])
            levels_start = perf_counter()
            if commit_kind is None:
                commit_kind = CommitKind.FULL_MONTAGE_INITIAL if getattr(geometry, "montage", None) is not None else CommitKind.FULL_NORMAL
            context = self._render_request_context(
                document_key=document_key,
                request_key=request_key,
                render_generation=render_generation,
                semantic_key=montage_level_key,
            )
            decision = decide_presentation(
                PresentationInput(
                    payload=DisplayPayload(
                        image=display_image,
                        geometry=geometry,
                        viewport_policy=viewport_policy,
                        rgb_already_windowed=bool(getattr(display_image, "rgb_already_windowed", False)),
                        histogram_plot_data=histogram_plot_data,
                        montage_dirty_tiles=montage_dirty_tiles,
                        montage_tile_source_ids=montage_tile_source_ids,
                        montage_tile_payloads=montage_tile_payloads,
                    ),
                    context=context,
                    previous_frame=previous_frame,
                    window_mode=window_mode,
                    force_auto=force_auto,
                    commit_kind=commit_kind,
                    semantic_source=semantic_source,
                    applied_level_source=applied_level_source,
                    level_bounds=normalize_bounds(level_bounds),
                )
            )
            self._last_levels_histogram_ms = (perf_counter() - levels_start) * 1000.0

            set_image_start = perf_counter()
            backend_decision = self._montage_backend_decision_for_display(geometry, display_image.data)
            use_tile_layer = backend_decision.backend == "tile_layer" and hasattr(self.img_view, "setMontageTileLayerPresentation")
            if use_tile_layer:
                frame = self._display_committer().commit_tile_layer(decision.display_presentation, context.frame_key)
                actual_backend = "tile_layer"
            else:
                frame = self._display_committer().commit_full(decision.display_presentation, context.frame_key)
                actual_backend = "canvas"
            self._record_montage_backend_commit(backend_decision, actual_backend)
            self._last_set_image_ms = (perf_counter() - set_image_start) * 1000.0
            self.display_geometry = geometry
            self._set_committed_display_frame(frame)
            self._note_display_level_source(decision)
            if defer_side_panels:
                self._deferred_side_panel_refresh_pending = True
            else:
                self._update_operation_dock()
        
            # Apply axis flips after setting the image
            self.apply_axis_flips()
            self.img_view.setImageStale(False)
            self.img_view.setEvaluationOverlay(False)
            if defer_side_panels:
                self._deferred_side_panel_refresh_pending = True
            else:
                self._refresh_inspection_dock()
        
        except Exception as e:
            handle_ui_exception("image update", e)
            show_status_message(self, f"Image update failed: {e}")
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
        commit_kind=CommitKind.PROGRESSIVE_MONTAGE_PATCH,
        document_key=None,
        request_key=None,
        render_generation=None,
        montage_level_key=None,
        montage_dirty_tiles=None,
        montage_tile_source_ids=None,
        montage_tile_payloads=None,
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
            decision = decide_presentation(
                PresentationInput(
                    payload=DisplayPayload(
                        image=display_image,
                        geometry=geometry,
                        viewport_policy=viewport_policy,
                        rgb_already_windowed=bool(getattr(display_image, "rgb_already_windowed", False)),
                        histogram_plot_data=histogram_plot_data,
                        montage_dirty_tiles=montage_dirty_tiles,
                        montage_tile_source_ids=montage_tile_source_ids,
                        montage_tile_payloads=montage_tile_payloads,
                    ),
                    context=context,
                    previous_frame=previous_frame,
                    window_mode=window_mode,
                    force_auto=force_auto,
                    commit_kind=commit_kind,
                    semantic_source=semantic_source,
                    applied_level_source=applied_level_source,
                    level_bounds=normalize_bounds(level_bounds),
                )
            )
            self._last_levels_histogram_ms = (perf_counter() - levels_start) * 1000.0
            set_image_start = perf_counter()
            can_fast = (
                decision.allow_fast_commit
                and viewport_policy == ViewportPolicy.PRESERVE
                and getattr(self.img_view, "image", None) is not None
                and tuple(getattr(self.img_view.image, "shape", ())[:2]) == tuple(display_image.data.shape[:2])
                and hasattr(self.img_view, "updateImagePresentationFast")
            )
            backend_decision = self._montage_backend_decision_for_display(geometry, display_image.data)
            use_tile_layer = backend_decision.backend == "tile_layer" and hasattr(self.img_view, "setMontageTileLayerPresentation")
            if use_tile_layer:
                frame = self._display_committer().commit_tile_layer(decision.display_presentation, context.frame_key)
                actual_backend = "tile_layer"
            elif can_fast:
                frame = self._display_committer().commit_fast(decision.display_presentation, context.frame_key)
                actual_backend = "canvas"
            else:
                frame = self._display_committer().commit_full(decision.display_presentation, context.frame_key)
                actual_backend = "canvas"
            self._record_montage_backend_commit(backend_decision, actual_backend)
            self._last_set_image_ms = (perf_counter() - set_image_start) * 1000.0
            self.display_geometry = geometry
            self._set_committed_display_frame(frame)
            self._note_display_level_source(decision)
            self.apply_axis_flips()
            self.img_view.setImageStale(False)
        except Exception as e:
            handle_ui_exception("progressive image update", e)
            show_status_message(self, f"Image update failed: {e}")
        finally:
            self._last_progressive_commit_ms = (perf_counter() - commit_start) * 1000.0
            self._last_display_commit_ms = self._last_progressive_commit_ms

    def _should_use_montage_tile_layer_for_display(self, geometry, data) -> bool:
        policy = getattr(self, "_montage_tile_layer_policy", None)
        if policy is None:
            return False
        return bool(policy(geometry, data))

    def _montage_backend_decision_for_display(self, geometry, data) -> MontageBackendDecision:
        policy = getattr(self, "_montage_backend_policy", None)
        if policy is None or getattr(geometry, "montage", None) is None:
            return MontageBackendDecision("canvas", "not a montage display")
        return policy(geometry, data)

    def _record_montage_backend_commit(self, decision: MontageBackendDecision, actual_backend: str) -> None:
        self._last_montage_backend_choice = decision
        self._last_montage_backend_actual = str(actual_backend)
        self._last_montage_backend_warning = backend_warning_for_actual_commit(decision, actual_backend)

    def _display_committer(self) -> DisplayCommitter:
        committer = getattr(self, "_display_committer_instance", None)
        if committer is None or getattr(committer, "image_view", None) is not self.img_view:
            committer = DisplayCommitter(self.img_view)
            self._display_committer_instance = committer
        return committer

    def _previous_display_frame_for_policy(self, *, force_auto: bool) -> CommittedDisplayFrame | None:
        if force_auto:
            return None
        frame = getattr(self, "_committed_display_frame", None)
        if frame is None:
            return None
        return frame if self._is_level_history_frame_usable(frame) else None

    def _is_level_history_frame_usable(self, frame: CommittedDisplayFrame | None) -> bool:
        if frame is None or getattr(self, "_closing", False):
            return False
        if frame.key.document_key != _document_key(self.document):
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
            document_key = _document_key(self.document)
        if request_key is None:
            request_key = ("display", document_key, self.view_state)
        if render_generation is None:
            render_generation = self._capture_render_generation()
        return RenderRequestContext(
            document_key=document_key,
            request_key=request_key,
            render_generation=int(render_generation),
            semantic_key=semantic_key,
        )

    def _note_display_level_source(self, decision) -> None:
        frame = getattr(self, "_committed_display_frame", None)
        session = getattr(self, "_montage_session", None)
        if session is None or frame is None or frame.key.semantic_key != getattr(session, "level_key", None):
            return
        source = getattr(decision, "applied_level_source", None)
        if source is not None:
            session.applied_level_source = source

    def _viewport_bridge(self) -> ViewportBridge:
        bridge = getattr(self, "_viewport_bridge_instance", None)
        if bridge is None:
            bridge = ViewportBridge(self)
            self._viewport_bridge_instance = bridge
        return bridge

    def _display_frame_key(self, *, document_key=None, request_key=None, render_generation=None, semantic_key=None) -> DisplayFrameKey:
        if document_key is None:
            document_key = _document_key(self.document)
        if request_key is None:
            request_key = ("display", document_key, self.view_state)
        if render_generation is None:
            render_generation = self._capture_render_generation()
        return DisplayFrameKey(
            document_key=document_key,
            request_key=request_key,
            render_generation=int(render_generation),
            semantic_key=semantic_key,
        )

    def _set_committed_display_frame(self, frame: CommittedDisplayFrame) -> None:
        self._committed_display_request_key = frame.key.request_key
        self._committed_display_frame = frame

    def _on_display_levels_changed(self) -> None:
        try:
            levels = normalize_bounds(self.img_view.getLevels())
        except Exception:
            levels = None
        if levels is None:
            return
        try:
            histogram_range = normalize_bounds(self.img_view.getHistogramDataBounds())
        except Exception:
            histogram_range = None
        mode = self._current_window_mode()
        source = LevelSource(
            levels=levels,
            histogram_range=histogram_range or levels,
            rank=LevelSourceRank.EXPLICIT_USER if mode == "absolute" else LevelSourceRank.PREVIOUS_COMMITTED,
            source_count=0,
            expected_count=0,
            semantic_key=getattr(getattr(self, "_montage_session", None), "level_key", None),
            mode=mode,
        )
        self._explicit_user_level_source = source
        session = getattr(self, "_montage_session", None)
        if session is not None:
            session.applied_level_source = source

        frame = getattr(self, "_committed_display_frame", None)
        if frame is None or not self._is_level_history_frame_usable(frame):
            return
        self._committed_display_frame = replace(frame, levels=levels, histogram_range=histogram_range or frame.histogram_range)
