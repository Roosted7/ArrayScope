"""Normal image render orchestration for ArrayScope windows.

This module owns the non-montage visible-image path.  It intentionally contains
real rendering behavior, not a facade back into ``render.py``.
"""

from __future__ import annotations

from time import perf_counter

import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception
from arrayscope.core.cache_status import CacheStatus, CacheStatusSnapshot
from arrayscope.core.compute_policy import ComputeLane
from arrayscope.core.view_state import ChannelMode
from arrayscope.display.geometry import DisplayGeometry
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.operations.chunked import evaluate_image_snapshot_chunked
from arrayscope.operations.evaluator import _document_key, evaluate_image_snapshot, stage_document_key
from arrayscope.operations.render_plan import (
    RenderDecisionKind,
    degraded_view_state,
    estimate_visible_render_context,
    choose_visible_render_decision as _default_choose_visible_render_decision,
)
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.evaluation_controller import EvalPriority
from arrayscope.window.stage_warmup import schedule_stage_warmup


class NormalImageRenderMixin:
    def _interactive_render_cache_hit(self) -> bool:
        """Return whether the current normal-image request is already cached."""

        view_state = getattr(self, "view_state", None)
        if view_state is None or view_state.image_axes is None or view_state.montage_axis is not None:
            return False
        evaluator = getattr(self, "operation_evaluator", None)
        if evaluator is None:
            return False
        shader_display = bool(image_view_backend_capabilities(self.img_view).shader_windowing)
        colormap_lut = self._evaluation_colormap_lut(view_state, shader_display=shader_display)
        return evaluator.cached_image(
            view_state,
            colormap_lut=colormap_lut,
            shader_display=shader_display,
        ) is not None

    def update_image_view(self, *, force_autolevel: bool = False, defer_side_panels: bool = False):
        if self.view_state.image_axes is None: # No image view for 1D data
            return
        if self.view_state.montage_axis is not None:
            return self.update_montage_view(force_autolevel=force_autolevel, defer_side_panels=defer_side_panels)
        force_auto = force_autolevel or getattr(self, '_force_autolevel', False)
        window_mode = self._current_window_mode()
        # Capture presentation history before clearing montage/session state.
        previous_frame = self._previous_display_frame_for_policy(force_auto=force_auto)

        self._montage_session = None
        self._stop_montage_session_slow_overlay()
        self._current_montage_geometry = None
        self._current_montage_plan = None
        self._current_montage_canvas = None
        if hasattr(self.img_view, "clearMontageTileOverlays"):
            self.img_view.clearMontageTileOverlays()
            
        # reset the one-shot flag after using it
        if getattr(self, '_force_autolevel', False):
            self._force_autolevel = False

        view_state = self.view_state
        document = self.document
        shader_display = bool(image_view_backend_capabilities(self.img_view).shader_windowing)
        colormap_lut = self._evaluation_colormap_lut(view_state, shader_display=shader_display)
        request_key = self.operation_evaluator.image_key(view_state, colormap_lut=colormap_lut, document=document, shader_display=shader_display)
        render_generation = self._capture_render_generation()
        cached = self.operation_evaluator.cached_image(view_state, colormap_lut=colormap_lut, shader_display=shader_display)
        if cached is not None:
            self._last_render_was_degraded = False
            geometry = DisplayGeometry(view_state=view_state, display_shape=cached.data.shape[:2])
            self._apply_display_image(
                cached,
                geometry=geometry,
                window_mode=window_mode,
                previous_frame=previous_frame,
                force_auto=force_auto,
                defer_side_panels=defer_side_panels,
                document_key=_document_key(document),
                request_key=request_key,
                render_generation=render_generation,
            )
            return
        self._refresh_memory_policy(active_render=self.visible_evaluation_controller.is_busy())
        planning_start = perf_counter()
        estimated_bytes = self._estimated_image_display_bytes(view_state)
        context = estimate_visible_render_context(
            document,
            view_state,
            display_bytes=estimated_bytes,
            render_budget_bytes=self._visible_render_budget_bytes(),
        )
        self._last_planning_ms = (perf_counter() - planning_start) * 1000.0
        decision = _choose_visible_render_decision(context)
        self._last_render_context = context
        self._last_render_decision = decision
        self._last_render_request_key = None
        self._last_render_error = ""
        self._last_render_completed_ms = None
        if decision.kind == RenderDecisionKind.REFUSE:
            self.operation_evaluator.note_render_refused(decision.reason)
            show_status_message(
                self,
                decision.status_text or f"Image view would allocate {format_bytes(estimated_bytes)}. Reduce image-axis ranges or switch axes.",
                timeout=6000,
            )
            if getattr(self.img_view, "image", None) is not None:
                self.img_view.setImageStale(True)
                self.img_view.setEvaluationOverlay(True, decision.overlay_text or "Exact view over budget")
            if defer_side_panels:
                self._deferred_side_panel_refresh_pending = True
            else:
                self._update_operation_dock()
            return

        self._last_render_request_key = str(request_key)
        self.prefetch_evaluation_controller.cancel_prefetch()

        if decision.kind == RenderDecisionKind.DEGRADED_PREVIEW:
            preview_state = degraded_view_state(view_state, factor=decision.degraded_factor)
            preview_key = ("degraded_preview", request_key, decision.degraded_factor)
            submitted_at = perf_counter()

            def evaluate_preview(token):
                self._last_worker_queue_wait_ms = (perf_counter() - submitted_at) * 1000.0
                context = self._evaluation_context(ComputeLane.VISIBLE, token)
                return evaluate_image_snapshot(
                    document,
                    preview_state,
                    colormap_lut=colormap_lut,
                    cancellation_token=token,
                    degraded=True,
                    shader_display=shader_display,
                    stage_cache=self.operation_evaluator.stage_cache,
                    stage_document_key=stage_document_key(document),
                    evaluation_context=context,
                )

            def done_preview(result):
                if not self._is_current_render_generation(render_generation):
                    return
                if request_key != self.operation_evaluator.image_key(self.view_state, colormap_lut=colormap_lut, shader_display=shader_display):
                    return
                self.operation_evaluator.note_render_degraded()
                display_image = result.value
                self._degraded_rendered_view = display_image
                self._last_render_was_degraded = True
                geometry = DisplayGeometry(view_state=preview_state, display_shape=display_image.data.shape[:2])
                self._apply_display_image(
                    display_image,
                    geometry=geometry,
                    window_mode=window_mode,
                    previous_frame=previous_frame,
                    force_auto=force_auto,
                    defer_side_panels=defer_side_panels,
                    document_key=_document_key(document),
                    request_key=preview_key,
                    render_generation=render_generation,
                )
                self.img_view.setImageStale(True)
                self.img_view.setEvaluationOverlay(True, decision.overlay_text)
                show_status_message(self, decision.status_text, timeout=6000)

            self.visible_evaluation_controller.start_latest(
                evaluate_preview,
                key=preview_key,
                priority=EvalPriority.VISIBLE_IMAGE,
                replace_group="visible-image",
                on_done=done_preview,
                on_error=lambda exc: show_status_message(self, f"Preview update failed: {exc}"),
                on_stale=lambda: None,
                on_slow=lambda: self.img_view.setEvaluationOverlay(True, "Updating preview..."),
                pass_token=True,
            )
            return

        def evaluate(token):
            self._last_worker_queue_wait_ms = (perf_counter() - submitted_at) * 1000.0
            context = self._evaluation_context(ComputeLane.VISIBLE, token)
            if decision.kind == RenderDecisionKind.ASYNC_CHUNKED:
                return evaluate_image_snapshot_chunked(
                    document,
                    view_state,
                    chunk_axis=decision.chunk_axis,
                    chunk_size=decision.chunk_size,
                    colormap_lut=colormap_lut,
                    cancellation_token=token,
                    shader_display=shader_display,
                    stage_cache=self.operation_evaluator.stage_cache,
                    stage_document_key=stage_document_key(document),
                    evaluation_context=context,
                )
            return evaluate_image_snapshot(
                document,
                view_state,
                colormap_lut=colormap_lut,
                cancellation_token=token,
                shader_display=shader_display,
                stage_cache=self.operation_evaluator.stage_cache,
                stage_document_key=stage_document_key(document),
                evaluation_context=context,
            )

        def slow():
            self.img_view.setImageStale(True)
            text = "Updating view in chunks..." if decision.kind == RenderDecisionKind.ASYNC_CHUNKED else "Updating view..."
            self.img_view.setEvaluationOverlay(True, text)
            self.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.COMPUTING, "Evaluating image view")
            if defer_side_panels:
                self._deferred_side_panel_refresh_pending = True
            else:
                self._update_operation_dock()

        def done(result):
            if not self._is_current_render_generation(render_generation):
                return
            if request_key != self.operation_evaluator.image_key(self.view_state, colormap_lut=colormap_lut, shader_display=shader_display):
                return
            self._last_render_completed_ms = float(getattr(result, "eval_ms", 0.0) or 0.0)
            self._last_render_was_degraded = False
            self._degraded_rendered_view = None
            display_image = self.operation_evaluator.store_image_result(view_state, colormap_lut, result, shader_display=shader_display)
            geometry = DisplayGeometry(view_state=view_state, display_shape=display_image.data.shape[:2])
            self._apply_display_image(
                display_image,
                geometry=geometry,
                window_mode=window_mode,
                previous_frame=previous_frame,
                force_auto=force_auto,
                defer_side_panels=defer_side_panels,
                document_key=_document_key(document),
                request_key=request_key,
                render_generation=render_generation,
            )
            schedule_stage_warmup(self, view_state)
            self._schedule_prefetch_nearby_slices(view_state, colormap_lut)

        def error(exc):
            self._last_render_error = str(exc)
            self.img_view.setImageStale(False)
            self.img_view.setEvaluationOverlay(False)
            show_status_message(self, f"Image update failed: {exc}")

        submitted_at = perf_counter()
        self.visible_evaluation_controller.start_latest(
            evaluate,
            key=request_key,
            priority=EvalPriority.VISIBLE_IMAGE,
            replace_group="visible-image",
            on_done=done,
            on_error=error,
            on_stale=lambda: self.operation_evaluator.note_render_cancelled(),
            on_slow=slow,
            pass_token=True,
        )


def _choose_visible_render_decision(context):
    try:
        from arrayscope.window import render as render_module
        chooser = getattr(render_module, "choose_visible_render_decision")
    except (ImportError, AttributeError):
        chooser = _default_choose_visible_render_decision
    return chooser(context)
