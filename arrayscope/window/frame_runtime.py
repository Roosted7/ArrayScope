"""Montage runtime UI helpers for render orchestration."""

from __future__ import annotations

import sys
from time import perf_counter

import numpy as np
import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception
from arrayscope.core.cache_status import CacheStatus, CacheStatusSnapshot
from arrayscope.core.gui_callback_budget import GuiCallbackBudget
from arrayscope.kernel import Lane as WorkLane, WorkItem, complete_inline_work as _complete_inline_work
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.imageview2d import MontageTileOverlay
from arrayscope.display.montage import MontageTileState, montage_rect_for_viewport
from arrayscope.display.viewport import ViewportMode, view_ranges_near
from arrayscope.display.planning import normalize_bounds
from arrayscope.display.pyramid import PyramidCache
from arrayscope.operations.evaluator import _document_key
from arrayscope.render import effects as render_effects
from arrayscope.render.ladder import LadderPolicy, LodLadder
from arrayscope.render.pipeline import FramePipeline
from arrayscope.render.stages import LodAdmissionScope, RenderIntent
from arrayscope.ui.toasts import show_revert_action
from arrayscope.render import lod as render_lod
from arrayscope.window import frame_effects as montage_commit
from arrayscope.window.frame_effects import FramePipelineEffects
from arrayscope.window.montage_viewport import prioritize_montage_tiles, square_montage_fit_view_range
from arrayscope.window.render_contract import (
    montage_work_token as _montage_work_token,
    montage_work_token_is_current as _montage_work_token_is_current,
)


MONTAGE_AUTOFIT_VISIBLE_FRACTION = 0.80
MONTAGE_AUTOFIT_RESCUE_VISIBLE_FRACTION = 0.35


class FrameRuntimeMixin:
    def set_tile_truth_overlay_enabled(self, enabled: bool) -> None:
        self._tile_truth_overlay_enabled = bool(enabled)
        self._refresh_tile_truth_overlay()

    def _refresh_tile_truth_overlay(self) -> None:
        setter = getattr(self.win.img_view, "setTileTruthOverlayRows", None)
        if not callable(setter):
            return
        if not bool(getattr(self, "_tile_truth_overlay_enabled", False)):
            setter(())
            return
        session = getattr(self, "_frame_session", None)
        if session is None or not self._frame_session_is_current(session):
            setter(())
            return
        setter(session.diagnostic_tile_identity_rows(limit=12))

    def _on_montage_tile_slow(self, session_id):
        session = getattr(self, "_frame_session", None)
        # Not the shared predicate: on_slow callbacks capture only the session
        # id, so this is intentionally an id-only currency check.
        if session is None or int(session.session_id) != int(session_id):
            return
        self._show_frame_session_loading_overlay(session)

    def show_frame_session_slow_overlay(self, session):
        self._show_frame_session_loading_overlay_if_current(int(session.session_id), session.key)

    def _stop_frame_session_slow_overlay(self):
        self._frame_session_slow_key = None

    def _show_frame_session_loading_overlay_if_current(self, session_id, key):
        session = getattr(self, "_frame_session", None)
        if session is None or not self._is_current_frame_session(session_id, key):
            return
        if session.visible_plan_complete():
            return
        if not session.pending_tiles and not session.loading_tiles and not session.stage_fan_in.attached_requests:
            return
        self._show_frame_session_loading_overlay(session)

    def _show_frame_session_loading_overlay(self, session):
        if not self._frame_session_is_current(session):
            return
        if session.visible_plan_complete():
            return
        session.show_loading_overlays = True
        # The loading chrome is an overlay, not a frame transaction. The
        # caller has already committed/submitted the current presentation;
        # requesting it again here duplicated the initial backend commit and
        # made a cold render callback pay the full tile-layer walk twice.
        # Later payload completions naturally carry ``show_loading_overlays``
        # through their own bounded presentation requests.
        self.win.img_view.setImageStale(True)
        self.win.img_view.setEvaluationOverlay(True, "Updating image frame...")
        rect = montage_rect_for_viewport(session.plan, view_range=session.view_range, viewport_shape=session.viewport_shape)
        overlay_start = perf_counter()
        self._update_montage_tile_overlays_for_plan(session.plan, tuple(session.tile_states), rect)
        self._last_montage_overlay_update_ms = (perf_counter() - overlay_start) * 1000.0
        self.win.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.COMPUTING, "Evaluating image frame")
        if getattr(session, "defer_side_panels", False) or _viewport_interaction_active(self):
            self.win._deferred_side_panel_refresh_pending = True
        else:
            self.win._update_operation_dock()

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

    def _montage_quality_policy_mode(self) -> str:
        return render_lod.policy_mode_for_renderer(self)

    def _montage_pyramid_cache(self) -> PyramidCache:
        return render_lod.pyramid_cache_for_renderer(self)

    def _montage_render_intent(self, session) -> RenderIntent:
        return RenderIntent(
            semantic_key=session.key,
            viewport_key=session.view_range,
            presentation_key=(
                str(session.window_mode),
                normalize_bounds(getattr(session, "user_levels_override", None)),
                bool(getattr(session, "force_auto", False)),
            ),
            view_range=session.view_range,
            viewport_shape=tuple(session.viewport_shape),
            interactive=_interactive_active(self),
            tile_source_ids=tuple(
                (
                    int(tile.montage_index),
                    session.tile_semantic_source_id(int(tile.source_index)),
                )
                for tile in tuple(getattr(session.plan, "tiles", ()) or ())
            ),
            tile_source_indices=tuple(
                (int(tile.montage_index), int(tile.source_index))
                for tile in tuple(getattr(session.plan, "tiles", ()) or ())
            ),
        )

    def _lod_admission_scope(self, session, intent: RenderIntent) -> LodAdmissionScope:
        frame_plan = getattr(session, "frame_plan", None)
        onscreen_tiles = getattr(session, "onscreen_tile_numbers", None)
        visible_source = (
            tuple(onscreen_tiles())
            if bool(getattr(session, "display_committed", False)) and callable(onscreen_tiles)
            else tuple(getattr(session, "visible_tile_numbers", ()) or ())
        )
        visible = frozenset(int(tile) for tile in visible_source)
        # Session visibility intentionally retains a coverage ring so camera
        # motion never reveals black edges. It is not visible admission:
        # FramePlan.active is the canonical on-screen set, while coverage and
        # near tiles remain lower-priority retained/speculative work.
        coverage = set(
            int(tile)
            for tile in tuple(getattr(session, "visible_tile_numbers", ()) or ())
        )
        coverage.update(visible)
        coverage.update(int(tile) for tile in tuple(getattr(session, "loading_tiles", ()) or ()))
        coverage.update(int(tile) for tile in tuple(getattr(session, "active_tile_requests", ()) or ()))
        if (
            not bool(getattr(session, "shader_display", False))
            and (
                bool(getattr(session, "_cpu_atomic_successor_pending", False))
                or bool(getattr(session, "source_window_changed_pending", False))
            )
        ):
            # A CPU source-window handoff is one indivisible presentation.
            # FramePlan.active is normally the admission scope, but the
            # atomic successor must materialize every tile in the session's
            # zero-margin visible coverage. Excluding an edge tile here left
            # the successor permanently waiting with no pending producer.
            visible = frozenset((*visible, *coverage))
        near = tuple(getattr(frame_plan, "near_region_ids", ()) or ())
        if not near:
            near = (
                tuple(getattr(session, "_near_tile_numbers", lambda **_kwargs: ())(margin_tiles=2))
                if hasattr(session, "_near_tile_numbers")
                else ()
            )
        skipped = set(int(tile) for tile in tuple(getattr(session, "skipped_tiles", ()) or ()))
        missing = 0
        for tile_number in visible:
            if int(tile_number) in skipped:
                continue
            matches = getattr(session, "_tile_presentation_matches_current_plan", None)
            if not callable(matches) or not bool(matches(int(tile_number))):
                missing += 1
        return LodAdmissionScope(
            visible_tile_numbers=visible,
            coverage_tile_numbers=frozenset(coverage),
            near_tile_numbers=frozenset(int(tile) for tile in tuple(near or ())),
            viewport_key=intent.viewport_key,
            interactive=bool(intent.interactive),
            visible_missing_count=missing,
        )

    def _frame_pipeline_for_session(self, session) -> FramePipeline:
        pipeline = getattr(session, "pipeline", None)
        if pipeline is None:
            seed_tile = next(iter(tuple(getattr(session.plan, "tiles", ()) or ())), None)
            reduced_input_available = bool(
                seed_tile is not None
                and render_effects.preview_pipeline_commutes_for_display_lod(session, seed_tile)
            )
            pipeline = FramePipeline(
                self.win.kernel,
                FramePipelineEffects(self, session),
                LodLadder(
                    LadderPolicy(
                        mode=str(getattr(session, "lod_policy_mode", "native-only") or "native-only"),
                        floor_level=max(1, int(getattr(session, "lod_preview_level", 0) or 0)),
                        preview_level=max(1, int(getattr(session, "lod_preview_level", 0) or 0)),
                        # Per-tile preview rungs are cheap only for pipelines
                        # whose display-LOD result is independently tileable.
                        # Non-commuting but reduced-input-suitable pipelines
                        # such as FFT use the shared transform-preview path.
                        reduced_input_available=reduced_input_available,
                    )
                ),
                commit_max_items=8,
            )
            session.pipeline = pipeline
        self._frame_pipeline = pipeline
        return pipeline

    def request_montage_replan(self, session) -> None:
        """Coalesced ladder replan: at most one per event-loop turn.

        Per-completion callbacks (level ready, stage done/stale, declined
        admissions) must NOT call `retarget_frame_pipeline` directly: a
        full replan snapshots every tile, so N completions × N tiles was an
        O(N²) GUI-thread storm (272-tile fill: 204 replans, 2.5–14 s
        event-loop gaps). They mark their own bounded state and request one
        replan here. Category: zero-delay coalescing continuation.
        """

        if bool(getattr(self, "_montage_replan_gate_armed", False)):
            return
        self._montage_replan_gate_armed = True

        def fire() -> None:
            self._montage_replan_gate_armed = False
            # Replan whatever session is current when the gate fires — NOT the
            # one captured when it was armed.  During scroll churn the session
            # is reused/superseded between arm and fire (retarget_index_window
            # bumps session_id/key on the same object).  Gating on the captured
            # identity dropped the replan for the *new* current session: that
            # session's own request_montage_replan had already no-opped against
            # the still-armed gate, and this fire then bailed as "stale", so
            # freshly demoted tiles sat at their coarse floor with no wakeup
            # left (the onscreen-only mixed-LOD scroll stall — a timing race the
            # busier onscreen event loop loses and the offscreen one wins).
            # retarget_frame_pipeline re-validates the session and is
            # idempotent, so replanning the current session is always safe.
            current = getattr(self, "_frame_session", None)
            if current is None or not self._frame_session_is_current(current):
                return
            self.retarget_frame_pipeline(current)

        # Timer category: UI cosmetic. Event-turn barrier for coalescing a
        # current-session replan; kernel completion still owns work delivery.
        # A zero timer re-armed by completion/presentation callbacks can stay
        # continuously ready and outrun Qt's input/heartbeat timers. One
        # millisecond preserves event-turn coalescing while forcing a real
        # dispatcher opportunity between replans.
        Qt.QtCore.QTimer.singleShot(1, self, fire)

    def retarget_frame_pipeline(self, session, *, force_commit: bool = False) -> int:
        if session is None or not self._frame_session_is_current(session):
            return 0
        render_lod.selected_lod_factor(session)
        intent = self._montage_render_intent(session)
        scope = self._lod_admission_scope(session, intent)
        pipeline = self._frame_pipeline_for_session(session)
        submitted = pipeline.effects.submit_shared_transform_floor(scope)
        if montage_commit.complete_deferred_stage_fan_in(self, session):
            pipeline.effects.release_display_owned_pending(scope)
            return submitted
        montage_commit.rearm_ready_stage_dependents(session)
        submitted += pipeline.retarget(intent, session.lod_policy_decision.demand, scope)
        pipeline.effects.release_display_owned_pending(scope)
        if getattr(session, "pending_level_tiles", None) or int(getattr(session, "level_scan_remaining_tiles", 0) or 0) > 0:
            self._schedule_montage_cached_level_stats(session)
        if force_commit or session.flush_pending or session.final_commit_pending:
            self.apply_ready_montage_display(session)
        if not submitted:
            self._finish_frame_session_if_complete(session)
        unsettled = bool(
            getattr(session, "pending_tiles", None)
            or getattr(session, "active_tile_requests", None)
            or getattr(session, "loading_tiles", None)
            or getattr(session.stage_fan_in, "active_requests", None)
            or getattr(session.stage_fan_in, "attached_requests", None)
            or getattr(session.stage_fan_in, "tile_stage_keys", None)
            or getattr(session, "dirty_payloads", None)
            or getattr(session, "pending_payload_upserts", None)
        )
        if unsettled:
            self._ensure_montage_watchdog()
        return int(submitted)

    def replan_deferred_interactive_native_quality(self) -> bool:
        """Admit native-quality rungs deferred while an interaction was active."""

        session = getattr(self, "_frame_session", None)
        if session is None or not self._frame_session_is_current(session):
            return False
        pipeline = getattr(session, "pipeline", None)
        counters = getattr(pipeline, "counters", None)
        deferred = int(getattr(counters, "interactive_native_deferred", 0) or 0)
        residency_deferred = bool(getattr(session, "_interactive_residency_deferred", False))
        if (
            not residency_deferred
            and deferred <= int(getattr(self, "_montage_native_deferred_replanned", 0) or 0)
        ):
            return False
        session._interactive_residency_deferred = False
        self._montage_native_deferred_replanned = deferred
        self.retarget_frame_pipeline(session, force_commit=residency_deferred)
        return True

    # -- stall assertion probe (ADR 0051) -------------------------------------
    # The live render path must make lost wakeups impossible by construction.
    # This probe is diagnostics-only: when the diagnostics dialog is visible it
    # records a frozen unsettled signature, but it never mutates session state
    # or schedules work.

    def _montage_assertion_probe_enabled(self) -> bool:
        dialog = getattr(self.win, "_diagnostics_dialog", None)
        return bool(dialog is not None and dialog.isVisible())

    def _ensure_montage_watchdog(self) -> None:
        if not self._montage_assertion_probe_enabled():
            self._montage_watchdog_stop()
            return
        timer = getattr(self, "_montage_watchdog_timer", None)
        if timer is None:
            # Timer category: anti-hang fallback. Diagnostics-only stall
            # assertion probe; it never mutates rendering state.
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
        if not self._montage_assertion_probe_enabled():
            self._montage_watchdog_stop()
            return
        session = getattr(self, "_frame_session", None)
        if session is None or not self._frame_session_is_current(session):
            self._montage_watchdog_stop()
            return
        pending = len(session.pending_tiles)
        evaluating = len(session.lifecycle.evaluating_tiles)
        active = len(session.active_tile_requests)
        dirty = len(session.dirty_payloads)
        upserts = len(session.pending_payload_upserts)
        lod_pending = len(getattr(session, "pending_rung_materializations", ()) or ())
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
            # Deferred planning is scheduled work, not a stall; if it wedges,
            # the stable signature below reports it without re-kicking work.
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
            len(session.lifecycle.presented_tiles),
            len(session.rendered_tiles),
        )
        previous = getattr(self, "_montage_watchdog_state", None)
        self._montage_watchdog_state = signature
        if previous != signature:
            return  # work is progressing; stay armed.
        kernel = getattr(self.win, "kernel", None)
        kernel_diag = None if kernel is None else kernel.diagnostics()
        completion_queue = None if kernel is None else getattr(kernel, "completions", None)
        kernel_idle = bool(
            kernel_diag is not None
            and int(getattr(kernel_diag, "queued", 0) or 0) == 0
            and int(getattr(kernel_diag, "running", 0) or 0) == 0
            and int(getattr(kernel_diag, "active", 0) or 0) == 0
            and int(getattr(kernel_diag, "parked_deps", 0) or 0) == 0
            and int(getattr(kernel_diag, "parked_quota", 0) or 0) == 0
            and (completion_queue is None or completion_queue.empty())
        )
        if kernel_idle and active:
            pipeline = getattr(session, "pipeline", None)
            effects = getattr(pipeline, "effects", None)
            release = getattr(effects, "release_idle_evaluation_claims", None)
            released = 0 if release is None else int(release(session.active_tile_requests))
            if released:
                print(
                    "[arrayscope] STALL REPAIR: "
                    f"released_idle_evaluation_claims={released}",
                    file=sys.stderr,
                    flush=True,
                )
                self._montage_watchdog_state = None
                return
        probe = getattr(session, "diagnostic_tile_identity_rows", lambda **_kwargs: ())()
        actionable_probe = tuple(row for row in tuple(probe) if _stall_tile_probe_row_actionable(row))
        if not actionable_probe and session.visible_plan_complete():
            self._montage_watchdog_last_refinement_backlog = signature
            self._montage_watchdog_stop()
            return
        self._montage_stall_assertions = int(getattr(self, "_montage_stall_assertions", 0) or 0) + 1
        self._montage_watchdog_last_stall = signature
        print(
            "[arrayscope] STALL ASSERTION PROBE FIRED (ADR 0051): "
            f"signature={signature} "
            f"stage_active={len(session.stage_fan_in.active_requests)} "
            f"stage_attached={len(session.stage_fan_in.attached_requests)} "
            f"stage_deps={len(getattr(session.stage_fan_in, 'tile_stage_keys', {}))} "
            f"loading={len(session.loading_tiles)} "
            f"flush_pending={session.flush_pending} final={session.final_commit_pending}",
            file=sys.stderr,
            flush=True,
        )
        for row in actionable_probe[:20]:
            print(f"[arrayscope] STALL TILE PROBE: {row}", file=sys.stderr, flush=True)

    def _update_montage_tile_overlays_for_plan(self, plan, tile_states, viewport_rect) -> None:
        FrameRuntimeMixin._refresh_tile_truth_overlay(self)
        if not hasattr(self.win.img_view, "setMontageTileOverlays"):
            return
        session = getattr(self, "_frame_session", None)
        candidate_numbers: tuple[int, ...] | None = None
        tile_state_revision = None
        if session is not None and getattr(session, "plan", None) is plan:
            tile_state_revision = int(getattr(session, "tile_state_revision", 0) or 0)
            candidates = set(int(tile) for tile in getattr(session, "skipped_tiles", ()) or ())
            candidate_numbers = tuple(sorted(candidates))
            key = (
                id(plan),
                tuple(int(value) for value in viewport_rect),
                tile_state_revision,
                candidate_numbers,
            )
            if key == getattr(self, "_last_montage_overlay_update_key", None):
                return
            if not candidate_numbers:
                self._last_montage_overlay_update_key = key
                if int(getattr(self.win.img_view, "montageTileOverlayCount", lambda: 0)() or 0) != 0:
                    self.win.img_view.setMontageTileOverlays(())
                return
            self._last_montage_overlay_update_key = key
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
            # In-progress work is represented once by the evaluation overlay
            # and diagnostics. Per-tile loading rectangles doubled scene item
            # count, covered compatible predecessor pixels, and made a valid
            # retained frame look partially black. Only durable skipped/error
            # slots belong in the committed scene.
            if state != MontageTileState.SKIPPED:
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
        decision = getattr(self.win, "_gui_callback_budget_decision", lambda *args, **kwargs: None)(
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

    def retarget_montage_viewport(self) -> None:
        if getattr(self, "_montage_viewport_update_running", False):
            self.win._montage_viewport_update_pending = True
            return
        session = getattr(self, "_frame_session", None)
        self._montage_viewport_update_token = None if session is None else _montage_work_token(session, "viewport_update")
        self.apply_montage_viewport_retarget()

    def retarget_montage_priority_from_hover(self) -> None:
        if getattr(self.win, "_closing", False):
            return
        if getattr(self.win.view_state, "montage_axis", None) is None:
            return
        session = getattr(self, "_frame_session", None)
        if not self._frame_session_is_current(session):
            return
        if not session.pending_tiles:
            return
        self._montage_priority_retarget_token = _montage_work_token(session, "priority_retarget")
        self._montage_priority_retarget_pending = True
        self.apply_montage_priority_retarget()

    def refresh_montage_priority_targets(self, session) -> int:
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
        if not session.pending_tiles:
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
        total = len(session.pending_tiles)
        return session.retarget_tile_priority(
            focus=focus,
            max_items=max(1, int(total)),
            view_range=view_range,
        )

    def apply_montage_priority_retarget(self) -> None:
        self._montage_priority_retarget_pending = False
        if getattr(self.win, "_closing", False):
            return
        session = getattr(self, "_frame_session", None)
        if not self._frame_session_is_current(session):
            return
        token = getattr(self, "_montage_priority_retarget_token", None)
        if not _montage_work_token_is_current(session, token, "priority_retarget"):
            return
        if not session.pending_tiles:
            return
        budget = self._montage_callback_budget(
            "montage_priority_retarget",
            interactive=True,
            work_class="queue_metadata",
            item_cap=max(1, len(session.pending_tiles)),
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
            self.retarget_frame_pipeline(session)

    def apply_montage_viewport_retarget(self) -> None:
        if getattr(self.win, "_closing", False):
            return
        session = getattr(self, "_frame_session", None)
        token = getattr(self, "_montage_viewport_update_token", None)
        if token is not None and (
            session is None or not _montage_work_token_is_current(session, token, "viewport_update")
        ):
            return
        if getattr(self, "_montage_viewport_update_running", False):
            self.win._montage_viewport_update_pending = True
            return
        self._montage_viewport_update_running = True
        self.win._montage_viewport_update_pending = False
        try:
            if self._try_update_montage_viewport_only():
                retargeted = getattr(self, "_frame_session", None)
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
        if not getattr(self.win, "_montage_viewport_update_pending", False):
            return
        if not getattr(self, "_montage_viewport_continue_immediately", False):
            # A newer retarget already submitted current correctness work.
            return
        # Budgeted additions remain. One viewport slice per receiver-owned
        # event turn keeps fit/zoom callbacks bounded and lets paints/input run
        # between additions; the pending flag is the lost-wakeup obligation.
        self._montage_viewport_continue_immediately = False
        schedule = getattr(self, "_schedule_frame_viewport_update", None)
        if not callable(schedule):
            raise RuntimeError("montage viewport continuation has no receiver-owned gate")
        schedule(delay_ms=1)

    def _publish_montage_content_extent(self, plan) -> bool:
        """Publish the semantic montage extent as the viewport content shape.

        A first frame may publish this at plan time. Replacements publish only
        after the backend commit succeeds: the acknowledged frame owns camera
        coordinates, and moving the camera to uncommitted layout geometry can
        make still-resident old pixels appear black.
        """

        geometry = getattr(plan, "geometry", plan)
        montage = getattr(geometry, "montage", geometry)
        set_extent = getattr(self.win.img_view, "setViewportContentExtent", None)
        if not callable(set_extent):
            return False
        if montage is None or not getattr(montage, "indices", ()):
            return bool(set_extent(None))
        (x0, x1), (y0, y1) = _montage_full_view_range(montage)
        return bool(set_extent((max(1, int(round(y1 - y0))), max(1, int(round(x1 - x0))))))

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
        plan = getattr(session, "plan", None)
        plan_tiles_tuple = tuple(getattr(plan, "tiles", ()) or ())
        if (
            getattr(session, "_tile_source_ids_plan", None) is plan
            and len(source_ids) == len(plan_tiles_tuple)
        ):
            return source_ids
        plan_tiles = {
            int(tile.montage_index): tile
            for tile in plan_tiles_tuple
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
        session._tile_source_ids_plan = plan
        return source_ids
    def _classify_visible_montage_tiles(self, session) -> None:
        if not render_lod.native_missing_tile_queue_required(
            str(getattr(session, "lod_policy_mode", "")),
            getattr(getattr(session, "lod_policy_decision", None), "demand", None),
        ):
            return
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
            if index not in pending:
                newly_pending.append(tile)
                pending.add(index)
        for tile in prioritize_montage_tiles(
            newly_pending,
            view_range=((rect[0], rect[2]), (rect[1], rect[3])),
            focus=_montage_priority_focus(self, session.view_range),
        ):
            _enqueue_session_pending_tile(session, tile)
        if newly_pending:
            self.request_montage_replan(session)



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
    near_auto = getattr(viewport_controller, "is_near_auto", None)
    if callable(near_auto) and bool(near_auto(view_range)):
        return True
    if view_ranges_near(view_range, full_range):
        return True
    visible_fraction = max(0, int(visible_count)) / float(max(1, int(tile_count)))
    if visible_fraction <= MONTAGE_AUTOFIT_RESCUE_VISIBLE_FRACTION:
        return True
    return False


def _viewport_controller_auto_active_for_range(viewport_controller, view_range) -> bool:
    if viewport_controller is None:
        return False
    active = getattr(viewport_controller, "is_auto_active", None)
    if callable(active) and bool(active()):
        return True
    near_auto = getattr(viewport_controller, "is_near_auto", None)
    if callable(near_auto) and bool(near_auto(view_range)):
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


def _enqueue_session_pending_tile(session, tile) -> None:
    enqueue = getattr(session, "enqueue_pending_tile", None)
    if callable(enqueue):
        enqueue(tile)
        return
    session.pending_tiles.append(tile)


def _stall_tile_probe_row_actionable(row: dict[str, object]) -> bool:
    """Return whether a diagnostic row is evidence of a visible stall.

    R4 allows a tile to be visibly complete with preview/coarser pixels while
    exact/native refinement remains queued.  That state is useful telemetry,
    but it is not a stall assertion.  Keep source mismatches and live work
    claims actionable.
    """

    if not bool(row.get("visible_first_pixel_complete")):
        return True
    for key in ("loading", "active", "dirty", "pending_upsert"):
        if bool(row.get(key)):
            return True
    if row.get("evaluation_claim_source_index") is not None and not bool(
        row.get("evaluation_claim_matches_current_source")
    ):
        return True
    desired_present = row.get("desired_payload_source_index") is not None
    state_present = row.get("state_payload_source_index") is not None
    if desired_present and not bool(row.get("desired_matches_current_source")):
        return True
    if state_present and not bool(row.get("state_matches_current_source")):
        return True
    if desired_present and row.get("backend_source") not in (None, "") and not bool(
        row.get("backend_matches_desired")
    ):
        return True
    if not desired_present and state_present and row.get("backend_source") not in (None, "") and not bool(
        row.get("backend_matches_state")
    ):
        return True
    return False


def _view_range_center(view_range) -> tuple[float, float] | None:
    try:
        (x0, x1), (y0, y1) = view_range
        return ((float(x0) + float(x1)) * 0.5, (float(y0) + float(y1)) * 0.5)
    except Exception:
        return None


def _montage_priority_focus(window, view_range) -> tuple[float, float] | None:
    try:
        viewport_controller = getattr(getattr(window.win, "img_view", None), "viewport_controller", None)
        if viewport_controller is not None:
            focus = viewport_controller.priority_focus(view_range)
            if focus is not None:
                return focus
    except Exception:
        pass
    try:
        plan = getattr(getattr(window, "_frame_session", None), "plan", None)
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


def _viewport_interaction_active(window) -> bool:
    return bool(getattr(window.win, "_viewport_interaction_active", False))
