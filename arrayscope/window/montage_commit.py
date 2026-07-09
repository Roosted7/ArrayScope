"""Montage pipeline effects that cross the GUI/backend presentation boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception
from arrayscope.core.gui_callback_budget import WARNING_THRESHOLD_MS
from arrayscope.core.compute_policy import ComputeLane
from arrayscope.kernel import Lane as WorkLane, Priority, Supersession, TaskSpec, WorkItem, complete_inline_work
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.geometry import DisplayGeometry, display_geometry_coordinates_equal
from arrayscope.display.model.commit import CommitKind, DisplayPayload, PresentationInput
from arrayscope.display.model.frame import TiledValueSource, display_tile_payload_has_semantics
from arrayscope.display.montage import montage_rect_for_viewport
from arrayscope.display.planning import LevelSourceRank, decide_presentation, normalize_bounds
from arrayscope.display.pyramid import reduce_box_mean
from arrayscope.display.slice_engine import DisplayImage
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.operations.chunked_stage import (
    materialize_stage_candidate_chunked,
    stage_materialization_allowed_chunk_axes,
)
from arrayscope.operations.evaluator import _document_key, stage_document_key
from arrayscope.operations.planner import final_region_for_request
from arrayscope.operations.regions import region_contains
from arrayscope.operations.slabs import plan_slab, request_for_image, stage_key_for_candidate
from arrayscope.operations.stage_fanin import StageFanInState
from arrayscope.render import effects as render_effects
from arrayscope.render import lod as render_lod
from arrayscope.render.ladder import Rung
from arrayscope.render.stages import CommitBatch, LodAdmissionScope
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.display_presenter import tile_residency_budget_bytes
from arrayscope.window.montage_session import _base_source_id
from arrayscope.window.montage_payload_cache import (
    payload_lod_matches,
    previous_tiled_payloads_by_base_source,
)


class _StaleBatchIntent:
    def __init__(self, semantic_key) -> None:
        self.semantic_key = semantic_key


@dataclass(frozen=True)
class _EvaluationClaim:
    semantic_key: object
    rung: int
    level: int
    source_index: int
    source_id: object

    def matches_step(self, intent, step) -> bool:
        return (
            self.semantic_key == getattr(intent, "semantic_key", None)
            and self.rung == int(step.rung)
            and self.level == int(step.level)
        )

    def matches_tile(self, intent, step, tile, source_id) -> bool:
        return (
            self.matches_step(intent, step)
            and self.source_index == int(tile.source_index)
            and self.source_id == source_id
        )


class MontagePipelineEffects:
    """Concrete pipeline effects for one live ``MontageRenderSession``.

    Worker-side evaluators stay in ``render.effects``. This class owns the
    GUI-thread gateway: converting ready rung payloads into session state,
    building a bounded ``DisplayTiledPresentation``, presenting through the
    shared surface contract, and feeding the backend acknowledgement back into
    ``TileLifecycle`` via ``MontageRenderSession``.
    """

    def __init__(self, renderer, session) -> None:
        self.renderer = renderer
        self.session = session

    def _evaluation_claim(self, intent, step, tile) -> _EvaluationClaim:
        source_index = int(tile.source_index)
        return _EvaluationClaim(
            semantic_key=getattr(intent, "semantic_key", getattr(self.session, "key", None)),
            rung=int(step.rung),
            level=int(step.level),
            source_index=source_index,
            source_id=self.session.tile_semantic_source_id(source_index),
        )

    def evaluate_rung(self, intent, step):
        if not self._session_is_current(intent):
            return lambda _token=None: None
        session = self.session
        tile = self._tile_for_step(step)
        if tile is None:
            return lambda _token=None: None

        if self._step_evaluates_reduced_display_payload(step, tile):
            # FLOOR/PREVIEW are degraded first-pixel rungs: whenever the tile
            # has nothing presentable yet, they must hand back their coarse
            # level so it is not left black — even at native scale (desired 0).
            # ingest_lod_demand() withholds a demand at native scale only to
            # keep cold *native* workers from reducing-at-ingest, so fall back
            # to the current policy demand. The payload is labeled
            # quality="preview": it never satisfies the native target, and
            # DESIRED/EXACT still follow and replace it without being cleared.
            demand = session.ingest_lod_demand()
            if demand is None:
                demand = getattr(getattr(session, "lod_policy_decision", None), "demand", None)
            semantic_source_id = session.tile_semantic_source_id(tile.source_index) if demand is not None else None

            def evaluate_preview(token=None):
                if demand is None or semantic_source_id is None:
                    return None
                return render_effects.evaluate_preview_tile(
                    session,
                    tile,
                    demand=demand,
                    semantic_source_id=semantic_source_id,
                    level=int(step.level),
                    cancellation_token=token,
                    shader_display=bool(getattr(session, "shader_display", False)),
                    evaluation_context=self.renderer.win._evaluation_context(ComputeLane.MONTAGE_TILE, token),
                )

            return evaluate_preview

        tile_number = int(tile.montage_index)
        request = None
        if step.rung == Rung.DESIRED:
            request = self.session.lifecycle.materialization_request_for(tile_number, self._level_key_for_step(tile, step))
        if step.rung == Rung.DESIRED and request is not None:
            pyramid = getattr(session, "pyramid_cache", None)

            def evaluate_materialization(token=None, request=request, pyramid=pyramid):
                if pyramid is None:
                    return None
                plane = request.source
                for level_key, rel in tuple(getattr(request, "chain", ()) or ((request.key, request.reduce_factor_xy),)):
                    plane = reduce_box_mean(plane, rel)
                    if level_key is not None:
                        plane = pyramid.admit(level_key, plane)
                return ("materialized", request)

            return evaluate_materialization

        def evaluate_target(token=None):
            demand = session.lod_policy_decision.demand
            return render_effects.evaluate_target_tile(
                session,
                tile,
                level=int(step.level),
                demand=demand,
                semantic_source_id=session.tile_semantic_source_id(tile.source_index),
                stage_cache=self.renderer.win.operation_evaluator.stage_cache,
                stage_materializer=self.renderer.win.operation_evaluator.stage_materializer,
                cancellation_token=token,
                shader_display=bool(getattr(session, "shader_display", False)),
                evaluation_context=self.renderer.win._evaluation_context(ComputeLane.MONTAGE_TILE, token),
            )

        return evaluate_target

    def tile_states(self, intent, demand, scope: LodAdmissionScope):
        if not self._session_is_current(intent):
            return ()
        self._release_inactive_evaluation_claims(getattr(scope, "visible_tile_numbers", ()))
        return render_effects.tile_lod_states(self.session, demand, scope=scope)

    def prepare_rung(self, intent, step) -> bool:
        tile = self._tile_for_step(step)
        if tile is None or not self._session_is_current(intent):
            return False
        tile_number = int(tile.montage_index)
        if self._step_evaluates_reduced_display_payload(step, tile):
            if self.session.lifecycle.preview_claim_matches(tile_number, int(step.rung), int(step.level)):
                return False
            if (
                step.rung == Rung.DESIRED
                and self.session.lifecycle.preview_claim_matches(tile_number, int(Rung.PREVIEW), int(step.level))
            ):
                return False
            return self.session.lifecycle.preview_claimed(tile_number, int(step.rung), int(step.level))
        if step.rung == Rung.DESIRED and int(step.level) > 0 and tile_number in self.session.rendered_tiles:
            materialization_key = (tile_number, int(step.rung), int(step.level))
            pyramid = getattr(self.session, "pyramid_cache", None)
            if pyramid is None:
                return False
            rendered = self.session.rendered_tiles.get(tile_number)
            demand = self.session.lod_policy_decision.demand
            level_key = self.session._pyramid_key_for(rendered, demand=demand, level=int(step.level))
            if self.session.lifecycle.materialization_request_for(tile_number, level_key) is not None:
                return False
            if pyramid.peek(level_key) is not None:
                self.session.lifecycle.level_resident(tile_number, level_key)
                return False
            if not pyramid.begin_pending(level_key):
                return False
            request = self.session._lod_materialization_request(
                rendered,
                demand=demand,
                level=int(step.level),
                key=level_key,
            )
            self.session.pending_rung_materializations.append(request)
            self.session.pending_rung_materializations.mark_started(request)
            return True
        if step.rung == Rung.DESIRED and int(step.level) > 0 and bool(step.reduce_from_native):
            if self._shared_transform_owns_display_target(tile, step):
                self.session.discard_pending_tile(tile_number)
                return False
            if self.session.lifecycle.preview_claim_matches(tile_number, int(Rung.DESIRED), int(step.level)):
                self.session.discard_pending_tile(tile_number)
                return False
            if self._display_payload_covers_display_target(tile_number, tile, step):
                self.session.discard_pending_tile(tile_number)
                return False
        if step.rung in (Rung.DESIRED, Rung.EXACT):
            if self._shared_preview_claim_covers_cold_tile(tile_number):
                return False
            if tile_number in self.session.rendered_tiles or tile_number in self.session.skipped_tiles:
                return False
            if tile_number in self.session.active_tile_requests:
                claim = self.session.lifecycle.evaluation_claim_for(tile_number)
                current_claim = self._evaluation_claim(intent, step, tile)
                if claim is not None and claim.matches_tile(intent, step, tile, current_claim.source_id):
                    if self._tile_task_is_live(tile_number):
                        return False
                    self._release_evaluation_claim(tile_number, marker=claim, request_replan=False)
                else:
                    self._release_evaluation_claim(tile_number, marker=claim, request_replan=False)
        return True

    def rung_admitted(self, intent, step, task_key) -> None:
        tile = self._tile_for_step(step)
        if tile is None or not self._session_is_current(intent):
            return
        tile_number = int(tile.montage_index)
        if self._step_evaluates_reduced_display_payload(step, tile):
            return
        if step.rung in (Rung.DESIRED, Rung.EXACT) and tile_number not in self.session.rendered_tiles:
            stage_key = self.session.stage_fan_in.tile_stage_keys.get(tile_number)
            stage_producer_key = self._stage_producer_key(stage_key)
            if stage_producer_key is None:
                stage_key = None
            self.session.mark_loading(tile)
            self.session.lifecycle.evaluation_claimed(tile_number, self._evaluation_claim(intent, step, tile))
            self.session.tile_ledger.task_admitted(
                tile_number,
                task_key,
                stage_key=stage_key,
                stage_producer_key=stage_producer_key,
            )

    def _step_evaluates_reduced_display_payload(self, step, tile) -> bool:
        if step.rung in (Rung.FLOOR, Rung.PREVIEW):
            return True
        tile_number = int(getattr(tile, "montage_index", getattr(step, "tile_number", -1)))
        if tile_number in getattr(self.session, "rendered_tiles", {}):
            return False
        return bool(
            step.rung == Rung.DESIRED
            and int(step.level) > 0
            and not bool(getattr(step, "reduce_from_native", True))
        )

    def _display_payload_covers_display_target(self, tile_number: int, tile, step) -> bool:
        payload = self.session.display_tile_payloads.get(int(tile_number))
        if not self._display_payload_is_current(tile_number, tile, payload=payload):
            return False
        lod = getattr(payload, "lod", None)
        if lod is None:
            return False
        return int(getattr(lod, "level", 0) or 0) <= int(step.level)

    def _display_payload_is_current(self, tile_number: int, tile, *, payload=None) -> bool:
        payload = self.session.display_tile_payloads.get(int(tile_number)) if payload is None else payload
        if payload is None:
            return False
        if int(getattr(payload, "source_index", -1)) != int(getattr(tile, "source_index", -2)):
            return False
        semantic_id = self.session.tile_semantic_source_id(tile.source_index)
        payload_source_id = getattr(payload, "source_id", None)
        if payload_source_id != semantic_id and _base_source_id(payload_source_id) != semantic_id:
            return False
        return int(tile_number) in set(getattr(self.session.lifecycle, "presented_tiles", ()) or ())

    def _display_payload_owns_pending_tile(self, tile_number: int, tile) -> bool:
        payload = self.session.display_tile_payloads.get(int(tile_number))
        if not self._display_payload_is_current(tile_number, tile, payload=payload):
            return False
        if self._shared_transform_owns_tile_display_target(tile):
            return True
        lod = getattr(payload, "lod", None)
        if lod is None or int(getattr(lod, "level", 0) or 0) <= 0:
            return False
        if not bool(getattr(self.session, "shader_display", False)):
            return False
        if str(getattr(payload, "quality", "exact") or "exact") != "preview":
            return False
        return bool(
            getattr(payload, "level_stats", None) is not None
            or getattr(payload, "level_data", None) is not None
            or getattr(payload, "histogram_data", None) is not None
        )

    def _shared_transform_owns_display_target(self, tile, step) -> bool:
        if not self._shared_transform_owns_tile_display_target(tile):
            return False
        return int(getattr(step, "level", 0) or 0) > 0

    def _shared_transform_owns_tile_display_target(self, tile) -> bool:
        demand = getattr(getattr(self.session, "lod_policy_decision", None), "demand", None)
        if demand is None:
            return False
        if int(getattr(demand, "desired_level", 0) or 0) <= 0:
            return False
        if render_effects.preview_pipeline_commutes_for_display_lod(self.session, tile):
            return False
        return render_effects.shared_preview_is_useful(
            self.session,
            tile,
            demand,
            upload_preview_useful=True,
        )

    def _shared_preview_claim_covers_cold_tile(self, tile_number: int) -> bool:
        if int(tile_number) in self.session.rendered_tiles:
            return False
        if self.session.display_tile_payloads.get(int(tile_number)) is not None:
            return False
        rec = self.session.lifecycle.peek(int(tile_number))
        return bool(rec is not None and rec.preview_claims)

    def rung_deps(self, intent, step) -> tuple[object, ...]:
        if not self._session_is_current(intent):
            return ()
        tile_number = int(step.tile_number)
        stage_key = self.session.stage_fan_in.tile_stage_keys.get(tile_number)
        if stage_key is None or stage_key in self.session.stage_fan_in.values:
            return ()
        stage_producer_key = self._stage_producer_key(stage_key)
        if stage_producer_key is None:
            return ()
        return (stage_producer_key,)

    def _stage_producer_key(self, stage_key):
        if stage_key is None:
            return None
        kernel = getattr(self.renderer, "kernel", None)
        if kernel is None:
            win = getattr(self.renderer, "win", None)
            kernel = getattr(win, "kernel", None)
        if kernel is None:
            return None
        if bool(getattr(kernel, "has_live_task", lambda _key: False)(stage_key)):
            return stage_key
        if bool(getattr(kernel, "has_completed_task", lambda _key: False)(stage_key)):
            return stage_key
        return None

    def _tile_task_is_live(self, tile_number: int) -> bool:
        row = self.session.tile_ledger.row(int(tile_number))
        claim = row.task_claim
        task_key = None if claim is None else claim.task_key
        if task_key is None:
            return False
        kernel = getattr(self.renderer, "kernel", None)
        if kernel is None:
            win = getattr(self.renderer, "win", None)
            kernel = getattr(win, "kernel", None)
        if kernel is None:
            return True
        return bool(getattr(kernel, "has_live_task", lambda _key: True)(task_key))

    def _release_inactive_evaluation_claims(self, tile_numbers=()) -> int:
        active_requests = getattr(self.session, "active_tile_requests", None)
        if active_requests is None:
            return 0
        visible = {int(tile) for tile in tuple(tile_numbers or ())}
        scope = visible or set(int(tile) for tile in tuple(active_requests or ()))
        released = 0
        for tile_number in sorted(scope):
            if tile_number not in active_requests:
                continue
            if self._tile_task_is_live(tile_number):
                continue
            released += int(self._release_evaluation_claim(tile_number, request_replan=False))
        return released

    def rung_dropped(self, intent, step) -> None:
        tile = self._tile_for_step(step)
        if tile is None:
            return
        tile_number = int(tile.montage_index)
        if step.rung in (Rung.FLOOR, Rung.PREVIEW):
            self.session.lifecycle.preview_released(tile_number, int(step.rung), int(step.level))
            return
        if step.rung == Rung.DESIRED and int(step.level) > 0:
            self.session.lifecycle.preview_released(tile_number, int(step.rung), int(step.level))
        if step.rung == Rung.DESIRED:
            request = self.session.lifecycle.materialization_request_for(tile_number, self._level_key_for_step(tile, step))
            if request is not None:
                self.session.pending_rung_materializations._apply_release_effects(
                    self.session.pending_rung_materializations.release(request)
                )
        if step.rung in (Rung.DESIRED, Rung.EXACT):
            marker = self._evaluation_claim(intent, step, tile)
            self._release_evaluation_claim(tile_number, marker=marker, request_replan=True)
            self.session.tile_ledger.task_released(tile_number, reason="dropped")

    def retained_native_source_available(self, intent, step) -> bool:
        if not self._session_is_current(intent):
            return False
        if step.rung not in (Rung.DESIRED, Rung.EXACT):
            return False
        if step.rung == Rung.DESIRED and int(step.level) > 0 and not bool(step.reduce_from_native):
            return False
        tile = self._tile_for_step(step)
        if tile is None:
            return False
        tile_number = int(tile.montage_index)
        stage_key = self.session.stage_fan_in.tile_stage_keys.get(tile_number)
        candidate = self.session.stage_fan_in.tile_stage_candidates.get(tile_number)
        if stage_key is None and candidate is not None:
            stage_key = self.renderer.win.operation_evaluator.stage_materializer.key_for_candidate(
                stage_document_key(self.session.document),
                candidate,
            )
        if stage_key is None:
            return False
        if stage_key in self.session.stage_fan_in.values:
            return True
        stage_cache = self.renderer.win.operation_evaluator.stage_cache
        getter = stage_cache.get_containing if hasattr(stage_cache, "get_containing") else stage_cache.get
        return getter(stage_key) is not None

    def release_display_owned_pending(self, scope: LodAdmissionScope | None = None) -> int:
        if not self._session_is_current():
            return 0
        pending_numbers = set(self.session.pending_tile_numbers())
        if not pending_numbers:
            return 0
        visible = None
        if scope is not None:
            visible = {int(tile) for tile in tuple(getattr(scope, "visible_tile_numbers", ()) or ())}
        released = 0
        for tile in tuple(getattr(getattr(self.session, "plan", None), "tiles", ()) or ()):
            tile_number = int(tile.montage_index)
            if tile_number not in pending_numbers:
                continue
            if visible is not None and tile_number not in visible:
                continue
            if not self._display_payload_owns_pending_tile(tile_number, tile):
                continue
            if self.session.discard_pending_tile(tile_number):
                released += 1
        return released

    def _release_evaluation_claim(self, tile_number: int, *, marker=None, request_replan: bool = True) -> bool:
        tile_number = int(tile_number)
        claim = self.session.lifecycle.evaluation_claim_for(tile_number)
        if marker is not None:
            if isinstance(marker, _EvaluationClaim):
                matched = (
                    claim is not None
                    and claim.semantic_key == marker.semantic_key
                    and claim.rung == marker.rung
                    and claim.level == marker.level
                )
            else:
                matched = claim == marker
            if not matched:
                return False
        elif claim is None and tile_number not in self.session.active_tile_requests:
            return False
        self.session.lifecycle.evaluation_request_cleared(tile_number)
        self.session.loading_tiles.discard(tile_number)
        self.session.tile_ledger.task_released(tile_number, reason="dropped")
        if tile_number not in self.session.rendered_tiles and tile_number not in self.session.display_tile_payloads:
            self.session.dirty_payloads.pop(tile_number, None)
            self.session.pending_payload_upserts.pop(tile_number, None)
        self.session.lifecycle.evaluation_declined(tile_number)
        if request_replan and self._session_is_current():
            self.renderer.request_montage_replan(self.session)
        return True

    def apply_commit(self, batch: CommitBatch) -> None:
        """Admit ready rung payloads; presentation happens through the gate.

        Admission (session/lifecycle bookkeeping) is cheap and runs per
        drained completion. The *presentation* commit (classify + geometry +
        delta walk + backend apply) is deliberately NOT run here: running it
        per completion was the R2 commit storm (272 tiles → 272 full commits
        inside bridge drains, multi-second event-loop gaps, and a visually
        all-at-once fill because paints never interleaved).
        """

        if batch.semantic_key is not None and batch.semantic_key != getattr(self.session, "key", batch.semantic_key):
            stale_intent = _StaleBatchIntent(batch.semantic_key)
            for row in tuple(batch.upserts or ()):
                if isinstance(row, tuple) and len(row) == 2:
                    step, _payload = row
                    self.rung_dropped(stale_intent, step)
            return
        self._admit_ready_payloads(batch.upserts)
        if not self._session_is_current():
            return
        self.request_presentation()

    def request_presentation(self) -> None:
        """Coalesced presentation continuation: at most one per loop turn.

        Category: zero-delay coalescing continuation (like the render
        coordinator), NOT wall-clock pacing — it exists so paints and input
        events interleave with bounded commits. Re-armed by the commit
        itself while backlog (flush/final/dirty/upserts) remains, so a
        bounded commit can never strand its remainder (the ADR 0051
        lost-wakeup rule: every deferral leaves a wakeup armed).
        """

        if bool(getattr(self.renderer, "_montage_presentation_gate_armed", False)):
            return
        self.renderer._montage_presentation_gate_armed = True
        # Timer category: UI cosmetic. Event-turn commit gate that coalesces
        # backend presentation updates; kernel/lifecycle own data readiness.
        Qt.QtCore.QTimer.singleShot(0, self.renderer, self._on_presentation_gate)

    def _on_presentation_gate(self) -> None:
        self.renderer._montage_presentation_gate_armed = False
        if not self._session_is_current():
            return
        self.commit_pending_session()

    def admit_tile_result(self, tile, result) -> int:
        """Admit one native target result into session/lifecycle state.

        Kernel bridge callbacks are already bounded; the old
        frame_renderer-side result deque/timer was a second fan-in queue.
        """

        return self._admit_evaluation_result(tile, result)

    def submit_shared_transform_floor(self, scope: LodAdmissionScope | None = None) -> int:
        """Shared display-target pass for reduced-input pipelines.

        FFT-over-montage-axis and similar pipelines can evaluate one reduced
        display volume and fan it out to every montage tile. That is both the
        first-pixel floor and the demanded target pass; per-tile native output
        is reserved for true level-0 semantic targets.
        """

        session = self.session
        renderer = self.renderer
        demand = session.ingest_lod_demand()
        if demand is None or not self._session_is_current():
            return 0
        visible_scope = None if scope is None else tuple(getattr(scope, "visible_tile_numbers", ()) or ())
        plan_tiles = tuple(
            tile
            for tile in tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ())
            if visible_scope is None or int(tile.montage_index) in set(int(value) for value in visible_scope)
        )
        if not plan_tiles:
            return 0
        seed = plan_tiles[0]
        if render_effects.preview_pipeline_commutes_for_display_lod(session, seed):
            return 0  # commuting pipelines get per-tile FLOOR/DESIRED rungs.
        if not render_effects.shared_preview_is_useful(session, seed, demand):
            return 0
        desired = int(getattr(demand, "desired_level", 0) or 0)
        preview_level = int(render_effects.preview_evaluation_level(session, demand))
        submitted = 0
        if preview_level > desired:
            preview_tiles = tuple(
                render_effects.shared_transform_candidate_tiles(
                    session,
                    level=preview_level,
                    tile_numbers=visible_scope,
                    include_missing=True,
                    require_presented_preview=False,
                )
            )
            submitted += self._submit_shared_transform_target(
                demand=demand,
                level=preview_level,
                tiles=preview_tiles,
                priority=Priority.INTERACTIVE,
                lane=WorkLane.DISPLAY_PREVIEW,
            )
            if desired > 0:
                target_tiles = tuple(
                    render_effects.shared_transform_candidate_tiles(
                        session,
                        level=desired,
                        tile_numbers=visible_scope,
                        include_missing=False,
                        require_presented_preview=True,
                    )
                )
                submitted += self._submit_shared_transform_target(
                    demand=demand,
                    level=desired,
                    tiles=target_tiles,
                    priority=Priority.VISIBLE_IMAGE,
                    lane=WorkLane.DISPLAY_PREPARATION,
                )
            return submitted
        tiles = tuple(
            render_effects.shared_transform_candidate_tiles(
                session,
                level=desired,
                tile_numbers=visible_scope,
                include_missing=True,
                require_presented_preview=False,
            )
        )
        return self._submit_shared_transform_target(
            demand=demand,
            level=desired,
            tiles=tiles,
            priority=Priority.VISIBLE_IMAGE,
            lane=WorkLane.DISPLAY_PREPARATION,
        )

    def _submit_shared_transform_target(self, *, demand, level: int, tiles, priority, lane) -> int:
        session = self.session
        renderer = self.renderer
        level = int(level)
        tiles = tuple(tiles or ())
        if not tiles:
            return 0
        shader_display = bool(getattr(session, "shader_display", False))
        marker = _shared_transform_marker(
            session,
            demand=demand,
            level=level,
            tiles=tiles,
            shader_display=shader_display,
        )
        if not self._claim_shared_transform_target(tiles, level=level, lane=lane):
            return 0

        def evaluate(token=None, tiles=tiles, level=level):
            return render_effects.evaluate_shared_preview(
                session,
                tiles[0],
                tiles,
                demand=demand,
                level=level,
                cancellation_token=token,
                shader_display=shader_display,
                evaluation_context=renderer.win._evaluation_context(ComputeLane.MONTAGE_TILE, token),
                upload_preview_useful=True,
            )

        def done(rows):
            self._release_shared_transform_claims(tiles, level=level, lane=lane)
            if not rows or not self._session_is_current():
                return
            admitted = self._admit_preview_payload(int(rows[0][0]), tuple(rows))
            if admitted:
                self.request_presentation()

        def dropped():
            self._release_shared_transform_claims(tiles, level=level, lane=lane)

        handle = renderer.win.kernel.submit(
            TaskSpec(
                key=("shared-target", marker),
                fn=evaluate,
                lane=lane,
                priority=priority,
                scope=f"montage:{session.key!r}",
                supersession=Supersession(("shared-target", session.key, level), marker),
                reusable=True,
                pass_token=True,
            ),
            on_done=done,
            on_stale=dropped,
            on_error=lambda exc: (dropped(), handle_ui_exception("shared transform target", exc)),
        )
        if handle is None:
            dropped()
            return 0
        return 1

    def commit_pending_session(self) -> None:
        if not self._session_is_current():
            return
        if bool(getattr(self.renderer, "_montage_commit_drain_active", False)):
            self.session.final_commit_pending = True
            self.session.flush_pending = True
            self.request_presentation()
            return
        self.renderer._montage_commit_drain_active = True
        try:
            self.renderer._classify_visible_montage_tiles(self.session)
            direct_presentation = self.direct_tile_layer_presentation()
            if direct_presentation is None:
                raise RuntimeError("montage presentation could not be built")
            self._commit_tile_layer(direct_presentation, commit_start=perf_counter())
        finally:
            self.renderer._montage_commit_drain_active = False
        self._rearm_if_backlog()

    def _backlog_signature(self) -> tuple:
        """Everything a presentation commit can make progress on.

        Progress dimensions MUST all appear here: the gate stops re-arming
        when the signature repeats, so an invisible dimension turns steady
        progress into a false no-progress stop. Level convergence proved
        this: bounded per-commit level rewindows shrank the stale count
        while flush/dirty stayed constant, and the gate stalled a 272-tile
        FFT level refinement at 145 stale tiles.
        """

        session = self.session
        level_pending = bool(session.has_pending_level_update())
        level_stale = 0
        if level_pending:
            snapshot = session.level_presentation_snapshot()
            level_stale = int(getattr(snapshot, "stale_count", 0) or 0)
        return (
            bool(getattr(session, "flush_pending", False)),
            bool(getattr(session, "final_commit_pending", False)),
            len(tuple(getattr(session, "dirty_payloads", ()) or ())),
            len(tuple(getattr(session, "pending_payload_upserts", ()) or ())),
            len(tuple(getattr(session, "pending_removals", ()) or ())),
            level_pending,
            level_stale,
            int(getattr(session, "level_revision", 0) or 0),
            len(tuple(_call(session, "backend_identity_mismatch_tiles") or ())),
        )

    def _rearm_if_backlog(self) -> None:
        """Re-arm the gate while a *shrinking* backlog remains.

        A commit that leaves the identical backlog signature is not making
        progress; re-arming would spin the event loop. It is either waiting
        on an external completion (level scans, evaluations — each of those
        completion paths calls ``request_presentation`` itself) or a real
        wedge, counted here as a bug report (ADR 0051: rescues hide bugs).
        """

        signature = self._backlog_signature()
        if not any(signature):
            self.renderer._montage_gate_last_backlog = None
            return
        previous = getattr(self.renderer, "_montage_gate_last_backlog", None)
        self.renderer._montage_gate_last_backlog = signature
        if previous == signature:
            self.renderer._montage_gate_no_progress = (
                int(getattr(self.renderer, "_montage_gate_no_progress", 0) or 0) + 1
            )
            return
        self.request_presentation()

    def direct_tile_layer_presentation(self):
        session = self.session
        tile_states = session.ensure_tile_states()
        placeholder = montage_tile_layer_placeholder(session)
        geometry = DisplayGeometry(
            view_state=session.view_state,
            display_shape=tuple(placeholder.shape[:2]),
            montage=session.plan.geometry,
            montage_origin_x=0,
            montage_origin_y=0,
            montage_tile_states=tile_states,
        )
        decision = self.renderer._montage_backend_policy(geometry, placeholder)
        if decision.backend != "tile_layer":
            return None
        return DisplayImage(data=placeholder, histogram_data=None, rgb_already_windowed=False), geometry

    # ------------------------------------------------------------------ admit

    def _admit_ready_payloads(self, rows) -> None:
        for row in tuple(rows or ()):
            if not isinstance(row, tuple) or len(row) != 2:
                continue
            step, payload = row
            if payload is None:
                continue
            if (
                step.rung == Rung.DESIRED
                and isinstance(payload, tuple)
                and len(payload) == 2
                and payload[0] == "materialized"
            ):
                self._admit_materialized_rung(step, payload[1])
                continue
            tile = self._tile_for_step(step)
            if tile is None:
                continue
            if getattr(payload, "value", None) is not None:
                self._admit_evaluation_result(tile, payload)
                continue
            self.session.lifecycle.preview_released(int(step.tile_number), int(step.rung), int(step.level))
            self._admit_preview_payload(int(step.tile_number), payload)

    def _admit_materialized_rung(self, step, request) -> None:
        tile_number = int(step.tile_number)
        self.session.lifecycle.preview_released(tile_number, int(step.rung), int(step.level))
        self.session.pending_rung_materializations.mark_resident(request)
        self.session.lod_materializations_completed = (
            int(getattr(self.session, "lod_materializations_completed", 0) or 0) + 1
        )
        if tile_number in self.session.rendered_tiles:
            self.session.dirty_payloads[tile_number] = None
            self.session.flush_pending = True
            self.session.final_commit_pending = True
        else:
            claim = self.session.lifecycle.evaluation_claim_for(tile_number)
            if claim is not None and claim.rung == int(step.rung) and claim.level == int(step.level):
                self._release_evaluation_claim(tile_number, marker=claim, request_replan=False)
        self.renderer.request_montage_replan(self.session)

    def _admit_evaluation_result(self, tile, result) -> int:
        session = self.session
        if not self._session_is_current():
            return 0
        self.session.lifecycle.evaluation_request_cleared(int(tile.montage_index))
        rendered = self.renderer.win.operation_evaluator.store_montage_tile_result(
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
            session.stage_backed_tiles_pending = max(0, int(session.stage_backed_tiles_pending) - 1)
            session.tile_compute_waiting_for_stage = max(0, int(session.tile_compute_waiting_for_stage) - 1)
        else:
            session.tile_compute_direct += 1
            session.tile_compute_direct_ms += eval_ms
            session.tile_compute_direct_max_ms = max(float(session.tile_compute_direct_max_ms), eval_ms)
        self.renderer._update_montage_level_bounds_from_rendered(
            session.level_key,
            rendered,
            expected_indices=self.renderer._montage_level_expected_indices(session),
        )
        self.renderer._queue_montage_level_refinement(session, rendered)
        session.mark_materialized(rendered)
        session.tile_ledger.task_released(int(tile.montage_index), reason="completed")
        session.dirty_tiles.append(int(tile.montage_index))
        return rendered_tile_nbytes(rendered)

    def _admit_preview_payload(self, tile_number: int, payload) -> bool:
        session = self.session
        if not self._session_is_current():
            return False
        rows = payload if _looks_like_shared_preview_rows(payload) else ((int(tile_number), *payload),)
        upserted = False
        admitted_any = False
        visible_previews = 0
        for row in tuple(rows or ()):
            tile_number, key, plane, histogram, shader_mapping, texture_kind, level_data, level_stats = preview_row_parts(row)
            admitted = session.admit_preview_plane(
                tile_number,
                key,
                plane,
                histogram,
                shader_mapping=shader_mapping,
                texture_kind=texture_kind,
                level_data=level_data,
                level_stats=level_stats,
            )
            if not admitted:
                continue
            admitted_any = True
            session._ensure_floor_payloads((tile_number,))
            preview_upserted = (
                int(tile_number) in session.pending_payload_upserts
                and str(getattr(session.display_tile_payloads.get(int(tile_number)), "quality", "exact")) == "preview"
            )
            visible_previews += int(preview_upserted)
            upserted = upserted or preview_upserted
        if visible_previews:
            session.lod_preview_presentations = (
                int(getattr(session, "lod_preview_presentations", 0) or 0) + int(visible_previews)
            )
        if upserted:
            session.flush_pending = True
            session.final_commit_pending = True
        return bool(admitted_any)

    # ------------------------------------------------------------------ commit

    def _commit_tile_layer(self, direct_presentation, *, commit_start: float) -> None:
        renderer = self.renderer
        session = self.session
        display_image, rendered_geometry = direct_presentation
        dirty_tiles = session.consume_dirty_tiles()
        tile_source_ids = renderer._montage_tile_source_ids(session)
        renderer._montage_committed_tile_upserts_last_flush = len(dirty_tiles)
        renderer._current_montage_geometry = session.plan.geometry
        renderer._current_montage_plan = session.plan
        renderer._next_viewport_policy = ViewportPolicy.PRESERVE
        renderer._montage_presentation_commit_active = True
        _reset_commit_timings(renderer)
        try:
            payload_start = perf_counter()
            selected_lod_factor = int(session._selected_lod_factor())
            reuse_any_lod = bool(getattr(session, "_resident_lod_active", lambda: False)())
            if not session.lifecycle.presented_tiles:
                previous_payloads = {
                    key: payload
                    for key, payload in previous_tiled_payloads_by_base_source(
                        getattr(renderer.win, "_committed_display_frame", None)
                    ).items()
                    if reuse_any_lod or payload_lod_matches(payload, selected_lod_factor)
                }
                retained_payloads = renderer._retained_tiled_payload_store().payloads_by_base_source(
                    lod_factor=None if reuse_any_lod else selected_lod_factor
                )
                if retained_payloads:
                    previous_payloads.update(retained_payloads)
                if previous_payloads:
                    session.seed_display_tile_payloads(previous_payloads, tile_source_ids, tile_numbers=tuple(session.dirty_payloads))
                    if reuse_any_lod:
                        session.mark_ladder_swaps_for_viewport()
            base_tile_state = session.tile_presentation_state
            fast_drain = persistent_tile_layer_fast_drain_enabled(renderer, session)
            renderer._persistent_tile_layer_fast_drain_last_enabled = bool(fast_drain)
            renderer._persistent_tile_layer_fast_drain_enabled_count = int(
                getattr(renderer, "_persistent_tile_layer_fast_drain_enabled_count", 0) or 0
            ) + int(bool(fast_drain))
            limits = tile_layer_upsert_limits(renderer, session)
            cold_deadline_ms = None
            if limits:
                limits = dict(limits)
                cold_deadline_ms = limits.pop("cold_deadline_ms", None)
            tile_state, tile_delta = session.build_tile_presentation(
                tile_source_ids,
                cold_deadline_ms=cold_deadline_ms,
                **limits,
            )
            if persistent_tile_residency_backend(renderer, session):
                dirty_tiles = tuple(int(tile) for tile in dirty_tiles if int(tile) in set(tile_delta.upserts))
            active_payloads = tile_state.active_payloads(tile_delta)
            first_display_commit = not bool(session.display_committed)
            requested_levels = session_requested_levels(session)
            explicit_auto = bool(getattr(session, "force_auto", False) and requested_levels is None)
            if self._empty_first_commit_can_wait(first_display_commit, explicit_auto, active_payloads, tile_delta):
                session.final_commit_pending = False
                session.flush_pending = False
                renderer._last_montage_tile_payload_build_ms = (perf_counter() - payload_start) * 1000.0
                renderer.request_montage_replan(session)
                return
            if self._empty_progressive_commit_settled(first_display_commit, explicit_auto, tile_delta):
                session.final_commit_pending = False
                session.flush_pending = False
                renderer._last_montage_tile_payload_build_ms = (perf_counter() - payload_start) * 1000.0
                session.note_committed()
                self._finish_after_noop_commit()
                return
            if _commit_should_queue_level_stats(renderer, first_display_commit=first_display_commit):
                level_payloads = active_payloads if first_display_commit else dict(tile_delta.upserts)
                renderer._queue_montage_level_stats_for_payloads(session, level_payloads)
            rendered_geometry = replace(rendered_geometry, montage_tile_states=session.ensure_tile_states())
            self._install_warm_residency_scheduler(rendered_geometry)
            renderer._last_montage_tile_payload_build_ms = (perf_counter() - payload_start) * 1000.0
            prepare_apply_start = perf_counter()
            prepare_stats_start = perf_counter()
            level_stats = renderer._montage_level_stats_for_session(session)
            renderer._last_montage_tile_prepare_stats_ms = (perf_counter() - prepare_stats_start) * 1000.0
            semantic_commit = tiled_payloads_include_semantics(active_payloads)
            decision_force_auto = bool(explicit_auto and semantic_commit)
            if tile_layer_auto_levels_wait_for_complete_source(
                renderer,
                session,
                decision_force_auto,
                level_stats,
            ):
                session.final_commit_pending = True
                session.flush_pending = True
                if not getattr(session, "pending_level_tiles", None) and int(getattr(session, "level_scan_remaining_tiles", 0) or 0) <= 0:
                    renderer._mark_montage_level_scan_pending(session)
                renderer._schedule_montage_cached_level_stats(session)
                return
            prepare_metadata_start = perf_counter()
            metadata_can_advance = bool(first_display_commit or self._side_work_visible_settled())
            level_metadata_improved = bool(
                metadata_can_advance and renderer._should_publish_montage_level_metadata(session, level_stats)
            )
            publish_auto_metadata = bool(
                explicit_auto and metadata_can_advance and (first_display_commit or level_metadata_improved)
            )
            # The aggregate histogram plot is derived from the level stats, so
            # it must (re)publish whenever those stats first arrive or improve
            # — not only on the first display commit. On startup the first
            # commit runs before any tile has level stats, so a first-commit-
            # only publish left the histogram panel empty until an unrelated
            # action (index change, channel toggle) forced a fresh first
            # commit (field defect 2026-07).
            publish_histogram_plot = bool(first_display_commit or level_metadata_improved)
            publish_metadata = publish_auto_metadata or publish_histogram_plot or level_metadata_improved
            renderer._last_montage_tile_prepare_metadata_ms = (perf_counter() - prepare_metadata_start) * 1000.0
            prepare_source_start = perf_counter()
            semantic_source = renderer._montage_level_source_for_session(session, allow_partial=publish_metadata)
            renderer._last_montage_tile_prepare_source_ms = (perf_counter() - prepare_source_start) * 1000.0
            prepare_histogram_start = perf_counter()
            histogram_plot_data = (
                renderer._montage_histogram_plot_data_for_session(session, allow_partial=publish_metadata)
                if publish_histogram_plot
                else None
            )
            renderer._last_montage_tile_prepare_histogram_ms = (perf_counter() - prepare_histogram_start) * 1000.0
            renderer._last_montage_tile_prepare_apply_ms = (perf_counter() - prepare_apply_start) * 1000.0
            applied = self._present_tile_delta(
                display_image,
                rendered_geometry,
                tile_state=tile_state,
                base_tile_state=base_tile_state,
                tile_delta=tile_delta,
                semantic_source=semantic_source,
                applied_level_source=session.applied_level_source,
                histogram_plot_data=histogram_plot_data,
                first_display_commit=first_display_commit,
                explicit_auto=explicit_auto,
                requested_levels=requested_levels,
                semantic_commit=semantic_commit,
                decision_force_auto=decision_force_auto,
            )
            if not applied:
                return
            self._acknowledge_and_publish(tile_delta, tile_state, rendered_geometry, active_payloads, commit_start=commit_start)
        finally:
            renderer._montage_presentation_commit_active = False

    def _present_tile_delta(
        self,
        display_image,
        geometry,
        *,
        tile_state,
        base_tile_state,
        tile_delta,
        semantic_source,
        applied_level_source,
        histogram_plot_data,
        first_display_commit: bool,
        explicit_auto: bool,
        requested_levels,
        semantic_commit: bool,
        decision_force_auto: bool,
    ) -> bool:
        renderer = self.renderer
        session = self.session
        apply_start = perf_counter()
        if first_display_commit and self._commit_direct_delta(
            display_image,
            geometry,
            tile_state=tile_state,
            base_tile_state=base_tile_state,
            tile_delta=tile_delta,
            semantic_source=semantic_source,
            applied_level_source=applied_level_source,
            histogram_plot_data=histogram_plot_data,
            explicit_auto=explicit_auto,
            requested_levels=requested_levels,
            semantic_commit=semantic_commit,
            allow_uncommitted_persistent=True,
        ):
            pass
        elif first_display_commit:
            renderer._apply_full_display_image(
                display_image,
                geometry=geometry,
                window_mode=session.window_mode,
                previous_frame=getattr(renderer.win, "_committed_display_frame", None),
                force_auto=decision_force_auto,
                defer_side_panels=getattr(session, "defer_side_panels", False),
                semantic_source=semantic_source,
                applied_level_source=applied_level_source,
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
        elif self._commit_direct_delta(
            display_image,
            geometry,
            tile_state=tile_state,
            base_tile_state=base_tile_state,
            tile_delta=tile_delta,
            semantic_source=semantic_source,
            applied_level_source=applied_level_source,
            histogram_plot_data=histogram_plot_data,
            explicit_auto=explicit_auto,
            requested_levels=requested_levels,
            semantic_commit=semantic_commit,
        ):
            pass
        else:
            renderer._apply_progressive_display_image(
                display_image,
                geometry=geometry,
                window_mode=session.window_mode,
                previous_frame=getattr(renderer.win, "_committed_display_frame", None),
                force_auto=False,
                viewport_policy=ViewportPolicy.PRESERVE,
                semantic_source=semantic_source,
                applied_level_source=applied_level_source,
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
        renderer._last_montage_tile_layer_apply_ms = (perf_counter() - apply_start) * 1000.0
        return True

    def _commit_direct_delta(
        self,
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
        requested_levels,
        semantic_commit: bool,
        allow_uncommitted_persistent: bool = False,
    ) -> bool:
        renderer = self.renderer
        session = self.session
        if not direct_montage_tile_delta_commit_enabled(
            renderer,
            session,
            allow_uncommitted_persistent=allow_uncommitted_persistent,
        ):
            return False
        previous_frame = getattr(renderer.win, "_committed_display_frame", None)
        if previous_frame is None or not isinstance(getattr(previous_frame, "value_source", None), TiledValueSource):
            return False
        previous_geometry = getattr(previous_frame, "geometry", None)
        if not safe_tiled_payload_geometry_retarget(previous_geometry, geometry):
            return False
        context = renderer._render_request_context(
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
                    tile_residency_budget_bytes=tile_residency_budget_bytes(renderer._memory_policy()),
                ),
                context=context,
                previous_frame=previous_frame,
                window_mode=session.window_mode,
                force_auto=False,
                commit_kind=CommitKind.EXPLICIT_AUTO_WINDOW if explicit_auto else CommitKind.PROGRESSIVE_FRAME_PATCH,
                semantic_source=semantic_source,
                applied_level_source=applied_level_source,
                user_levels=requested_levels,
            )
        )
        set_image_start = perf_counter()
        backend_decision = renderer._montage_backend_decision_for_display(geometry, display_image.data)
        if backend_decision.backend != "tile_layer":
            return False
        committer = renderer._display_committer()
        committer.commit_tiled_delta(decision.display_presentation)
        renderer._record_montage_backend_commit(backend_decision, "tile_layer")
        renderer._last_set_image_ms = (perf_counter() - set_image_start) * 1000.0
        renderer.display_geometry = geometry
        report = getattr(committer, "last_tile_commit_report", None)
        semantic_frame_commit = bool(semantic_commit and bool(getattr(report, "presented_tiles", ())))
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
                renderer._set_committed_display_frame(frame)
                renderer._consume_pending_display_levels(session.user_levels_override)
                renderer._note_display_level_source(decision)
                _call(renderer.win, "_apply_viewport_continuity_when_ready")
                _call(renderer, "_show_pending_montage_view_revert")
                _call(renderer, "_refresh_hover_after_display_commit")
        renderer.win.apply_axis_flips()
        renderer.win.img_view.setImageStale(False)
        return True

    def _acknowledge_and_publish(self, tile_delta, tile_state, geometry, active_payloads, *, commit_start: float) -> None:
        renderer = self.renderer
        session = self.session
        report = getattr(renderer._display_committer(), "last_tile_commit_report", None)
        acknowledge_start = perf_counter()
        acknowledged = session.acknowledge_tile_presentation(tile_delta, report, levels=normalize_bounds(renderer.win.img_view.getLevels()))
        renderer._last_montage_tile_acknowledge_ms = (perf_counter() - acknowledge_start) * 1000.0
        retained_start = perf_counter()
        renderer._retained_tiled_payload_store().remember_acknowledged(
            accepted_tiled_payloads(acknowledged.payloads, tile_delta, report)
        )
        renderer._last_montage_tile_retained_store_ms = (perf_counter() - retained_start) * 1000.0
        state_start = perf_counter()
        if not session.has_stale_level_presentations():
            session.set_level_update_pending(False)
        presented_tiles = active_payloads if report is None else getattr(report, "presented_tiles", active_payloads)
        session.mark_presented(presented_tiles)
        released_display_pending = self.release_display_owned_pending()
        if released_display_pending:
            renderer._montage_display_owned_pending_released = (
                int(getattr(renderer, "_montage_display_owned_pending_released", 0) or 0)
                + int(released_display_pending)
            )
        if active_payloads:
            renderer._queue_montage_level_stats_for_payloads(session, active_payloads)
        session.display_committed = bool(session.lifecycle.presented_tiles)
        geometry = replace(geometry, montage_tile_states=session.ensure_tile_states())
        renderer._last_montage_tile_state_publish_ms = (perf_counter() - state_start) * 1000.0
        geometry_start = perf_counter()
        renderer._sync_committed_montage_geometry(
            geometry,
            semantic_commit=tiled_payloads_include_semantics(active_payloads),
        )
        renderer._last_montage_tile_geometry_sync_ms = (perf_counter() - geometry_start) * 1000.0
        if not bool(getattr(session, "display_committed", False)):
            renderer.refresh_montage_priority_targets(session)
        overlay_start = perf_counter()
        rect = montage_rect_for_viewport(session.plan, view_range=session.view_range, viewport_shape=session.viewport_shape)
        renderer._update_montage_tile_overlays_for_plan(session.plan, tuple(session.tile_states), rect)
        renderer._last_montage_overlay_update_ms = (perf_counter() - overlay_start) * 1000.0
        self._finish_commit(report, tile_state, commit_start=commit_start)

    def _finish_commit(self, report, tile_state, *, commit_start: float) -> None:
        renderer = self.renderer
        session = self.session
        identity_start = perf_counter()
        identity_mismatches = session.backend_identity_mismatch_tiles()
        renderer._last_montage_tile_identity_check_ms = (perf_counter() - identity_start) * 1000.0
        if identity_mismatches:
            renderer._montage_identity_repair_commits = int(getattr(renderer, "_montage_identity_repair_commits", 0) or 0) + 1
            session.final_commit_pending = True
            session.flush_pending = True
        rearmed_parked = tuple(getattr(session, "rearm_visible_parked_payloads", lambda: ())() or ())
        renderer._last_montage_tile_commit_ms = (perf_counter() - commit_start) * 1000.0
        complete_inline_work(
            renderer,
            WorkItem(
                key=("montage_backend_commit", session.key, int(session.session_id), int(tile_state.revision), "tile_layer"),
                lane=WorkLane.BACKEND_COMMIT,
                frame_target=session.frame_plan.target,
                supersession_key=("montage-backend-commit", session.key),
                supersession_value=int(session.session_id),
                estimated_cpu_ms=float(renderer._last_montage_tile_commit_ms),
                estimated_bytes=int(getattr(report, "texture_upload_bytes", 0) or 0),
            ),
        )
        self._record_commit_feedback(report)
        upload_backlog = bool(
            getattr(session, "dirty_payloads", None)
            or getattr(session, "pending_removals", None)
            or getattr(session, "pending_payload_upserts", None)
            or (session.has_pending_level_update() and session.has_stale_level_presentations())
            or rearmed_parked
        )
        followup_start = perf_counter()
        session.note_committed()
        renderer._notify_file_session_montage_committed()
        if upload_backlog:
            session.final_commit_pending = True
            session.flush_pending = True
        if (
            upload_backlog
            or identity_mismatches
            or _commit_report_accepted_preview_payload(report, tile_state)
        ):
            renderer.request_montage_replan(session)
        renderer._settle_montage_visible_plan_if_complete(session)
        renderer._finish_montage_session_if_complete(session)
        if not upload_backlog:
            from arrayscope.window.montage_prefetch import schedule_near_viewport_montage_prefetch

            schedule_near_viewport_montage_prefetch(renderer, session)
        renderer._retry_live_profile_after_montage_tile()
        renderer._last_montage_tile_followup_ms = (perf_counter() - followup_start) * 1000.0

    def _record_commit_feedback(self, report) -> None:
        renderer = self.renderer
        feedback = latency_feedback(renderer)
        if feedback is None and not hasattr(renderer.win, "_record_ui_work"):
            return
        cold_count = int(getattr(report, "cold_count", 0) or 0)
        warm_count = int(getattr(report, "existing_items_shown", 0) or 0) + int(getattr(report, "relocated_tiles", 0) or 0)
        processed_count = tile_layer_commit_processed_count(report)
        texture_upload_bytes = int(getattr(report, "texture_upload_bytes", 0) or 0)
        storage_rebuilds = int(getattr(report, "storage_rebuilds", 0) or 0)
        backend_name = image_view_backend_capabilities(renderer.win.img_view).name
        details = _commit_feedback_details(renderer)
        cold_ms = 0.0
        if cold_count > 0 and storage_rebuilds <= 0:
            cold_ms = float(getattr(report, "cold_work_ms", 0.0) or 0.0) or renderer._last_montage_tile_commit_ms
            _observe_ui(renderer, "montage_cold_commit", cold_ms, cold_count, texture_upload_bytes, "texture_upload", backend_name)
        commit_feedback_ms = renderer._last_montage_tile_commit_ms
        commit_feedback_bytes = texture_upload_bytes
        vispy_backend = str(backend_name).lower() == "vispy"
        vispy_uniform_only = bool(vispy_backend and cold_count == 0 and warm_count == 0 and texture_upload_bytes == 0)
        if vispy_backend and storage_rebuilds > 0 and cold_count > 0:
            _observe_ui(renderer, "montage_layout_commit", renderer._last_montage_tile_commit_ms, processed_count, texture_upload_bytes, "presentation_layout", backend_name)
            commit_feedback_ms = 0.0
            commit_feedback_bytes = 0
        elif vispy_backend and (cold_count > 0 or vispy_uniform_only):
            _observe_ui(renderer, "montage_present_total", renderer._last_montage_tile_commit_ms, processed_count, texture_upload_bytes, "presentation_upsert", backend_name)
            commit_feedback_ms = max(0.0, renderer._last_montage_tile_commit_ms - cold_ms)
            commit_feedback_bytes = 0
            if vispy_uniform_only:
                commit_feedback_ms = 0.0
        _observe_ui(renderer, "montage_commit", commit_feedback_ms, processed_count, commit_feedback_bytes, "presentation_upsert", backend_name)
        _observe_ui(
            renderer,
            "tile_layer_commit",
            renderer._last_montage_tile_commit_ms,
            processed_count,
            max(texture_upload_bytes, commit_feedback_bytes),
            "tile_layer_commit",
            backend_name,
            details=details,
        )

    def _install_warm_residency_scheduler(self, geometry) -> None:
        view = getattr(self.renderer.win, "img_view", None)
        if view is None or not hasattr(view, "_vispy_warm_tile_scheduler"):
            return
        session = self.session
        renderer = self.renderer
        viewport_value = (
            int(getattr(session, "session_id", 0) or 0),
            int(getattr(session, "viewport_revision", 0) or 0),
            tuple(getattr(session, "visible_tile_numbers", ()) or ()),
        )

        def schedule(process) -> None:
            if not callable(process):
                return
            def cancel_pending() -> None:
                view._vispy_pending_warm_tile_payloads = {}
                view._vispy_pending_warm_tile_context = {}

            if not self._side_work_visible_settled():
                cancel_pending()
                return
            submitted_key = getattr(renderer, "_vispy_warm_residency_submitted_key", None)
            if submitted_key == viewport_value:
                cancel_pending()
                return

            def admit():
                return True

            def done(_value=None):
                if not self._side_work_visible_settled():
                    cancel_pending()
                    return
                renderer._vispy_warm_residency_submitted_key = viewport_value
                process()

            renderer.win.kernel.submit_speculative_batch(
                kind="vispy-warm-residency",
                scope=f"montage:{session.key!r}:warm-residency",
                generation=viewport_value,
                key=("vispy-warm-residency", session.key, viewport_value),
                fn=admit,
                priority=Priority.PREFETCH,
                lane=WorkLane.SPECULATIVE_RESIDENCY,
                max_items=len(getattr(view, "_vispy_pending_warm_tile_payloads", {}) or ()),
                on_done=done,
                on_stale=lambda: None,
                on_error=lambda exc: handle_ui_exception("vispy warm residency", exc),
            )

        view._vispy_warm_tile_scheduler = schedule

    def _side_work_visible_settled(self) -> bool:
        session = self.session
        return bool(
            self._session_is_current()
            and int(getattr(self.renderer.win.kernel, "visible_backlog", 0) or 0) <= 0
            and session.visible_plan_complete()
            and not getattr(session, "dirty_payloads", None)
            and not getattr(session, "pending_payload_upserts", None)
            and not getattr(session, "pending_removals", None)
            and not bool(getattr(session, "flush_pending", False))
            and not bool(getattr(session, "final_commit_pending", False))
        )

    def _empty_first_commit_can_wait(self, first_display_commit, explicit_auto, active_payloads, tile_delta) -> bool:
        session = self.session
        return bool(
            first_display_commit
            and not explicit_auto
            and not active_payloads
            and not getattr(tile_delta, "upserts", None)
            and not getattr(tile_delta, "removals", None)
            and not (session.has_pending_level_update() and session.has_stale_level_presentations())
        )

    def _empty_progressive_commit_settled(self, first_display_commit, explicit_auto, tile_delta) -> bool:
        session = self.session
        return bool(
            not first_display_commit
            and not explicit_auto
            and not getattr(tile_delta, "force_refresh", False)
            and not getattr(tile_delta, "upserts", None)
            and not getattr(tile_delta, "removals", None)
            and not bool(getattr(session, "presentation_geometry_changed", False))
            and session.visible_plan_complete()
            and not (session.has_pending_level_update() and session.has_stale_level_presentations())
        )

    def _finish_after_noop_commit(self) -> None:
        session = self.session
        self.renderer._settle_montage_visible_plan_if_complete(session)
        self.renderer._finish_montage_session_if_complete(session)
        if not (getattr(session, "dirty_payloads", None) or getattr(session, "pending_removals", None)):
            from arrayscope.window.montage_prefetch import schedule_near_viewport_montage_prefetch

            schedule_near_viewport_montage_prefetch(self.renderer, session)
        self.renderer.request_montage_replan(session)
        self.renderer._retry_live_profile_after_montage_tile()

    def _tile_for_step(self, step):
        tile_number = int(getattr(step, "tile_number", -1))
        for tile in tuple(getattr(getattr(self.session, "plan", None), "tiles", ()) or ()):
            if int(getattr(tile, "montage_index", -2)) == tile_number:
                return tile
        return None

    def _level_key_for_step(self, tile, step):
        rendered = self.session.rendered_tiles.get(int(tile.montage_index))
        if rendered is None:
            return None
        demand = self.session.lod_policy_decision.demand
        return self.session._pyramid_key_for(rendered, demand=demand, level=int(step.level))

    def _shared_claim_rung(self, *, level: int, lane) -> int:
        if WorkLane(str(lane)) == WorkLane.DISPLAY_PREVIEW:
            return int(Rung.PREVIEW)
        return int(Rung.DESIRED)

    def _claim_shared_transform_target(self, tiles, *, level: int, lane) -> bool:
        rung = self._shared_claim_rung(level=level, lane=lane)
        changed = False
        for tile in tuple(tiles or ()):
            changed = self.session.lifecycle.preview_claimed(
                int(tile.montage_index),
                rung,
                int(level),
            ) or changed
        return bool(changed)

    def _release_shared_transform_claims(self, tiles, *, level: int, lane) -> None:
        rung = self._shared_claim_rung(level=level, lane=lane)
        for tile in tuple(tiles or ()):
            self.session.lifecycle.preview_released(int(tile.montage_index), rung, int(level))

    def _intent_matches_session(self, intent) -> bool:
        return (
            intent is None
            or getattr(intent, "semantic_key", getattr(self.session, "key", None))
            == getattr(self.session, "key", None)
        )

    def _session_is_current(self, intent=None) -> bool:
        if not self._intent_matches_session(intent):
            return False
        predicate = getattr(self.renderer, "_montage_session_is_current", None)
        if callable(predicate):
            return bool(predicate(self.session))
        return True


def _shared_transform_marker(session, *, demand, level: int, tiles, shader_display: bool) -> tuple:
    """Identity of one shared reduced-display batch.

    Montage slot numbers alone are not stable content: index-window retargets
    intentionally reuse slots for different source indices.  The marker must
    therefore include the current semantic source for each covered tile, plus
    the demand fields that affect the pyramid key.
    """

    desired_level = int(getattr(demand, "desired_level", 0) or 0)
    desired_factor_xy = tuple(int(value) for value in tuple(getattr(demand, "desired_factor_xy", ()) or ()))
    acceptable_levels = tuple(int(value) for value in tuple(getattr(demand, "acceptable_levels", ()) or ()))
    tile_identity = tuple(
        (
            int(tile.montage_index),
            int(tile.source_index),
            session.tile_semantic_source_id(tile.source_index),
        )
        for tile in tuple(tiles or ())
    )
    return (
        "shared-transform",
        getattr(session, "semantic_key", None),
        int(level),
        desired_level,
        desired_factor_xy,
        acceptable_levels,
        bool(shader_display),
        tile_identity,
    )


def _commit_report_accepted_preview_payload(report, tile_state) -> bool:
    committed = getattr(report, "committed_upserts", None)
    if not committed:
        return False
    payloads = dict(getattr(tile_state, "payloads", {}) or {})
    for tile_number in tuple(committed or ()):
        payload = payloads.get(int(tile_number))
        if payload is not None and str(getattr(payload, "quality", "exact")) == "preview":
            return True
    return False


def _commit_should_queue_level_stats(renderer, *, first_display_commit: bool) -> bool:
    capabilities = image_view_backend_capabilities(renderer.win.img_view)
    if str(getattr(capabilities, "name", "")).lower() != "vispy":
        return True
    return bool(first_display_commit)



def plan_stage_fan_in_candidates(document, missing_tiles, *, cancellation_token=None) -> dict[str, object]:
    document_key = stage_document_key(document)
    groups: dict[object, dict[str, object]] = {}
    tile_stage_plans = {}
    tile_stage_candidates = {}
    for tile in tuple(missing_tiles):
        if bool(getattr(cancellation_token, "cancelled", False)):
            return {"groups": {}, "tile_stage_plans": {}, "tile_stage_candidates": {}}
        request = request_for_image(tile.view_state)
        plan = plan_slab(document, request)
        candidates = tuple(getattr(plan.region_plan, "cache_candidates", ()))
        retained = tuple(candidate for candidate in candidates if getattr(candidate, "retain", True))
        if not retained:
            continue
        candidate = retained[-1]
        key = stage_key_for_candidate(document_key, candidate)
        groups.setdefault(key, {"candidate": candidate, "tiles": [], "plan": plan})
        groups[key]["tiles"].append(tile)
        tile_stage_plans[int(tile.montage_index)] = plan
        tile_stage_candidates[int(tile.montage_index)] = candidate
    return {
        "groups": groups,
        "tile_stage_plans": tile_stage_plans,
        "tile_stage_candidates": tile_stage_candidates,
    }


def hot_cached_stage_fan_in_plan(renderer, document, missing_tiles) -> dict[str, object]:
    """Attach already-resident stage values without GUI-thread slab planning."""

    missing_tiles = tuple(missing_tiles or ())
    stage_cache = getattr(getattr(renderer.win, "operation_evaluator", None), "stage_cache", None)
    resident_items = getattr(stage_cache, "resident_items", None)
    if not missing_tiles or not callable(resident_items):
        return deferred_stage_fan_in_plan()
    document_key = stage_document_key(document)
    entries = []
    for key, value in resident_items():
        if getattr(key, "document_key", None) != document_key:
            continue
        if bool(getattr(value, "prefetch_only", False)):
            continue
        if not bool(getattr(value, "visible_reuse", True)):
            continue
        entries.append((key, value))
    if not entries:
        return deferred_stage_fan_in_plan()
    entries.sort(
        key=lambda item: (
            len(tuple(getattr(item[0], "operation_prefix", ()) or ())),
            int(getattr(item[1], "stage_index", -1) or -1),
            int(getattr(item[1], "nbytes", 0) or 0),
        ),
        reverse=True,
    )
    tile_stage_keys = {}
    stage_values = {}
    for tile in missing_tiles:
        tile_number = int(tile.montage_index)
        try:
            final_region = final_region_for_request(document.current_shape, request_for_image(tile.view_state))
        except Exception:
            continue
        for key, value in entries:
            if tuple(getattr(key, "shape", ()) or ()) != tuple(int(size) for size in document.current_shape):
                continue
            try:
                contains = region_contains(value.region, final_region, key.shape)
            except Exception:
                contains = False
            if not contains:
                continue
            tile_stage_keys[tile_number] = key
            stage_values[key] = value
            break
    if not tile_stage_keys:
        return deferred_stage_fan_in_plan()
    return {
        "tile_stage_keys": tile_stage_keys,
        "tile_stage_plans": {},
        "tile_stage_candidates": {},
        "stage_values": stage_values,
        "lead_stage_warmups": {},
        "stage_requests": [],
        "attached_stage_keys": set(),
        "waiting_indices": set(),
        "lead_direct_tiles": 0,
        "retained_stage_index": max(
            (int(getattr(value, "stage_index", -1) or -1) for value in stage_values.values()),
            default=None,
        ),
        "retained_stage_decision": "cached-hot-stage",
        "repeated_expensive_stage_per_tile": False,
    }


def build_stage_fan_in_plan(renderer, document, missing_tiles, *, existing_only: bool = False, candidate_plan=None) -> dict[str, object]:
    document_key = stage_document_key(document)
    candidate_plan = (
        plan_stage_fan_in_candidates(document, missing_tiles)
        if candidate_plan is None
        else dict(candidate_plan or {})
    )
    groups: dict[object, dict[str, object]] = dict(candidate_plan.get("groups", {}) or {})
    tile_stage_plans = dict(candidate_plan.get("tile_stage_plans", {}) or {})
    tile_stage_candidates = dict(candidate_plan.get("tile_stage_candidates", {}) or {})

    tile_stage_keys = {}
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
        if existing_only:
            stage_cache = renderer.win.operation_evaluator.stage_cache
            getter = stage_cache.get_containing if hasattr(stage_cache, "get_containing") else stage_cache.get
            value = getter(key)
            if value is not None:
                stage_values[key] = value
                for tile in tiles:
                    tile_stage_keys[int(tile.montage_index)] = key
                    tile_stage_plans[int(tile.montage_index)] = tile_stage_plans.get(int(tile.montage_index), group["plan"])
                    tile_stage_candidates[int(tile.montage_index)] = candidate
                continue
            in_flight = getattr(renderer.win.operation_evaluator.stage_materializer, "_in_flight", {})
            request = in_flight.get(key)
            if request is not None:
                attached_stage_keys.add(key)
                for tile in tiles:
                    tile_stage_keys[int(tile.montage_index)] = key
                    tile_stage_plans[int(tile.montage_index)] = tile_stage_plans.get(int(tile.montage_index), group["plan"])
                    tile_stage_candidates[int(tile.montage_index)] = candidate
                    waiting_indices.add(int(tile.montage_index))
            continue
        result = renderer.win.operation_evaluator.stage_materializer.request_stage(document_key, candidate)
        retained_stage_decision = result.decision
        if result.decision == "hit":
            stage_values[key] = result.value
            for tile in tiles:
                tile_stage_keys[int(tile.montage_index)] = key
                tile_stage_plans[int(tile.montage_index)] = tile_stage_plans.get(int(tile.montage_index), group["plan"])
                tile_stage_candidates[int(tile.montage_index)] = candidate
            continue
        if result.decision == "scheduled":
            for tile in tiles:
                tile_stage_keys[int(tile.montage_index)] = key
                tile_stage_plans[int(tile.montage_index)] = tile_stage_plans.get(int(tile.montage_index), group["plan"])
                tile_stage_candidates[int(tile.montage_index)] = candidate
                waiting_indices.add(int(tile.montage_index))
            stage_requests.append((result.request, group["plan"]))
            continue
        if result.decision == "attached":
            attached_stage_keys.add(key)
            for tile in tiles:
                tile_stage_keys[int(tile.montage_index)] = key
                tile_stage_plans[int(tile.montage_index)] = tile_stage_plans.get(int(tile.montage_index), group["plan"])
                tile_stage_candidates[int(tile.montage_index)] = candidate
                waiting_indices.add(int(tile.montage_index))
            continue
        for tile in tiles:
            tile_stage_keys.pop(int(tile.montage_index), None)
        if len(tiles) > 1:
            repeated_expensive_stage_per_tile = True
    return {
        "tile_stage_keys": tile_stage_keys,
        "tile_stage_plans": tile_stage_plans,
        "tile_stage_candidates": tile_stage_candidates,
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


def submit_deferred_stage_fan_in_plan(renderer, session, missing_tiles) -> bool:
    missing_tiles = tuple(missing_tiles or ())
    if not missing_tiles or not renderer._montage_session_is_current(session):
        return False
    kernel = getattr(renderer.win, "kernel", None)
    if kernel is None:
        return False
    if bool(getattr(session, "stage_planning_async", False)):
        return True
    session.stage_planning_deferred = True
    session.stage_planning_async = True
    session.deferred_missing_tiles = missing_tiles

    def plan(token=None):
        return plan_stage_fan_in_candidates(session.document, missing_tiles, cancellation_token=token)

    def done(candidate_plan, session_id=session.session_id, session_key=session.key):
        current = getattr(renderer, "_montage_session", None)
        if current is None or not renderer._is_current_montage_session(session_id, session_key):
            return
        if not renderer._is_current_render_generation(current.render_generation):
            return
        current.stage_planning_async = False
        current.stage_planning_deferred = False
        current.deferred_missing_tiles = ()
        stage_plan = build_stage_fan_in_plan(
            renderer,
            current.document,
            (),
            candidate_plan=candidate_plan,
        )
        merge_stage_fan_in_plan(current, stage_plan)
        if render_lod.native_missing_tile_queue_required(
            str(getattr(current, "lod_policy_mode", "")),
            getattr(getattr(current, "lod_policy_decision", None), "demand", None),
        ):
            queued = set(current.pending_tile_numbers())
            for tile in missing_tiles:
                index = int(tile.montage_index)
                if index not in queued and index not in current.rendered_tiles and index not in current.skipped_tiles:
                    current.enqueue_pending_tile(tile)
                    queued.add(index)
        submit_stage_tasks(renderer, current, stage_plan["stage_requests"])
        renderer.retarget_montage_pipeline(current)

    def stale(session_id=session.session_id, session_key=session.key):
        current = getattr(renderer, "_montage_session", None)
        if current is not None and renderer._is_current_montage_session(session_id, session_key):
            current.stage_planning_async = False

    handle = kernel.submit(
        TaskSpec(
            key=("montage-stage-plan", session.key, int(session.session_id)),
            fn=plan,
            lane=WorkLane.VISIBLE_PLANNING,
            priority=Priority.INTERACTIVE,
            scope=f"montage-stage-plan:{session.key!r}",
            supersession=Supersession(
                ("montage-stage-plan", id(renderer)),
                (session.key, int(session.session_id)),
            ),
            pass_token=True,
        ),
        on_done=done,
        on_stale=stale,
        on_error=lambda exc: (stale(), handle_ui_exception("montage stage planning", exc)),
    )
    if handle is None:
        stale()
        return False
    return True


def deferred_stage_fan_in_plan() -> dict[str, object]:
    return {
        "tile_stage_keys": {},
        "tile_stage_plans": {},
        "tile_stage_candidates": {},
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


def stage_fan_in_plan_has_existing_sources(stage_plan) -> bool:
    return bool(stage_plan.get("stage_values") or stage_plan.get("attached_stage_keys"))


def merge_stage_fan_in_plan(session, stage_plan) -> None:
    session.stage_fan_in.merge_plan(stage_plan)
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


def stage_fan_in_state(stage_plan) -> StageFanInState:
    return StageFanInState(
        tile_stage_keys=stage_plan["tile_stage_keys"],
        tile_stage_plans=stage_plan["tile_stage_plans"],
        tile_stage_candidates=stage_plan["tile_stage_candidates"],
        attached_requests=stage_plan["attached_stage_keys"],
        values=stage_plan["stage_values"],
        lead_warmups=stage_plan["lead_stage_warmups"],
    )


def attach_stage_fan_in_plan(session, stage_plan) -> None:
    session.attach_stage_fan_in(stage_fan_in_state(stage_plan))
    session.tile_compute_waiting_for_stage = len(stage_plan["waiting_indices"])
    session.stage_backed_tiles_pending = len(stage_plan["waiting_indices"])
    session.lead_direct_tiles = stage_plan["lead_direct_tiles"]
    session.retained_stage_index = stage_plan["retained_stage_index"]
    session.retained_stage_decision = stage_plan["retained_stage_decision"]
    session.repeated_expensive_stage_per_tile = stage_plan["repeated_expensive_stage_per_tile"]


def complete_deferred_stage_fan_in(renderer, session) -> bool:
    if not renderer._montage_session_is_current(session):
        return False
    if not bool(getattr(session, "stage_planning_deferred", False)):
        return False
    if bool(getattr(session, "stage_planning_async", False)):
        return False
    missing_tiles = tuple(getattr(session, "deferred_missing_tiles", ()) or ())
    if submit_deferred_stage_fan_in_plan(renderer, session, missing_tiles):
        return False
    if viewport_interaction_active(renderer):
        return False
    session.stage_planning_deferred = False
    session.deferred_missing_tiles = ()
    stage_plan_start = perf_counter()
    stage_plan = build_stage_fan_in_plan(renderer, session.document, missing_tiles)
    renderer._last_montage_stage_plan_ms = (perf_counter() - stage_plan_start) * 1000.0
    attach_stage_fan_in_plan(session, stage_plan)
    if render_lod.native_missing_tile_queue_required(
        str(getattr(session, "lod_policy_mode", "")),
        getattr(getattr(session, "lod_policy_decision", None), "demand", None),
    ):
        for tile in missing_tiles:
            session.enqueue_pending_tile(tile)
    submit_stage_tasks(renderer, session, stage_plan["stage_requests"])
    renderer.retarget_montage_pipeline(session)
    return True


def submit_stage_tasks(renderer, session, stage_requests) -> None:
    if not renderer._montage_session_is_current(session):
        return
    for request, plan in tuple(stage_requests):
        if request is None or request.key in session.stage_fan_in.active_requests:
            continue
        session.stage_fan_in.active_requests.add(request.key)

        def evaluate(token, request=request, plan=plan):
            context = renderer.win._evaluation_context(ComputeLane.STAGE, token)
            return materialize_stage_candidate_chunked(
                session.document,
                plan.region_plan,
                request.candidate,
                stage_cache=renderer.win.operation_evaluator.stage_cache,
                document_key=request.document_key,
                cancellation_token=token,
                evaluation_context=context,
                memory_policy=context.memory_policy,
                allowed_chunk_axes=stage_materialization_allowed_chunk_axes(request.candidate.shape),
            )

        def done(value, session_id=session.session_id, session_key=session.key, key=request.key):
            current = getattr(renderer, "_montage_session", None)
            renderer.win.operation_evaluator.stage_materializer.complete(key, value)
            if current is None or not renderer._is_current_montage_session(session_id, session_key):
                return
            if not renderer._is_current_render_generation(current.render_generation):
                return
            batch = current.stage_fan_in.activate_value(key, value)
            enqueue_stage_dependent_tiles(current, batch.tiles)
            # Per-completion: coalesced replan, never a direct O(tiles) one.
            renderer.request_montage_replan(current)

        def stale(key=request.key):
            renderer.win.operation_evaluator.stage_materializer.cancel(key)
            current = getattr(renderer, "_montage_session", None)
            if current is not None:
                current.stage_fan_in.active_requests.discard(key)
                release_stage_dependents_to_direct(current, key)
                renderer.request_montage_replan(current)

        def failed(exc, session_id=session.session_id, session_key=session.key, key=request.key):
            current = getattr(renderer, "_montage_session", None)
            renderer.win.operation_evaluator.stage_materializer.fail(key, exc)
            if current is None or not renderer._is_current_montage_session(session_id, session_key):
                return
            for tile_number, stage_key in tuple(current.stage_fan_in.tile_stage_keys.items()):
                tile_number = int(tile_number)
                if stage_key == key and 0 <= tile_number < len(current.plan.tiles):
                    current.stage_fan_in.tile_stage_keys.pop(tile_number, None)
                    current.mark_skipped(current.plan.tiles[tile_number])
            current.stage_fan_in.fail(key)
            show_status_message(renderer.win, f"Montage stage update failed: {exc}", timeout=4000)
            renderer.retarget_montage_pipeline(current, force_commit=True)

        handle = renderer.win.kernel.submit(
            TaskSpec(
                key=request.key,
                fn=evaluate,
                lane=WorkLane.STAGE_MATERIALIZATION,
                priority=Priority.VISIBLE_IMAGE,
                scope=f"montage:{session.key!r}",
                supersession=Supersession(
                    ("montage-stage", request.key),
                    (session.key, int(session.session_id)),
                ),
                estimated_bytes=int(getattr(request.candidate, "estimated_nbytes", 0) or 0),
                reusable=True,
                pass_token=True,
            ),
            on_done=done,
            on_error=failed,
            on_stale=stale,
            on_reuse=done,
        )
        if handle is None:
            session.stage_fan_in.active_requests.discard(request.key)
            renderer.win.operation_evaluator.stage_materializer.cancel(request.key)
            release_stage_dependents_to_direct(session, request.key)
            renderer._montage_stage_admission_declined = (
                int(getattr(renderer, "_montage_stage_admission_declined", 0) or 0) + 1
            )
            renderer.request_montage_replan(session)


def enqueue_stage_dependent_tiles(session, tile_numbers) -> int:
    """Requeue stage-backed tiles whose retained source is now usable."""

    queued = set(session.pending_tile_numbers())
    busy = (
        set(int(tile) for tile in getattr(session, "loading_tiles", ()) or ())
        | set(int(tile) for tile in getattr(session, "active_tile_requests", ()) or ())
    )
    added = 0
    for tile_number in tuple(tile_numbers or ()):
        tile_number = int(tile_number)
        if tile_number in queued or tile_number in busy:
            continue
        if tile_number in session.rendered_tiles or tile_number in session.skipped_tiles:
            continue
        if 0 <= tile_number < len(session.plan.tiles):
            if session.enqueue_pending_tile(session.plan.tiles[tile_number]):
                queued.add(tile_number)
                added += 1
    return int(added)


def rearm_ready_stage_dependents(session) -> int:
    """Keep stage-backed retained pixels from idling after their source is ready."""

    values = dict(getattr(session.stage_fan_in, "values", {}) or {})
    rearmed = 0
    for key in tuple(values):
        rearmed += release_stage_dependents_to_direct(session, key)

    bound_keys = set(getattr(session.stage_fan_in, "tile_stage_keys", {}).values())
    owned_keys = (
        set(getattr(session.stage_fan_in, "active_requests", set()) or set())
        | set(getattr(session.stage_fan_in, "attached_requests", set()) or set())
        | set(values)
    )
    for key in tuple(bound_keys - owned_keys):
        rearmed += release_stage_dependents_to_direct(session, key)
    return int(rearmed)


def release_stage_dependents_to_direct(session, key) -> int:
    queued = set(session.pending_tile_numbers())
    unbound = 0
    queued_count = 0
    for tile_number, stage_key in tuple(session.stage_fan_in.tile_stage_keys.items()):
        tile_number = int(tile_number)
        if stage_key != key:
            continue
        session.stage_fan_in.tile_stage_keys.pop(tile_number, None)
        unbound += 1
        if tile_number in session.rendered_tiles or tile_number in session.skipped_tiles:
            continue
        if 0 <= tile_number < len(session.plan.tiles) and tile_number not in queued:
            tile = session.plan.tiles[tile_number]
            session.enqueue_pending_tile(tile)
            queued.add(tile_number)
            queued_count += 1
    session.stage_fan_in.detach_unbound_requests()
    if unbound:
        session.tile_compute_waiting_for_stage = max(0, int(session.tile_compute_waiting_for_stage) - unbound)
        session.stage_backed_tiles_pending = max(0, int(session.stage_backed_tiles_pending) - unbound)
    return int(queued_count)


def montage_tile_layer_placeholder(session) -> np.ndarray:
    height, width = (max(1, int(value)) for value in session.plan.display_shape)
    if bool(getattr(session, "rgb", False)):
        base = np.zeros((1, 1, 3), dtype=np.uint8)
        return np.broadcast_to(base, (height, width, 3))
    base = np.zeros((1, 1), dtype=np.float32)
    return np.broadcast_to(base, (height, width))


def session_requested_levels(session) -> tuple[float, float] | None:
    return normalize_bounds(getattr(session.level_generation, "target_levels", None)) or normalize_bounds(
        getattr(session, "user_levels_override", None)
    )


def persistent_tile_layer_fast_drain_enabled(window, session) -> bool:
    if not bool(getattr(session, "display_committed", False)):
        return False
    return persistent_gpu_tile_residency_backend(window, session)


def direct_montage_tile_delta_commit_enabled(window, session, *, allow_uncommitted_persistent: bool = False) -> bool:
    capabilities = image_view_backend_capabilities(getattr(window.win, "img_view", None))
    if not bool(getattr(session, "display_committed", False)) and not (
        bool(allow_uncommitted_persistent) and bool(capabilities.persistent_tile_residency)
    ):
        return False
    return bool(capabilities.persistent_tile_residency or not capabilities.shader_windowing)


def persistent_tile_residency_backend(window, _session=None) -> bool:
    capabilities = image_view_backend_capabilities(getattr(window.win, "img_view", None))
    return bool(capabilities.persistent_tile_residency)


def persistent_gpu_tile_residency_backend(window, _session=None) -> bool:
    capabilities = image_view_backend_capabilities(getattr(window.win, "img_view", None))
    kind = str(getattr(capabilities, "tile_residency_kind", "none") or "none")
    return bool(capabilities.persistent_tile_residency and capabilities.shader_windowing and kind in {"gpu_atlas", "none"})


def tile_layer_upsert_limits(window, session) -> dict[str, object]:
    if persistent_gpu_tile_residency_backend(window, session):
        return _persistent_tile_upsert_limits(window, session)
    capabilities = image_view_backend_capabilities(getattr(window.win, "img_view", None))
    if not (
        not capabilities.shader_windowing
        and (
            getattr(session, "dirty_payloads", None)
            or getattr(session, "pending_payload_upserts", None)
            or getattr(session, "pending_removals", None)
            or (session.has_pending_level_update() and session.has_stale_level_presentations())
        )
    ):
        return {}
    interactive = interactive_active(window)
    decision = _commit_batch_decision(window, interactive=interactive)
    batch_limit = int(getattr(decision, "batch_limit", 0) or 0)
    byte_cap = int(getattr(decision, "byte_cap", 0) or 0)
    if batch_limit <= 0:
        feedback = latency_feedback(window)
        batch_limit = 8 if feedback is None else int(feedback.batch_limit("tile_layer_commit", interactive=interactive))
    if byte_cap <= 0:
        byte_cap = 8 * 1024 * 1024 if interactive else 32 * 1024 * 1024
    return {
        "max_upserts": max(1, int(batch_limit)),
        "max_upsert_bytes": max(1024, int(byte_cap)),
        "cold_deadline_ms": presentation_upload_control_budget_ms(window, "tile_layer_commit", decision, interactive=interactive),
        "upsert_cost_fn": pyqtgraph_payload_upload_nbytes,
        "pace_resident_retargets": True,
    }


def presentation_upload_control_budget_ms(window, channel: str, decision, *, interactive: bool) -> float:
    budget = float(getattr(decision, "budget_ms", 0.0) or 0.0)
    feedback = latency_feedback(window)
    target = 4.0 if interactive else 8.0
    if feedback is not None:
        tuning = getattr(feedback, "tuning", None)
        target = float(getattr(tuning, "target_interactive_ms" if interactive else "target_idle_ms", target))
    if budget <= 0.0:
        budget = target
    return max(float(budget), float(WARNING_THRESHOLD_MS) + float(target))


def tile_layer_commit_processed_count(report) -> int:
    cold_count = int(getattr(report, "cold_count", 0) or 0)
    warm_count = int(getattr(report, "existing_items_shown", 0) or 0) + int(getattr(report, "relocated_tiles", 0) or 0)
    acknowledged_upserts = len(tuple(getattr(report, "committed_upserts", ()) or ()))
    return max(1, cold_count + warm_count, acknowledged_upserts)


def accepted_tiled_payloads(payloads, delta, report) -> dict[int, object]:
    if report is None or delta is None:
        return {}
    accepted = report.accepted_upserts(delta)
    return {int(tile): payloads[int(tile)] for tile in accepted if int(tile) in dict(payloads or {})}


def safe_tiled_payload_geometry_retarget(previous_geometry, geometry) -> bool:
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


def tile_layer_auto_levels_wait_for_complete_source(window, session, decision_force_auto: bool, level_stats) -> bool:
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


def preview_payload_parts(preview):
    if preview is None:
        return (None, None, None, None, None, None, None)
    if len(preview) == 3:
        key, plane, histogram = preview
        return key, plane, histogram, None, None, None, None
    if len(preview) == 7:
        return preview
    raise ValueError(f"unexpected preview payload shape: {len(preview)}")


def preview_row_parts(row):
    if len(row) == 4:
        tile_number, key, plane, histogram = row
        return int(tile_number), key, plane, histogram, None, None, None, None
    if len(row) == 8:
        tile_number, key, plane, histogram, shader_mapping, texture_kind, level_data, level_stats = row
        return int(tile_number), key, plane, histogram, shader_mapping, texture_kind, level_data, level_stats
    raise ValueError(f"unexpected shared preview payload shape: {len(row)}")


def rendered_tile_nbytes(rendered) -> int:
    total = 0
    for name in ("image", "histogram_data", "semantic_data", "semantic_histogram_data", "level_data"):
        value = getattr(rendered, name, None)
        if value is not None:
            total += int(getattr(np.asarray(value), "nbytes", 0) or 0)
    return int(total)


def interactive_active(window) -> bool:
    coordinator = getattr(window.win, "render_coordinator", None)
    return bool(coordinator is not None and getattr(coordinator, "interactive_active", False) or viewport_interaction_active(window))


def viewport_interaction_active(window) -> bool:
    return bool(getattr(window.win, "_viewport_interaction_active", False))


def latency_feedback(window):
    return getattr(window.win, "latency_feedback", None)


def pyqtgraph_payload_upload_nbytes(payload) -> int:
    image = getattr(payload, "image", None)
    return max(1, 0 if image is None else int(getattr(np.asarray(image), "nbytes", 0) or 0))


def vispy_payload_upload_nbytes(payload) -> int:
    texture = getattr(payload, "texture_data", None)
    if texture is None:
        texture = getattr(payload, "image", None)
    total = 0 if texture is None else int(getattr(np.asarray(texture), "nbytes", 0) or 0)
    histogram = getattr(payload, "histogram_data", None)
    if histogram is not None and histogram is not texture:
        total += int(getattr(np.asarray(histogram), "nbytes", 0) or 0)
    return max(1, int(total))


def _persistent_tile_upsert_limits(window, session) -> dict[str, object]:
    if not persistent_gpu_tile_residency_backend(window, session):
        return {}
    interactive = interactive_active(window)
    decision = _commit_batch_decision(window, interactive=interactive)
    upload_decision = decision
    batch_limit = int(getattr(decision, "batch_limit", 0) or 0)
    upload_batch_limit = int(getattr(upload_decision, "batch_limit", 0) or 0)
    if not bool(getattr(session, "display_committed", False)) and upload_batch_limit > 0:
        batch_limit = upload_batch_limit if batch_limit <= 0 else min(int(batch_limit), int(upload_batch_limit))
    byte_cap = max(int(getattr(decision, "byte_cap", 0) or 0), int(getattr(upload_decision, "byte_cap", 0) or 0))
    if batch_limit <= 0:
        feedback = latency_feedback(window)
        batch_limit = 4 if feedback is None else int(feedback.batch_limit("montage_present_total", interactive=interactive))
    if byte_cap <= 0:
        byte_cap = 8 * 1024 * 1024 if interactive else 32 * 1024 * 1024
    return {
        "max_upserts": max(1, int(batch_limit)),
        "max_upsert_bytes": max(1024, int(byte_cap)),
        "upsert_cost_fn": vispy_payload_upload_nbytes,
        "item_free_upsert_fn": vispy_preview_payload_item_free,
        "max_item_free_upserts": preview_payload_burst_limit(batch_limit),
    }


def preview_payload_burst_limit(batch_limit: int) -> int:
    """Bound VisPy preview/floor first-fill bursts.

    Preview payloads are cheaper than exact/native uploads and should not be
    limited to the cold item cap one-by-one. They still have per-item GUI and
    atlas bookkeeping cost, so the burst is finite and follows the existing
    feedback batch size instead of introducing a second pacing system.
    """

    return max(8, min(64, max(1, int(batch_limit)) * 16))


def vispy_preview_payload_item_free(payload) -> bool:
    """Preview/floor pixels are byte-budgeted, not item-budgeted, on VisPy.

    Persistent GPU presentation pays real upload bytes for these payloads, but
    the per-item cap is the wrong limiter for a first-pixel fill made of many
    tiny reduced-LOD tiles.  Native/exact uploads still count as ordinary
    items, so refinement remains paced.
    """

    return str(getattr(payload, "quality", "exact") or "exact") == "preview"


def _commit_batch_decision(window, *, interactive: bool):
    governor = getattr(window.win, "resource_governor", None)
    decide = getattr(governor, "decide_commit_batch", None)
    if callable(decide):
        return decide(interactive=interactive)
    provider = getattr(window.win, "_gui_callback_budget_decision", None)
    if callable(provider):
        return provider("presentation_commit", interactive=interactive)
    return None


def _reset_commit_timings(renderer) -> None:
    for name in (
        "_last_montage_tile_prepare_apply_ms",
        "_last_montage_tile_prepare_stats_ms",
        "_last_montage_tile_prepare_metadata_ms",
        "_last_montage_tile_prepare_source_ms",
        "_last_montage_tile_prepare_histogram_ms",
        "_last_montage_tile_layer_apply_ms",
        "_last_montage_tile_acknowledge_ms",
        "_last_montage_tile_retained_store_ms",
        "_last_montage_tile_state_publish_ms",
        "_last_montage_tile_geometry_sync_ms",
        "_last_montage_tile_identity_check_ms",
        "_last_montage_tile_followup_ms",
    ):
        setattr(renderer, name, 0.0)


def _commit_feedback_details(renderer) -> tuple[str, ...]:
    return (
        f"payload={float(getattr(renderer, '_last_montage_tile_payload_build_ms', 0.0) or 0.0):.3f}",
        f"prepare={float(getattr(renderer, '_last_montage_tile_prepare_apply_ms', 0.0) or 0.0):.3f}",
        f"prep_stats={float(getattr(renderer, '_last_montage_tile_prepare_stats_ms', 0.0) or 0.0):.3f}",
        f"prep_meta={float(getattr(renderer, '_last_montage_tile_prepare_metadata_ms', 0.0) or 0.0):.3f}",
        f"prep_source={float(getattr(renderer, '_last_montage_tile_prepare_source_ms', 0.0) or 0.0):.3f}",
        f"prep_hist={float(getattr(renderer, '_last_montage_tile_prepare_histogram_ms', 0.0) or 0.0):.3f}",
        f"apply={float(getattr(renderer, '_last_montage_tile_layer_apply_ms', 0.0) or 0.0):.3f}",
        f"ack={float(getattr(renderer, '_last_montage_tile_acknowledge_ms', 0.0) or 0.0):.3f}",
        f"retain={float(getattr(renderer, '_last_montage_tile_retained_store_ms', 0.0) or 0.0):.3f}",
        f"state={float(getattr(renderer, '_last_montage_tile_state_publish_ms', 0.0) or 0.0):.3f}",
        f"geometry={float(getattr(renderer, '_last_montage_tile_geometry_sync_ms', 0.0) or 0.0):.3f}",
        f"overlay={float(getattr(renderer, '_last_montage_overlay_update_ms', 0.0) or 0.0):.3f}",
        f"identity={float(getattr(renderer, '_last_montage_tile_identity_check_ms', 0.0) or 0.0):.3f}",
        f"followup={float(getattr(renderer, '_last_montage_tile_followup_ms', 0.0) or 0.0):.3f}",
    )


def _observe_ui(renderer, channel, ms, count, byte_count, work_class, backend, *, details=None) -> None:
    if hasattr(renderer.win, "_record_ui_work"):
        kwargs = {"work_class": work_class, "backend": backend}
        if details is not None:
            kwargs["details"] = details
        renderer.win._record_ui_work(channel, ms, count=count, byte_count=byte_count, **kwargs)
        return
    feedback = latency_feedback(renderer)
    if feedback is not None:
        feedback.observe(channel, ms, count=count, byte_count=byte_count)


def _looks_like_shared_preview_rows(payload) -> bool:
    return bool(
        isinstance(payload, tuple)
        and payload
        and isinstance(payload[0], tuple)
        and len(payload[0]) in {4, 8}
    )


def tiled_payloads_include_semantics(payloads) -> bool:
    return any(display_tile_payload_has_semantics(payload) for payload in dict(payloads or {}).values())


def _call(target, name: str, *args, **kwargs):
    fn = getattr(target, name, None)
    if callable(fn):
        return fn(*args, **kwargs)
    return None


__all__ = [
    "MontagePipelineEffects",
    "accepted_tiled_payloads",
    "direct_montage_tile_delta_commit_enabled",
    "interactive_active",
    "montage_tile_layer_placeholder",
    "persistent_gpu_tile_residency_backend",
    "persistent_tile_layer_fast_drain_enabled",
    "persistent_tile_residency_backend",
    "preview_payload_parts",
    "preview_row_parts",
    "rendered_tile_nbytes",
    "safe_tiled_payload_geometry_retarget",
    "session_requested_levels",
    "tile_layer_commit_processed_count",
    "tile_layer_upsert_limits",
    "tiled_payloads_include_semantics",
    "viewport_interaction_active",
]
