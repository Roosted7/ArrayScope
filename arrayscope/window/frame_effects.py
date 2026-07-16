"""Montage pipeline effects that cross the GUI/backend presentation boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter, thread_time

import numpy as np
import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception
from arrayscope.core.gui_callback_budget import WARNING_THRESHOLD_MS
from arrayscope.core.trace import emit_trace
from arrayscope.core.compute_policy import ComputeLane
from arrayscope.core.window_levels import WindowLevelController
from arrayscope.kernel import Lane as WorkLane, Priority, Supersession, TaskSpec, WorkItem, complete_inline_work
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.geometry import DisplayGeometry, display_geometry_coordinates_equal
from arrayscope.display.model.commit import CommitKind, DisplayPayload, PresentationInput
from arrayscope.display.model.frame import TiledValueSource, display_tile_payload_has_semantics
from arrayscope.display.model.montage_levels import LevelEvidenceQuality
from arrayscope.display.model.presentation_generation import levels_match
from arrayscope.display.model.tile_identity import acknowledged_identity_satisfies_target, tile_ack_identity
from arrayscope.display.model.tile_priority import prioritize_tile_numbers, prioritize_tiles
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
from arrayscope.operations.regions import region_contains, region_is_full
from arrayscope.operations.slabs import plan_slab, request_for_image, stage_key_for_candidate
from arrayscope.operations.stage_fanin import StageFanInState
from arrayscope.render import effects as render_effects
from arrayscope.render import lod as render_lod
from arrayscope.render.ladder import Rung
from arrayscope.render.stages import CommitBatch, LodAdmissionScope
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.display_presenter import tile_residency_budget_bytes
from arrayscope.window.frame_session import _base_source_id
from arrayscope.window.montage_payload_cache import (
    payload_lod_matches,
    previous_tiled_payloads_by_base_source,
)


class _StaleBatchIntent:
    def __init__(self, semantic_key) -> None:
        self.semantic_key = semantic_key


_PRESENTATION_GATE_EVENT_TYPE = Qt.QtCore.QEvent.Type(
    Qt.QtCore.QEvent.registerEventType()
)
_LOW_PRIORITY_CALLBACK_EVENT_TYPE = Qt.QtCore.QEvent.Type(
    Qt.QtCore.QEvent.registerEventType()
)


class _PresentationGateEvent(Qt.QtCore.QEvent):
    def __init__(self, effects) -> None:
        super().__init__(_PRESENTATION_GATE_EVENT_TYPE)
        self.effects = effects


class _LowPriorityCallbackEvent(Qt.QtCore.QEvent):
    def __init__(self, callback) -> None:
        super().__init__(_LOW_PRIORITY_CALLBACK_EVENT_TYPE)
        self.callback = callback


class _PresentationGateReceiver(Qt.QtCore.QObject):
    """Receiver-owned low-priority continuation for one presentation turn."""

    def event(self, event) -> bool:
        if event.type() == _PRESENTATION_GATE_EVENT_TYPE:
            event.effects._on_presentation_gate()
            return True
        if event.type() == _LOW_PRIORITY_CALLBACK_EVENT_TYPE:
            event.callback()
            return True
        return super().event(event)


def _presentation_gate_receiver(renderer) -> _PresentationGateReceiver:
    receiver = getattr(renderer, "_montage_presentation_gate_receiver", None)
    if receiver is None:
        receiver = _PresentationGateReceiver(renderer)
        renderer._montage_presentation_gate_receiver = receiver
    return receiver


def _post_low_priority_callback(renderer, callback) -> None:
    Qt.QtCore.QCoreApplication.postEvent(
        _presentation_gate_receiver(renderer),
        _LowPriorityCallbackEvent(callback),
        int(Qt.QtCore.Qt.EventPriority.LowEventPriority.value),
    )


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


def _finish_presentation_commit(renderer) -> None:
    """Release the commit guard and replay camera-induced retarget intent."""

    renderer._montage_presentation_commit_active = False
    if not bool(getattr(renderer, "_frame_viewport_retarget_after_commit", False)):
        return
    renderer._frame_viewport_retarget_after_commit = False
    schedule_viewport = getattr(renderer, "_schedule_frame_viewport_update", None)
    if not callable(schedule_viewport):
        raise RuntimeError("committed extent camera change has no viewport retarget gate")
    schedule_viewport(delay_ms=1)


class FramePipelineEffects:
    """Concrete pipeline effects for one live ``FrameSession``.

    Worker-side evaluators stay in ``render.effects``. This class owns the
    GUI-thread gateway: converting ready rung payloads into session state,
    building a bounded ``DisplayTiledPresentation``, presenting through the
    shared surface contract, and feeding the backend acknowledgement back into
    ``TileLifecycle`` via ``FrameSession``.
    """

    def __init__(self, renderer, session) -> None:
        self.renderer = renderer
        self.session = session

    def _evaluation_claim(self, intent, step, tile) -> _EvaluationClaim:
        source_index_for_tile = getattr(intent, "source_index_for_tile", None)
        intent_source_index = (
            source_index_for_tile(int(step.tile_number))
            if callable(source_index_for_tile)
            else None
        )
        source_index = int(tile.source_index if intent_source_index is None else intent_source_index)
        source_id_for_tile = getattr(intent, "source_id_for_tile", None)
        intent_source_id = (
            source_id_for_tile(int(step.tile_number))
            if callable(source_id_for_tile)
            else None
        )
        return _EvaluationClaim(
            semantic_key=getattr(intent, "semantic_key", getattr(self.session, "key", None)),
            rung=int(step.rung),
            level=int(step.level),
            source_index=source_index,
            source_id=(
                self.session.tile_semantic_source_id(source_index)
                if intent_source_id is None
                else intent_source_id
            ),
        )

    def _preview_claim_identity(self, intent, tile) -> tuple[object, object]:
        """Identity one reduced-display claim by frame family and source."""

        semantic_key = getattr(intent, "semantic_key", self.session.key)
        source_id_for_tile = getattr(intent, "source_id_for_tile", None)
        source_id = (
            source_id_for_tile(int(tile.montage_index))
            if callable(source_id_for_tile)
            else None
        )
        if source_id is None:
            source_id = self.session.tile_semantic_source_id(int(tile.source_index))
        return semantic_key, source_id

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
        states = render_effects.tile_lod_states(self.session, demand, scope=scope)
        plan_tiles = {
            int(tile.montage_index): tile
            for tile in tuple(getattr(getattr(self.session, "plan", None), "tiles", ()) or ())
        }
        first_state = next(iter(states), None)
        first_tile = (
            None
            if first_state is None
            else plan_tiles.get(int(first_state.tile_number))
        )
        shared_owned = (
            {int(state.tile_number) for state in states}
            if first_tile is not None
            and self._shared_transform_owns_tile_display_target(first_tile)
            else set()
        )
        if shared_owned:
            # Non-commuting pipelines (FFT across the montage axis) have one
            # shared reduced-volume owner. Planning ordinary per-tile preview
            # rungs as well reduced every source plane independently while the
            # shared transform computed the same visible target — 60-100
            # memory-bandwidth-heavy tasks per scroll frame. The shared owner
            # supplies both first pixels and the demanded reduced target.
            states = tuple(
                replace(state, allow_preview=False)
                if int(state.tile_number) in shared_owned
                else state
                for state in states
            )
        if (
            not bool(getattr(self.session, "shader_display", False))
            and bool(getattr(self.session, "source_window_changed_pending", False))
            and _compatible_successor_payload_count(self.session) > 0
        ):
            # Preserve an already-compatible predecessor (notably a one-index
            # shift) until its few cold replacements are ready. A fully cold
            # successor has no compatible pixels to preserve and streams in
            # canonical priority order instead.
            return tuple(replace(state, allow_preview=False) for state in states)
        return states

    def shared_first_pass_barrier_pending(self, scope: LodAdmissionScope | None) -> bool:
        """Hold shared quality work behind acknowledged first-pass pixels.

        Shared transforms sit outside ``FramePipeline`` because one evaluation
        fans out to every tile.  Their planned/admitted rows therefore cannot
        prove physical coverage; only the canonical lifecycle can open the
        target pass.  Shader backends additionally wait for the first-pass
        levels which give those pixels their intended meaning.
        """

        if not self._session_is_current():
            return False
        session = self.session
        if not bool(session.required_first_pixels_presented()):
            return True
        return bool(
            getattr(session, "shader_display", False)
            and getattr(session, "first_pass_quality", None) == "preview"
            and not bool(getattr(session, "first_pass_histogram_published", False))
        )

    def prepare_rung(self, intent, step) -> bool:
        tile = self._tile_for_step(step)
        if tile is None or not self._session_is_current(intent):
            return False
        if (
            bool(getattr(self.session, "shader_display", False))
            and getattr(self.session, "first_pass_quality", None) == "preview"
            and not bool(getattr(self.session, "first_pass_histogram_published", False))
            and step.rung in (Rung.DESIRED, Rung.EXACT)
        ):
            # The preview pass is one coherent evidence/display phase.  Do not
            # let target work race its final physical acknowledgement and
            # rough histogram publication.
            return False
        tile_number = int(tile.montage_index)
        semantic_key = self._preview_claim_identity(intent, tile)
        if self._step_evaluates_reduced_display_payload(step, tile):
            if self.session.lifecycle.preview_claim_matches(
                tile_number,
                int(step.rung),
                int(step.level),
                semantic_key,
            ):
                return False
            if (
                step.rung == Rung.DESIRED
                and self.session.lifecycle.preview_claim_matches(
                    tile_number,
                    int(Rung.PREVIEW),
                    int(step.level),
                    semantic_key,
                )
            ):
                return False
            return self.session.lifecycle.preview_claimed(
                tile_number,
                int(step.rung),
                int(step.level),
                semantic_key,
            )
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
            if self.session.lifecycle.preview_claim_matches(
                tile_number,
                int(Rung.DESIRED),
                int(step.level),
                semantic_key,
            ):
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
            self.session.lifecycle.task_admitted(
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
        if int(getattr(lod, "level", 0) or 0) > int(step.level):
            return False
        # Currency is not satisfiability: a presented-but-retargeted payload
        # whose typed identity can never satisfy the tile's current lifecycle
        # target is rejected by every backend commit, so counting it as
        # coverage would deny the tile its only producer — the per-tile
        # analog of the shared-coverage starvation behind the session-148
        # stall (render.effects.payload_identity_dead).
        record = self.session.lifecycle.peek(int(tile_number))
        target = None if record is None or record.target is None else record.target.identity
        return acknowledged_identity_satisfies_target(tile_ack_identity(payload), target)

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
        desired_level = int(getattr(demand, "desired_level", 0) or 0)
        if desired_level <= 0:
            return False
        # Shared-transform ownership depends on the document pipeline, display
        # axes, session generation and demanded level—not on the scalar source
        # index of each montage tile. Capability/slab planning here used to run
        # ~60 times for every pan event (9k calls in one R8 zoom stress).
        view_state = getattr(tile, "view_state", None)
        cache_key = (
            int(getattr(self.session, "session_id", 0) or 0),
            getattr(self.session, "key", None),
            desired_level,
            tuple(getattr(view_state, "image_axes", ()) or ()),
        )
        cached = getattr(self.session, "_shared_transform_ownership_cache", None)
        if cached is not None and cached[0] == cache_key:
            return bool(cached[1])
        owns = bool(
            not render_effects.preview_pipeline_commutes_for_display_lod(self.session, tile)
            and render_effects.shared_preview_is_useful(
                self.session,
                tile,
                demand,
                upload_preview_useful=True,
            )
        )
        self.session._shared_transform_ownership_cache = (cache_key, owns)
        return owns

    def _shared_preview_claim_covers_cold_tile(self, tile_number: int) -> bool:
        if int(tile_number) in self.session.rendered_tiles:
            return False
        if self.session.display_tile_payloads.get(int(tile_number)) is not None:
            return False
        tile = self._tile_for_number(tile_number)
        if tile is None:
            return False
        return self.session.lifecycle.preview_claim_active(
            int(tile_number),
            self._preview_claim_identity(None, tile),
        )

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
        row = self.session.lifecycle.row(int(tile_number))
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
        semantic_key = self._preview_claim_identity(intent, tile)
        reduced_display_step = self._step_evaluates_reduced_display_payload(step, tile)
        if step.rung in (Rung.FLOOR, Rung.PREVIEW):
            self.session.lifecycle.preview_released(
                tile_number,
                int(step.rung),
                int(step.level),
                semantic_key,
            )
            if reduced_display_step and self._session_is_current(intent):
                self.renderer.request_montage_replan(self.session)
            return
        if step.rung == Rung.DESIRED and int(step.level) > 0:
            self.session.lifecycle.preview_released(
                tile_number,
                int(step.rung),
                int(step.level),
                semantic_key,
            )
        if step.rung == Rung.DESIRED:
            request = self.session.lifecycle.materialization_request_for(tile_number, self._level_key_for_step(tile, step))
            if request is not None:
                self.session.pending_rung_materializations._apply_release_effects(
                    self.session.pending_rung_materializations.release(request)
                )
        if step.rung in (Rung.DESIRED, Rung.EXACT):
            marker = self._evaluation_claim(intent, step, tile)
            self._release_evaluation_claim(tile_number, marker=marker, request_replan=True)
            self.session.lifecycle.task_released(tile_number, reason="dropped")
            if reduced_display_step and self._session_is_current(intent):
                # Reduced display rungs do not own an evaluation claim, so
                # _release_evaluation_claim cannot arm their retry. A current
                # rung may still be cancelled by a coalesced render/layout
                # generation; leaving only the released preview claim strands
                # the final cold edge with no producer.
                self.renderer.request_montage_replan(self.session)

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
        self.session.lifecycle.task_released(tile_number, reason="dropped")
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

        emit_trace(
            "commit_batch",
            phase="admit",
            session_id=int(getattr(self.session, "session_id", 0) or 0),
            semantic_key=batch.semantic_key,
            upserts=int(len(tuple(batch.upserts or ()))),
        )

        if batch.semantic_key is not None and batch.semantic_key != getattr(self.session, "key", batch.semantic_key):
            stale_intent = _StaleBatchIntent(batch.semantic_key)
            for row in tuple(batch.upserts or ()):
                if isinstance(row, tuple) and len(row) == 2:
                    step, _payload = row
                    self.rung_dropped(stale_intent, step)
            return
        # A worker can legitimately produce no preview (for example a
        # capability/region check that became false after planning). It is a
        # released claim, not a ready payload. Leaving the claim live makes
        # the ladder suppress every retry for that slot and can strand one
        # tile outside an otherwise complete atomic CPU successor.
        current_intent = _StaleBatchIntent(batch.semantic_key)
        for row in tuple(batch.upserts or ()):
            if isinstance(row, tuple) and len(row) == 2 and row[1] is None:
                self.rung_dropped(current_intent, row[0])
        replan_needed = self._admit_ready_payloads(batch.upserts)
        if not self._session_is_current():
            return
        if replan_needed:
            # A reduced rung can complete after a newer viewport demand has
            # made its payload inadmissible. Releasing its lifecycle claim is
            # necessary, but it also removes the only producer that caused a
            # later replan to skip this tile. An admitted FLOOR/PREVIEW also
            # unlocks the finer rung which was deliberately held behind first
            # pixels. Both transitions therefore own an explicit replan
            # wakeup; a presentation wake alone can strand the visible tile at
            # its fallback LOD when the resident backend needs no new upsert.
            self.renderer.request_montage_replan(self.session)
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
        receiver = _presentation_gate_receiver(self.renderer)
        if (
            not bool(image_view_backend_capabilities(self.renderer.win.img_view).shader_windowing)
            and bool(getattr(self.session, "display_committed", False))
        ):
            # Timer category: UI cosmetic. A short one-shot coalescer lets
            # several completed CPU-windowed tiles share one scene rebuild.
            # Without it, one worker completion caused one whole-frame commit
            # (600+ commits in a five-second scrub). Generation/current-session
            # checks remain in `_on_presentation_gate`.
            Qt.QtCore.QTimer.singleShot(
                4,
                receiver,
                lambda effects=self: effects._on_presentation_gate(),
            )
            return
        # Low priority is the fairness contract: already queued input, paint,
        # heartbeat, and kernel-delivery events run before the next bounded
        # presentation slice. The receiver is parented to the orchestrator,
        # and `_on_presentation_gate` rechecks the session generation.
        Qt.QtCore.QCoreApplication.postEvent(
            receiver,
            _PresentationGateEvent(self),
            int(Qt.QtCore.Qt.EventPriority.LowEventPriority.value),
        )

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
        visible_scope = (
            None
            if scope is None
            else frozenset(int(value) for value in tuple(getattr(scope, "visible_tile_numbers", ()) or ()))
        )
        plan_tiles = tuple(
            tile
            for tile in tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ())
            if visible_scope is None or int(tile.montage_index) in visible_scope
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
            # Materialized/admitted preview rows are not a completed first
            # pass.  Wait until every required row is backend-acknowledged
            # (and shader levels are published) before admitting the shared
            # target.  The former per-tile candidate gate let the center
            # refine while the outer preview was still black, then stranded
            # retained finer fallbacks after a scroll/LOD retarget.
            if (
                desired > 0
                and not preview_tiles
                and not self.shared_first_pass_barrier_pending(scope)
            ):
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
                exact_pass=True,
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
        tiles = prioritize_tiles(
            tuple(tiles or ()),
            context=session.tile_priority_context(),
        )
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
        semantic_key = session.key
        if not self._claim_shared_transform_target(
            tiles,
            level=level,
            lane=lane,
            semantic_key=semantic_key,
        ):
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
            if not rows or not self._session_is_current():
                self._release_shared_transform_claims(
                    tiles,
                    level=level,
                    lane=lane,
                    semantic_key=semantic_key,
                )
                return
            desired = int(getattr(demand, "desired_level", 0) or 0)
            quality = "exact" if int(level) <= desired else "preview"
            pending_rows = {
                int(row[0]): row
                for row in tuple(rows)
            }

            def admit_next() -> None:
                if not self._session_is_current():
                    self._release_shared_transform_claims(
                        tiles,
                        level=level,
                        lane=lane,
                        semantic_key=semantic_key,
                    )
                    return
                # Shared-transform pixels are independent of montage layout,
                # but their presentation order is not.  The window may have
                # settled from (for example) 24 to 22 columns while the one
                # whole-volume job was running.  Re-rank every bounded
                # fan-out slice from the session's current canonical context;
                # retaining the submission-time row order visibly filled two
                # remote regions of the successor grid.
                ordered_rows = _prioritize_shared_preview_rows(
                    session,
                    tuple(pending_rows.values()),
                )
                batch = tuple(ordered_rows[:8])
                for row in batch:
                    pending_rows.pop(int(row[0]), None)
                context = session.tile_priority_context()
                emit_trace(
                    "shared_fanout_batch",
                    level=int(level),
                    quality=quality,
                    columns=int(getattr(getattr(session, "plan", None), "columns", 0) or 0),
                    focus=getattr(context, "focus", None),
                    tiles=tuple(int(row[0]) for row in batch),
                )
                admitted = bool(
                    batch
                    and self._admit_reduced_display_payload(
                        None,
                        int(batch[0][0]),
                        batch,
                        quality=quality,
                    )
                )
                if admitted:
                    self.request_presentation()
                if pending_rows:
                    _post_low_priority_callback(renderer, admit_next)
                    return
                self._release_shared_transform_claims(
                    tiles,
                    level=level,
                    lane=lane,
                    semantic_key=semantic_key,
                )
                renderer.request_montage_replan(session)

            _post_low_priority_callback(renderer, admit_next)

        def dropped():
            self._release_shared_transform_claims(
                tiles,
                level=level,
                lane=lane,
                semantic_key=semantic_key,
            )

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
        first_pass_publication_pending = self._first_pass_publication_pending()
        level_pending = bool(session.has_pending_level_update())
        level_stale = 0
        if level_pending:
            snapshot = session.level_presentation_snapshot()
            level_stale = int(getattr(snapshot, "stale_count", 0) or 0)
        obligation_tiles = tuple(
            dict.fromkeys(
                (
                    *(int(tile) for tile in tuple(getattr(session, "dirty_payloads", ()) or ())),
                    *(int(tile) for tile in tuple(getattr(session, "pending_payload_upserts", ()) or ())),
                )
            )
        )
        display_payloads = getattr(session, "display_tile_payloads", {}) or {}
        rendered_tiles = getattr(session, "rendered_tiles", {}) or {}
        def payload_marker(tile: int):
            payload = display_payloads.get(int(tile))
            source_id = None if payload is None else getattr(payload, "source_id", None)
            try:
                return hash(source_id)
            except TypeError:
                return id(source_id)

        obligation_identities = tuple(
            (int(tile), payload_marker(int(tile)), id(rendered_tiles.get(int(tile))))
            for tile in obligation_tiles
        )
        return (
            bool(getattr(session, "flush_pending", False)),
            bool(getattr(session, "final_commit_pending", False)),
            tuple(int(tile) for tile in tuple(getattr(session, "dirty_payloads", ()) or ())),
            tuple(int(tile) for tile in tuple(getattr(session, "pending_payload_upserts", ()) or ())),
            tuple(sorted(int(tile) for tile in tuple(getattr(session, "pending_removals", ()) or ()))),
            level_pending,
            level_stale,
            first_pass_publication_pending,
            int(getattr(session, "level_revision", 0) or 0),
            len(tuple(_call(session, "backend_identity_mismatch_tiles") or ())),
            obligation_identities,
        )

    def _first_pass_publication_pending(self) -> bool:
        session = self.session
        return bool(
            getattr(session, "first_pass_quality", None) is not None
            and not bool(getattr(session, "first_pass_histogram_published", False))
            and _call(self.renderer, "_first_pass_level_evidence_complete", session)
        )

    def _rearm_if_backlog(self) -> None:
        """Re-arm the gate while a *shrinking* backlog remains.

        A commit that leaves the identical backlog signature is not making
        progress; re-arming would spin the event loop. It is either waiting
        on an external completion (level scans, evaluations — each of those
        completion paths calls ``request_presentation`` itself) or a real
        wedge, counted here as a bug report (ADR 0051: rescues hide bugs).
        """

        session = self.session
        has_backlog = bool(
            getattr(session, "flush_pending", False)
            or getattr(session, "final_commit_pending", False)
            or getattr(session, "dirty_payloads", ())
            or getattr(session, "pending_payload_upserts", ())
            or getattr(session, "pending_removals", ())
            or session.has_pending_level_update()
            or self._first_pass_publication_pending()
            or _call(session, "backend_identity_mismatch_tiles")
        )
        if not has_backlog:
            self.renderer._montage_gate_last_backlog = None
            return
        signature = self._backlog_signature()
        previous = getattr(self.renderer, "_montage_gate_last_backlog", None)
        self.renderer._montage_gate_last_backlog = signature
        if previous == signature:
            self.renderer._montage_gate_no_progress = (
                int(getattr(self.renderer, "_montage_gate_no_progress", 0) or 0) + 1
            )
            # The gate stopping is a load-bearing decision: from here on only
            # an external completion (or replan) re-arms presentation. Say so
            # in the trace — a stall whose last events are repeated
            # no-progress stops with queued upserts is a lost-wakeup proof.
            emit_trace(
                "commit_gate_no_progress",
                count=int(self.renderer._montage_gate_no_progress),
                outcome=str(getattr(self.renderer, "_last_montage_commit_outcome", "") or ""),
                pending_upserts=len(getattr(session, "pending_payload_upserts", ()) or ()),
                dirty_payloads=len(getattr(session, "dirty_payloads", ()) or ()),
                flush=bool(getattr(session, "flush_pending", False)),
                final=bool(getattr(session, "final_commit_pending", False)),
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
        # The geometry above always carries the session's montage plan, so
        # this is unconditionally a tile-layer presentation.
        return DisplayImage(data=placeholder, histogram_data=None, rgb_already_windowed=False), geometry

    # ------------------------------------------------------------------ admit

    def _admit_ready_payloads(self, rows) -> bool:
        replan_needed = False
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
                replan_needed = True
                continue
            if getattr(payload, "value", None) is not None:
                self._admit_evaluation_result(tile, payload)
                continue
            claim_identity = self._preview_claim_identity(None, tile)
            self.session.lifecycle.preview_released(
                int(step.tile_number),
                int(step.rung),
                int(step.level),
                claim_identity,
            )
            admitted = self._admit_reduced_display_payload(step, int(step.tile_number), payload)
            if not admitted or step.rung in (Rung.FLOOR, Rung.PREVIEW):
                replan_needed = True
        return bool(replan_needed)

    def _admit_materialized_rung(self, step, request) -> None:
        tile_number = int(step.tile_number)
        tile = self._tile_for_step(step)
        if tile is None:
            return
        claim_identity = self._preview_claim_identity(None, tile)
        self.session.lifecycle.preview_released(
            tile_number,
            int(step.rung),
            int(step.level),
            claim_identity,
        )
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
            # The source tile disappeared while this reusable materialization
            # was running (for example an index-window retarget). It cannot
            # build a current payload, so a new producer must be planned.
            self.renderer.request_montage_replan(self.session)
        # This is the DESIRED rung's terminal materialization. The dirty
        # payload and presentation wake below are sufficient to select and
        # acknowledge the resident level; replanning every completed tile
        # rebuilt the whole visible ladder N times during a broad zoom.

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
        self.renderer._admit_first_pass_level_evidence(session, rendered, quality="exact")
        session.mark_materialized(rendered)
        session.lifecycle.task_released(int(tile.montage_index), reason="completed")
        session.dirty_tiles.append(int(tile.montage_index))
        return rendered_tile_nbytes(rendered)

    def _admit_reduced_display_payload(self, step, tile_number: int, payload, *, quality: str | None = None) -> bool:
        session = self.session
        if not self._session_is_current():
            return False
        rows = payload if _looks_like_shared_preview_rows(payload) else ((int(tile_number), *payload),)
        quality = str(quality or ("exact" if step is not None and step.rung == Rung.DESIRED else "preview"))
        is_preview = quality == "preview"
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
                quality=quality,
            )
            if not admitted:
                continue
            admitted_any = True
            session._ensure_floor_payloads((tile_number,))
            display_payload = session.display_tile_payloads.get(int(tile_number))
            if display_payload is not None:
                rendered = self.renderer._rendered_tile_for_current_payload(
                    session,
                    int(tile_number),
                    display_payload,
                )
                if rendered is not None:
                    self.renderer._admit_first_pass_level_evidence(
                        session,
                        rendered,
                        quality=quality,
                    )
            pending_upserted = int(tile_number) in session.pending_payload_upserts
            preview_upserted = bool(
                is_preview
                and pending_upserted
                and str(getattr(session.display_tile_payloads.get(int(tile_number)), "quality", "exact")) == "preview"
            )
            visible_previews += int(preview_upserted)
            upserted = upserted or pending_upserted
        if visible_previews:
            session.lod_preview_presentations = (
                int(getattr(session, "lod_preview_presentations", 0) or 0) + int(visible_previews)
            )
        if upserted:
            session.flush_pending = True
            session.final_commit_pending = True
        return bool(admitted_any)

    # ------------------------------------------------------------------ commit

    def _note_commit_bail(self, outcome: str, *, wakeup: str, **details) -> None:
        """Every early commit return names its outcome AND its armed wakeup.

        The 2026-07-16 churn stall (dossier
        docs/redesign/stale-empty-tiles-2026-07-16.md, open follow-up) sat
        exactly in these returns: upserts stayed queued while commits ran
        and bailed silently, so the trace showed healthy replans and no
        reason. The renderer attribute feeds diagnostics, the counter feeds
        the JSONL snapshots, and the ``commit_bail`` event makes a bail loop
        visible on the first read of any stall trace.
        """

        renderer = self.renderer
        session = self.session
        renderer._last_montage_commit_outcome = str(outcome)
        counts = getattr(renderer, "_montage_commit_outcome_counts", None)
        if counts is None:
            counts = {}
            renderer._montage_commit_outcome_counts = counts
        counts[str(outcome)] = int(counts.get(str(outcome), 0)) + 1
        emit_trace(
            "commit_bail",
            outcome=str(outcome),
            wakeup=str(wakeup),
            session_id=int(getattr(session, "session_id", 0) or 0),
            pending_upserts=len(getattr(session, "pending_payload_upserts", ()) or ()),
            dirty_payloads=len(getattr(session, "dirty_payloads", ()) or ()),
            flush=bool(getattr(session, "flush_pending", False)),
            final=bool(getattr(session, "final_commit_pending", False)),
            interactive=bool(interactive_active(renderer)),
            source_window_pending=bool(getattr(session, "source_window_changed_pending", False)),
            **details,
        )

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
        renderer._last_montage_commit_outcome = "started"
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
            capabilities = image_view_backend_capabilities(renderer.win.img_view)
            cpu_backend = not bool(capabilities.shader_windowing)
            cpu_atomic_successor = bool(
                cpu_backend
                and bool(getattr(session, "source_window_changed_pending", False))
                and not session.atomic_source_successor_committed()
                and _compatible_successor_payload_count(session) > 0
            )
            predecessor_frame = getattr(renderer.win, "_committed_display_frame", None)
            predecessor_source = getattr(predecessor_frame, "value_source", None)
            shader_successor_candidate = bool(
                capabilities.shader_windowing
                and not bool(getattr(session, "display_committed", False))
                and isinstance(predecessor_source, TiledValueSource)
                and bool(getattr(predecessor_source, "payloads", None))
            )
            shader_source_successor = bool(
                capabilities.shader_windowing
                and bool(getattr(session, "source_window_changed_pending", False))
                and not session.atomic_source_successor_committed()
                and isinstance(predecessor_source, TiledValueSource)
                and bool(getattr(predecessor_source, "payloads", None))
                and _compatible_successor_payload_count(session) > 0
            )
            renderer._last_montage_atomic_source_committed_before = (
                session.atomic_source_successor_committed()
            )
            renderer._last_montage_source_window_pending_before = bool(
                getattr(session, "source_window_changed_pending", False)
            )
            renderer._last_montage_shader_source_successor = bool(
                shader_source_successor
            )
            if cpu_atomic_successor:
                lod_factor = int(session._selected_lod_factor())
                for tile_number in tuple(getattr(session, "visible_tile_numbers", ()) or ()):
                    rendered = session.rendered_tiles.get(int(tile_number))
                    if rendered is not None:
                        session._ensure_display_tile_payload(
                            int(tile_number),
                            rendered,
                            tile_source_ids,
                            lod_factor=lod_factor,
                        )
                if not _cpu_successor_payloads_ready(session):
                    session.final_commit_pending = False
                    session.flush_pending = False
                    renderer._last_montage_tile_payload_build_ms = (
                        perf_counter() - payload_start
                    ) * 1000.0
                    self._note_commit_bail("cpu-compatible-successor-wait", wakeup="replan")
                    renderer.request_montage_replan(session)
                    return
            requested_levels = session_requested_levels(session)
            if cpu_backend and requested_levels is None:
                # Automatic widget synchronization is deliberately silent: it
                # must not masquerade as user input. Register the refined
                # semantic target here, before building the CPU delta, so the
                # pixels, delta revision, and acknowledgement all describe
                # the same generation. Otherwise the widget shows successor
                # levels while level_generation remains stuck on the
                # predecessor window and all tiles stay stale forever.
                current_level_source = renderer._montage_level_source_for_session(
                    session,
                    allow_partial=False,
                )
                current_levels = normalize_bounds(
                    getattr(current_level_source, "levels", None)
                )
                if current_level_source is not None:
                    current_level_source = WindowLevelController().decide(
                        previous=getattr(session, "applied_level_source", None),
                        candidate=current_level_source,
                        explicit_auto=bool(getattr(session, "force_auto", False)),
                        mode=session.window_mode,
                    ).as_level_source()
                    current_levels = normalize_bounds(current_level_source.levels)
                    session.applied_level_source = current_level_source
                if (
                    current_level_source is not None
                    and current_levels is not None
                    and getattr(current_level_source, "semantic_key", None) == session.level_key
                    and (
                        getattr(session.level_generation, "semantic_key", None) != session.level_key
                        or not levels_match(
                            getattr(session.level_generation, "target_levels", None),
                            current_levels,
                        )
                    )
                ):
                    session.begin_level_presentation_update(current_levels)
            elif requested_levels is None:
                # Shader levels are one global physical uniform, but the
                # payload identities crossing this transaction must name the
                # same acknowledged generation. Register the provisional
                # source before building the delta, then rebind the current
                # wrappers without rebuilding or re-uploading pixels.
                current_level_source = renderer._montage_level_source_for_session(
                    session,
                    allow_partial=not bool(
                        getattr(session, "first_pass_histogram_published", False)
                    ),
                )
                if bool(getattr(session, "first_pass_histogram_published", False)):
                    current_summary = renderer._montage_level_tracker().summary_for(
                        session.level_key
                    )
                    if not bool(getattr(current_summary, "refined", False)):
                        current_level_source = None
                if current_level_source is None:
                    # Withhold an unrefined *candidate*, not the already
                    # accepted source. The histogram callback can advance the
                    # generation just before the controller retains a broader
                    # applied window. Re-target convergence to that accepted
                    # source or refinement remains blocked behind six stale
                    # uniforms with no external work left to wake them.
                    applied_source = getattr(session, "applied_level_source", None)
                    if (
                        getattr(applied_source, "semantic_key", None)
                        == session.level_key
                    ):
                        current_level_source = applied_source
                if current_level_source is not None:
                    current_level_source = WindowLevelController().decide(
                        previous=getattr(session, "applied_level_source", None),
                        candidate=current_level_source,
                        explicit_auto=bool(getattr(session, "force_auto", False)),
                        mode=session.window_mode,
                    ).as_level_source()
                    # The applied source and convergence target are separate:
                    # the target may already name these levels while a prior
                    # display decision still owns the applied source. Publish
                    # every accepted controller decision, not only revisions.
                    session.applied_level_source = current_level_source
                current_levels = normalize_bounds(
                    getattr(current_level_source, "levels", None)
                )
                if (
                    current_level_source is not None
                    and current_levels is not None
                    and getattr(current_level_source, "semantic_key", None) == session.level_key
                    and (
                        getattr(session.level_generation, "semantic_key", None) != session.level_key
                        or not levels_match(
                            getattr(session.level_generation, "target_levels", None),
                            current_levels,
                        )
                    )
                ):
                    session.begin_level_presentation_update(
                        current_levels,
                        source=current_level_source,
                    )
                if current_levels is not None:
                    session.bind_payloads_to_level_generation()
            limits = tile_layer_upsert_limits(renderer, session)
            if cpu_atomic_successor or shader_source_successor:
                # Hidden warming bounds GPU uploads; the eventual source-slot
                # handoff itself is one complete transaction and must not be
                # built once under the upload cap and then rebuilt unbounded.
                limits = {}
            cold_deadline_ms = None
            if limits:
                limits = dict(limits)
                cold_deadline_ms = limits.pop("cold_deadline_ms", None)
            renderer._last_montage_commit_dirty_before = len(getattr(session, "dirty_payloads", ()) or ())
            renderer._last_montage_commit_pending_before = len(
                getattr(session, "pending_payload_upserts", ()) or ()
            )
            renderer._last_montage_commit_presented_before = len(session.lifecycle.presented_tiles)
            prepared_atomic = getattr(session, "_atomic_prepared_transaction", None)
            prepared_atomic_current = _prepared_atomic_transaction_current(
                session,
                prepared_atomic,
            )
            renderer._last_montage_atomic_prepared_reused = bool(
                prepared_atomic_current
            )
            renderer._last_montage_atomic_fast_built = False
            renderer._last_montage_atomic_fast_reject_reason = ""
            if prepared_atomic_current:
                base_tile_state = prepared_atomic["base_tile_state"]
                tile_state = prepared_atomic["tile_state"]
                tile_delta = prepared_atomic["tile_delta"]
            else:
                session._atomic_prepared_transaction = None
                fast_atomic = (
                    session.build_atomic_source_successor_presentation()
                    if shader_source_successor
                    else None
                )
                renderer._last_montage_atomic_fast_reject_reason = str(
                    getattr(session, "_atomic_fast_reject_reason", "") or ""
                )
                if fast_atomic is not None:
                    tile_state, tile_delta = fast_atomic
                    renderer._last_montage_atomic_fast_built = True
                else:
                    tile_state, tile_delta = session.build_tile_presentation(
                        tile_source_ids,
                        cold_deadline_ms=cold_deadline_ms,
                        **limits,
                    )
            active_payloads = tile_state.active_payloads(tile_delta)
            acknowledged_payloads = dict(
                getattr(session.tile_presentation_state, "payloads", {}) or {}
            )
            lod_handoff = bool(
                capabilities.shader_windowing
                and getattr(session, "display_committed", False)
                and any(
                    previous is not None
                    and _base_source_id(getattr(previous, "source_id", None))
                    == _base_source_id(getattr(payload, "source_id", None))
                    and int(getattr(getattr(previous, "lod", None), "level", 0) or 0)
                    != int(getattr(getattr(payload, "lod", None), "level", 0) or 0)
                    for tile, payload in dict(tile_delta.upserts or {}).items()
                    for previous in (acknowledged_payloads.get(int(tile)),)
                )
            )
            if lod_handoff and interactive_active(renderer):
                session._interactive_residency_deferred = True
                session.final_commit_pending = False
                session.flush_pending = False
                self._note_commit_bail("interactive-lod-handoff-deferred", wakeup="interaction-stop-edge")
                return
            first_display_commit = not bool(session.display_committed)
            renderer._last_montage_commit_first_display = bool(first_display_commit)
            atomic_query = getattr(renderer.win.img_view, "tiledSuccessorRequiresAtomicCommit", None)
            shader_atomic_successor = bool(
                shader_source_successor
                or (
                    shader_successor_candidate
                    and callable(atomic_query)
                    and atomic_query(
                        tile_delta.upserts,
                        rgb_already_windowed=bool(display_image.rgb_already_windowed),
                    )
                )
            )
            if shader_atomic_successor and not (
                shader_source_successor or prepared_atomic_current
            ):
                # An incompatible atlas mode (scalar/complex/color) also needs
                # a coverage-complete first transaction. Unlike a known source
                # successor, this decision requires inspecting the initially
                # bounded delta, so rebuild it once without the ordinary cap.
                # A prepared transaction and source successor are already
                # complete and must not pay this O(tiles) work twice.
                tile_state, tile_delta = session.build_tile_presentation(tile_source_ids)
                active_payloads = tile_state.active_payloads(tile_delta)
            renderer._last_montage_commit_delta_upserts = len(tile_delta.upserts)
            if persistent_tile_residency_backend(renderer, session):
                upserted_tiles = set(int(tile) for tile in tile_delta.upserts)
                dirty_tiles = tuple(int(tile) for tile in dirty_tiles if int(tile) in upserted_tiles)
            if shader_atomic_successor and len(active_payloads) < len(tuple(tile_delta.active_tiles)):
                session.final_commit_pending = False
                session.flush_pending = False
                renderer._last_montage_tile_payload_build_ms = (perf_counter() - payload_start) * 1000.0
                self._note_commit_bail(
                    "shader-atomic-successor-wait",
                    wakeup="replan",
                    active_payloads=len(active_payloads),
                    active_tiles=len(tuple(tile_delta.active_tiles)),
                )
                renderer.request_montage_replan(session)
                return
            explicit_auto = bool(getattr(session, "force_auto", False) and requested_levels is None)
            if _commit_should_queue_level_stats(
                renderer,
                session,
                first_display_commit=first_display_commit,
            ):
                level_key = getattr(session, "level_key", None)
                existing_level_stats = None if level_key is None else renderer._montage_level_tracker().summary_for(level_key)
                level_stats_seeded = bool(
                    existing_level_stats is not None and existing_level_stats.source_indices
                )
                level_payloads = active_payloads if first_display_commit or not level_stats_seeded else dict(tile_delta.upserts)
                if level_payloads:
                    renderer._queue_montage_level_stats_for_payloads(session, level_payloads)
            if self._empty_first_commit_can_wait(first_display_commit, explicit_auto, active_payloads, tile_delta):
                session.final_commit_pending = False
                session.flush_pending = False
                renderer._last_montage_tile_payload_build_ms = (perf_counter() - payload_start) * 1000.0
                self._note_commit_bail("empty-first-commit-wait", wakeup="replan")
                renderer.request_montage_replan(session)
                return
            prepare_apply_start = perf_counter()
            prepare_stats_start = perf_counter()
            level_stats = renderer._montage_level_stats_for_session(session)
            renderer._last_montage_tile_prepare_stats_ms = (perf_counter() - prepare_stats_start) * 1000.0
            semantic_commit = tiled_payloads_include_semantics(active_payloads)
            # Automatic windowing applies to first-pixel payloads too.  A
            # preview may intentionally omit committed value semantics, but it
            # still needs the rough semantic source used by the shader and
            # histogram; tying levels to ``semantic_commit`` displayed those
            # first tiles at the widget fallback (0, 1).
            decision_force_auto = bool(explicit_auto and active_payloads)
            prepare_metadata_start = perf_counter()
            level_generation_semantic_key = getattr(
                getattr(session, "level_generation", None),
                "semantic_key",
                None,
            )
            semantic_level_supersession = bool(
                level_generation_semantic_key is not None
                and level_generation_semantic_key != session.level_key
            )
            metadata_can_advance = bool(
                first_display_commit
                or semantic_level_supersession
                or (
                    bool(getattr(session, "shader_display", False))
                    and not bool(getattr(session, "first_pass_histogram_published", False))
                )
                or self._side_work_visible_settled()
            )
            level_metadata_improved = bool(
                metadata_can_advance
                and (
                    semantic_level_supersession
                    or renderer._should_publish_montage_level_metadata(session, level_stats)
                )
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
            first_pass_complete = renderer._first_pass_level_evidence_complete(session)
            publish_first_pass_histogram = bool(
                first_pass_complete
                and not bool(getattr(session, "first_pass_histogram_published", False))
            )
            publish_refined_histogram = bool(
                level_metadata_improved
                and bool(getattr(level_stats, "refined", False))
            )
            publish_histogram_plot = bool(
                publish_first_pass_histogram or publish_refined_histogram
            )
            publish_metadata = publish_auto_metadata or publish_histogram_plot or level_metadata_improved
            # Instrumentation: record why a commit did/did not (re)apply level
            # metadata, so the levels/histogram-stranding path is observable from
            # runtime diagnostics without a debugger.
            renderer._last_montage_level_decision = {
                "first_display_commit": bool(first_display_commit),
                "semantic_commit": bool(semantic_commit),
                "decision_force_auto": bool(decision_force_auto),
                "explicit_auto": bool(explicit_auto),
                "metadata_can_advance": bool(metadata_can_advance),
                "semantic_level_supersession": bool(semantic_level_supersession),
                "level_metadata_improved": bool(level_metadata_improved),
                "publish_metadata": bool(publish_metadata),
                "level_stats_rank": None if level_stats is None else str(getattr(level_stats, "rank", None)),
                "level_stats_quality": int(getattr(level_stats, "evidence_quality", 0) or 0),
                "active_payload_count": len(dict(active_payloads or {})),
            }
            renderer._montage_level_decision_count = int(getattr(renderer, "_montage_level_decision_count", 0)) + 1
            renderer._last_montage_tile_prepare_metadata_ms = (perf_counter() - prepare_metadata_start) * 1000.0
            if (
                self._empty_progressive_commit_settled(first_display_commit, explicit_auto, tile_delta)
                and not publish_metadata
            ):
                session.final_commit_pending = False
                session.flush_pending = False
                renderer._last_montage_tile_payload_build_ms = (perf_counter() - payload_start) * 1000.0
                session.note_committed()
                self._note_commit_bail("empty-progressive-settled", wakeup="noop-finish")
                self._finish_after_noop_commit()
                return
            rendered_geometry = replace(rendered_geometry, montage_tile_states=session.ensure_tile_states())
            self._install_warm_residency_scheduler(rendered_geometry)
            renderer._last_montage_tile_payload_build_ms = (perf_counter() - payload_start) * 1000.0
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
            if tile_layer_first_pixels_wait_for_level_source(
                renderer,
                session,
                first_display_commit,
                level_stats,
            ):
                capabilities = image_view_backend_capabilities(renderer.win.img_view)
                publish_rough_histogram = getattr(renderer.win.img_view, "applyHistogramMetadata", None)
                if (
                    not bool(capabilities.shader_windowing)
                    and semantic_source is not None
                    and histogram_plot_data is not None
                    and callable(publish_rough_histogram)
                ):
                    publish_rough_histogram(
                        histogramData=histogram_plot_data,
                        histogramPlotData=histogram_plot_data,
                        levels=semantic_source.levels,
                        histogramRange=semantic_source.histogram_range,
                    )
                    renderer._note_montage_level_source_applied(session, semantic_source, explicit=False)
                session.final_commit_pending = True
                session.flush_pending = True
                if not getattr(session, "pending_level_tiles", None) and int(getattr(session, "level_scan_remaining_tiles", 0) or 0) <= 0:
                    renderer._mark_montage_level_scan_pending(session)
                renderer._schedule_montage_cached_level_stats(session)
                self._note_commit_bail("level-evidence-wait", wakeup="level-scan")
                return
            warm_levels = normalize_bounds(requested_levels)
            if warm_levels is None and semantic_source is not None:
                warm_levels = normalize_bounds(getattr(semantic_source, "levels", None))
            if warm_levels is None:
                warm_levels = normalize_bounds(renderer.win.img_view.getLevels())
            cpu_backend = not bool(capabilities.shader_windowing)
            atomic_successor = bool(cpu_atomic_successor or shader_source_successor)
            renderer._last_montage_atomic_source_successor = bool(atomic_successor)
            resident_predicate = getattr(renderer.win.img_view, "tiledPayloadResident", None)
            cold_gpu_successor = bool(
                not cpu_backend
                and bool(getattr(session, "display_committed", False))
                and callable(resident_predicate)
                and any(
                    not bool(resident_predicate(payload))
                    for payload in dict(tile_delta.upserts or {}).values()
                )
            )
            if cold_gpu_successor and interactive_active(renderer):
                # Keep the acknowledged pixels during the gesture. The
                # interaction-stop edge replans this exact dirty transaction;
                # hidden residency then absorbs allocation/upload cost before
                # the semantic swap reaches the canvas.
                session._interactive_residency_deferred = True
                session.final_commit_pending = False
                session.flush_pending = False
                self._note_commit_bail("interactive-residency-deferred", wakeup="interaction-stop-edge")
                return
            requires_hidden_warm = bool(atomic_successor or cold_gpu_successor)
            if (
                requires_hidden_warm
                and bool(tile_delta.upserts)
                and warm_levels is not None
                and not _warm_atomic_successor_residency(
                    renderer,
                    session,
                    rendered_geometry,
                    tile_delta,
                    levels=warm_levels,
                    rgb_already_windowed=bool(getattr(display_image, "rgb_already_windowed", False)),
                    payloads=(
                        active_payloads
                        if atomic_successor
                        else dict(tile_delta.upserts or {})
                    ),
                    batch_size=2,
                )
            ):
                session._atomic_prepared_transaction = {
                    "session_id": int(getattr(session, "session_id", 0) or 0),
                    "level_revision": int(
                        getattr(getattr(session, "level_generation", None), "revision", 0)
                        or 0
                    ),
                    "marker_kind": (
                        "cpu-compatible" if cpu_atomic_successor else "shader-source"
                    ),
                    "base_tile_state": base_tile_state,
                    "tile_state": tile_state,
                    "tile_delta": tile_delta,
                    "payload_markers": {
                        int(tile): (
                            (
                                _cpu_transaction_payload_marker(payload)
                                if cpu_atomic_successor
                                else _shader_source_transaction_payload_marker(payload)
                            )
                        )
                        for tile, payload in active_payloads.items()
                    },
                }
                # Residency preparation owns its receiver-bound continuation
                # and requests exactly one semantic replan when complete.
                # Replanning the whole frame after every warmed holder is an
                # O(tiles^2) event-loop starvation loop during scroll churn.
                session.final_commit_pending = False
                session.flush_pending = False
                self._note_commit_bail("hidden-warm-residency-wait", wakeup="warm-residency-continuation")
                return
            renderer._last_montage_tile_prepare_apply_ms = (perf_counter() - prepare_apply_start) * 1000.0
            # "Emitted" means handed across the backend boundary, not merely
            # selected by the transaction builder. In particular, the
            # refined-level gate above may defer a complete PyQtGraph frame;
            # marking it emitted before that decision strands the lifecycle
            # at TARGET_EMITTED even though no scene update occurred.
            session.lifecycle.commit_emitted(tile_delta.upserts)
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
                self._note_commit_bail("backend-declined", wakeup="rearm-if-backlog")
                return
            renderer._last_montage_commit_outcome = "backend-applied"
            session._atomic_prepared_transaction = None
            self._acknowledge_and_publish(
                tile_delta,
                tile_state,
                rendered_geometry,
                active_payloads,
                commit_start=commit_start,
                atomic_source_successor=atomic_successor,
                first_pass_histogram_published=bool(
                    publish_first_pass_histogram and histogram_plot_data is not None
                ),
            )
        finally:
            _finish_presentation_commit(renderer)

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
        renderer._last_montage_decision_levels = normalize_bounds(decision.levels)
        renderer._last_montage_decision_source_rank = int(
            getattr(getattr(decision, "applied_level_source", None), "rank", 0) or 0
        )
        set_image_start = perf_counter()
        if getattr(geometry, "montage", None) is None:
            return False
        committer = renderer._display_committer()
        committer.commit_tiled_delta(decision.display_presentation)
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
        elif bool(getattr(report, "presented_tiles", ())):
            renderer._note_display_level_source(decision)
        renderer.win.apply_axis_flips()
        renderer.win.img_view.setImageStale(False)
        return True

    def _acknowledge_and_publish(
        self,
        tile_delta,
        tile_state,
        geometry,
        active_payloads,
        *,
        commit_start: float,
        atomic_source_successor: bool,
        first_pass_histogram_published: bool,
    ) -> None:
        renderer = self.renderer
        session = self.session
        report = getattr(renderer._display_committer(), "last_tile_commit_report", None)
        renderer._last_montage_report_presented = len(tuple(getattr(report, "presented_tiles", ()) or ()))
        renderer._last_montage_report_committed = len(tuple(getattr(report, "committed_upserts", ()) or ()))
        renderer._last_montage_report_stale = bool(getattr(report, "stale", False))
        renderer._last_montage_report_delta_key = getattr(report, "delta_key", None)
        renderer._last_montage_report_generation = getattr(report, "transaction_generation", None)
        renderer._last_montage_report_acknowledges = bool(
            report is not None and report.acknowledges(tile_delta)
        )
        preview_transition = _commit_report_accepts_new_preview(
            session,
            report,
            tile_delta,
            tile_state,
        )
        acknowledge_start = perf_counter()
        committed_levels = normalize_bounds(renderer.win.img_view.getLevels())
        renderer._last_montage_physical_levels = committed_levels
        presented_before = set(session.lifecycle.presented_tiles)
        first_pixels_before = bool(session.required_first_pixels_presented())
        acknowledged = session.acknowledge_tile_presentation(tile_delta, report, levels=committed_levels)
        if atomic_source_successor:
            # A source successor is complete only after the lifecycle accepts
            # one coverage-complete backend report.  Backend submission alone
            # cannot suppress the next attempt after a stale/partial commit.
            session.acknowledge_atomic_source_successor(
                tile_delta,
                report,
                acknowledged,
            )
        renderer._last_montage_tile_acknowledge_ms = (perf_counter() - acknowledge_start) * 1000.0
        _call(renderer, "_refresh_tile_truth_overlay")
        retained_start = perf_counter()
        accepted_payloads = accepted_tiled_payloads(acknowledged.payloads, tile_delta, report)
        retention_started_at = getattr(renderer, "_slice_retention_started_at", None)
        retention_session_id = getattr(renderer, "_slice_retention_session_id", None)
        if (
            accepted_payloads
            and retention_started_at is not None
            and retention_session_id == int(session.session_id)
        ):
            replacement_ms = max(0.0, (perf_counter() - float(retention_started_at)) * 1000.0)
            renderer._slice_retention_replacements = (
                int(getattr(renderer, "_slice_retention_replacements", 0) or 0) + 1
            )
            renderer._slice_retention_last_replacement_ms = replacement_ms
            renderer._slice_retention_max_replacement_ms = max(
                float(getattr(renderer, "_slice_retention_max_replacement_ms", 0.0) or 0.0),
                replacement_ms,
            )
            renderer._slice_retention_started_at = None
            renderer._slice_retention_session_id = None
            emit_trace(
                "slice_retention_replaced",
                session_id=int(session.session_id),
                elapsed_ms=float(replacement_ms),
                accepted_tiles=tuple(sorted(int(tile) for tile in accepted_payloads)),
                cache_hits=int(getattr(session, "tile_compute_cache_hits", 0) or 0),
                stage_backed=int(getattr(session, "tile_compute_stage_backed", 0) or 0),
                uploads=int(getattr(report, "texture_uploads", 0) or 0),
            )
        renderer._retained_tiled_payload_store().remember_acknowledged(
            accepted_payloads
        )
        renderer._last_montage_tile_retained_store_ms = (perf_counter() - retained_start) * 1000.0
        state_start = perf_counter()
        presented_tiles = active_payloads if report is None else getattr(report, "presented_tiles", active_payloads)
        session.mark_presented(presented_tiles)
        presented_after = set(session.lifecycle.presented_tiles)
        first_pixels_presented = bool(session.required_first_pixels_presented())
        first_pixel_transition = bool(
            first_pixels_presented and not first_pixels_before
        )
        renderer._last_montage_ack_new_presented = len(presented_after - presented_before)
        renderer._last_montage_ack_lost_presented = len(presented_before - presented_after)
        if first_pixels_presented:
            extent_changed = renderer._publish_montage_content_extent(session.plan)
            if extent_changed:
                refresh_extent_intent = getattr(
                    renderer.win.img_view,
                    "refreshViewportContentExtentIntent",
                    None,
                )
                if callable(refresh_extent_intent):
                    if bool(refresh_extent_intent()):
                        # The range-change signal fires while the enclosing
                        # commit intentionally suppresses viewport retargeting.
                        # Preserve that semantic obligation and replay it only
                        # after acknowledgement/commit teardown, otherwise the
                        # camera shows the successor extent while the active
                        # plan remains the predecessor's partial tile set.
                        renderer._frame_viewport_retarget_after_commit = True
        if committed_levels is not None and image_view_backend_capabilities(renderer.win.img_view).shader_windowing:
            session.update_level_presentation_scope()
            session.acknowledge_uniform_level_presentation(committed_levels)
        if not session.has_stale_level_presentations():
            session.set_level_update_pending(False)
        released_display_pending = self.release_display_owned_pending()
        if released_display_pending:
            renderer._montage_display_owned_pending_released = (
                int(getattr(renderer, "_montage_display_owned_pending_released", 0) or 0)
                + int(released_display_pending)
            )
        if accepted_payloads:
            # Evidence quality can advance only after the backend accepts the
            # payload. Scan the current active population at that transition
            # so a bounded final batch can close coherent first-pass evidence.
            # Re-scanning it on reports with *no* accepted upserts created the
            # idle level-stats -> presentation feedback loop caught by
            # trace_verify.
            renderer._queue_montage_level_stats_for_payloads(session, active_payloads)
        first_pass_publication_transition = bool(
            first_pass_histogram_published
            and not bool(getattr(session, "first_pass_histogram_published", False))
        )
        if first_pass_publication_transition:
            session.first_pass_histogram_published = True
        elif (
            renderer._first_pass_level_evidence_complete(session)
            and not bool(getattr(session, "first_pass_histogram_published", False))
        ):
            # The final first-pass acknowledgement makes the rough histogram
            # eligible.  Preserve that metadata-only obligation before the
            # preview transition can replan target quality.
            session.flush_pending = True
            session.final_commit_pending = True
        session.display_committed = bool(session.lifecycle.presented_tiles)
        semantic_progress = getattr(session, "semantic_level_evidence_progress", None)
        semantic_evidence_waiting = bool(
            semantic_progress is not None
            and (
                semantic_progress.inflight_generation is not None
                or int(semantic_progress.pending_batches) > 0
            )
        )
        if first_pixel_transition or semantic_evidence_waiting:
            # Physical first-pixel completion changes DISPLAY_PREPARATION and
            # side-work eligibility without a kernel completion after this
            # acknowledgement. Refined semantic evidence has the same shape.
            # Reconcile the existing quotas at their canonical lifecycle edge
            # rather than polling or letting the preview-era quota strand the
            # exact rung planned by the follow-up replan.
            reconcile_quotas = getattr(renderer.win, "_apply_resource_governor_decisions", None)
            if callable(reconcile_quotas):
                reconcile_quotas(refresh_telemetry=False)
        if first_pixel_transition:
            # Physical first-pass closure is a planning edge.  No worker
            # completion follows the backend acknowledgement, so the quality
            # barrier needs an explicit receiver-owned replan wakeup.
            renderer.request_montage_replan(session)
        if session.visible_plan_complete():
            session.source_window_changed_pending = False
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
        self._finish_commit(
            report,
            tile_state,
            tile_delta,
            commit_start=commit_start,
            preview_transition=preview_transition,
        )
        if first_pass_publication_transition:
            renderer.request_montage_replan(session)

    def _finish_commit(
        self,
        report,
        tile_state,
        tile_delta,
        *,
        commit_start: float,
        preview_transition: bool,
    ) -> None:
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
        emit_trace(
            "commit_batch",
            phase="backend_complete",
            session_id=int(getattr(session, "session_id", 0) or 0),
            revision=int(getattr(tile_state, "revision", 0) or 0),
            elapsed_ms=float(renderer._last_montage_tile_commit_ms),
            presented_tiles=tuple(getattr(report, "presented_tiles", ()) or ()),
            committed_upserts=tuple(getattr(report, "committed_upserts", ()) or ()),
            identity_rejected=tuple(
                sorted(getattr(report, "identity_rejected_tiles", ()) or ())
            ),
            delta_upserts=tuple(int(tile) for tile in tile_delta.upserts),
            uploads=int(getattr(report, "texture_uploads", 0) or 0),
            upload_bytes=int(getattr(report, "texture_upload_bytes", 0) or 0),
            resident_rebinds=int(getattr(report, "resident_rebinds", 0) or 0),
            vertex_uploads=int(getattr(report, "vertex_uploads", 0) or 0),
            level_revision=int(
                getattr(getattr(session, "level_generation", None), "revision", 0)
                or 0
            ),
            level_target=getattr(
                getattr(session, "level_generation", None), "target_levels", None
            ),
            decision_levels=getattr(renderer, "_last_montage_decision_levels", None),
            decision_source_rank=int(
                getattr(renderer, "_last_montage_decision_source_rank", 0) or 0
            ),
            physical_levels=getattr(renderer, "_last_montage_physical_levels", None),
            stale_level_tiles=tuple(
                sorted(
                    getattr(
                        getattr(session, "level_generation", None),
                        "stale_active_tiles",
                        (),
                    )
                    or ()
                )
            ),
            atomic_source_successor_committed=(
                session.atomic_source_successor_committed()
            ),
            atomic_source_successor_generation=getattr(
                session, "atomic_source_successor_generation", None
            ),
            atomic_source_committed_before=bool(
                getattr(
                    renderer,
                    "_last_montage_atomic_source_committed_before",
                    False,
                )
            ),
            source_window_pending_before=bool(
                getattr(
                    renderer,
                    "_last_montage_source_window_pending_before",
                    False,
                )
            ),
            shader_source_successor=bool(
                getattr(renderer, "_last_montage_shader_source_successor", False)
            ),
            atomic_source_successor=bool(
                getattr(renderer, "_last_montage_atomic_source_successor", False)
            ),
            atomic_fast_built=bool(
                getattr(renderer, "_last_montage_atomic_fast_built", False)
            ),
            atomic_fast_reject_reason=str(
                getattr(renderer, "_last_montage_atomic_fast_reject_reason", "") or ""
            ),
            atomic_prepared_reused=bool(
                getattr(renderer, "_last_montage_atomic_prepared_reused", False)
            ),
            preview_transition=bool(preview_transition),
        )
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
        # Bound identical-delta re-commits (session-148 follow-up): when every
        # upsert of this commit was identity-rejected and the previous commit
        # already rejected the byte-identical delta, re-arming the flush from
        # those queued upserts only replays a guaranteed rejection at full
        # flush rate (~25 ms of geometry sync per cycle in the field trace).
        # One retry is allowed — a retarget can race the commit — and any
        # payload or target-identity change re-arms normally.  Producers own
        # recovery: dead payloads are regenerated via the replan below.
        rejected_signature = _identity_rejected_delta_signature(tile_delta, report)
        identical_rejected_recommit = bool(
            rejected_signature is not None
            and rejected_signature
            == getattr(session, "_identity_rejected_delta_signature", None)
        )
        session._identity_rejected_delta_signature = rejected_signature
        session._identity_rejected_backoff_tiles = (
            frozenset(entry[0] for entry in rejected_signature)
            if identical_rejected_recommit
            else frozenset()
        )
        if identical_rejected_recommit:
            repeats = int(getattr(session, "_identity_rejected_delta_repeats", 0) or 0) + 1
            session._identity_rejected_delta_repeats = repeats
            emit_trace(
                "identity_rejected_recommit",
                session_id=int(getattr(session, "session_id", 0) or 0),
                revision=int(getattr(tile_state, "revision", 0) or 0),
                tiles=tuple(sorted(int(tile) for tile in tile_delta.upserts)),
                repeats=repeats,
                elapsed_ms=float(renderer._last_montage_tile_commit_ms),
            )
        else:
            session._identity_rejected_delta_repeats = 0
        backlog_dirty = dict(getattr(session, "dirty_payloads", None) or {})
        backlog_pending = dict(getattr(session, "pending_payload_upserts", None) or {})
        if identical_rejected_recommit:
            rejected_tiles = {entry[0] for entry in rejected_signature}
            backlog_dirty = {
                tile: value for tile, value in backlog_dirty.items() if int(tile) not in rejected_tiles
            }
            backlog_pending = {
                tile: value for tile, value in backlog_pending.items() if int(tile) not in rejected_tiles
            }
        upload_backlog = bool(
            backlog_dirty
            or getattr(session, "pending_removals", None)
            or backlog_pending
            or (session.has_pending_level_update() and session.has_stale_level_presentations())
            or rearmed_parked
        )
        followup_start = perf_counter()
        session.note_committed()
        renderer._notify_file_session_montage_committed()
        if upload_backlog:
            session.final_commit_pending = True
            session.flush_pending = True
        # A bounded backend slice is not a new semantic target. Its remaining
        # prepared upserts are re-armed by ``_rearm_if_backlog`` after this
        # commit; sending them through the full ladder retarget was the R2
        # O(tiles * commits) storm. Only lifecycle changes that can unlock new
        # quality work need a semantic replan.
        if identity_mismatches or preview_transition or identical_rejected_recommit:
            renderer.request_montage_replan(session)
        renderer._settle_montage_visible_plan_if_complete(session)
        renderer._finish_frame_session_if_complete(session)
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
        vertex_uploads = int(getattr(report, "vertex_uploads", 0) or 0)
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
        # Atlas allocation is a one-off fixed cost and must not permanently
        # collapse the incremental batch. Vertex submission, however, occurs
        # on virtually every changed VisPy page: excluding it left the
        # controller blind to the complete 15-30 ms interaction callback and
        # held the batch at its maximum. Feed that end-to-end cost back while
        # retaining the layout-specific observation for diagnosis.
        if vispy_backend and storage_rebuilds > 0:
            _observe_ui(renderer, "montage_layout_commit", renderer._last_montage_tile_commit_ms, processed_count, texture_upload_bytes, "presentation_layout", backend_name)
            commit_feedback_ms = 0.0
            commit_feedback_bytes = 0
        elif vispy_backend and vertex_uploads > 0:
            _observe_ui(renderer, "montage_layout_commit", renderer._last_montage_tile_commit_ms, processed_count, texture_upload_bytes, "presentation_layout", backend_name)
            _observe_ui(renderer, "montage_present_total", renderer._last_montage_tile_commit_ms, processed_count, texture_upload_bytes, "presentation_upsert", backend_name)
        elif vispy_backend and cold_count > 0:
            _observe_ui(renderer, "montage_present_total", renderer._last_montage_tile_commit_ms, processed_count, texture_upload_bytes, "presentation_upsert", backend_name)
            commit_feedback_ms = max(0.0, renderer._last_montage_tile_commit_ms - cold_ms)
            commit_feedback_bytes = 0
        elif vispy_uniform_only:
            # A uniform/no-payload turn has no admission-dependent work and
            # therefore cannot teach the tile batch size. Its total callback
            # remains recorded by the outer tile-layer observation/gate.
            commit_feedback_ms = 0.0
            commit_feedback_bytes = 0
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
            tuple(sorted(int(tile) for tile in tuple(getattr(session, "visible_tile_numbers", ()) or ()))),
        )

        def current_viewport_value() -> tuple:
            return (
                int(getattr(session, "session_id", 0) or 0),
                int(getattr(session, "viewport_revision", 0) or 0),
                tuple(sorted(int(tile) for tile in tuple(getattr(session, "visible_tile_numbers", ()) or ()))),
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
                if current_viewport_value() != viewport_value or not self._side_work_visible_settled():
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
        from arrayscope.window.frame_runtime import _interactive_active

        required_settled = getattr(session, "required_target_settled", None)
        if not callable(required_settled):
            raise RuntimeError("live frame session has no required-tile owner")
        pixels_settled = bool(required_settled())
        return bool(
            self._session_is_current()
            and not _interactive_active(self.renderer)
            and int(getattr(self.renderer.win.kernel, "visible_backlog", 0) or 0) <= 0
            and pixels_settled
        )

    def _empty_first_commit_can_wait(self, first_display_commit, explicit_auto, active_payloads, tile_delta) -> bool:
        session = self.session
        return bool(
            first_display_commit
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
        self.renderer._finish_frame_session_if_complete(session)
        if not (getattr(session, "dirty_payloads", None) or getattr(session, "pending_removals", None)):
            from arrayscope.window.montage_prefetch import schedule_near_viewport_montage_prefetch

            schedule_near_viewport_montage_prefetch(self.renderer, session)
        self.renderer.request_montage_replan(session)
        self.renderer._retry_live_profile_after_montage_tile()

    def _tile_for_step(self, step):
        return self._tile_for_number(int(getattr(step, "tile_number", -1)))

    def _tile_for_number(self, tile_number: int):
        tile_number = int(tile_number)
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

    def _claim_shared_transform_target(self, tiles, *, level: int, lane, semantic_key=None) -> bool:
        rung = self._shared_claim_rung(level=level, lane=lane)
        semantic_key = self.session.key if semantic_key is None else semantic_key
        changed = False
        for tile in tuple(tiles or ()):
            claim_identity = (
                semantic_key,
                self.session.tile_semantic_source_id(int(tile.source_index)),
            )
            changed = self.session.lifecycle.preview_claimed(
                int(tile.montage_index),
                rung,
                int(level),
                claim_identity,
            ) or changed
        return bool(changed)

    def _release_shared_transform_claims(self, tiles, *, level: int, lane, semantic_key=None) -> None:
        rung = self._shared_claim_rung(level=level, lane=lane)
        semantic_key = self.session.key if semantic_key is None else semantic_key
        for tile in tuple(tiles or ()):
            claim_identity = (
                semantic_key,
                self.session.tile_semantic_source_id(int(tile.source_index)),
            )
            self.session.lifecycle.preview_released(
                int(tile.montage_index),
                rung,
                int(level),
                claim_identity,
            )

    def _intent_matches_session(self, intent) -> bool:
        return (
            intent is None
            or getattr(intent, "semantic_key", getattr(self.session, "key", None))
            == getattr(self.session, "key", None)
        )

    def _session_is_current(self, intent=None) -> bool:
        if not self._intent_matches_session(intent):
            return False
        predicate = getattr(self.renderer, "_frame_session_is_current", None)
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


def _prioritize_shared_preview_rows(session, rows) -> tuple:
    """Project current session priority onto materialized shared rows."""

    rows_by_tile = {
        int(row[0]): row
        for row in tuple(rows or ())
    }
    ordered = prioritize_tile_numbers(
        tuple(rows_by_tile),
        plan_tiles=tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ()),
        context=session.tile_priority_context(),
    )
    return tuple(rows_by_tile[int(tile)] for tile in ordered if int(tile) in rows_by_tile)


def _commit_report_accepts_new_preview(session, report, tile_delta, tile_state) -> bool:
    if report is None or not report.acknowledges(tile_delta):
        return False
    committed = report.accepted_upserts(tile_delta)
    payloads = dict(getattr(tile_state, "payloads", {}) or {})
    previously_presented = dict(
        getattr(getattr(session, "lifecycle", None), "backend_presented_identities", {})
        or {}
    )
    for tile_number in tuple(committed or ()):
        payload = payloads.get(int(tile_number))
        if payload is None or str(getattr(payload, "quality", "exact")) not in {
            "preview",
            "fallback",
        }:
            continue
        if previously_presented.get(int(tile_number)) != tile_ack_identity(payload):
            return True
    return False


def _commit_should_queue_level_stats(renderer, session, *, first_display_commit: bool) -> bool:
    capabilities = image_view_backend_capabilities(renderer.win.img_view)
    if str(getattr(capabilities, "name", "")).lower() != "vispy":
        return True
    if bool(first_display_commit):
        return True
    stats = renderer._montage_level_tracker().summary_for(session.level_key)
    return bool(stats is None or not stats.source_indices)



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
    limits = {
        "groups": groups,
        "tile_stage_plans": tile_stage_plans,
        "tile_stage_candidates": tile_stage_candidates,
    }
    return limits


def hot_cached_stage_fan_in_plan(renderer, document, missing_tiles) -> dict[str, object]:
    """Attach already-resident stage values without GUI-thread slab planning."""

    missing_tiles = tuple(missing_tiles or ())
    stage_cache = getattr(getattr(renderer.win, "operation_evaluator", None), "stage_cache", None)
    resident_items = getattr(stage_cache, "resident_items", None)
    if not missing_tiles or not callable(resident_items):
        return deferred_stage_fan_in_plan()
    profile_start = thread_time()
    document_key = stage_document_key(document)
    resident_start = thread_time()
    resident = resident_items()
    renderer._last_hot_stage_resident_cpu_ms = (thread_time() - resident_start) * 1000.0
    filter_start = thread_time()
    entries = []
    for key, value in resident:
        if getattr(key, "document_key", None) != document_key:
            continue
        if bool(getattr(value, "prefetch_only", False)):
            continue
        if not bool(getattr(value, "visible_reuse", True)):
            continue
        entries.append((key, value))
    renderer._last_hot_stage_filter_cpu_ms = (thread_time() - filter_start) * 1000.0
    if not entries:
        return deferred_stage_fan_in_plan()
    sort_start = thread_time()
    entries.sort(
        key=lambda item: (
            len(tuple(getattr(item[0], "operation_prefix", ()) or ())),
            int(getattr(item[1], "stage_index", -1) or -1),
            int(getattr(item[1], "nbytes", 0) or 0),
        ),
        reverse=True,
    )
    renderer._last_hot_stage_sort_cpu_ms = (thread_time() - sort_start) * 1000.0
    entry_signature = (
        id(document),
        tuple(sorted(id(key) for key, _value in entries)),
    )
    if getattr(renderer, "_hot_stage_match_signature", None) != entry_signature:
        renderer._hot_stage_match_signature = entry_signature
        renderer._hot_stage_match_cache = {}
    match_cache = getattr(renderer, "_hot_stage_match_cache", None)
    if match_cache is None:
        match_cache = {}
        renderer._hot_stage_match_cache = match_cache
    cache_hits = 0
    cache_misses = 0
    # Whole-volume retained stages are the common/valuable case for
    # montage-axis FFT pipelines. One such value contains every tile by
    # construction; do not perform O(tiles * cache_entries) request-region
    # planning on every interactive index-window retarget.
    for key, value in entries:
        if tuple(getattr(key, "shape", ()) or ()) != tuple(int(size) for size in document.current_shape):
            continue
        if not region_is_full(value.region):
            continue
        tile_stage_keys = {int(tile.montage_index): key for tile in missing_tiles}
        renderer._last_hot_stage_total_cpu_ms = (thread_time() - profile_start) * 1000.0
        return {
            "tile_stage_keys": tile_stage_keys,
            "tile_stage_plans": {},
            "tile_stage_candidates": {},
            "stage_values": {key: value},
            "lead_stage_warmups": {},
            "stage_requests": [],
            "attached_stage_keys": set(),
            "waiting_indices": set(),
            "lead_direct_tiles": 0,
            "retained_stage_index": int(getattr(value, "stage_index", -1) or -1),
            "retained_stage_decision": "cached-hot-full-stage",
            "repeated_expensive_stage_per_tile": False,
        }
    tile_stage_keys = {}
    stage_values = {}
    for tile in missing_tiles:
        tile_number = int(tile.montage_index)
        request = request_for_image(tile.view_state)
        axis_ranges = tuple(
            None
            if int(axis) >= len(tuple(getattr(tile.view_state, "axis_range_indices", ()) or ()))
            or tile.view_state.axis_range_indices[int(axis)] is None
            else tuple(int(value) for value in tile.view_state.axis_range_indices[int(axis)])
            for axis in request.keep_axes
        )
        # Montage window text/indices are presentation state and deliberately
        # differ on every scrub. Stage containment depends only on the slab
        # request's kept axes, slice coordinates, and explicit keep-axis
        # ranges; key the cache by those facts.
        view_key = (
            tuple(int(axis) for axis in request.keep_axes),
            tuple(int(index) for index in request.slice_indices),
            axis_ranges,
        )
        sentinel = object()
        match = match_cache.get(view_key, sentinel)
        if match is sentinel:
            cache_misses += 1
            try:
                final_region = final_region_for_request(document.current_shape, request)
            except Exception:
                match = None
            else:
                match = None
                for key, value in entries:
                    if tuple(getattr(key, "shape", ()) or ()) != tuple(int(size) for size in document.current_shape):
                        continue
                    try:
                        contains = region_contains(value.region, final_region, key.shape)
                    except Exception:
                        contains = False
                    if contains:
                        match = (key, value)
                        break
            if len(match_cache) >= 4096:
                match_cache.clear()
            match_cache[view_key] = match
        else:
            cache_hits += 1
        if match is None:
            continue
        key, value = match
        tile_stage_keys[tile_number] = key
        stage_values[key] = value
    renderer._hot_stage_match_cache_hits = int(
        getattr(renderer, "_hot_stage_match_cache_hits", 0) or 0
    ) + int(cache_hits)
    renderer._hot_stage_match_cache_misses = int(
        getattr(renderer, "_hot_stage_match_cache_misses", 0) or 0
    ) + int(cache_misses)
    renderer._last_hot_stage_total_cpu_ms = (thread_time() - profile_start) * 1000.0
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
        stage_cache = renderer.win.operation_evaluator.stage_cache
        resident_probe = getattr(stage_cache, "peek_containing_resident", None)
        resident_value = resident_probe(key) if callable(resident_probe) else None
        if resident_value is not None:
            retained_stage_decision = "hit"
            stage_values[key] = resident_value
            for tile in tiles:
                tile_stage_keys[int(tile.montage_index)] = key
                tile_stage_plans[int(tile.montage_index)] = tile_stage_plans.get(
                    int(tile.montage_index), group["plan"]
                )
                tile_stage_candidates[int(tile.montage_index)] = candidate
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
    if not missing_tiles or not renderer._frame_session_is_current(session):
        return False
    kernel = getattr(renderer.win, "kernel", None)
    if kernel is None:
        return False
    if bool(getattr(session, "stage_planning_async", False)):
        return True
    session.stage_planning_deferred = True
    session.stage_planning_async = True
    session.deferred_missing_tiles = missing_tiles
    emit_trace(
        "stage_plan_submitted",
        session_id=int(session.session_id),
        session_obj=id(session),
        missing=len(missing_tiles),
    )

    def plan(token=None):
        return plan_stage_fan_in_candidates(session.document, missing_tiles, cancellation_token=token)

    def done(candidate_plan, session_id=session.session_id, session_key=session.key):
        current = getattr(renderer, "_frame_session", None)
        if current is None or not renderer._is_current_frame_session(session_id, session_key):
            emit_trace(
                "stage_plan_done",
                decision="bailed-session",
                task_session_id=int(session_id),
                current_session_id=int(getattr(current, "session_id", -1) or -1),
                same_object=bool(current is session),
                same_key=bool(getattr(current, "key", None) == session_key),
                async_flag=bool(getattr(session, "stage_planning_async", False)),
            )
            return
        # Session currency has exactly one owner: the ``(session_id, key)``
        # check above.  This callback used to ALSO validate the session's
        # ``render_generation`` stamp against the renderer's global counter —
        # but that counter advances on every render request while the stamp
        # only refreshes on session build/retarget, so after any unrelated
        # repaint the stamp could never match again.  First that bail leaked
        # the async flag (phantom planner, field stall 2026-07-16 09:14);
        # then the flag-clearing repair turned it into a discard/resubmit
        # livelock that starved the deferred tiles of producers forever
        # (churn harness 2026-07-16: 5,200 plan computations, 46 pending
        # tiles, kernel idle).  A completed plan for the current session
        # applies unconditionally.
        current.stage_planning_async = False
        current.stage_planning_deferred = False
        current.deferred_missing_tiles = ()
        emit_trace(
            "stage_plan_done",
            decision="applied",
            task_session_id=int(session_id),
            missing=len(missing_tiles),
        )
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
        renderer.retarget_frame_pipeline(current)

    def stale(session_id=session.session_id, session_key=session.key):
        current = getattr(renderer, "_frame_session", None)
        cleared = bool(current is not None and renderer._is_current_frame_session(session_id, session_key))
        if cleared:
            current.stage_planning_async = False
        emit_trace(
            "stage_plan_stale",
            cleared=cleared,
            task_session_id=int(session_id),
            same_object=bool(current is session),
            async_flag=bool(getattr(session, "stage_planning_async", False)),
        )

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
    if not renderer._frame_session_is_current(session):
        return False
    if not bool(getattr(session, "stage_planning_deferred", False)):
        return False
    if bool(getattr(session, "stage_planning_async", False)):
        # Deferring to an in-flight async planner. If no planner task exists
        # this is the phantom-flag stall shape; the repeated event with a
        # growing count is the trace signature.
        emit_trace(
            "stage_plan_async_defer",
            session_id=int(getattr(session, "session_id", 0) or 0),
            pending=len(tuple(getattr(session, "pending_tiles", ()) or ())),
            deferred_missing=len(tuple(getattr(session, "deferred_missing_tiles", ()) or ())),
        )
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
    renderer.retarget_frame_pipeline(session)
    return True


def submit_stage_tasks(renderer, session, stage_requests) -> None:
    if not renderer._frame_session_is_current(session):
        return
    for request, plan in tuple(stage_requests):
        if request is None or request.key in session.stage_fan_in.active_requests:
            continue
        session.stage_fan_in.active_requests.add(request.key)
        scheduling_rank = _stage_consumer_scheduling_rank(session, request.key)

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
            current = getattr(renderer, "_frame_session", None)
            renderer.win.operation_evaluator.stage_materializer.complete(key, value)
            if current is None or not renderer._is_current_frame_session(session_id, session_key):
                return
            # No render-generation bail here either: the stage value is
            # session-current by the check above, and the stale-stamp proxy
            # (see submit_deferred_stage_fan_in_plan.done) silently skipped
            # both activation and the replan wakeup after any unrelated
            # repaint, parking the waiting tiles until the next replan.
            batch = current.stage_fan_in.activate_value(key, value)
            enqueue_stage_dependent_tiles(current, batch.tiles)
            # Per-completion: coalesced replan, never a direct O(tiles) one.
            renderer.request_montage_replan(current)

        def stale(key=request.key):
            renderer.win.operation_evaluator.stage_materializer.cancel(key)
            current = getattr(renderer, "_frame_session", None)
            if current is not None:
                current.stage_fan_in.active_requests.discard(key)
                release_stage_dependents_to_direct(current, key)
                renderer.request_montage_replan(current)

        def failed(exc, session_id=session.session_id, session_key=session.key, key=request.key):
            current = getattr(renderer, "_frame_session", None)
            renderer.win.operation_evaluator.stage_materializer.fail(key, exc)
            if current is None or not renderer._is_current_frame_session(session_id, session_key):
                return
            for tile_number, stage_key in tuple(current.stage_fan_in.tile_stage_keys.items()):
                tile_number = int(tile_number)
                if stage_key == key and 0 <= tile_number < len(current.plan.tiles):
                    current.stage_fan_in.tile_stage_keys.pop(tile_number, None)
                    current.mark_skipped(current.plan.tiles[tile_number])
            current.stage_fan_in.fail(key)
            show_status_message(renderer.win, f"Montage stage update failed: {exc}", timeout=4000)
            renderer.retarget_frame_pipeline(current, force_commit=True)

        handle = renderer.win.kernel.submit(
            TaskSpec(
                key=request.key,
                fn=evaluate,
                lane=WorkLane.STAGE_MATERIALIZATION,
                priority=Priority.VISIBLE_IMAGE,
                scheduling_rank=scheduling_rank,
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


def _stage_consumer_scheduling_rank(session, stage_key) -> int:
    """Return the best canonical tile rank among one stage's consumers."""

    from arrayscope.kernel import UNRANKED_SCHEDULING_RANK

    consumers = {
        int(tile_number)
        for tile_number, bound_key in dict(session.stage_fan_in.tile_stage_keys).items()
        if bound_key == stage_key
    }
    if not consumers:
        return int(UNRANKED_SCHEDULING_RANK)
    ordered = session._prioritized_tile_numbers(range(len(tuple(session.plan.tiles))))
    ranks = {
        int(tile_number): int(rank)
        for rank, tile_number in enumerate(ordered)
    }
    return min(
        (ranks[tile_number] for tile_number in consumers if tile_number in ranks),
        default=int(UNRANKED_SCHEDULING_RANK),
    )


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
    """Return explicit user intent, never an automatic convergence target.

    ``level_generation.target_levels`` is an acknowledgement detail. Feeding
    an early automatic target back as ``user_levels`` froze the display at
    partial-source bounds while the semantic histogram later expanded to the
    full frame (the GPU identity ramp saturated after tile 8). User intent has
    one owner: ``user_levels_override``.
    """

    return normalize_bounds(getattr(session, "user_levels_override", None))


def _compatible_successor_payload_count(session) -> int:
    plan_tiles = {
        int(tile.montage_index): tile
        for tile in tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ())
    }
    return sum(
        1
        for tile_number, payload in dict(
            getattr(session, "display_tile_payloads", {}) or {}
        ).items()
        if int(tile_number) in plan_tiles
        and int(getattr(payload, "source_index", -1))
        == int(plan_tiles[int(tile_number)].source_index)
    )


def _cpu_successor_payloads_ready(session) -> bool:
    required = set(int(tile) for tile in session.required_tile_numbers())
    required.difference_update(int(tile) for tile in getattr(session, "skipped_tiles", ()) or ())
    plan_tiles = {
        int(tile.montage_index): tile
        for tile in tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ())
    }
    payloads = dict(getattr(session, "display_tile_payloads", {}) or {})
    return bool(required) and all(
        int(tile_number) in payloads
        and int(getattr(payloads[int(tile_number)], "source_index", -1))
        == int(plan_tiles[int(tile_number)].source_index)
        for tile_number in required
    )


def _cpu_transaction_payload_marker(payload) -> tuple:
    lod = getattr(payload, "lod", None)
    return (
        getattr(payload, "source_id", None),
        int(getattr(payload, "source_index", -1)),
        str(getattr(payload, "quality", "exact") or "exact"),
        int(getattr(lod, "level", 0) or 0),
        id(getattr(payload, "image", None)),
    )


def _shader_source_transaction_payload_marker(payload) -> tuple:
    """Stable source-window marker that deliberately ignores LOD upgrades."""

    return (
        _base_source_id(getattr(payload, "source_id", None)),
        int(getattr(payload, "source_index", -1)),
    )


def _prepared_atomic_transaction_current(session, prepared) -> bool:
    if not isinstance(prepared, dict):
        return False
    if int(prepared.get("session_id", -1)) != int(getattr(session, "session_id", -2)):
        return False
    if int(prepared.get("level_revision", -1)) != int(
        getattr(getattr(session, "level_generation", None), "revision", -2)
    ):
        return False
    delta = prepared.get("tile_delta")
    if delta is None or int(getattr(delta, "base_revision", -1)) != int(
        getattr(getattr(session, "tile_presentation_state", None), "revision", -2)
    ):
        return False
    planned = set(int(tile) for tile in getattr(session, "visible_tile_numbers", ()) or ())
    planned.difference_update(int(tile) for tile in getattr(session, "skipped_tiles", ()) or ())
    if set(int(tile) for tile in getattr(delta, "active_tiles", ()) or ()) != planned:
        return False
    markers = dict(prepared.get("payload_markers", {}) or {})
    payloads = dict(getattr(session, "display_tile_payloads", {}) or {})
    marker_kind = prepared.get("marker_kind")
    if marker_kind == "shader-source":
        marker_fn = _shader_source_transaction_payload_marker
    elif marker_kind == "cpu-compatible":
        marker_fn = _cpu_transaction_payload_marker
    else:
        return False
    return bool(markers) and all(
        int(tile) in payloads
        and marker_fn(payloads[int(tile)]) == marker
        for tile, marker in markers.items()
    )


def _warm_atomic_successor_residency(
    renderer,
    session,
    geometry,
    tile_delta,
    *,
    levels: tuple[float, float],
    rgb_already_windowed: bool,
    payloads=None,
    batch_size: int = 1,
) -> bool:
    """Prepare one bounded hidden residency batch before an atomic frame swap.

    Returns true only when every current upsert identity is already warm. A
    receiver-bound continuation prepares the outstanding holders without
    re-entering semantic planning, then requests one final replan. The extra
    callback boundary prevents the last warm batch and the semantic swap from
    accidentally combining into one over-budget GUI callback.
    """

    warm = getattr(getattr(renderer.win, "img_view", None), "warmTiledResidency", None)
    if not callable(warm):
        return True
    payloads = dict(payloads or getattr(tile_delta, "upserts", {}) or {})
    if not payloads:
        return True
    level_key = (float(levels[0]), float(levels[1]))
    warmed = getattr(session, "_atomic_warmed_payloads", None)
    if warmed is None:
        warmed = {}
        session._atomic_warmed_payloads = warmed
    warmed_identities = getattr(session, "_atomic_warmed_identities", None)
    if warmed_identities is None:
        warmed_identities = set(warmed.values())
        session._atomic_warmed_identities = warmed_identities
    pending = tuple(
        int(tile)
        for tile, payload in payloads.items()
        if (getattr(payload, "source_id", None), level_key) not in warmed_identities
    )
    if not pending:
        return True
    signature = (
        int(getattr(session, "session_id", 0) or 0),
        getattr(session, "key", None),
        int(getattr(session, "viewport_revision", 0) or 0),
        level_key,
        tuple((int(tile), id(payloads[int(tile)])) for tile in sorted(payloads)),
    )
    active_job = getattr(session, "_atomic_warm_job", None)
    if active_job is not None and active_job.get("signature") == signature:
        return False
    job = {
        "signature": signature,
        "pending": list(pending),
        "payloads": payloads,
    }
    session._atomic_warm_job = job

    def continue_warm() -> None:
        if getattr(session, "_atomic_warm_job", None) is not job:
            return
        if not renderer._frame_session_is_current(session):
            session._atomic_warm_job = None
            return
        current_signature = (
            int(getattr(session, "session_id", 0) or 0),
            getattr(session, "key", None),
            int(getattr(session, "viewport_revision", 0) or 0),
        )
        if current_signature != signature[:3]:
            session._atomic_warm_job = None
            return
        admitted = tuple(job["pending"][: max(1, int(batch_size))])
        del job["pending"][: len(admitted)]
        batch = {int(tile): job["payloads"][int(tile)] for tile in admitted}
        warm(
            payloads=batch,
            geometry=geometry,
            levels=level_key,
            rgb_already_windowed=bool(rgb_already_windowed),
            tile_delta=tile_delta,
            tile_residency_budget_bytes=tile_residency_budget_bytes(renderer._memory_policy()),
            frame_plan=getattr(session, "frame_plan", None),
        )
        for tile in admitted:
            payload = job["payloads"][int(tile)]
            marker = (getattr(payload, "source_id", None), level_key)
            warmed[int(tile)] = marker
            warmed_identities.add(marker)
        if job["pending"]:
            _post_low_priority_callback(renderer, continue_warm)
            return
        session._atomic_warm_job = None
        session.final_commit_pending = True
        session.flush_pending = True
        renderer.request_montage_replan(session)

    _post_low_priority_callback(renderer, continue_warm)
    return False


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
    if not capabilities.shader_windowing and not bool(getattr(session, "display_committed", False)):
        # PyQtGraph cannot show a partially windowed first frame: unlike the
        # shader backend, every first-pixel tile must already carry refined
        # CPU levels. Build and acknowledge that one complete transaction;
        # capping it creates TARGET_EMITTED subsets that no backend commit may
        # accept and therefore strands the first frame at zero pixels.
        return {}
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
    if interactive:
        batch_limit = min(8, max(4, int(batch_limit)))
    if byte_cap <= 0:
        byte_cap = 8 * 1024 * 1024 if interactive else 32 * 1024 * 1024
    limits = {
        "max_upserts": max(1, int(batch_limit)),
        "max_upsert_bytes": max(1024, int(byte_cap)),
        "cold_deadline_ms": presentation_upload_control_budget_ms(window, "tile_layer_commit", decision, interactive=interactive),
        "upsert_cost_fn": pyqtgraph_payload_upload_nbytes,
        "pace_resident_retargets": True,
    }
    return limits


def presentation_upload_control_budget_ms(window, channel: str, decision, *, interactive: bool) -> float:
    budget = float(getattr(decision, "budget_ms", 0.0) or 0.0)
    feedback = latency_feedback(window)
    target = 4.0 if interactive else 8.0
    if feedback is not None:
        tuning = getattr(feedback, "tuning", None)
        target = float(getattr(tuning, "target_interactive_ms" if interactive else "target_idle_ms", target))
    if budget <= 0.0:
        budget = target
    # This deadline bounds only the backend's cold upsert walk; payload build,
    # acknowledgement, state publication, and paint still run in the same GUI
    # callback.  Granting WARNING_THRESHOLD_MS + target to that inner walk made
    # over-budget commits the intended steady state. Keep a four-millisecond
    # margin beneath the hard callback warning for the surrounding work.
    safe_inner_budget = max(0.25, float(WARNING_THRESHOLD_MS) - 4.0)
    return max(0.25, min(float(budget), safe_inner_budget))


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


def tile_layer_first_pixels_wait_for_level_source(
    window,
    session,
    first_display_commit: bool,
    level_stats,
) -> bool:
    """Whether first successor pixels still lack backend-required evidence.

    Explicit user levels change the window choice, not the semantic histogram
    source. Every first tiled frame still needs rough evidence for a shader
    backend and refined/full evidence for the CPU-windowed backend before its
    pixels can be acknowledged.
    """

    if not bool(first_display_commit):
        return False
    shader_windowing = bool(image_view_backend_capabilities(window.win.img_view).shader_windowing)
    has_rough_source = bool(
        level_stats is not None
        and getattr(level_stats, "bounds", None) is not None
        and getattr(level_stats, "source_indices", None)
        and int(getattr(level_stats, "evidence_quality", 0) or 0) >= int(LevelEvidenceQuality.ROUGH_PREVIEW)
    )
    if shader_windowing:
        return not has_rough_source
    if bool(
        has_rough_source
        and bool(getattr(level_stats, "refined", False))
        and getattr(level_stats, "rank", None) == LevelSourceRank.MONTAGE_SAMPLED_FULL
    ):
        return False
    return True


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


def _identity_rejected_delta_signature(tile_delta, report):
    """Signature of a delta whose upserts were ALL identity-rejected.

    Returns ``None`` unless the report proves every upsert in the delta was
    refused at the identity gate with nothing committed.  The signature pairs
    each payload's acknowledgement identity with the target identity it was
    judged against, so replacing the payload OR retargeting the tile both
    change it and re-arm normal commits.
    """

    upserts = dict(getattr(tile_delta, "upserts", {}) or {})
    if not upserts:
        return None
    if tuple(getattr(report, "committed_upserts", ()) or ()):
        return None
    rejected = {int(tile) for tile in (getattr(report, "identity_rejected_tiles", ()) or ())}
    if not rejected.issuperset(int(tile) for tile in upserts):
        return None
    targets = dict(getattr(tile_delta, "target_identities", {}) or {})
    return tuple(
        (int(tile), tile_ack_identity(payload), targets.get(int(tile)))
        for tile, payload in sorted(upserts.items(), key=lambda item: int(item[0]))
    )


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
    if interactive:
        # One early shader/driver realization must not collapse the whole
        # gesture to one cold tile per commit. Bytes remain capped separately;
        # 4-8 uploads amortize full-plan/backend overhead while preserving the
        # callback target on the reference 60-tile workflow.
        batch_limit = min(8, max(4, int(batch_limit)))
    else:
        # A tiled VisPy transaction has a measured fixed cost even when one
        # small texture is uploaded. Treating that entire transaction as the
        # cost of one item makes feedback collapse an idle drain to one tile
        # and repeats planning/presentation hundreds of times. The byte cap
        # remains authoritative for large textures; four is only the minimum
        # item cohort when the byte budget admits it.
        batch_limit = max(4, int(batch_limit))
    if byte_cap <= 0:
        byte_cap = 8 * 1024 * 1024 if interactive else 32 * 1024 * 1024
    limits: dict[str, object] = {
        "max_upserts": max(1, int(batch_limit)),
        "max_upsert_bytes": max(1024, int(byte_cap)),
        "upsert_cost_fn": vispy_payload_upload_nbytes,
        # Zero-upload remaps bypass the cold item cap, but they are not
        # literally free: page geometry and lifecycle publication still scale
        # with count. Bound that separate work class without letting cold
        # feedback collapse it to one tile.
        "max_free_retargets": 8 if interactive else 12,
        # Resident swaps still publish identities and page geometry. Pace
        # them separately from cold uploads so a broad LOD transition yields
        # between bounded physical commits.
        "pace_resident_retargets": True,
    }
    resident = getattr(getattr(window.win, "img_view", None), "tiledPayloadResident", None)
    if callable(resident):
        limits["item_free_upsert_fn"] = resident
        limits["max_item_free_upserts"] = 8 if interactive else 12
    return limits


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
    renderer._last_montage_atomic_prepared_reused = False
    renderer._last_montage_atomic_fast_built = False
    renderer._last_montage_atomic_fast_reject_reason = ""
    renderer._last_montage_decision_levels = None
    renderer._last_montage_decision_source_rank = 0
    renderer._last_montage_physical_levels = None
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
        "backlog="
        f"{int(getattr(renderer, '_last_montage_commit_dirty_before', 0) or 0)}/"
        f"{int(getattr(renderer, '_last_montage_commit_pending_before', 0) or 0)}/"
        f"{int(getattr(renderer, '_last_montage_commit_presented_before', 0) or 0)}"
        f"->{int(getattr(renderer, '_last_montage_commit_delta_upserts', 0) or 0)}",
        f"payload={float(getattr(renderer, '_last_montage_tile_payload_build_ms', 0.0) or 0.0):.3f}",
        f"atomic_reuse={int(bool(getattr(renderer, '_last_montage_atomic_prepared_reused', False)))}",
        f"atomic_fast={int(bool(getattr(renderer, '_last_montage_atomic_fast_built', False)))}",
        f"atomic_reject={str(getattr(renderer, '_last_montage_atomic_fast_reject_reason', '') or '-')}",
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
    "FramePipelineEffects",
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
