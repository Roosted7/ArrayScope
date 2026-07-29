"""Montage pipeline effects that cross the GUI/backend presentation boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter, thread_time

import numpy as np
import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception, traceback_text
from arrayscope.core.compute_policy import ComputeLane
from arrayscope.core.gui_callback_budget import WARNING_THRESHOLD_MS
from arrayscope.core.trace import emit_trace
from arrayscope.core.window_levels import WindowLevelController
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.geometry import DisplayGeometry, display_geometry_coordinates_equal
from arrayscope.display.model.commit import CommitKind, DisplayPayload, PresentationInput
from arrayscope.display.model.frame import (
    TiledValueSource,
    canonical_plane_payload_for,
    display_tile_payload_can_commit_frame,
    display_tile_payload_has_semantics,
)
from arrayscope.display.model.montage_levels import LevelEvidenceQuality
from arrayscope.display.model.presentation_generation import levels_match
from arrayscope.display.model.tile_identity import (
    acknowledged_identity_satisfies_target,
    tile_ack_identity,
)
from arrayscope.display.model.tile_priority import prioritize_tile_numbers
from arrayscope.display.montage import montage_rect_for_viewport
from arrayscope.display.planning import LevelSourceRank, decide_presentation, normalize_bounds
from arrayscope.display.pyramid import materialize_lod_page
from arrayscope.display.slice_engine import DisplayImage
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.kernel import Lane as WorkLane
from arrayscope.kernel import Priority, Supersession, TaskSpec, WorkItem, complete_inline_work
from arrayscope.operations.chunked_stage import (
    materialize_stage_candidate_chunked,
    stage_materialization_allowed_chunk_axes,
)
from arrayscope.operations.evaluator import _document_key, stage_document_key
from arrayscope.operations.planner import final_region_for_request
from arrayscope.operations.regions import region_contains, region_is_full
from arrayscope.operations.slabs import plan_slab, request_for_image, stage_key_for_candidate
from arrayscope.operations.stage_fanin import StageFanInState
from arrayscope.presentation import payload_ref_from_display_payload
from arrayscope.render import effects as render_effects
from arrayscope.render import lod as render_lod
from arrayscope.render.ladder import Rung
from arrayscope.render.stages import CommitBatch, LodAdmissionScope
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.display_presenter import tile_residency_budget_bytes
from arrayscope.window.frame_session import _base_source_id, forget_admission_verdict
from arrayscope.window.montage_payload_cache import (
    payload_lod_matches,
    previous_tiled_payloads,
    previous_tiled_payloads_by_base_source,
)


class _StaleBatchIntent:
    def __init__(self, semantic_key) -> None:
        self.semantic_key = semantic_key


_PRESENTATION_GATE_EVENT_TYPE = Qt.QtCore.QEvent.Type(Qt.QtCore.QEvent.registerEventType())
_LOW_PRIORITY_CALLBACK_EVENT_TYPE = Qt.QtCore.QEvent.Type(Qt.QtCore.QEvent.registerEventType())

# GUI budget for one atomic-successor warm callback: several batch_size-bounded
# backend warm calls share one event-loop turn up to this elapsed budget, so a
# wide successor warms in a handful of turns while each turn stays well inside
# the 50 ms callback bar.
_ATOMIC_WARM_BUDGET_MS = 8.0


class _PresentationGateEvent(Qt.QtCore.QEvent):
    def __init__(self, effects, owner) -> None:
        super().__init__(_PRESENTATION_GATE_EVENT_TYPE)
        self.effects = effects
        self.owner = owner


class _LowPriorityCallbackEvent(Qt.QtCore.QEvent):
    def __init__(self, callback) -> None:
        super().__init__(_LOW_PRIORITY_CALLBACK_EVENT_TYPE)
        self.callback = callback


class _PresentationGateReceiver(Qt.QtCore.QObject):
    """Receiver-owned low-priority continuation for one presentation turn."""

    def event(self, event) -> bool:
        if event.type() == _PRESENTATION_GATE_EVENT_TYPE:
            event.effects._on_presentation_gate(event.owner)
            return True
        if event.type() == _LOW_PRIORITY_CALLBACK_EVENT_TYPE:
            event.callback()
            return True
        return super().event(event)


def _session_display_transposed(session) -> bool:
    """Whether a canonical session's committed display is X/Y transposed.

    Only canonical-orientation sessions store tiles in sorted-axis order, so the
    value-readout swap applies solely there; a reversed image-axis pair marks
    the transpose.
    """

    if not bool(getattr(session, "canonical_orientation", False)):
        return False
    axes = tuple(getattr(getattr(session, "view_state", None), "image_axes", None) or ())
    return len(axes) == 2 and int(axes[0]) > int(axes[1])


def canonical_plane_memo_bytes(memo) -> int:
    """CPU bytes the resident-crop canonical-plane memo currently pins.

    Both the memo's own budget and its release accounting measure the same
    thing: the semantic planes it holds strong references to.
    """

    return sum(
        int(np.asarray(entry.semantic_data).nbytes)
        for entry in dict(memo or {}).values()
        if getattr(entry, "semantic_data", None) is not None
    )


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


def _post_visible_path_callback(renderer, callback) -> None:
    """Post a receiver-bound callback at NORMAL event priority.

    Continuations on the visible critical path (atomic successor warming)
    must share the event queue fairly with completion-drain signal
    deliveries.  Posting them at LowEventPriority starved them behind a
    saturated drain storm: during the 2026-07-24 completion-drain freeze the
    warm continuation ran ~3 times/second while workers kept the queue full,
    holding a 100-tile atomic transaction (and the pixels behind it) hostage
    for tens of seconds.
    """

    Qt.QtCore.QCoreApplication.postEvent(
        _presentation_gate_receiver(renderer),
        _LowPriorityCallbackEvent(callback),
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
    session = getattr(renderer, "_frame_session", None)
    if session is not None and bool(session.scheduling_policy.verdict.coverage_open):
        # Phase 1 fully completes before the camera's rescope replays: the
        # retarget re-derives LOD demand and supersedes the lifecycle scope,
        # which mid-coverage disturbs the entry choreography (first-pixel
        # ordering, evidence barriers).  The flag stays armed; commits keep
        # arriving through coverage close, so the first teardown after the
        # pass closes owns the replay.
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

    def scheduling_verdict(self):
        """Read/advance the sole required-generation phase owner."""

        policy = self.session.scheduling_policy
        policy.observe(
            self.session.lifecycle,
            on_refinement_replan=(
                (lambda: self.renderer.request_montage_replan(self.session))
                if self._session_is_current()
                else None
            ),
        )
        return policy.verdict

    def _evaluation_claim(self, intent, step, tile) -> _EvaluationClaim:
        source_index_for_tile = getattr(intent, "source_index_for_tile", None)
        intent_source_index = (
            source_index_for_tile(int(step.tile_number))
            if callable(source_index_for_tile)
            else None
        )
        source_index = int(
            tile.source_index if intent_source_index is None else intent_source_index
        )
        source_id_for_tile = getattr(intent, "source_id_for_tile", None)
        intent_source_id = (
            source_id_for_tile(int(step.tile_number)) if callable(source_id_for_tile) else None
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
            source_id_for_tile(int(tile.montage_index)) if callable(source_id_for_tile) else None
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

        if self._step_produces_page_payload(step, tile) and not (
            step.rung == Rung.DESIRED and bool(step.reduce_from_native)
        ):
            # FLOOR is the degraded first-pixel rung: whenever the tile
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
            semantic_source_id = (
                session.tile_semantic_source_id(tile.source_index) if demand is not None else None
            )

            warm_canonical_plane = self._canonical_plane_warm_enabled()

            def evaluate_preview(token=None):
                if demand is None or semantic_source_id is None:
                    return None
                if step.rung == Rung.DESIRED:
                    # DESIRED is the target-quality owner even when it returns
                    # the same page-backed tuple shape as FLOOR.  Keeping both
                    # rungs on ``evaluate_preview_tile`` made the target lose
                    # resident-crop's native-plane sidecar: preview-first
                    # therefore committed L4 then L1 pages, but never the
                    # canonical L0 pages later crop/axis retargets rebind.
                    return render_effects.evaluate_target_tile(
                        session,
                        tile,
                        level=int(step.level),
                        demand=demand,
                        semantic_source_id=semantic_source_id,
                        cancellation_token=token,
                        shader_display=bool(getattr(session, "shader_display", False)),
                        evaluation_context=self.renderer.win._evaluation_context(
                            ComputeLane.MONTAGE_TILE, token
                        ),
                        stage_cache=self.renderer.win.operation_evaluator.stage_cache,
                        stage_materializer=(
                            self.renderer.win.operation_evaluator.stage_materializer
                        ),
                        warm_canonical_plane=warm_canonical_plane,
                    )
                return render_effects.evaluate_preview_tile(
                    session,
                    tile,
                    demand=demand,
                    semantic_source_id=semantic_source_id,
                    level=int(step.level),
                    cancellation_token=token,
                    shader_display=bool(getattr(session, "shader_display", False)),
                    evaluation_context=self.renderer.win._evaluation_context(
                        ComputeLane.MONTAGE_TILE, token
                    ),
                    stage_cache=self.renderer.win.operation_evaluator.stage_cache,
                    stage_materializer=self.renderer.win.operation_evaluator.stage_materializer,
                    warm_canonical_plane=warm_canonical_plane,
                )

            return evaluate_preview

        tile_number = int(tile.montage_index)
        request = None
        if step.rung == Rung.DESIRED:
            request = self.session.lifecycle.materialization_request_for(
                tile_number, self._level_key_for_step(tile, step)
            )
        if step.rung == Rung.DESIRED and request is not None:
            pyramid = getattr(session, "lod_page_cache", None)

            def evaluate_materialization(token=None, request=request, pyramid=pyramid):
                if pyramid is None:
                    return None
                if not pyramid.begin_owner_work(request.owner):
                    # A request cancelled before worker entry owns no claims.
                    # If another producer completed its set meanwhile, the
                    # normal resident path will observe it on the next replan.
                    return None
                try:
                    for plan in request.claimed_plans:
                        if token is not None and bool(getattr(token, "cancelled", False)):
                            from arrayscope.operations.cancellation import EvaluationCancelled

                            raise EvaluationCancelled()
                        page = materialize_lod_page(
                            request.source,
                            source_origin_yx=request.source_origin_yx,
                            plan=plan,
                        )
                        pyramid.admit_as(plan.key, page, owner=request.owner)
                    # Some requested pages may be owned by a concurrent
                    # prefetch/shifted-window singleflight. This worker is
                    # terminal once *its* pages are admitted; GUI admission
                    # below checks whole-set exactness and declines/replans if
                    # the foreign subset has not landed yet. Raising here made
                    # a normal partial attachment look like worker failure and
                    # could strand the set without a visible replan wakeup.
                    return ("materialized", request)
                finally:
                    pyramid.finish_owner_work(request.owner)

            return evaluate_materialization

        semantic_source_id = session.tile_semantic_source_id(tile.source_index)
        warm_canonical_plane = self._canonical_plane_warm_enabled()

        def evaluate_target(token=None, semantic_source_id=semantic_source_id):
            demand = session.lod_policy_decision.demand
            return render_effects.evaluate_target_tile(
                session,
                tile,
                level=int(step.level),
                demand=demand,
                semantic_source_id=semantic_source_id,
                stage_cache=self.renderer.win.operation_evaluator.stage_cache,
                stage_materializer=self.renderer.win.operation_evaluator.stage_materializer,
                cancellation_token=token,
                shader_display=bool(getattr(session, "shader_display", False)),
                evaluation_context=self.renderer.win._evaluation_context(
                    ComputeLane.MONTAGE_TILE, token
                ),
                warm_canonical_plane=warm_canonical_plane,
            )

        return evaluate_target

    def tile_states(self, intent, demand, scope: LodAdmissionScope):
        if not self._session_is_current(intent):
            return ()
        self._release_inactive_evaluation_claims(getattr(scope, "visible_tile_numbers", ()))
        self._seed_resident_crop_rebinds(scope)
        states = render_effects.tile_lod_states(self.session, demand, scope=scope)
        if (
            not bool(getattr(self.session, "shader_display", False))
            and bool(getattr(self.session, "atomic_successor_pending", False))
            and _compatible_successor_payload_count(self.session) > 0
        ):
            # Preserve an already-compatible predecessor (notably a one-index
            # shift) until its few cold replacements are ready. A fully cold
            # successor has no compatible pixels to preserve and streams in
            # canonical priority order instead.
            return tuple(replace(state, allow_preview=False) for state in states)
        return states

    def _resident_crop_rebind_enabled(self) -> bool:
        """Read the resident-crop-rebind capability once per session.

        Default ON.  The rebind eliminates the per-tile evaluation storm of a
        resident crop scrub and is pixel-correct, and a rebound window
        re-anchors its own auto levels: its carried evidence is demoted to
        preview quality (it describes the predecessor window) and the semantic
        level-evidence owner re-samples the new window off the display lane, so
        the maturity contract switches once, atomically.  Measured on a 50-tile
        row-gradient montage: window-exact levels ~220 ms after a 130 ms settle,
        zero display-preparation producers, and identical settled levels to the
        ordinary evaluation that costs 550-770 ms and 50 producers per step.
        Under ``CenteredFFT(axis=2)`` the same holds on the shared-stage montage
        (settle ~105 ms against ~330 ms, identical settled levels).  The toggle
        survives as the field isolation switch, not as a caveat.

        The GPU-histogram route is refuted, not merely unwired: the resident
        histogram (``DispatchHistogram`` over ``DataChunkKey`` blocks) is
        whole-page and carries no source sub-rectangle, while a scrubbed crop
        tile re-samples the SAME whole-plane pages at a shifted origin.  Do not
        retry it.
        """

        cached = getattr(self, "_resident_crop_rebind_flag", None)
        if cached is not None:
            return bool(cached)
        enabled = False
        win = getattr(self.renderer, "win", None)
        # Read the first-class setting object (settings -> window -> pipeline),
        # the same path every sibling toggle flows; a menu toggle drops the
        # cache via ``invalidate_resident_crop_rebind_flag`` so the next
        # retarget re-reads it without an app restart.
        app_settings = getattr(win, "app_settings", None)
        if app_settings is not None:
            enabled = bool(getattr(app_settings, "resident_crop_rebind", False))
        self._resident_crop_rebind_flag = bool(enabled)
        return bool(enabled)

    def _canonical_plane_warm_enabled(self) -> bool:
        """Whether target work may establish a backend's canonical pages."""

        capabilities = image_view_backend_capabilities(self.renderer.win.img_view)
        return bool(
            self._resident_crop_rebind_enabled() and capabilities.canonical_source_plane_residency
        )

    def invalidate_resident_crop_rebind_flag(self) -> None:
        """Drop the per-session resident-crop-rebind capability snapshot.

        ``_resident_crop_rebind_enabled`` caches the setting once per
        ``FramePipelineEffects`` (it is consulted on every ``tile_states``
        retarget). A live menu toggle clears the snapshot here so the next crop
        retarget re-reads ``app_settings.resident_crop_rebind`` — the toggle
        takes effect on the next scrub, no new session or restart required.
        """

        self._resident_crop_rebind_flag = None

    def _seed_resident_crop_rebinds(self, scope) -> None:
        """Rebind a resident crop-window scrub before the ladder plans it.

        A displayed-axis crop retarget whose new source window is already
        physically resident under the current content identity short-circuits
        to a pure rebind: no kernel evaluation, no display-cache miss, just a
        resident-page rebind committed with no upload.  Running here — before
        ``tile_lod_states`` snapshots the ladder inputs — means a rebound tile is
        already recorded as target coverage, so it never enters ``missing_tiles``
        and no producer is scheduled.  The residency seam is backend-owned
        physical truth (``tiledPayloadResident``); backends without it keep the
        ordinary per-tile evaluation.

        Montage and single-slice presentations share this path.  The montage
        gate this replaced assumed the single-tile path "composes the
        source-anchored plane differently"; measured, it does not.  Both build
        a crop-local payload whose ``source_anchor`` names a sub-rect of the
        SAME canonical source plane, and ``_wgpu_payload_binding`` resolves
        both through one ``("wgpu-source-plane", content_key)`` page identity —
        a cropped montage tile is just as crop-local as a single-slice payload
        (both measured at a 200x200 native shape for a 200x200 window on a
        336x336 plane).  The non-montage anchor differs only in where its
        content key comes from (``source_anchoring_for_view`` directly, rather
        than the per-source-index cache), and that key folds the slider index,
        so advancing the slice still declines.
        """

        if not self._resident_crop_rebind_enabled():
            # A capability the user turned off may not keep whole planes pinned:
            # the memo holds STRONG references, so the toggle is also how the
            # memory is handed back.
            self._release_canonical_plane_payloads("capability-disabled")
            self._record_resident_crop_rebind(gate="disabled")
            return
        self._expire_canonical_plane_payloads_for_document()
        win = getattr(self.renderer, "win", None)
        img_view = getattr(win, "img_view", None)
        resident_fn = getattr(img_view, "tiledPayloadResident", None)
        if not callable(resident_fn):
            self._record_resident_crop_rebind(gate="no-residency-seam")
            return
        tile_numbers = tuple(getattr(scope, "visible_tile_numbers", ()) or ())
        if not tile_numbers:
            self._record_resident_crop_rebind(gate="no-visible-tiles")
            return
        # The last committed frame is the pre-scrub (predecessor-window) truth:
        # a new session's live payload map is empty at plan time, so the
        # committed wrappers are what a resident rebind clones from.
        previous_by_tile = previous_tiled_payloads(getattr(win, "_committed_display_frame", None))
        rebound = self.session.rebind_resident_crop_tiles(
            physical_resident_fn=resident_fn,
            tile_numbers=tile_numbers,
            previous_by_tile=previous_by_tile,
            canonical_by_tile=self._canonical_plane_memo(),
            remember_canonical=self._remember_canonical_plane_payload,
        )
        if rebound:
            # A rebound window is presented without any evaluation sampling it,
            # so the only producer that can anchor its levels is the semantic
            # evidence owner.  Arm it here, at the seam that knows the window
            # was rebound rather than computed.
            self.renderer.rearm_crop_rebind_level_evidence(self.session)
        self._record_resident_crop_rebind(
            gate="attempted",
            tile_numbers=len(tile_numbers),
            rebound=len(rebound),
            stats=dict(getattr(self.session, "resident_crop_rebind_stats", None) or {}),
        )

    def _record_resident_crop_rebind(
        self,
        *,
        gate: str,
        tile_numbers: int = 0,
        rebound: int = 0,
        stats: dict[str, int] | None = None,
    ) -> None:
        """Publish one seed outcome to the trace stream and the field counters.

        The rebind either happens or silently does not; a field session can only
        observe the producers it failed to avoid.  Recording the gate verdict and
        the per-tile decline histogram here means the next diagnostics JSONL
        answers "was it called, and why did it decline?" directly.  Counters live
        on the renderer, not the session, because a crop scrub retires sessions
        continuously and per-session totals would reset before anyone read them.
        """

        stats = dict(stats or {})
        renderer = self.renderer
        totals = getattr(renderer, "resident_crop_rebind_totals", None)
        if totals is None:
            totals = {}
            renderer.resident_crop_rebind_totals = totals
        totals[f"gate:{gate}"] = int(totals.get(f"gate:{gate}", 0)) + 1
        for key, value in stats.items():
            totals[key] = int(totals.get(key, 0)) + int(value)
        previous_gate = getattr(renderer, "resident_crop_rebind_last_gate", None)
        renderer.resident_crop_rebind_last_gate = str(gate)
        if gate != "attempted" and previous_gate == gate:
            # A refused gate repeats on every retarget of every session; one
            # event per transition is enough to prove which gate is closed.
            return
        emit_trace(
            "resident_crop_rebind",
            gate=gate,
            session=int(getattr(self.session, "session_id", 0) or 0),
            tile_numbers=int(tile_numbers),
            rebound=int(rebound),
            **{f"stat_{key}": int(value) for key, value in stats.items()},
        )

    # The memo holds a STRONG reference (a weak one dies the moment the crop
    # replaces the whole-plane payload, which is exactly when the scrub starts),
    # so it must state what it can pin.  This is the budget for the memo AS A
    # WHOLE.  A per-plane cap with a flat four-entry count was measured to be
    # both too loose and too tight at once: four planes could pin 256 MB, while
    # a 50-tile montage — every tile of which needs its own whole plane to
    # re-slice — could seat only four and therefore declined the rebind for the
    # other 46 (measured: 50 producers per scrub step where the pre-single-slice
    # montage rebind scheduled none).  Fifty 336x336 float32 planes are 22.6 MB,
    # comfortably inside this; a montage of genuinely large planes still stops
    # at the budget and keeps its ordinary evaluation.
    _CANONICAL_PLANE_MEMO_MAX_BYTES = 64 * 1024 * 1024

    def _canonical_plane_memo(self) -> dict[int, object]:
        """The renderer-owned memo of whole planes a rebind can re-slice.

        The memo lives on the RENDERER, not the session: a crop retarget builds
        a fresh ``FrameSession`` every step (measured: session ids 2, 3, 4, 5
        across one scrub), so a session-owned memo is empty exactly when the
        scrub needs it.  The renderer outlives the sessions, like the committed
        frame the predecessor wrappers already come from.
        """

        memo = getattr(self.renderer, "_resident_crop_canonical_planes", None)
        if memo is None:
            memo = {}
            self.renderer._resident_crop_canonical_planes = memo
        return memo

    def _remember_canonical_plane_payload(self, tile_number: int, committed):
        """Pin one committed payload's whole plane, or return ``None``.

        A source-anchored rebind may only reuse canonical pages that some
        earlier payload uploaded WHOLE, and a whole plane is likewise the only
        thing that can re-slice exact CPU semantics for a shifted window
        (``_rebind_reslice_planes``).  Two payloads supply one: a payload that
        covers its entire source plane (the backend's
        ``supplies_complete_pages`` precondition, mirrored on the CPU side),
        and a CROPPED payload whose evaluation was widened to the plane it
        presents a sub-rect of (``canonical_plane_payload_for``).  Without the
        second, a view cropped from its first frame onward has no whole-plane
        payload anywhere in its history and every step declined with
        ``no_reslicable_plane``.

        The rebind calls this only once it has proven the predecessor's content
        key is the window's CURRENT one, which is what keeps a retired
        generation out: after a document or operation change the committed
        frame still holds the previous identity's payloads, and pinning those
        would silently re-fill the memo the change just released.
        """

        memo = self._canonical_plane_memo()
        if str(getattr(committed, "quality", "exact") or "exact") != "exact":
            return None
        if getattr(committed, "semantic_data", None) is None:
            return None
        anchor = getattr(committed, "source_anchor", None)
        plane_shape = tuple(getattr(anchor, "plane_shape", None) or ())
        source_rect = tuple(getattr(anchor, "source_rect", None) or ())
        if len(plane_shape) != 2 or len(source_rect) != 4:
            return None
        y0, y1, x0, x1 = (int(value) for value in source_rect)
        payload = committed
        if (y0, x0) != (0, 0) or (y1, x1) != (int(plane_shape[0]), int(plane_shape[1])):
            # A cropped payload never covers its plane; it can still carry one.
            # Memoize the carried plane, not the window.
            payload = canonical_plane_payload_for(committed)
            if payload is None:
                return None
        plane_bytes = int(np.asarray(payload.semantic_data).nbytes)
        key = int(tile_number)
        replaced = memo.get(key)
        replaced_bytes = (
            0
            if replaced is None or getattr(replaced, "semantic_data", None) is None
            else int(np.asarray(replaced.semantic_data).nbytes)
        )
        pinned_bytes = canonical_plane_memo_bytes(memo)
        if pinned_bytes - replaced_bytes + plane_bytes > self._CANONICAL_PLANE_MEMO_MAX_BYTES:
            return None
        memo[key] = payload
        return payload

    def _release_canonical_plane_payloads(self, reason: str) -> None:
        """Unpin every memoized whole plane and say why.

        The memo is the only strong reference the rebind adds outside the frame
        it committed, so releasing it is the whole of its memory contract; the
        reason is recorded because a field session sees the freed bytes but not
        the cause.
        """

        renderer = self.renderer
        memo = getattr(renderer, "_resident_crop_canonical_planes", None)
        if not memo:
            return
        renderer._resident_crop_canonical_planes_released_bytes = canonical_plane_memo_bytes(memo)
        renderer._resident_crop_canonical_planes_release_reason = str(reason)
        memo.clear()

    def _expire_canonical_plane_payloads_for_document(self) -> None:
        """Release the pinned planes when the document/operation identity moves.

        The memo is keyed by tile number alone, so a document reload or an
        operation edit leaves planes of the RETIRED identity pinned under the
        live tile numbers.  Correctness never depended on this — the content-key
        check in ``rebind_resident_crop_tiles`` refuses to re-slice a stale plane
        — but nothing dropped them either, so a whole edited-away plane set
        (bounded by ``_CANONICAL_PLANE_MEMO_MAX_BYTES``) stayed resident for the
        rest of the session.  ``_document_key`` is the same identity the frame
        currency check uses, and it folds the operation steps, so an operation
        edit expires the memo exactly like a data reload.

        This runs on the seed path rather than a fresh document-change signal:
        the seed is consulted on every retarget, so the first render after any
        change reaches it, and no new notification has to stay in sync.
        """

        renderer = self.renderer
        win = getattr(renderer, "win", None)
        document = getattr(win, "document", None)
        if document is None:
            document = getattr(self.session, "document", None)
        key = None if document is None else _document_key(document)
        if key == getattr(renderer, "_resident_crop_canonical_planes_document_key", None):
            return
        self._release_canonical_plane_payloads("document-changed")
        renderer._resident_crop_canonical_planes_document_key = key

    def prepare_rung(self, intent, step) -> bool:
        tile = self._tile_for_step(step)
        if tile is None or not self._session_is_current(intent):
            return False
        if not self.scheduling_verdict().admits_lane(step.lane):
            return False
        tile_number = int(tile.montage_index)
        semantic_key = self._preview_claim_identity(intent, tile)
        if bool(getattr(step, "presentation_only", False)):
            # R2 has already proved that pixels satisfy this demand, either
            # as a resident page or a lifecycle-ready payload whose
            # completion lost its commit owner. Recover only a current
            # wrapper and arm presentation. If recovery fails, continue
            # into the ordinary producer path rather than stranding the
            # tile behind an unprovable "ready" claim.
            if step.rung == Rung.FLOOR:
                self.session._ensure_floor_payloads((tile_number,), max_count=1)
            payload = self.session.display_tile_payloads.get(tile_number)
            lifecycle = self.session.lifecycle
            if not lifecycle.payload_is_current(tile_number, payload):
                payload = lifecycle.current_presentable_payload(tile_number)
            if lifecycle.payload_is_current(tile_number, payload):
                self.session.display_tile_payloads[tile_number] = payload
                rearmed = self.session._rearm_required_first_pixel_payloads()
                presentation_armed = bool(
                    tile_number in rearmed
                    or tile_number in getattr(self.session, "dirty_payloads", ())
                    or tile_number in getattr(self.session, "pending_payload_upserts", ())
                )
                if presentation_armed:
                    self.request_presentation()
                    return False
        if self._step_produces_page_payload(step, tile):
            if step.rung == Rung.FLOOR and self._display_payload_covers_preview_step(
                tile_number, tile, step
            ):
                return False
            if step.rung == Rung.DESIRED and self._display_payload_covers_display_target(
                tile_number, tile, step
            ):
                return False
            if self.session.lifecycle.preview_claim_matches(
                tile_number,
                int(step.rung),
                int(step.level),
                semantic_key,
            ):
                return False
            if step.rung == Rung.DESIRED and self.session.lifecycle.preview_claim_matches(
                tile_number,
                int(Rung.FLOOR),
                int(step.level),
                semantic_key,
            ):
                return False
            claimed = self.session.lifecycle.preview_claimed(
                tile_number,
                int(step.rung),
                int(step.level),
                semantic_key,
            )
            if claimed and step.rung == Rung.FLOOR:
                # The phase owner must wait for acknowledged preview coverage,
                # not generic first pixels that may be retained from an exact
                # predecessor. CPU-composited backends do not seed first-pass
                # histogram evidence, so the FLOOR claim is their canonical
                # declaration that this scope has a preview pass.
                note_quality = getattr(self.session, "note_first_pass_quality", None)
                if callable(note_quality):
                    note_quality("preview")
            return claimed
        if (
            step.rung == Rung.DESIRED
            and int(step.level) > 0
            and tile_number in self.session.rendered_tiles
        ):
            pyramid = getattr(self.session, "lod_page_cache", None)
            if pyramid is None:
                return False
            rendered = self.session.rendered_tiles.get(tile_number)
            demand = self.session.lod_policy_decision.demand
            try:
                level_key = self.session._lod_page_set_key_for(
                    rendered, demand=demand, level=int(step.level)
                )
            except ValueError:
                return False
            if (
                self.session.lifecycle.materialization_request_for(tile_number, level_key)
                is not None
            ):
                return False
            if render_lod._page_set_exact(pyramid, level_key):
                self.session.lifecycle.level_resident(tile_number, level_key)
                return False
            request = self.session._lod_materialization_request(
                rendered,
                demand=demand,
                level=int(step.level),
                key=level_key,
            )
            if not request.claimed_plans:
                return False
            self.session.pending_rung_materializations.append(request)
            self.session.pending_rung_materializations.mark_started(request)
            return True
        if step.rung in (Rung.DESIRED, Rung.EXACT):
            if (
                tile_number in self.session.rendered_tiles
                or tile_number in self.session.skipped_tiles
            ):
                return False
            if tile_number in self.session.active_tile_requests:
                claim = self.session.lifecycle.evaluation_claim_for(tile_number)
                current_claim = self._evaluation_claim(intent, step, tile)
                if claim is not None and claim.matches_tile(
                    intent, step, tile, current_claim.source_id
                ):
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
        if step.rung == Rung.FLOOR and not str(
            getattr(self.session, "round_level_evidence_source", "") or ""
        ):
            self.renderer._expect_preview_cohort_level_evidence(self.session)
        tile_number = int(tile.montage_index)
        if self._step_produces_page_payload(step, tile):
            if step.rung == Rung.FLOOR and render_effects.can_evaluate_reduced_preview(
                self.session, tile
            ):
                renderer = self.renderer
                renderer._montage_preview_reduced_scheduled = (
                    int(getattr(renderer, "_montage_preview_reduced_scheduled", 0) or 0) + 1
                )
                renderer._montage_preview_reduced_last_gate = "submitted by per-tile coarse ladder"
            return
        if (
            step.rung in (Rung.DESIRED, Rung.EXACT)
            and tile_number not in self.session.rendered_tiles
        ):
            stage_key = self.session.stage_fan_in.tile_stage_keys.get(tile_number)
            stage_producer_key = self._stage_producer_key(stage_key)
            if stage_producer_key is None:
                stage_key = None
            self.session.mark_loading(tile)
            self.session.lifecycle.evaluation_claimed(
                tile_number, self._evaluation_claim(intent, step, tile)
            )
            self.session.lifecycle.task_admitted(
                tile_number,
                task_key,
                stage_key=stage_key,
                stage_producer_key=stage_producer_key,
            )

    def _step_produces_page_payload(self, step, tile) -> bool:
        """Whether this rung returns canonical display pages, not a native value.

        ``reduce_from_native`` selects the numeric source route only.  It must
        not select lifecycle ownership: every cold DESIRED rung above level
        zero returns the same page-backed display contract, whether its pages
        were derived from a retained reduction or directly from native data.
        """

        if step.rung == Rung.FLOOR:
            return True
        tile_number = int(getattr(tile, "montage_index", getattr(step, "tile_number", -1)))
        if tile_number in getattr(self.session, "rendered_tiles", {}):
            return False
        return bool(step.rung == Rung.DESIRED and int(step.level) > 0)

    def _display_payload_covers_display_target(self, tile_number: int, tile, step) -> bool:
        payload = self.session.display_tile_payloads.get(int(tile_number))
        if not self._display_payload_is_current(tile_number, tile, payload=payload):
            return False
        lod = getattr(payload, "lod", None)
        if lod is None:
            return False
        if int(getattr(lod, "level", 0) or 0) > int(step.level):
            return False
        # Backend drawability and target settlement are intentionally
        # different contracts.  A current equal-LOD fallback is safe to keep
        # on screen, but it still owes exact target work.  Use the lifecycle's
        # canonical quality/LOD predicate here; the backend identity predicate
        # also accepts safe fallbacks and would deny this tile its only exact
        # producer.
        record = self.session.lifecycle.peek(int(tile_number))
        if record is None or record.target is None:
            return False
        return payload_ref_from_display_payload(payload).satisfies_target(record.target)

    def _display_payload_covers_preview_step(self, tile_number: int, tile, step) -> bool:
        """Whether a current ready payload already covers one FLOOR request.

        Reduced results enter ``display_tile_payloads`` before their backend
        transaction is acknowledged.  Releasing the worker claim at that
        admission boundary is correct, but it used to let each intervening
        replan submit the same FLOOR evaluation again.  The ready payload is
        the canonical owner of that gap: it may suppress duplicate work while
        the phase owner still waits for physical acknowledgement.
        """

        payload = self.session.display_tile_payloads.get(int(tile_number))
        if payload is None:
            return False
        if int(getattr(payload, "source_index", -1)) != int(getattr(tile, "source_index", -2)):
            return False
        semantic_id = self.session.tile_semantic_source_id(tile.source_index)
        payload_source_id = getattr(payload, "source_id", None)
        if payload_source_id != semantic_id and _base_source_id(payload_source_id) != semantic_id:
            return False
        quality = str(getattr(payload, "quality", "exact") or "exact")
        if quality not in {"preview", "fallback"}:
            return False
        lod = getattr(payload, "lod", None)
        return bool(lod is not None and int(getattr(lod, "level", 0) or 0) <= int(step.level))

    def _display_payload_is_current(self, tile_number: int, tile, *, payload=None) -> bool:
        payload = (
            self.session.display_tile_payloads.get(int(tile_number)) if payload is None else payload
        )
        if payload is None:
            return False
        if int(getattr(payload, "source_index", -1)) != int(getattr(tile, "source_index", -2)):
            return False
        semantic_id = self.session.tile_semantic_source_id(tile.source_index)
        payload_source_id = getattr(payload, "source_id", None)
        if payload_source_id != semantic_id and _base_source_id(payload_source_id) != semantic_id:
            return False
        return int(tile_number) in set(getattr(self.session.lifecycle, "presented_tiles", ()) or ())

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
        if bool(getattr(kernel, "has_live_task", lambda _key: True)(task_key)):
            return True
        # A worker leaves the live-task table before its CompletionEvent is
        # drained on the GUI thread. The completed-key table spans that handoff.
        # Releasing the lifecycle claim in this interval lets a queued replan
        # submit the identical native evaluation a second time.
        return bool(getattr(kernel, "has_completed_task", lambda _key: False)(task_key))

    def _release_inactive_evaluation_claims(self, tile_numbers=()) -> int:
        active_requests = getattr(self.session, "active_tile_requests", None)
        if active_requests is None:
            return 0
        visible = {int(tile) for tile in tuple(tile_numbers or ())}
        scope = visible or {int(tile) for tile in tuple(active_requests or ())}
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
            if step.rung == Rung.DESIRED:
                tile_number = int(step.tile_number)
                pending = self.session.pending_rung_materializations
                request = self.session.lifecycle.materialization_request_for(tile_number)
                while request is not None:
                    pending._apply_release_effects(pending.release(request))
                    request = self.session.lifecycle.materialization_request_for(tile_number)
            if self._session_is_current(intent):
                self.renderer.request_montage_replan(self.session)
            return
        tile_number = int(tile.montage_index)
        semantic_key = self._preview_claim_identity(intent, tile)
        reduced_display_step = self._step_produces_page_payload(step, tile)
        if step.rung == Rung.FLOOR:
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
            request = self.session.lifecycle.materialization_request_for(
                tile_number, self._level_key_for_step(tile, step)
            )
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
        getter = (
            stage_cache.get_containing
            if hasattr(stage_cache, "get_containing")
            else stage_cache.get
        )
        return getter(stage_key) is not None

    def _release_evaluation_claim(
        self, tile_number: int, *, marker=None, request_replan: bool = True
    ) -> bool:
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
        if (
            tile_number not in self.session.rendered_tiles
            and tile_number not in self.session.display_tile_payloads
        ):
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
            upserts=len(tuple(batch.upserts or ())),
        )

        if batch.semantic_key is not None and batch.semantic_key != getattr(
            self.session, "key", batch.semantic_key
        ):
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
            # later replan to skip this tile. An admitted FLOOR also
            # unlocks the finer rung which was deliberately held behind first
            # pixels. Both transitions therefore own an explicit replan
            # wakeup; a presentation wake alone can strand the visible tile at
            # its fallback LOD when the resident backend needs no new upsert.
            self.renderer.request_montage_replan(self.session)
        self._prepare_backend_uploads(batch.upserts)
        self.request_presentation()

    def _prepare_backend_uploads(self, upserts) -> None:
        """Hand each admitted payload's upload preparation to a worker.

        Admission is the earliest moment the exact pixels of a future commit
        are known, and the commit that consumes them is several governed chunks
        away, so there is real time to use. The GUI thread does no preparation
        here: it only describes the work and submits it.

        Every part of this is best-effort by construction. A task that is
        superseded, dropped, or simply slower than the commit costs nothing but
        worker time — the backend's inline path is unchanged and still owns
        correctness. Nothing downstream may wait on a preparation.
        """

        if not tuple(upserts or ()):
            return
        window = getattr(self.renderer, "win", None)
        view = getattr(window, "img_view", None)
        plan = getattr(view, "tiledUploadPreparations", None)
        kernel = getattr(window, "kernel", None)
        if not callable(plan) or kernel is None:
            return
        session = self.session
        # The batch rows carry rung results, not display payloads — admission is
        # what turns one into the other. Take the tiles this batch just admitted
        # and read the payload the lifecycle now holds for each. Deliberately
        # not the whole dirty set: a payload stays dirty across many governed
        # chunks, and re-offering it every drain repeats work already published.
        lifecycle = session.lifecycle
        payload_resident = getattr(view, "tiledPayloadResident", None)
        payloads = {}
        resident_skips = 0
        for row in tuple(upserts or ()):
            if not isinstance(row, tuple) or len(row) != 2 or row[1] is None:
                continue
            tile = getattr(row[0], "tile_number", None)
            if tile is None:
                continue
            payload = lifecycle.current_presentable_payload(int(tile))
            if payload is None:
                payload = session.display_tile_payloads.get(int(tile))
            if payload is None:
                continue
            if callable(payload_resident) and bool(payload_resident(payload)):
                resident_skips += 1
                continue
            payloads[int(tile)] = payload
        mailbox = view.preparedTiledUploads
        if resident_skips:
            mailbox.note_resident(resident_skips)
        if not payloads:
            return
        level_generation = getattr(session, "level_generation", None)
        levels = normalize_bounds(getattr(level_generation, "target_levels", None))
        if levels is None:
            # Levels are not settled yet, so any PyQtGraph assembly prepared now
            # would bake against a window this round will not use (R3). WGPU
            # ignores the value; skipping is the conservative shared answer.
            return
        if getattr(level_generation, "semantic_key", None) != getattr(session, "level_key", None):
            # Present levels, but belonging to the *previous* round. Preparing
            # against them is not a gamble that sometimes pays: the commit will
            # ask under this round's window and refuse every buffer as stale,
            # so it is a guaranteed miss bought with worker time. Measured on a
            # cold scroll: 36 preparations, 36 stale takes, zero hits, all of
            # them baked at the pre-scroll (0, 71) while the commit wanted
            # (36, 107). The same predicate the commit path uses to decide the
            # level generation has caught up with the round.
            mailbox.note_stale_round(len(payloads))
            return
        session_id = int(getattr(session, "session_id", 0) or 0)
        try:
            preparations = plan(
                payloads,
                levels=levels,
                # Same session-derived form the residency warm path uses for
                # ahead-of-commit work (montage_prefetch): the frame's own
                # flag does not exist until the commit builds it, and a wrong
                # guess only costs a mailbox miss.
                rgb_already_windowed=not bool(getattr(session, "shader_display", False)),
            )
        except Exception as exc:  # pragma: no cover - defensive, never fatal
            handle_ui_exception("montage upload preparation planning", exc)
            return
        for slot, key, prepare in preparations:
            if mailbox.holds(slot, key):
                mailbox.note_deduped()
                continue
            mailbox.note_submitted()
            kernel.submit(
                TaskSpec(
                    key=("prepared-upload", session_id, slot, key),
                    fn=prepare,
                    # Speculative, not visible. Priority orders *selection*
                    # from the ready set; it cannot reclaim a worker already
                    # inside a task, so a preparation on a visible lane could
                    # and did hold a thread a pixel-producing task wanted —
                    # 458 ms of it across one recorded cold scroll, delaying
                    # 39 producers by up to 21 ms each. A non-visible lane is
                    # what makes the kernel's existing speculative governor
                    # apply: parked while any visible work is queued or
                    # running, capped at a fraction of the pool, and sorted
                    # behind every visible task rather than merely below it.
                    lane=WorkLane.SPECULATIVE_RESIDENCY,
                    priority=Priority.PREFETCH,
                    scope=f"montage:{session_id}",
                    session_id=session_id,
                    tile_number=int(slot),
                    supersession=Supersession(
                        family=("prepared-upload", session_id, slot),
                        value=key,
                    ),
                ),
            )

    def request_presentation(self) -> None:
        """Coalesced presentation continuation: at most one per loop turn.

        Category: zero-delay coalescing continuation (like the render
        coordinator), NOT wall-clock pacing — it exists so paints and input
        events interleave with bounded commits. Re-armed by the commit
        itself while backlog (flush/final/dirty/upserts) remains, so a
        bounded commit can never strand its remainder (the ADR 0051
        lost-wakeup rule: every deferral leaves a wakeup armed).
        """

        owner = (int(getattr(self.session, "session_id", 0) or 0), id(self.session))
        if getattr(self.renderer, "_montage_commit_failed_owner", None) == owner:
            # Worker completions already in flight can arrive after a commit
            # throws. They are not permission to retry the failed transaction:
            # keep the named failure terminal for this session generation.
            emit_trace("presentation_gate", action="suppressed-failed", owner_session=owner[0])
            return
        if getattr(self.renderer, "_montage_presentation_gate_owner", None) == owner:
            emit_trace("presentation_gate", action="coalesced", owner_session=owner[0])
            return
        self.renderer._montage_presentation_gate_armed = True
        self.renderer._montage_presentation_gate_owner = owner
        receiver = _presentation_gate_receiver(self.renderer)
        if not bool(
            image_view_backend_capabilities(self.renderer.win.img_view).shader_windowing
        ) and bool(getattr(self.session, "display_committed", False)):
            # Timer category: UI cosmetic. A short one-shot coalescer lets
            # several completed CPU-windowed tiles share one scene rebuild.
            # Without it, one worker completion caused one whole-frame commit
            # (600+ commits in a five-second scrub). Generation/current-session
            # checks remain in `_on_presentation_gate`.
            emit_trace("presentation_gate", action="armed-timer", owner_session=owner[0])
            Qt.QtCore.QTimer.singleShot(
                4,
                receiver,
                lambda effects=self, owner=owner: effects._on_presentation_gate(owner),
            )
            return
        emit_trace("presentation_gate", action="armed-post", owner_session=owner[0])
        # Low priority is the fairness contract: already queued input, paint,
        # heartbeat, and kernel-delivery events run before the next bounded
        # presentation slice. The receiver is parented to the orchestrator,
        # and `_on_presentation_gate` rechecks the session generation.
        Qt.QtCore.QCoreApplication.postEvent(
            receiver,
            _PresentationGateEvent(self, owner),
            int(Qt.QtCore.Qt.EventPriority.LowEventPriority.value),
        )

    def _on_presentation_gate(self, owner=None) -> None:
        live_owner = (int(getattr(self.session, "session_id", 0) or 0), id(self.session))
        owner = live_owner if owner is None else owner
        if getattr(self.renderer, "_montage_presentation_gate_owner", None) != owner:
            # A successor session armed its own continuation while this stale
            # callback was queued. It owns the shared gate now; this callback
            # must neither clear nor consume that wakeup.
            emit_trace("presentation_gate", action="fired-stale", owner_session=owner[0])
            return
        if live_owner != owner:
            # Index-window retargeting deliberately keeps the FrameSession and
            # effects object alive while advancing its generation. A callback
            # queued for the predecessor must not impersonate the successor by
            # reading the mutable session id only when it fires.
            self.renderer._montage_presentation_gate_owner = None
            self.renderer._montage_presentation_gate_armed = False
            emit_trace(
                "presentation_gate",
                action="fired-generation-stale",
                owner_session=owner[0],
                live_session=live_owner[0],
            )
            return
        self.renderer._montage_presentation_gate_owner = None
        self.renderer._montage_presentation_gate_armed = False
        if not self._session_is_current():
            emit_trace("presentation_gate", action="fired-not-current", owner_session=owner[0])
            return
        emit_trace("presentation_gate", action="fired-commit", owner_session=owner[0])
        try:
            self.commit_pending_session()
        except Exception as exc:
            # This is the outermost Python frame of a QEvent: without this,
            # Qt prints the traceback and continues, and the failure reaches
            # the operator only as an anonymous stall four seconds later.
            # Route it through the same policy every other Qt callback uses —
            # loud by default, fatal under ARRAYSCOPE_STRICT_UI.
            #
            # Note what is deliberately absent: no re-arm. The gate's owner
            # and armed flags were cleared above and `commit_pending_session`
            # resets its drain flag in a `finally`, so a throw corrupts no
            # gate state and a later session can still arm. What it strands
            # is this session's backlog, and re-arming that would replay a
            # delta about to throw again (ADR 0051; dossier
            # wgpu-pool-layer-leak-2026-07-26 §5a).
            emit_trace(
                "presentation_gate",
                action="fired-raised",
                owner_session=owner[0],
                exception_type=type(exc).__name__,
            )
            handle_ui_exception("montage presentation commit", exc)

    def admit_tile_result(self, tile, result) -> int:
        """Admit one native target result into session/lifecycle state.

        Kernel bridge callbacks are already bounded; the old
        frame_renderer-side result deque/timer was a second fan-in queue.
        """

        return self._admit_evaluation_result(tile, result)

    def commit_pending_session(self) -> None:
        if not self._session_is_current():
            return
        owner = (int(getattr(self.session, "session_id", 0) or 0), id(self.session))
        if getattr(self.renderer, "_montage_commit_failed_owner", None) == owner:
            return
        if bool(getattr(self.renderer, "_montage_commit_drain_active", False)):
            self.session.final_commit_pending = True
            self.session.flush_pending = True
            self.request_presentation()
            return
        self.renderer._montage_commit_drain_active = True
        try:
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
                    *(
                        int(tile)
                        for tile in tuple(getattr(session, "pending_payload_upserts", ()) or ())
                    ),
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
            tuple(
                int(tile) for tile in tuple(getattr(session, "pending_payload_upserts", ()) or ())
            ),
            tuple(
                sorted(int(tile) for tile in tuple(getattr(session, "pending_removals", ()) or ()))
            ),
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
            emit_trace(
                "commit_rearm",
                decision="no-backlog",
                session_id=int(getattr(session, "session_id", 0) or 0),
            )
            self.renderer._montage_gate_last_backlog = None
            return
        signature = self._backlog_signature()
        previous = getattr(self.renderer, "_montage_gate_last_backlog", None)
        self.renderer._montage_gate_last_backlog = signature
        emit_trace(
            "commit_rearm",
            decision="repeat" if previous == signature else "rearm",
            session_id=int(getattr(session, "session_id", 0) or 0),
            dirty=len(getattr(session, "dirty_payloads", ()) or ()),
            upserts=len(getattr(session, "pending_payload_upserts", ()) or ()),
            flush=bool(getattr(session, "flush_pending", False)),
            final=bool(getattr(session, "final_commit_pending", False)),
        )
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
        return DisplayImage(
            data=placeholder, histogram_data=None, rgb_already_windowed=False
        ), geometry

    # ------------------------------------------------------------------ admit

    def _admit_ready_payloads(self, rows) -> bool:
        replan_needed = False
        reduced_groups = {}
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
            if not isinstance(payload, tuple):
                admitted = self._admit_reduced_display_payload(
                    step,
                    int(step.tile_number),
                    payload,
                )
                if not admitted or step.rung == Rung.FLOOR:
                    replan_needed = True
                continue
            group_key = (int(step.rung), int(step.level))
            reduced_groups.setdefault(group_key, []).append((step, int(step.tile_number), payload))

        for (rung, _level), group in reduced_groups.items():
            step = group[0][0]
            payload_rows = tuple((tile_number, *payload) for _step, tile_number, payload in group)
            admitted = self._admit_reduced_display_payload(
                step,
                int(group[0][1]),
                payload_rows,
            )
            if not admitted or rung == int(Rung.FLOOR):
                replan_needed = True
        return bool(replan_needed)

    def _admit_materialized_rung(self, step, request) -> None:
        tile_number = int(step.tile_number)
        tile = self._tile_for_step(step)
        if tile is None:
            pending = self.session.pending_rung_materializations
            pending._apply_release_effects(pending.release(request))
            self.renderer.request_montage_replan(self.session)
            return
        claim_identity = self._preview_claim_identity(None, tile)
        self.session.lifecycle.preview_released(
            tile_number,
            int(step.rung),
            int(step.level),
            claim_identity,
        )
        if not self.session.pending_rung_materializations.mark_resident(request):
            # A bounded cache may evict the exact pages after the worker has
            # completed but before this GUI-thread admission runs.  That is a
            # normal declined result: release the lifecycle claim and let the
            # existing planner decide whether to retry or use an ancestor.
            self.renderer.request_montage_replan(self.session)
            return
        self.session.lod_materializations_completed = (
            int(getattr(self.session, "lod_materializations_completed", 0) or 0) + 1
        )
        if tile_number in self.session.rendered_tiles:
            self.session.dirty_payloads[tile_number] = None
            self.session.flush_pending = True
            self.session.final_commit_pending = True
        else:
            claim = self.session.lifecycle.evaluation_claim_for(tile_number)
            if (
                claim is not None
                and claim.rung == int(step.rung)
                and claim.level == int(step.level)
            ):
                self._release_evaluation_claim(tile_number, marker=claim, request_replan=False)
            # The source tile disappeared while this reusable materialization
            # was running (for example an index-window retarget). It cannot
            # build a current payload, so a new producer must be planned.
            self.renderer.request_montage_replan(self.session)
        # Newly resident pages can complete target sets owned by other tiles
        # or shifted-window requests that attached to the same singleflight.
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
            session.tile_compute_stage_backed_max_ms = max(
                float(session.tile_compute_stage_backed_max_ms), eval_ms
            )
            session.stage_backed_tiles_pending = max(0, int(session.stage_backed_tiles_pending) - 1)
            session.tile_compute_waiting_for_stage = max(
                0, int(session.tile_compute_waiting_for_stage) - 1
            )
        else:
            session.tile_compute_direct += 1
            session.tile_compute_direct_ms += eval_ms
            session.tile_compute_direct_max_ms = max(
                float(session.tile_compute_direct_max_ms), eval_ms
            )
        self.renderer._admit_first_pass_level_evidence(session, rendered, quality="exact")
        session.mark_materialized(rendered)
        session.lifecycle.task_released(int(tile.montage_index), reason="completed")
        session.dirty_tiles.append(int(tile.montage_index))
        return rendered_tile_nbytes(rendered)

    def _admit_reduced_display_payload(
        self, step, tile_number: int, payload, *, quality: str | None = None
    ) -> bool:
        session = self.session
        if not self._session_is_current():
            return False
        shared_preview_cohort = _looks_like_shared_preview_rows(payload)
        rows = payload if shared_preview_cohort else ((int(tile_number), *payload),)
        quality = str(
            quality or ("exact" if step is not None and step.rung == Rung.DESIRED else "preview")
        )
        is_preview = quality == "preview"
        upserted = False
        admitted_any = False
        visible_previews = 0
        admitted_tiles = []
        admitted_keys = {}
        for row in tuple(rows or ()):
            (
                tile_number,
                key,
                plane,
                histogram,
                shader_mapping,
                texture_kind,
                level_data,
                level_stats,
                native_residency_data,
                native_rendered,
            ) = preview_row_parts(row)
            admitted = session.admit_preview_plane(
                tile_number,
                key,
                plane,
                histogram,
                shader_mapping=shader_mapping,
                texture_kind=texture_kind,
                level_data=level_data,
                level_stats=level_stats,
                native_residency_data=native_residency_data,
                quality=quality,
            )
            if not admitted:
                continue
            if native_rendered is not None:
                session.remember_native_preview_result(native_rendered)
            admitted_any = True
            admitted_tiles.append(int(tile_number))
            admitted_keys[int(tile_number)] = key
        # A complete shared preview owns one result for the whole required
        # scope. Build its display payloads with the same scope shape:
        # ensure_floor_payloads constructs the visible-tile lookup once per
        # call, so invoking it per row turns 272 tiny planes into 272 full-set
        # scans before the first commit.
        session._ensure_floor_payloads(
            admitted_tiles,
            preferred_keys=admitted_keys,
        )
        first_pass_rendered = []
        for tile_number in admitted_tiles:
            display_payload = session.display_tile_payloads.get(int(tile_number))
            if display_payload is not None:
                rendered = self.renderer._rendered_tile_for_current_payload(
                    session,
                    int(tile_number),
                    display_payload,
                )
                if rendered is not None:
                    first_pass_rendered.append(rendered)
        if is_preview:
            expected_sources = {
                int(source)
                for source in tuple(getattr(session, "level_expected_indices", ()) or ())
            }
            cohort_rendered = tuple(first_pass_rendered)
            cohort_sources = {int(rendered.tile.source_index) for rendered in cohort_rendered}
            if cohort_sources != expected_sources and len(session.display_tile_payloads) >= len(
                expected_sources
            ):
                by_source = {}
                for current_tile, display_payload in tuple(session.display_tile_payloads.items()):
                    rendered = self.renderer._rendered_tile_for_current_payload(
                        session,
                        int(current_tile),
                        display_payload,
                    )
                    if rendered is not None:
                        by_source[int(rendered.tile.source_index)] = rendered
                if frozenset(by_source) == frozenset(expected_sources):
                    cohort_rendered = tuple(by_source[source] for source in expected_sources)
            self.renderer._admit_preview_cohort_level_evidence(
                session,
                cohort_rendered,
            )
        elif len(first_pass_rendered) > 1:
            self.renderer._admit_first_pass_level_evidence_batch(
                session,
                first_pass_rendered,
                quality=quality,
            )
        elif first_pass_rendered:
            self.renderer._admit_first_pass_level_evidence(
                session,
                first_pass_rendered[0],
                quality=quality,
            )
        for tile_number in admitted_tiles:
            pending_upserted = int(tile_number) in session.pending_payload_upserts
            preview_upserted = bool(
                is_preview
                and pending_upserted
                and str(
                    getattr(session.display_tile_payloads.get(int(tile_number)), "quality", "exact")
                )
                == "preview"
            )
            visible_previews += int(preview_upserted)
            upserted = upserted or pending_upserted
        if visible_previews:
            session.lod_preview_presentations = int(
                getattr(session, "lod_preview_presentations", 0) or 0
            ) + int(visible_previews)
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
            atomic_successor_pending=bool(getattr(session, "atomic_successor_pending", False)),
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
        forget_admission_verdict(session)
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
                retained_payloads = (
                    renderer._retained_tiled_payload_store().payloads_by_base_source(
                        lod_factor=None if reuse_any_lod else selected_lod_factor
                    )
                )
                if retained_payloads:
                    previous_payloads.update(retained_payloads)
                if previous_payloads:
                    session.seed_display_tile_payloads(
                        previous_payloads,
                        tile_source_ids,
                        tile_numbers=tuple(session.dirty_payloads),
                    )
                    # The pipeline retarget already owns the round's LOD
                    # decision and swap obligations. Re-running the complete
                    # 272-tile ladder scan inside the first presentation
                    # callback made a one-item governed target delta take
                    # 100+ ms before backend admission. Seed only; subsequent
                    # governed continuations consume the obligations already
                    # published by the retarget owner.
            payload_seed_done = perf_counter()
            base_tile_state = session.tile_presentation_state
            fast_drain = persistent_tile_layer_fast_drain_enabled(renderer, session)
            renderer._persistent_tile_layer_fast_drain_last_enabled = bool(fast_drain)
            renderer._persistent_tile_layer_fast_drain_enabled_count = int(
                getattr(renderer, "_persistent_tile_layer_fast_drain_enabled_count", 0) or 0
            ) + int(bool(fast_drain))
            capabilities = image_view_backend_capabilities(renderer.win.img_view)
            atomic_successor_pending = _atomic_successor_handoff_pending(session)
            cpu_atomic_successor, shader_atomic_successor = _atomic_successor_commit_modes(
                capabilities,
                pending=atomic_successor_pending,
            )
            limits = tile_layer_upsert_limits(renderer, session)
            if cpu_atomic_successor or shader_atomic_successor:
                # Hidden warming bounds GPU uploads; the eventual source-slot
                # handoff itself is one complete transaction and must not be
                # built once under the upload cap and then rebuilt unbounded.
                limits = {}
            renderer._last_montage_commit_max_upserts = int(
                dict(limits or {}).get("max_upserts", 0) or 0
            )
            renderer._last_montage_commit_unbounded_reason = (
                "atomic_successor"
                if cpu_atomic_successor or shader_atomic_successor
                else "first_cpu_frame"
                if not bool(getattr(session, "shader_display", False))
                and not bool(getattr(session, "display_committed", False))
                else ""
            )
            cold_deadline_ms = None
            renderer._last_montage_pass_budget_ms = None
            renderer._last_montage_governor_details = ()
            if limits:
                limits = dict(limits)
                cold_deadline_ms = limits.pop("cold_deadline_ms", None)
                renderer._last_montage_pass_budget_ms = limits.pop("pass_budget_ms", None)
                renderer._last_montage_governor_details = tuple(
                    limits.pop("governor_details", ()) or ()
                )
            cpu_backend = not bool(capabilities.shader_windowing)
            renderer._last_montage_atomic_successor_pending_before = bool(atomic_successor_pending)
            renderer._last_montage_shader_atomic_successor = bool(shader_atomic_successor)
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
                current_levels = normalize_bounds(getattr(current_level_source, "levels", None))
                if current_level_source is not None:
                    current_level_source = (
                        WindowLevelController()
                        .decide(
                            previous=getattr(session, "applied_level_source", None),
                            candidate=current_level_source,
                            explicit_auto=bool(getattr(session, "force_auto", False)),
                            mode=session.window_mode,
                        )
                        .as_level_source()
                    )
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
                current_level_source = shader_commit_level_source(
                    renderer,
                    session,
                    rehydrate_max_count=dict(limits or {}).get("max_upserts"),
                )
                if current_level_source is not None:
                    current_level_source = (
                        WindowLevelController()
                        .decide(
                            previous=getattr(session, "applied_level_source", None),
                            candidate=current_level_source,
                            explicit_auto=bool(getattr(session, "force_auto", False)),
                            mode=session.window_mode,
                        )
                        .as_level_source()
                    )
                    # The applied source and convergence target are separate:
                    # the target may already name these levels while a prior
                    # display decision still owns the applied source. Publish
                    # every accepted controller decision, not only revisions.
                    session.applied_level_source = current_level_source
                current_levels = normalize_bounds(getattr(current_level_source, "levels", None))
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
                    level_bind_tiles = tuple(
                        dict.fromkeys(
                            (
                                *(int(tile) for tile in dirty_tiles),
                                *(
                                    int(tile)
                                    for tile in getattr(session, "pending_payload_upserts", ())
                                ),
                            )
                        )
                    )
                    session.bind_payloads_to_level_generation(
                        level_bind_tiles,
                        max_count=dict(limits or {}).get("max_upserts"),
                    )
            payload_level_done = perf_counter()
            renderer._last_montage_commit_dirty_before = len(
                getattr(session, "dirty_payloads", ()) or ()
            )
            renderer._last_montage_commit_pending_before = len(
                getattr(session, "pending_payload_upserts", ()) or ()
            )
            renderer._last_montage_commit_presented_before = len(session.lifecycle.presented_tiles)
            prepared_atomic = getattr(session, "_atomic_prepared_transaction", None)
            prepared_atomic_current = bool(
                (cpu_atomic_successor or shader_atomic_successor)
                and _prepared_atomic_transaction_current(
                    session,
                    prepared_atomic,
                )
            )
            renderer._last_montage_atomic_prepared_reused = bool(prepared_atomic_current)
            renderer._last_montage_atomic_fast_built = False
            renderer._last_montage_atomic_fast_reject_reason = ""
            payload_build_call_start = perf_counter()
            if prepared_atomic_current:
                base_tile_state = prepared_atomic["base_tile_state"]
                tile_state = prepared_atomic["tile_state"]
                tile_delta = prepared_atomic["tile_delta"]
            else:
                session._atomic_prepared_transaction = None
                fast_atomic = (
                    session.build_atomic_successor_presentation()
                    if cpu_atomic_successor or shader_atomic_successor
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
                    if cpu_atomic_successor or shader_atomic_successor:
                        tile_delta = replace(tile_delta, atomic_handoff=True)
            payload_build_call_done = perf_counter()
            tile_delta = _priority_ordered_tile_delta(session, tile_delta)
            payload_priority_done = perf_counter()
            active_payloads = tile_state.active_payloads(tile_delta)
            payload_state_done = perf_counter()
            first_display_commit = not bool(session.display_committed)
            renderer._last_montage_commit_first_display = bool(first_display_commit)
            renderer._last_montage_commit_delta_upserts = len(tile_delta.upserts)
            if persistent_tile_residency_backend(renderer, session):
                upserted_tiles = {int(tile) for tile in tile_delta.upserts}
                dirty_tiles = tuple(
                    int(tile) for tile in dirty_tiles if int(tile) in upserted_tiles
                )
            atomic_required_tiles = (
                frozenset(int(tile) for tile in session.atomic_successor_required_scope())
                if cpu_atomic_successor or shader_atomic_successor
                else frozenset()
            )
            atomic_current_tiles = frozenset(
                int(tile)
                for tile, payload in active_payloads.items()
                if session.lifecycle.payload_is_current(int(tile), payload)
            )
            if atomic_required_tiles and not atomic_required_tiles.issubset(atomic_current_tiles):
                # The generic builder can legitimately return an empty active
                # scope after rejecting stale crop wrappers. Comparing against
                # that delta's own empty scope made the transaction look
                # complete and submitted a zero-tile atomic handoff, hiding
                # the complete predecessor. The transition owner already
                # froze the physical obligation; only that required scope can
                # authorize the backend swap.
                session.final_commit_pending = False
                session.flush_pending = False
                renderer._last_montage_tile_payload_build_ms = (
                    perf_counter() - payload_start
                ) * 1000.0
                self._note_commit_bail(
                    "atomic-successor-wait",
                    wakeup="replan",
                    active_payloads=len(atomic_current_tiles),
                    active_tiles=len(atomic_required_tiles),
                )
                renderer.request_montage_replan(session)
                return
            explicit_auto = bool(getattr(session, "force_auto", False) and requested_levels is None)
            level_key = getattr(session, "level_key", None)
            existing_level_stats = (
                None
                if level_key is None
                else renderer._montage_level_tracker().summary_for(level_key)
            )
            level_stats_seeded = bool(
                existing_level_stats is not None and existing_level_stats.source_indices
            )
            level_payloads = (
                active_payloads
                if first_display_commit or not level_stats_seeded
                else dict(tile_delta.upserts)
            )
            if level_payloads:
                renderer._queue_montage_level_stats_for_payloads(session, level_payloads)
            if self._empty_first_commit_can_wait(
                first_display_commit, explicit_auto, active_payloads, tile_delta
            ):
                session.final_commit_pending = False
                session.flush_pending = False
                renderer._last_montage_tile_payload_build_ms = (
                    perf_counter() - payload_start
                ) * 1000.0
                self._note_commit_bail("empty-first-commit-wait", wakeup="replan")
                renderer.request_montage_replan(session)
                return
            prepare_apply_start = perf_counter()
            prepare_stats_start = perf_counter()
            level_stats = renderer._montage_level_stats_for_session(session)
            renderer._last_montage_tile_prepare_stats_ms = (
                perf_counter() - prepare_stats_start
            ) * 1000.0
            semantic_commit = tiled_payloads_include_semantics(active_payloads)
            frame_commit = tiled_payloads_can_commit_frame(active_payloads)
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
                explicit_auto
                and metadata_can_advance
                and (first_display_commit or level_metadata_improved)
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
                level_metadata_improved and bool(getattr(level_stats, "refined", False))
            )
            histogram_metadata_pending = bool(getattr(session, "histogram_metadata_pending", False))
            publish_histogram_plot = bool(
                publish_first_pass_histogram
                or publish_refined_histogram
                or histogram_metadata_pending
            )
            publish_metadata = (
                publish_auto_metadata or publish_histogram_plot or level_metadata_improved
            )
            # Instrumentation: record why a commit did/did not (re)apply level
            # metadata, so the levels/histogram-stranding path is observable from
            # runtime diagnostics without a debugger.
            renderer._last_montage_level_decision = {
                "first_display_commit": bool(first_display_commit),
                "semantic_commit": bool(semantic_commit),
                "frame_commit": bool(frame_commit),
                "decision_force_auto": bool(decision_force_auto),
                "explicit_auto": bool(explicit_auto),
                "metadata_can_advance": bool(metadata_can_advance),
                "semantic_level_supersession": bool(semantic_level_supersession),
                "level_metadata_improved": bool(level_metadata_improved),
                "histogram_metadata_pending": bool(histogram_metadata_pending),
                "publish_histogram_plot": bool(publish_histogram_plot),
                "publish_metadata": bool(publish_metadata),
                "level_stats_rank": None
                if level_stats is None
                else str(getattr(level_stats, "rank", None)),
                "level_stats_quality": int(getattr(level_stats, "evidence_quality", 0) or 0),
                "active_payload_count": len(dict(active_payloads or {})),
            }
            renderer._montage_level_decision_count = (
                int(getattr(renderer, "_montage_level_decision_count", 0)) + 1
            )
            renderer._last_montage_tile_prepare_metadata_ms = (
                perf_counter() - prepare_metadata_start
            ) * 1000.0
            if (
                self._empty_progressive_commit_settled(
                    first_display_commit, explicit_auto, tile_delta
                )
                and not publish_metadata
            ):
                session.final_commit_pending = False
                session.flush_pending = False
                renderer._last_montage_tile_payload_build_ms = (
                    perf_counter() - payload_start
                ) * 1000.0
                session.note_committed()
                # The required scope is settled with nothing to upsert, so no
                # backend acknowledgement will follow this cycle.  Re-affirm
                # targets closed by retained payloads (idempotent per
                # requirement) so whole-workflow replay sees the closure even
                # for presentation state restored outside the ordinary
                # retarget/acknowledge events.
                session.lifecycle.note_retained_satisfaction(session.required_tile_numbers())
                self._note_commit_bail("empty-progressive-settled", wakeup="noop-finish")
                self._finish_after_noop_commit()
                return
            rendered_geometry = replace(
                rendered_geometry, montage_tile_states=session.ensure_tile_states()
            )
            renderer._last_montage_tile_payload_build_ms = (perf_counter() - payload_start) * 1000.0
            renderer._last_montage_tile_payload_seed_ms = (
                payload_seed_done - payload_start
            ) * 1000.0
            renderer._last_montage_tile_payload_level_ms = (
                payload_level_done - payload_seed_done
            ) * 1000.0
            renderer._last_montage_tile_payload_state_ms = (
                payload_state_done - payload_level_done
            ) * 1000.0
            renderer._last_montage_tile_payload_build_call_ms = (
                payload_build_call_done - payload_build_call_start
            ) * 1000.0
            renderer._last_montage_tile_payload_priority_ms = (
                payload_priority_done - payload_build_call_done
            ) * 1000.0
            renderer._last_montage_tile_payload_active_ms = (
                payload_state_done - payload_priority_done
            ) * 1000.0
            prepare_source_start = perf_counter()
            # A cold CPU-windowed first commit may window from a provisional
            # refined subset (first-batch acceptance in
            # tile_layer_first_pixels_wait_for_level_source); the partial
            # source must therefore be visible to the windowing decision on
            # exactly that commit, not only when metadata publication is due.
            first_cpu_commit_source = bool(
                first_display_commit and not bool(capabilities.shader_windowing)
            )
            semantic_source = renderer._montage_level_source_for_session(
                session,
                allow_partial=bool(publish_metadata or first_cpu_commit_source),
                rehydrate_max_count=(
                    dict(limits or {}).get("max_upserts")
                    if not bool(capabilities.shader_windowing)
                    else 0
                ),
            )
            renderer._last_montage_tile_prepare_source_ms = (
                perf_counter() - prepare_source_start
            ) * 1000.0
            prepare_histogram_start = perf_counter()
            histogram_plot_data = (
                renderer._montage_histogram_plot_data_for_session(
                    session, allow_partial=publish_metadata
                )
                if publish_histogram_plot
                else None
            )
            renderer._last_montage_tile_prepare_histogram_ms = (
                perf_counter() - prepare_histogram_start
            ) * 1000.0
            if tile_layer_first_pixels_wait_for_level_source(
                renderer,
                session,
                first_display_commit,
                level_stats,
            ):
                capabilities = image_view_backend_capabilities(renderer.win.img_view)
                publish_rough_histogram = getattr(
                    renderer.win.img_view, "applyHistogramMetadata", None
                )
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
                    renderer._note_montage_level_source_applied(
                        session, semantic_source, explicit=False
                    )
                session.final_commit_pending = True
                session.flush_pending = True
                if (
                    not getattr(session, "pending_level_tiles", None)
                    and int(getattr(session, "level_scan_remaining_tiles", 0) or 0) <= 0
                ):
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
            atomic_successor = bool(cpu_atomic_successor or shader_atomic_successor)
            renderer._last_montage_atomic_successor = bool(atomic_successor)
            resident_predicate = getattr(renderer.win.img_view, "tiledPayloadResident", None)
            cold_gpu_successor = _cold_gpu_successor_requires_hidden_warm(
                session=session,
                cpu_backend=cpu_backend,
                resident_predicate=resident_predicate,
                upserts=dict(tile_delta.upserts or {}),
            )
            if cold_gpu_successor and interactive_active(renderer):
                # Keep the acknowledged pixels during the gesture. The
                # interaction-stop edge replans this exact dirty transaction;
                # hidden residency then absorbs allocation/upload cost before
                # the semantic swap reaches the canvas.
                session._interactive_residency_deferred = True
                session.final_commit_pending = False
                session.flush_pending = False
                self._note_commit_bail(
                    "interactive-residency-deferred", wakeup="interaction-stop-edge"
                )
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
                    rgb_already_windowed=bool(
                        getattr(display_image, "rgb_already_windowed", False)
                    ),
                    payloads=(
                        active_payloads if atomic_successor else dict(tile_delta.upserts or {})
                    ),
                    batch_size=2,
                )
            ):
                session._atomic_prepared_transaction = {
                    "session_id": int(getattr(session, "session_id", 0) or 0),
                    "level_revision": int(
                        getattr(getattr(session, "level_generation", None), "revision", 0) or 0
                    ),
                    "marker_kind": (
                        "cpu-compatible" if cpu_atomic_successor else "shader-successor"
                    ),
                    "base_tile_state": base_tile_state,
                    "tile_state": tile_state,
                    "tile_delta": tile_delta,
                    "payload_markers": {
                        int(tile): (
                            _cpu_transaction_payload_marker(payload)
                            if cpu_atomic_successor
                            else _shader_successor_transaction_payload_marker(payload)
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
                self._note_commit_bail(
                    "hidden-warm-residency-wait", wakeup="warm-residency-continuation"
                )
                return
            renderer._last_montage_tile_prepare_apply_ms = (
                perf_counter() - prepare_apply_start
            ) * 1000.0
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
                semantic_commit=frame_commit,
                decision_force_auto=decision_force_auto,
            )
            if not applied:
                self._note_commit_bail("backend-declined", wakeup="rearm-if-backlog")
                return
            if histogram_metadata_pending and histogram_plot_data is not None:
                session.histogram_metadata_pending = False
            session.backend_refresh_pending = False
            renderer._last_montage_commit_outcome = "backend-applied"
            session._atomic_prepared_transaction = None
            self._acknowledge_and_publish(
                tile_delta,
                tile_state,
                rendered_geometry,
                active_payloads,
                commit_start=commit_start,
                atomic_successor=atomic_successor,
                first_pass_histogram_published=bool(
                    publish_first_pass_histogram and histogram_plot_data is not None
                ),
            )
        except BaseException as exc:
            # A throw is an early commit return that names neither its outcome
            # nor its wakeup, so it was the one commit exit invisible to the
            # trace. It reported `commit_outcome='started'` forever and the
            # stall guard four seconds later described a lost wakeup — the
            # same dump a real page-pool exhaustion and a typo'd AttributeError
            # both produced (dossier wgpu-pool-layer-leak-2026-07-26 §5a).
            # Name it like every other terminal bail and re-raise: the policy
            # decision about fatality belongs to the gate, not to this frame.
            self._note_commit_raised(exc, dirty_tiles=dirty_tiles)
            raise
        finally:
            _finish_presentation_commit(renderer)

    def _note_commit_raised(self, exc: BaseException, *, dirty_tiles=()) -> None:
        """Record a commit throw as the terminal bail it is.

        Deliberately NOT a wakeup: re-arming would replay a delta that is about
        to throw again at full flush rate, which the rescue ADR 0051 forbids.
        In-flight worker completions can nevertheless request presentation
        after this frame returns, so the terminal mark is generation-scoped:
        it blocks this failed owner without poisoning a successor generation.
        """

        renderer = self.renderer
        session = self.session
        tiles = tuple(int(tile) for tile in tuple(dirty_tiles or ()))
        renderer._montage_commit_failed_owner = (
            int(getattr(session, "session_id", 0) or 0),
            id(session),
        )
        renderer._last_montage_commit_exception = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback_text(exc),
            "session_id": int(getattr(session, "session_id", 0) or 0),
            "semantic_key": repr(getattr(session, "key", None)),
            "committing_tiles": tiles[:16],
            "committing_tile_count": len(tiles),
        }
        self._note_commit_bail(
            "raised",
            wakeup="none (terminal)",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            committing_tile_count=len(tiles),
        )

    def _configure_wgpu_evidence_obligation(self) -> None:
        """Install, defer, or clear the phase-1 resident-histogram obligation."""

        renderer = self.renderer
        session = self.session
        configure_gpu_evidence = getattr(
            renderer.win.img_view,
            "setResidentHistogramEvidenceRequired",
            None,
        )
        if not callable(configure_gpu_evidence):
            return
        gpu_evidence_required = bool(
            session.scheduling_policy.verdict.coverage_open
            and not getattr(session, "first_pass_histogram_published", False)
        )
        gpu_evidence_deferred = bool(gpu_evidence_required and interactive_active(renderer))
        session._wgpu_histogram_evidence_deferred = gpu_evidence_deferred
        # The backend evidence key already includes the committed plane
        # identity, resident frontier, and shader mapping. The phase key
        # supplies only the semantic population; session/viewport
        # generations are presentation ownership and must not churn a
        # content obligation during zoom/pan.
        evidence_obligation = (
            "wgpu-resident-histogram",
            session.level_key,
        )
        configure_gpu_evidence(
            gpu_evidence_required and not gpu_evidence_deferred,
            evidence_obligation if gpu_evidence_required else None,
        )
        if gpu_evidence_deferred:
            # A deferred COLD obligation is still phase-1 evidence debt.
            # Without the barrier, first pixels close COVERAGE evidence-empty
            # and the quiet-edge forced commit lands in REFINE, where
            # ``gpu_evidence_required`` is gated off — the rough histogram
            # never dispatches (codex review 2026-07-19, finding 1). The
            # publication transition in ``_acknowledge_and_publish`` releases
            # the barrier; a scope retarget resets it.
            session.scheduling_policy.set_coverage_evidence_pending(True)
            emit_trace(
                "wgpu_histogram_deferred",
                reason="interaction_active",
                session_id=int(getattr(session, "session_id", 0) or 0),
            )

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
        self._configure_wgpu_evidence_obligation()
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
            applied = True
        elif first_display_commit:
            applied = renderer._apply_full_display_image(
                display_image,
                geometry=geometry,
                window_mode=session.window_mode,
                previous_frame=getattr(renderer.win, "_committed_display_frame", None),
                force_auto=decision_force_auto,
                defer_side_panels=getattr(session, "defer_side_panels", False),
                semantic_source=semantic_source,
                applied_level_source=applied_level_source,
                histogram_plot_data=histogram_plot_data,
                commit_kind=CommitKind.EXPLICIT_AUTO_WINDOW
                if decision_force_auto
                else CommitKind.FULL_FRAME_INITIAL,
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
            applied = True
        else:
            applied = renderer._apply_progressive_display_image(
                display_image,
                geometry=geometry,
                window_mode=session.window_mode,
                previous_frame=getattr(renderer.win, "_committed_display_frame", None),
                force_auto=False,
                viewport_policy=ViewportPolicy.PRESERVE,
                semantic_source=semantic_source,
                applied_level_source=applied_level_source,
                histogram_plot_data=histogram_plot_data,
                commit_kind=CommitKind.EXPLICIT_AUTO_WINDOW
                if explicit_auto
                else CommitKind.PROGRESSIVE_FRAME_PATCH,
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
        # A commit the presenter caught and reported must decline here rather
        # than fall through to acknowledgement: there is no report describing
        # this delta, and the caller would otherwise acknowledge whatever the
        # previous transaction left behind.
        return bool(applied)

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
        if previous_frame is None or not isinstance(
            getattr(previous_frame, "value_source", None), TiledValueSource
        ):
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
                    rgb_already_windowed=bool(
                        getattr(display_image, "rgb_already_windowed", False)
                    ),
                    histogram_plot_data=histogram_plot_data,
                    tile_state=tile_state,
                    base_tile_state=base_tile_state,
                    tile_delta=tile_delta,
                    tile_residency_budget_bytes=tile_residency_budget_bytes(
                        renderer._memory_policy()
                    ),
                ),
                context=context,
                previous_frame=previous_frame,
                window_mode=session.window_mode,
                force_auto=False,
                commit_kind=CommitKind.EXPLICIT_AUTO_WINDOW
                if explicit_auto
                else CommitKind.PROGRESSIVE_FRAME_PATCH,
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
        # An X/Y axis-order swap on a canonical backend re-presents no tiles (the
        # payloads are unchanged), but the display transform and geometry DID
        # change, so the committed CPU frame -- its geometry and the value
        # source's ``transposed`` mapping -- must be rebuilt or hover/ROI keep
        # reading the pre-swap orientation off a stale frame.
        new_transposed = _session_display_transposed(session)
        orientation_changed = new_transposed != bool(
            getattr(getattr(previous_frame, "value_source", None), "transposed", False)
        )
        semantic_frame_commit = bool(
            semantic_commit and bool(getattr(report, "presented_tiles", ()))
        )
        if semantic_frame_commit or orientation_changed:
            committed_state = getattr(committer, "last_tile_committed_state", None)
            payloads = getattr(committed_state, "payloads", None)
            if not payloads:
                payloads = getattr(tile_state, "payloads", None)
            if payloads:
                frame = replace(
                    previous_frame,
                    key=context.frame_key,
                    geometry=geometry,
                    levels=decision.levels,
                    histogram_range=decision.histogram_range,
                    value_source=TiledValueSource(payloads, transposed=new_transposed),
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
        atomic_successor: bool,
        first_pass_histogram_published: bool,
    ) -> None:
        renderer = self.renderer
        session = self.session
        coverage_phase_before = bool(session.scheduling_policy.verdict.coverage_open)
        presented_before_commit = frozenset(
            int(tile)
            for tile in session.required_tile_numbers()
            if session.lifecycle.first_pixels_presented((int(tile),))
        )
        report = getattr(renderer._display_committer(), "last_tile_commit_report", None)
        renderer._last_montage_report_presented = len(
            tuple(getattr(report, "presented_tiles", ()) or ())
        )
        renderer._last_montage_report_committed = len(
            tuple(getattr(report, "committed_upserts", ()) or ())
        )
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
        acknowledged = session.acknowledge_tile_presentation(
            tile_delta, report, levels=committed_levels
        )
        # The acknowledged backend snapshot owns first-pass quality on every
        # backend. PyQtGraph complex correctly declines a reduced RGB rung;
        # its exact-only pass therefore has no FLOOR claim to seed quality,
        # and a WGPU-only observation leaves COVERAGE open forever after the
        # already-correct exact frame is on screen.
        session.observe_physically_presented_first_pass_quality(active_payloads)
        phase_closed = False
        if atomic_successor:
            # A successor is complete only after the lifecycle accepts
            # one coverage-complete backend report.  Backend submission alone
            # cannot suppress the next attempt after a stale/partial commit.
            session.acknowledge_atomic_successor(
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
        retained_budget = int(getattr(session, "tile_residency_budget_bytes", 0) or 0)
        renderer._retained_tiled_payload_store().remember_acknowledged(
            accepted_payloads,
            max_bytes=retained_budget if retained_budget > 0 else None,
        )
        renderer._last_montage_tile_retained_store_ms = (perf_counter() - retained_start) * 1000.0
        state_start = perf_counter()
        presented_tiles = (
            active_payloads
            if report is None
            else getattr(report, "presented_tiles", active_payloads)
        )
        session.mark_presented(presented_tiles)
        presented_after = set(session.lifecycle.presented_tiles)
        first_pixels_presented = bool(session.required_first_pixels_presented())
        first_pixel_transition = bool(first_pixels_presented and not first_pixels_before)
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
                if callable(refresh_extent_intent) and bool(refresh_extent_intent()):
                    # The range-change signal fires while the enclosing
                    # commit intentionally suppresses viewport retargeting.
                    # Preserve that semantic obligation and replay it only
                    # after acknowledgement/commit teardown, otherwise the
                    # camera shows the successor extent while the active
                    # plan remains the predecessor's partial tile set.
                    renderer._frame_viewport_retarget_after_commit = True
        if (
            committed_levels is not None
            and image_view_backend_capabilities(renderer.win.img_view).shader_windowing
        ):
            session.update_level_presentation_scope()
            session.acknowledge_uniform_level_presentation(committed_levels)
        if not session.has_stale_level_presentations():
            session.set_level_update_pending(False)
        resident_gpu_evidence = getattr(
            renderer.win.img_view,
            "residentHistogramEvidence",
            None,
        )
        gpu_evidence_waiting = bool(
            callable(resident_gpu_evidence) and resident_gpu_evidence(active_payloads)
        )
        if gpu_evidence_waiting:
            session.scheduling_policy.set_coverage_evidence_pending(True)
        if accepted_payloads or gpu_evidence_waiting:
            # Evidence quality can advance only after the backend accepts the
            # payload. Every accepted delta is accumulated by the level
            # tracker, so rescanning the complete active population on every
            # bounded target commit adds an O(active tiles) fixed term without
            # adding evidence. A metadata-only GPU evidence wake still scans
            # the active population because it has no accepted delta owner.
            evidence_payloads = accepted_payloads if accepted_payloads else active_payloads
            renderer._queue_montage_level_stats_for_payloads(session, evidence_payloads)
        first_pass_publication_transition = bool(
            first_pass_histogram_published
            and not bool(getattr(session, "first_pass_histogram_published", False))
        )
        if first_pass_publication_transition:
            session.first_pass_histogram_published = True
            session.scheduling_policy.set_coverage_evidence_pending(False)
        elif renderer._first_pass_level_evidence_complete(session) and not bool(
            getattr(session, "first_pass_histogram_published", False)
        ):
            # The final first-pass acknowledgement makes the rough histogram
            # eligible.  Preserve that metadata-only obligation before the
            # preview transition can replan target quality.
            session.flush_pending = True
            session.final_commit_pending = True
        phase_closed = session.scheduling_policy.observe(
            session,
            on_refinement_replan=lambda: renderer.request_montage_replan(session),
        )
        session.display_committed = bool(session.lifecycle.presented_tiles)
        if phase_closed:
            # Coverage close reclassifies any remaining evidence batches from
            # coverage work to refinement, which changes the lane they are
            # admitted on. Re-arm the owner on that edge so the reclassified
            # remainder is scheduled promptly; the sweep is never parked
            # waiting for this edge, it only changes lanes across it.
            renderer._schedule_semantic_level_evidence(session)
        semantic_progress = getattr(session, "semantic_level_evidence_progress", None)
        semantic_evidence_waiting = bool(
            semantic_progress is not None
            and (
                semantic_progress.inflight_generation is not None
                or int(semantic_progress.pending_batches) > 0
            )
        )
        if first_pixel_transition or phase_closed or semantic_evidence_waiting:
            # Physical first-pixel completion changes DISPLAY_PREPARATION and
            # side-work eligibility without a kernel completion after this
            # acknowledgement. Refined semantic evidence has the same shape.
            # Reconcile the existing quotas at their canonical lifecycle edge
            # rather than polling or letting the preview-era quota strand the
            # exact rung planned by the follow-up replan.
            reconcile_quotas = getattr(renderer.win, "_apply_resource_governor_decisions", None)
            if callable(reconcile_quotas):
                reconcile_quotas(refresh_telemetry=False)
        if phase_closed:
            # Coverage closes inside the presentation transaction. The
            # immediate reconciliation above can still see that transaction's
            # work as busy and legitimately leave refinement lanes at quota
            # zero. No kernel completion is owed after this callback, so a
            # queued histogram or target task would otherwise remain parked
            # forever while the screen is already correct.
            scheduling_generation = int(session.scheduling_policy.verdict.generation)
            session_id = int(session.session_id)

            def reconcile_after_phase_close():
                current = getattr(renderer.win, "_frame_session", None)
                if current is not session or int(getattr(current, "session_id", 0)) != session_id:
                    return
                verdict = session.scheduling_policy.verdict
                if int(verdict.generation) != scheduling_generation or bool(verdict.coverage_open):
                    return
                reconcile = getattr(
                    renderer.win,
                    "_apply_resource_governor_decisions",
                    None,
                )
                if callable(reconcile):
                    reconcile(refresh_telemetry=False)

            _post_low_priority_callback(renderer, reconcile_after_phase_close)
        geometry = replace(geometry, montage_tile_states=session.ensure_tile_states())
        renderer._last_montage_tile_state_publish_ms = (perf_counter() - state_start) * 1000.0
        geometry_start = perf_counter()
        renderer._sync_committed_montage_geometry(
            geometry,
            semantic_commit=tiled_payloads_can_commit_frame(active_payloads),
        )
        renderer._last_montage_tile_geometry_sync_ms = (perf_counter() - geometry_start) * 1000.0
        if not bool(getattr(session, "display_committed", False)):
            renderer.refresh_montage_priority_targets(session)
        overlay_start = perf_counter()
        rect = montage_rect_for_viewport(
            session.plan, view_range=session.view_range, viewport_shape=session.viewport_shape
        )
        renderer._update_montage_tile_overlays_for_plan(
            session.plan, tuple(session.tile_states), rect
        )
        renderer._last_montage_overlay_update_ms = (perf_counter() - overlay_start) * 1000.0
        self._finish_commit(
            report,
            tile_state,
            tile_delta,
            commit_start=commit_start,
            preview_transition=preview_transition,
            coverage_phase_before=coverage_phase_before,
            presented_before_commit=presented_before_commit,
        )

    def _finish_commit(
        self,
        report,
        tile_state,
        tile_delta,
        *,
        commit_start: float,
        preview_transition: bool,
        coverage_phase_before: bool = False,
        presented_before_commit: frozenset[int] = frozenset(),
    ) -> None:
        renderer = self.renderer
        session = self.session
        identity_start = perf_counter()
        identity_mismatches = session.backend_identity_mismatch_tiles()
        renderer._last_montage_tile_identity_check_ms = (perf_counter() - identity_start) * 1000.0
        if identity_mismatches:
            renderer._montage_identity_repair_commits = (
                int(getattr(renderer, "_montage_identity_repair_commits", 0) or 0) + 1
            )
            session.final_commit_pending = True
            session.flush_pending = True
        rearmed_parked = tuple(
            getattr(session, "rearm_visible_parked_payloads", lambda: ())() or ()
        )
        renderer._last_montage_tile_commit_ms = (perf_counter() - commit_start) * 1000.0
        priority_ranks = dict(getattr(tile_delta, "priority_ranks", {}) or {})
        lod_decision = getattr(session, "lod_policy_decision", None)
        lod_demand = getattr(lod_decision, "demand", None)
        chunk_qualities = {
            str(getattr(payload, "quality", "exact") or "exact")
            for payload in dict(getattr(tile_delta, "upserts", {}) or {}).values()
        }
        pass_kind = (
            "preview"
            if chunk_qualities and chunk_qualities <= {"preview", "fallback"}
            else "target"
            if chunk_qualities == {"exact"}
            else "mixed"
            if chunk_qualities
            else "metadata"
        )
        renderer._last_montage_pass_kind = pass_kind
        pass_chunk_budget_ms = getattr(renderer, "_last_montage_pass_budget_ms", None)
        required_tiles = tuple(int(tile) for tile in session.required_tile_numbers())
        pass_completed_atomically = bool(
            pass_kind in {"preview", "target"}
            and len(required_tiles) > 1
            and {int(tile) for tile in tile_delta.upserts} == set(required_tiles)
        )
        emit_trace(
            "commit_batch",
            phase="backend_complete",
            session_id=int(getattr(session, "session_id", 0) or 0),
            revision=int(getattr(tile_state, "revision", 0) or 0),
            elapsed_ms=float(renderer._last_montage_tile_commit_ms),
            pass_kind=pass_kind,
            pass_chunk_items=len(tuple(tile_delta.upserts)),
            pass_chunk_budget_ms=pass_chunk_budget_ms,
            governor_details=tuple(getattr(renderer, "_last_montage_governor_details", ()) or ()),
            pass_chunk_within_50ms=bool(renderer._last_montage_tile_commit_ms <= 50.0),
            pass_completed_atomically=pass_completed_atomically,
            presented_tiles=tuple(getattr(report, "presented_tiles", ()) or ()),
            committed_upserts=tuple(getattr(report, "committed_upserts", ()) or ()),
            cold_upsert_tiles=tuple(sorted(getattr(report, "cold_upsert_tiles", ()) or ())),
            identity_rejected=tuple(sorted(getattr(report, "identity_rejected_tiles", ()) or ())),
            delta_upserts=tuple(int(tile) for tile in tile_delta.upserts),
            delta_qualities=tuple(
                (
                    int(tile),
                    str(getattr(payload, "quality", "exact") or "exact"),
                    int(getattr(getattr(payload, "lod", None), "level", 0) or 0),
                )
                for tile, payload in tile_delta.upserts.items()
            ),
            delta_priority_ranks=tuple(
                (int(tile), priority_ranks.get(int(tile))) for tile in tile_delta.upserts
            ),
            max_upserts=int(getattr(renderer, "_last_montage_commit_max_upserts", 0) or 0),
            unbounded_reason=str(
                getattr(renderer, "_last_montage_commit_unbounded_reason", "") or ""
            ),
            # What made this batch this size. `max_upserts` is the cap in
            # force; `admission_limit` is the cap that BIT — "" means none did,
            # so the batch carried everything on offer and any smallness is
            # supply, not pacing. Read it with `admission_candidates`.
            admission_limit=str(getattr(session, "_last_admission_limit", "") or ""),
            admission_deferred=int(getattr(session, "_last_admission_deferred", 0) or 0),
            admission_candidates=int(getattr(session, "_last_admission_candidates", 0) or 0),
            first_display_commit=bool(
                getattr(renderer, "_last_montage_commit_first_display", False)
            ),
            # Compatibility fields consumed by the journey oracle. Their
            # value now comes directly from the scheduling-policy verdict;
            # neither commit tracing nor the oracle reconstructs coverage
            # from payload or lifecycle counters.
            preview_pass_open_before=bool(coverage_phase_before),
            coverage_pass_closed=bool(
                coverage_phase_before and not session.scheduling_policy.verdict.coverage_open
            ),
            scheduling_phase_before=("coverage" if coverage_phase_before else "refine"),
            required_tile_count=len(session.required_tile_numbers()),
            preview_missing_tile_count=sum(
                1
                for tile in session.required_tile_numbers()
                if int(tile) not in presented_before_commit
            ),
            first_pass_quality=getattr(session, "first_pass_quality", None),
            first_pass_pixels_presented=bool(session.first_pass_pixels_presented()),
            desired_level=(
                None if lod_demand is None else int(getattr(lod_demand, "desired_level", 0) or 0)
            ),
            applied_level=(
                None
                if lod_decision is None
                else int(getattr(lod_decision, "applied_level", 0) or 0)
            ),
            uploads=int(getattr(report, "texture_uploads", 0) or 0),
            upload_bytes=int(getattr(report, "texture_upload_bytes", 0) or 0),
            backend_cold_work_ms=float(getattr(report, "cold_work_ms", 0.0) or 0.0),
            backend_visibility_work_ms=float(getattr(report, "visibility_work_ms", 0.0) or 0.0),
            backend_total_ms=float(getattr(report, "total_ms", 0.0) or 0.0),
            backend_storage_rebuilds=int(getattr(report, "storage_rebuilds", 0) or 0),
            backend_pool_growth_ms=float(getattr(report, "pool_growth_ms", 0.0) or 0.0),
            backend_executor_initialization_ms=float(
                getattr(report, "executor_initialization_ms", 0.0) or 0.0
            ),
            prepare_ms=float(getattr(renderer, "_last_montage_tile_prepare_apply_ms", 0.0) or 0.0),
            payload_build_ms=float(
                getattr(renderer, "_last_montage_tile_payload_build_ms", 0.0) or 0.0
            ),
            payload_seed_ms=float(
                getattr(renderer, "_last_montage_tile_payload_seed_ms", 0.0) or 0.0
            ),
            payload_level_ms=float(
                getattr(renderer, "_last_montage_tile_payload_level_ms", 0.0) or 0.0
            ),
            payload_state_ms=float(
                getattr(renderer, "_last_montage_tile_payload_state_ms", 0.0) or 0.0
            ),
            payload_build_call_ms=float(
                getattr(renderer, "_last_montage_tile_payload_build_call_ms", 0.0) or 0.0
            ),
            payload_priority_ms=float(
                getattr(renderer, "_last_montage_tile_payload_priority_ms", 0.0) or 0.0
            ),
            payload_active_ms=float(
                getattr(renderer, "_last_montage_tile_payload_active_ms", 0.0) or 0.0
            ),
            prepare_stats_ms=float(
                getattr(renderer, "_last_montage_tile_prepare_stats_ms", 0.0) or 0.0
            ),
            prepare_metadata_ms=float(
                getattr(renderer, "_last_montage_tile_prepare_metadata_ms", 0.0) or 0.0
            ),
            prepare_source_ms=float(
                getattr(renderer, "_last_montage_tile_prepare_source_ms", 0.0) or 0.0
            ),
            prepare_histogram_ms=float(
                getattr(renderer, "_last_montage_tile_prepare_histogram_ms", 0.0) or 0.0
            ),
            backend_apply_ms=float(
                getattr(renderer, "_last_montage_tile_layer_apply_ms", 0.0) or 0.0
            ),
            # The hand-off split inside the backend callback: what a worker
            # could have prepared, and what only the GUI thread can submit.
            backend_texture_prepare_ms=float(getattr(report, "texture_prepare_ms", 0.0) or 0.0),
            backend_texture_submit_ms=float(getattr(report, "texture_submit_ms", 0.0) or 0.0),
            backend_texture_pack_ms=float(getattr(report, "texture_pack_ms", 0.0) or 0.0),
            # Whether the worker hand-off is actually landing. A commit that
            # packs inline because preparation never arrived looks identical to
            # one with no preparation at all, so say which it was.
            **_prepared_upload_counters(renderer),
            acknowledge_ms=float(
                getattr(renderer, "_last_montage_tile_acknowledge_ms", 0.0) or 0.0
            ),
            state_publish_ms=float(
                getattr(renderer, "_last_montage_tile_state_publish_ms", 0.0) or 0.0
            ),
            geometry_sync_ms=float(
                getattr(renderer, "_last_montage_tile_geometry_sync_ms", 0.0) or 0.0
            ),
            resident_rebinds=int(getattr(report, "resident_rebinds", 0) or 0),
            binding_fast_path_commits=int(getattr(report, "binding_fast_path_commits", 0) or 0),
            binding_incremental_commits=int(getattr(report, "binding_incremental_commits", 0) or 0),
            binding_full_republications=int(getattr(report, "binding_full_republications", 0) or 0),
            vertex_uploads=int(getattr(report, "vertex_uploads", 0) or 0),
            level_revision=int(
                getattr(getattr(session, "level_generation", None), "revision", 0) or 0
            ),
            level_target=getattr(getattr(session, "level_generation", None), "target_levels", None),
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
            atomic_successor_pending=bool(session.atomic_successor_pending),
            atomic_successor_pending_before=bool(
                getattr(
                    renderer,
                    "_last_montage_atomic_successor_pending_before",
                    False,
                )
            ),
            shader_atomic_successor=bool(
                getattr(renderer, "_last_montage_shader_atomic_successor", False)
            ),
            atomic_successor=bool(getattr(renderer, "_last_montage_atomic_successor", False)),
            atomic_fast_built=bool(getattr(renderer, "_last_montage_atomic_fast_built", False)),
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
                key=(
                    "montage_backend_commit",
                    session.key,
                    int(session.session_id),
                    int(tile_state.revision),
                    "tile_layer",
                ),
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
            and rejected_signature == getattr(session, "_identity_rejected_delta_signature", None)
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
                tile: value
                for tile, value in backlog_dirty.items()
                if int(tile) not in rejected_tiles
            }
            backlog_pending = {
                tile: value
                for tile, value in backlog_pending.items()
                if int(tile) not in rejected_tiles
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
        processed_count = tile_layer_commit_processed_count(report)
        texture_upload_bytes = int(getattr(report, "texture_upload_bytes", 0) or 0)
        storage_rebuilds = int(getattr(report, "storage_rebuilds", 0) or 0)
        backend_name = image_view_backend_capabilities(renderer.win.img_view).name
        details = _commit_feedback_details(renderer)
        cold_ms = 0.0
        if cold_count > 0 and storage_rebuilds <= 0:
            cold_ms = (
                float(getattr(report, "cold_work_ms", 0.0) or 0.0)
                or renderer._last_montage_tile_commit_ms
            )
            _observe_ui(
                renderer,
                "montage_cold_commit",
                cold_ms,
                cold_count,
                texture_upload_bytes,
                "texture_upload",
                backend_name,
            )
        commit_feedback_ms = renderer._last_montage_tile_commit_ms
        pool_growth_ms = min(
            commit_feedback_ms,
            max(0.0, float(getattr(report, "pool_growth_ms", 0.0) or 0.0)),
        )
        executor_initialization_ms = min(
            max(0.0, commit_feedback_ms - pool_growth_ms),
            max(
                0.0,
                float(getattr(report, "executor_initialization_ms", 0.0) or 0.0),
            ),
        )
        structural_backend_ms = pool_growth_ms + executor_initialization_ms
        steady_render_feedback_ms = max(0.0, commit_feedback_ms - structural_backend_ms)
        commit_feedback_bytes = texture_upload_bytes
        _observe_ui(
            renderer,
            "montage_commit",
            commit_feedback_ms,
            processed_count,
            commit_feedback_bytes,
            "presentation_upsert",
            backend_name,
        )
        # This is the exact channel ResourceGovernor.decide_commit_batch()
        # consumes. Recording only the diagnostic aliases left the governor
        # permanently at its cold-start maximum, so neither preview nor target
        # chunks adapted after an over-budget presentation callback.
        _observe_ui(
            renderer,
            "montage_present_total",
            commit_feedback_ms,
            processed_count,
            commit_feedback_bytes,
            "presentation_upsert",
            backend_name,
            details=details,
        )
        _observe_ui(
            renderer,
            (
                "montage_render_pass_target"
                if str(getattr(renderer, "_last_montage_pass_kind", "")) == "target"
                else "montage_render_pass_preview"
            ),
            steady_render_feedback_ms,
            max(1, int(getattr(renderer, "_last_montage_commit_delta_upserts", 0) or 0)),
            commit_feedback_bytes,
            "presentation_upsert",
            backend_name,
            details=(
                *details,
                f"callback_total_ms={commit_feedback_ms:.6f}",
                f"structural_pool_growth_ms={pool_growth_ms:.6f}",
                f"structural_executor_initialization_ms={executor_initialization_ms:.6f}",
            ),
        )
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

    def _empty_first_commit_can_wait(
        self, first_display_commit, explicit_auto, active_payloads, tile_delta
    ) -> bool:
        session = self.session
        return bool(
            first_display_commit
            and not active_payloads
            and not getattr(tile_delta, "upserts", None)
            and not getattr(tile_delta, "removals", None)
            and not (session.has_pending_level_update() and session.has_stale_level_presentations())
        )

    def _empty_progressive_commit_settled(
        self, first_display_commit, explicit_auto, tile_delta
    ) -> bool:
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
        if not (
            getattr(session, "dirty_payloads", None) or getattr(session, "pending_removals", None)
        ):
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
        return self.session._lod_page_set_key_for(rendered, demand=demand, level=int(step.level))

    def _intent_matches_session(self, intent) -> bool:
        return intent is None or getattr(
            intent, "semantic_key", getattr(self.session, "key", None)
        ) == getattr(self.session, "key", None)

    def _session_is_current(self, intent=None) -> bool:
        if not self._intent_matches_session(intent):
            return False
        predicate = getattr(self.renderer, "_frame_session_is_current", None)
        if callable(predicate):
            return bool(predicate(self.session))
        return True


def _priority_ordered_tile_delta(session, tile_delta):
    """Freeze current-camera order at the final backend command boundary."""

    if not tile_delta.upserts:
        return tile_delta
    plan_tiles = tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ())
    context = session.tile_priority_context()
    cache_key = (id(getattr(session, "plan", None)), id(context))
    cached = getattr(session, "_presentation_priority_rank_cache", None)
    if cached is not None and cached[0] == cache_key:
        priority_ranks = cached[1]
    else:
        priority_order = prioritize_tile_numbers(
            range(len(plan_tiles)),
            plan_tiles=plan_tiles,
            context=context,
        )
        priority_ranks = {int(tile): int(rank) for rank, tile in enumerate(priority_order)}
        session._presentation_priority_rank_cache = (cache_key, priority_ranks)
    ordered_upserts = {
        int(tile): tile_delta.upserts[int(tile)]
        for tile in sorted(
            tile_delta.upserts,
            key=lambda tile: priority_ranks.get(int(tile), len(priority_ranks)),
        )
        if int(tile) in tile_delta.upserts
    }
    return replace(
        tile_delta,
        upserts=ordered_upserts,
        priority_ranks={
            int(tile): priority_ranks[int(tile)]
            for tile in ordered_upserts
            if int(tile) in priority_ranks
        },
    )


def _commit_report_accepts_new_preview(session, report, tile_delta, tile_state) -> bool:
    if report is None or not report.acknowledges(tile_delta):
        return False
    committed = report.accepted_upserts(tile_delta)
    payloads = dict(getattr(tile_state, "payloads", {}) or {})
    previously_presented = dict(
        getattr(getattr(session, "lifecycle", None), "backend_presented_identities", {}) or {}
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


def plan_stage_fan_in_candidates(
    document, missing_tiles, *, cancellation_token=None
) -> dict[str, object]:
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
        retained = tuple(
            candidate for candidate in candidates if getattr(candidate, "retain", True)
        )
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
        if tuple(getattr(key, "shape", ()) or ()) != tuple(
            int(size) for size in document.current_shape
        ):
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
                    if tuple(getattr(key, "shape", ()) or ()) != tuple(
                        int(size) for size in document.current_shape
                    ):
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


def build_stage_fan_in_plan(
    renderer, document, missing_tiles, *, existing_only: bool = False, candidate_plan=None
) -> dict[str, object]:
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
            getter = (
                stage_cache.get_containing
                if hasattr(stage_cache, "get_containing")
                else stage_cache.get
            )
            value = getter(key)
            if value is not None:
                stage_values[key] = value
                for tile in tiles:
                    tile_stage_keys[int(tile.montage_index)] = key
                    tile_stage_plans[int(tile.montage_index)] = tile_stage_plans.get(
                        int(tile.montage_index), group["plan"]
                    )
                    tile_stage_candidates[int(tile.montage_index)] = candidate
                continue
            in_flight = getattr(
                renderer.win.operation_evaluator.stage_materializer, "_in_flight", {}
            )
            request = in_flight.get(key)
            if request is not None:
                attached_stage_keys.add(key)
                for tile in tiles:
                    tile_stage_keys[int(tile.montage_index)] = key
                    tile_stage_plans[int(tile.montage_index)] = tile_stage_plans.get(
                        int(tile.montage_index), group["plan"]
                    )
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
        result = renderer.win.operation_evaluator.stage_materializer.request_stage(
            document_key, candidate
        )
        retained_stage_decision = result.decision
        if result.decision == "hit":
            stage_values[key] = result.value
            for tile in tiles:
                tile_stage_keys[int(tile.montage_index)] = key
                tile_stage_plans[int(tile.montage_index)] = tile_stage_plans.get(
                    int(tile.montage_index), group["plan"]
                )
                tile_stage_candidates[int(tile.montage_index)] = candidate
            continue
        if result.decision == "scheduled":
            for tile in tiles:
                tile_stage_keys[int(tile.montage_index)] = key
                tile_stage_plans[int(tile.montage_index)] = tile_stage_plans.get(
                    int(tile.montage_index), group["plan"]
                )
                tile_stage_candidates[int(tile.montage_index)] = candidate
                waiting_indices.add(int(tile.montage_index))
            stage_requests.append((result.request, group["plan"]))
            continue
        if result.decision == "attached":
            attached_stage_keys.add(key)
            for tile in tiles:
                tile_stage_keys[int(tile.montage_index)] = key
                tile_stage_plans[int(tile.montage_index)] = tile_stage_plans.get(
                    int(tile.montage_index), group["plan"]
                )
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
        return plan_stage_fan_in_candidates(
            session.document, missing_tiles, cancellation_token=token
        )

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
        submit_stage_tasks(renderer, current, stage_plan["stage_requests"])
        renderer.retarget_frame_pipeline(current)

    def stale(session_id=session.session_id, session_key=session.key):
        current = getattr(renderer, "_frame_session", None)
        cleared = bool(
            current is not None and renderer._is_current_frame_session(session_id, session_key)
        )
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
        session.repeated_expensive_stage_per_tile or stage_plan["repeated_expensive_stage_per_tile"]
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
            target_unsettled=len(tuple(session.required_target_unsettled_tiles())),
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
                allowed_chunk_axes=stage_materialization_allowed_chunk_axes(
                    request.candidate.shape
                ),
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
            current.stage_fan_in.activate_value(key, value)
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
    ranks = {int(tile_number): int(rank) for rank, tile_number in enumerate(ordered)}
    return min(
        (ranks[tile_number] for tile_number in consumers if tile_number in ranks),
        default=int(UNRANKED_SCHEDULING_RANK),
    )


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
    released = session.stage_fan_in.release_missing(key)
    unbound = len(tuple(released.tiles or ()))
    if unbound:
        session.tile_compute_waiting_for_stage = max(
            0, int(session.tile_compute_waiting_for_stage) - unbound
        )
        session.stage_backed_tiles_pending = max(
            0, int(session.stage_backed_tiles_pending) - unbound
        )
    return int(unbound)


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


def shader_commit_level_source(renderer, session, *, rehydrate_max_count: int | None = None):
    """Select the level source a shader commit may offer the controller.

    Before the first-pass histogram publishes, bounded partial evidence is the
    honest provisional source.  Afterwards a candidate may move the applied
    window only when the tracked summary is mature under the retention
    contract — complete coverage at target quality (``ROUGH_TARGET``) or
    better; anything less falls back to the already accepted source so
    partial/preview ranges never disturb a published window.

    Gating on full refinement here froze the applied window at whatever
    anchored it first: on a cold load that anchor is preview/reduced-LOD
    evidence whose averaged extremes clip, and on wgpu the refined pass is
    owned by the semantic evidence producer which may never arm, so the
    mature target-quality evidence could never re-anchor the window while a
    scroll to the same slice settled on it (path-dependent levels,
    2026-07-24).
    """

    published = bool(getattr(session, "first_pass_histogram_published", False))
    current_level_source = renderer._montage_level_source_for_session(
        session,
        allow_partial=not published,
        rehydrate_max_count=rehydrate_max_count,
    )
    if published:
        current_summary = renderer._montage_level_tracker().summary_for(session.level_key)
        if int(getattr(current_summary, "evidence_quality", 0) or 0) < int(
            LevelEvidenceQuality.ROUGH_TARGET
        ):
            current_level_source = None
    if current_level_source is None:
        # Withhold an immature *candidate*, not the already accepted source.
        # The histogram callback can advance the generation just before the
        # controller retains a broader applied window. Re-target convergence
        # to that accepted source or refinement remains blocked behind six
        # stale uniforms with no external work left to wake them.
        applied_source = getattr(session, "applied_level_source", None)
        if getattr(applied_source, "semantic_key", None) == session.level_key:
            current_level_source = applied_source
    return current_level_source


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


def _atomic_successor_handoff_pending(session) -> bool:
    """Whether a retained tiled predecessor requires one complete successor.

    ``FrameSession.atomic_successor_pending`` is armed only by the transition
    owner after it proves complete predecessor coverage.  Do not re-decide
    that physical obligation from the committed semantic frame or successor
    payload wrappers: either can legitimately lag the persistent tile layer,
    and doing so permits partial commits that can never acknowledge the
    already-armed handoff.
    """

    if not bool(getattr(session, "atomic_successor_pending", False)):
        return False
    required_scope = getattr(session, "atomic_successor_required_scope", None)
    if callable(required_scope) and not tuple(required_scope()):
        # Camera/plan changes may supersede the scope after the transition
        # owner armed the handoff.  Revalidate at the consumption boundary so
        # an obsolete all-slot transaction cannot wait on tiles the current
        # frame no longer requires.
        session.atomic_successor_pending = False
        session._atomic_prepared_transaction = None
        return False
    return True


def _atomic_successor_commit_modes(capabilities, *, pending: bool) -> tuple[bool, bool]:
    """Classify one semantic handoff without weakening it by render phase.

    A compatible predecessor is the complete visible frame. Shader windowing
    can refine that frame in place, but it cannot make a mixed-source successor
    semantically valid. Both renderer classes therefore wait for the same
    complete successor scope once the transition owner arms it.
    """

    shader = bool(getattr(capabilities, "shader_windowing", False))
    return bool(pending and not shader), bool(pending and shader)


def _cpu_transaction_payload_marker(payload) -> tuple:
    lod = getattr(payload, "lod", None)
    return (
        getattr(payload, "source_id", None),
        int(getattr(payload, "source_index", -1)),
        str(getattr(payload, "quality", "exact") or "exact"),
        int(getattr(lod, "level", 0) or 0),
        id(getattr(payload, "image", None)),
    )


def _shader_successor_transaction_payload_marker(payload) -> tuple:
    """Stable successor marker that deliberately ignores LOD upgrades."""

    return (
        _base_source_id(getattr(payload, "source_id", None)),
        int(getattr(payload, "source_index", -1)),
    )


def _prepared_upload_counters(renderer) -> dict[str, int]:
    """Mailbox hit/stale/miss counts for the commit trace, or nothing."""

    view = getattr(getattr(renderer, "win", None), "img_view", None)
    mailbox = getattr(view, "_prepared_tiled_uploads", None)
    if mailbox is None:
        return {}
    counters = mailbox.counters()
    return {
        # Planning: what was decided about each payload, including the two
        # ways a preparation is deliberately not made.
        "prepared_upload_submitted": int(counters.submitted),
        "prepared_upload_deduped": int(counters.deduped),
        "prepared_upload_skipped_resident": int(counters.skipped_resident),
        "prepared_upload_skipped_no_work": int(counters.skipped_no_work),
        "prepared_upload_skipped_stale_round": int(counters.skipped_stale_round),
        # Execution: what a worker did, and what the scheduler dropped first.
        "prepared_upload_executed": int(counters.executed),
        "prepared_upload_superseded_before_execution": int(counters.superseded_before_execution),
        "prepared_upload_superseded_publish": int(counters.superseded_publish),
        "prepared_upload_published": int(counters.published),
        # Displacement and consumption.
        "prepared_upload_replaced": int(counters.replaced),
        "prepared_upload_evicted": int(counters.evicted),
        "prepared_upload_hits": int(counters.hits),
        "prepared_upload_stale": int(counters.stale),
        "prepared_upload_misses": int(counters.misses),
        "prepared_upload_inline_fallbacks": int(counters.inline_fallbacks),
        "prepared_upload_resident": int(counters.resident_entries),
        "prepared_upload_peak_in_flight": int(counters.peak_in_flight),
    }


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
    required_scope = getattr(session, "atomic_successor_required_scope", None)
    required = (
        {int(tile) for tile in required_scope()}
        if callable(required_scope)
        else {int(tile) for tile in getattr(session, "visible_tile_numbers", ()) or ()}
        - {int(tile) for tile in getattr(session, "skipped_tiles", ()) or ()}
    )
    if {int(tile) for tile in getattr(delta, "active_tiles", ()) or ()} != required:
        return False
    target_getter = getattr(session, "tile_target_identities", None)
    if callable(target_getter):
        frozen_targets = dict(getattr(delta, "target_identities", {}) or {})
        current_targets = dict(target_getter(required) or {})
        if frozen_targets != current_targets:
            return False
        delta_payloads = dict(getattr(delta, "upserts", {}) or {})
        if any(
            tile not in delta_payloads
            or not acknowledged_identity_satisfies_target(
                tile_ack_identity(delta_payloads[tile]),
                frozen_targets.get(tile),
            )
            for tile in required
        ):
            return False
    markers = dict(prepared.get("payload_markers", {}) or {})
    payloads = dict(getattr(session, "display_tile_payloads", {}) or {})
    marker_kind = prepared.get("marker_kind")
    if marker_kind == "shader-successor":
        marker_fn = _shader_successor_transaction_payload_marker
    elif marker_kind == "cpu-compatible":
        marker_fn = _cpu_transaction_payload_marker
    else:
        return False
    return bool(markers) and all(
        int(tile) in payloads and marker_fn(payloads[int(tile)]) == marker
        for tile, marker in markers.items()
    )


def _cold_gpu_successor_requires_hidden_warm(
    *,
    session,
    cpu_backend: bool,
    resident_predicate,
    upserts,
) -> bool:
    """Whether a cold GPU transaction has complete pixels to preserve.

    ``display_committed`` is intentionally not consulted: it becomes true
    after any accepted tile and therefore cannot prove that a complete
    predecessor covers the required scope.  Only lifecycle-confirmed first
    pixels may transfer the remaining work to hidden successor warming.
    """

    first_pixels = getattr(session, "required_first_pixels_presented", None)
    if cpu_backend or not callable(first_pixels) or not bool(first_pixels()):
        return False
    if not callable(resident_predicate):
        return False
    return any(not bool(resident_predicate(payload)) for payload in dict(upserts or {}).values())


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
    resident = getattr(
        getattr(renderer.win, "img_view", None),
        "tiledPayloadResident",
        None,
    )
    commit_slot_owned = getattr(
        getattr(renderer.win, "img_view", None),
        "tiledPayloadCommitSlotOwned",
        None,
    )
    physical_truth_available = callable(resident) or callable(commit_slot_owned)

    def physically_warm(payload) -> bool:
        return bool(
            (callable(resident) and bool(resident(payload)))
            or (callable(commit_slot_owned) and bool(commit_slot_owned(payload)))
        )

    pending = tuple(
        int(tile)
        for tile, payload in payloads.items()
        if (
            not physically_warm(payload)
            if physical_truth_available
            else (getattr(payload, "source_id", None), level_key) not in warmed_identities
        )
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
        # Time-budgeted warming: one receiver-owned callback prepares as many
        # bounded batches as fit in the GUI budget instead of exactly one.
        # With one fixed-size batch per callback, a 100-tile successor needed
        # 50 event-loop turns; combined with LowEventPriority starvation the
        # 2026-07-24 field freeze warmed ~6 tiles/second while the drain
        # storm kept the queue full.  Each warm() call stays batch_size-
        # bounded, so individual backend calls remain small; the budget only
        # governs how many of them share one callback turn.
        warm_budget_start = perf_counter()
        while True:
            admitted = tuple(job["pending"][: max(1, int(batch_size))])
            del job["pending"][: len(admitted)]
            batch = {int(tile): job["payloads"][int(tile)] for tile in admitted}
            # This coordinator already owns the bounded GUI-thread continuation.
            # Tell the backend to complete this batch synchronously instead of
            # queueing it behind the background warm scheduler, whose admission
            # correctly requires a settled visible target.  Queueing a target
            # successor there created a cycle: target settlement waited for warm
            # residency while warm residency waited for target settlement.
            warm_delta = (
                tile_delta
                if bool(getattr(tile_delta, "atomic_handoff", False))
                else replace(tile_delta, atomic_handoff=True)
            )
            warm(
                payloads=batch,
                geometry=geometry,
                levels=level_key,
                rgb_already_windowed=bool(rgb_already_windowed),
                tile_delta=warm_delta,
                tile_residency_budget_bytes=tile_residency_budget_bytes(renderer._memory_policy()),
                frame_plan=getattr(session, "frame_plan", None),
            )
            unresolved = tuple(
                int(tile) for tile in admitted if not physically_warm(job["payloads"][int(tile)])
            )
            for tile in admitted:
                if int(tile) in unresolved:
                    continue
                payload = job["payloads"][int(tile)]
                marker = (getattr(payload, "source_id", None), level_key)
                warmed[int(tile)] = marker
                warmed_identities.add(marker)
            if unresolved:
                job["pending"].extend(unresolved)
                if len(unresolved) == len(admitted):
                    # No page became resident: stop this bounded job instead of
                    # spinning a callback forever. Re-arm the visible
                    # presentation owner before returning: the settlement guard
                    # is observational and cannot own these dirty upserts. A
                    # later memory/interaction edge can now retry, and the global
                    # deadline still reports a persistent capacity failure.
                    session._atomic_warm_job = None
                    session.final_commit_pending = True
                    session.flush_pending = True
                    renderer.request_montage_replan(session)
                    return
            if not job["pending"]:
                break
            if (perf_counter() - warm_budget_start) * 1000.0 >= _ATOMIC_WARM_BUDGET_MS:
                _post_visible_path_callback(renderer, continue_warm)
                return
        session._atomic_warm_job = None
        session.final_commit_pending = True
        session.flush_pending = True
        renderer.request_montage_replan(session)

    _post_visible_path_callback(renderer, continue_warm)
    return False


def persistent_tile_layer_fast_drain_enabled(window, session) -> bool:
    if not bool(getattr(session, "display_committed", False)):
        return False
    return persistent_gpu_tile_residency_backend(window, session)


def direct_montage_tile_delta_commit_enabled(
    window, session, *, allow_uncommitted_persistent: bool = False
) -> bool:
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
    return bool(
        capabilities.persistent_tile_residency
        and capabilities.shader_windowing
        and kind in {"gpu_atlas", "none"}
    )


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
            # Floor-progress commits carry no dirty/pending work at decision
            # time — the build's floor pass materializes preview upserts
            # during assembly (frontier tiles at a zoom-in). Unsettled
            # required targets mean such upserts can appear, and an
            # ungoverned batch here failed the journey matrix's
            # bounded-commit oracle (pyqtgraph zoom_in, v19/v11/2026-07-19
            # v2-v4: max_upserts=0, unbounded_reason="").
            or getattr(session, "required_target_unsettled_tiles", tuple)()
        )
    ):
        return {}
    interactive = interactive_active(window)
    decision = _commit_batch_decision(
        window,
        interactive=interactive,
        pass_token=_render_pass_token(session),
        remaining_items=_render_pass_remaining_items(session),
        session=session,
    )
    batch_limit = int(getattr(decision, "batch_limit", 0) or 0)
    byte_cap = int(getattr(decision, "byte_cap", 0) or 0)
    if batch_limit <= 0:
        feedback = latency_feedback(window)
        batch_limit = (
            8
            if feedback is None
            else int(feedback.batch_limit("tile_layer_commit", interactive=interactive))
        )
    if byte_cap <= 0:
        byte_cap = 8 * 1024 * 1024 if interactive else 32 * 1024 * 1024
    limits = {
        "max_upserts": max(1, int(batch_limit)),
        "max_upsert_bytes": max(1024, int(byte_cap)),
        "pass_budget_ms": float(getattr(decision, "budget_ms", 0.0) or 0.0),
        "cold_deadline_ms": presentation_upload_control_budget_ms(
            window, "tile_layer_commit", decision, interactive=interactive
        ),
        "upsert_cost_fn": pyqtgraph_payload_upload_nbytes,
        "pace_resident_retargets": True,
        "governor_details": tuple(getattr(decision, "details", ()) or ()),
    }
    return limits


def presentation_upload_control_budget_ms(
    window, channel: str, decision, *, interactive: bool
) -> float:
    budget = float(getattr(decision, "budget_ms", 0.0) or 0.0)
    feedback = latency_feedback(window)
    target = 4.0 if interactive else 8.0
    if feedback is not None:
        tuning = getattr(feedback, "tuning", None)
        target = float(
            getattr(tuning, "target_interactive_ms" if interactive else "target_idle_ms", target)
        )
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
    warm_count = int(getattr(report, "existing_items_shown", 0) or 0) + int(
        getattr(report, "relocated_tiles", 0) or 0
    )
    acknowledged_upserts = len(tuple(getattr(report, "committed_upserts", ()) or ()))
    return max(1, cold_count + warm_count, acknowledged_upserts)


def accepted_tiled_payloads(payloads, delta, report) -> dict[int, object]:
    if report is None or delta is None:
        return {}
    accepted = report.accepted_upserts(delta)
    return {
        int(tile): payloads[int(tile)] for tile in accepted if int(tile) in dict(payloads or {})
    }


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
    backend. A CPU-windowed backend needs the complete round-owned population:
    its levels are baked into pixels and cannot be repaired by convergence
    after the first commit.
    """

    if not bool(first_display_commit):
        return False
    if str(getattr(session, "round_level_evidence_source", "") or "") == "preview-cohort-pending":
        # The fallback semantic sweep is deliberately parked while the preview
        # cohort owns the round-level decision. That is safe only while no tile
        # from the round can reach a backend. A complete cohort is admitted
        # atomically, so holding both backend families here costs no normal
        # shared-preview progress and keeps the parked window unobservable.
        return True
    shader_windowing = bool(image_view_backend_capabilities(window.win.img_view).shader_windowing)
    has_rough_source = bool(
        level_stats is not None
        and getattr(level_stats, "bounds", None) is not None
        and getattr(level_stats, "source_indices", None)
        and int(getattr(level_stats, "evidence_quality", 0) or 0)
        >= int(LevelEvidenceQuality.ROUGH_PREVIEW)
    )
    if shader_windowing:
        return not has_rough_source
    complete_round_source = bool(
        has_rough_source
        and getattr(level_stats, "rank", None)
        in {LevelSourceRank.MONTAGE_COMPLETE, LevelSourceRank.MONTAGE_SAMPLED_FULL}
        and int(getattr(level_stats, "evidence_quality", 0) or 0)
        >= int(LevelEvidenceQuality.ROUGH_TARGET)
    )
    return not complete_round_source


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
        return int(tile_number), key, plane, histogram, None, None, None, None, None, None
    if len(row) == 8:
        (
            tile_number,
            key,
            plane,
            histogram,
            shader_mapping,
            texture_kind,
            level_data,
            level_stats,
        ) = row
        return (
            int(tile_number),
            key,
            plane,
            histogram,
            shader_mapping,
            texture_kind,
            level_data,
            level_stats,
            None,
            None,
        )
    if len(row) == 9:
        return (int(row[0]), *row[1:], None)
    if len(row) == 10:
        return (int(row[0]), *row[1:])
    raise ValueError(f"unexpected shared preview payload shape: {len(row)}")


def rendered_tile_nbytes(rendered) -> int:
    total = 0
    for name in (
        "image",
        "histogram_data",
        "semantic_data",
        "semantic_histogram_data",
        "level_data",
    ):
        value = getattr(rendered, name, None)
        if value is not None:
            total += int(getattr(np.asarray(value), "nbytes", 0) or 0)
    return int(total)


def interactive_active(window) -> bool:
    coordinator = getattr(window.win, "render_coordinator", None)
    return bool(
        (coordinator is not None and getattr(coordinator, "interactive_active", False))
        or viewport_interaction_active(window)
    )


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


def texture_payload_upload_nbytes(payload) -> int:
    texture = getattr(payload, "texture_data", None)
    if texture is None:
        texture = getattr(payload, "image", None)
    total = 0 if texture is None else int(getattr(np.asarray(texture), "nbytes", 0) or 0)
    histogram = getattr(payload, "histogram_data", None)
    if histogram is not None and histogram is not texture:
        total += int(getattr(np.asarray(histogram), "nbytes", 0) or 0)
    return max(1, int(total))


def wgpu_native_plane_warm_payload(payload) -> bool:
    """Will this payload's commit upload a whole canonical plane instead?

    Mirrors ``_wgpu_reusable_native_texture``'s decision on the pipeline side,
    where no binding is available: a payload warms the canonical plane when it
    carries the WHOLE plane and presents less than all of it — either reduced
    (a coarser level) or cropped (a sub-rect of the plane), or both.
    """

    native = getattr(payload, "native_residency_data", None)
    if native is None:
        return False
    anchor = getattr(payload, "source_anchor", None)
    plane_shape = tuple(int(value) for value in (getattr(anchor, "plane_shape", ()) or ()))
    if len(plane_shape) != 2 or tuple(int(value) for value in np.shape(native)[:2]) != plane_shape:
        return False
    if int(getattr(getattr(payload, "lod", None), "level", 0) or 0) > 0:
        return True
    source_rect = tuple(int(value) for value in (getattr(anchor, "source_rect", ()) or ()))
    return len(source_rect) == 4 and source_rect != (0, plane_shape[0], 0, plane_shape[1])


def wgpu_page_rounded_nbytes(texture) -> int:
    """Bytes a WGPU texture upload PHYSICALLY writes: whole pages, always.

    ``_wgpu_page_block`` zero-fills a full ``(PAGE, PAGE)`` block and
    ``write_texture`` writes extent ``(PAGE, PAGE, 1)``, so a payload's logical
    ``nbytes`` is never what the upload costs.  A 336² float32 plane is four
    256² pages: 451 584 B logical against 1 048 576 B written, a **2.32×**
    undercharge.  A reduced 21² payload that does not warm its plane is worse
    still — 1 764 B against one whole page.

    Every byte-governed decision in the commit path (``max_upsert_bytes``, the
    admission cost function, residency accounting) takes this as its input, so
    a cap denominated in bytes could not mean what it said while the input was
    wrong by a factor of two.  Measured 2026-07-26; harmless until then only
    because ``_idle_backlog_cohort``'s item ceiling of 32 happened to bind
    first, and 32 MiB ÷ 1 MiB is also 32.
    """

    from arrayscope.gpu.wgpu_executor import PAGE

    data = np.asarray(texture)
    if data.ndim < 2:
        return int(data.nbytes)
    rows, columns = (int(value) for value in data.shape[:2])
    pages = -(-rows // PAGE) * -(-columns // PAGE)
    texel_nbytes = int(data.itemsize) * int(np.prod(data.shape[2:], dtype=int))
    return pages * PAGE * PAGE * texel_nbytes


def wgpu_payload_upload_nbytes(payload) -> int:
    """Physical bytes for WGPU, including hidden exact source-page warming."""

    if wgpu_native_plane_warm_payload(payload):
        # The WGPU warm path uses the exact native pages instead of also
        # uploading the redundant reduced/cropped payload.
        native = getattr(payload, "native_residency_data", None)
        return max(1, wgpu_page_rounded_nbytes(native))
    texture = getattr(payload, "texture_data", None)
    if texture is None:
        texture = getattr(payload, "image", None)
    if texture is None:
        return texture_payload_upload_nbytes(payload)
    # The histogram array is separate from the page grid, so it is charged as
    # itself rather than rounded to a texture page.
    total = wgpu_page_rounded_nbytes(texture)
    histogram = getattr(payload, "histogram_data", None)
    if histogram is not None and histogram is not texture:
        total += int(getattr(np.asarray(histogram), "nbytes", 0) or 0)
    return max(1, int(total))


def _persistent_tile_upsert_limits(window, session) -> dict[str, object]:
    if not persistent_gpu_tile_residency_backend(window, session):
        return {}
    capabilities = image_view_backend_capabilities(getattr(window.win, "img_view", None))
    interactive = interactive_active(window)
    decision = _commit_batch_decision(
        window,
        interactive=interactive,
        pass_token=_render_pass_token(session),
        remaining_items=_render_pass_remaining_items(session),
        session=session,
    )
    batch_limit = int(getattr(decision, "batch_limit", 0) or 0)
    byte_cap = int(getattr(decision, "byte_cap", 0) or 0)
    if batch_limit <= 0:
        feedback = latency_feedback(window)
        batch_limit = (
            4
            if feedback is None
            else int(feedback.batch_limit("montage_present_total", interactive=interactive))
        )
    if byte_cap <= 0:
        byte_cap = 8 * 1024 * 1024 if interactive else 32 * 1024 * 1024
    limits: dict[str, object] = {
        "max_upserts": max(1, int(batch_limit)),
        "max_upsert_bytes": max(1024, int(byte_cap)),
        "pass_budget_ms": float(getattr(decision, "budget_ms", 0.0) or 0.0),
        "upsert_cost_fn": (
            wgpu_payload_upload_nbytes
            if capabilities.name == "wgpu"
            else texture_payload_upload_nbytes
        ),
        "cold_deadline_ms": presentation_upload_control_budget_ms(
            window, "montage_present_total", decision, interactive=interactive
        ),
        # Physical page residency is authoritative for WGPU mapping-only
        # retargets. Those bindings form one visibility transaction: pacing
        # them through the cold item cohort tears the retained frame and can
        # leave the ready ledger with no completion to arm its successor.
        # Cold uploads remain governed by both caps below.
        "pace_resident_retargets": False,
        "governor_details": tuple(getattr(decision, "details", ()) or ()),
    }
    resident = getattr(getattr(window.win, "img_view", None), "tiledPayloadResident", None)
    if callable(resident):
        limits["physical_resident_fn"] = resident
    return limits


def _render_pass_token(session) -> tuple[int, str, bool]:
    verdict = getattr(getattr(session, "scheduling_policy", None), "verdict", None)
    return (
        int(getattr(session, "session_id", 0) or 0),
        str(getattr(session, "render_round_id", "") or ""),
        bool(getattr(verdict, "coverage_open", False)),
    )


def _render_pass_remaining_items(session) -> int:
    pending = {
        *tuple(int(tile) for tile in tuple(getattr(session, "dirty_payloads", ()) or ())),
        *tuple(int(tile) for tile in tuple(getattr(session, "pending_payload_upserts", ()) or ())),
    }
    unsettled = getattr(session, "required_target_unsettled_tiles", None)
    if callable(unsettled):
        pending.update(int(tile) for tile in tuple(unsettled() or ()))
    return max(1, len(pending))


def _commit_batch_decision(
    window,
    *,
    interactive: bool,
    pass_token=None,
    remaining_items: int | None = None,
    session=None,
):
    governor = getattr(window.win, "resource_governor", None)
    pass_kind = (
        "preview"
        if isinstance(pass_token, tuple) and pass_token and bool(pass_token[-1])
        else "target"
    )
    begin_pass = getattr(governor, "begin_render_pass", None)
    if callable(begin_pass):
        stable_pass_token = (
            pass_token[:-1]
            if isinstance(pass_token, tuple) and len(pass_token) >= 2
            else pass_token
        )
        structural_key, representation_key = _render_pass_cost_context(window, session)
        begin_pass(
            stable_pass_token,
            pass_kind=pass_kind,
            structural_key=structural_key,
            representation_key=representation_key,
        )
    decide_pass = getattr(governor, "decide_render_pass", None)
    if callable(decide_pass):
        return decide_pass(
            interactive=interactive,
            pass_kind=pass_kind,
            remaining_items=remaining_items,
        )
    decide = getattr(governor, "decide_commit_batch", None)
    if callable(decide):
        return decide(interactive=interactive)
    provider = getattr(window.win, "_gui_callback_budget_decision", None)
    if callable(provider):
        return provider("presentation_commit", interactive=interactive)
    return None


def _render_pass_cost_context(window, session) -> tuple[object, object]:
    """Identify only the facts that change transaction cost-model terms."""

    capabilities = image_view_backend_capabilities(getattr(window.win, "img_view", None))
    plan = getattr(session, "plan", None)
    tiles = tuple(getattr(plan, "tiles", ()) or ())
    structural_key = (
        str(getattr(capabilities, "name", "") or "unknown"),
        (
            "persistent-delta"
            if bool(getattr(capabilities, "persistent_tile_residency", False))
            else "cpu-delta"
        ),
        tuple(int(value) for value in tuple(getattr(plan, "tile_shape", ()) or ())),
        int(getattr(plan, "columns", 0) or 0),
        int(getattr(plan, "rows", 0) or 0),
        int(getattr(plan, "gap", 0) or 0),
        len(tiles),
    )
    view_state = getattr(session, "view_state", None)
    channel = getattr(getattr(view_state, "channel", None), "value", None)
    representation_key = (
        str(getattr(session, "output_dtype", "") or ""),
        str(channel or ""),
    )
    return structural_key, representation_key


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
        f"atomic_reject={getattr(renderer, '_last_montage_atomic_fast_reject_reason', '') or '-'!s}",
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


def _observe_ui(
    renderer, channel, ms, count, byte_count, work_class, backend, *, details=None
) -> None:
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
    """Whether this is a cohort of per-tile rows rather than one tile's payload.

    Discriminate on WHERE the page-set key sits, not on tuple length: a cohort
    row is ``(tile_number, key, pages, ...)`` and a single payload is
    ``(key, pages, ...)``. The length set below is only a secondary guard.

    That distinction is load bearing because ``LodPageSetKey`` is itself
    tuple-shaped, so a single payload whose leading key happened to match a row
    arity was read as a cohort. Adding one field to the native-output preview
    payload was enough to trigger it, and any future field on either shape
    moves the lengths again. The key's position does not move.
    """

    if (
        not isinstance(payload, tuple)
        or not payload
        or isinstance(payload[0], render_lod.LodPageSetKey)
    ):
        return False
    return bool(
        all(
            isinstance(row, tuple)
            and len(row) in {4, 8, 9, 10}
            and isinstance(row[0], (int, np.integer))
            and isinstance(row[1], render_lod.LodPageSetKey)
            for row in payload
        )
    )


def tiled_payloads_include_semantics(payloads) -> bool:
    return any(
        display_tile_payload_has_semantics(payload) for payload in dict(payloads or {}).values()
    )


def tiled_payloads_can_commit_frame(payloads) -> bool:
    return any(
        display_tile_payload_can_commit_frame(payload) for payload in dict(payloads or {}).values()
    )


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
    "tiled_payloads_can_commit_frame",
    "tiled_payloads_include_semantics",
    "viewport_interaction_active",
]
