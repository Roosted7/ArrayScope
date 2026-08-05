"""Frame render orchestration for ArrayScope windows.

The frame path owns the single visible image surface for every image view.  A
single slice and a multi-region montage both become one semantic frame with
region payloads; backends differ only in physical presentation mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter, thread_time

import numpy as np
import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception
from arrayscope.core.cache_status import CacheStatus, CacheStatusSnapshot
from arrayscope.core.frame_targets import FrameTarget
from arrayscope.core.memory_budget import estimate_display_image_bytes, format_bytes
from arrayscope.core.trace import emit_trace
from arrayscope.core.view_state import ChannelMode
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.backends.base import surface_for_view, tiled_presentation_visible
from arrayscope.display.frame_planner import FramePlanner
from arrayscope.display.lod import LOD_POLICY_RESIDENT, select_lod_demand
from arrayscope.display.model.montage_levels import (
    MontageLevelStats,
)
from arrayscope.display.montage import (
    ACTIVE_MARGIN_TILES,
    COVERAGE_MARGIN_TILES,
    RETENTION_HALO_TILES,
    VISIBLE_MARGIN_TILES,
    MontageTileState,
    RenderedTile,
    make_montage_plan,
    viewport_tiles,
)
from arrayscope.display.planning import fallback_level_source, normalize_bounds
from arrayscope.display.pyramid import preview_level_for_tile_shape
from arrayscope.kernel import Lane as WorkLane
from arrayscope.kernel import WorkItem
from arrayscope.kernel import complete_inline_work as _complete_inline_work
from arrayscope.operations.evaluator import _document_key
from arrayscope.render import lod as render_lod
from arrayscope.render.level_stats import LevelStatsService
from arrayscope.render.stages import RenderIntent
from arrayscope.ui.toasts import show_status_message
from arrayscope.window import frame_effects as montage_commit
from arrayscope.window.display_presenter import tile_residency_budget_bytes
from arrayscope.window.frame_effects import FramePipelineEffects, session_order_lock_section
from arrayscope.window.frame_runtime import (
    FrameRuntimeMixin,
)
from arrayscope.window.frame_session import FrameSession, plan_presentation_transition
from arrayscope.window.montage_payload_cache import (
    RetainedTiledPayloadStore,
)
from arrayscope.window.montage_payload_cache import (
    payload_compatible_with_tile as _payload_compatible_with_tile,
)
from arrayscope.window.montage_payload_cache import (
    payload_lod_matches as _payload_lod_matches,
)
from arrayscope.window.montage_payload_cache import (
    previous_tiled_payloads_by_base_source as _previous_tiled_payloads_by_base_source,
)
from arrayscope.window.montage_prefetch import (
    drain_deferred_out_of_view_invalidation,
    drain_out_of_view_upgrade,
    schedule_near_viewport_montage_prefetch,
)
from arrayscope.window.montage_viewport import (
    MontageViewportPlan,
    effective_montage_columns,
    frame_session_key,
    montage_latched_columns_for_plan,
    montage_tile_semantic_key,
    montage_viewport_intent,
    montage_viewport_retarget_policy,
    plan_full_view_range,
    remap_montage_roi_selections,
    retarget_montage_viewport_plan,
    square_montage_fit_view_range,
)
from arrayscope.window.montage_viewport import (
    montage_priority_focus as _montage_priority_focus,
)
from arrayscope.window.render_contract import (
    session_token_is_current as _session_token_is_current,
)

MONTAGE_VERY_SLOW_UPLOAD_MS = 100.0


@dataclass(frozen=True)
class MontageCacheResolutionContext:
    """Per-retarget snapshot of the tile-from-cache lookup inputs.

    Valid for exactly one retarget pass: it captures the retained-store scan
    and the committed frame's payload map at build time, so it must not
    outlive the pass that built it (a later commit or eviction would leave it
    describing stale residency).
    """

    reuse_any_lod: bool
    selected_lod_factor: int
    retained_store: RetainedTiledPayloadStore
    previous_payloads: dict[object, object]
    tile_key_for: object


class FrameControllerMixin(FrameRuntimeMixin, LevelStatsService):
    def _interactive_frame_cache_hit(self) -> bool:
        view_state = getattr(self.win, "view_state", None)
        if view_state is None or view_state.image_axes is None:
            return False
        evaluator = getattr(self.win, "operation_evaluator", None)
        if evaluator is None:
            return False
        shader_display = bool(image_view_backend_capabilities(self.win.img_view).shader_windowing)
        colormap_lut = self._evaluation_colormap_lut(view_state, shader_display=shader_display)
        viewport_plan = self._montage_viewport_plan(view_state)
        expected_key = frame_session_key(
            _document_key(self.win.document),
            view_state,
            viewport_plan,
            colormap_lut,
        )
        frame = getattr(self.win, "_committed_display_frame", None)
        frame_key = None if frame is None else getattr(frame, "key", None)
        if getattr(frame_key, "request_key", None) != expected_key:
            return False
        for tile in viewport_plan.candidate_tiles(margin_tiles=VISIBLE_MARGIN_TILES):
            if (
                evaluator.cached_montage_tile(
                    tile.view_state,
                    montage_axis=viewport_plan.axis,
                    source_index=tile.source_index,
                    colormap_lut=colormap_lut,
                    document=self.win.document,
                    shader_display=shader_display,
                )
                is None
            ):
                return False
        return True

    def _interactive_render_supersedes_presentation(self, *, reason: str) -> bool:
        view_state = getattr(self.win, "view_state", None)
        if view_state is None or view_state.image_axes is None:
            return False
        frame = getattr(self.win, "_committed_display_frame", None)
        frame_key = None if frame is None else getattr(frame, "key", None)
        capabilities = image_view_backend_capabilities(self.win.img_view)
        shader_display = bool(capabilities.shader_windowing)
        colormap_lut = self._evaluation_colormap_lut(view_state, shader_display=shader_display)
        viewport_plan = self._montage_viewport_plan(view_state)
        semantic_key = RenderIntent.semantic_key_for_montage(
            _document_key(self.win.document),
            view_state,
            viewport_plan,
            colormap_lut,
        )
        return getattr(frame_key, "request_key", None) != semantic_key

    def _montage_frame_planner(self) -> FramePlanner:
        provider = getattr(self, "_frame_planner", None)
        if provider is not None:
            return provider()
        planner = getattr(self, "_montage_frame_planner_instance", None)
        if planner is None:
            planner = FramePlanner()
            self._montage_frame_planner_instance = planner
        return planner

    def _montage_tile_layer_policy(self, geometry, data) -> bool:
        # Every montage display commits through the tile layer; the historical
        # per-size/per-backend chooser collapsed to this predicate.
        return getattr(geometry, "montage", None) is not None

    def _retained_tiled_payload_store(self) -> RetainedTiledPayloadStore:
        store = getattr(self, "_montage_retained_tiled_payloads", None)
        if not isinstance(store, RetainedTiledPayloadStore):
            store = RetainedTiledPayloadStore()
            self._montage_retained_tiled_payloads = store
        return store

    def _retains_out_of_view_payloads(self) -> bool:
        """Whether this backend keeps display payloads for out-of-view tiles.

        Only ``gpu_atlas``: its eviction is reachable solely from
        ``_ensure_free_pool_layer`` under pool-layer pressure, so a tile leaving
        the viewport keeps its pages either way and dropping the payload buys
        nothing but a whole-montage republication on the way back.  A
        ``cpu_item`` backend owns real per-tile scene items with their own
        residency policy, and ``none`` has no persistent residency to build on.
        """

        capabilities = image_view_backend_capabilities(getattr(self.win, "img_view", None))
        return bool(
            capabilities.persistent_tile_residency
            and str(getattr(capabilities, "tile_residency_kind", "none") or "none") == "gpu_atlas"
        )

    def _hide_superseded_presentation(self, session, reason: str, tiles) -> int:
        """Stop drawing tiles a data change superseded.

        The one seam behind the product rule "a quick pan must never reveal
        stale tiles".  It is deliberately the cheap half of invalidation: it
        hides, and runs on the path that owes the screen new pixels, while the
        removals — which re-enter the ``removals`` tuple of every commit until
        the backend acknowledges them — wait in
        ``deferred_out_of_view_invalidations`` for the visible set to finish.

        The caller names the superseded tiles.  Passing "everything outside the
        viewport" instead looked equivalent and is not: a camera move can reach
        the index-window retarget with every source unchanged, and hiding on
        that would undo the whole point of out-of-view retention — caught by
        ``test_wgpu_pan_away_and_back_reuses_the_retained_presentation_scope``,
        which went from 0 to 1 republication per pan leg.

        Nothing here is a new staleness notion.  A tile hidden by this seam
        returns only through the ordinary identity-checked upsert path, and the
        payload-level guards (``acknowledged_identity_satisfies_target`` on
        ``cpu_item``, ``payload_compatible_with_tile``'s
        ``semantic_generation`` on both) are what refuse the superseded pixels
        if anything asks for them in the meantime.
        """

        superseded = frozenset(int(tile) for tile in tiles or ())
        if not superseded or session is None:
            return 0
        if not bool(getattr(session, "defer_out_of_view_removals", True)):
            return 0
        surface = surface_for_view(self.win.img_view)
        hide = getattr(surface, "hide_superseded_tiled_presentation", None)
        if not callable(hide):
            return 0
        hidden = int(hide(str(reason), tiles=superseded) or 0)
        if hidden:
            emit_trace(
                "superseded_presentation_hidden",
                session_id=int(session.session_id),
                reason=str(reason),
                hidden=int(hidden),
                superseded=len(superseded),
            )
        self._superseded_tiles_hidden = (
            int(getattr(self, "_superseded_tiles_hidden", 0) or 0) + hidden
        )
        return hidden

    def _session_source_anchoring(self, document, view_state, axis):
        """ADR 0055 G3: window-invariant payload anchoring for the session.

        GPU-resident sessions stamp exact payloads with a window-free content
        identity so the backend can keep source pages across display-window
        shifts.  Montage planes add their source index when the payload anchor
        is built; the shared anchoring state deliberately normalizes the
        montage slider index here.
        """

        capabilities = image_view_backend_capabilities(self.win.img_view)
        if getattr(capabilities, "tile_residency_kind", None) != "gpu_atlas":
            return None
        from arrayscope.display.source_anchoring import source_anchoring_for_view

        anchoring_state = view_state
        if axis is not None:
            anchoring_state = view_state.with_montage_axis(
                None,
                columns=None,
                indices=None,
                text=None,
            ).with_slice(int(axis), 0)
        return source_anchoring_for_view(
            document,
            anchoring_state,
            canonical_orientation=bool(getattr(capabilities, "display_axis_transpose", False)),
        )

    def _montage_viewport_plan(self, view_state, *, view_range=None) -> MontageViewportPlan:
        axis = view_state.montage_axis
        all_indices = (
            (0,)
            if axis is None
            else tuple(view_state.montage_indices or tuple(range(int(view_state.shape[axis]))))
        )
        viewport_size = self.win.img_view.graphicsView.viewport().size()
        viewport_shape = (max(1, viewport_size.height()), max(1, viewport_size.width()))
        tile_shape = self._montage_tile_shape(view_state)
        pending_restore_range = None
        pending_restore_columns = None
        if view_range is None:
            pending_restore = getattr(self.win, "_pending_viewport_continuity_range", None)
            pending_restore_range = pending_restore() if callable(pending_restore) else None
            if pending_restore_range is None:
                # A continuity restore that has been applied but whose window
                # shape has not settled yet still owns the target range; the
                # live camera is transitional and would misanchor everything
                # derived from this plan (priority focus, tile bands).
                active_restore = getattr(self.win, "_active_viewport_continuity_range", None)
                pending_restore_range = active_restore() if callable(active_restore) else None
            pending_columns = getattr(self.win, "_pending_viewport_continuity_columns", None)
            pending_restore_columns = pending_columns() if callable(pending_columns) else None
        # The live Qt camera range is only trustworthy once this montage owns
        # the viewport. Entering montage mode from a single plane (or a montage
        # on a different axis), and the whole first fill before the initial
        # commit rescales the camera, leave the camera on the *previous* view —
        # a tiny single-plane range that reads as native scale (desired_level
        # 0). Seeding the plan from that stale range makes the first montage
        # render evaluate native tiles instead of the cheap reduced fit LOD.
        # When the camera is not yet on this montage, fall through to the fit
        # extent (`_initial_montage_planning_view_range`) below.
        current_session = getattr(self, "_frame_session", None)
        camera_on_montage = bool(
            current_session is not None
            and getattr(current_session, "montage_axis", None) == axis
            and getattr(current_session, "display_committed", False)
        )
        current_range = (
            view_range
            if view_range is not None
            else (
                pending_restore_range
                if pending_restore_range is not None
                else (
                    self._current_montage_global_view_range()
                    if getattr(self.win.img_view, "image", None) is not None and camera_on_montage
                    else None
                )
            )
        )
        # A committed montage holds its column layout forward under a manual
        # view, except at the zoom extremes (montage_latched_columns_for_plan).
        committed_plan = getattr(current_session, "plan", None) if camera_on_montage else None
        latched_columns = montage_latched_columns_for_plan(committed_plan, current_range)
        columns = self._effective_montage_columns(
            view_state,
            all_indices=all_indices,
            tile_shape=tile_shape,
            viewport_shape=viewport_shape,
            view_range=current_range,
            restored_columns=pending_restore_columns,
            latched_columns=latched_columns,
        )
        plan = make_montage_plan(
            view_state,
            axis=axis,
            indices=all_indices,
            tile_shape=tile_shape,
            columns=columns,
            viewport_shape=viewport_shape,
        )
        viewport_controller = getattr(self.win.img_view, "viewport_controller", None)
        planning_intent = montage_viewport_intent(viewport_controller, current_range)
        # AUTO/FIT describes the intended SUCCESSOR camera: when this plan's
        # layout differs from the committed session's, the fit extent — not
        # the predecessor camera — is the planning truth, so a CPU backend
        # cannot commit only the predecessor-sized subset and expose holes
        # when acknowledgement applies the new fit. But when the camera is
        # already on THIS layout, the live range IS the truth even under
        # AUTO intent: a zoom gesture does not change intent, and replacing
        # it with the fit froze the LOD demand at the fit level forever
        # (field defect 2026-07-18, both backends; red journey gate
        # tests/ui/test_lod_demand_freshness.py).
        camera_on_this_layout = bool(
            camera_on_montage
            and getattr(getattr(current_session, "plan", None), "geometry", None) == plan.geometry
        )
        if (
            view_range is None
            and pending_restore_range is None
            and (
                planning_intent.fit_locked
                or (planning_intent.auto_active and not camera_on_this_layout)
            )
        ) or current_range is None:
            current_range = _initial_montage_planning_view_range(
                plan,
                viewport_shape,
                viewport_controller,
            )
        priority_focus = _montage_priority_focus(self, current_range)
        capabilities = image_view_backend_capabilities(self.win.img_view)
        return MontageViewportPlan(
            axis=None if axis is None else int(axis),
            all_indices=all_indices,
            viewport_shape=viewport_shape,
            tile_shape=tile_shape,
            plan=plan,
            view_range=current_range,
            shader_display=bool(capabilities.shader_windowing),
            persistent_tile_residency=bool(capabilities.persistent_tile_residency),
            priority_focus=priority_focus,
        )

    def _effective_montage_columns(
        self,
        view_state,
        *,
        all_indices,
        tile_shape,
        viewport_shape,
        view_range,
        restored_columns=None,
        latched_columns=None,
    ) -> int | None:
        if not all_indices:
            return None
        viewport_controller = getattr(
            getattr(self.win, "img_view", None), "viewport_controller", None
        )
        requested_columns = getattr(view_state, "montage_columns", None)
        if requested_columns is None and restored_columns is not None:
            try:
                requested_columns = max(1, int(restored_columns))
            except (TypeError, ValueError):
                requested_columns = None
        intent = montage_viewport_intent(viewport_controller, view_range)
        return effective_montage_columns(
            len(all_indices),
            tile_shape,
            viewport_shape,
            requested_columns=requested_columns,
            fit_locked=intent.fit_locked,
            auto_active=intent.auto_active,
            latched_columns=latched_columns,
        )

    def _on_image_viewport_resized(
        self, *, previous_viewport_size=None, base_view_range=None, resize_focus=None
    ) -> None:
        if getattr(self.win, "_closing", False):
            return
        continuity_shape = getattr(self.win, "_viewport_continuity_shape_target", lambda: None)()
        if continuity_shape is not None:
            # Child layouts can resize after an outer-window continuity resize
            # has looked settled.
            # A genuine top-level user resize releases the transaction in the
            # window's resizeEvent before this callback, so an extant target is
            # still authoritative and must be restored instead of reflowed.
            restore_viewport_shape = getattr(
                self.win, "_restore_viewport_continuity_shape_after_layout", None
            )
            if callable(restore_viewport_shape):
                restore_viewport_shape()
            return
        active_continuity_range = getattr(
            self.win, "_active_viewport_continuity_range", lambda: None
        )()
        if active_continuity_range is not None:
            restore_viewport_shape = getattr(
                self.win, "_restore_viewport_continuity_shape_after_layout", None
            )
            if callable(restore_viewport_shape):
                restore_viewport_shape()
            self._set_montage_view_range(active_continuity_range)
            self.retarget_montage_viewport()
            return
        if getattr(self.win.view_state, "montage_axis", None) is None:
            # A single (non-montage) image has no tiles to re-flow: the image
            # view's own resize (_apply_viewport_resize) already preserved the
            # camera. Running the montage retarget below would re-apply a
            # second, conflicting camera and visibly rescale the content -- the
            # reason a manual single-image view shrinks on resize while montages
            # (which take ONLY this path via _viewport_resize_owned_by_parent)
            # do not. Continuity restores above still run for single images.
            return
        self._montage_live_layout_reflow = True
        self.win._montage_viewport_update_pending = False
        viewport_plan = self._retarget_montage_resize_camera(
            previous_viewport_size=_montage_viewport_shape_from_qt_size_tuple(
                previous_viewport_size
            ),
            base_view_range=base_view_range,
            resize_focus=resize_focus,
        )
        if viewport_plan is not None:
            self._retarget_montage_resize_payloads(viewport_plan)
        self.retarget_montage_viewport()

    def _current_montage_resize_focus(self, view_range) -> tuple[float, float] | None:
        return _montage_priority_focus(self, view_range)

    def _retarget_montage_resize_camera(
        self,
        *,
        previous_viewport_size=None,
        base_view_range=None,
        resize_focus=None,
    ) -> MontageViewportPlan | None:
        session = getattr(self, "_frame_session", None)
        if not self._frame_session_is_current(session):
            return None
        view_state = self.win.view_state
        if view_state.image_axes is None:
            # The live view state no longer presents an image (e.g. a reduction
            # switched to line-plot mode) while the montage session from the
            # previous state is still attached. There is nothing to retarget;
            # the pending render replaces the session.
            return None
        capabilities = image_view_backend_capabilities(self.win.img_view)
        viewport_plan = self._montage_viewport_plan(view_state, view_range=base_view_range)
        colormap_lut = self._evaluation_colormap_lut(
            view_state,
            shader_display=bool(capabilities.shader_windowing),
        )
        expected_key = frame_session_key(
            _document_key(self.win.document),
            view_state,
            viewport_plan,
            colormap_lut,
        )
        # Not the shared currency predicate: this compares the session against
        # the key re-derived from the *live* view state (semantic-match check).
        if session.key != expected_key:
            return None
        previous_plan = getattr(session, "plan", None)
        viewport_plan = self._retargeted_montage_viewport_plan(
            session,
            viewport_plan,
            previous_viewport_shape=previous_viewport_size,
            focus=resize_focus,
        )
        self._remap_montage_rois_for_layout_reflow(previous_plan, viewport_plan.plan)
        return viewport_plan

    @session_order_lock_section
    def _retarget_montage_resize_payloads(self, viewport_plan: MontageViewportPlan) -> bool:
        session = getattr(self, "_frame_session", None)
        if not self._frame_session_is_current(session):
            return False
        capabilities = image_view_backend_capabilities(self.win.img_view)
        try:
            display_mode = str(self.win.img_view.montageDisplayMode())
        except Exception:
            display_mode = ""
        if not montage_viewport_retarget_policy(capabilities, display_mode).enabled:
            return False
        view_state = self.win.view_state
        # A resize is a live drag, so it deliberately does NOT widen the
        # production request: ``coverage`` stays at the drawn set and no tile
        # is asked for that the settled camera will not need. The retention
        # halo is a different kind of thing and gets the ordinary ring.
        #
        # This used to pass 0 for both, unexplained. It came in with "Reduce
        # VisPy zoom path further" (19d36f45) alongside
        # ``_cancel_vispy_speculative_work`` and ``priority_retarget_limit=1``;
        # both of those were deleted with VisPy (54d1fd5b, 934081fb) and this
        # line was left behind. Its cost was not the coverage half but the
        # halo half: ``retarget_viewport`` feeds ``near`` straight into
        # ``retarget_tile_priority``, so every window resize replaced the
        # session's priority context with one whose NEAR band was empty --
        # flattening the ordering that prefetch and per-commit admission both
        # read -- until some later camera move happened to rebuild it.
        _additions, presentation_changed = session.retarget_viewport(
            view_range=viewport_plan.view_range,
            viewport_shape=viewport_plan.viewport_shape,
            plan=viewport_plan.plan,
            coverage_margin_tiles=VISIBLE_MARGIN_TILES,
            near_margin_tiles=RETENTION_HALO_TILES,
            priority_focus=viewport_plan.priority_focus,
        )
        memory_policy = self._memory_policy() if hasattr(self, "_memory_policy") else None
        session.frame_plan = self._montage_frame_planner().plan(
            target=FrameTarget(
                semantic_key=session.key,
                viewport_key=viewport_plan.view_range,
                presentation_key=(
                    str(session.window_mode),
                    normalize_bounds(getattr(session, "user_levels_override", None)),
                    bool(getattr(session, "force_auto", False)),
                ),
                quality="exact-visible",
            ),
            view_state=view_state,
            display_shape=viewport_plan.plan.display_shape,
            backend_capabilities=capabilities,
            viewport_shape=viewport_plan.viewport_shape,
            view_range=viewport_plan.view_range,
            memory_policy=memory_policy,
            montage_plan=viewport_plan.plan,
        )
        session.sync_lifecycle_scope()
        if memory_policy is not None:
            session.tile_residency_budget_bytes = tile_residency_budget_bytes(memory_policy)
        lod_swap_ready = session.mark_ladder_swaps_for_viewport(refresh_demand=False)
        self.retarget_frame_pipeline(
            session,
            prepared_lod_swap_ready=lod_swap_ready,
        )
        if presentation_changed or lod_swap_ready:
            self._commit_montage_resize_presentation_retarget(session)
        return True

    def _commit_montage_resize_presentation_retarget(self, session) -> None:
        self.apply_montage_presentation(session)

    def _publish_first_frame_content_extent(self, plan) -> bool:
        """Publish planned camera coordinates only when no frame can be retained."""

        if getattr(self.win, "_committed_display_frame", None) is not None:
            return False
        self._publish_montage_content_extent(plan)
        return True

    # The session-rebirth funnel: it closes the dying pipeline, releases its
    # LOD claims, inherits the backend identity mirror and swaps
    # ``_frame_session``. One span for the whole entry, because a rebirth
    # observed half-done by the owner thread is exactly the state no currency
    # check can describe.
    @session_order_lock_section
    def update_image_view(self, *, force_autolevel: bool = False, defer_side_panels: bool = False):
        for attribute in (
            "_last_montage_viewport_plan_ms",
            "_last_montage_cache_resolve_ms",
            "_last_montage_stage_plan_ms",
            "_last_frame_session_setup_ms",
            "_last_montage_initial_commit_ms",
        ):
            setattr(self, attribute, None)
        axis = self.win.view_state.montage_axis
        if self.win.view_state.image_axes is None or axis in self.win.view_state.image_axes:
            return
        plan_start = perf_counter()
        policy = self._refresh_memory_policy(active_render=self._montage_render_active())
        if policy is None:
            policy = self._memory_policy()
        user_levels = self._pending_display_levels_for_render()
        if force_autolevel and user_levels is not None:
            self._queue_display_levels(None)
            user_levels = None
        force_auto = bool(
            force_autolevel
            or (getattr(self.win, "_force_autolevel", False) and user_levels is None)
        )
        if getattr(self.win, "_force_autolevel", False):
            self.win._force_autolevel = False
        window_mode = self._current_window_mode()
        previous_frame = self._previous_display_frame_for_policy(force_auto=force_auto)
        pending_auto_level_source = (
            getattr(self, "_pending_auto_level_source", None) if force_auto else None
        )
        carried_user_level_source = None
        if not force_auto:
            candidate_user_source = getattr(self, "_explicit_user_level_source", None)
            if normalize_bounds(getattr(candidate_user_source, "levels", None)) is not None:
                carried_user_level_source = candidate_user_source

        view_state = self.win.view_state
        shader_display = bool(image_view_backend_capabilities(self.win.img_view).shader_windowing)
        colormap_lut = self._evaluation_colormap_lut(view_state, shader_display=shader_display)
        document = self.win.document
        viewport_plan = self._montage_viewport_plan(view_state)
        all_indices = viewport_plan.all_indices
        viewport_shape = viewport_plan.viewport_shape
        tile_shape = viewport_plan.tile_shape
        plan = viewport_plan.plan
        # The auto-fit writes the camera, Qt emits ``sigRangeChanged``
        # synchronously, and ``ViewportBridge`` used to answer a programmatic
        # montage range change by calling ``retarget_montage_viewport()`` right
        # here -- which cannot take its viewport-only path (the live session is
        # still keyed to the OUTGOING view state, so
        # ``_try_update_montage_viewport_only`` rejects on ``session.key``) and
        # so ran a whole nested ``update_image_view`` inside this one, inside
        # the ``plan_start`` span.  Measured on a 144-tile wgpu montage return,
        # 7 returns per arm, median: 2 entries and ``viewport_plan_ms`` 22.4 ms
        # of a 23.1 ms callback, against 1 entry and 1.9 ms of 20.3 ms with
        # this flag.  A depth-resolved run put the nested entry at 29.5 ms wall
        # inside a 34.4 ms outer one, whose own plan phase was 1.0 ms — the
        # plan was never slow, the label was billing a second render.  It was
        # invisible in the bill of materials because the nested entry writes
        # the same five phase attributes this one then overwrites.
        #
        # The flag makes that camera write announce itself so the bridge can
        # DEFER instead, and the two lines below are why deferring loses
        # nothing: this entry immediately replans against the very range the
        # signal was reporting.  try/finally rather than a plain pair, because
        # a flag left set would silently disable the bridge's synchronous
        # retarget for the life of the window.
        self._montage_entry_autofit_active = True
        try:
            fitted = self._maybe_auto_fit_montage_tiles(plan)
        finally:
            self._montage_entry_autofit_active = False
        if fitted:
            viewport_plan = self._montage_viewport_plan(
                view_state,
                view_range=self._current_montage_global_view_range(),
            )
            viewport_shape = viewport_plan.viewport_shape
            tile_shape = viewport_plan.tile_shape
            plan = viewport_plan.plan
        # The committed frame owns coordinate semantics. On a first frame there
        # is no prior surface to preserve, so publishing the planned extent is
        # useful for startup fit. A replacement keeps the acknowledged extent
        # until its pixels commit; moving the camera to uncommitted geometry
        # makes compatible retained tiles appear to go black.
        self._publish_first_frame_content_extent(plan)
        self._montage_live_layout_reflow = False
        previous_session_plan = getattr(getattr(self, "_frame_session", None), "plan", None)
        self._remap_montage_rois_for_layout_reflow(previous_session_plan, plan)
        current_range = viewport_plan.view_range
        display_tiles = viewport_plan.candidate_tiles(margin_tiles=VISIBLE_MARGIN_TILES)
        candidate_tiles = viewport_plan.candidate_tiles(
            margin_tiles=(
                COVERAGE_MARGIN_TILES
                if viewport_plan.persistent_tile_residency
                else VISIBLE_MARGIN_TILES
            ),
            prioritize=True,
        )
        shader_display = viewport_plan.shader_display
        canonical_orientation = bool(
            image_view_backend_capabilities(self.win.img_view).display_axis_transpose
        )
        # Single derivation point: mirror the active backend's display-transpose
        # capability onto the evaluator so every tile key/materialization for
        # this frame (and the prefetch/retarget paths that persist until the
        # next build) shares one orientation policy.
        self.win.operation_evaluator.canonical_orientation = canonical_orientation
        output_dtype = getattr(
            getattr(self.win, "data", None),
            "dtype",
            getattr(document.base_data, "dtype", np.dtype(float)),
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
                self.win,
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
        if self._maybe_retarget_frame_session(
            getattr(self, "_frame_session", None),
            document=document,
            axis=axis,
            view_state=view_state,
            viewport_plan=viewport_plan,
            plan=plan,
            policy=policy,
            colormap_lut=colormap_lut,
            window_mode=window_mode,
            force_auto=force_auto,
            user_levels=user_levels,
            output_dtype=output_dtype,
            shader_display=shader_display,
            cached_tiles=cached_tiles,
            missing_tiles=missing_tiles,
            skipped_tiles=skipped_tiles,
            all_indices=all_indices,
            display_tiles=display_tiles,
            current_range=current_range,
            viewport_shape=viewport_shape,
        ):
            return
        render_generation = self._capture_render_generation()
        lod_policy_mode = self._montage_quality_policy_mode()
        allow_anisotropic_lod = bool(
            image_view_backend_capabilities(self.win.img_view).anisotropic_lod_pages
        )
        initial_demand = select_lod_demand(
            current_range,
            viewport_shape,
            plan.tile_shape,
            allow_anisotropy=allow_anisotropic_lod,
        )
        # Interaction fast path: during a scrub/pan burst, supersedable stage
        # planning runs only for the step the user lands on. Native policy and
        # first-ever montage builds still plan inline because they have no
        # retained screen to carry.
        previous_session = getattr(self, "_frame_session", None)
        defer_stage_planning = bool(
            missing_tiles
            and _viewport_interaction_active(self)
            and self._montage_quality_policy_mode() == LOD_POLICY_RESIDENT
            and previous_session is not None
            # Only a committed predecessor on the same axis can carry the
            # screen; a first fill or axis change is not a scrub step.
            and bool(getattr(previous_session, "display_committed", False))
            and getattr(previous_session, "montage_axis", None) == axis
        )
        native_target_required = bool(
            not defer_stage_planning
            and render_lod.missing_tiles_require_native_target(
                lod_policy_mode,
                initial_demand,
            )
        )
        stage_plan_start = perf_counter()
        if missing_tiles and not native_target_required and not defer_stage_planning:
            stage_plan = montage_commit.deferred_stage_fan_in_plan()
        elif defer_stage_planning:
            stage_plan = montage_commit.hot_cached_stage_fan_in_plan(
                self,
                document,
                missing_tiles,
            )
            if not montage_commit.stage_fan_in_plan_has_existing_sources(stage_plan):
                stage_plan = montage_commit.deferred_stage_fan_in_plan()
            self._montage_stage_plans_deferred = (
                int(getattr(self, "_montage_stage_plans_deferred", 0) or 0) + 1
            )
        else:
            stage_plan = montage_commit.build_stage_fan_in_plan(self, document, missing_tiles)
        self._last_montage_stage_plan_ms = (perf_counter() - stage_plan_start) * 1000.0
        session_setup_start = perf_counter()
        session_key = frame_session_key(
            _document_key(document), view_state, viewport_plan, colormap_lut
        )
        level_key = self._montage_level_key(document, view_state, all_indices, colormap_lut)
        frame_plan = self._montage_frame_planner().plan(
            target=FrameTarget(
                semantic_key=session_key,
                viewport_key=current_range,
                presentation_key=(
                    str(window_mode),
                    normalize_bounds(user_levels),
                    bool(force_auto),
                ),
                quality="exact-visible",
            ),
            view_state=view_state,
            display_shape=plan.display_shape,
            backend_capabilities=image_view_backend_capabilities(self.win.img_view),
            viewport_shape=viewport_shape,
            view_range=current_range,
            memory_policy=policy,
            montage_plan=plan,
        )
        session_id = int(getattr(self, "_frame_session_id", 0)) + 1
        self._frame_session_id = session_id
        lod_preview_level = (
            preview_level_for_tile_shape(
                plan.tile_shape, min_level=render_lod.PREVIEW_FLOOR_MIN_LEVEL
            )
            if lod_policy_mode == LOD_POLICY_RESIDENT
            else 0
        )
        session = FrameSession(
            session_id=session_id,
            key=session_key,
            semantic_key=montage_tile_semantic_key(
                _document_key(document),
                view_state,
                viewport_plan,
                colormap_lut,
                canonical_orientation=canonical_orientation,
            ),
            render_generation=render_generation,
            level_key=level_key,
            level_expected_indices=tuple(int(index) for index in all_indices),
            frame_plan=frame_plan,
            plan=plan,
            view_state=view_state,
            document=document,
            montage_axis=axis,
            source_anchoring=self._session_source_anchoring(document, view_state, axis),
            colormap_lut=colormap_lut,
            viewport_shape=viewport_shape,
            view_range=current_range,
            output_dtype=np.dtype(output_dtype),
            rgb=view_state.channel == ChannelMode.COMPLEX,
            window_mode=window_mode,
            force_auto=force_auto,
            visible_tiles=tuple(display_tiles),
            rendered_tiles={
                int(rendered.tile.montage_index): rendered for rendered in cached_tiles
            },
            loading_tiles=set(),
            skipped_tiles={int(tile.montage_index) for tile in skipped_tiles},
            shader_display=bool(shader_display),
            canonical_orientation=canonical_orientation,
            stage_fan_in=montage_commit.stage_fan_in_state(stage_plan),
            defer_side_panels=bool(defer_side_panels),
            applied_level_source=(
                pending_auto_level_source
                if pending_auto_level_source is not None
                else (
                    carried_user_level_source
                    if carried_user_level_source is not None
                    else (None if previous_frame is None else fallback_level_source(previous_frame))
                )
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
            lod_policy_mode=lod_policy_mode,
            lod_native_reason=render_lod.native_policy_reason_for_renderer(self),
            lod_anisotropic_pages=allow_anisotropic_lod,
            lod_preview_level=lod_preview_level,
            lod_preview_min_level=lod_preview_level,
            tile_residency_budget_bytes=tile_residency_budget_bytes(policy),
            retain_out_of_view_payloads=self._retains_out_of_view_payloads(),
            lod_page_cache=(
                self._lod_page_cache() if lod_policy_mode == LOD_POLICY_RESIDENT else None
            ),
            # Handed over together with the pyramid, always: an index of keys
            # into a cache the session does not have would name nothing.
            lod_floor_index=(
                self._lod_floor_index() if lod_policy_mode == LOD_POLICY_RESIDENT else None
            ),
        )
        stage_planning_deferred = bool(
            defer_stage_planning or (missing_tiles and not native_target_required)
        )
        session.stage_planning_deferred = stage_planning_deferred
        session.stage_planning_async = False
        session.deferred_missing_tiles = tuple(missing_tiles) if stage_planning_deferred else ()
        # The dying session's planned-but-undrained LOD requests hold
        # singleflight claims in the shared pyramid; scrubbing back to the
        # same slice would find those levels permanently claimed (stale
        # wrong-LOD tiles).  Balance them before the replacement takes over.
        dying_session = getattr(self, "_frame_session", None)
        dying_pipeline = getattr(dying_session, "pipeline", None)
        if dying_pipeline is not None:
            # Each session owns its own FramePipeline. A semantic/layout
            # rebirth never retargets the dying pipeline, so its ordinary
            # semantic-key supersession hook cannot clear the old scope.
            # Close it explicitly before the successor starts: otherwise its
            # queued preview tasks keep consuming the shared workers ahead of
            # the successor's complete-scope preview batch.
            dying_pipeline.close()
        render_lod.release_session_claims(dying_session)
        # Backend slots outlive sessions (persistent tile residency), so the
        # identity ground truth from the last report stays valid — but a fresh
        # lifecycle starts without backend slot truth, blind to inherited
        # stale slots until its own first report.  Inherit the backend snapshot
        # into the new lifecycle; tiles absent from the new plan fall out
        # naturally (no current payload → mismatch scan skips them).
        if dying_session is not None:
            inherited = getattr(dying_session.lifecycle, "backend_presented_identities", None)
            if inherited:
                session.lifecycle.backend_presented_snapshot(inherited)
        # Activating a new target is an atomic visibility boundary.  Backend
        # residency outlives a FrameSession, but inherited slot mappings do
        # not: until the new session explicitly rebinds and acknowledges them,
        # they are not evidence for the new target (including viewport-only
        # retargets whose semantic key intentionally remains stable).
        #
        # "Pixels stay visible" is a separate axis from "mappings are not
        # evidence" (ADR 0051).  A slice-index-only rebirth keeps DRAWING the
        # predecessor's plane — a stale-but-honest preview, exactly like a
        # video player between frames — while all session bookkeeping below
        # still starts cold: no acknowledgement inheritance, no seeded
        # evidence (payload seeding requires exact source identities), and
        # the successor's first commit swaps the drawn pixels atomically.
        # The transition planner keeps the predecessor as independently named
        # physical truth while compatible successor semantics are prepared.
        # It rejects a different staged document or surface/backend contract;
        # derived representation, view state, and auto layout belong to the
        # complete successor transaction and cannot justify a black flash.
        surface = surface_for_view(self.win.img_view)
        transition = plan_presentation_transition(
            dying_session,
            session,
            predecessor_visible=tiled_presentation_visible(surface),
        )
        retain_stale_pixels = bool(transition.retain_pixels)
        session.atomic_successor_pending = bool(transition.atomic_successor)
        # A bridge successor replaced before its first commit leaves the
        # ORIGINAL bridge predecessor on the surface; the flag lets the next
        # rebirth continue that bridge instead of blanking against a
        # pixel-less dying session (plan_presentation_transition).
        session.presentation_bridge_pending = bool(
            transition.retain_pixels and transition.reason == "montage-axis-bridge"
        )
        emit_trace(
            "presentation_transition_retention",
            session_id=int(session.session_id),
            predecessor_session_id=int(getattr(dying_session, "session_id", 0) or 0),
            retained=bool(retain_stale_pixels),
            reason=str(transition.reason),
            detail=str(transition.detail),
            force_auto=bool(session.force_auto),
            montage_axis=getattr(session, "montage_axis", None),
            atomic_successor_pending=bool(session.atomic_successor_pending),
        )
        self._frame_session_transition_retained_pixels = bool(retain_stale_pixels)
        if retain_stale_pixels:
            self._frame_session_transitions_retained = (
                int(getattr(self, "_frame_session_transitions_retained", 0) or 0) + 1
            )
            self._slice_retention_started_at = perf_counter()
            self._slice_retention_session_id = int(session.session_id)
            emit_trace(
                "slice_retention_started",
                session_id=int(session.session_id),
                predecessor_session_id=int(getattr(dying_session, "session_id", 0) or 0),
                transition_count=int(self._frame_session_transitions_retained),
            )
        else:
            self._slice_retention_started_at = None
            self._slice_retention_session_id = None
        surface.invalidate_tiled_presentation(
            "frame-session-transition",
            hide_pixels=not retain_stale_pixels,
        )
        if retain_stale_pixels:
            # The bridge is honest only where the camera is.  A rebirth starts
            # a cold lifecycle, so nothing the surface is still drawing belongs
            # to the successor; everything outside its viewport is superseded
            # and a pan is a camera uniform with no per-tile predicate in the
            # path to refuse it.  The viewport itself keeps its stale-but-
            # honest pixels, which is what the bridge is for.
            self._hide_superseded_presentation(
                session,
                "frame-session-transition",
                {
                    int(tile.montage_index)
                    for tile in tuple(getattr(session.plan, "tiles", ()) or ())
                    if int(tile.montage_index) not in session.visible_tile_numbers
                },
            )
        self._frame_session = session
        # A live session can receive cached/refined level evidence as soon as
        # its level work is queued below.  Attach its one effects owner first
        # so an immediate kernel completion cannot observe a current session
        # without the presentation gate needed to publish that evidence.
        self._frame_pipeline_for_session(session)
        # A viewport-update token armed for the dying session would make every
        # later apply_montage_viewport_retarget bail as stale — a dead
        # continuation (lost-wakeup class).  The new session's construction
        # subsumes any pending retarget; clear the token so future runs (timer
        # or explicit) act on the current session.
        self._montage_viewport_update_token = None
        self._ensure_montage_watchdog()
        apply_restored_viewport = getattr(self.win, "_apply_viewport_continuity_when_ready", None)
        if callable(apply_restored_viewport):
            apply_restored_viewport()
        _complete_inline_work(
            self,
            WorkItem(
                key=("montage_visible_planning", session.key, int(session.session_id)),
                lane=WorkLane.VISIBLE_PLANNING,
                frame_target=session.frame_plan.target,
                supersession_key=("montage-visible", session.key),
                supersession_value=int(session.session_id),
                estimated_cpu_ms=float(self._last_montage_viewport_plan_ms or 0.0)
                + float(self._last_montage_cache_resolve_ms or 0.0)
                + float(self._last_montage_stage_plan_ms or 0.0),
                estimated_bytes=int(single_estimate) * max(1, len(tuple(display_tiles or ()))),
            ),
        )
        if pending_auto_level_source is not None:
            self._pending_auto_level_source = None
        self._ensure_montage_level_stats(level_key, expected_indices=all_indices)
        self._queue_montage_cached_level_stats(session, cached_tiles, seed_if_empty=True)
        self._last_frame_session_setup_ms = (perf_counter() - session_setup_start) * 1000.0
        initial_commit_start = perf_counter()
        try:
            # Execute the first *governed* slice on the retarget edge. This is
            # not the former whole-plan reconcile: the governor's item, byte,
            # and deadline decision already bounds the transaction below.
            # Deferring even this slice loses the only wakeup when the new
            # target is fully resident and therefore produces no worker
            # completion. It also exposes an empty backend between a retained
            # predecessor and its mapping-only successor.
            self.commit_frame_session_presentation(session)
        except MemoryError as exc:
            show_status_message(self.win, str(exc), timeout=6000)
            return
        finally:
            self._last_montage_initial_commit_ms = (perf_counter() - initial_commit_start) * 1000.0
        if session.is_complete():
            self._finish_frame_session_if_complete(session)
            if defer_side_panels or _viewport_interaction_active(self):
                self.win._deferred_side_panel_refresh_pending = True
            else:
                self.win._update_operation_dock()
            self._schedule_montage_cached_level_stats(session)
            return
        visible_complete = self._settle_montage_visible_plan_if_complete(session)
        self.win.prefetch_evaluation_controller.cancel_prefetch()
        if not visible_complete:
            self.win.operation_evaluator.last_status = CacheStatusSnapshot(
                CacheStatus.COMPUTING, "Evaluating image frame"
            )
        if defer_side_panels or _viewport_interaction_active(self):
            self.win._deferred_side_panel_refresh_pending = True
        else:
            self.win._update_operation_dock()
        if not visible_complete:
            self.show_frame_session_slow_overlay(session)
        self._schedule_montage_cached_level_stats(session)
        if defer_stage_planning:
            montage_commit.submit_deferred_stage_fan_in_plan(self, session, missing_tiles)
            self.request_montage_replan(session)
        else:
            montage_commit.submit_stage_tasks(self, session, stage_plan["stage_requests"])
            self.request_montage_replan(session)

    def _maybe_retarget_frame_session(
        self,
        previous_session,
        *,
        document,
        axis,
        view_state,
        viewport_plan,
        plan,
        policy,
        colormap_lut,
        window_mode,
        force_auto,
        user_levels,
        output_dtype,
        shader_display,
        cached_tiles,
        missing_tiles,
        skipped_tiles,
        all_indices,
        display_tiles,
        current_range,
        viewport_shape,
    ) -> bool:
        """Retarget the live session to a new index window instead of a rebirth.

        ADR 0051 P2 (session-rebirth cost): index-window scrubs and
        camera-only viewport changes reuse the session object. Backend
        acknowledgement state and drawn payloads survive; missing stage
        planning is either attached from retained source data immediately or
        submitted as kernel-owned supersedable work. Returns True when the
        retarget handled the step.
        """

        def _reject(reason: str) -> bool:
            self._frame_session_retarget_last_reject = reason
            rejects = getattr(self, "_frame_session_retarget_rejects", None)
            if rejects is None:
                rejects = {}
                self._frame_session_retarget_rejects = rejects
            rejects[reason] = int(rejects.get(reason, 0)) + 1
            return False

        session = previous_session
        if session is None or session is not getattr(self, "_frame_session", None):
            return _reject("no-session")
        if axis is None or getattr(session, "montage_axis", None) is None:
            # Normal sliced images share this tiled path with axis=None; a
            # slice change is new semantic content behind an unchanged
            # layout, not an index-window move.  Only true montage sessions
            # retarget.
            return _reject("no-axis")
        if not bool(getattr(session, "display_committed", False)):
            return _reject("uncommitted")
        if getattr(session, "montage_axis", None) != axis:
            return _reject("axis")
        # An image-axis ORDER change (X/Y swap / transpose) keeps the montage
        # axis, layout, and source indices, so the index-window remap would
        # treat every tile as an unchanged/remappable slot and re-present the
        # previously materialized plane in its OLD orientation.  On a legacy
        # (non-canonical) backend a transpose changes what every tile's pixels
        # mean, not which sources are shown, so it is a rebirth, not an
        # index-window move: reject retarget and let the caller rebuild the
        # session with freshly oriented tiles.  (Axis FLIPS never reach here --
        # they are a viewbox transform applied without a montage re-render.)
        #
        # When the backend applies the swap as a DISPLAY transform
        # (``canonical_orientation``), the tiles are materialized canonically
        # and the retarget legitimately reuses them: a square-tile transpose
        # keeps the geometry and re-commits with the new display mapping, and a
        # non-square transpose falls through to the geometry reject below (a
        # re-layout that still hits the canonical cache / GPU residency).
        if not bool(getattr(session, "canonical_orientation", False)) and tuple(
            getattr(session.view_state, "image_axes", None) or ()
        ) != tuple(getattr(view_state, "image_axes", None) or ()):
            return _reject("image-axes-order")
        if bool(force_auto):
            return _reject("force-auto")
        if bool(skipped_tiles) or bool(getattr(session, "skipped_tiles", None)):
            return _reject("skipped-tiles")
        if _document_key(session.document) != _document_key(document):
            return _reject("document")
        if session.lod_policy_mode != self._montage_quality_policy_mode():
            return _reject("lod-policy")
        if session.window_mode != window_mode:
            return _reject("window-mode")
        if session.user_levels_override != user_levels:
            return _reject("user-levels")
        if session.colormap_lut is not colormap_lut:
            return _reject("colormap")
        if bool(getattr(session, "shader_display", False)) != bool(shader_display):
            return _reject("shader-display")
        if session.output_dtype != np.dtype(output_dtype):
            return _reject("dtype")
        if bool(session.rgb) != bool(view_state.channel == ChannelMode.COMPLEX):
            return _reject("rgb")
        # Level convergence no longer forces rebirth: level upserts are
        # machine-gated, and retarget_index_window resets per-window evidence
        # while kept tiles continue converging.
        previous_geometry = getattr(session.plan, "geometry", None)
        geometry = getattr(plan, "geometry", None)
        if previous_geometry is None or geometry is None:
            return _reject("geometry-missing")
        if (
            tuple(previous_geometry.tile_shape) != tuple(geometry.tile_shape)
            or int(previous_geometry.columns) != int(geometry.columns)
            or int(previous_geometry.rows) != int(geometry.rows)
            or int(previous_geometry.gap) != int(geometry.gap)
            or len(previous_geometry.indices) != len(geometry.indices)
        ):
            return _reject("geometry")
        session_key = frame_session_key(
            _document_key(document), view_state, viewport_plan, colormap_lut
        )
        viewport_changed = (
            tuple(session.viewport_shape) != tuple(viewport_shape)
            or session.view_range != current_range
        )
        if session_key == session.key and viewport_changed:
            # Viewport retargeting may create new work tokens and commit a
            # presentation without rebuilding the session.  Carry the render
            # request's ordering generation into that work before it starts.
            # If the specialized retarget declines, restore the old stamp so
            # the ordinary rejection/rebirth path still owns the transition.
            previous_render_generation = int(session.render_generation)
            session.render_generation = self._capture_render_generation()
            if self._try_update_montage_viewport_only():
                self._frame_session_reuses = int(getattr(self, "_frame_session_reuses", 0) or 0) + 1
                self._refresh_committed_display_frame_generation(session.render_generation)
                return True
            session.render_generation = previous_render_generation
            return _reject("viewport-retarget")
        if session_key == session.key:
            # Same identity: refresh the generation stamp, commit genuine
            # dirt, and let the standing machinery converge without rebirth.
            self._frame_session_reuses = int(getattr(self, "_frame_session_reuses", 0) or 0) + 1
            session.render_generation = self._capture_render_generation()
            self._montage_viewport_update_token = None
            self._ensure_montage_watchdog()
            reuse_commit_start = perf_counter()
            try:
                # Only genuinely pending presentation work commits here; the
                # flush/level continuations own their own pacing, and a
                # settled re-render must stay a true no-op.
                if (
                    session.dirty_payloads
                    or session.pending_removals
                    or session.pending_payload_upserts
                    or not session.display_committed
                ):
                    self.commit_frame_session_presentation(session)
            finally:
                self._last_montage_initial_commit_ms = (
                    perf_counter() - reuse_commit_start
                ) * 1000.0
            # A settled reuse commits nothing, so the committed display frame is
            # never re-stamped with the generation just captured above.  Realign
            # its ordering stamp here or the currency predicate reports an idle
            # ``committed_frame_stale`` stall on a frame whose pixels are current.
            self._refresh_committed_display_frame_generation(session.render_generation)
            self._last_montage_stage_plan_ms = 0.0
            self._last_frame_session_setup_ms = 0.0
            if session.defer_side_panels or _viewport_interaction_active(self):
                self.win._deferred_side_panel_refresh_pending = True
            else:
                self.win._update_operation_dock()
            return True
        # Change detection uses the same memoized semantic key batch that the
        # lazy tile_source_ids fill uses, so unchanged tiles are exact hits.
        source_ids_start = perf_counter()
        try:
            tile_key_for = self.win.operation_evaluator.montage_tile_key_batch(
                colormap_lut=colormap_lut,
                document=document,
                shader_display=bool(shader_display),
            )
            new_source_ids = {
                int(tile.montage_index): tile_key_for(tile.view_state)
                for tile in tuple(getattr(plan, "tiles", ()) or ())
            }
        except Exception:
            return False
        self._last_montage_retarget_source_ids_ms = (perf_counter() - source_ids_start) * 1000.0
        semantic_key = montage_tile_semantic_key(
            _document_key(document),
            view_state,
            viewport_plan,
            colormap_lut,
            canonical_orientation=bool(
                getattr(self.win.operation_evaluator, "canonical_orientation", False)
            ),
        )
        level_key = self._montage_level_key(document, view_state, all_indices, colormap_lut)
        frame_plan_start = perf_counter()
        frame_plan = self._montage_frame_planner().plan(
            target=FrameTarget(
                semantic_key=session_key,
                viewport_key=current_range,
                presentation_key=(
                    str(window_mode),
                    normalize_bounds(user_levels),
                    bool(force_auto),
                ),
                quality="exact-visible",
            ),
            view_state=view_state,
            display_shape=plan.display_shape,
            backend_capabilities=image_view_backend_capabilities(self.win.img_view),
            viewport_shape=viewport_shape,
            view_range=current_range,
            memory_policy=policy,
            montage_plan=plan,
        )
        self._last_montage_retarget_frame_plan_ms = (perf_counter() - frame_plan_start) * 1000.0
        render_generation = self._capture_render_generation()
        # Undrained pyramid claims from the old window are balanced exactly as
        # on a rebirth; source identities are window-agnostic, so re-plans
        # re-claim cheaply.
        release_start = perf_counter()
        render_lod.release_session_claims(session)
        self._last_montage_retarget_release_ms = (perf_counter() - release_start) * 1000.0
        session_id = int(getattr(self, "_frame_session_id", 0)) + 1
        self._frame_session_id = session_id
        setup_start = perf_counter()
        model_start = perf_counter()
        deferred_before = frozenset(session.deferred_out_of_view_invalidations)
        stats = session.retarget_index_window(
            session_id=session_id,
            key=session_key,
            semantic_key=semantic_key,
            level_key=level_key,
            render_generation=render_generation,
            view_state=view_state,
            source_anchoring=self._session_source_anchoring(document, view_state, axis),
            plan=plan,
            frame_plan=frame_plan,
            all_indices=all_indices,
            new_source_ids=new_source_ids,
            cached_tiles={int(rendered.tile.montage_index): rendered for rendered in cached_tiles},
            visible_tiles=tuple(display_tiles),
        )
        self._last_montage_retarget_model_ms = (perf_counter() - model_start) * 1000.0
        # An index-window retarget is a data change that never reaches
        # ``invalidate_tiled_presentation``: the session object survives, so
        # nothing tells the surface to stop drawing the predecessor.  Exactly
        # the tiles this retarget just deferred are the superseded out-of-view
        # ones — the tiles a quick pan would reveal — so hide those and nothing
        # else.  A camera move that reaches this path with every source
        # unchanged defers nothing and therefore hides nothing.
        hide_start = perf_counter()
        self._hide_superseded_presentation(
            session,
            "index-window-retarget",
            frozenset(session.deferred_out_of_view_invalidations) - deferred_before,
        )
        self._last_montage_retarget_hide_superseded_ms = (perf_counter() - hide_start) * 1000.0
        session.tile_residency_budget_bytes = tile_residency_budget_bytes(policy)
        session.force_auto = bool(force_auto)
        session.user_levels_override = user_levels
        stage_start = perf_counter()
        stage_cpu_start = thread_time()
        stage_plan_deferred = False
        if missing_tiles:
            hot_call_cpu_start = thread_time()
            hot_stage_plan = montage_commit.hot_cached_stage_fan_in_plan(
                self,
                session.document,
                missing_tiles,
            )
            self._last_montage_retarget_hot_call_cpu_ms = (
                thread_time() - hot_call_cpu_start
            ) * 1000.0
            hot_predicate_cpu_start = thread_time()
            has_existing_stage = montage_commit.stage_fan_in_plan_has_existing_sources(
                hot_stage_plan
            )
            self._last_montage_retarget_hot_predicate_cpu_ms = (
                thread_time() - hot_predicate_cpu_start
            ) * 1000.0
            if not has_existing_stage:
                deferred_cpu_start = thread_time()
                hot_stage_plan = montage_commit.deferred_stage_fan_in_plan()
                stage_plan_deferred = True
                self._last_montage_retarget_hot_deferred_cpu_ms = (
                    thread_time() - deferred_cpu_start
                ) * 1000.0
        else:
            hot_stage_plan = montage_commit.deferred_stage_fan_in_plan()
        self._last_montage_retarget_hot_stage_ms = (perf_counter() - stage_start) * 1000.0
        self._last_montage_retarget_hot_stage_cpu_ms = (thread_time() - stage_cpu_start) * 1000.0
        attach_start = perf_counter()
        session.attach_stage_fan_in(montage_commit.stage_fan_in_state(hot_stage_plan))
        self._last_montage_retarget_attach_ms = (perf_counter() - attach_start) * 1000.0
        session.stage_planning_deferred = bool(missing_tiles and stage_plan_deferred)
        session.stage_planning_async = False
        # The deferred backlog is only meaningful WITH the deferral flag: it is
        # the argument ``complete_deferred_stage_fan_in`` replans from, and that
        # owner bails unless ``stage_planning_deferred`` is set.  Recording it
        # unconditionally left a retarget that found an existing stage plan
        # (an operation-pipeline montage whose shared stage is already warm)
        # holding a backlog nobody would ever drain, and ``is_complete`` reads
        # it directly — so such a session never reported complete even after its
        # pixels settled, stranding every completion continuation behind it
        # (the semantic level-evidence owner among them).  The session-build
        # path already pairs them this way.
        session.deferred_missing_tiles = (
            tuple(missing_tiles) if session.stage_planning_deferred else ()
        )
        session.tile_compute_cache_hits = int(stats["hits"])
        session.tile_compute_waiting_for_stage = len(hot_stage_plan["waiting_indices"])
        session.stage_backed_tiles_pending = len(hot_stage_plan["waiting_indices"])
        session.lead_direct_tiles = 0
        session.retained_stage_index = hot_stage_plan["retained_stage_index"]
        session.retained_stage_decision = (
            hot_stage_plan["retained_stage_decision"] or "deferred-retarget"
        )
        session.repeated_expensive_stage_per_tile = bool(
            hot_stage_plan["repeated_expensive_stage_per_tile"]
        )
        if session.stage_planning_deferred:
            self._montage_stage_plans_deferred = (
                int(getattr(self, "_montage_stage_plans_deferred", 0) or 0) + 1
            )
        self._frame_session_retargets = int(getattr(self, "_frame_session_retargets", 0) or 0) + 1
        self._montage_cached_tiles_last_session = int(stats["hits"])
        self._montage_missing_tiles_last_session = len(missing_tiles)
        self._montage_viewport_update_token = None
        self._ensure_montage_watchdog()
        level_setup_start = perf_counter()
        self._ensure_montage_level_stats(level_key, expected_indices=all_indices)
        self._queue_montage_cached_level_stats(session, tuple(cached_tiles), seed_if_empty=True)
        self._last_montage_retarget_level_setup_ms = (perf_counter() - level_setup_start) * 1000.0
        self._last_montage_stage_plan_ms = 0.0
        self._last_frame_session_setup_ms = (perf_counter() - setup_start) * 1000.0
        _complete_inline_work(
            self,
            WorkItem(
                key=("montage_visible_planning", session.key, int(session.session_id)),
                lane=WorkLane.VISIBLE_PLANNING,
                frame_target=session.frame_plan.target,
                supersession_key=("montage-visible", session.key),
                supersession_value=int(session.session_id),
                estimated_cpu_ms=float(self._last_montage_viewport_plan_ms or 0.0)
                + float(self._last_montage_cache_resolve_ms or 0.0)
                + float(self._last_frame_session_setup_ms or 0.0),
                estimated_bytes=0,
            ),
        )
        initial_commit_start = perf_counter()
        try:
            self.commit_frame_session_presentation(session)
        finally:
            self._last_montage_initial_commit_ms = (perf_counter() - initial_commit_start) * 1000.0
        if session.defer_side_panels or _viewport_interaction_active(self):
            self.win._deferred_side_panel_refresh_pending = True
        else:
            self.win._update_operation_dock()
        if missing_tiles:
            if session.stage_planning_deferred:
                montage_commit.submit_deferred_stage_fan_in_plan(self, session, missing_tiles)
            else:
                montage_commit.submit_stage_tasks(self, session, hot_stage_plan["stage_requests"])
        # Retained/floor cache hits can be valid first pixels without satisfying
        # the successor target.  They produce no ``missing_tiles`` entry, so
        # index-window retargeting must always give the ladder one chance to
        # admit the exact/desired follow-up for any unsettled target.
        self.retarget_frame_pipeline(session)
        return True

    def _montage_cache_resolution_context(
        self,
        *,
        document,
        colormap_lut,
        shader_display: bool,
    ) -> MontageCacheResolutionContext:
        """Snapshot the shared lookup state for resolving tiles from cache.

        The retained-store scan and the committed-frame payload map are
        identical for every tile of one retarget pass; building them per tile
        turned the budgeted viewport-additions loop into a full store copy per
        addition (O(N^2) at field size).
        """

        selected_lod_factor = 1
        # ADR 0050: under the resident policy a retained payload at any LOD
        # level is reusable evidence of the tile's native content (its exact
        # image/semantic planes are always native).  Rejecting reduced-level
        # payloads here forced pan/scroll re-evaluation of tiles that were
        # already on screen, which is what read as black/placeholder flashes
        # during LOD transitions.  The session re-selects the presented level
        # from live demand and streams the swap.
        reuse_any_lod = self._montage_quality_policy_mode() == LOD_POLICY_RESIDENT
        retained_store = self._retained_tiled_payload_store()
        previous_payloads = retained_store.payloads_by_base_source(
            lod_factor=None if reuse_any_lod else selected_lod_factor
        )
        previous_payloads.update(
            {
                key: payload
                for key, payload in _previous_tiled_payloads_by_base_source(
                    getattr(self.win, "_committed_display_frame", None)
                ).items()
                if reuse_any_lod or _payload_lod_matches(payload, selected_lod_factor)
            }
        )
        tile_key_for = self.win.operation_evaluator.montage_tile_key_batch(
            colormap_lut=colormap_lut,
            document=document,
            shader_display=shader_display,
        )
        return MontageCacheResolutionContext(
            reuse_any_lod=reuse_any_lod,
            selected_lod_factor=selected_lod_factor,
            retained_store=retained_store,
            previous_payloads=previous_payloads,
            tile_key_for=tile_key_for,
        )

    def _resolve_montage_tiles_from_cache(
        self,
        tiles,
        *,
        document,
        axis: int,
        colormap_lut,
        shader_display: bool,
        context: MontageCacheResolutionContext | None = None,
    ) -> tuple[list[RenderedTile], list[object]]:
        """Resolve only the supplied tiles from semantic CPU caches."""

        if context is None:
            context = self._montage_cache_resolution_context(
                document=document,
                colormap_lut=colormap_lut,
                shader_display=shader_display,
            )
        reuse_any_lod = context.reuse_any_lod
        selected_lod_factor = context.selected_lod_factor
        retained_store = context.retained_store
        previous_payloads = context.previous_payloads
        tile_key_for = context.tile_key_for
        cached_tiles: list[RenderedTile] = []
        missing_tiles: list[object] = []
        total_lookup_ms = 0.0
        last_hit = False
        tile_tuple = tuple(tiles)
        for tile in tile_tuple:
            display_cache_start = perf_counter()
            tile_key = tile_key_for(tile.view_state)
            cached = self.win.operation_evaluator.cached_montage_tile_by_key(tile_key)
            total_lookup_ms += (perf_counter() - display_cache_start) * 1000.0
            last_hit = cached is not None
            if cached is None:
                previous_payload = previous_payloads.get(tile_key)
                if previous_payload is None:
                    previous_payload = retained_store.resolve(
                        tile_key,
                        None if reuse_any_lod else selected_lod_factor,
                        tile.view_state,
                        shader_display=shader_display,
                    )
                if previous_payload is None or not _payload_compatible_with_tile(
                    previous_payload,
                    tile.view_state,
                    shader_display=shader_display,
                ):
                    missing_tiles.append(tile)
                else:
                    cached_tiles.append(
                        _rendered_tile_from_previous_payload(tile, previous_payload)
                    )
            else:
                cached_tiles.append(_rendered_tile_from_cached_display(tile, cached))
        self._last_montage_display_cache_lookup_ms = total_lookup_ms
        self._last_montage_display_cache_hit = last_hit if tile_tuple else False
        if cached_tiles and reuse_any_lod:
            # Each cached resolve under the resident LOD policy is a pipeline
            # evaluation a display-LOD-driven rebuild did not have to run.
            self._montage_quality_pipeline_reruns_avoided = int(
                getattr(self, "_montage_quality_pipeline_reruns_avoided", 0) or 0
            ) + len(cached_tiles)
        return cached_tiles, missing_tiles

    def _published_scope_covers_camera(self, session, viewport_plan) -> bool:
        """Whether skipping the pass can leave the session's scope short.

        ``retarget_viewport`` is not only the planner's front door: it is the
        SOLE publisher of the session's notion of what the user is looking at.
        ``visible_tiles``/``visible_tile_numbers``, the lifecycle target set,
        the level-presentation scope, the tile priority order and -- through
        the frame plan it precedes -- ``required_tile_numbers`` all come from
        it, and other actors read every one of them. A skipped pass leaves all
        of that describing the camera of the last pass that ran.

        That is why the visible ring's IDENTITY was a sound term to key on,
        and it is the half of the question ``camera_pending_tiles`` does not
        answer: the pending set proves the PLANNER would do nothing, not that
        the published scope still describes the screen.

        It is also why the gate does not simply stop at the planner. Timed
        inside one full pass, 272 tiles, wgpu, latch disabled so every call is
        a full body (10.09 ms):

            retarget_frame_pipeline   6.04 ms   60 %   <- the planner
            retarget_viewport         2.13 ms   21 %   <- publication
            sync_lifecycle_scope      1.69 ms   17 %   <- publication
            mark_ladder_swaps         0.27 ms    3 %
            frame planner .plan()     0.10 ms    1 %

        Gating only the planner is unconditionally sound and buys 60 %.
        Gating the whole pass buys all of it -- measured 10.09 ms -> 2.5 ms,
        the residual being work that happens BEFORE the gate (the viewport
        plan build, the colormap LUT, the session key, the prefetch) -- but
        only if the other 40 % was not owed. This predicate is what decides
        that, and it decides it by containment rather than by argument.

        Containment is what makes the drift safe rather than unbounded. A
        SUPERSET is harmless -- every consumer asks "is everything in scope
        settled" or "does everything in scope have work", and both are
        conservative when the scope carries extra tiles. A scope MISSING a
        tile the camera shows is the dangerous direction: an actor that
        replans over ``required_tile_numbers`` while the camera sits still
        would leave that tile out, and nothing would come back for it. So this
        refuses the skip the moment the camera reaches past what was
        published, at which point the full pass republishes around the new
        camera.

        Both rings are checked because they come from different publishers at
        different margins: ``visible_tile_numbers`` is the session's own
        ``VISIBLE_MARGIN_TILES`` set and ``required_tile_numbers`` is the
        frame planner's ``ACTIVE_MARGIN_TILES`` one, and containment in one
        does not imply containment in the other.
        """

        plan = viewport_plan.plan
        view_range = viewport_plan.view_range
        published_visible = getattr(session, "visible_tile_numbers", None)
        if not published_visible:
            return False
        for margin, published in (
            (VISIBLE_MARGIN_TILES, published_visible),
            (ACTIVE_MARGIN_TILES, frozenset(session.required_tile_numbers())),
        ):
            for tile in viewport_tiles(plan, view_range=view_range, margin_tiles=margin):
                if int(tile.montage_index) not in published:
                    return False
        return True

    def _relatch_converged_viewport(self, session, viewport_plan, *, key=None) -> None:
        """Record the convergence latch from the state this pass ends in.

        After a FULL pass the key is re-derived, never reused from the top of
        the method. It carries the ledger and page-cache epochs and a pass
        moves both (``retarget_viewport``, ``sync_lifecycle_scope``, the ladder
        and the presentation apply all write records). Latching an entry key
        would latch an epoch that is already stale, so the very next camera
        move would miss and pay a second full pass to latch the same
        convergence -- measured as 2 planner passes over 12 sub-tile pans that
        should have cost none. What convergence means is "nothing to do from
        HERE", so it has to be read after everything the pass does.

        After a SUPPRESSED pass the caller hands its entry key back instead,
        and that is not a shortcut with a different meaning: the pass proved
        that key describes a convergence, so re-deriving it can only reproduce
        it or -- if the prefetch the caller scheduled first moved the ledger --
        produce one whose epochs no longer describe anything that was proved.
        Keeping the proven one is the conservative half of that choice, and it
        saves the second 0.5 ms derivation (at 272 tiles) on the path whose
        entire point is to be cheap.

        The session epoch is stored BESIDE the key rather than inside it, and
        on the renderer rather than on the session, because the assignment
        below is itself a session write: a self-carried reading would name the
        instant before it was stored and be one behind for ever. It is paired
        with the session id so an index-window retarget (same object, new id)
        cannot inherit it; the key itself dies with the session, so a stale
        pairing alone can never suppress anything.
        """

        session._viewport_converged_key = (
            session.viewport_invariant_key(
                viewport_plan.plan, viewport_plan.view_range, viewport_plan.viewport_shape
            )
            if key is None
            else key
        )
        self._montage_viewport_converged_revision = (
            int(getattr(session, "session_id", 0) or 0),
            int(session.mutation_revision),
        )

    # ------------------------------------------------------------------
    # The planner's own convergence gate.
    #
    # ``_try_update_montage_viewport_only`` above gates the VIEWPORT caller.
    # Attributed by caller over a 272-tile wgpu workflow (fill + scroll +
    # zoom/pan, 558 recorded ``retarget_frame_pipeline`` calls, 92.5 % of them
    # ``submitted=0`` costing 3108 ms), that caller is a MINORITY of the
    # stream:
    #
    #     caller                                          calls zero  ms
    #     request_montage_replan's fire()                   207 183 1406
    #     _maybe_retarget_frame_session                      83  83  457
    #     _try_update_montage_viewport_only, deferred fan-in  33  32  401
    #     frame_effects.retarget_on_gui                      95  92  387
    #     _try_update_montage_viewport_only, main pass       90  79  318
    #     replan_deferred_interactive_native_quality          7   6   85
    #     _try_update_montage_viewport_only, interactive     37  37   38
    #     _retarget_montage_resize_payloads                   6   4   15
    #
    # 70 % of the calls and 76 % of the wasted milliseconds enter through a
    # door the viewport latch does not stand in, which is why
    # ``viewport_retargets_suppressed`` reads 48 / 341 in the field traces
    # while 93 % of all retargets submit nothing.  So the containment is
    # lifted to the planner, where every caller passes.
    #
    # Timed inside the body (561 calls, 3270 ms, 2625 ms of it in passes that
    # submitted nothing), gating here rather than at ``pipeline.retarget``
    # buys the whole pass rather than 59 % of it:
    #
    #     pipeline.retarget            1933 ms   <- effects.tile_states 1246
    #     mark_ladder_swaps             908 ms
    #     everything else               429 ms
    #
    # ``mark_ladder_swaps_for_current_demand`` alone is 28 % and runs BEFORE
    # the planner, so a gate inside the pipeline cannot reach it.
    # ------------------------------------------------------------------

    def _montage_planner_outstanding_is_camera_local(self, session) -> bool:
        """Whether nothing is outstanding for a slot the camera cannot see.

        Only the CAMERA arm needs this, and reading the refusal counts is what
        showed why. Over 575 planner passes on a 272-tile wgpu workflow the
        two arms refuse 419 (epoch moved) and 409 (a camera tile is genuinely
        pending) times, so the arms — not the tail obligations — are what the
        gate turns on. An earlier draft vetoed on the whole obligation list and
        suppressed 48 passes; the list is the wrong instrument for most of it
        (``flush_pending`` alone was the first refusal 198 times, and the tail
        DISCHARGES that rather than needing to plan for it).

        What is left here is the one thing the tile statement cannot cover.
        Every set below is already in :meth:`FrameSession.camera_pending_tiles`'
        ``outstanding`` union — but only for tiles the camera reaches. So a
        non-empty set either intersects the camera, in which case the camera
        arm has already refused and this clause costs nothing, or it lies
        entirely OUTSIDE the retention halo, which is the one shape the arm
        cannot see: ``_lod_admission_scope`` folds ``loading_tiles`` and
        ``active_tile_requests`` into coverage wherever they are. Refusing
        there is therefore free in hit rate and removes the whole class.

        Biased to VETO on an unreadable state: every default is the "something
        is outstanding" value. A tile that never appears is far worse than a
        replan that finds nothing.
        """

        return not (
            getattr(session, "dirty_payloads", True)
            or getattr(session, "pending_payload_upserts", True)
            or getattr(session, "pending_removals", True)
            or getattr(session, "loading_tiles", True)
            or getattr(session, "active_tile_requests", True)
            or getattr(session, "skipped_tiles", True)
            or getattr(session, "deferred_out_of_view_invalidations", True)
        )

    def _montage_planner_guard_terms(self, session) -> tuple | None:
        """The planner's inputs that the session mutation epoch does not carry.

        ``interactive`` is the load-bearing one and it is not session state at
        all: it lives on the render coordinator, and the ladder defers native
        rungs on it (``_defer_native_quality_during_interaction``).  A gesture
        ending changes what an otherwise identical plan admits, which is
        exactly what ``replan_deferred_interactive_native_quality`` exists to
        pick up -- so it has to break the latch, and it does, here.

        The round terms are session state, and therefore covered by the epoch
        arm.  They are named anyway because the CAMERA arm deliberately
        survives a moved epoch: a round that reopens at a different preview or
        target floor plans differently over the same settled tiles.
        """

        shedding = getattr(getattr(session, "round_load_shedding", None), "verdict", None)
        if shedding is None:
            return None
        return (
            bool(_interactive_active(self)),
            getattr(session, "key", None),
            str(getattr(session, "render_round_id", "") or ""),
            int(getattr(session, "round_preview_level", 0) or 0),
            int(getattr(session, "round_target_level", 0) or 0),
            bool(getattr(shedding, "drops_target_pass", True)),
        )

    def montage_planner_pass_is_idle(self, session, *, force_commit: bool):
        """The live invariant key when this planner pass provably decides nothing.

        ``None`` on every ambiguity, which is the whole contract: an
        unreadable epoch, a missing latch, a backend that drops out-of-view
        pixels, an obligation still owed -- all of them mean PLAN.  The two
        arms are the same two the viewport gate rests on, and for the same
        reasons; what differs is that the publication half
        (``_published_scope_covers_camera``) is not needed here, because the
        planner is not the publisher.  ``retarget_viewport`` is, and it runs
        in the caller above this gate, not inside it.

        **Epoch arm** -- argument from determinism.  Same session, same
        mutation epoch, same key, same non-session inputs: the pass has
        literally the same inputs as one that already returned zero, so it
        returns zero.  This is the arm that must hold when a caller replans
        speculatively.

        **Camera arm** -- argument from the present, for the case the epoch
        cannot serve.  The replan gate fires BECAUSE a completion wrote to the
        session, so its epoch has always moved by the time it runs and the
        first arm can never hold for it; that caller is 45 % of the wasted
        milliseconds.  What survives a commit is the tile statement: every
        tile the camera reaches is drawn at the demanded level with nothing
        outstanding, nothing is outstanding anywhere else either
        (:meth:`_montage_planner_obligations_clear`), and the camera terms the
        tiles cannot answer for -- chiefly ``preview_floor`` -- are unchanged.

        The full key equality in the epoch arm is not redundant with the
        epoch.  ``lifecycle_revision`` and ``page_cache_revision`` live in the
        key and neither is session state, so the session epoch alone would let
        a ledger write through.
        """

        if force_commit:
            # A forced commit is a caller saying "present regardless"; it has
            # never been a question about the plan.
            return None
        if not montage_commit.viewport_convergence_latch_backend(self, session):
            return None
        if getattr(session, "atomic_successor_pending", True):
            # A live successor handoff makes ``tile_states`` answer differently
            # (``allow_preview`` is forced off while compatible predecessor
            # pixels are held), and that is not a function of the camera.
            return None
        pending_admissions = getattr(
            getattr(session, "pipeline", None), "has_pending_admissions", None
        )
        if pending_admissions is None or pending_admissions():
            # A pipeline that has not been built yet reads ``None`` and is
            # refused, which is right: there is no evidence either way, and a
            # session with no planner has never planned.
            return None
        latched = getattr(self, "_montage_planner_converged", None)
        if latched is None:
            return None
        latched_session_id, latched_revision, latched_key, latched_guard = latched
        if int(latched_session_id) != int(getattr(session, "session_id", -1) or -1):
            return None
        guard = self._montage_planner_guard_terms(session)
        if guard is None or guard != latched_guard:
            return None
        entry_revision = getattr(session, "mutation_revision", None)
        invariant_key = session.viewport_invariant_key(
            getattr(session, "plan", None),
            getattr(session, "view_range", None),
            getattr(session, "viewport_shape", None),
        )
        if invariant_key is None:
            return None
        epoch_held = entry_revision is not None and int(entry_revision) == int(latched_revision)
        if epoch_held and latched_key == invariant_key:
            return invariant_key
        if (
            not invariant_key.camera_pending_tiles
            and latched_key.camera_terms == invariant_key.camera_terms
            and self._montage_planner_outstanding_is_camera_local(session)
        ):
            return invariant_key
        return None

    def relatch_converged_planner(self, session) -> None:
        """Record the planner latch from the state the pass ENDS in.

        Read after everything the pass did, for the reason
        :meth:`_relatch_converged_viewport` documents: the epoch moves on every
        session write and the pass is a writer, so an entry reading would name
        a state the pass itself has already left and could never match again.

        Unlike the viewport latch this never takes a key from its caller, not
        even on the suppressed path. That path is not read-only here -- it
        discharges the flush, the stage re-arm and the completion edge -- so
        the key it entered with describes a state it has already left.
        """

        if not montage_commit.viewport_convergence_latch_backend(self, session):
            # Nothing will ever read it, and deriving the key is not free: it
            # walks the retention halo. Also the reason this asks the backend
            # rather than trusting the caller -- a latch derived on a surface
            # that drops out-of-view pixels is a key nobody can use.
            self._montage_planner_converged = None
            return
        guard = self._montage_planner_guard_terms(session)
        if guard is None:
            self._montage_planner_converged = None
            return
        key = session.viewport_invariant_key(
            getattr(session, "plan", None),
            getattr(session, "view_range", None),
            getattr(session, "viewport_shape", None),
        )
        revision = getattr(session, "mutation_revision", None)
        if key is None or revision is None:
            self._montage_planner_converged = None
            return
        self._montage_planner_converged = (
            int(getattr(session, "session_id", 0) or 0),
            int(revision),
            key,
            guard,
        )

    @session_order_lock_section
    def _try_update_montage_viewport_only(self) -> bool:
        """Retarget a persistent tiled session without restarting evaluation."""

        session = getattr(self, "_frame_session", None)
        if not self._frame_session_is_current(session):
            return False
        # BEFORE this method writes anything. The epoch moves on every session
        # write and this method is a writer, so a reading taken later in the
        # body would be compared against a latch recorded before those writes
        # and could never match (``dde88c56``: 553 bumps for a settled
        # retarget, 1399 during a fill). Both places that record the latch
        # below re-read it at the very END of the pass, so what this compares
        # is exactly "did anything write to the session since I last looked".
        entry_revision = getattr(session, "mutation_revision", None)
        capabilities = image_view_backend_capabilities(self.win.img_view)
        try:
            display_mode = str(self.win.img_view.montageDisplayMode())
        except Exception:
            display_mode = ""
        retarget_policy = montage_viewport_retarget_policy(capabilities, display_mode)
        if not retarget_policy.enabled:
            return False

        view_state = self.win.view_state
        if view_state.montage_axis is None:
            return False
        viewport_plan = self._montage_viewport_plan(view_state)
        colormap_lut = self._evaluation_colormap_lut(
            view_state,
            shader_display=bool(capabilities.shader_windowing),
        )
        expected_key = frame_session_key(
            _document_key(self.win.document),
            view_state,
            viewport_plan,
            colormap_lut,
        )
        # Not the shared currency predicate: this compares the session against
        # the key re-derived from the *live* view state (semantic-match check).
        if session.key != expected_key:
            self._montage_live_layout_reflow = False
            return False
        self._montage_live_layout_reflow = False
        previous_plan = getattr(session, "plan", None)
        viewport_plan = self._retargeted_montage_viewport_plan(session, viewport_plan)
        self._remap_montage_rois_for_layout_reflow(previous_plan, viewport_plan.plan)

        # Threshold gate (persistent-residency backends). A pan or zoom
        # only needs to replan when it crosses a tile boundary or a zoom
        # threshold; between crossings the camera moves without the engine
        # restating anything per tile -- on WGPU through the camera uniform
        # (tile instances hold world rects since e9956846), on PyQtGraph
        # through the ViewBox transform over retained ImageItem scene
        # positions. ``viewport_invariant_key`` is that boundary, evaluated in
        # O(1); ``_viewport_converged_key`` records the key at which the last
        # full pass proved convergence by submitting zero work. When they match,
        # the retarget is provably a no-op, so skip it whole -- retarget_viewport,
        # frame plan, lifecycle sync, ladder and all. Any tile-boundary or zoom
        # crossing changes the key and takes the full path, which re-latches.
        persistent_residency = montage_commit.persistent_gpu_tile_residency_backend(self, session)
        invariant_key = (
            session.viewport_invariant_key(
                viewport_plan.plan, viewport_plan.view_range, viewport_plan.viewport_shape
            )
            if montage_commit.viewport_convergence_latch_backend(self, session)
            else None
        )
        # The epoch half of the latch. ``None`` on either side fails OPEN:
        # an unreadable epoch is a reason to plan, never a reason to assume
        # nothing changed.
        converged_revision = getattr(self, "_montage_viewport_converged_revision", None)
        epoch_held = (
            entry_revision is not None
            and converged_revision is not None
            and converged_revision
            == (int(getattr(session, "session_id", 0) or 0), int(entry_revision))
        )
        # The SECOND arm, and the one that makes a pan over a settled montage
        # free. The first is an argument from HISTORY: a full pass proved this
        # camera converged and no session write has happened since. That
        # argument cannot survive its own consequences -- a pass that crosses a
        # tile boundary applies a presentation, the commit that follows
        # acknowledges it, and acknowledging writes tile records, so the latch
        # is always recorded in a state the commit is about to leave and the
        # next gesture misses. One full 272-tile replan per gesture for ever,
        # which is what ``viewport_retargets_suppressed`` 18 / 451 measures.
        #
        # This one is an argument from the PRESENT for the half the epoch
        # cannot carry: every tile this camera reaches, out to the widest ring
        # any consumer of the camera uses, is already on the screen at the
        # demanded level with nothing outstanding that names it, so the pass
        # would find no addition, no ladder swap, no admission and no stale
        # level whatever happened in between. What it still takes from history
        # is only what the tiles cannot answer -- the CAMERA terms, chiefly
        # ``preview_floor``, which steps where ``desired_level`` does not.
        # None of those move when a commit lands, so this arm survives one.
        #
        # It is still gated on an observed convergence (``latched is not
        # None``), and on ``_published_scope_covers_camera`` -- the pending set
        # proves the PLANNER would do nothing, not that the scope this pass
        # would have published is still adequate. See there.
        #
        # The layout guard is the one thing neither argument covers:
        # ``retarget_viewport`` treats a geometry change as a presentation
        # change regardless of tile state, so a reflow takes the full path even
        # over a montage with nothing to do.
        latched = session._viewport_converged_key
        layout_unchanged = getattr(viewport_plan.plan, "geometry", None) == getattr(
            getattr(session, "plan", None), "geometry", object()
        )
        camera_has_no_work = (
            invariant_key is not None
            and latched is not None
            and not invariant_key.camera_pending_tiles
            and layout_unchanged
            and latched.camera_terms == invariant_key.camera_terms
            and self._published_scope_covers_camera(session, viewport_plan)
        )
        if invariant_key is not None and (
            camera_has_no_work or (epoch_held and latched == invariant_key)
        ):
            # A suppressed retarget must be visible in a stall dump, or a
            # montage that goes stale under a gesture has no first question.
            # The counter is the field instrument (``MontageRuntimeDiagnostics``);
            # the trace line names the key that did the suppressing.
            self._montage_viewport_retargets_suppressed = (
                int(getattr(self, "_montage_viewport_retargets_suppressed", 0) or 0) + 1
            )
            emit_trace(
                "viewport_retarget_suppressed",
                session_id=int(getattr(session, "session_id", 0) or 0),
                camera_pending_tiles=len(invariant_key.camera_pending_tiles),
                desired_level=int(invariant_key.desired_level),
                desired_factor=int(invariant_key.desired_factor),
                preview_floor=int(invariant_key.preview_floor),
                lifecycle_revision=int(invariant_key.lifecycle_revision),
                page_cache_revision=int(invariant_key.page_cache_revision),
                session_revision=int(entry_revision or 0),
                arm="camera" if camera_has_no_work else "epoch",
            )
            # Prefetch is interaction-gated: it no-ops mid-drag, runs at settle.
            schedule_near_viewport_montage_prefetch(self, session)
            # Re-latched with THIS camera's key and an epoch read after the
            # prefetch above. Without it the latch would keep naming the camera
            # and the epoch of whichever pass last converged, so the very next
            # move would compare against a reading this call has already
            # invalidated and pay a full pass to prove the same nothing.
            # Absorbing this call's own writes is sound because it decided
            # nothing and scheduled only speculative prefetch, whose
            # completions come back through ``request_montage_replan``.
            self._relatch_converged_viewport(session, viewport_plan, key=invariant_key)
            return True
        self._montage_viewport_retargets_planned = (
            int(getattr(self, "_montage_viewport_retargets_planned", 0) or 0) + 1
        )

        # Hidden residency is keyed to the viewport that selected its near
        additions, presentation_changed = session.retarget_viewport(
            view_range=viewport_plan.view_range,
            viewport_shape=viewport_plan.viewport_shape,
            plan=viewport_plan.plan,
            coverage_margin_tiles=retarget_policy.coverage_margin_tiles,
            near_margin_tiles=retarget_policy.near_margin_tiles,
            priority_focus=viewport_plan.priority_focus,
            settled_payloads_are_known=persistent_residency,
        )
        # The key just changed (we did not take the early-out), so any prior
        # convergence latch is stale; it is re-earned below only if this full
        # pass submits nothing. Both halves, or the epoch pairing would outlive
        # the key it belongs to.
        session._viewport_converged_key = None
        self._montage_viewport_converged_revision = None

        memory_policy = self._memory_policy() if hasattr(self, "_memory_policy") else None
        session.frame_plan = self._montage_frame_planner().plan(
            target=FrameTarget(
                semantic_key=session.key,
                viewport_key=viewport_plan.view_range,
                presentation_key=(
                    str(session.window_mode),
                    normalize_bounds(getattr(session, "user_levels_override", None)),
                    bool(getattr(session, "force_auto", False)),
                ),
                quality="exact-visible",
            ),
            view_state=view_state,
            display_shape=viewport_plan.plan.display_shape,
            backend_capabilities=capabilities,
            viewport_shape=viewport_plan.viewport_shape,
            view_range=viewport_plan.view_range,
            memory_policy=memory_policy,
            montage_plan=viewport_plan.plan,
        )
        session.sync_lifecycle_scope()
        if memory_policy is not None:
            session.tile_residency_budget_bytes = tile_residency_budget_bytes(memory_policy)
        # Camera-only changes must retarget the LOD decision immediately:
        # black/too-coarse tiles improve immediately, while already-presented
        # finer tiles stay put unless visible residency pressure requires a
        # quality demotion. Demand math only; reduction stays on worker lanes.
        lod_swap_ready = session.mark_ladder_swaps_for_viewport(refresh_demand=False)
        submitted = self.retarget_frame_pipeline(
            session,
            prepared_lod_swap_ready=lod_swap_ready,
        )
        required_tile_numbers = frozenset(session.required_tile_numbers())
        # The frame pipeline derives visible work from the lifecycle's
        # required scope. Keep the viewport-only cache/evaluation path on that
        # same scope.
        # Coverage-margin misses are speculative and are owned by
        # ``schedule_near_viewport_montage_prefetch`` once visible work is
        # quiet; admitting them here leaves dormant speculative debt with no
        # pipeline consumer and later suppresses the cache lookup when the
        # tile actually becomes visible.
        additions = tuple(
            tile
            for tile in viewport_plan.prioritize_tiles(additions)
            if int(tile.montage_index) in required_tile_numbers
        )
        self._prune_stale_montage_tile_work(session)
        if not additions:
            if presentation_changed or lod_swap_ready:
                self.apply_montage_presentation(session)
            self._finish_frame_session_if_complete(session)
            # This full pass submitted no kernel work and admitted no cached
            # additions: convergence is OBSERVED, not predicted. Latch the
            # invariant key so the next pan that stays within the same tile/zoom
            # bounds short-circuits at the top. A lod swap means the scene is
            # still moving, so do not latch then. (Parked work drains via
            # scheduler callbacks, never this path, so zero submissions here
            # genuinely means nothing left to do.)
            #
            # ``presentation_changed`` used to veto the latch too, and that
            # veto is what pinned the field ratio at 0 suppressed / 112 planned
            # in trace 6's 4K phase: it is set by ``retarget_viewport`` for ANY
            # change of the visible tile numbers, and the operator's visible
            # set swings 30 <-> 272, so every pan across a tile boundary
            # refused to latch and the next one paid for it again. It is gone
            # because ``apply_montage_presentation`` has already run for the
            # new visible set two lines above, and because what the resulting
            # commit goes on to change retires the latch when it LANDS -- it
            # moves the ledger epoch and it makes tiles pending -- rather than
            # having to be predicted from here.
            schedule_near_viewport_montage_prefetch(self, session)
            if invariant_key is not None and int(submitted or 0) == 0 and not lod_swap_ready:
                self._relatch_converged_viewport(session, viewport_plan)
            return True
        additions_to_process = tuple(additions)
        budget = self._montage_callback_budget(
            "montage_viewport_update",
            interactive=_interactive_active(self),
            work_class="viewport_cached_additions",
            item_cap=max(1, len(additions_to_process)),
        )
        cached_tiles = []
        missing_tiles = []
        processed_additions = 0
        # One lookup-state snapshot for the whole budgeted loop: nothing the
        # snapshot captures changes between iterations (commits happen after
        # this loop), and rebuilding it per tile made the loop quadratic in
        # the retained-store size.
        #
        # Built only when there is an addition to spend it on. The per-tile
        # version was reached exclusively from inside the loop body, so an
        # empty additions tuple used to touch none of what the snapshot reads
        # -- hoisting it unguarded made a no-op pass newly demand
        # `win.operation_evaluator` and the retained store.
        resolution_context = (
            self._montage_cache_resolution_context(
                document=session.document,
                colormap_lut=session.colormap_lut,
                shader_display=bool(getattr(session, "shader_display", False)),
            )
            if additions_to_process
            else None
        )
        for tile in additions_to_process:
            display_cached, display_missing = self._resolve_montage_tiles_from_cache(
                (tile,),
                document=session.document,
                axis=session.montage_axis,
                colormap_lut=session.colormap_lut,
                shader_display=bool(getattr(session, "shader_display", False)),
                context=resolution_context,
            )
            cached_tiles.extend(display_cached)
            missing_tiles.extend(display_missing)
            processed_additions += 1
            byte_count = sum(
                montage_commit.rendered_tile_nbytes(rendered) for rendered in display_cached
            )
            budget.record_item(byte_count=byte_count)
            if budget.should_yield():
                break
        self._record_gui_budget(budget)
        remaining_additions = max(0, len(additions_to_process) - int(processed_additions))
        self._last_montage_viewport_deferred_additions = int(remaining_additions)
        native_target_additions = render_lod.missing_tiles_require_native_target(
            str(getattr(session, "lod_policy_mode", "")),
            getattr(getattr(session, "lod_policy_decision", None), "demand", None),
        )
        if remaining_additions and native_target_additions:
            self.win._montage_viewport_update_pending = True
            self._montage_viewport_continue_immediately = True
        session.tile_compute_cache_hits += len(cached_tiles)
        for rendered in cached_tiles:
            session.mark_materialized(rendered)
        self._queue_montage_cached_level_stats(session, cached_tiles, seed_if_empty=False)

        self._montage_cached_tiles_last_session = len(cached_tiles)
        self._montage_missing_tiles_last_session = len(missing_tiles)
        if presentation_changed or cached_tiles or lod_swap_ready:
            self.apply_montage_presentation(session)
        if cached_tiles:
            self._schedule_montage_cached_level_stats(session)
        if not missing_tiles:
            self._finish_frame_session_if_complete(session)
            schedule_near_viewport_montage_prefetch(self, session)
            return True
        missing_tiles = viewport_plan.prioritize_tiles(missing_tiles)
        if not native_target_additions:
            self.win._montage_viewport_update_pending = False
            self._montage_viewport_continue_immediately = False
            session.stage_planning_deferred = bool(missing_tiles)
            session.stage_planning_async = False
            session.deferred_missing_tiles = tuple(missing_tiles)
            montage_commit.submit_deferred_stage_fan_in_plan(self, session, missing_tiles)
            self.retarget_frame_pipeline(session)
            return True

        if _viewport_interaction_active(self):
            stage_plan = montage_commit.hot_cached_stage_fan_in_plan(
                self, session.document, missing_tiles
            )
            if montage_commit.stage_fan_in_plan_has_existing_sources(stage_plan):
                montage_commit.merge_stage_fan_in_plan(session, stage_plan)
            session.stage_planning_deferred = True
            session.stage_planning_async = False
            session.deferred_missing_tiles = tuple(missing_tiles)
            montage_commit.submit_deferred_stage_fan_in_plan(self, session, missing_tiles)
            self.retarget_frame_pipeline(session)
            return True

        stage_plan = montage_commit.build_stage_fan_in_plan(self, session.document, missing_tiles)
        montage_commit.merge_stage_fan_in_plan(session, stage_plan)

        self.win.prefetch_evaluation_controller.cancel_prefetch()
        self.win.operation_evaluator.last_status = CacheStatusSnapshot(
            CacheStatus.COMPUTING,
            "Extending montage viewport",
        )
        self.show_frame_session_slow_overlay(session)
        montage_commit.submit_stage_tasks(self, session, stage_plan["stage_requests"])
        self.retarget_frame_pipeline(session)
        return True

    @session_order_lock_section
    def _prune_stale_montage_tile_work(self, session: FrameSession) -> None:
        if not _viewport_interaction_active(self):
            return
        # Through ``viewport_tiles``, like every other halo consumer. This one
        # got the constant's name without its derivation, which leaves it
        # naming a *narrower* set than the halo it is spelled after: it cancels
        # in-flight work mid-drag, so a raw-rect ``keep`` un-loads tiles the
        # session is protecting and the drag then re-requests them.
        keep = {
            int(tile.montage_index)
            for tile in viewport_tiles(
                session.plan,
                view_range=session.view_range,
                margin_tiles=RETENTION_HALO_TILES,
            )
        }
        if not keep:
            return
        stale = (set(session.loading_tiles) - set(session.active_tile_requests)) - keep
        if stale:
            for index in sorted(stale):
                session.loading_tiles.discard(int(index))
                if 0 <= int(index) < len(session.tile_states):
                    session.tile_states[int(index)] = MontageTileState.UNLOADED
                    session.invalidate_tile_states()
        pruned = len(stale)
        if pruned > 0:
            self._last_montage_pruned_tile_work = int(
                getattr(self, "_last_montage_pruned_tile_work", 0) or 0
            ) + int(pruned)

    def _warn_montage_tiles_skipped(
        self, *, skipped_count: int, tile_bytes: int, budget_bytes: int, tile_shape
    ) -> None:
        message = (
            f"Montage skipped {int(skipped_count)} tile(s) because each tile would allocate "
            f"{format_bytes(int(tile_bytes))}, over the visible render budget of {format_bytes(int(budget_bytes))}. "
            f"Tile shape is {tuple(int(size) for size in tile_shape)}. Zoom in, crop/range the image axes, "
            "or increase Performance > Render Memory Budget."
        )
        show_status_message(self.win, message, timeout=8000)
        warning_key = (
            int(skipped_count),
            int(tile_bytes),
            int(budget_bytes),
            tuple(int(size) for size in tile_shape),
        )
        if getattr(self, "_last_montage_skip_warning_key", None) == warning_key:
            return
        self._last_montage_skip_warning_key = warning_key
        try:
            Qt.QtWidgets.QMessageBox.warning(self.win, "Montage tiles skipped", message)
        except Exception as exc:
            handle_ui_exception("montage skipped warning", exc)

    def _retargeted_montage_viewport_plan(
        self,
        session,
        viewport_plan: MontageViewportPlan,
        *,
        previous_viewport_shape=None,
        focus=None,
    ) -> MontageViewportPlan:
        continuity = getattr(self.win, "_viewport_continuity_transaction", lambda: None)()
        skip_remap = bool(
            continuity is not None
            and not continuity.released
            and continuity.range_applied
            and continuity.view_range is not None
        )

        viewport_controller = getattr(self.win.img_view, "viewport_controller", None)
        current_range = viewport_plan.view_range
        intent = montage_viewport_intent(viewport_controller, current_range)
        camera_focus = (
            focus
            if focus is not None
            else (None if current_range is None else _montage_priority_focus(self, current_range))
        )
        reflow = retarget_montage_viewport_plan(
            getattr(session, "plan", None),
            viewport_plan,
            previous_viewport_shape
            or getattr(session, "viewport_shape", viewport_plan.viewport_shape),
            fit_locked=intent.fit_locked,
            auto_active=intent.auto_active,
            skip_remap=skip_remap,
            focus=camera_focus,
        )
        if viewport_controller is not None and reflow.last_auto_view_range is not None:
            viewport_controller.last_auto_view_range = reflow.last_auto_view_range
        if reflow.view_range_to_apply is not None:
            self._set_montage_view_range(reflow.view_range_to_apply)
        if skip_remap:
            complete_continuity = getattr(
                self.win, "_complete_viewport_continuity_if_settled", None
            )
            if callable(complete_continuity):
                complete_continuity()
        view_range = reflow.viewport_plan.view_range
        priority_focus = (
            focus
            if focus is not None
            else (None if view_range is None else _montage_priority_focus(self, view_range))
        )
        return replace(
            reflow.viewport_plan,
            priority_focus=priority_focus,
        )

    def _remap_montage_rois_for_layout_reflow(self, previous_plan, next_plan) -> None:
        if previous_plan is None or next_plan is None:
            return
        if getattr(previous_plan, "geometry", None) == getattr(next_plan, "geometry", None):
            return
        img_view = getattr(self.win, "img_view", None)
        selections_fn = getattr(img_view, "roiSelections", None)
        if not callable(selections_fn):
            return
        selections = tuple(selections_fn() or ())
        remapped_selections = remap_montage_roi_selections(previous_plan, next_plan, selections)
        updates = [
            (previous, current.geometry)
            for previous, current in zip(selections, remapped_selections, strict=False)
            if current.geometry != previous.geometry
        ]
        if not updates:
            return
        set_geometry = getattr(img_view, "_set_roi_geometry", None)
        if callable(set_geometry):
            for selection, geometry in updates:
                set_geometry(selection.id, geometry, emit=True, sync_item=True)
        else:
            selected_id = getattr(getattr(self.win, "roi_store", None), "selected_id", None)
            setter = getattr(img_view, "setRoiSelections", None)
            if callable(setter):
                setter(remapped_selections, selected_id=selected_id)
        roi_store = getattr(self.win, "roi_store", None)
        if roi_store is not None:
            selected_id = getattr(roi_store, "selected_id", None)
            self.win.roi_store = roi_store.replace_all(selections_fn()).select(selected_id)
            dock = getattr(self.win, "inspection_dock", None)
            set_rois = getattr(dock, "set_rois", None)
            if callable(set_rois):
                set_rois(self.win.roi_store.selections)
        schedule_refresh = getattr(self.win, "_schedule_refresh_inspection_dock", None)
        if callable(schedule_refresh):
            schedule_refresh("montage-layout-reflow")

    # Also reached from outside this module (the level-update presenter), so
    # the span lives on the funnel rather than on that caller.
    @session_order_lock_section
    def apply_montage_presentation(self, session) -> None:
        if not self._frame_session_is_current(session):
            return
        pipeline = getattr(session, "pipeline", None)
        if pipeline is None:
            # Commit-only test/teardown sessions do not own a live pipeline.
            self.commit_frame_session_presentation(session)
        else:
            pipeline.effects.request_presentation()
        if (
            getattr(session, "show_loading_overlays", False)
            and not session.visible_plan_complete()
            and not session.is_complete()
        ):
            self.win.img_view.setImageStale(True)
            self.win.img_view.setEvaluationOverlay(True, "Updating image frame...")

    @session_order_lock_section
    def commit_frame_session_presentation(self, session) -> None:
        if not self._frame_session_is_current(session):
            return
        # One effects instance per session: it owns the in-flight rung guards
        # and the presentation gate flag. A throwaway instance would fork
        # that state (exactly the parallel-bookkeeping defect ADR 0051 bans).
        # Commit-only callers without a live pipeline (tests, teardown) get a
        # transient instance — they never submit rungs, so no guard state is
        # forked.
        pipeline = getattr(session, "pipeline", None)
        effects = pipeline.effects if pipeline is not None else FramePipelineEffects(self, session)
        effects.commit_pending_session()

    @session_order_lock_section
    def apply_ready_montage_display(self, session) -> None:
        if not self._frame_session_is_current(session):
            return
        has_commit_work = (
            getattr(session, "dirty_rects", None)
            or getattr(session, "dirty_tiles", None)
            or getattr(session, "dirty_payloads", None)
            or getattr(session, "pending_payload_upserts", None)
            or getattr(session, "pending_removals", None)
            or getattr(session, "backend_refresh_pending", False)
            or (session.has_pending_level_update() and session.has_stale_level_presentations())
        )
        if not has_commit_work:
            return
        session.final_commit_pending = True
        session.flush_pending = True
        self.apply_montage_presentation(session)

    def _finish_frame_session_if_complete(self, session) -> bool:
        if not self._frame_session_is_current(session):
            return False
        if not session.is_complete():
            return False
        self._settle_montage_visible_plan_if_complete(session)
        if not session.scheduling_policy.verdict.refinement_admissible:
            return True
        # Montage level phasing: rough -> hold -> refined. Refined evidence is
        # queued after visible settlement and admitted through the kernel.
        if bool(getattr(session, "shader_display", False)):
            # Payload evidence has already closed the rough pass. The
            # statistics-only semantic owner now refines the full population;
            # target display payloads must not enter a second rough/refinement
            # owner.
            self._schedule_semantic_level_evidence(session)
        else:
            self._queue_montage_current_level_evidence(session)
            self._queue_montage_final_level_refinements(session)
            self._schedule_montage_refined_level_stats(session)
        return True

    def _settle_montage_visible_plan_if_complete(self, session) -> bool:
        if not self._frame_session_is_current(session):
            return False
        if not session.visible_plan_complete():
            return False
        session.show_loading_overlays = False
        self._stop_frame_session_slow_overlay()
        self.win.operation_evaluator.last_status = CacheStatusSnapshot(
            CacheStatus.READY, "Image frame ready"
        )
        self.win.img_view.setImageStale(False)
        self.win.img_view.setEvaluationOverlay(False)
        if hasattr(self.win.img_view, "clearMontageTileOverlays"):
            self.win.img_view.clearMontageTileOverlays()
        # "In view complete" is exactly the edge the deferred out-of-view
        # invalidation is waiting for, so this is where it is invited.  The
        # prefetch entry invites it too; the drain owns its own gates so the
        # ordering property does not depend on which owner arrives first, and
        # a montage that stops committing (nothing left to converge) still
        # drains instead of holding its queue until the next user action.
        drain_deferred_out_of_view_invalidation(self, session)
        # The parked out-of-view quality upgrades share that edge and those
        # gates; removals go first because a removal is owed to correctness
        # and an upgrade is not.
        drain_out_of_view_upgrade(self, session)
        # NOT invited here: the halo warm.  This is the moment the product rule
        # names for it — visible work done, session idle — and prefetch has no
        # invitation of its own at that moment (every other one hangs off a
        # commit, so a montage that finished converging after a data change
        # gets none; measured on a settled 272-tile montage after a slice step,
        # the 24-tile halo ring stayed 0/24 display-cached through a second of
        # idle with zero prefetch tiles scheduled).  Adding the invitation was
        # measured on this branch and reverted: it costs
        # ``test_wgpu_pan_away_and_back_reuses_the_retained_presentation_scope``
        # one whole-montage republication per pan leg, 0 -> 1.  Bisected to
        # ``_warm_prefetched_tiled_residency`` — with the invitation kept and
        # only that warm disabled, the republication goes away.  So the
        # out-of-view retention did NOT unblock the halo warm that
        # retention-halo-2026-07-30.md §3 rejected: the warm deposits into the
        # commit-scoped payload map for a tile no commit will present, and the
        # retention predicate cannot rescue it because it requires the backend
        # to be DRAWING the payload already, which a warm never is.
        return True

    def _sync_committed_montage_geometry(self, geometry, *, semantic_commit: bool = True) -> None:
        self.display_geometry = geometry
        frame = getattr(self.win, "_committed_display_frame", None)
        if bool(semantic_commit) and frame is not None and frame.geometry != geometry:
            self._set_committed_display_frame(replace(frame, geometry=geometry, scene=None))
        refresh_hover = getattr(self, "_refresh_hover_after_display_commit", None)
        if callable(refresh_hover):
            refresh_hover()
        if bool(semantic_commit):
            schedule_refresh = getattr(self.win, "_schedule_refresh_inspection_dock", None)
            if callable(schedule_refresh):
                schedule_refresh("montage-semantic-commit")

    def _notify_file_session_montage_committed(self) -> None:
        restore = getattr(self.win, "_schedule_viewport_continuity_when_ready", None)
        if callable(restore):
            restore()
        if not bool(getattr(self.win, "_file_session_roi_refresh_pending", False)):
            return
        schedule_roi_refresh = getattr(self.win, "_schedule_file_session_roi_refresh", None)
        if callable(schedule_roi_refresh):
            schedule_roi_refresh("montage-semantic-commit")

    def _should_publish_montage_level_metadata(self, session, stats: MontageLevelStats) -> bool:
        # Histogram metadata is independent from whether display levels are
        # allowed to move.  Publishing better semantic stats lets absolute mode
        # update the histogram while preserving numeric levels, and lets
        # relative mode remap through WindowLevelController.
        #
        # Reduced-input display montages (opaque/FFT and reduced-LOD scalar
        # fits) present entirely through preview payloads and never populate
        # `rendered_tiles`, so gating on it alone left them stranded at default
        # (0, 1) levels with an empty histogram.  Their rough sampled stats are
        # the correct first-pass source; refinement improves them later.
        if not session.rendered_tiles and not getattr(session, "display_tile_payloads", None):
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
        if int(stats.rank) > applied_rank:
            return True
        if int(getattr(stats, "evidence_quality", 0) or 0) > int(
            getattr(applied, "evidence_quality", 0) or 0
        ):
            return True
        if (  # noqa: SIM103
            bool(getattr(session, "shader_display", False))
            and not bool(getattr(session, "first_pass_histogram_published", False))
            and len(getattr(stats, "source_indices", ()) or ())
            > int(getattr(applied, "source_count", 0) or 0)
        ):
            # First-pass shader levels intentionally track bounded coverage
            # growth. Histogram plotting remains coalesced until the physical
            # pass is complete.
            return True
        # Coverage growth inside one rank is intentionally coalesced. A
        # 60-source fill otherwise rebuilds the histogram plot after every
        # small evidence batch, starving paints/input while showing a sequence
        # of transient partial distributions. Rank/quality transitions still
        # publish immediately (first rough pixels, complete population,
        # refined evidence), so correctness and useful progress stay visible.
        return False

    def _note_montage_level_source_applied(self, session, source, *, explicit: bool) -> None:
        if source is None:
            return
        # Store partial as well as complete semantic sources.  The presentation
        # controller expands same-key sources monotonically and protects explicit
        # user locks, so storing partial coverage is safe and prevents fallback
        # to stale tiny placeholder ranges.
        session.applied_level_source = source

    def _is_current_frame_session(self, session_id, key) -> bool:
        return _session_token_is_current(getattr(self, "_frame_session", None), session_id, key)

    def _frame_session_is_current(self, session) -> bool:
        """Shared staleness predicate for a montage session object.

        Delegates to :meth:`_is_current_frame_session` so tests that stub the
        (session_id, key) predicate keep controlling both forms. Callbacks that
        captured a raw ``(session_id, key)`` pair call the canonical predicate
        directly.
        """
        if session is None:
            return False
        return self._is_current_frame_session(session.session_id, session.key)

    def _montage_tile_shape(self, view_state):
        primary_axis, secondary_axis = view_state.image_axes
        primary_indices = view_state.axis_range_indices[primary_axis]
        secondary_indices = view_state.axis_range_indices[secondary_axis]
        return (
            len(primary_indices)
            if primary_indices is not None
            else int(view_state.shape[primary_axis]),
            len(secondary_indices)
            if secondary_indices is not None
            else int(view_state.shape[secondary_axis]),
        )

    def _current_montage_global_view_range(self):
        try:
            view_range = self.win.img_view.getView().viewRange()
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


def _montage_viewport_shape_from_qt_size_tuple(size):
    if size is None:
        return None
    try:
        width, height = size
        return (max(1, round(float(height))), max(1, round(float(width))))
    except Exception:
        return None


def _montage_full_view_range(montage):
    height = int(montage.rows) * int(montage.tile_height) + max(0, int(montage.rows) - 1) * int(
        montage.gap
    )
    width = int(montage.columns) * int(montage.tile_width) + max(0, int(montage.columns) - 1) * int(
        montage.gap
    )
    return ((0.0, float(max(1, width))), (0.0, float(max(1, height))))


def _initial_montage_planning_view_range(plan, viewport_shape, viewport_controller):
    """Measure startup LOD from layout intent before a backend image exists."""

    if plan is None:
        return None
    intent = montage_viewport_intent(viewport_controller, None)
    if intent.fit_locked:
        return plan_full_view_range(plan)
    return square_montage_fit_view_range(plan, viewport_shape)


def _rendered_tile_from_previous_payload(tile, payload) -> RenderedTile:
    semantic = None if payload.semantic_data is None else np.asarray(payload.semantic_data)
    image = semantic if semantic is not None else np.asarray(payload.image)
    semantic_histogram = (
        None
        if getattr(payload, "semantic_histogram_data", None) is None
        else np.asarray(payload.semantic_histogram_data)
    )
    histogram = (
        semantic_histogram
        if semantic_histogram is not None
        else (None if payload.histogram_data is None else np.asarray(payload.histogram_data))
    )
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
        semantic_histogram_data=semantic_histogram,
        lod_source_data=getattr(payload, "lod_source_data", None),
        lod=getattr(payload, "lod", None),
        level_data=getattr(payload, "level_data", None),
        level_stats=getattr(payload, "level_stats", None),
        quality=str(getattr(payload, "quality", "exact") or "exact"),
    )


def _rendered_tile_from_cached_display(tile, cached) -> RenderedTile:
    if hasattr(cached, "bind"):
        return cached.bind(tile)
    if hasattr(cached, "payload"):
        return cached.payload().bind(tile)
    image = np.asarray(cached.data)
    histogram = (
        None
        if getattr(cached, "histogram_data", None) is None
        else np.asarray(cached.histogram_data)
    )
    return RenderedTile(
        tile=tile,
        image=image,
        histogram_data=histogram,
        eval_ms=0.0,
        slab_shape=tuple(image.shape),
        slab_nbytes=int(image.nbytes),
        shader_mapping=getattr(cached, "shader_mapping", None),
        texture_kind=getattr(cached, "texture_kind", None),
        semantic_data=getattr(cached, "semantic_data", None),
        semantic_histogram_data=getattr(cached, "semantic_histogram_data", None),
        lod_source_data=getattr(cached, "lod_source_data", None),
        lod=getattr(cached, "lod", None),
        level_data=getattr(cached, "level_data", None),
        level_stats=getattr(cached, "level_stats", None),
        quality=str(getattr(cached, "quality", "exact") or "exact"),
    )


def _interactive_active(window) -> bool:
    coordinator = getattr(window.win, "render_coordinator", None)
    return bool(
        (coordinator is not None and getattr(coordinator, "interactive_active", False))
        or _viewport_interaction_active(window)
    )


def _viewport_interaction_active(window) -> bool:
    return bool(getattr(window.win, "_viewport_interaction_active", False))
