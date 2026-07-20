"""Per-window sync participation: publish own changes, apply peer state.

``WindowSyncController`` binds one ``ArrayScopeWindow`` to the shared
``SyncBus``. Participation is per facet (window/level, dimension indexing,
operations, ROIs) and off by default; the sync toggle buttons call
``set_facet_enabled``. The bus starts lazily on the first enabled facet, so
windows that never link pay nothing.

Loop prevention is layered:

- the bus never delivers a window's own messages back to it;
- every state message carries ``(origin, revision)`` and receivers drop
  duplicates they have already applied;
- while a remote payload is being applied the facet is marked in
  ``_applying`` so apply-triggered widget/render callbacks do not republish;
- publishing is leading-edge (a change after a quiet period goes out
  immediately) with a short trailing coalesce timer for bursts, and skipped
  when the built payload equals the last payload sent or applied for that
  facet.
"""

from __future__ import annotations

from time import monotonic
from uuid import uuid4

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph.Qt as Qt

from arrayscope.core.view_session import roi_from_mapping, roi_to_mapping
from arrayscope.core.window_levels import LevelSourceRank, normalize_bounds
from arrayscope.operations.recipes import recipe_from_steps, steps_from_recipe
from arrayscope.sync.bus import SyncBus
from arrayscope.sync.messages import (
    FACET_DIMS,
    FACET_LEVELS,
    FACET_OPERATIONS,
    FACET_ROIS,
    FACETS,
    KIND_REQUEST,
    KIND_STATE,
    dimension_state_payload,
    merged_dimension_state,
    request_message,
    state_message,
)
from arrayscope.ui.toasts import show_status_message

PUBLISH_COALESCE_MS = 120


class WindowSyncController(Qt.QtCore.QObject):
    """Sync one window's facets with the default linked-window group."""

    def __init__(self, window, *, bus: SyncBus | None = None, server_name: str | None = None):
        super().__init__(window)
        self.win = window
        self.window_id = uuid4().hex
        self.bus = bus if bus is not None else SyncBus(server_name=server_name, parent=self)
        self.bus.messageReceived.connect(self._on_message)
        self.bus.peerCountChanged.connect(self._on_peer_count_changed)
        self._enabled = dict.fromkeys(FACETS, False)
        self._revisions = dict.fromkeys(FACETS, 0)
        self._last_applied = {}  # facet -> (origin, revision)
        self._last_payload = {}  # facet -> last payload sent or applied
        self._applying = set()
        self._publish_timers: dict[str, Qt.QtCore.QTimer] = {}
        self._last_publish_monotonic: dict[str, float] = {}
        self._pending_requests = set()
        self._ignore_join_state = set()
        self._pending_peer_publishes = set()
        self._connect_window_signals()

    # ------------------------------------------------------------------
    # Participation

    def facet_enabled(self, facet: str) -> bool:
        return bool(self._enabled.get(facet, False))

    def set_facet_enabled(self, facet: str, enabled: bool) -> None:
        if facet not in FACETS:
            raise ValueError(f"unknown sync facet: {facet!r}")
        enabled = bool(enabled)
        if self._enabled[facet] == enabled:
            return
        self._enabled[facet] = enabled
        if not enabled:
            self._last_payload.pop(facet, None)
            self._pending_peer_publishes.discard(facet)
            if not any(self._enabled.values()):
                self.bus.stop()
            return
        if not self.bus.is_running():
            self.bus.start()
        # Joining pulls the group's current state instead of pushing ours:
        # enabling sync must not stomp what the group is already looking at.
        self._pending_requests.add(facet)
        self.bus.publish(request_message(facet, self.window_id))

    def shutdown(self) -> None:
        for timer in self._publish_timers.values():
            timer.stop()
        self.bus.stop()

    # ------------------------------------------------------------------
    # Publishing

    def schedule_publish(self, facet: str) -> None:
        """Publish this window's state for ``facet``: leading edge + coalesce.

        The first change after a quiet period publishes immediately, so a
        discrete action (one slice step, a level nudge, an ROI drop) reaches
        linked windows without waiting out the coalesce window. Further
        changes inside the window are coalesced through the trailing timer,
        which also guarantees a final publish once a continuous drag pauses.
        The leading edge doubles as the periodic flush during sustained
        drags — a pure trailing debounce would keep re-arming and never
        publish until the drag ended.
        """

        if not self.facet_enabled(facet) or facet in self._applying:
            return
        now = monotonic()
        last = self._last_publish_monotonic.get(facet)
        if last is None or (now - last) * 1000.0 >= PUBLISH_COALESCE_MS:
            timer = self._publish_timers.get(facet)
            if timer is not None:
                timer.stop()
            self._publish_now(facet)
            return
        timer = self._publish_timers.get(facet)
        if timer is None:
            # Timer category: UI cosmetic. Coalesces sync publications so UI
            # toggles do not emit redundant local-bus messages.
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda facet=facet: self._publish_now(facet))
            self._publish_timers[facet] = timer
        timer.start(PUBLISH_COALESCE_MS)

    def _publish_now(self, facet: str, *, force: bool = False) -> bool:
        if not self.facet_enabled(facet) or not self.bus.is_running():
            return False
        try:
            payload = self._build_payload(facet)
        except Exception:
            return False
        if payload is None:
            return False
        if not force and self._last_payload.get(facet) == payload:
            return True
        broker_without_peers = (
            getattr(self.bus, "role", None) == "broker"
            and int(getattr(self.bus, "peer_count", 0) or 0) == 0
        )
        if broker_without_peers:
            self._pending_peer_publishes.add(facet)
        revision = self._revisions[facet] + 1
        sent = self.bus.publish(state_message(facet, self.window_id, revision, payload))
        if not sent and not broker_without_peers:
            return False
        if sent:
            self._pending_peer_publishes.discard(facet)
        self._last_payload[facet] = payload
        self._revisions[facet] = revision
        self._last_publish_monotonic[facet] = monotonic()
        return True

    def _on_peer_count_changed(self, count: int) -> None:
        if int(count) <= 0:
            return
        for facet in tuple(self._pending_peer_publishes):
            self._publish_now(facet, force=True)

    def _build_payload(self, facet: str):
        if facet == FACET_LEVELS:
            levels = normalize_bounds(self.win.img_view.getLevels())
            if levels is None:
                return None
            return {
                "levels": [float(levels[0]), float(levels[1])],
                "window_mode": str(self.win._current_window_mode()),
            }
        if facet == FACET_DIMS:
            return dimension_state_payload(self.win.view_state)
        if facet == FACET_OPERATIONS:
            return {"recipe": recipe_from_steps(self.win.document.steps)}
        if facet == FACET_ROIS:
            store = self.win.roi_store
            return {
                "rois": [roi_to_mapping(selection) for selection in store.selections],
                "selected_roi_id": store.selected_id,
            }
        return None

    # ------------------------------------------------------------------
    # Receiving

    def _on_message(self, message) -> None:
        facet = message.get("facet")
        if not self.facet_enabled(facet):
            return
        origin = message.get("origin")
        if origin == self.window_id:
            return
        if message.get("kind") == KIND_REQUEST:
            # A peer joined this facet; give it our current state even when
            # it matches what we last sent.
            self._publish_now(facet, force=True)
            return
        if message.get("kind") != KIND_STATE:
            return
        if facet in self._ignore_join_state:
            self._ignore_join_state.discard(facet)
            self._pending_requests.discard(facet)
            return
        self._pending_requests.discard(facet)
        key = (origin, int(message.get("revision", 0)))
        if self._last_applied.get(facet) == key:
            return
        payload = message.get("payload") or {}
        self._applying.add(facet)
        try:
            self._apply_payload(facet, payload)
        except Exception as exc:
            show_status_message(self.win, f"Sync update skipped: {exc}", timeout=4000)
            return
        finally:
            self._applying.discard(facet)
        self._last_applied[facet] = key
        # Remember the applied payload so the local change handlers this
        # apply just triggered do not echo it back to the group.
        self._last_payload[facet] = payload

    def _apply_payload(self, facet: str, payload) -> None:
        if facet == FACET_LEVELS:
            self._apply_levels(payload)
        elif facet == FACET_DIMS:
            self._apply_dims(payload)
        elif facet == FACET_OPERATIONS:
            self._apply_operations(payload)
        elif facet == FACET_ROIS:
            self._apply_rois(payload)

    def _apply_levels(self, payload) -> None:
        win = self.win
        mode = payload.get("window_mode")
        if mode in ("relative", "absolute") and mode != win._current_window_mode():
            win._on_window_mode_changed(mode)
        levels = normalize_bounds(payload.get("levels"))
        if levels is None:
            return
        current = normalize_bounds(win.img_view.getLevels())
        if current == levels:
            return
        win.renderer._apply_display_level_override(
            levels,
            emit_user=False,
            source_rank=LevelSourceRank.EXPLICIT_USER,
        )

    def _apply_dims(self, payload) -> None:
        win = self.win
        merged = merged_dimension_state(win.view_state, payload)
        win._apply_synced_dimension_state(merged)

    def _apply_operations(self, payload) -> None:
        win = self.win
        recipe = payload.get("recipe")
        steps = steps_from_recipe(recipe, win.base_data.shape)
        if tuple(steps) == tuple(win.document.steps):
            return
        win.operation_coordinator.load_steps(steps)
        win._set_document(win.operation_coordinator.document)
        win.render(reason="sync-operations", force_autolevel=True)

    def _apply_rois(self, payload) -> None:
        selections = tuple(roi_from_mapping(mapping) for mapping in payload.get("rois", ()))
        self.win._restore_roi_session(selections, selected_id=payload.get("selected_roi_id"))

    # ------------------------------------------------------------------
    # Local change sources

    def _connect_window_signals(self) -> None:
        win = self.win
        img_view = getattr(win, "img_view", None)
        if img_view is not None:
            img_view.userLevelsChanged.connect(lambda: self.schedule_publish(FACET_LEVELS))
            img_view.autoWindowRequested.connect(lambda: self.schedule_publish(FACET_LEVELS))
            img_view.roiCreated.connect(lambda _selection: self.schedule_publish(FACET_ROIS))
            img_view.roiChanged.connect(
                lambda _roi_id, _geometry: self.schedule_publish(FACET_ROIS)
            )
            img_view.roiDeleted.connect(lambda _roi_id: self.schedule_publish(FACET_ROIS))
        toolbar = getattr(win, "display_toolbar", None)
        if toolbar is not None:
            toolbar.windowModeChanged.connect(lambda _mode: self.schedule_publish(FACET_LEVELS))
            toolbar.autoWindowRequested.connect(lambda: self.schedule_publish(FACET_LEVELS))
