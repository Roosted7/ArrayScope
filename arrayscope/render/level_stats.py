"""Montage histogram and level-stat maintenance for the render pipeline."""

from __future__ import annotations

from collections import deque
from time import perf_counter

import numpy as np
import pyqtgraph.Qt as Qt

from arrayscope.core.gui_callback_budget import GuiCallbackBudget
from arrayscope.kernel import Lane as WorkLane, WorkItem, complete_inline_work as _complete_inline_work
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.model.montage_levels import (
    MontageLevelStats,
    MontageLevelTracker,
    montage_level_key,
    provisional_tile_level_stats,
    sample_tile_level_stats,
)
from arrayscope.display.montage import RenderedTile
from arrayscope.display.planning import LevelSourceRank, normalize_bounds
from arrayscope.operations.evaluator import _document_key
from arrayscope.render import effects as render_effects
from arrayscope.window.evaluation_controller import EvalPriority
from arrayscope.window import montage_commit


MONTAGE_LEVEL_STATS_COMMIT_BATCH = 8
MONTAGE_LEVEL_STATS_BACKGROUND_BATCH = 4
MONTAGE_LEVEL_STATS_BACKGROUND_BUDGET_MS = 4.0


class LevelStatsService:
    def _montage_level_key(self, document, view_state, all_indices, colormap_lut):
        return montage_level_key(
            _document_key(document),
            view_state,
            all_indices,
            colormap_lut,
        )

    def _montage_level_expected_indices(self, session) -> tuple[int, ...]:
        expected = tuple(int(index) for index in getattr(session, "level_expected_indices", ()) or ())
        if expected:
            return expected
        return tuple(int(tile.source_index) for tile in getattr(session.plan, "tiles", ()))

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

    def _preview_evidence_can_refine(self) -> bool:
        """Whether preview evidence may close the refined level pass."""

        # Preview/reduced tiles are useful first-display evidence, including on
        # shader-windowing backends where levels can be applied cheaply. They
        # must still remain provisional: promoting them to refined can suppress
        # the final target-quality histogram/window-level pass.
        return False

    def _update_montage_level_bounds_from_rendered(self, level_key, rendered, *, expected_indices=None, refined: bool = False) -> None:
        if expected_indices is None:
            previous_stats = self._montage_level_tracker().stats_for(level_key)
            expected_indices = () if previous_stats is None else previous_stats.expected_indices
        tracker = self._montage_level_tracker()
        tracker.ensure_expected(level_key, expected_indices)
        source_index = int(rendered.tile.source_index)
        if refined and _rendered_tile_is_preview(rendered) and not self._preview_evidence_can_refine():
            refined = False
        refined = bool(refined)
        level_stats = getattr(rendered, "level_stats", None)
        existing_refined = tracker.has_source(level_key, source_index, refined=True)
        existing_any = existing_refined or tracker.has_source(level_key, source_index)
        if existing_refined or (
            existing_any
            and not bool(refined)
            and (level_stats is None or not bool(getattr(level_stats, "refined", False)))
        ):
            return
        if level_stats is not None and not refined:
            tracker.update_from_stats(level_key, level_stats, aggregate=False)
            return
        level_data = getattr(rendered, "level_data", None)
        if level_data is not None and not refined:
            stats = provisional_tile_level_stats(level_data, source_index)
            if stats is not None:
                tracker.update_from_stats(level_key, stats, aggregate=False)
                return
        stats = sample_tile_level_stats(
            render_effects.montage_refined_level_values(rendered),
            source_index,
            refined=bool(refined),
        )
        if stats is not None:
            tracker.update_from_stats(level_key, stats, aggregate=False)
        elif refined:
            # Nothing finite to sample: record that as refined evidence, or
            # level convergence re-queues this source forever and an
            # explicit-auto flush parked on the rank can never re-commit.
            tracker.record_vacuous_source(level_key, source_index)

    def _update_montage_level_bounds_from_prepared(self, level_key, rendered, *, expected_indices=None, require_refined: bool = False) -> bool:
        """Merge already-prepared level evidence without sampling source pixels."""

        if expected_indices is None:
            previous_stats = self._montage_level_tracker().stats_for(level_key)
            expected_indices = () if previous_stats is None else previous_stats.expected_indices
        tracker = self._montage_level_tracker()
        tracker.ensure_expected(level_key, expected_indices)
        source_index = int(rendered.tile.source_index)
        if bool(require_refined) and _rendered_tile_is_preview(rendered) and not self._preview_evidence_can_refine():
            return False
        if tracker.has_source(level_key, source_index, refined=bool(require_refined)):
            return True
        if require_refined and tracker.has_source(level_key, source_index):
            return False
        level_stats = getattr(rendered, "level_stats", None)
        if level_stats is not None:
            if require_refined and not bool(getattr(level_stats, "refined", False)):
                return False
            tracker.update_from_stats(level_key, level_stats, aggregate=False)
            return True
        if require_refined:
            return False
        level_data = getattr(rendered, "level_data", None)
        if level_data is not None:
            stats = provisional_tile_level_stats(level_data, source_index)
            if stats is not None:
                tracker.update_from_stats(level_key, stats, aggregate=False)
                return True
        return False

    def _queue_montage_level_refinement(self, session, rendered) -> None:
        if _rendered_tile_is_preview(rendered) and not self._preview_evidence_can_refine():
            return
        tracker = self._montage_level_tracker()
        source_index = int(rendered.tile.source_index)
        if tracker.has_source(session.level_key, source_index, refined=True):
            return
        pending = getattr(session, "pending_refined_level_tiles", None)
        if pending is None:
            pending = deque()
            session.pending_refined_level_tiles = pending
        pending_sources = getattr(session, "pending_refined_level_sources", None)
        if pending_sources is None:
            pending_sources = {int(item.tile.source_index) for item in pending}
            session.pending_refined_level_sources = pending_sources
        if source_index in pending_sources:
            return
        pending.append(rendered)
        pending_sources.add(source_index)

    def _rendered_tile_for_current_payload(self, session, tile_number: int, payload) -> RenderedTile | None:
        rendered = getattr(session, "rendered_tiles", {}).get(int(tile_number))
        if rendered is not None:
            return rendered
        plan_tiles = tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ())
        if 0 <= int(tile_number) < len(plan_tiles):
            tile = plan_tiles[int(tile_number)]
            if int(getattr(tile, "montage_index", int(tile_number))) == int(tile_number):
                if int(getattr(payload, "source_index", -1)) != int(getattr(tile, "source_index", -2)):
                    return None
                return _rendered_tile_from_previous_payload(tile, payload)
        for offset, tile in enumerate(plan_tiles):
            if int(getattr(tile, "montage_index", offset)) != int(tile_number):
                continue
            if int(getattr(payload, "source_index", -1)) != int(getattr(tile, "source_index", -2)):
                return None
            return _rendered_tile_from_previous_payload(tile, payload)
        return None

    def _queue_montage_current_level_evidence(self, session) -> int:
        """Seed rough/full semantic level evidence from currently displayed payloads."""

        tracker = self._montage_level_tracker()
        expected = self._montage_level_expected_indices(session)
        tracker.ensure_expected(session.level_key, expected)
        summary = tracker.summary_for(session.level_key)
        complete = summary is not None and summary.rank in {
            LevelSourceRank.MONTAGE_COMPLETE,
            LevelSourceRank.MONTAGE_SAMPLED_FULL,
        }
        if complete:
            return 0
        payloads = {
            int(tile): payload
            for tile, payload in (getattr(session, "display_tile_payloads", {}) or {}).items()
            if self._rendered_tile_for_current_payload(session, int(tile), payload) is not None
        }
        if not payloads:
            return 0
        return self._queue_montage_level_stats_for_payloads(session, payloads)

    def _queue_montage_final_level_refinements(self, session) -> None:
        """Queue settled final/target payloads for refined stats.

        First-display preview stats are deliberately provisional. Once the
        montage itself is complete, exact target payloads can feed the refined
        histogram/window-level pass without competing with visible rendering.
        """

        queued_tiles: set[int] = set()
        for tile_number, rendered in tuple((getattr(session, "rendered_tiles", {}) or {}).items()):
            if _rendered_tile_is_preview(rendered):
                continue
            queued_tiles.add(int(tile_number))
            self._queue_montage_level_refinement(session, rendered)

        for tile_number, payload in tuple((getattr(session, "display_tile_payloads", {}) or {}).items()):
            tile_number = int(tile_number)
            if tile_number in queued_tiles:
                continue
            if str(getattr(payload, "quality", "exact") or "exact") == "preview":
                continue
            rendered = self._rendered_tile_for_current_payload(session, tile_number, payload)
            if rendered is None:
                continue
            self._queue_montage_level_refinement(session, rendered)

    def _montage_level_stats_for_session(self, session) -> MontageLevelStats:
        expected = self._montage_level_expected_indices(session)
        self._montage_level_tracker().ensure_expected(session.level_key, expected)
        stats = self._montage_level_tracker().summary_for(session.level_key)
        if stats is None:
            return self._ensure_montage_level_stats(session.level_key, expected_indices=expected)
        return stats

    def _montage_level_bounds_for_session(self, session, *, allow_partial: bool = False):
        source = self._montage_level_source_for_session(session, allow_partial=allow_partial)
        return None if source is None else source.histogram_range

    def _montage_level_source_for_session(self, session, *, allow_partial: bool = False):
        # Partial semantic tile coverage is a valid provisional level source.
        # It must not be confused with viewport pixels; the level key is semantic
        # and excludes zoom/pan.  WindowLevelController keeps updates monotonic.
        tracker = self._montage_level_tracker()
        stats = tracker.summary_for(session.level_key)
        if stats is None:
            return None
        if not allow_partial and stats.rank not in {LevelSourceRank.MONTAGE_COMPLETE, LevelSourceRank.MONTAGE_SAMPLED_FULL}:
            return None
        return tracker.source_for_stats(session.level_key, stats)

    def _montage_histogram_plot_data_for_session(self, session, *, allow_partial: bool = False):
        tracker = self._montage_level_tracker()
        stats = tracker.stats_for(session.level_key)
        if stats is None:
            return None
        if not allow_partial and stats.rank not in {LevelSourceRank.MONTAGE_COMPLETE, LevelSourceRank.MONTAGE_SAMPLED_FULL}:
            return None
        return tracker.histogram_data_for_stats(stats)

    def _montage_level_tracker(self) -> MontageLevelTracker:
        tracker = getattr(self, "_montage_level_tracker_instance", None)
        if tracker is None:
            tracker = MontageLevelTracker()
            self._montage_level_tracker_instance = tracker
        return tracker

    def _schedule_montage_cached_level_stats(self, session) -> None:
        if (
            not getattr(session, "pending_level_tiles", None)
            and int(getattr(session, "level_scan_remaining_tiles", 0) or 0) <= 0
        ):
            return
        timer = getattr(self, "_montage_level_stats_timer", None)
        if timer is None:
            # Bounded continuation. Cached level stats are secondary UI work
            # and each slice is budgeted by `_process_montage_cached_level_stats`;
            # remove when histogram refinement is fully kernel-owned.
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._process_montage_cached_level_stats)
            self._montage_level_stats_timer = timer
        if not timer.isActive():
            timer.start(0)

    def _pending_montage_level_sources(self, session):
        pending = getattr(session, "pending_level_tiles", None)
        if pending is None:
            pending = deque()
            session.pending_level_tiles = pending
        queued_sources = getattr(session, "pending_level_sources", None)
        if queued_sources is None:
            queued_sources = {int(item.tile.source_index) for item in pending}
            session.pending_level_sources = queued_sources
        return pending, queued_sources

    def _mark_montage_level_scan_pending(self, session) -> None:
        # Restart a FULL pass even mid-scan: a tile that materializes after
        # the cursor already passed its position would otherwise fall through
        # a completed pass and no continuation would ever sample it (level
        # rank then never completes for exactly that source).  Passes are
        # cheap — already-merged sources are skip-checked — and arrivals are
        # bounded by the plan, so restarts terminate.
        tile_count = len(getattr(getattr(session, "plan", None), "tiles", ()) or ())
        if tile_count <= 0:
            return
        session.level_scan_remaining_tiles = tile_count

    def _queue_montage_cached_level_stats(self, session, rendered_tiles, *, seed_if_empty: bool) -> None:
        """Admit cached-payload level work without making commit latency scale with tile count.

        Prepared per-tile evidence can be merged immediately for a small,
        bounded batch.  Anything requiring source-pixel sampling is queued for
        the same timer-driven maintenance path used by later cached residents.
        """

        tracker = self._montage_level_tracker()
        expected = self._montage_level_expected_indices(session)
        tracker.ensure_expected(session.level_key, expected)
        pending, queued_sources = self._pending_montage_level_sources(session)
        require_refined = _montage_level_evidence_requires_refined(self, session)
        summary = tracker.summary_for(session.level_key)
        inspected = 0
        seeded = bool(summary is not None and summary.source_indices)
        for rendered in rendered_tiles or ():
            if inspected >= MONTAGE_LEVEL_STATS_COMMIT_BATCH:
                self._mark_montage_level_scan_pending(session)
                break
            inspected += 1
            source_index = int(rendered.tile.source_index)
            if source_index in queued_sources or tracker.has_source(session.level_key, source_index, refined=require_refined):
                continue
            if self._update_montage_level_bounds_from_prepared(
                session.level_key,
                rendered,
                expected_indices=expected,
                require_refined=require_refined,
            ):
                seeded = True
                self._queue_montage_level_refinement(session, rendered)
                continue
            pending.append(rendered)
            queued_sources.add(source_index)
            if seed_if_empty and not seeded:
                seeded = True
        self._montage_pending_level_tiles_last_session = len(pending or ())

    def _queue_montage_level_stats_for_payloads(self, session, payloads) -> int:
        """Request level evidence for a presentation delta without scanning it inline."""

        tracker = self._montage_level_tracker()
        expected = self._montage_level_expected_indices(session)
        tracker.ensure_expected(session.level_key, expected)
        stats_start = perf_counter()
        merged = 0
        inspected = 0
        pending, queued_sources = self._pending_montage_level_sources(session)
        require_refined = _montage_level_evidence_requires_refined(self, session)
        tiles_by_number = {
            int(getattr(tile, "montage_index", offset)): tile
            for offset, tile in enumerate(tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ()))
        }
        for tile_number in payloads or ():
            if inspected >= MONTAGE_LEVEL_STATS_COMMIT_BATCH:
                self._mark_montage_level_scan_pending(session)
                break
            inspected += 1
            payload = payloads.get(int(tile_number)) if isinstance(payloads, dict) else None
            rendered = getattr(session, "rendered_tiles", {}).get(int(tile_number))
            if rendered is None and payload is not None:
                tile = tiles_by_number.get(int(tile_number))
                if tile is not None:
                    rendered = _rendered_tile_from_previous_payload(tile, payload)
            if rendered is None:
                continue
            source_index = int(rendered.tile.source_index)
            if tracker.has_source(session.level_key, source_index, refined=require_refined):
                continue
            if self._update_montage_level_bounds_from_prepared(
                session.level_key,
                rendered,
                expected_indices=expected,
                require_refined=require_refined,
            ):
                self._queue_montage_level_refinement(session, rendered)
                queued_sources.discard(source_index)
                merged += 1
            elif source_index not in queued_sources:
                pending.append(rendered)
                queued_sources.add(source_index)
        self._last_montage_level_stats_ms = (perf_counter() - stats_start) * 1000.0
        self._montage_level_sources_added_last_commit = int(merged)
        self._montage_pending_level_tiles_last_session = len(getattr(session, "pending_level_tiles", ()) or ())
        self._schedule_montage_cached_level_stats(session)
        return int(merged)

    def _scan_montage_level_stats_from_session(self, session, *, expected, stats_start: float, processed: int, budget: GuiCallbackBudget) -> int:
        tile_count = len(getattr(getattr(session, "plan", None), "tiles", ()) or ())
        remaining = int(getattr(session, "level_scan_remaining_tiles", 0) or 0)
        if tile_count <= 0 or remaining <= 0:
            session.level_scan_remaining_tiles = 0
            return int(processed)
        pending, queued_sources = self._pending_montage_level_sources(session)
        tracker = self._montage_level_tracker()
        require_refined = _montage_level_evidence_requires_refined(self, session)
        cursor = int(getattr(session, "level_scan_cursor", 0) or 0) % tile_count
        inspected = 0
        while remaining > 0 and inspected < int(budget.item_cap):
            rendered = getattr(session, "rendered_tiles", {}).get(cursor)
            if rendered is None:
                payload = (getattr(session, "display_tile_payloads", {}) or {}).get(cursor)
                if payload is not None:
                    rendered = self._rendered_tile_for_current_payload(session, cursor, payload)
            cursor = (cursor + 1) % tile_count
            remaining -= 1
            inspected += 1
            if rendered is None:
                continue
            source_index = int(rendered.tile.source_index)
            if source_index in queued_sources or tracker.has_source(session.level_key, source_index, refined=require_refined):
                continue
            if self._update_montage_level_bounds_from_prepared(
                session.level_key,
                rendered,
                expected_indices=expected,
                require_refined=require_refined,
            ):
                self._queue_montage_level_refinement(session, rendered)
                processed += 1
            else:
                pending.append(rendered)
                queued_sources.add(source_index)
            budget.record_item(byte_count=montage_commit.rendered_tile_nbytes(rendered))
            if budget.should_yield():
                break
        session.level_scan_cursor = int(cursor)
        session.level_scan_remaining_tiles = max(0, int(remaining))
        return int(processed)

    def _process_montage_cached_level_stats(self) -> None:
        session = getattr(self, "_montage_session", None)
        if not self._montage_session_is_current(session):
            return
        pending = getattr(session, "pending_level_tiles", None)
        if not pending and int(getattr(session, "level_scan_remaining_tiles", 0) or 0) <= 0:
            return
        budget = self._montage_callback_budget(
            "montage_level_evidence",
            interactive=_interactive_active(self),
            work_class="semantic_level_evidence",
            item_cap=MONTAGE_LEVEL_STATS_BACKGROUND_BATCH,
            target_ms=MONTAGE_LEVEL_STATS_BACKGROUND_BUDGET_MS,
        )
        stats_start = perf_counter()
        expected = self._montage_level_expected_indices(session)
        require_refined = _montage_level_evidence_requires_refined(self, session)
        processed = 0
        while pending and processed < int(budget.item_cap):
            rendered = pending.popleft()
            source_index = int(rendered.tile.source_index)
            pending_sources = getattr(session, "pending_level_sources", None)
            if pending_sources is not None:
                pending_sources.discard(source_index)
            if not self._montage_level_tracker().has_source(session.level_key, source_index, refined=require_refined):
                self._update_montage_level_bounds_from_rendered(
                    session.level_key,
                    rendered,
                    expected_indices=expected,
                    refined=require_refined,
                )
            self._queue_montage_level_refinement(session, rendered)
            processed += 1
            budget.record_item(byte_count=montage_commit.rendered_tile_nbytes(rendered))
            if budget.should_yield():
                break
        if not pending and not budget.should_yield():
            processed = self._scan_montage_level_stats_from_session(
                session,
                expected=expected,
                stats_start=stats_start,
                processed=processed,
                budget=budget,
            )
        self._last_montage_level_stats_ms = (perf_counter() - stats_start) * 1000.0
        self._montage_pending_level_tiles_last_session = len(pending or ())
        self._record_gui_budget(budget)
        if processed:
            _complete_inline_work(
                self,
                WorkItem(
                    key=(
                        "montage_level_evidence",
                        session.key,
                        int(session.session_id),
                        int(getattr(session, "level_revision", 0) or 0),
                        int(processed),
                    ),
                    lane=WorkLane.HISTOGRAM_REFINEMENT,
                    quality="retained",
                    supersession_key=("montage-level-evidence", session.key),
                    supersession_value=int(session.session_id),
                    estimated_cpu_ms=float(self._last_montage_level_stats_ms or 0.0),
                    estimated_bytes=int(budget.processed_bytes),
                ),
            )
        # A histogram/level refinement is presentation metadata.  It must not
        # force a full tiled-payload refresh or replay stale removals after a
        # viewport change.  Normal display commits will publish richer sources;
        # when there is no visible upload backlog, a non-forced commit can
        # update uniforms/histogram without invalidating residency.
        # Evidence-drain pacing (wedge cost fix 2026-07-05): while a parked
        # explicit-auto flush waits on level evidence, committing after EVERY
        # budget slice re-runs the full payload build per handful of tiles
        # (~68 no-op commits for a 272-tile scene).  Commit when the evidence
        # queue actually drained — the parked flush re-checks the rank then —
        # or when nothing is parked (metadata refresh for a settled session).
        evidence_remaining = bool(pending) or int(getattr(session, "level_scan_remaining_tiles", 0) or 0) > 0
        flush_parked = bool(getattr(session, "flush_pending", False) or getattr(session, "final_commit_pending", False))
        if (
            processed
            and not (evidence_remaining and flush_parked)
            and not getattr(session, "dirty_payloads", ())
            and not getattr(session, "pending_removals", ())
        ):
            self.apply_montage_presentation(session)
        self._schedule_montage_cached_level_stats(session)

    def _schedule_montage_refined_level_stats(self, session) -> None:
        if not self._montage_session_is_current(session):
            return
        pending = getattr(session, "pending_refined_level_tiles", None)
        if not pending:
            return
        controller = getattr(self.win, "histogram_evaluation_controller", None)
        if controller is None:
            return
        scheduled = 0
        while pending and scheduled < 4:
            rendered = pending.popleft()
            source_index = int(rendered.tile.source_index)
            if self._montage_level_tracker().has_source(session.level_key, source_index, refined=True):
                pending_sources = getattr(session, "pending_refined_level_sources", None)
                if pending_sources is not None:
                    pending_sources.discard(source_index)
                continue
            source = render_effects.montage_refined_level_values(rendered)
            key = ("montage_refined_level_stats", session.level_key, source_index)

            def evaluate(source=source, source_index=source_index):
                return sample_tile_level_stats(source, int(source_index), refined=True)

            def done(
                stats,
                session_id=session.session_id,
                session_key=session.key,
                level_key=session.level_key,
                source_index=source_index,
            ):
                self._on_montage_refined_level_stats_done(session_id, session_key, level_key, source_index, stats)

            started = controller.start_latest(
                evaluate,
                on_done=done,
                key=key,
                priority=EvalPriority.HISTOGRAM,
                replace_group=f"montage_level_refinement:{source_index}",
                memory_budget_bytes=self._memory_policy().display_cache_budget_bytes,
            )
            if started is None:
                pending.appendleft(rendered)
                break
            scheduled += 1

    def _on_montage_refined_level_stats_done(self, session_id, session_key, level_key, source_index, stats) -> None:
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session_id, session_key):
            return
        pending_sources = getattr(session, "pending_refined_level_sources", None)
        if pending_sources is not None:
            pending_sources.discard(int(source_index))
        if stats is not None:
            self._montage_level_tracker().update_from_stats(level_key, stats, aggregate=False)
            summary = self._montage_level_tracker().summary_for(level_key)
            if (
                summary is not None
                and session.display_committed
                and not getattr(session, "dirty_payloads", ())
                and not getattr(session, "pending_removals", ())
                and getattr(session, "user_levels_override", None) is None
                and self._should_publish_montage_level_metadata(session, summary)
            ):
                self.apply_montage_presentation(session)
        self._schedule_montage_refined_level_stats(session)



def _rendered_tile_from_previous_payload(tile, payload) -> RenderedTile:
    semantic = None if payload.semantic_data is None else np.asarray(payload.semantic_data)
    image = semantic if semantic is not None else np.asarray(payload.image)
    semantic_histogram = (
        None
        if getattr(payload, "semantic_histogram_data", None) is None
        else np.asarray(payload.semantic_histogram_data)
    )
    histogram = semantic_histogram if semantic_histogram is not None else (
        None if payload.histogram_data is None else np.asarray(payload.histogram_data)
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
        lod=getattr(payload, "lod", None),
        level_data=getattr(payload, "level_data", None),
        level_stats=getattr(payload, "level_stats", None),
        quality=str(getattr(payload, "quality", "exact") or "exact"),
    )


def _rendered_tile_is_preview(rendered) -> bool:
    return str(getattr(rendered, "quality", "exact") or "exact") == "preview"


def _interactive_active(window) -> bool:
    coordinator = getattr(window.win, "render_coordinator", None)
    return bool(
        coordinator is not None and getattr(coordinator, "interactive_active", False)
        or _viewport_interaction_active(window)
    )


def _montage_level_evidence_requires_refined(window, session) -> bool:
    level_generation = getattr(session, "level_generation", None)
    requested_levels = (
        normalize_bounds(getattr(level_generation, "target_levels", None))
        or normalize_bounds(getattr(session, "user_levels_override", None))
    )
    return bool(
        requested_levels is None
        and getattr(session, "force_auto", False)
        and not image_view_backend_capabilities(window.win.img_view).shader_windowing
    )


def _viewport_interaction_active(window) -> bool:
    return bool(getattr(window.win, "_viewport_interaction_active", False))
