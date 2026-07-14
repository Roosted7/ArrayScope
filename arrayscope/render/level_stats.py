"""Montage histogram and level-stat maintenance for the render pipeline."""

from __future__ import annotations

from collections import deque
from time import perf_counter

import numpy as np

from arrayscope.app.errors import handle_ui_exception
from arrayscope.core.bounded_cache import BoundedCache
from arrayscope.kernel import (
    Lane as WorkLane,
    Priority,
    Supersession,
    TaskSpec,
    UNRANKED_SCHEDULING_RANK,
    WorkItem,
    complete_inline_work as _complete_inline_work,
)
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.model.montage_levels import (
    AGGREGATE_SAMPLE_LIMIT,
    LevelEvidenceQuality,
    MontageLevelStats,
    MontageLevelTracker,
    REFINED_TILE_SAMPLE_LIMIT,
    montage_level_key,
    provisional_tile_level_stats,
    sample_tile_level_stats,
    tile_level_stats_with_quality,
)
from arrayscope.display.montage import RenderedTile
from arrayscope.display.planning import LevelSourceRank, normalize_bounds
from arrayscope.operations.evaluator import (
    _document_key,
    evaluate_level_evidence_snapshot,
    stage_document_key,
)
from arrayscope.render import effects as render_effects
from arrayscope.window import frame_effects as montage_commit
from arrayscope.window.frame_session import (
    SemanticLevelEvidenceProgress,
    SemanticLevelEvidenceTarget,
)


MONTAGE_LEVEL_STATS_COMMIT_BATCH = 4
MONTAGE_LEVEL_STATS_BACKGROUND_BATCH = 2
# Refined first-frame evidence is worker-side NumPy sampling. Four sources per
# submission made a 60-tile PyQtGraph successor wait through 15 kernel/Qt
# round-trips (~1.2 s before the first atomic frame). Sixteen keeps the merge
# callback bounded while reducing the visible dependency to four handoffs.
MONTAGE_LEVEL_STATS_FIRST_CPU_BATCH = 16
MONTAGE_LEVEL_STATS_BACKGROUND_BUDGET_MS = 4.0


class LevelStatsService:
    def _montage_source_level_cache(self) -> BoundedCache:
        cache = getattr(self, "_montage_source_level_cache_instance", None)
        if cache is None:
            cache = BoundedCache(max_entries=4096, max_bytes=32 * 1024 * 1024)
            self._montage_source_level_cache_instance = cache
        return cache

    def _remember_montage_source_level_stats(self, level_key, stats) -> None:
        if stats is None:
            return
        cache = self._montage_source_level_cache()
        cache_key = (_montage_level_family_key(level_key), int(stats.source_index))
        previous = cache.peek(cache_key)
        if (
            previous is not None
            and int(getattr(previous, "evidence_quality", 0) or 0)
            > int(getattr(stats, "evidence_quality", 0) or 0)
        ):
            return
        sample = np.asarray(getattr(stats, "sample", ()))
        cache.put(
            cache_key,
            stats,
            nbytes=int(sample.nbytes),
        )

    def _cached_montage_source_level_stats(self, level_key, source_index: int, quality):
        stats = self._montage_source_level_cache().get(
            (_montage_level_family_key(level_key), int(source_index))
        )
        if stats is None:
            return None
        return (
            stats
            if int(getattr(stats, "evidence_quality", 0) or 0) >= int(quality)
            else None
        )

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

    def _update_montage_level_bounds_from_rendered(
        self,
        level_key,
        rendered,
        *,
        expected_indices=None,
        refined: bool = False,
        evidence_quality: LevelEvidenceQuality | int | str | None = None,
    ) -> None:
        if expected_indices is None:
            previous_stats = self._montage_level_tracker().stats_for(level_key)
            expected_indices = () if previous_stats is None else previous_stats.expected_indices
        tracker = self._montage_level_tracker()
        tracker.ensure_expected(level_key, expected_indices)
        source_index = int(rendered.tile.source_index)
        quality = _rendered_level_evidence_quality(
            rendered,
            refined=bool(refined),
            evidence_quality=evidence_quality,
        )
        if refined and quality == LevelEvidenceQuality.ROUGH_PREVIEW and not self._preview_evidence_can_refine():
            refined = False
            quality = LevelEvidenceQuality.ROUGH_PREVIEW
        refined = bool(refined)
        level_stats = getattr(rendered, "level_stats", None)
        if tracker.has_source_quality(level_key, source_index, quality):
            return
        if level_stats is not None and (not refined or bool(getattr(level_stats, "refined", False))):
            stats = tile_level_stats_with_quality(
                level_stats,
                quality,
                source_index=source_index,
            )
            tracker.update_from_stats(
                level_key,
                stats,
                aggregate=False,
            )
            self._remember_montage_source_level_stats(level_key, stats)
            return
        level_data = getattr(rendered, "level_data", None)
        if level_data is not None and not refined:
            stats = provisional_tile_level_stats(level_data, source_index, evidence_quality=quality)
            if stats is not None:
                tracker.update_from_stats(level_key, stats, aggregate=False)
                self._remember_montage_source_level_stats(level_key, stats)
                return
        stats = sample_tile_level_stats(
            render_effects.montage_refined_level_values(rendered),
            source_index,
            refined=bool(refined),
            evidence_quality=quality,
        )
        if stats is not None:
            tracker.update_from_stats(level_key, stats, aggregate=False)
            self._remember_montage_source_level_stats(level_key, stats)
            if refined:
                self._montage_refined_level_applied_count = int(
                    getattr(self, "_montage_refined_level_applied_count", 0)
                ) + 1
        elif refined:
            # Nothing finite to sample: record that as refined evidence, or
            # level convergence re-queues this source forever and an
            # explicit-auto flush parked on the rank can never re-commit.
            tracker.record_vacuous_source(level_key, source_index)

    def _update_montage_level_bounds_from_prepared(
        self,
        level_key,
        rendered,
        *,
        expected_indices=None,
        require_refined: bool = False,
        evidence_quality: LevelEvidenceQuality | int | str | None = None,
    ) -> bool:
        """Merge already-prepared level evidence without sampling source pixels."""

        if expected_indices is None:
            previous_stats = self._montage_level_tracker().stats_for(level_key)
            expected_indices = () if previous_stats is None else previous_stats.expected_indices
        tracker = self._montage_level_tracker()
        tracker.ensure_expected(level_key, expected_indices)
        source_index = int(rendered.tile.source_index)
        quality = _rendered_level_evidence_quality(
            rendered,
            refined=bool(require_refined),
            evidence_quality=evidence_quality,
        )
        if bool(require_refined) and quality == LevelEvidenceQuality.ROUGH_PREVIEW and not self._preview_evidence_can_refine():
            return False
        if tracker.has_source_quality(level_key, source_index, quality):
            return True
        level_stats = getattr(rendered, "level_stats", None)
        if level_stats is not None:
            if require_refined and not bool(getattr(level_stats, "refined", False)):
                return False
            stats = tile_level_stats_with_quality(
                level_stats,
                quality,
                source_index=source_index,
            )
            tracker.update_from_stats(
                level_key,
                stats,
                aggregate=False,
            )
            self._remember_montage_source_level_stats(level_key, stats)
            return True
        if require_refined:
            return False
        level_data = getattr(rendered, "level_data", None)
        if level_data is not None:
            stats = provisional_tile_level_stats(level_data, source_index, evidence_quality=quality)
            if stats is not None:
                tracker.update_from_stats(level_key, stats, aggregate=False)
                self._remember_montage_source_level_stats(level_key, stats)
                return True
        return False

    def _queue_montage_level_refinement(self, session, rendered) -> None:
        if bool(getattr(session, "shader_display", False)):
            # Shader first pixels use payload-prepared rough evidence. Final
            # refinement belongs exclusively to the statistics-only semantic
            # evidence owner, including sources absent from display payloads.
            return
        quality = _rendered_level_evidence_quality_for_session(session, rendered, refined=True)
        if quality == LevelEvidenceQuality.ROUGH_PREVIEW and not self._preview_evidence_can_refine():
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

    def _admit_first_pass_level_evidence(self, session, rendered, *, quality: str) -> bool:
        """Merge worker-prepared rough evidence for the one physical first pass."""

        if not bool(getattr(session, "shader_display", False)):
            return False
        if not session.note_first_pass_quality(quality):
            return False
        evidence_quality = (
            LevelEvidenceQuality.ROUGH_PREVIEW
            if str(quality) == "preview"
            else LevelEvidenceQuality.ROUGH_TARGET
        )
        return bool(
            self._update_montage_level_bounds_from_prepared(
                session.level_key,
                rendered,
                expected_indices=self._montage_level_expected_indices(session),
                require_refined=False,
                evidence_quality=evidence_quality,
            )
        )

    def _first_pass_level_evidence_complete(self, session) -> bool:
        if not bool(getattr(session, "shader_display", False)):
            return False
        if not session.first_pass_pixels_presented():
            return False
        plan_tiles = {
            int(getattr(tile, "montage_index", offset)): tile
            for offset, tile in enumerate(
                tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ())
            )
        }
        required = getattr(session, "required_tile_numbers", None)
        if not callable(required):
            raise RuntimeError("live frame session has no required-tile owner")
        tile_numbers = tuple(required())
        expected = {
            int(plan_tiles[int(tile_number)].source_index)
            for tile_number in tile_numbers
            if int(tile_number) in plan_tiles
        }
        summary = self._montage_level_tracker().summary_for(session.level_key)
        covered = set() if summary is None else set(int(source) for source in summary.source_indices)
        return bool(expected and expected <= covered)

    @staticmethod
    def _first_pass_rough_evidence_closed(session) -> bool:
        return bool(
            getattr(session, "shader_display", False)
            and getattr(session, "first_pass_histogram_published", False)
        )

    def _queue_montage_final_level_refinements(self, session) -> None:
        """Queue settled final/target payloads for refined stats.

        First-display preview stats are deliberately provisional. Once the
        montage itself is complete, exact target payloads can feed the refined
        histogram/window-level pass without competing with visible rendering.
        """

        queued_tiles: set[int] = set()
        for tile_number, rendered in tuple((getattr(session, "rendered_tiles", {}) or {}).items()):
            if _rendered_level_evidence_quality_for_session(session, rendered, refined=True) == LevelEvidenceQuality.ROUGH_PREVIEW:
                continue
            queued_tiles.add(int(tile_number))
            self._queue_montage_level_refinement(session, rendered)

        for tile_number, payload in tuple((getattr(session, "display_tile_payloads", {}) or {}).items()):
            tile_number = int(tile_number)
            if tile_number in queued_tiles:
                continue
            rendered = self._rendered_tile_for_current_payload(session, tile_number, payload)
            if rendered is None:
                continue
            if _rendered_level_evidence_quality_for_session(session, rendered, refined=True) == LevelEvidenceQuality.ROUGH_PREVIEW:
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

    def _ensure_semantic_level_evidence_target(self, session):
        document = getattr(session, "document", None)
        view_state = getattr(session, "view_state", None)
        if document is None or view_state is None or getattr(session, "montage_axis", None) is None:
            return None
        expected = self._montage_level_expected_indices(session)
        generation = (
            int(getattr(session, "session_id", 0) or 0),
            getattr(session, "level_key", None),
            expected,
        )
        current = getattr(session, "semantic_level_evidence_target", None)
        if current is not None and current.generation == generation:
            return current
        target = SemanticLevelEvidenceTarget(
            generation=generation,
            level_key=session.level_key,
            expected_sources=expected,
            pixel_limit=int(REFINED_TILE_SAMPLE_LIMIT),
            aggregate_sample_limit=int(AGGREGATE_SAMPLE_LIMIT),
            blocking_batch_limit=int(MONTAGE_LEVEL_STATS_FIRST_CPU_BATCH),
            background_batch_limit=int(MONTAGE_LEVEL_STATS_BACKGROUND_BATCH),
        )
        visible_dependency = bool(
            _montage_level_evidence_requires_refined(self, session)
            and not bool(getattr(session, "display_committed", False))
        )
        progress = SemanticLevelEvidenceProgress(
            target=target,
            current_batch_limit=(
                target.blocking_batch_limit
                if visible_dependency
                else target.background_batch_limit
            ),
        )
        session.semantic_level_evidence_target = target
        session.semantic_level_evidence_progress = progress
        self._montage_level_tracker().ensure_expected(session.level_key, expected)
        return target

    def _take_semantic_level_evidence_sources(self, session) -> tuple[int, ...]:
        target = self._ensure_semantic_level_evidence_target(session)
        progress = getattr(session, "semantic_level_evidence_progress", None)
        if target is None or progress is None:
            return ()
        tracker = self._montage_level_tracker()
        limit = max(1, int(progress.current_batch_limit))
        sources: list[int] = []
        inspected = 0
        while progress.cursor < target.target_population and inspected < limit:
            source_index = int(target.expected_sources[progress.cursor])
            progress.cursor += 1
            inspected += 1
            cached = self._cached_montage_source_level_stats(
                target.level_key,
                source_index,
                LevelEvidenceQuality.REFINED,
            )
            if cached is not None:
                tracker.update_from_stats(target.level_key, cached, aggregate=False)
            if tracker.has_source_quality(
                target.level_key,
                source_index,
                LevelEvidenceQuality.REFINED,
            ):
                progress.record_covered(source_index)
                continue
            sources.append(source_index)
        return tuple(sources)

    def _schedule_semantic_level_evidence(self, session) -> None:
        if not self._frame_session_is_current(session):
            return
        if bool(getattr(session, "shader_display", False)):
            if not bool(getattr(session, "first_pass_histogram_published", False)):
                return
            if not _montage_side_work_visible_settled(self, session):
                return
        target = self._ensure_semantic_level_evidence_target(session)
        progress = getattr(session, "semantic_level_evidence_progress", None)
        if target is None or progress is None or progress.inflight_generation is not None:
            return
        visible_dependency = bool(
            _montage_level_evidence_requires_refined(self, session)
            and not bool(getattr(session, "display_committed", False))
        )
        progress.current_batch_limit = int(
            target.blocking_batch_limit if visible_dependency else target.background_batch_limit
        )
        if len(progress.covered_sources) >= target.target_population:
            progress.blocking_reason = "ready"
            return
        sources = self._take_semantic_level_evidence_sources(session)
        if not sources and progress.cursor >= target.target_population:
            progress.blocking_reason = "ready"
            self._maybe_publish_after_level_evidence(session, processed=1)
            return

        generation = target.generation
        progress.inflight_generation = generation
        progress.blocking_reason = "worker-in-flight"
        document = session.document
        view_state = session.view_state
        evaluator = getattr(getattr(self, "win", None), "operation_evaluator", None)
        stage_cache = None if evaluator is None else evaluator.stage_cache
        document_key = stage_document_key(document)

        def evaluate(token, sources=sources):
            return evaluate_level_evidence_snapshot(
                document,
                view_state,
                sources,
                pixel_limit=target.pixel_limit,
                cancellation_token=token,
                stage_cache=stage_cache,
                stage_document_key=document_key,
            )

        def release(owner) -> bool:
            owner_progress = getattr(owner, "semantic_level_evidence_progress", None)
            if owner_progress is None or owner_progress.inflight_generation != generation:
                return False
            owner_progress.inflight_generation = None
            return True

        def done(result):
            current = getattr(self, "_frame_session", None)
            current_target = None if current is None else getattr(current, "semantic_level_evidence_target", None)
            if current is not session or current_target is None or current_target.generation != generation:
                release(session)
                return
            release(current)
            tracker = self._montage_level_tracker()
            merged = 0
            for source in tuple(result.sources):
                source_index = int(source.source_index)
                if source.stats is None:
                    tracker.record_vacuous_source(current.level_key, source_index)
                else:
                    tracker.update_from_stats(current.level_key, source.stats, aggregate=False)
                    self._remember_montage_source_level_stats(current.level_key, source.stats)
                if progress.record_covered(source_index):
                    merged += 1
            self._semantic_level_evidence_last_merged = int(merged)
            self._last_montage_level_stats_ms = float(result.elapsed_ms)
            if len(progress.covered_sources) >= target.target_population:
                progress.blocking_reason = "ready"
            else:
                progress.blocking_reason = "waiting-semantic-sources"
            self._maybe_publish_after_level_evidence(current, processed=int(merged))
            self._schedule_semantic_level_evidence(current)
            if (
                len(progress.covered_sources) >= target.target_population
                and _montage_side_work_visible_settled(self, current)
            ):
                # Completion is the atomic refined levels+histogram edge.
                # Request it directly from the guarded worker completion;
                # there is no later payload transition guaranteed to wake the
                # presentation gate.
                self._request_level_metadata_presentation(current)

        def stale():
            if release(session):
                progress.blocking_reason = "superseded"

        def failed(exc):
            if release(session):
                progress.blocking_reason = f"error:{type(exc).__name__}"
            handle_ui_exception("semantic montage level evidence", exc)

        # An all-cached cursor slice still advances only one bounded GUI pass;
        # the no-op worker completion is the continuation for the next slice.
        max_items = max(1, len(sources))
        handle = self.win.kernel.submit_speculative_batch(
            kind="semantic-level-evidence",
            scope="montage:semantic-level-evidence",
            generation=generation,
            key=("semantic-level-evidence", generation, int(progress.cursor)),
            fn=evaluate,
            on_done=done,
            on_stale=stale,
            on_error=failed,
            priority=Priority.HISTOGRAM,
            lane=WorkLane.HISTOGRAM_REFINEMENT,
            max_items=max_items,
            pass_token=True,
        )
        if handle is None:
            release(session)
            progress.blocking_reason = "kernel-admission"

    def _schedule_montage_cached_level_stats(self, session) -> None:
        if self._first_pass_rough_evidence_closed(session):
            getattr(session, "pending_level_tiles", deque()).clear()
            getattr(session, "pending_level_sources", set()).clear()
            session.level_scan_remaining_tiles = 0
            self._schedule_semantic_level_evidence(session)
            return
        if (
            not getattr(session, "pending_level_tiles", None)
            and int(getattr(session, "level_scan_remaining_tiles", 0) or 0) <= 0
        ):
            self._schedule_semantic_level_evidence(session)
            return
        self._process_montage_cached_level_stats()

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
            quality = _rendered_level_evidence_quality_for_session(
                session,
                rendered,
                refined=bool(require_refined),
            )
            if source_index in queued_sources or tracker.has_source_quality(session.level_key, source_index, quality):
                continue
            if self._update_montage_level_bounds_from_prepared(
                session.level_key,
                rendered,
                expected_indices=expected,
                require_refined=require_refined,
                evidence_quality=quality,
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
            if bool(getattr(session, "shader_display", False)):
                payload_quality = str(getattr(rendered, "quality", "exact") or "exact")
                first_pass_quality = getattr(session, "first_pass_quality", None)
                if (
                    first_pass_quality is not None
                    and not session.first_pass_accepts_quality(payload_quality)
                ):
                    continue
            source_index = int(rendered.tile.source_index)
            quality = _rendered_level_evidence_quality_for_session(
                session,
                rendered,
                refined=bool(require_refined),
            )
            cached_stats = self._cached_montage_source_level_stats(
                session.level_key,
                source_index,
                quality,
            )
            if cached_stats is not None:
                tracker.update_from_stats(session.level_key, cached_stats, aggregate=False)
                queued_sources.discard(source_index)
                merged += 1
                continue
            if tracker.has_source_quality(session.level_key, source_index, quality):
                continue
            if self._update_montage_level_bounds_from_prepared(
                session.level_key,
                rendered,
                expected_indices=expected,
                require_refined=require_refined,
                evidence_quality=quality,
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

    def _take_montage_level_evidence_batch(
        self,
        session,
        *,
        expected,
        require_refined: bool,
        batch_limit: int = MONTAGE_LEVEL_STATS_BACKGROUND_BATCH,
    ) -> tuple[RenderedTile, ...]:
        scan_started = perf_counter()
        batch: list[RenderedTile] = []
        tracker = self._montage_level_tracker()
        pending = getattr(session, "pending_level_tiles", None)
        if pending is None:
            pending = deque()
            session.pending_level_tiles = pending
        pending_sources = getattr(session, "pending_level_sources", None)
        if pending_sources is None:
            pending_sources = {int(item.tile.source_index) for item in pending}
            session.pending_level_sources = pending_sources
        while (
            pending
            and len(batch) < int(batch_limit)
            and (
                not batch
                or (perf_counter() - scan_started) * 1000.0
                < MONTAGE_LEVEL_STATS_BACKGROUND_BUDGET_MS
            )
        ):
            rendered = pending.popleft()
            source_index = int(rendered.tile.source_index)
            pending_sources.discard(source_index)
            quality = _rendered_level_evidence_quality_for_session(
                session,
                rendered,
                refined=bool(require_refined),
            )
            cached_stats = self._cached_montage_source_level_stats(
                session.level_key,
                source_index,
                quality,
            )
            if cached_stats is not None:
                tracker.update_from_stats(session.level_key, cached_stats, aggregate=False)
                continue
            if tracker.has_source_quality(session.level_key, source_index, quality):
                continue
            batch.append(rendered)

        tile_count = len(getattr(getattr(session, "plan", None), "tiles", ()) or ())
        remaining = int(getattr(session, "level_scan_remaining_tiles", 0) or 0)
        if tile_count <= 0 or remaining <= 0:
            session.level_scan_remaining_tiles = 0
            self._montage_pending_level_tiles_last_session = len(pending or ())
            return tuple(batch)
        cursor = int(getattr(session, "level_scan_cursor", 0) or 0) % tile_count
        inspected = 0
        while (
            remaining > 0
            and len(batch) < int(batch_limit)
            and (
                inspected == 0
                or (perf_counter() - scan_started) * 1000.0
                < MONTAGE_LEVEL_STATS_BACKGROUND_BUDGET_MS
            )
        ):
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
            quality = _rendered_level_evidence_quality_for_session(
                session,
                rendered,
                refined=bool(require_refined),
            )
            if source_index in pending_sources or tracker.has_source_quality(session.level_key, source_index, quality):
                continue
            cached_stats = self._cached_montage_source_level_stats(
                session.level_key,
                source_index,
                quality,
            )
            if cached_stats is not None:
                tracker.update_from_stats(session.level_key, cached_stats, aggregate=False)
                continue
            if self._update_montage_level_bounds_from_prepared(
                session.level_key,
                rendered,
                expected_indices=expected,
                require_refined=require_refined,
                evidence_quality=quality,
            ):
                self._queue_montage_level_refinement(session, rendered)
            else:
                batch.append(rendered)
        session.level_scan_cursor = int(cursor)
        session.level_scan_remaining_tiles = max(0, int(remaining))
        self._montage_pending_level_tiles_last_session = len(pending or ())
        return tuple(batch)

    def _process_montage_cached_level_stats(self) -> None:
        session = getattr(self, "_frame_session", None)
        if not self._frame_session_is_current(session):
            return
        pending = getattr(session, "pending_level_tiles", None)
        if not pending and int(getattr(session, "level_scan_remaining_tiles", 0) or 0) <= 0:
            self._schedule_semantic_level_evidence(session)
            return
        if getattr(session, "level_evidence_inflight", False):
            return
        stats_start = perf_counter()
        expected = self._montage_level_expected_indices(session)
        require_refined = _montage_level_evidence_requires_refined(self, session)
        visible_level_dependency = bool(require_refined and not getattr(session, "display_committed", False))
        batch = self._take_montage_level_evidence_batch(
            session,
            expected=expected,
            require_refined=require_refined,
            batch_limit=(
                MONTAGE_LEVEL_STATS_FIRST_CPU_BATCH
                if visible_level_dependency
                else MONTAGE_LEVEL_STATS_BACKGROUND_BATCH
            ),
        )
        if not batch:
            self._last_montage_level_stats_ms = (perf_counter() - stats_start) * 1000.0
            if int(getattr(session, "level_scan_remaining_tiles", 0) or 0) > 0:
                self._invite_montage_level_evidence_continuation(session)
            else:
                # `_take_montage_level_evidence_batch` can finish the pass by
                # merging prepared per-payload stats inline, leaving no worker
                # batch. That is still a real evidence transition. Without
                # this publication, PyQtGraph's first frame remains parked at
                # TARGET_EMITTED forever whenever the final source takes this
                # fast path (profiling changes made the race deterministic).
                self._maybe_publish_after_level_evidence(session, processed=1)
            return
        generation = (
            session.key,
            int(session.session_id),
            int(getattr(session, "viewport_revision", 0) or 0),
            session.level_key,
            bool(require_refined),
        )
        session.level_evidence_inflight = True
        session.level_evidence_generation = generation

        def release_generation(current) -> bool:
            if getattr(current, "level_evidence_generation", None) != generation:
                return False
            current.level_evidence_inflight = False
            current.level_evidence_generation = None
            return True

        def evaluate(batch=batch, require_refined=require_refined):
            start = perf_counter()
            rows = []
            total_bytes = 0
            for rendered in batch:
                source_index = int(rendered.tile.source_index)
                quality = _rendered_level_evidence_quality_for_session(
                    session,
                    rendered,
                    refined=bool(require_refined),
                )
                stats = _sample_rendered_level_evidence(
                    rendered,
                    refined=bool(require_refined),
                    evidence_quality=quality,
                )
                rows.append((source_index, stats))
                total_bytes += montage_commit.rendered_tile_nbytes(rendered)
            return tuple(rows), (perf_counter() - start) * 1000.0, int(total_bytes)

        def done(result, batch=batch, generation=generation, expected=expected, require_refined=require_refined):
            rows, elapsed_ms, total_bytes = result
            # Worker results are immutable per-source evidence. A scroll may
            # supersede the presentation generation before this callback, but
            # the sampled source values remain reusable by overlapping/future
            # windows in the same semantic family. Cache them before the
            # presentation-generation guard; only live tracker/publication
            # mutation stays guarded below.
            for _source_index, stats in tuple(rows or ()):
                if stats is not None:
                    self._remember_montage_source_level_stats(generation[3], stats)
            current = getattr(self, "_frame_session", None)
            current_generation = None if current is None else (
                current.key,
                int(current.session_id),
                int(getattr(current, "viewport_revision", 0) or 0),
                current.level_key,
                bool(require_refined),
            )
            if current_generation != generation:
                reuse(result)
                release_generation(session)
                if current is session and current.level_key == generation[3]:
                    self._mark_montage_level_scan_pending(current)
                    self._schedule_montage_cached_level_stats(current)
                return
            release_generation(current)
            processed = 0
            for rendered, (source_index, stats) in zip(batch, tuple(rows or ())):
                source_index = int(source_index)
                quality = _rendered_level_evidence_quality_for_session(
                    current,
                    rendered,
                    refined=bool(require_refined),
                )
                if not self._montage_level_tracker().has_source_quality(current.level_key, source_index, quality):
                    if stats is not None:
                        self._montage_level_tracker().update_from_stats(current.level_key, stats, aggregate=False)
                        self._remember_montage_source_level_stats(current.level_key, stats)
                    elif require_refined:
                        self._montage_level_tracker().record_vacuous_source(current.level_key, source_index)
                self._queue_montage_level_refinement(current, rendered)
                processed += 1
            self._last_montage_level_stats_ms = float(elapsed_ms)
            self._montage_pending_level_tiles_last_session = len(getattr(current, "pending_level_tiles", ()) or ())
            if processed:
                _complete_inline_work(
                    self,
                    WorkItem(
                        key=(
                            "montage_level_evidence",
                            current.key,
                            int(current.session_id),
                            int(getattr(current, "level_revision", 0) or 0),
                            int(processed),
                        ),
                        lane=WorkLane.HISTOGRAM_REFINEMENT,
                        quality="retained",
                        supersession_key=("montage-level-evidence", current.key),
                        supersession_value=int(current.session_id),
                        estimated_cpu_ms=float(elapsed_ms or 0.0),
                        estimated_bytes=int(total_bytes),
                    ),
                )
            self._maybe_publish_after_level_evidence(current, processed=processed)
            self._schedule_montage_cached_level_stats(current)

        def reuse(result, generation=generation, require_refined=require_refined):
            """Retain immutable evidence even when its presentation was superseded."""

            rows, _elapsed_ms, _total_bytes = result
            for source_index, stats in tuple(rows or ()):
                if stats is not None:
                    self._remember_montage_source_level_stats(generation[3], stats)
            current = getattr(self, "_frame_session", None)
            if current is None or current.level_key != generation[3]:
                return
            tracker = self._montage_level_tracker()
            tracker.ensure_expected(
                current.level_key,
                self._montage_level_expected_indices(current),
            )
            processed = 0
            for source_index, stats in tuple(rows or ()):
                source_index = int(source_index)
                if stats is not None:
                    tracker.update_from_stats(current.level_key, stats, aggregate=False)
                elif require_refined:
                    tracker.record_vacuous_source(current.level_key, source_index)
                processed += 1
            self._maybe_publish_after_level_evidence(current, processed=processed)

        def stale(session=session, batch=batch):
            current = getattr(self, "_frame_session", None)
            release_generation(session)
            if current is session and current.level_key == generation[3]:
                # A retarget can supersede a running evidence batch while
                # retaining the same semantic level population. Completed
                # work arrives through ``reuse``; work dropped before running
                # is rediscovered by a fresh bounded scan of current payloads.
                self._mark_montage_level_scan_pending(current)
                self._schedule_montage_cached_level_stats(current)

        def failed(exc, session=session, batch=batch):
            stale(session=session, batch=batch)
            handle_ui_exception("montage level evidence", exc)

        task_key = ("montage_level_evidence", session.key, int(session.session_id), generation)
        if visible_level_dependency:
            handle = self.win.kernel.submit(
                TaskSpec(
                    key=task_key,
                    fn=evaluate,
                    lane=WorkLane.VISIBLE_MATERIALIZATION,
                    priority=Priority.VISIBLE_IMAGE,
                    scheduling_rank=UNRANKED_SCHEDULING_RANK,
                    scope=f"montage:{session.key!r}:histogram",
                    supersession=Supersession(("montage-level-evidence", session.key), generation),
                    reusable=True,
                    pass_token=False,
                ),
                on_done=done,
                on_stale=stale,
                on_reuse=reuse,
                on_error=failed,
            )
        else:
            handle = self.win.kernel.submit_speculative_batch(
                kind="montage-level-evidence",
                scope=f"montage:{session.key!r}:histogram",
                generation=generation,
                key=task_key,
                fn=evaluate,
                on_done=done,
                on_stale=stale,
                on_error=failed,
                priority=Priority.HISTOGRAM,
                lane=WorkLane.HISTOGRAM_REFINEMENT,
                max_items=len(batch),
            )
        if handle is None:
            if release_generation(session):
                self._requeue_montage_level_evidence(session, batch)

    def _requeue_montage_level_evidence(self, session, batch) -> None:
        pending, pending_sources = self._pending_montage_level_sources(session)
        for rendered in reversed(tuple(batch or ())):
            source_index = int(rendered.tile.source_index)
            if source_index in pending_sources:
                continue
            pending.appendleft(rendered)
            pending_sources.add(source_index)

    def _invite_montage_level_evidence_continuation(self, session) -> None:
        if getattr(session, "level_evidence_inflight", False):
            return
        generation = (
            session.key,
            int(session.session_id),
            int(getattr(session, "viewport_revision", 0) or 0),
            session.level_key,
        )
        session.level_evidence_inflight = True
        session.level_evidence_generation = generation

        def release_generation(current) -> bool:
            if getattr(current, "level_evidence_generation", None) != generation:
                return False
            current.level_evidence_inflight = False
            current.level_evidence_generation = None
            return True

        def done(_value=None, generation=generation):
            current = getattr(self, "_frame_session", None)
            current_generation = None if current is None else (
                current.key,
                int(current.session_id),
                int(getattr(current, "viewport_revision", 0) or 0),
                current.level_key,
            )
            if current_generation != generation:
                release_generation(session)
                if current is session and current.level_key == generation[3]:
                    self._process_montage_cached_level_stats()
                return
            release_generation(current)
            self._process_montage_cached_level_stats()

        def stale(session=session):
            current = getattr(self, "_frame_session", None)
            release_generation(session)
            if current is session and current.level_key == generation[3]:
                self._process_montage_cached_level_stats()

        def failed(exc, session=session):
            stale(session=session)
            handle_ui_exception("montage level evidence continuation", exc)

        task_key = ("montage_level_evidence_continuation", session.key, int(session.session_id), generation)
        visible_level_dependency = bool(
            _montage_level_evidence_requires_refined(self, session)
            and not getattr(session, "display_committed", False)
        )
        if visible_level_dependency:
            # This continuation advances the scan that gates PyQtGraph's very
            # first pixels. It is correctness work, not speculative histogram
            # refinement; parking it behind the optional-lane quota leaves the
            # app with histogram/ROI metadata but zero tiles forever.
            handle = self.win.kernel.submit(
                TaskSpec(
                    key=task_key,
                    fn=lambda: True,
                    lane=WorkLane.VISIBLE_MATERIALIZATION,
                    priority=Priority.VISIBLE_IMAGE,
                    scheduling_rank=UNRANKED_SCHEDULING_RANK,
                    scope=f"montage:{session.key!r}:histogram",
                    supersession=Supersession(("montage-level-evidence-continuation", session.key), generation),
                    pass_token=False,
                ),
                on_done=done,
                on_stale=stale,
                on_error=failed,
            )
        else:
            handle = self.win.kernel.submit_speculative_batch(
                kind="montage-level-evidence-continuation",
                scope=f"montage:{session.key!r}:histogram",
                generation=generation,
                key=task_key,
                fn=lambda: True,
                on_done=done,
                on_stale=stale,
                on_error=failed,
                priority=Priority.HISTOGRAM,
                lane=WorkLane.HISTOGRAM_REFINEMENT,
                max_items=1,
            )
        if handle is None:
            release_generation(session)

    def _maybe_publish_after_level_evidence(self, session, *, processed: int) -> None:
        pending = getattr(session, "pending_level_tiles", None)
        self._montage_pending_level_tiles_last_session = len(pending or ())
        if processed:
            self._publish_first_cpu_histogram(session)
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
        semantic_progress = getattr(session, "semantic_level_evidence_progress", None)
        semantic_remaining = bool(
            semantic_progress is not None
            and (
                semantic_progress.inflight_generation is not None
                or int(semantic_progress.pending_batches) > 0
            )
        )
        evidence_remaining = bool(
            pending
            or int(getattr(session, "level_scan_remaining_tiles", 0) or 0) > 0
            or semantic_remaining
        )
        flush_parked = bool(getattr(session, "flush_pending", False) or getattr(session, "final_commit_pending", False))
        can_resume_parked_flush = bool(flush_parked and not evidence_remaining)
        can_refresh_settled_metadata = bool(
            not flush_parked and _montage_side_work_visible_settled(self, session)
        )
        payload_backlog_clear = bool(
            not getattr(session, "dirty_payloads", ())
            and not getattr(session, "pending_payload_upserts", ())
            and not getattr(session, "pending_removals", ())
        )
        if processed and (
            can_resume_parked_flush
            or (payload_backlog_clear and can_refresh_settled_metadata)
        ):
            if bool(getattr(self, "_montage_commit_drain_active", False)):
                # The current presentation callback synchronously drained the
                # last evidence item. Posting the same receiver event from
                # inside its handler can leave the coalescing bit armed after
                # Qt consumes the event. Keep the semantic obligation on the
                # session; commit_pending_session's normal backlog rearm posts
                # exactly one continuation after this callback exits.
                session.flush_pending = True
                session.final_commit_pending = True
                self._montage_gate_last_backlog = None
                return
            pipeline = getattr(session, "pipeline", None)
            effects = None if pipeline is None else getattr(pipeline, "effects", None)
            request_presentation = None if effects is None else getattr(effects, "request_presentation", None)
            if not callable(request_presentation):
                raise RuntimeError("live frame session has no presentation effect gate")
            request_presentation()

    def _publish_first_cpu_histogram(self, session) -> bool:
        """Show semantic evidence while PyQtGraph still waits for final levels."""

        if bool(getattr(session, "display_committed", False)):
            return False
        win = getattr(self, "win", None)
        image_view = None if win is None else getattr(win, "img_view", None)
        if image_view is None or image_view_backend_capabilities(image_view).shader_windowing:
            return False
        summary = self._montage_level_tracker().summary_for(session.level_key)
        if (
            summary is None
            or summary.rank != LevelSourceRank.MONTAGE_SAMPLED_FULL
            or not self._should_publish_montage_level_metadata(session, summary)
        ):
            return False
        source = self._montage_level_source_for_session(session, allow_partial=True)
        histogram = self._montage_histogram_plot_data_for_session(session, allow_partial=True)
        publish = getattr(image_view, "applyHistogramMetadata", None)
        if source is None or histogram is None or not callable(publish):
            return False
        if not publish(
            histogramData=histogram,
            histogramPlotData=histogram,
            levels=source.levels,
            histogramRange=source.histogram_range,
        ):
            return False
        note = getattr(self, "_note_montage_level_source_applied", None)
        if callable(note):
            note(session, source, explicit=False)
        self._montage_first_cpu_histogram_publications = int(
            getattr(self, "_montage_first_cpu_histogram_publications", 0) or 0
        ) + 1
        return True

    def _schedule_montage_refined_level_stats(self, session) -> None:
        if not self._frame_session_is_current(session):
            return
        pending = getattr(session, "pending_refined_level_tiles", None)
        if not pending:
            return
        if not _montage_side_work_visible_settled(self, session):
            return
        batch = []
        while pending and len(batch) < 4:
            rendered = pending.popleft()
            source_index = int(rendered.tile.source_index)
            quality = _rendered_level_evidence_quality_for_session(session, rendered, refined=True)
            if self._montage_level_tracker().has_source_quality(session.level_key, source_index, quality):
                pending_sources = getattr(session, "pending_refined_level_sources", None)
                if pending_sources is not None:
                    pending_sources.discard(source_index)
                continue
            source = render_effects.montage_refined_level_values(rendered)
            batch.append((source_index, source, rendered))

        if not batch:
            return

        generation = (
            session.key,
            int(session.session_id),
            int(getattr(session, "viewport_revision", 0) or 0),
            session.level_key,
        )
        sources = tuple(int(source_index) for source_index, _source, _rendered in batch)

        def evaluate(batch=batch):
            return tuple(
                (
                    int(source_index),
                    sample_tile_level_stats(
                        source,
                        int(source_index),
                        refined=True,
                        evidence_quality=LevelEvidenceQuality.REFINED,
                    ),
                )
                for source_index, source, _rendered in batch
            )

        def done(
            rows,
            session_id=session.session_id,
            session_key=session.key,
            level_key=session.level_key,
            generation=generation,
        ):
            current = getattr(self, "_frame_session", None)
            current_generation = None if current is None else (
                current.key,
                int(current.session_id),
                int(getattr(current, "viewport_revision", 0) or 0),
                current.level_key,
            )
            if current_generation != generation:
                return
            metadata_improved = False
            for source_index, stats in tuple(rows or ()):
                metadata_improved = bool(self._on_montage_refined_level_stats_done(
                    session_id,
                    session_key,
                    level_key,
                    int(source_index),
                    stats,
                    schedule_next=False,
                )) or metadata_improved
            self._schedule_montage_refined_level_stats(current)
            if metadata_improved and not getattr(current, "pending_refined_level_tiles", None):
                self._request_level_metadata_presentation(current)

        handle = self.win.kernel.submit_speculative_batch(
            kind="montage-refined-level-stats",
            scope=f"montage:{session.key!r}:histogram",
            generation=generation,
            key=("montage_refined_level_stats", session.key, int(session.session_id), sources),
            fn=evaluate,
            on_done=done,
            on_stale=lambda: None,
            on_error=lambda exc: handle_ui_exception("montage refined level stats", exc),
            priority=Priority.HISTOGRAM,
            lane=WorkLane.HISTOGRAM_REFINEMENT,
            max_items=len(batch),
        )
        if handle is None:
            for _source_index, _source, rendered in reversed(batch):
                pending.appendleft(rendered)

    def _on_montage_refined_level_stats_done(
        self,
        session_id,
        session_key,
        level_key,
        source_index,
        stats,
        *,
        schedule_next: bool = True,
    ) -> bool:
        session = getattr(self, "_frame_session", None)
        if session is None or not self._is_current_frame_session(session_id, session_key):
            return False
        pending_sources = getattr(session, "pending_refined_level_sources", None)
        if pending_sources is not None:
            pending_sources.discard(int(source_index))
        metadata_improved = False
        if stats is not None:
            self._montage_level_tracker().update_from_stats(level_key, stats, aggregate=False)
            summary = self._montage_level_tracker().summary_for(level_key)
            first_cpu_frame_ready = bool(
                summary is not None
                and not session.display_committed
                and _montage_level_evidence_requires_refined(self, session)
                and bool(getattr(summary, "refined", False))
                and getattr(summary, "rank", None) == LevelSourceRank.MONTAGE_SAMPLED_FULL
            )
            metadata_improved = bool(
                first_cpu_frame_ready
                or (
                    summary is not None
                    and session.display_committed
                    and _montage_side_work_visible_settled(self, session)
                    and getattr(session, "user_levels_override", None) is None
                    and self._should_publish_montage_level_metadata(session, summary)
                )
            )
        if schedule_next:
            self._schedule_montage_refined_level_stats(session)
            if metadata_improved and not getattr(session, "pending_refined_level_tiles", None):
                self._request_level_metadata_presentation(session)
        return metadata_improved

    @staticmethod
    def _request_level_metadata_presentation(session) -> None:
        pipeline = getattr(session, "pipeline", None)
        effects = None if pipeline is None else getattr(pipeline, "effects", None)
        request_presentation = None if effects is None else getattr(effects, "request_presentation", None)
        if not callable(request_presentation):
            raise RuntimeError("live frame session has no presentation effect gate")
        request_presentation()


def _montage_side_work_visible_settled(renderer, session) -> bool:
    kernel = getattr(getattr(renderer, "win", None), "kernel", None)
    pixels_settled = False
    required_settled = getattr(session, "required_target_settled", None)
    if not callable(required_settled):
        raise RuntimeError("live frame session has no required-tile owner")
    pixels_settled = bool(required_settled())
    return bool(
        kernel is not None
        and int(getattr(kernel, "visible_backlog", 0) or 0) <= 0
        and pixels_settled
        and not getattr(session, "dirty_payloads", None)
        and not getattr(session, "pending_payload_upserts", None)
        and not getattr(session, "pending_removals", None)
        and not bool(getattr(session, "flush_pending", False))
        and not bool(getattr(session, "final_commit_pending", False))
    )



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


def _rendered_level_evidence_quality(rendered, *, refined: bool, evidence_quality=None) -> LevelEvidenceQuality:
    if evidence_quality is not None:
        return LevelEvidenceQuality(int(evidence_quality))
    if bool(refined) and not _rendered_tile_is_preview(rendered):
        return LevelEvidenceQuality.REFINED
    stats = getattr(rendered, "level_stats", None)
    if stats is not None and bool(getattr(stats, "refined", False)) and not _rendered_tile_is_preview(rendered):
        return LevelEvidenceQuality.REFINED
    if _rendered_tile_is_preview(rendered):
        return LevelEvidenceQuality.ROUGH_PREVIEW
    return LevelEvidenceQuality.ROUGH_TARGET


def _rendered_level_evidence_quality_for_session(session, rendered, *, refined: bool) -> LevelEvidenceQuality:
    quality = _rendered_level_evidence_quality(rendered, refined=bool(refined))
    if quality != LevelEvidenceQuality.ROUGH_PREVIEW:
        return quality
    demand = getattr(getattr(session, "lod_policy_decision", None), "demand", None)
    desired = int(getattr(demand, "desired_level", 0) or 0)
    lod_level = int(getattr(getattr(rendered, "lod", None), "level", desired + 1) or 0)
    if lod_level <= desired:
        return LevelEvidenceQuality.REFINED if bool(refined) else LevelEvidenceQuality.ROUGH_TARGET
    return quality


def _sample_rendered_level_evidence(rendered, *, refined: bool, evidence_quality=None):
    source_index = int(rendered.tile.source_index)
    quality = _rendered_level_evidence_quality(
        rendered,
        refined=bool(refined),
        evidence_quality=evidence_quality,
    )
    level_stats = getattr(rendered, "level_stats", None)
    if level_stats is not None and (not refined or bool(getattr(level_stats, "refined", False))):
        return tile_level_stats_with_quality(
            level_stats,
            quality,
            source_index=int(rendered.tile.source_index),
        )
    if not refined:
        level_data = getattr(rendered, "level_data", None)
        if level_data is not None:
            stats = provisional_tile_level_stats(level_data, source_index, evidence_quality=quality)
            if stats is not None:
                return stats
    return sample_tile_level_stats(
        render_effects.montage_refined_level_values(rendered),
        source_index,
        refined=bool(refined),
        evidence_quality=quality,
    )


def _montage_level_family_key(level_key):
    """Level identity without the selected montage population.

    Per-source evidence is valid across overlapping index windows; only the
    aggregate population changes. Keep document/pipeline, channel/axes and
    montage axis semantics while removing selected indices/column layout.
    """

    try:
        marker, document_key, scope_state, axis = level_key
        family_state = scope_state.with_montage_axis(
            axis,
            columns=None,
            indices=None,
            text=None,
        )
        return marker, document_key, family_state, axis
    except Exception:
        return level_key


def _interactive_active(window) -> bool:
    coordinator = getattr(window.win, "render_coordinator", None)
    return bool(
        coordinator is not None and getattr(coordinator, "interactive_active", False)
        or _viewport_interaction_active(window)
    )


def _montage_level_evidence_requires_refined(window, session) -> bool:
    requested_levels = normalize_bounds(getattr(session, "user_levels_override", None))
    cpu_windowed = not image_view_backend_capabilities(window.win.img_view).shader_windowing
    return bool(
        cpu_windowed
        and (
            not bool(getattr(session, "display_committed", False))
            or (requested_levels is None and getattr(session, "force_auto", False))
        )
    )


def _viewport_interaction_active(window) -> bool:
    return bool(getattr(window.win, "_viewport_interaction_active", False))
