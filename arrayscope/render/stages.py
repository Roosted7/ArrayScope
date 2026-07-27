"""Typed boundaries between rendering pipeline stages.

Every stage consumes and produces exactly these types; a stage never reaches
into another stage's internals. This is the "modular chunks with well-defined
task and state" contract:

    RenderIntent -> plan -> TileWork* -> materialize/reduce (kernel tasks)
                 -> CommitBatch -> apply (GUI/GPU gateway) -> AckExpectation
                 -> TileLifecycle events

Data only: no Qt, no numpy payload manipulation, no scheduling decisions.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from arrayscope.kernel.task import Lane, Priority
from arrayscope.render.ladder import Rung


@dataclass(frozen=True)
class RenderIntent:
    """What the user currently means: one semantic target + one viewport.

    Immutable snapshot taken on the GUI thread at interaction time. Every
    downstream task carries the keys, so staleness checks never re-derive
    them from live state.
    """

    semantic_key: object
    viewport_key: object
    presentation_key: object
    view_range: tuple | None
    viewport_shape: tuple[int, int] | None
    interactive: bool = False
    render_round_id: str = ""
    round_preview_level: int | None = None
    round_target_level: int | None = None
    tile_source_ids: tuple[tuple[int, object], ...] = ()
    tile_source_indices: tuple[tuple[int, int], ...] = ()

    def source_id_for_tile(self, tile_number: int):
        tile_number = int(tile_number)
        for candidate, source_id in self.tile_source_ids:
            if int(candidate) == tile_number:
                return source_id
        return None

    def source_index_for_tile(self, tile_number: int) -> int | None:
        tile_number = int(tile_number)
        for candidate, source_index in self.tile_source_indices:
            if int(candidate) == tile_number:
                return int(source_index)
        return None

    @staticmethod
    def semantic_key_for_montage(document_key, view_state, viewport_plan, colormap_lut):
        """Return the montage session key used for pipeline supersession."""

        from arrayscope.window.montage_viewport import frame_session_key

        return frame_session_key(document_key, view_state, viewport_plan, colormap_lut)


@dataclass(frozen=True)
class LodAdmissionScope:
    """Current viewport-owned scope for LOD admission.

    Visible tile numbers are the only tiles allowed onto visible LOD lanes.
    Coverage/near tiles are carried for later speculative residency and
    prefetch decisions, after the kernel has no visible backlog.
    """

    visible_tile_numbers: frozenset[int] = field(default_factory=frozenset)
    coverage_tile_numbers: frozenset[int] = field(default_factory=frozenset)
    near_tile_numbers: frozenset[int] = field(default_factory=frozenset)
    viewport_key: object = None
    interactive: bool = False
    visible_missing_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "visible_tile_numbers",
            frozenset(int(tile) for tile in tuple(self.visible_tile_numbers or ())),
        )
        object.__setattr__(
            self,
            "coverage_tile_numbers",
            frozenset(int(tile) for tile in tuple(self.coverage_tile_numbers or ())),
        )
        object.__setattr__(
            self,
            "near_tile_numbers",
            frozenset(int(tile) for tile in tuple(self.near_tile_numbers or ())),
        )
        object.__setattr__(
            self, "visible_missing_count", max(0, int(self.visible_missing_count or 0))
        )


@dataclass(frozen=True)
class TileWork:
    """One schedulable unit of tile work, ready for kernel submission."""

    tile_number: int
    kind: str  # "materialize" | "reduce" | "stats" | "upload_prep"
    level: int
    lane: Lane
    priority: Priority
    key: object = None
    deps: tuple = ()
    estimated_bytes: int = 0
    payload: Any = None  # stage-specific request object (kept opaque here)


@dataclass(frozen=True)
class CommitBatch:
    """A bounded set of ready tile results for one GUI/GPU application.

    The apply step must respect the batch bounds; if it cannot, it commits a
    prefix and the pipeline re-batches the remainder. Placeholders/dirty
    state clear only on acknowledgement (ADR 0051), never on batch emission.
    """

    semantic_key: object
    presentation_key: object
    upserts: tuple = ()
    releases: tuple = ()
    level_metadata: Any = None
    max_items: int = 8
    max_bytes: int = 0

    @property
    def empty(self) -> bool:
        return not self.upserts and not self.releases


@dataclass(frozen=True)
class AckExpectation:
    """What the backend must acknowledge before lifecycle may advance.

    Identity-aware: slots are compared by emitted payload identity, so a
    late/foreign report can never satisfy a newer commit (ADR 0051 X5b).
    """

    semantic_key: object
    presentation_key: object
    slot_identities: tuple = ()
    committed_at_ns: int = 0


@dataclass
class PipelineCounters:
    """Deterministic counters; the only mutable state in this module."""

    intents: int = 0
    ladder_plans: int = 0
    tasks_submitted: int = 0
    interactive_native_deferred: int = 0
    commit_batches: int = 0
    acks_confirmed: int = 0
    acks_rejected_stale: int = 0

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


class RungEvaluationTimings:
    """Evaluation call count and wall time per ``(rung, level)``.

    The claim this exists to settle is "reduced-input evaluation is ~16x
    cheaper for operation pipelines".  Wall-clock cannot show it — this
    machine's raw montage stage spreads 4.0-4.9 s run to run — so the cost of
    each rung at each level has to be counted directly.

    Worker threads record; the GUI thread reads.  One short lock and three
    integer dict bumps per *evaluation* (not per tile row, page, or texel), so
    a 272-tile montage pays ~600 of them against multi-millisecond
    evaluations.
    """

    __slots__ = ("_calls", "_discarded", "_lock", "_max_ns", "_total_ns")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[tuple[int, int], int] = {}
        self._total_ns: dict[tuple[int, int], int] = {}
        self._max_ns: dict[tuple[int, int], int] = {}
        self._discarded: dict[tuple[int, int], int] = {}

    def record(self, rung: int, level: int, elapsed_ns: int) -> None:
        """Account one finished evaluation, however it ended.

        Cancelled and failed evaluations are recorded too: work spent and
        thrown away is exactly what a preview-rung audit must see.
        """

        bucket = (int(rung), int(level))
        elapsed_ns = max(0, int(elapsed_ns))
        with self._lock:
            self._calls[bucket] = self._calls.get(bucket, 0) + 1
            self._total_ns[bucket] = self._total_ns.get(bucket, 0) + elapsed_ns
            if elapsed_ns > self._max_ns.get(bucket, 0):
                self._max_ns[bucket] = elapsed_ns

    def record_discarded(self, rung: int, level: int) -> None:
        """Account one evaluation whose *display payload* was never committed.

        Counted, not timed: the discard is learned on the GUI thread when the
        result arrives too late to commit, by which point the worker's own
        elapsed reading is gone.  Price it as ``discarded * total_ms / calls``
        from the same row — on the 272-tile FFT montage that reads 8 discarded
        level-1 evaluations against a ~950 ms mean, i.e. ~7.6 s of a 3.9 s
        stage.

        **A discard is not proof of waste, and this counter must not be read as
        one.** It says a payload did not reach the screen; it says nothing about
        whether the evaluation was on the critical path or whether removing it
        would make anything faster.  Measured on exactly that FFT montage: a
        configuration doing 9.7 s *less* total evaluation (15.5 s -> 5.9 s) with
        zero discards finished **1.35 s slower**, because this pipeline is
        serialization-bound on the presentation path, not worker-bound.  See
        `docs/redesign/discarded-rung-evaluation-2026-07-26.md` §6.
        """

        bucket = (int(rung), int(level))
        with self._lock:
            self._discarded[bucket] = self._discarded.get(bucket, 0) + 1

    def rows(self) -> tuple[dict[str, object], ...]:
        """One row per observed ``(rung, level)``, coarse rungs first."""

        with self._lock:
            buckets = sorted(set(self._calls) | set(self._discarded))
            calls = dict(self._calls)
            totals = dict(self._total_ns)
            maxima = dict(self._max_ns)
            discarded = dict(self._discarded)
        return tuple(
            {
                "rung": int(rung),
                "rung_name": Rung(rung).name.lower() if rung in _RUNG_VALUES else str(rung),
                "level": int(level),
                "calls": int(calls.get((rung, level), 0)),
                "discarded": int(discarded.get((rung, level), 0)),
                "total_ms": totals.get((rung, level), 0) / 1e6,
                "max_ms": maxima.get((rung, level), 0) / 1e6,
            }
            for rung, level in buckets
        )


_RUNG_VALUES = frozenset(int(rung) for rung in Rung)


__all__ = [
    "AckExpectation",
    "CommitBatch",
    "LodAdmissionScope",
    "PipelineCounters",
    "RenderIntent",
    "RungEvaluationTimings",
    "TileWork",
]
