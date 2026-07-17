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

from dataclasses import dataclass, field
from typing import Any

from arrayscope.kernel.task import Lane, Priority


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
        object.__setattr__(self, "visible_missing_count", max(0, int(self.visible_missing_count or 0)))


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


__all__ = [
    "AckExpectation",
    "CommitBatch",
    "LodAdmissionScope",
    "PipelineCounters",
    "RenderIntent",
    "TileWork",
]
