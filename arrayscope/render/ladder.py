"""The unified LOD ladder: one owner for per-tile quality progression.

Pre-redesign, "what quality should this tile get next" was answered in at
least four places: `render.lod.plan_materialization` (desired LOD),
the preview/floor methods on `FrameRenderMixin` (Plans 04/05), ingest
reduction admission, and native-only policy checks. They cooperated through
shared mutable session state and could disagree — the source of the
stall/loop defect class documented in ADR 0051.

Here the ladder is a *pure planner*: given immutable per-tile state and the
viewport demand, it returns the ordered rung steps a tile still needs. It
holds no collections, performs no I/O, and never mutates lifecycle state —
`TileLifecycle` remains the single owner of tile state; the pipeline turns
steps into kernel tasks and lifecycle claims.

Rungs (roadmap "unified LOD ladder", ADR 0050/0052 lineage):

    FLOOR    retained coarse preview — cheapest first pixels, survives
             retargets; never blocks on exact work.
    PREVIEW  first display rung at reduced input; operations run once per
             rung — commuting ops evaluate on reduced input directly.
    DESIRED  refinement to the demanded display level.
    EXACT    native resolution. Exact inspection values are ALWAYS computed
             from native data regardless of displayed rung (policy
             constraint: display LOD never contaminates inspection).

Invariants:

1. Steps are ordered coarse→fine; a finer rung never runs before a coarser
   rung that is still useful for first pixels.
2. A rung is skipped when an acknowledged resident level already satisfies
   it (no re-computation on retarget — resident levels come from lifecycle
   claims, not "we probably uploaded it").
3. Level values derived from preview rungs stay authoritative until a
   coordinated refinement pass replaces them (`levels_authoritative_rung`).
4. Native-only policy collapses the ladder to a single EXACT step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from arrayscope.display.lod import LodDemand
from arrayscope.kernel.task import Lane, Priority


class Rung(IntEnum):
    FLOOR = 0
    PREVIEW = 1
    DESIRED = 2
    EXACT = 3


@dataclass(frozen=True)
class TileLodState:
    """Immutable per-tile snapshot the ladder plans from.

    ``resident_levels`` are backend-acknowledged levels (lifecycle claims),
    ``presented_level`` is the currently presented level (None = placeholder
    still visible), ``floor_available`` means a retained floor payload can be
    committed without new evaluation.
    """

    tile_number: int
    resident_levels: tuple[int, ...] = ()
    presented_level: int | None = None
    floor_available: bool = False
    exact_requested: bool = False  # inspection or user demanded native


@dataclass(frozen=True)
class RungStep:
    """One planned unit of quality progression for one tile."""

    tile_number: int
    rung: Rung
    level: int
    # Level to reduce from when reduced input is available with reduction (None = evaluate
    # the full operation pipeline at native, then reduce the output).
    reduce_from_native: bool
    lane: Lane
    priority: Priority
    reason: str


@dataclass(frozen=True)
class LadderPolicy:
    """Tunable ladder policy. Defaults follow ADR 0050 + Plan 04/05 landings.

    TODO(redesign R3): re-derive `preview_level` bounds and the DESIRED
    priority from fresh A/B evidence (roadmap X5 queue item 2) before
    changing defaults; ADR 0046 forbids theory-driven tuning.
    """

    mode: str = "resident"  # "resident" | "native-only"
    floor_level: int = 4
    preview_level: int = 2
    reduced_input_available: bool = True
    levels_authoritative_rung: Rung = Rung.PREVIEW


class LodLadder:
    """Pure rung planner. Construct once per policy; plan per tile+demand."""

    def __init__(self, policy: LadderPolicy | None = None) -> None:
        self.policy = policy or LadderPolicy()

    # ------------------------------------------------------------ planning

    def plan_tile(self, state: TileLodState, demand: LodDemand) -> tuple[RungStep, ...]:
        """Return the ordered steps ``state`` still needs to satisfy ``demand``.

        An empty result means the tile is converged for this demand.
        """

        policy = self.policy
        if policy.mode == "native-only":
            return self._native_only_plan(state)

        desired = max(0, int(demand.desired_level))
        acceptable = tuple(demand.acceptable_levels or (desired,))
        steps: list[RungStep] = []

        presented = state.presented_level
        resident = frozenset(int(level) for level in state.resident_levels)

        def finest_available() -> float:
            """Finest (lowest) level that exists or is already planned.

            "Exists" means committable: presented on screen, acknowledged
            resident, or produced by an earlier step of this same plan.
            Committing an existing resident level is the pipeline's job and
            needs no compute step here.
            """

            candidates = [float(level) for level in resident]
            if presented is not None:
                candidates.append(float(presented))
            candidates.extend(float(step.level) for step in steps)
            return min(candidates) if candidates else float("inf")

        # Pre-native rungs are planned only when they are actually cheap:
        # reduced-input-capable pipelines evaluate on reduced input directly, and a
        # retained floor plane commits with no evaluation at all. For
        # opaque pipelines (FFT and friends) a per-tile "preview" would cost a
        # FULL native evaluation per tile just to throw resolution away —
        # twice the work of going native once. Their cheap first pass is the
        # shared transform-preview queue, not a per-tile native evaluation.
        cheap_pre_native = policy.reduced_input_available or state.floor_available

        # 1) FLOOR — only while the tile has nothing committable at all.
        if presented is None and not resident and cheap_pre_native:
            floor_level = max(policy.floor_level, desired)
            steps.append(
                RungStep(
                    tile_number=state.tile_number,
                    rung=Rung.FLOOR,
                    level=floor_level,
                    reduce_from_native=False,
                    lane=Lane.DISPLAY_PREVIEW,
                    priority=Priority.INTERACTIVE,
                    reason=(
                        "retained floor commit" if state.floor_available else "cold floor fill"
                    ),
                )
            )

        # 2) PREVIEW — the first *display-quality* rung; skipped when
        # something at least as fine already exists or DESIRED covers it.
        preview_level = max(policy.preview_level, desired)
        if (
            policy.reduced_input_available
            and preview_level < finest_available()
            and preview_level != desired
        ):
            steps.append(
                RungStep(
                    tile_number=state.tile_number,
                    rung=Rung.PREVIEW,
                    level=preview_level,
                    reduce_from_native=False,
                    lane=Lane.DISPLAY_PREVIEW,
                    priority=Priority.VISIBLE_IMAGE,
                    reason="preview rung (reduced-input display)",
                )
            )

        # 3) DESIRED — converge the displayed level to the demanded level.
        # Native/exact content is a valid source for reduction, but it is
        # not display-level convergence when the viewport demands a coarser
        # resident level; otherwise a tile can sit at applied factor 1 while
        # demand asks for factor 4 with no wakeup left to demote it.
        desired_resident = desired in resident or presented == desired
        if not desired_resident and (desired > 0 or desired < finest_available()):
            steps.append(
                RungStep(
                    tile_number=state.tile_number,
                    rung=Rung.DESIRED,
                    level=desired,
                    reduce_from_native=not policy.reduced_input_available,
                    lane=Lane.DISPLAY_PREPARATION,
                    priority=(
                        Priority.VISIBLE_IMAGE
                        if presented is None or presented not in acceptable
                        else Priority.HOVER
                    ),
                    reason=str(demand.reason or "desired display level"),
                )
            )

        # 4) EXACT — only on explicit request (inspection) or when the
        # demand itself is native, and nothing native exists yet.
        if (state.exact_requested or desired == 0) and finest_available() > 0:
            steps.append(
                RungStep(
                    tile_number=state.tile_number,
                    rung=Rung.EXACT,
                    level=0,
                    reduce_from_native=True,
                    lane=Lane.VISIBLE_MATERIALIZATION,
                    priority=Priority.VISIBLE_IMAGE if desired == 0 else Priority.HOVER,
                    reason="exact/native rung",
                )
            )

        return tuple(steps)

    def plan(self, states, demand: LodDemand) -> tuple[RungStep, ...]:
        """Plan every tile, coarse rungs across tiles before fine rungs.

        Cross-tile ordering matters for perceived progress: every visible
        tile should reach FLOOR/PREVIEW before any tile spends budget on
        DESIRED/EXACT (Plan 05 floor-first-fill, generalized).
        """

        per_tile = [self.plan_tile(state, demand) for state in states]
        ordered: list[RungStep] = []
        for rung in Rung:
            for steps in per_tile:
                ordered.extend(step for step in steps if step.rung == rung)
        return tuple(ordered)

    # ------------------------------------------------------------- helpers

    def _native_only_plan(self, state: TileLodState) -> tuple[RungStep, ...]:
        if state.presented_level == 0 or 0 in set(state.resident_levels):
            return ()
        return (
            RungStep(
                tile_number=state.tile_number,
                rung=Rung.EXACT,
                level=0,
                reduce_from_native=True,
                lane=Lane.VISIBLE_MATERIALIZATION,
                priority=Priority.VISIBLE_IMAGE,
                reason="native-only policy",
            ),
        )


__all__ = ["LadderPolicy", "LodLadder", "Rung", "RungStep", "TileLodState"]
