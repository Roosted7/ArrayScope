"""The unified LOD ladder: one owner for per-tile quality progression.

Pre-redesign, "what quality should this tile get next" was answered in at
least four places: `render.lod.plan_materialization` (desired LOD),
the preview/floor methods on `FrameControllerMixin` (Plans 04/05), ingest
reduction admission, and native-only policy checks. They cooperated through
shared mutable session state and could disagree — the source of the
stall/loop defect class documented in ADR 0051.

Here the ladder is a *pure planner*: given immutable per-tile state and the
viewport demand, it returns the ordered rung steps a tile still needs. It
holds no collections, performs no I/O, and never mutates lifecycle state —
`TileLifecycle` remains the single owner of tile state; the pipeline turns
steps into kernel tasks and lifecycle claims.

Rungs (roadmap "unified LOD ladder", ADR 0050/0052/0059 lineage):

    FLOOR    the one coarse preview rung. Its level is chosen for retention;
             it evaluates reduced display input when that route is available.
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
from arrayscope.render.progressive_scheduling import SchedulingVerdict

COARSE_RUNG_ENABLED_DEFAULT = True
COARSE_RUNG_MIN_LEVEL_DELTA = 2
COARSE_RUNG_MIN_SCREEN_PIXELS_PER_TEXEL = 3.0
COARSE_RUNG_MAX_SCREEN_PIXELS_PER_TEXEL = 6.0


def coarse_rung_level(
    *,
    demand: LodDemand,
    retention_level: int,
) -> int:
    """Return the preview level required by live screen scale and target quality.

    Two LOD levels are four times coarser per axis and sixteen times fewer
    texels. Within that invariant, the preview texel footprint follows the
    continuous viewport scale: one preview texel spans 3–6 screen pixels on
    the dominant axis. A retained level is only reusable when it remains
    inside that screen-space ceiling; tile count never decides image quality.
    """

    desired = max(0, int(demand.desired_level))
    retention = max(0, int(retention_level))
    level = desired + COARSE_RUNG_MIN_LEVEL_DELTA
    source_texels_per_pixel = max(
        (float(value) for value in demand.source_texels_per_pixel_xy),
        default=0.0,
    )
    if source_texels_per_pixel > 0.0:
        while (
            (2**level) / source_texels_per_pixel
            < COARSE_RUNG_MIN_SCREEN_PIXELS_PER_TEXEL
        ):
            level += 1
        retention_footprint = (2**retention) / source_texels_per_pixel
        if (
            retention > level
            and retention_footprint <= COARSE_RUNG_MAX_SCREEN_PIXELS_PER_TEXEL
        ):
            level = retention
    return level


class Rung(IntEnum):
    FLOOR = 0
    DESIRED = 2
    EXACT = 3


@dataclass(frozen=True)
class TileLodState:
    """Immutable per-tile snapshot the ladder plans from.

    ``resident_levels`` are backend-acknowledged levels (lifecycle claims),
    ``presented_level`` is the currently presented level (None = placeholder
    still visible), ``ready_level`` is lifecycle-owned materialized payload
    awaiting backend acknowledgement, and ``floor_available`` means a retained
    floor payload can be committed without new evaluation.
    """

    tile_number: int
    resident_levels: tuple[int, ...] = ()
    presented_level: int | None = None
    ready_level: int | None = None
    ready_quality: str = ""
    floor_available: bool = False
    presented_quality: str = "exact"
    current_presentation_quality: str = "exact"
    allow_preview: bool = True
    target_quality_available: bool = False
    exact_requested: bool = False  # inspection or user demanded native
    scheduling_rank: int = 0


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
    scheduling_rank: int = 0


@dataclass(frozen=True)
class LadderPolicy:
    """Tunable ladder policy. Defaults follow ADR 0050 + Plan 04/05 landings.

    TODO(redesign R3): re-derive coarse-level bounds and the DESIRED
    priority from fresh A/B evidence (roadmap X5 queue item 2) before
    changing defaults; ADR 0046 forbids theory-driven tuning.
    """

    mode: str = "resident"  # "resident" | "native-only"
    floor_level: int = 4
    reduced_input_available: bool = True
    coarse_rung_enabled: bool = COARSE_RUNG_ENABLED_DEFAULT
    levels_authoritative_rung: Rung = Rung.FLOOR


#: Why a tile got no coarse FLOOR rung, in the order `plan_tile`
#: decides it.  Interned constants, not f-strings: the refusal is asked once per
#: tile per replan (272 tiles x ~25 replans on a cold montage fill), so the
#: answer must cost a few comparisons and no allocation.  The numbers that go
#: with a verdict travel beside it, once per plan, not once per tile.
COARSE_RUNG_NATIVE_ONLY = "native-only policy: no coarse rung exists"
COARSE_RUNG_DISABLED = "coarse rung disabled by measured delivery policy"
COARSE_RUNG_PREVIEW_NOT_ALLOWED = "allow_preview false: tile is covered or too few missing"
COARSE_RUNG_NO_REDUCED_INPUT = "no reduced input and no retained floor"
COARSE_RUNG_ALREADY_COVERED = "tile already has committable coverage"
COARSE_RUNG_LANE_NOT_ADMITTED = "scheduling verdict does not admit the coarse lane"
COARSE_RUNG_PLANNED = ""


class LodLadder:
    """Pure rung planner. Construct once per policy; plan per tile+demand."""

    def __init__(self, policy: LadderPolicy | None = None) -> None:
        self.policy = policy or LadderPolicy()

    # ------------------------------------------------------------ reporting

    def coarse_rung_refusal(
        self,
        state: TileLodState,
        demand: LodDemand,
        verdict: SchedulingVerdict | None = None,
    ) -> str:
        """Why this tile gets no FLOOR step, or `""` if it gets one.

        "The ladder planned no coarse rung" is otherwise an absence, and an
        absence names no cause: the 2026-07-26 preview-LOD work attributed a
        missing FFT preview to `allow_preview` by reading code, and the
        attribution was wrong.  This answers the question from the same
        expressions `plan_tile` decides with, evaluated in the same order.
        """

        policy = self.policy
        if policy.mode == "native-only":
            return COARSE_RUNG_NATIVE_ONLY
        if not policy.coarse_rung_enabled:
            return COARSE_RUNG_DISABLED
        if not bool(state.allow_preview):
            return COARSE_RUNG_PREVIEW_NOT_ALLOWED
        if not (policy.reduced_input_available or state.floor_available):
            return COARSE_RUNG_NO_REDUCED_INPUT
        blank = (
            state.presented_level is None
            and state.ready_level is None
            and not state.resident_levels
        )
        if blank:
            lane = Lane.DISPLAY_PREVIEW
            if verdict is not None and not verdict.admits_lane(lane):
                return COARSE_RUNG_LANE_NOT_ADMITTED
            return COARSE_RUNG_PLANNED
        return COARSE_RUNG_ALREADY_COVERED

    # ------------------------------------------------------------ planning

    def plan_tile(
        self,
        state: TileLodState,
        demand: LodDemand,
        verdict: SchedulingVerdict | None = None,
    ) -> tuple[RungStep, ...]:
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
        # Both vocabularies circulate: the lifecycle records a retained floor
        # as quality="fallback" while its display payload is labelled
        # "preview" (tile_identity maps between them), and `ready_preview`
        # below has always accepted the pair. Accepting only "preview" here
        # made a presented fallback AT the desired level read as converged and
        # plan no refinement at all -- member 5 of the 2026-07-16 churn
        # starvation family, whose belt-and-braces guard in the shared exact
        # pass ADR 0059 retired. A non-exact payload never satisfies a demand,
        # whatever it is called.
        presented_preview = str(getattr(state, "presented_quality", "exact") or "exact") in {
            "preview",
            "fallback",
        }
        resident = frozenset(int(level) for level in state.resident_levels)
        ready = None if state.ready_level is None else int(state.ready_level)
        ready_preview = str(state.ready_quality or "") in {"preview", "fallback"}

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
            if ready is not None:
                candidates.append(float(ready))
            candidates.extend(float(step.level) for step in steps)
            return min(candidates) if candidates else float("inf")

        # Pre-native rungs are planned only when admission says they are useful
        # for first pixels.  Target-ready, caught-up tiles skip the preview
        # detour and go straight to DESIRED/EXACT.
        # The admission predicate permits genuinely tile-local pipelines and
        # montage-axis expansions whose identical reduced region is backed by
        # one cacheable real-document stage.
        preview_level = coarse_rung_level(
            demand=demand,
            retention_level=policy.floor_level,
        )
        preview_target_has_finer_followup = preview_level > desired
        cheap_pre_native = (
            bool(policy.coarse_rung_enabled)
            and bool(state.allow_preview)
            and preview_target_has_finer_followup
            and (policy.reduced_input_available or state.floor_available)
        )
        # 1) FLOOR — the one coarse rung, only while the tile is blank.
        if presented is None and ready is None and not resident and cheap_pre_native:
            steps.append(
                RungStep(
                    tile_number=state.tile_number,
                    rung=Rung.FLOOR,
                    level=preview_level,
                    reduce_from_native=False,
                    lane=Lane.DISPLAY_PREVIEW,
                    priority=Priority.INTERACTIVE,
                    reason=(
                        "retained floor commit" if state.floor_available else "cold floor fill"
                    ),
                    scheduling_rank=int(state.scheduling_rank),
                )
            )

        # 2) DESIRED — fill or refine only when the displayed tile is below
        # the demand. A coarser demand is a minimum acceptable quality, not a
        # command to replace already-presented finer data. Demotion belongs to
        # memory/eviction policy; the ladder must keep camera-only zooms from
        # churning materialization and presentation.
        ready_satisfies_display_demand = bool(
            ready is not None and not ready_preview and ready == desired
        )
        desired_resident = (
            ready_satisfies_display_demand
            or (desired in resident and not presented_preview)
            or (presented == desired and not presented_preview)
        )
        if presented is not None and int(presented) <= desired and not presented_preview:
            desired_resident = True
        if not desired_resident and (desired > 0 or desired < finest_available()):
            # DESIRED is usually refinement and belongs to preparation.  But
            # when no floor/preview/current payload exists it is also the
            # successor target's first and only presentable rung.  Classify
            # that work by its semantic role, not by the historical rung
            # name: during a retained slice scrub DISPLAY_PREPARATION is
            # intentionally parked, while one DISPLAY_PREVIEW worker remains
            # available to replace the stale predecessor atomically.
            has_first_pixel = bool(
                presented is not None
                or ready is not None
                or resident
                or any(step.rung == Rung.FLOOR for step in steps)
            )
            steps.append(
                RungStep(
                    tile_number=state.tile_number,
                    rung=Rung.DESIRED,
                    level=desired,
                    reduce_from_native=not policy.reduced_input_available,
                    # A target rung is phase-2 work whenever this target has a
                    # FLOOR path or any current first pixel. It must not share
                    # the coverage lane: doing so lets already-covered tiles
                    # consume workers while required tiles are still waiting
                    # for preview. The only DESIRED work on DISPLAY_PREVIEW is
                    # the first-and-only presentable rung of a pipeline with no
                    # coarse producer.
                    lane=(
                        Lane.DISPLAY_PREVIEW if not has_first_pixel else Lane.DISPLAY_PREPARATION
                    ),
                    priority=(
                        Priority.VISIBLE_IMAGE
                        if presented is None or presented not in acceptable
                        else Priority.HOVER
                    ),
                    reason=str(demand.reason or "desired display level"),
                    scheduling_rank=int(state.scheduling_rank),
                )
            )

        # 3) EXACT — only on explicit request (inspection) or when the
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
                    scheduling_rank=int(state.scheduling_rank),
                )
            )

        if verdict is not None:
            steps = [step for step in steps if verdict.admits_lane(step.lane)]
        return tuple(steps)

    def plan(
        self,
        states,
        demand: LodDemand,
        verdict: SchedulingVerdict | None = None,
    ) -> tuple[RungStep, ...]:
        """Plan every tile, coarse rungs across tiles before fine rungs.

        Cross-tile ordering matters for perceived progress: every visible
        tile should reach FLOOR before any tile spends budget on
        DESIRED/EXACT (Plan 05 floor-first-fill, generalized).
        """

        per_tile = [self.plan_tile(state, demand, verdict) for state in states]
        ordered: list[RungStep] = []
        for rung in Rung:
            for steps in per_tile:
                ordered.extend(step for step in steps if step.rung == rung)
        return tuple(ordered)

    # ------------------------------------------------------------- helpers

    def _native_only_plan(self, state: TileLodState) -> tuple[RungStep, ...]:
        ready_native = state.ready_level == 0 and str(state.ready_quality or "") not in {
            "preview",
            "fallback",
        }
        if state.presented_level == 0 or 0 in set(state.resident_levels) or ready_native:
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
                scheduling_rank=int(state.scheduling_rank),
            ),
        )


__all__ = [
    "COARSE_RUNG_ALREADY_COVERED",
    "COARSE_RUNG_DISABLED",
    "COARSE_RUNG_ENABLED_DEFAULT",
    "COARSE_RUNG_LANE_NOT_ADMITTED",
    "COARSE_RUNG_MAX_SCREEN_PIXELS_PER_TEXEL",
    "COARSE_RUNG_MIN_LEVEL_DELTA",
    "COARSE_RUNG_MIN_SCREEN_PIXELS_PER_TEXEL",
    "COARSE_RUNG_NATIVE_ONLY",
    "COARSE_RUNG_NO_REDUCED_INPUT",
    "COARSE_RUNG_PLANNED",
    "COARSE_RUNG_PREVIEW_NOT_ALLOWED",
    "LadderPolicy",
    "LodLadder",
    "Rung",
    "RungStep",
    "TileLodState",
    "coarse_rung_level",
]
