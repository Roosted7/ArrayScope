"""Backend-neutral display frame and presentation models."""

from arrayscope.display.model.commit import (
    CommitKind,
    CommitPlan,
    DisplayPayload,
    DisplayPresentation,
    DisplayTiledPresentation,
    PresentationDecision,
    PresentationInput,
    RenderRequestContext,
)
from arrayscope.display.model.frame import (
    CommittedDisplayFrame,
    DisplayFrameKey,
    DisplayTilePayload,
    FrameValueSource,
    PageBackedPresentation,
    TiledValueSource,
    TilePresentationDelta,
    TilePresentationState,
)
from arrayscope.display.model.montage_levels import (
    MontageLevelStats,
    MontageLevelTracker,
    TileLevelStats,
    montage_level_key,
)
from arrayscope.display.model.presentation_generation import (
    LevelPresentationTarget,
    PresentationGenerationSnapshot,
    PresentationGenerationTracker,
)
from arrayscope.display.model.tile_identity import (
    ArrayPlaneIdentity,
    TileIdentity,
    TileLodIdentity,
    TilePresentationIdentity,
    acknowledged_identity_satisfies_target,
    array_plane_identities,
    complex_mapping_identity,
    tile_ack_identity,
    tile_truth_record,
)
from arrayscope.display.model.tile_priority import (
    MontageTilePriorityQueue,
    TilePriorityContext,
    prioritize_tiles,
    tile_priority_key,
)

__all__ = [
    "ArrayPlaneIdentity",
    "CommitKind",
    "CommitPlan",
    "CommittedDisplayFrame",
    "DisplayFrameKey",
    "DisplayPayload",
    "DisplayPresentation",
    "DisplayTilePayload",
    "DisplayTiledPresentation",
    "FrameValueSource",
    "LevelPresentationTarget",
    "MontageLevelStats",
    "MontageLevelTracker",
    "MontageTilePriorityQueue",
    "PageBackedPresentation",
    "PresentationDecision",
    "PresentationGenerationSnapshot",
    "PresentationGenerationTracker",
    "PresentationInput",
    "RenderRequestContext",
    "TileIdentity",
    "TileLevelStats",
    "TileLodIdentity",
    "TilePresentationDelta",
    "TilePresentationIdentity",
    "TilePresentationState",
    "TilePriorityContext",
    "TiledValueSource",
    "acknowledged_identity_satisfies_target",
    "array_plane_identities",
    "complex_mapping_identity",
    "montage_level_key",
    "prioritize_tiles",
    "tile_ack_identity",
    "tile_priority_key",
    "tile_truth_record",
]
