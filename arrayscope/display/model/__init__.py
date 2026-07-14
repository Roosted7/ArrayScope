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
    TilePresentationDelta,
    TilePresentationState,
    TiledValueSource,
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
from arrayscope.display.model.tile_priority import (
    MontageTilePriorityQueue,
    TilePriorityContext,
    prioritize_tiles,
    tile_priority_key,
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

__all__ = [
    "CommitKind",
    "ArrayPlaneIdentity",
    "CommitPlan",
    "CommittedDisplayFrame",
    "DisplayFrameKey",
    "DisplayPayload",
    "DisplayPresentation",
    "DisplayTilePayload",
    "DisplayTiledPresentation",
    "FrameValueSource",
    "MontageLevelStats",
    "MontageTilePriorityQueue",
    "MontageLevelTracker",
    "LevelPresentationTarget",
    "PresentationDecision",
    "PresentationGenerationSnapshot",
    "PresentationGenerationTracker",
    "PresentationInput",
    "RenderRequestContext",
    "TilePresentationDelta",
    "TileIdentity",
    "TileLodIdentity",
    "TilePresentationIdentity",
    "TilePresentationState",
    "TilePriorityContext",
    "prioritize_tiles",
    "tile_priority_key",
    "TileLevelStats",
    "TiledValueSource",
    "montage_level_key",
    "acknowledged_identity_satisfies_target",
    "array_plane_identities",
    "complex_mapping_identity",
    "tile_ack_identity",
    "tile_truth_record",
]
