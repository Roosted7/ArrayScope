"""Single-owner tile lifecycle (ADR 0051).

Qt-free. The state machine in :mod:`tile_lifecycle` is the only writer of
per-tile lifecycle state; window/renderer/LOD/backend components are event
sources and effect executors.
"""

from .tile_lifecycle import (
    ClaimOwner,
    LevelPhase,
    Presentation,
    ReleaseClaim,
    Semantic,
    TileLifecycle,
    TileRecord,
)
from .tile_obligations import TileObligation, TileObligationPlan, build_tile_obligation_plan

__all__ = [
    "ClaimOwner",
    "LevelPhase",
    "Presentation",
    "ReleaseClaim",
    "Semantic",
    "TileObligation",
    "TileObligationPlan",
    "TileLifecycle",
    "TileRecord",
    "build_tile_obligation_plan",
]
