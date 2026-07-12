"""The modular rendering pipeline (redesign R2/R3).

This package replaces the monolithic `window/frame_controller.py` +
legacy window-level LOD orchestration with modular chunks, each owning one
well-defined task and its state:

- `render.stages`   — typed boundaries between pipeline stages. Data only.
- `render.ladder`   — the unified LOD ladder: ONE owner for "which quality
                      rung does each tile need next" (floor → preview →
                      desired → exact). Pure planning, no I/O, no Qt.
- `render.pipeline` — FramePipeline: turns ladder plans into kernel tasks,
                      consumes TileLifecycle events, emits bounded commit
                      batches through an injected Effects protocol.

Ownership rules:

1. `TileLifecycle` (presentation/tile_lifecycle.py) remains the single owner
   of tile state; this package never keeps a parallel tile collection.
2. The kernel (arrayscope.kernel) is the only executor. No per-purpose pools, no
   pacing timers here; the GUI thread only applies commit batches.
3. Backends declare capabilities (shader windowing, atlas residency, uniform
   level changes); the ladder and pipeline branch on capabilities, never on
   backend names. Both backends are first-class: VisPy exploits GPU
   residency/uniforms, PyQtGraph gets correct (bounded CPU) equivalents.
4. Operations run once per rung on reduced input when they commute with
   reduction (`operations.capabilities`), else once at native + reduce.
"""

from arrayscope.render.ladder import LodLadder, Rung, RungStep, TileLodState
from arrayscope.render.pipeline import FramePipeline
from arrayscope.render.stages import (
    CommitBatch,
    RenderIntent,
    TileWork,
)

__all__ = [
    "CommitBatch",
    "FramePipeline",
    "LodLadder",
    "RenderIntent",
    "Rung",
    "RungStep",
    "TileLodState",
    "TileWork",
]
