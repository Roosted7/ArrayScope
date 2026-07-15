"""Qt-free, VisPy-free GPU residency engine (ADR 0055, G-program).

This package owns the three-way vocabulary that ADR 0055 separates:

- :class:`ViewTileKey` — a rectangular output region of the presentation;
- :class:`DataChunkKey` — an N-dimensional block of evaluated array values;
- :class:`PageSlot` / :class:`PageTable` — physical placement of a chunk in
  a backend memory pool, with explicit "not resident" state.

Rendering backends consume this engine; they do not define it. Nothing here
may import Qt, VisPy, pyqtgraph, or ``arrayscope.display`` — the dependency
points the other way.
"""

from arrayscope.gpu.keys import ChunkLod, DataChunkKey, ViewTileKey
from arrayscope.gpu.chunk_grid import ChunkGrid, WindowDelta
from arrayscope.gpu.page_table import PageSlot, PageTable, ResidencyEntry
from arrayscope.gpu.chunk_store import (
    CapacityError,
    ChunkStore,
    ChunkStoreDiagnostics,
    Residency,
    SlotPool,
)

__all__ = [
    "CapacityError",
    "ChunkGrid",
    "ChunkLod",
    "ChunkStore",
    "ChunkStoreDiagnostics",
    "DataChunkKey",
    "PageSlot",
    "PageTable",
    "Residency",
    "ResidencyEntry",
    "SlotPool",
    "ViewTileKey",
    "WindowDelta",
]
