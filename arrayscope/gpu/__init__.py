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

from arrayscope.gpu.chunk_grid import ChunkGrid, WindowDelta
from arrayscope.gpu.chunk_store import (
    CapacityError,
    ChunkStore,
    ChunkStoreDiagnostics,
    Residency,
    SlotPool,
)
from arrayscope.gpu.chunk_summary import (
    HISTOGRAM_NORMALIZED_L1_TOLERANCE,
    ChunkHistogramAggregate,
    ChunkHistogramSummary,
    aggregate_chunk_summaries,
    chunk_key_frontier,
    chunk_summary_frontier,
    representative_sample_from_histogram,
    summarize_chunk,
)
from arrayscope.gpu.keys import ChunkLod, DataChunkKey, ViewTileKey
from arrayscope.gpu.page_table import PageResolution, PageSlot, PageTable, ResidencyEntry

__all__ = [
    "HISTOGRAM_NORMALIZED_L1_TOLERANCE",
    "CapacityError",
    "ChunkGrid",
    "ChunkHistogramAggregate",
    "ChunkHistogramSummary",
    "ChunkLod",
    "ChunkStore",
    "ChunkStoreDiagnostics",
    "DataChunkKey",
    "PageResolution",
    "PageSlot",
    "PageTable",
    "Residency",
    "ResidencyEntry",
    "SlotPool",
    "ViewTileKey",
    "WindowDelta",
    "aggregate_chunk_summaries",
    "chunk_key_frontier",
    "chunk_summary_frontier",
    "representative_sample_from_histogram",
    "summarize_chunk",
]
