"""Backend-neutral renderer command protocol (tensor-engine seam).

This module formalizes the semantic command table recorded in
``docs/proposals/tensor-engine-endpoint.md`` (renderer strategy): the frame
planner speaks *these* commands; a renderer backend (wgpu today, VisPy's
executor once migrated, Datoviz/native-Vulkan if ever re-opened) merely
executes them.  Nothing here may assume WGSL, GL texture names, Datoviz IDs,
Qt, or one-physical-texture-per-tile — the protocol carries ADR 0055/0056
identities (:class:`~arrayscope.gpu.keys.DataChunkKey`) and normalized
geometry only.

Like the rest of :mod:`arrayscope.gpu`, this module is passive data + an
executor interface: no scheduling happens here (ADR 0053 — the kernel owns
ordering; a frame submission is already-ordered work).

Command → engine-seam mapping (from the endpoint doc):

========================  ====================================================
``EnsureChunkResident``   ``ChunkStore.ensure`` / pool chunk plan (G1/G3b-2)
``EvictChunk``            ``PageTable.unbind``
``UpdateTileInstances``   draw parts / quad emission (G3c)
``SetDisplayMapping``     shader-mapping uniforms + physical-truth audit
``DispatchHistogram``     G6 reduction
``PresentGeneration``     ``TileCommitReport`` acknowledgement + physical audit
========================  ====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from arrayscope.gpu.keys import DataChunkKey

#: Display mapping modes for complex-valued chunks. Scalar chunks use
#: ``"real"`` (imaginary plane is zero by construction).
MAPPING_MODES = ("magnitude", "phase", "real", "imag")


#: Size of a display LUT: 256 RGBA8 entries.
LUT_BYTES = 256 * 4


@dataclass(frozen=True)
class DisplayMapping:
    """Shader-on-read display state: complex component + levels + LUT.

    ``lut`` is the resolved 256-entry RGBA8 table (raw bytes) — the protocol
    carries the table itself, never a colormap *name*, so backends need no
    knowledge of ArrayScope's colormap library. ``None`` means the backend's
    neutral grayscale ramp. Levels-normalized values index the table by
    nearest entry (``round(g * 255)``), matching the CPU display mirror.
    """

    mode: str = "magnitude"
    level_lo: float = 0.0
    level_hi: float = 1.0
    lut: bytes | None = None

    def __post_init__(self) -> None:
        if self.mode not in MAPPING_MODES:
            raise ValueError(
                f"unknown mapping mode {self.mode!r}; expected one of {MAPPING_MODES}"
            )
        object.__setattr__(self, "level_lo", float(self.level_lo))
        object.__setattr__(self, "level_hi", float(self.level_hi))
        if not self.level_hi > self.level_lo:
            raise ValueError(
                f"levels window must be non-empty, got [{self.level_lo}, {self.level_hi}]"
            )
        if self.lut is not None:
            lut = bytes(self.lut)
            if len(lut) != LUT_BYTES:
                raise ValueError(
                    f"lut must be {LUT_BYTES} bytes (256 RGBA8 entries), got {len(lut)}"
                )
            object.__setattr__(self, "lut", lut)


@dataclass(frozen=True)
class TileInstance:
    """One drawn tile: destination rect + source window + requested LOD.

    ``dst_rect`` is ``(x, y, w, h)`` in normalized target space ([0, 1] with
    y down); ``src_origin``/``src_size`` are in native (LOD-0) source pixels.
    The backend resolves source pixels through its page table at
    ``lod_level``, falling back to coarser resident ancestors (never black,
    ADR 0056 §5); which chunks are resident is NOT part of tile identity —
    that is the whole point of the tile/chunk/page split (ADR 0055).
    """

    dst_rect: tuple[float, float, float, float]
    src_origin: tuple[float, float]
    src_size: tuple[float, float]
    lod_level: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dst_rect", tuple(float(v) for v in self.dst_rect)
        )
        object.__setattr__(
            self, "src_origin", tuple(float(v) for v in self.src_origin)
        )
        object.__setattr__(
            self, "src_size", tuple(float(v) for v in self.src_size)
        )
        if len(self.dst_rect) != 4 or len(self.src_origin) != 2 or len(self.src_size) != 2:
            raise ValueError("malformed tile instance geometry")
        object.__setattr__(self, "lod_level", max(0, int(self.lod_level)))


# ---- commands ---------------------------------------------------------------


@dataclass(frozen=True)
class EnsureChunkResident:
    """Make one chunk's values resident; ``payload`` is the evaluated block.

    ``payload`` is an ndarray-like in the chunk's stored representation
    (``(h, w)`` scalar or ``(h, w, 2)`` complex-as-planes).  Re-ensuring an
    already-resident key is a no-op (zero uploads) — the upload counter in
    the frame report is the residency oracle.
    """

    key: DataChunkKey
    payload: object
    pinned: bool = False


@dataclass(frozen=True)
class EvictChunk:
    key: DataChunkKey


@dataclass(frozen=True)
class UpdateTileInstances:
    tiles: tuple[TileInstance, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tiles", tuple(self.tiles))


@dataclass(frozen=True)
class SetDisplayMapping:
    mapping: DisplayMapping


@dataclass(frozen=True)
class DispatchHistogram:
    """Magnitude histogram over the given resident chunks (G6 evidence)."""

    keys: tuple[DataChunkKey, ...]
    bins: int = 64
    lo: float = 0.0
    hi: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "bins", int(self.bins))
        object.__setattr__(self, "lo", float(self.lo))
        object.__setattr__(self, "hi", float(self.hi))
        if self.bins <= 0:
            raise ValueError("histogram bins must be positive")


@dataclass(frozen=True)
class PresentGeneration:
    """Render the current tiles/mapping as generation ``generation``."""

    generation: int


Command = (
    EnsureChunkResident
    | EvictChunk
    | UpdateTileInstances
    | SetDisplayMapping
    | DispatchHistogram
    | PresentGeneration
)


# ---- submission / report ----------------------------------------------------


@dataclass(frozen=True)
class FrameSubmission:
    """One ordered batch of commands, already prioritized by the kernel."""

    generation: int
    commands: tuple[Command, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "commands", tuple(self.commands))


@dataclass
class FrameReport:
    """What physically happened; the auditable half of the contract.

    ``uploads`` counts texel uploads performed by THIS submission (the
    zero-upload oracles read it); ``histograms`` maps DispatchHistogram
    order-index → bins array; ``wait_completed`` blocks until the GPU
    finished the submitted work — the completion token that page/staging
    recycling requires (renderer gate 3).
    """

    generation: int
    presented: bool = False
    uploads: int = 0
    evictions: int = 0
    histograms: dict[int, object] = field(default_factory=dict)
    wait_completed: object = None  # callable () -> None


@runtime_checkable
class RendererExecutor(Protocol):
    """The one interface a renderer backend implements."""

    def submit(self, submission: FrameSubmission) -> FrameReport: ...
