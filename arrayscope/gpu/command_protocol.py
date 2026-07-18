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
``BindContentPlanes``     session/montage plane set → flat-table spans
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

from arrayscope.gpu.keys import REPRESENTATIONS, DataChunkKey

#: Display mapping modes for complex-valued chunks. Scalar chunks use
#: ``"real"`` (imaginary plane is zero by construction).
MAPPING_MODES = ("magnitude", "phase", "real", "imag")

#: Display-scale formulas applied after component extraction and before
#: levels normalization.
MAPPING_SCALES = ("linear", "log", "symlog")


#: Size of a display LUT: 256 RGBA8 entries.
LUT_BYTES = 256 * 4


@dataclass(frozen=True)
class DisplayMapping:
    """Shader-on-read display state: component + scale + levels + LUT.

    ``lut`` is the resolved 256-entry RGBA8 table (raw bytes) — the protocol
    carries the table itself, never a colormap *name*, so backends need no
    knowledge of ArrayScope's colormap library. ``None`` means the backend's
    neutral grayscale ramp. Levels-normalized values index the table by
    nearest entry (``round(g * 255)``), matching the CPU display mirror.
    ``phase_color`` makes the LUT coordinate cyclic phase; for a non-phase
    component the normalized component modulates that color's intensity.
    """

    mode: str = "magnitude"
    level_lo: float = 0.0
    level_hi: float = 1.0
    lut: bytes | None = None
    scale: str = "linear"
    symlog_constant: float = 0.0
    phase_color: bool = False

    def __post_init__(self) -> None:
        if self.mode not in MAPPING_MODES:
            raise ValueError(
                f"unknown mapping mode {self.mode!r}; expected one of {MAPPING_MODES}"
            )
        if self.scale not in MAPPING_SCALES:
            raise ValueError(
                f"unknown mapping scale {self.scale!r}; expected one of {MAPPING_SCALES}"
            )
        object.__setattr__(self, "level_lo", float(self.level_lo))
        object.__setattr__(self, "level_hi", float(self.level_hi))
        object.__setattr__(self, "symlog_constant", float(self.symlog_constant))
        object.__setattr__(self, "phase_color", bool(self.phase_color))
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
class ContentPlane:
    """One bound 2-D content plane: identity + geometry + stored representation.

    A plane is *which values* a tile instance samples from — the montage/
    session unit.  ``plane_shape`` is ``(h, w)`` in native (LOD-0) pixels;
    ``max_lod`` is the deepest reduction level a backend may fall back to for
    this plane's chunks (ADR 0056 §5).  Binding carries no payloads and no
    residency: chunks are ensured separately, and chunks of planes that are
    currently *unbound* stay warm in the backend's page table.
    """

    document_generation: object
    operation_key: object
    plane_shape: tuple[int, int]
    max_lod: int = 0
    representation: str = "scalar_r32f"

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.plane_shape)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError(f"plane shape must be positive (h, w), got {self.plane_shape}")
        object.__setattr__(self, "plane_shape", shape)
        object.__setattr__(self, "max_lod", max(0, int(self.max_lod)))
        representation = str(self.representation)
        if representation not in REPRESENTATIONS:
            raise ValueError(
                f"unknown plane representation {representation!r}; "
                f"expected one of {REPRESENTATIONS}"
            )
        object.__setattr__(self, "representation", representation)


@dataclass(frozen=True)
class TileInstance:
    """One drawn tile: destination rect + source window + requested LOD.

    ``dst_rect`` is ``(x, y, w, h)`` in normalized target space ([0, 1] with
    y down); ``src_origin``/``src_size`` are in native (LOD-0) source pixels
    of the bound content plane selected by ``plane_index``.  The backend
    resolves source pixels through its page table at ``lod_level``, falling
    back to coarser resident ancestors (never black, ADR 0056 §5); which
    chunks are resident is NOT part of tile identity — that is the whole
    point of the tile/chunk/page split (ADR 0055).
    """

    dst_rect: tuple[float, float, float, float]
    src_origin: tuple[float, float]
    src_size: tuple[float, float]
    lod_level: int = 0
    plane_index: int = 0

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
        plane_index = int(self.plane_index)
        if plane_index < 0:
            raise ValueError(f"plane_index must be non-negative, got {plane_index}")
        object.__setattr__(self, "plane_index", plane_index)


# ---- commands ---------------------------------------------------------------


@dataclass(frozen=True)
class BindContentPlanes:
    """Replace the full set of bound content planes for tile sampling.

    ``TileInstance.plane_index`` indexes this tuple.  Binding is descriptor
    work only: it never uploads and never evicts — resident chunks of planes
    dropped from the bound set stay warm (scroll-back across planes is
    zero-upload while their pages survive eviction pressure).
    """

    planes: tuple[ContentPlane, ...]

    def __post_init__(self) -> None:
        planes = tuple(self.planes)
        for plane in planes:
            if not isinstance(plane, ContentPlane):
                raise TypeError(f"bound planes must be ContentPlane, got {type(plane).__name__}")
        object.__setattr__(self, "planes", planes)


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
    """Mapped-scalar histogram over the given resident chunks (G6 evidence).

    ``lo``/``hi`` may both be omitted to ask the executor to derive exact
    finite bounds on the GPU before binning. Scalar pages and the scalar
    level signal packed with windowable RGB ignore ``mode``; complex pages
    use the same component and scale vocabulary as :class:`DisplayMapping`.
    """

    keys: tuple[DataChunkKey, ...]
    bins: int = 64
    lo: float | None = 0.0
    hi: float | None = 1.0
    mode: str = "magnitude"
    scale: str = "linear"
    symlog_constant: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "bins", int(self.bins))
        if (self.lo is None) != (self.hi is None):
            raise ValueError("histogram lo/hi must both be set or both be omitted")
        if self.lo is not None:
            object.__setattr__(self, "lo", float(self.lo))
            object.__setattr__(self, "hi", float(self.hi))
            if not self.hi > self.lo:
                raise ValueError("histogram range must be non-empty")
        if self.mode not in MAPPING_MODES:
            raise ValueError(f"unknown histogram mapping mode {self.mode!r}")
        if self.scale not in MAPPING_SCALES:
            raise ValueError(f"unknown histogram mapping scale {self.scale!r}")
        object.__setattr__(self, "symlog_constant", float(self.symlog_constant))
        if self.bins <= 0:
            raise ValueError("histogram bins must be positive")


@dataclass(frozen=True)
class PresentGeneration:
    """Render the current tiles/mapping as generation ``generation``."""

    generation: int


Command = (
    BindContentPlanes
    | EnsureChunkResident
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
    histogram_bounds: dict[int, tuple[float, float] | None] = field(default_factory=dict)
    wait_completed: object = None  # callable () -> None


@runtime_checkable
class RendererExecutor(Protocol):
    """The one interface a renderer backend implements."""

    def submit(self, submission: FrameSubmission) -> FrameReport: ...
