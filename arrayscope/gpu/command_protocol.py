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
``UpdateOverlayGeometry`` shell overlay state -> flat primitive buffer
``UpdateGlyphAtlas``      CPU-baked glyph alpha atlas -> sampled overlay texture
``SetOverlayCamera``      world-space overlay camera (uniform-only)
``SetDisplayMapping``     shader-mapping uniforms + physical-truth audit
``GenerateLodPages``      G6 resident-page reduction
``DispatchHistogram``     G6 histogram reduction
``PresentGeneration``     ``TileCommitReport`` acknowledgement + physical audit
========================  ====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from arrayscope.gpu.keys import REDUCERS, REDUCER_MEAN, REPRESENTATIONS, DataChunkKey

#: Display mapping modes for complex-valued chunks. Scalar chunks use
#: ``"real"`` (imaginary plane is zero by construction).
MAPPING_MODES = ("magnitude", "phase", "real", "imag")

#: Display-scale formulas applied after component extraction and before
#: levels normalization.
MAPPING_SCALES = ("linear", "log", "symlog")

#: Flat overlay primitive kinds.  These are draw instructions, never scene
#: objects: one instance is one line segment, filled world rectangle,
#: screen-sized handle quad, screen-sized filled rectangle, or one textured
#: glyph quad sampling the executor's glyph atlas.
OVERLAY_PRIMITIVE_KINDS = (
    "line",
    "world_rect",
    "handle_quad",
    "screen_rect",
    "glyph_quad",
)


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
    ``lod_reducer`` selects the derived-value family for the plane's flat LOD
    spans; native pages are shared input, but incompatible reduced families
    never occupy the same physical lookup entry (ADR 0056 §3).
    """

    document_generation: object
    operation_key: object
    plane_shape: tuple[int, int]
    max_lod: int = 0
    representation: str = "scalar_r32f"
    lod_reducer: str = REDUCER_MEAN

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
        reducer = str(self.lod_reducer)
        if reducer not in REDUCERS:
            raise ValueError(
                f"unknown plane LOD reducer {reducer!r}; expected one of {REDUCERS}"
            )
        object.__setattr__(self, "lod_reducer", reducer)


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


@dataclass(frozen=True)
class OverlayPrimitive:
    """One backend-neutral, world-anchored flat overlay primitive.

    ``line`` uses ``p0``/``p1`` as endpoints and ``width`` as pixel line
    width. ``world_rect`` uses them as opposite world-space corners and is
    filled. ``handle_quad`` is centered on ``p0`` with pixel side length
    ``width``. ``visibility_anchor`` lets cursor geometry disappear when its
    semantic anchor leaves the viewport without rebuilding the instance
    buffer on camera changes.

    ``screen_rect`` and ``glyph_quad`` are screen-space-sized quads anchored
    at world point ``p0``: ``screen_offset`` is the quad's top-left offset
    from the anchor in physical pixels (y down) and ``size`` its physical
    pixel extent — constant on screen under zoom, moving with the image
    under pan because the anchor is world space.  ``glyph_quad``
    additionally samples the executor's glyph atlas over normalized
    ``uv_rect`` ``(u0, v0, u1, v1)`` as an alpha mask on ``rgba``.
    """

    kind: str
    p0: tuple[float, float]
    p1: tuple[float, float] = (0.0, 0.0)
    rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    width: float = 1.0
    visibility_anchor: tuple[float, float] | None = None
    screen_offset: tuple[float, float] = (0.0, 0.0)
    size: tuple[float, float] = (0.0, 0.0)
    uv_rect: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        kind = str(self.kind)
        if kind not in OVERLAY_PRIMITIVE_KINDS:
            raise ValueError(
                f"unknown overlay primitive kind {kind!r}; "
                f"expected one of {OVERLAY_PRIMITIVE_KINDS}"
            )
        p0 = tuple(float(value) for value in self.p0)
        p1 = tuple(float(value) for value in self.p1)
        rgba = tuple(float(value) for value in self.rgba)
        if len(p0) != 2 or len(p1) != 2:
            raise ValueError("overlay primitive points must be 2-D")
        if len(rgba) != 4 or any(value < 0.0 or value > 1.0 for value in rgba):
            raise ValueError("overlay primitive RGBA must contain four values in [0, 1]")
        width = float(self.width)
        if width <= 0.0:
            raise ValueError("overlay primitive width must be positive")
        anchor = self.visibility_anchor
        if anchor is not None:
            anchor = tuple(float(value) for value in anchor)
            if len(anchor) != 2:
                raise ValueError("overlay visibility anchor must be 2-D")
        screen_offset = tuple(float(value) for value in self.screen_offset)
        size = tuple(float(value) for value in self.size)
        uv_rect = tuple(float(value) for value in self.uv_rect)
        if len(screen_offset) != 2 or len(size) != 2:
            raise ValueError("overlay screen offset/size must be 2-D")
        if len(uv_rect) != 4:
            raise ValueError("overlay uv rect must contain four values")
        if kind in ("screen_rect", "glyph_quad") and (size[0] <= 0.0 or size[1] <= 0.0):
            raise ValueError(f"{kind} primitives need a positive pixel size")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "p0", p0)
        object.__setattr__(self, "p1", p1)
        object.__setattr__(self, "rgba", rgba)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "visibility_anchor", anchor)
        object.__setattr__(self, "screen_offset", screen_offset)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "uv_rect", uv_rect)


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
class GenerateLodPages:
    """Reduce resident child pages into one derived parent page.

    ``source_keys`` are the one-to-four valid children at the immediately
    finer reduction level.  The destination is bound to the executor's page
    table only after the GPU pass has been submitted; no CPU payload or texel
    upload is involved.  Reducer-family honesty is validated again by the
    executor, which currently implements only component-wise ``mean``.
    """

    source_keys: tuple[DataChunkKey, ...]
    destination_key: DataChunkKey

    def __post_init__(self) -> None:
        sources = tuple(self.source_keys)
        if not 1 <= len(sources) <= 4:
            raise ValueError("LOD generation requires one to four child pages")
        if any(not isinstance(key, DataChunkKey) for key in sources):
            raise TypeError("LOD generation sources must be DataChunkKey instances")
        if not isinstance(self.destination_key, DataChunkKey):
            raise TypeError("LOD generation destination must be a DataChunkKey")
        object.__setattr__(self, "source_keys", sources)


@dataclass(frozen=True)
class UpdateTileInstances:
    tiles: tuple[TileInstance, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tiles", tuple(self.tiles))


@dataclass(frozen=True)
class UpdateOverlayGeometry:
    """Replace the executor's one flat overlay instance buffer.

    Only semantic overlay changes emit this command. Camera changes use
    :class:`SetOverlayCamera`, leaving this geometry physically untouched.
    """

    primitives: tuple[OverlayPrimitive, ...]

    def __post_init__(self) -> None:
        primitives = tuple(self.primitives)
        if any(not isinstance(value, OverlayPrimitive) for value in primitives):
            raise TypeError("overlay geometry must contain OverlayPrimitive instances")
        object.__setattr__(self, "primitives", primitives)


@dataclass(frozen=True)
class UpdateGlyphAtlas:
    """Replace the executor's one glyph alpha atlas (rare, off the frame path).

    ``data`` is a tightly-packed ``height`` x ``width`` single-channel alpha
    image (uint8).  The atlas is baked on the CPU (Qt is allowed there — it
    happens only when new glyphs appear); a frame whose glyphs are all cached
    must emit NO atlas command, and :attr:`FrameReport.glyph_atlas_uploads`
    is the oracle that it did not.  Glyph quads reference this atlas via
    normalized ``uv_rect`` coordinates, so a re-uploaded atlas must come with
    (or precede) overlay geometry laid out against the same atlas revision.
    """

    width: int
    height: int
    data: bytes

    def __post_init__(self) -> None:
        width = int(self.width)
        height = int(self.height)
        data = bytes(self.data)
        if width <= 0 or height <= 0:
            raise ValueError(f"glyph atlas must be non-empty, got {width}x{height}")
        if len(data) != width * height:
            raise ValueError(
                f"glyph atlas data must be {width * height} bytes "
                f"({width}x{height} alpha8), got {len(data)}"
            )
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "data", data)


@dataclass(frozen=True)
class SetOverlayCamera:
    """Set the sorted world viewport and axis direction for overlay drawing.

    This is uniform-only state. ``world_rect`` is ``(x0, y0, x1, y1)``;
    target pixel size is supplied by the executor's presentation surface.
    """

    world_rect: tuple[float, float, float, float]
    x_inverted: bool = False
    y_inverted: bool = True

    def __post_init__(self) -> None:
        rect = tuple(float(value) for value in self.world_rect)
        if len(rect) != 4:
            raise ValueError("overlay camera world rect must contain four values")
        x0, y0, x1, y1 = rect
        if not x1 > x0 or not y1 > y0:
            raise ValueError(f"overlay camera world rect must be non-empty, got {rect}")
        object.__setattr__(self, "world_rect", rect)
        object.__setattr__(self, "x_inverted", bool(self.x_inverted))
        object.__setattr__(self, "y_inverted", bool(self.y_inverted))


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

    Residency contract (2026-07-19 dogfood crash): keys referenced here must
    survive the *same submission's* earlier residency work — an executor's
    own LRU eviction (triggered by this batch's ensures/LOD generation) must
    shield them for the submission's duration while its pool budget allows.
    When pool pressure genuinely exceeds that shield, or a key was never
    resident, the executor skips the key and reports it in
    :attr:`FrameReport.histogram_missing` — it never fails the submission.
    Consumers must treat evidence with missing keys as unsatisfied and retry
    through their normal scheduling machinery.
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
    | GenerateLodPages
    | UpdateTileInstances
    | UpdateOverlayGeometry
    | UpdateGlyphAtlas
    | SetOverlayCamera
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
    zero-upload oracles read it); ``glyph_atlas_uploads`` counts glyph-atlas
    texture writes the same way (zero once every drawn glyph is cached);
    ``lod_pages_generated`` names pages created
    wholly inside the resident pool; ``histograms`` maps DispatchHistogram
    order-index → bins array over the keys that were resident at dispatch;
    ``histogram_missing`` maps the same order-index → keys the executor had
    to skip (not resident at dispatch, e.g. sacrificed to pool pressure
    within this very submission) — evidence with missing keys is partial and
    must not be consumed as truth; ``wait_completed`` blocks until the GPU
    finished the submitted work — the completion token that page/staging
    recycling requires (renderer gate 3).
    """

    generation: int
    presented: bool = False
    uploads: int = 0
    overlay_buffer_writes: int = 0
    glyph_atlas_uploads: int = 0
    evictions: int = 0
    lod_pages_generated: tuple[DataChunkKey, ...] = ()
    histograms: dict[int, object] = field(default_factory=dict)
    histogram_bounds: dict[int, tuple[float, float] | None] = field(default_factory=dict)
    histogram_missing: dict[int, tuple[DataChunkKey, ...]] = field(default_factory=dict)
    wait_completed: object = None  # callable () -> None


@runtime_checkable
class RendererExecutor(Protocol):
    """The one interface a renderer backend implements."""

    def submit(self, submission: FrameSubmission) -> FrameReport: ...
