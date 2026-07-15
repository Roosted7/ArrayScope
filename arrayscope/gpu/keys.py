"""Identity keys for the GPU residency engine (ADR 0055).

Three identities that today's renderer conflates 1:1:

- a **view tile** is *where* content is drawn (presentation-relative);
- a **data chunk** is *which values* exist (evaluation-relative);
- a **page slot** (:mod:`arrayscope.gpu.page_table`) is *where bytes live*.

``DataChunkKey`` deliberately mirrors the vocabulary of
``display.model.tile_identity.TileIdentity`` (``document_generation``,
``operation_key``, LOD triple) so the two derivations can be checked against
each other, without importing the display layer from here.
"""

from __future__ import annotations

from dataclasses import dataclass

# Canonical representation labels for resident chunk values. "Raw" values
# stay raw on the GPU; display mapping is shader work (ADR 0055 §4). These
# match the atlas storage modes in the VisPy backend.
SCALAR_R32F = "scalar_r32f"
COMPLEX_RG32F = "complex_rg32f"
RGB8 = "rgb8"
REPRESENTATIONS = (SCALAR_R32F, COMPLEX_RG32F, RGB8)


@dataclass(frozen=True)
class ChunkLod:
    """Level-of-detail identity of a chunk's values (mirrors TileLodIdentity)."""

    level: int = 0
    factor: int = 1
    gutter: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", max(0, int(self.level)))
        object.__setattr__(self, "factor", max(1, int(self.factor)))
        object.__setattr__(self, "gutter", max(0, int(self.gutter)))


@dataclass(frozen=True)
class DataChunkKey:
    """Identity of one N-dimensional block of evaluated array values.

    Independent of whether, where, or how often the block is drawn. Two view
    tiles that show overlapping windows share the chunks under them; a window
    shift changes which chunks a tile references, not the chunks' identity.
    """

    document_generation: object
    operation_key: object
    lod: ChunkLod
    chunk_origin: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    dtype: str
    representation: str = SCALAR_R32F

    def __post_init__(self) -> None:
        if not isinstance(self.lod, ChunkLod):
            object.__setattr__(self, "lod", ChunkLod(*self.lod))
        origin = tuple(int(value) for value in self.chunk_origin)
        shape = tuple(int(value) for value in self.chunk_shape)
        if len(origin) != len(shape):
            raise ValueError(f"chunk origin {origin} and shape {shape} rank mismatch")
        if any(value < 0 for value in origin):
            raise ValueError(f"chunk origin must be non-negative, got {origin}")
        if any(value <= 0 for value in shape):
            raise ValueError(f"chunk shape must be positive, got {shape}")
        object.__setattr__(self, "chunk_origin", origin)
        object.__setattr__(self, "chunk_shape", shape)
        object.__setattr__(self, "dtype", str(self.dtype))
        representation = str(self.representation)
        if representation not in REPRESENTATIONS:
            raise ValueError(
                f"unknown chunk representation {representation!r}; expected one of {REPRESENTATIONS}"
            )
        object.__setattr__(self, "representation", representation)

    @property
    def rank(self) -> int:
        return len(self.chunk_origin)

    @property
    def stop(self) -> tuple[int, ...]:
        return tuple(o + s for o, s in zip(self.chunk_origin, self.chunk_shape))


@dataclass(frozen=True)
class ViewTileKey:
    """Identity of one rectangular output region of a presentation.

    Owned by the shared semantic layer; carries no payload and no residency.
    The mapping view tile → chunks lives in presentation planning, never on
    the key itself.
    """

    presentation_key: object
    tile_number: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "tile_number", int(self.tile_number))
