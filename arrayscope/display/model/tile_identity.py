"""Typed semantic and presentation identities for tiled display truth."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arrayscope.display.shader_mapping import TexturePlaneKind


def _enum_value(value):
    return None if value is None else getattr(value, "value", value)


@dataclass(frozen=True)
class ArrayPlaneIdentity:
    """Identity of the array plane handed to a backend upload/mapping path."""

    component: str
    pointer: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class TileLodIdentity:
    level: int = 0
    factor: int = 1
    gutter: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", max(0, int(self.level)))
        object.__setattr__(self, "factor", max(1, int(self.factor)))
        object.__setattr__(self, "gutter", max(0, int(self.gutter)))


@dataclass(frozen=True)
class TileIdentity:
    """Exact semantic identity a backend target or acknowledgement describes."""

    document_generation: object
    operation_key: object
    source_index: int
    image_axes: tuple[int, ...]
    axis_flips: tuple[bool, ...]
    channel: str
    complex_mapping: object
    texture_kind: TexturePlaneKind
    semantic_generation: object
    lod: TileLodIdentity = TileLodIdentity()
    quality: str = "exact"
    # Physical upload-plane provenance is diagnostic evidence, not pixel
    # semantics: equal source pixels may be copied into a different buffer or
    # rebound from existing residency without changing what the tile means.
    real_plane: ArrayPlaneIdentity | None = field(default=None, compare=False, hash=False)
    imag_plane: ArrayPlaneIdentity | None = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_index", int(self.source_index))
        object.__setattr__(self, "image_axes", tuple(int(axis) for axis in self.image_axes))
        object.__setattr__(self, "axis_flips", tuple(bool(value) for value in self.axis_flips))
        object.__setattr__(self, "channel", str(_enum_value(self.channel)))
        kind = self.texture_kind
        if not isinstance(kind, TexturePlaneKind):
            kind = TexturePlaneKind(_enum_value(kind))
        object.__setattr__(self, "texture_kind", kind)
        if not isinstance(self.lod, TileLodIdentity):
            object.__setattr__(self, "lod", TileLodIdentity(*self.lod))
        quality = "fallback" if str(self.quality) == "preview" else str(self.quality or "exact")
        if quality not in {"exact", "fallback"}:
            raise ValueError(f"tile identity quality must be exact/fallback, got {quality!r}")
        object.__setattr__(self, "quality", quality)

    @property
    def semantic_key(self) -> tuple[object, ...]:
        return (
            self.document_generation,
            self.operation_key,
            self.source_index,
            self.image_axes,
            self.axis_flips,
            self.channel,
            self.complex_mapping,
            self.texture_kind,
            self.semantic_generation,
        )

    def compatible_fallback_for(self, target: "TileIdentity") -> bool:
        """Whether this is an explicitly safe lower-quality target fallback."""

        if not isinstance(target, TileIdentity) or self.semantic_key != target.semantic_key:
            return False
        if self == target:
            return True
        return bool(
            self.quality == "fallback"
            or int(self.lod.level) > int(target.lod.level)
            or int(self.lod.factor) > int(target.lod.factor)
        )

    def satisfies_target(self, target: "TileIdentity") -> bool:
        """Whether this payload is exact target quality or a safe fallback."""

        if not isinstance(target, TileIdentity) or self.semantic_key != target.semantic_key:
            return False
        if self.quality == "exact" and int(self.lod.level) <= int(target.lod.level):
            return True
        return self.compatible_fallback_for(target)


@dataclass(frozen=True)
class TilePresentationIdentity:
    """Presentation state kept separate from source-pixel identity."""

    levels_generation: int
    levels: tuple[float, float] | None = None
    scale: object = None
    lut_identity: object = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels_generation", max(0, int(self.levels_generation)))
        if self.levels is not None:
            object.__setattr__(self, "levels", tuple(float(value) for value in self.levels))
        object.__setattr__(self, "scale", _enum_value(self.scale))


def array_plane_identities(data) -> tuple[ArrayPlaneIdentity | None, ArrayPlaneIdentity | None]:
    """Return inspectable real/imag plane identities for backend-facing data."""

    values = np.asarray(data)
    if np.iscomplexobj(values):
        return _array_plane_identity(values.real, "real"), _array_plane_identity(values.imag, "imag")
    if values.ndim >= 3 and values.shape[-1] == 2 and np.issubdtype(values.dtype, np.floating):
        return _array_plane_identity(values[..., 0], "real"), _array_plane_identity(values[..., 1], "imag")
    return _array_plane_identity(values, "real"), None


def complex_mapping_identity(mapping) -> object:
    """Pixel-meaning part of shader mapping, excluding levels/LUT/scale."""

    if mapping is None:
        return None
    return (
        _enum_value(getattr(mapping, "display_mode", None)),
        _enum_value(getattr(mapping, "component", None)),
        str(getattr(mapping, "histogram_source_policy", "mapped")),
    )


def tile_ack_identity(payload) -> object:
    """Return the exact typed identity a backend must acknowledge."""

    identity = getattr(payload, "tile_identity", None)
    return identity if identity is not None else getattr(payload, "source_id", None)


def acknowledged_identity_satisfies_target(acknowledged, target) -> bool:
    """Return whether backend truth is safe to draw for a typed target.

    A missing target means the caller is outside the R8A typed contract and is
    intentionally left unchanged.  Once a typed target exists, opaque legacy
    source ids are never sufficient evidence.
    """

    if target is None:
        return True
    return bool(
        isinstance(acknowledged, TileIdentity)
        and isinstance(target, TileIdentity)
        and acknowledged.satisfies_target(target)
    )


def tile_truth_record(*, tile_number: int, target, acknowledged, payload=None) -> dict[str, object]:
    """Build one compact, JSON-friendly R8A tile truth record."""

    identity = acknowledged if isinstance(acknowledged, TileIdentity) else None
    source = identity if identity is not None else getattr(payload, "tile_identity", None)
    if source is None:
        texture_kind = None
        real_plane = None
        imag_plane = None
        complex_mapping = None
        lod = None
    else:
        texture_kind = source.texture_kind.value
        real_plane = source.real_plane
        imag_plane = source.imag_plane
        complex_mapping = source.complex_mapping
        lod = source.lod
    presentation = getattr(payload, "presentation_identity", None)
    return {
        "tile": int(tile_number),
        "target_identity": None if target is None else repr(target),
        "acknowledged_identity": None if acknowledged is None else repr(acknowledged),
        "texture_kind": texture_kind,
        "real_plane_identity": None if real_plane is None else repr(real_plane),
        "imag_plane_identity": None if imag_plane is None else repr(imag_plane),
        "complex_mapping": complex_mapping,
        "lod": None
        if lod is None
        else {"level": int(lod.level), "factor": int(lod.factor), "gutter": int(lod.gutter)},
        "levels_generation": None
        if presentation is None
        else int(presentation.levels_generation),
        "drawable": acknowledged_identity_satisfies_target(acknowledged, target),
        "placeholder": bool(target is not None and not acknowledged_identity_satisfies_target(acknowledged, target)),
    }


def _array_plane_identity(values: np.ndarray, component: str) -> ArrayPlaneIdentity:
    array = np.asarray(values)
    return ArrayPlaneIdentity(
        component=str(component),
        pointer=int(array.__array_interface__["data"][0]),
        shape=tuple(int(value) for value in array.shape),
        strides=tuple(int(value) for value in array.strides),
        dtype=str(array.dtype),
    )


__all__ = [
    "ArrayPlaneIdentity",
    "TileIdentity",
    "TileLodIdentity",
    "TilePresentationIdentity",
    "array_plane_identities",
    "acknowledged_identity_satisfies_target",
    "complex_mapping_identity",
    "tile_ack_identity",
    "tile_truth_record",
]
