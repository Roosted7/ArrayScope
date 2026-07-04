"""Qt-free helpers for reusing committed montage tile payloads."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arrayscope.core.bounded_cache import BoundedCache
from arrayscope.core.view_state import ChannelMode
from arrayscope.display.shader_mapping import TexturePlaneKind


def previous_tiled_payloads(frame) -> dict[int, object]:
    source = None if frame is None else getattr(frame, "value_source", None)
    payloads = getattr(source, "payloads", None)
    return {} if payloads is None else dict(payloads)


def base_tile_source_id(source_id) -> object | None:
    if isinstance(source_id, tuple) and len(source_id) >= 3 and source_id[1] == "texture_kind":
        return source_id[0]
    if isinstance(source_id, tuple) and "texture_kind" in source_id:
        marker = source_id.index("texture_kind")
        prefix = source_id[:marker]
        if not prefix:
            return None
        return prefix[0] if len(prefix) == 1 else prefix
    return source_id


def previous_tiled_payloads_by_base_source(frame) -> dict[object, object]:
    return {
        base_tile_source_id(payload.source_id): payload
        for payload in previous_tiled_payloads(frame).values()
        if base_tile_source_id(payload.source_id) is not None
    }


@dataclass
class RetainedTiledPayloadStore:
    """Acknowledged tiled payloads retained outside the current frame.

    The committed frame may temporarily describe a single image or a narrowed
    tiled view.  This store keeps the broader resident source set available for
    the next compatible tiled montage session without treating requested-but-not
    acknowledged deltas as reusable presentation state.
    """

    limit: int = 4096
    _payloads: BoundedCache = field(default_factory=BoundedCache)
    last_clear_reason: str = ""

    def __post_init__(self) -> None:
        self.limit = max(1, int(self.limit))
        self._payloads.resize(max_entries=self.limit)

    def remember_acknowledged(self, payloads, *, limit: int | None = None) -> None:
        max_items = max(1, int(self.limit if limit is None else limit))
        # Bound the store before insertion.  Large montage commits can hand us
        # thousands of acknowledged payloads; retaining all of them and only
        # then trimming creates an avoidable allocation spike.
        self._payloads.resize(max_entries=max_items)
        values = () if payloads is None else payloads.values()
        for payload in values:
            key = base_tile_source_id(getattr(payload, "source_id", None))
            if key is None or not payload_matches_texture_kind(payload):
                continue
            self._payloads.put(key, payload)

    def resolve(self, tile_key, lod_factor: int | None, tile_state, *, shader_display: bool):
        """Return a retained payload for ``tile_key`` or None.

        ``lod_factor=None`` accepts any retained level (ADR 0050 resident
        policy): the payload carries native exact/semantic planes, so the
        session re-selects the presented level from the live demand and a
        stale-level texture is presented until its replacement is resident,
        never a placeholder.
        """

        payload = self._payloads.peek(tile_key)
        if payload is None:
            return None
        if lod_factor is not None and not payload_lod_matches(payload, lod_factor):
            return None
        if not payload_compatible_with_tile(payload, tile_state, shader_display=shader_display):
            return None
        return payload

    def payloads_by_base_source(self, *, lod_factor: int | None = None) -> dict[object, object]:
        payloads = dict(self._payloads.items())
        if lod_factor is None:
            return payloads
        return {
            key: payload
            for key, payload in payloads.items()
            if payload_lod_matches(payload, lod_factor)
        }

    def clear_for_document_or_context_change(self, reason: str) -> None:
        self._payloads.clear()
        self.last_clear_reason = str(reason)


def payload_lod_matches(payload, factor: int) -> bool:
    lod = getattr(payload, "lod", None)
    payload_factor = int(getattr(lod, "factor", 1) or 1)
    return payload_factor == max(1, int(factor))


def payload_matches_texture_kind(payload) -> bool:
    kind = _coerce_texture_kind(getattr(payload, "texture_kind", None))
    texture = getattr(payload, "texture_data", None)
    if texture is None:
        texture = getattr(payload, "semantic_data", None)
    if texture is None:
        texture = getattr(payload, "image", None)
    if texture is None:
        return False
    arr = np.asarray(texture)
    if arr.ndim < 2:
        return False
    if kind == TexturePlaneKind.COMPLEX_RG32F:
        return np.iscomplexobj(arr) or (arr.ndim == 3 and arr.shape[-1] == 2)
    if kind == TexturePlaneKind.RGB8:
        return arr.ndim == 3 and arr.shape[-1] in (3, 4)
    if kind == TexturePlaneKind.SCALAR_R32F:
        return arr.ndim == 2 and not np.iscomplexobj(arr)
    return False


def payload_compatible_with_tile(payload, tile_state, *, shader_display: bool) -> bool:
    """Return whether a previous tile payload may be reused for this tile.

    The semantic source key already proves that the tile refers to the same
    data request.  This guard checks the presentation contract: a payload
    marked as complex must actually carry complex/RG texture data, and complex
    shader montages must not resurrect old RGB/windowed tile wrappers.
    """

    if not payload_matches_texture_kind(payload):
        return False
    channel = getattr(tile_state, "channel", None)
    try:
        channel = ChannelMode(getattr(channel, "value", channel))
    except Exception:
        channel = None
    kind = _coerce_texture_kind(getattr(payload, "texture_kind", None))
    if bool(shader_display) and channel in {ChannelMode.COMPLEX, ChannelMode.ANGLE}:
        return kind == TexturePlaneKind.COMPLEX_RG32F
    return True


def _coerce_texture_kind(kind) -> TexturePlaneKind:
    if kind is None:
        return TexturePlaneKind.SCALAR_R32F
    if isinstance(kind, TexturePlaneKind):
        return kind
    return TexturePlaneKind(getattr(kind, "value", kind))
