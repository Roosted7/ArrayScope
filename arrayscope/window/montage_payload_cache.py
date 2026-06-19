"""Qt-free helpers for reusing committed montage tile payloads."""

from __future__ import annotations

import numpy as np

from arrayscope.core.view_state import ChannelMode
from arrayscope.display.shader_mapping import TexturePlaneKind


def previous_tiled_payloads(frame) -> dict[int, object]:
    source = None if frame is None else getattr(frame, "value_source", None)
    payloads = getattr(source, "payloads", None)
    return {} if payloads is None else dict(payloads)


def base_tile_source_id(source_id) -> object | None:
    if isinstance(source_id, tuple) and len(source_id) >= 3 and source_id[1] == "texture_kind":
        return source_id[0]
    return source_id


def previous_tiled_payloads_by_base_source(frame) -> dict[object, object]:
    return {
        base_tile_source_id(payload.source_id): payload
        for payload in previous_tiled_payloads(frame).values()
        if base_tile_source_id(payload.source_id) is not None
    }


def limited_payload_cache(existing, payloads, *, limit: int = 4096) -> dict[object, object]:
    cache = dict(existing or {})
    for payload in dict(payloads or {}).values():
        key = base_tile_source_id(payload.source_id)
        if key is not None:
            cache[key] = payload
    if len(cache) <= int(limit):
        return cache
    return dict(tuple(cache.items())[-int(limit) :])


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
