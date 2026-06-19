"""Qt-free helpers for reusing committed montage tile payloads."""

from __future__ import annotations


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
