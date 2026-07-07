"""Feedback signatures for tiled presentation work.

The adaptive governor owns feedback math.  Tiled presentation owns the work
identity: backend, payload quality, LOD, texture class, and shader/content
shape are what make one learned batch size transferable to another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TilePresentationFeedbackSignature:
    backend: str
    cost_class: str
    lifecycle: tuple[tuple[object, ...], ...]
    payload_quality: str
    lod: tuple[object, ...]
    texture: tuple[object, ...]
    shader_display: bool
    has_operations: bool


def tile_presentation_feedback_signature(session, *, backend: str) -> TilePresentationFeedbackSignature:
    """Return the presentation work class for adaptive feedback reuse."""

    payloads = dict(getattr(session, "display_tile_payloads", {}) or {})
    candidates = _candidate_tiles(session, payloads)
    lifecycle_signature = ()
    lifecycle = getattr(session, "lifecycle", None)
    feedback_signature = getattr(lifecycle, "feedback_signature", None)
    if callable(feedback_signature):
        lifecycle_signature = feedback_signature(candidates)
    return TilePresentationFeedbackSignature(
        backend=str(backend or "").lower(),
        cost_class=_tile_cost_class(session),
        lifecycle=tuple(lifecycle_signature),
        payload_quality=_payload_quality(session),
        lod=_lod_signature(session),
        texture=_texture_signature(session),
        shader_display=bool(getattr(session, "shader_display", False)),
        has_operations=bool(tuple(getattr(getattr(session, "document", None), "enabled_operations", ()) or ())),
    )


def tile_presentation_feedback_conservative_start(signature: TilePresentationFeedbackSignature) -> bool:
    return str(signature.cost_class) == "rgb_or_complex"


def _tile_cost_class(session) -> str:
    if bool(getattr(session, "rgb", False)):
        return "rgb_or_complex"
    try:
        if np.issubdtype(np.dtype(getattr(session, "output_dtype", np.float32)), np.complexfloating):
            return "rgb_or_complex"
    except TypeError:
        pass
    payloads = dict(getattr(session, "display_tile_payloads", {}) or {})
    for tile in _candidate_tiles(session, payloads):
        payload = payloads.get(int(tile))
        image = None if payload is None else getattr(payload, "image", None)
        rendered = dict(getattr(session, "rendered_tiles", {}) or {}).get(int(tile))
        if image is None and rendered is not None:
            image = getattr(rendered, "image", None)
        if _array_requires_cpu_windowed_tile_commit(image):
            return "rgb_or_complex"
    return "scalar"


def _payload_quality(session) -> str:
    payloads = dict(getattr(session, "display_tile_payloads", {}) or {})
    qualities = {
        str(getattr(payloads.get(int(tile)), "quality", "exact") or "exact")
        for tile in _candidate_tiles(session, payloads)
        if payloads.get(int(tile)) is not None
    }
    if not qualities:
        return "pending"
    if qualities == {"preview"}:
        return "preview"
    if qualities == {"exact"}:
        return "exact"
    return "mixed"


def _lod_signature(session) -> tuple[object, ...]:
    demand = getattr(getattr(session, "lod_policy_decision", None), "demand", None)
    desired = None if demand is None else int(getattr(demand, "desired_level", 0) or 0)
    applied = getattr(session, "lod_applied_level", None)
    preview = getattr(session, "lod_preview_level", None)
    payload_levels = set()
    for payload in dict(getattr(session, "display_tile_payloads", {}) or {}).values():
        lod = getattr(payload, "lod", None)
        if lod is not None:
            payload_levels.add(int(getattr(lod, "level", 0) or 0))
    return (
        desired,
        None if applied is None else int(applied),
        None if preview is None else int(preview),
        tuple(sorted(payload_levels)),
    )


def _texture_signature(session) -> tuple[object, ...]:
    payloads = dict(getattr(session, "display_tile_payloads", {}) or {})
    shapes = set()
    dtypes = set()
    kinds = set()
    for tile in _candidate_tiles(session, payloads):
        payload = payloads.get(int(tile))
        if payload is None:
            continue
        texture = getattr(payload, "texture_data", None)
        if texture is None:
            texture = getattr(payload, "image", None)
        if texture is None:
            continue
        array = np.asarray(texture)
        shapes.add(tuple(int(value) for value in array.shape))
        dtypes.add(str(array.dtype))
        kind = getattr(payload, "texture_kind", None)
        kinds.add(None if kind is None else str(getattr(kind, "value", kind)))
    if not shapes:
        return ("pending",)
    return tuple(sorted(shapes)), tuple(sorted(dtypes)), tuple(sorted(kinds, key=str))


def _candidate_tiles(session, payloads) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(
            (
                *(int(tile) for tile in getattr(session, "dirty_payloads", ()) or ()),
                *(int(tile) for tile in getattr(session, "pending_payload_upserts", ()) or ()),
                *(int(tile) for tile in payloads),
            )
        )
    )


def _array_requires_cpu_windowed_tile_commit(image) -> bool:
    if image is None:
        return False
    array = np.asarray(image)
    return bool(np.iscomplexobj(array) or (array.ndim == 3 and array.shape[-1] in (3, 4)))


__all__ = [
    "TilePresentationFeedbackSignature",
    "tile_presentation_feedback_conservative_start",
    "tile_presentation_feedback_signature",
]
