"""Qt-free typed message vocabulary for linked-window sync.

Messages are one-line UTF-8 JSON envelopes. Every state message carries an
``origin`` (the publishing window's id) and a per-facet ``revision`` so
receivers can drop echoes and stale duplicates instead of re-broadcasting
them (feedback-loop prevention through origin/revision ids, per the roadmap
"Linked windows and inspection groups" item).
"""

from __future__ import annotations

import json

from arrayscope.core.axis_utils import clamp_index


SYNC_PROTOCOL_VERSION = 1

DEFAULT_GROUP = "default"

FACET_LEVELS = "levels"
FACET_DIMS = "dims"
FACET_OPERATIONS = "operations"
FACET_ROIS = "rois"
FACETS = (FACET_LEVELS, FACET_DIMS, FACET_OPERATIONS, FACET_ROIS)

KIND_STATE = "state"
KIND_REQUEST = "request"
KINDS = (KIND_STATE, KIND_REQUEST)


def state_message(facet, origin, revision, payload, *, group=DEFAULT_GROUP):
    """Build a facet state envelope."""

    if facet not in FACETS:
        raise ValueError(f"unknown sync facet: {facet!r}")
    return {
        "v": SYNC_PROTOCOL_VERSION,
        "kind": KIND_STATE,
        "group": str(group),
        "facet": str(facet),
        "origin": str(origin),
        "revision": int(revision),
        "payload": dict(payload),
    }


def request_message(facet, origin, *, group=DEFAULT_GROUP):
    """Build a join-time request: peers respond with their current state."""

    if facet not in FACETS:
        raise ValueError(f"unknown sync facet: {facet!r}")
    return {
        "v": SYNC_PROTOCOL_VERSION,
        "kind": KIND_REQUEST,
        "group": str(group),
        "facet": str(facet),
        "origin": str(origin),
    }


def parse_message(mapping):
    """Return a validated message mapping, or ``None`` if it is not usable."""

    if not isinstance(mapping, dict):
        return None
    if mapping.get("v") != SYNC_PROTOCOL_VERSION:
        return None
    if mapping.get("kind") not in KINDS:
        return None
    if mapping.get("facet") not in FACETS:
        return None
    origin = mapping.get("origin")
    if not isinstance(origin, str) or not origin:
        return None
    if mapping.get("kind") == KIND_STATE:
        if not isinstance(mapping.get("payload"), dict):
            return None
        if not isinstance(mapping.get("revision"), int):
            return None
    return mapping


def encode_message(mapping) -> bytes:
    """Encode one message as a newline-terminated compact JSON line."""

    return json.dumps(mapping, separators=(",", ":")).encode("utf-8") + b"\n"


def decode_lines(buffer: bytes):
    """Split ``buffer`` into parsed messages plus the unterminated remainder.

    Unparseable or invalid lines are dropped; the transport must never raise
    on peer input.
    """

    messages = []
    remainder = buffer
    while b"\n" in remainder:
        line, remainder = remainder.split(b"\n", 1)
        line = line.strip()
        if not line:
            continue
        try:
            mapping = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        message = parse_message(mapping)
        if message is not None:
            messages.append(message)
    return messages, remainder


def merged_slice_indices(shape, current_indices, incoming_indices):
    """Merge a peer's slice indices into this window's shape.

    Dimensions are matched by position. Received indices are clamped per
    dimension; dimensions the receiver does not have are ignored; dimensions
    the sender did not have keep their current index.
    """

    shape = tuple(int(size) for size in shape)
    current = tuple(int(index) for index in current_indices)
    incoming = tuple(int(index) for index in incoming_indices)
    merged = []
    for axis in range(len(shape)):
        index = incoming[axis] if axis < len(incoming) else current[axis]
        merged.append(clamp_index(shape, axis, index))
    return tuple(merged)
