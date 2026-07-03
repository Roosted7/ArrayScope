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
from arrayscope.core.view_state import ViewState


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

DIMENSION_STATE_FIELDS = (
    "shape",
    "image_axes",
    "line_axis",
    "slice_indices",
    "axis_flipped",
    "axis_fftshifted",
    "montage_axis",
    "montage_columns",
    "montage_indices",
    "montage_text",
    "axis_range_indices",
    "axis_range_text",
)


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


def dimension_state_payload(state: ViewState) -> dict[str, object]:
    """Serialize the dimension-owned subset of ``ViewState``.

    Channel, scale, levels, and colormap are intentionally outside this facet.
    """

    return {
        "shape": [int(size) for size in state.shape],
        "image_axes": None if state.image_axes is None else [int(axis) for axis in state.image_axes],
        "line_axis": state.line_axis,
        "slice_indices": [int(index) for index in state.slice_indices],
        "axis_flipped": [bool(value) for value in state.axis_flipped],
        "axis_fftshifted": [bool(value) for value in state.axis_fftshifted],
        "montage_axis": state.montage_axis,
        "montage_columns": state.montage_columns,
        "montage_indices": None if state.montage_indices is None else [int(index) for index in state.montage_indices],
        "montage_text": state.montage_text,
        "axis_range_indices": [None if value is None else [int(index) for index in value] for value in state.axis_range_indices],
        "axis_range_text": list(state.axis_range_text),
    }


def merged_dimension_state(current_state: ViewState, payload) -> ViewState:
    """Merge a peer's dimension state into this window's current shape.

    Full dimension payloads are interpreted against the sender shape first and
    then migrated to the receiver shape. Legacy thin payloads that only carry
    slice indices keep their historical behavior.
    """

    if not isinstance(payload, dict):
        payload = {}
    if not any(field in payload for field in DIMENSION_STATE_FIELDS if field not in {"shape", "slice_indices"}):
        merged = merged_slice_indices(current_state.shape, current_state.slice_indices, payload.get("slice_indices", ()))
        return current_state.with_slice_indices(merged)

    sender_shape = _shape_from_payload(payload.get("shape"), current_state.shape)
    base = current_state.for_shape(sender_shape, preserve_flags=True)
    incoming = ViewState(
        ndim=len(sender_shape),
        shape=sender_shape,
        image_axes=_optional_axis_pair(payload.get("image_axes", base.image_axes)),
        line_axis=payload.get("line_axis", base.line_axis),
        slice_indices=_fixed_int_sequence(payload.get("slice_indices"), base.slice_indices, len(sender_shape)),
        channel=current_state.channel,
        scale=current_state.scale,
        axis_flipped=_fixed_bool_sequence(payload.get("axis_flipped"), base.axis_flipped, len(sender_shape)),
        axis_fftshifted=_fixed_bool_sequence(payload.get("axis_fftshifted"), base.axis_fftshifted, len(sender_shape)),
        montage_axis=payload.get("montage_axis", base.montage_axis),
        montage_columns=payload.get("montage_columns", base.montage_columns),
        montage_indices=_optional_int_tuple(payload.get("montage_indices", base.montage_indices)),
        montage_text=payload.get("montage_text", base.montage_text),
        axis_range_indices=_fixed_optional_int_sequences(
            payload.get("axis_range_indices"),
            base.axis_range_indices,
            len(sender_shape),
        ),
        axis_range_text=_fixed_optional_text_sequence(
            payload.get("axis_range_text"),
            base.axis_range_text,
            len(sender_shape),
        ),
    )
    return incoming.for_shape(current_state.shape, preserve_flags=True)


def _shape_from_payload(value, fallback) -> tuple[int, ...]:
    shape = tuple(int(size) for size in (fallback if value is None else value))
    if not shape:
        raise ValueError("dimension sync payload shape must not be empty")
    return shape


def _optional_axis_pair(value):
    if value is None:
        return None
    values = tuple(int(axis) for axis in value)
    if len(values) != 2:
        raise ValueError("dimension sync image_axes must contain exactly two axes")
    return values


def _optional_int_tuple(value):
    if value is None:
        return None
    return tuple(int(item) for item in value)


def _fixed_int_sequence(value, fallback, length: int) -> tuple[int, ...]:
    values = list(fallback if value is None else value)
    fallback_values = tuple(int(item) for item in fallback)
    while len(values) < length:
        values.append(fallback_values[len(values)] if len(values) < len(fallback_values) else 0)
    return tuple(int(item) for item in values[:length])


def _fixed_bool_sequence(value, fallback, length: int) -> tuple[bool, ...]:
    values = list(fallback if value is None else value)
    fallback_values = tuple(bool(item) for item in fallback)
    while len(values) < length:
        values.append(fallback_values[len(values)] if len(values) < len(fallback_values) else False)
    return tuple(bool(item) for item in values[:length])


def _fixed_optional_int_sequences(value, fallback, length: int) -> tuple[tuple[int, ...] | None, ...]:
    values = list(fallback if value is None else value)
    fallback_values = tuple(fallback)
    while len(values) < length:
        values.append(fallback_values[len(values)] if len(values) < len(fallback_values) else None)
    return tuple(None if item is None else tuple(int(index) for index in item) for item in values[:length])


def _fixed_optional_text_sequence(value, fallback, length: int) -> tuple[str | None, ...]:
    values = list(fallback if value is None else value)
    fallback_values = tuple(fallback)
    while len(values) < length:
        values.append(fallback_values[len(values)] if len(values) < len(fallback_values) else None)
    return tuple(None if item is None else str(item) for item in values[:length])
