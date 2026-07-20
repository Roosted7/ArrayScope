import pytest

from arrayscope.core.view_state import ChannelMode, ScaleMode, ViewState
from arrayscope.sync.messages import (
    FACET_DIMS,
    FACET_LEVELS,
    FACETS,
    decode_lines,
    dimension_state_payload,
    encode_message,
    merged_dimension_state,
    merged_slice_indices,
    parse_message,
    request_message,
    state_message,
)


def test_state_message_round_trip_through_codec():
    message = state_message(
        FACET_LEVELS, "origin-a", 3, {"levels": [0.0, 1.5], "window_mode": "relative"}
    )
    messages, remainder = decode_lines(encode_message(message))
    assert remainder == b""
    assert messages == [message]


def test_decode_lines_keeps_partial_line_and_drops_junk():
    good = state_message(FACET_DIMS, "origin-a", 1, {"shape": [4], "slice_indices": [2]})
    buffer = b"not json\n" + encode_message(good) + b'{"v":1,"kind":"state"}\n' + b'{"partial'
    messages, remainder = decode_lines(buffer)
    assert messages == [good]
    assert remainder == b'{"partial'


def test_parse_message_rejects_wrong_version_kind_facet_and_origin():
    valid = state_message(FACET_DIMS, "origin-a", 1, {})
    assert parse_message(valid) is valid
    assert parse_message({**valid, "v": 2}) is None
    assert parse_message({**valid, "kind": "gossip"}) is None
    assert parse_message({**valid, "facet": "viewport"}) is None
    assert parse_message({**valid, "origin": ""}) is None
    assert parse_message({**valid, "revision": "7"}) is None
    assert parse_message({**valid, "payload": None}) is None


def test_request_message_has_no_payload_requirement():
    for facet in FACETS:
        message = request_message(facet, "origin-b")
        assert parse_message(message) is message


def test_unknown_facet_raises_for_builders():
    with pytest.raises(ValueError):
        state_message("viewport", "o", 1, {})
    with pytest.raises(ValueError):
        request_message("viewport", "o")


def test_merged_slice_indices_clamps_per_dimension():
    # Peer index 9 exceeds axis size 5 -> clamp; negative clamps to 0.
    assert merged_slice_indices((5, 3), (0, 0), (9, -2)) == (4, 0)


def test_merged_slice_indices_ignores_extra_peer_dimensions():
    assert merged_slice_indices((5, 3), (1, 2), (4, 1, 7, 7)) == (4, 1)


def test_merged_slice_indices_keeps_own_index_for_missing_dimensions():
    assert merged_slice_indices((5, 3, 8), (1, 2, 6), (0,)) == (0, 2, 6)


def test_dimension_state_payload_round_trips_full_view_state_subset():
    incoming = (
        ViewState.from_shape((4, 5, 6))
        .with_image_axes(1, 2)
        .with_slice(0, 3)
        .with_axis_flipped(2, True)
        .with_axis_fftshifted(1, True)
        .with_axis_range(1, (0, 2, 4), "0:2:5")
    )
    current = (
        ViewState.from_shape((4, 5, 6)).with_channel(ChannelMode.ABS).with_scale(ScaleMode.LOG)
    )

    merged = merged_dimension_state(current, dimension_state_payload(incoming))

    assert merged.image_axes == incoming.image_axes
    assert merged.line_axis == incoming.line_axis
    assert merged.slice_indices == incoming.slice_indices
    assert merged.axis_flipped == incoming.axis_flipped
    assert merged.axis_fftshifted == incoming.axis_fftshifted
    assert merged.axis_range_indices == incoming.axis_range_indices
    assert merged.axis_range_text == incoming.axis_range_text
    assert merged.channel == ChannelMode.ABS
    assert merged.scale == ScaleMode.LOG
    payload = dimension_state_payload(merged)
    assert "channel" not in payload
    assert "scale" not in payload


def test_merged_dimension_state_clamps_for_smaller_receiver_and_ignores_extra_axes():
    incoming = ViewState.from_shape((5, 3, 8)).with_slice(0, 4).with_slice(1, 2).with_slice(2, 7)
    current = ViewState.from_shape((3, 2)).with_image_axes(1, 0)

    merged = merged_dimension_state(current, dimension_state_payload(incoming))

    assert merged.shape == (3, 2)
    assert merged.slice_indices == (2, 1)
    assert merged.image_axes == (0, 1)


def test_merged_dimension_state_migrates_image_axes_when_sender_axis_is_missing():
    incoming = ViewState.from_shape((4, 5, 6)).with_image_axes(1, 2)
    current = ViewState.from_shape((4, 5))

    merged = merged_dimension_state(current, dimension_state_payload(incoming))

    assert merged.image_axes == (1, 0)


def test_merged_dimension_state_preserves_valid_montage_range_on_receiver():
    incoming = ViewState.from_shape((4, 5, 6)).with_montage_axis(
        2, columns=2, indices=(0, 3, 5), text="0 3 5"
    )
    current = ViewState.from_shape((4, 5, 4))

    merged = merged_dimension_state(current, dimension_state_payload(incoming))

    assert merged.montage_axis == 2
    assert merged.montage_columns == 2
    assert merged.montage_indices == (0, 3)
    assert merged.montage_text == "0 3 5"


def test_merged_dimension_state_accepts_legacy_slice_only_payload():
    current = (
        ViewState.from_shape((5, 3))
        .with_image_axes(1, 0)
        .with_axis_flipped(1, True)
        .with_channel(ChannelMode.ABS)
    )

    merged = merged_dimension_state(current, {"shape": [5, 3, 8], "slice_indices": [4, 1, 7]})

    assert merged.slice_indices == (4, 1)
    assert merged.image_axes == (1, 0)
    assert merged.axis_flipped == current.axis_flipped
    assert merged.channel == ChannelMode.ABS
