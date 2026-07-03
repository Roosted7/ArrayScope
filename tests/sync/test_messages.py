import pytest

from arrayscope.sync.messages import (
    FACET_DIMS,
    FACET_LEVELS,
    FACETS,
    decode_lines,
    encode_message,
    merged_slice_indices,
    parse_message,
    request_message,
    state_message,
)


def test_state_message_round_trip_through_codec():
    message = state_message(FACET_LEVELS, "origin-a", 3, {"levels": [0.0, 1.5], "window_mode": "relative"})
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
