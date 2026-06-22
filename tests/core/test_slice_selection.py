import pytest

from arrayscope.core.slice_selection import (
    center_index,
    parse_slice_selection,
    selection_text_is_allowed,
    shift_slice_selection_text,
)


def test_center_index_uses_fft_center_convention():
    assert center_index(1) == 0
    assert center_index(5) == 2
    assert center_index(6) == 3


def test_python_slice_is_default_and_stop_exclusive():
    selection = parse_slice_selection("0:100:2", 12)

    assert selection.indices == (0, 2, 4, 6, 8, 10)
    assert selection.text == "0:100:2"
    assert selection.style == "python"


def test_full_and_reverse_python_slices():
    assert parse_slice_selection(":", 4).indices == (0, 1, 2, 3)
    assert parse_slice_selection("::-1", 4).indices == (3, 2, 1, 0)


def test_python_negative_scalar_and_slice_bounds():
    assert parse_slice_selection("-1", 5).indices == (4,)
    assert parse_slice_selection("-4:-1", 5).indices == (1, 2, 3)


def test_zero_step_is_rejected():
    with pytest.raises(ValueError):
        parse_slice_selection("0:0:4", 8)


def test_matlab_fallback_preserves_valid_matlab_text():
    selection = parse_slice_selection("0:2:100", 10)

    assert selection.indices == (0, 2, 4, 6, 8)
    assert selection.text == "0:2:100"
    assert selection.style == "matlab"


def test_descending_matlab_fallback():
    selection = parse_slice_selection("10:-1:0", 12)

    assert selection.indices == tuple(range(10, -1, -1))
    assert selection.text == "10:-1:0"
    assert selection.style == "matlab"


def test_ambiguous_three_part_range_prefers_python_when_both_are_meaningful():
    selection = parse_slice_selection("0:4:2", 10)

    assert selection.indices == (0, 2)
    assert selection.style == "python"


def test_dash_ranges_are_repaired_to_python_text():
    assert parse_slice_selection("0-4", 10).text == "0:4"
    selection = parse_slice_selection("0 - 4", 10)
    assert selection.indices == (0, 1, 2, 3)
    assert selection.text == "0:4"


def test_raw_index_lists_normalize_to_spaces():
    assert parse_slice_selection("0 5 8", 10).text == "0 5 8"
    assert parse_slice_selection("0,5,8", 10).text == "0 5 8"
    assert parse_slice_selection("0; 5, 8", 10).text == "0 5 8"
    assert parse_slice_selection("0 50", 10).text == "0 9"


def test_invalid_characters_and_ranges_are_rejected():
    assert not selection_text_is_allowed("abc")
    assert not selection_text_is_allowed("0#4")
    with pytest.raises(ValueError):
        parse_slice_selection("abc", 10)
    with pytest.raises(ValueError):
        parse_slice_selection("0:1:2:3", 10)


def test_shifting_scalar_range_and_list():
    assert shift_slice_selection_text("5", 1, 10) == "6"
    assert shift_slice_selection_text("0:4", 1, 10) == "1:5"
    assert shift_slice_selection_text("0:2:8", 1, 12) == "2:2:10"
    assert shift_slice_selection_text("0 3 5", 1, 10) == "1 4 6"


def test_shifting_clamps_whole_selection_window():
    assert shift_slice_selection_text("0:4", -1, 10) == "0:4"
    assert shift_slice_selection_text("6:10", 1, 10) == "6:10"
