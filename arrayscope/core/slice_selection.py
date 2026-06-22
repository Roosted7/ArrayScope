"""Pure helpers for parsing and shifting dimension slice selections."""

from __future__ import annotations

from dataclasses import dataclass
import re


_ALLOWED_CHARACTERS_RE = re.compile(r"^[0-9\s:;,+\-]*$")
_REPAIR_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


@dataclass(frozen=True)
class SliceSelection:
    kind: str
    indices: tuple[int, ...]
    text: str
    style: str
    step: int | None = None
    explicit_step: bool = False


def center_index(size: int) -> int:
    """Return the centered scalar index for an axis."""

    return max(0, int(size) // 2)


def selection_text_is_allowed(text: str) -> bool:
    return bool(_ALLOWED_CHARACTERS_RE.match(str(text)))


def parse_slice_selection(text: str, axis_size: int) -> SliceSelection:
    """Parse a scalar, range, or explicit index-list selection."""

    axis_size = _validate_axis_size(axis_size)
    original = str(text).strip()
    if original == "":
        raise ValueError("selection is empty")
    if not selection_text_is_allowed(original):
        raise ValueError("selection contains unsupported characters")

    repaired = _repair_range_text(original)
    if repaired is not None:
        indices = _python_slice_indices(repaired, axis_size)
        if not indices:
            raise ValueError("selection is empty")
        return SliceSelection("range", indices, repaired, "python", step=1, explicit_step=False)

    if ":" in original:
        return _parse_range_selection(original, axis_size)

    if _looks_like_index_list(original):
        indices = tuple(_parse_list_index(part, axis_size) for part in _list_parts(original))
        if not indices:
            raise ValueError("selection is empty")
        return SliceSelection("list", indices, " ".join(str(index) for index in indices), "list")

    index = _parse_single_index(original, axis_size)
    return SliceSelection("scalar", (index,), str(index), "scalar")


def shift_slice_selection_text(text: str, delta: int, axis_size: int) -> str:
    """Shift a parsed selection while keeping its width/spacing where possible."""

    selection = parse_slice_selection(text, axis_size)
    axis_size = _validate_axis_size(axis_size)
    delta = int(delta)
    if delta == 0:
        return selection.text
    if selection.kind == "scalar":
        index = _clamp(selection.indices[0] + delta, 0, axis_size - 1)
        return str(index)

    step = abs(int(selection.step or 1))
    requested_shift = delta * step
    shifted = _bounded_shift(selection.indices, requested_shift, axis_size)
    if selection.kind == "list":
        return " ".join(str(index) for index in shifted)
    if selection.style == "matlab":
        return _format_matlab_range(shifted, selection.step or 1, selection.explicit_step)
    return _format_python_range(shifted, selection.step or 1, selection.explicit_step, axis_size)


def _validate_axis_size(axis_size: int) -> int:
    axis_size = int(axis_size)
    if axis_size < 1:
        raise ValueError(f"axis_size must be at least 1, got {axis_size}")
    return axis_size


def _repair_range_text(text: str) -> str | None:
    match = _REPAIR_RANGE_RE.match(text)
    if match is None:
        return None
    return f"{int(match.group(1))}:{int(match.group(2))}"


def _looks_like_index_list(text: str) -> bool:
    return "," in text or ";" in text or len(text.split()) > 1


def _list_parts(text: str) -> tuple[str, ...]:
    return tuple(part for part in re.split(r"[\s,;]+", text.strip()) if part)


def _parse_single_index(text: str, axis_size: int) -> int:
    try:
        index = int(str(text).strip())
    except ValueError as exc:
        raise ValueError(f"invalid index {text!r}") from exc
    if index < 0:
        index += int(axis_size)
    if index < 0 or index >= int(axis_size):
        raise ValueError(f"index {index} is out of bounds for axis size {axis_size}")
    return int(index)


def _parse_list_index(text: str, axis_size: int) -> int:
    try:
        index = int(str(text).strip())
    except ValueError as exc:
        raise ValueError(f"invalid index {text!r}") from exc
    if index < 0:
        index += int(axis_size)
    return _clamp(index, 0, int(axis_size) - 1)


def _parse_range_selection(text: str, axis_size: int) -> SliceSelection:
    python = _try_python_range(text, axis_size)
    matlab = _try_matlab_range(text, axis_size)
    if python is None and matlab is None:
        raise ValueError("selection range is invalid")
    if python is None:
        return matlab
    if matlab is not None and (len(python.indices) == 0 or (len(python.indices) <= 1 and len(matlab.indices) > 1)):
        return matlab
    if not python.indices:
        raise ValueError("selection is empty")
    return python


def _try_python_range(text: str, axis_size: int) -> SliceSelection | None:
    raw_parts = str(text).split(":")
    if len(raw_parts) > 3:
        return None
    try:
        parts = [_parse_optional_int(part) for part in raw_parts]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.append(None)
    start, stop, step = parts[:3]
    if step == 0:
        return None
    step = 1 if step is None else int(step)
    indices = tuple(range(*slice(start, stop, step).indices(int(axis_size))))
    return SliceSelection(
        "range",
        indices,
        str(text).strip(),
        "python",
        step=step,
        explicit_step=len(raw_parts) == 3,
    )


def _try_matlab_range(text: str, axis_size: int) -> SliceSelection | None:
    raw_parts = str(text).split(":")
    if len(raw_parts) != 3 or raw_parts[1].strip() == "":
        return None
    try:
        start = _parse_optional_int(raw_parts[0])
        step = _parse_optional_int(raw_parts[1])
        stop = _parse_optional_int(raw_parts[2])
    except ValueError:
        return None
    if step is None or step == 0:
        return None
    step = int(step)
    if start is None:
        start = 0 if step > 0 else int(axis_size) - 1
    else:
        start = _resolve_endpoint(start, axis_size)
    if stop is None:
        stop = int(axis_size) - 1 if step > 0 else 0
    else:
        stop = _resolve_endpoint(stop, axis_size)

    current = _clamp(start, 0, int(axis_size) - 1)
    end = _clamp(stop, 0, int(axis_size) - 1)
    indices: list[int] = []
    if step > 0:
        while current <= end:
            indices.append(current)
            current += step
    else:
        while current >= end:
            indices.append(current)
            current += step
    if not indices:
        return None
    return SliceSelection("range", tuple(indices), str(text).strip(), "matlab", step=step, explicit_step=True)


def _parse_optional_int(text: str) -> int | None:
    text = str(text).strip()
    if text == "":
        return None
    return int(text)


def _python_slice_indices(text: str, axis_size: int) -> tuple[int, ...]:
    selection = _try_python_range(text, axis_size)
    if selection is None:
        raise ValueError("selection range is invalid")
    return selection.indices


def _resolve_endpoint(value: int, axis_size: int) -> int:
    value = int(value)
    if value < 0:
        value += int(axis_size)
    return value


def _bounded_shift(indices: tuple[int, ...], requested_shift: int, axis_size: int) -> tuple[int, ...]:
    if not indices:
        return ()
    low = min(indices)
    high = max(indices)
    requested_shift = int(requested_shift)
    actual = _clamp(requested_shift, -low, int(axis_size) - 1 - high)
    return tuple(_clamp(index + actual, 0, int(axis_size) - 1) for index in indices)


def _format_matlab_range(indices: tuple[int, ...], step: int, explicit_step: bool) -> str:
    start = indices[0]
    stop = indices[-1]
    if explicit_step:
        return f"{start}:{int(step)}:{stop}"
    return f"{start}:{stop}"


def _format_python_range(indices: tuple[int, ...], step: int, explicit_step: bool, axis_size: int) -> str:
    start = indices[0]
    stop = _python_exclusive_stop(indices[-1], int(step), int(axis_size))
    stop_text = "" if stop is None else str(stop)
    if explicit_step or int(step) != 1:
        return f"{start}:{stop_text}:{int(step)}"
    return f"{start}:{stop_text}"


def _python_exclusive_stop(last: int, step: int, axis_size: int) -> int | None:
    stop = int(last) + int(step)
    if step > 0:
        return min(int(axis_size), stop)
    if stop < 0:
        return None
    return stop


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(int(lower), min(int(upper), int(value)))
