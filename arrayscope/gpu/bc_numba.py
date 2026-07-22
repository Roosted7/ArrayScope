"""Optional, explicitly prewarmed Numba accelerator for BC4 encoding.

Importing this module defines the JIT kernel but does not compile it. Call
``prewarm()`` from background work before using the non-blocking entry points.
"""

from __future__ import annotations

import threading

import numba
import numpy as np

_READY = threading.Event()
_WARM_LOCK = threading.Lock()


@numba.njit(nogil=True, cache=True)
def _encode_u8(
    field_u8: np.ndarray,
    field_unit: np.ndarray,
    valid_height: int,
    valid_width: int,
) -> tuple[np.ndarray, float, float, int]:
    """Encode quantized input and accumulate exact round-trip error."""

    height, width = field_u8.shape
    padded_height = ((height + 3) // 4) * 4
    padded_width = ((width + 3) // 4) * 4
    blocks_x = padded_width // 4
    encoded = np.empty((padded_height // 4) * blocks_x * 8, dtype=np.uint8)
    squared_error = 0.0
    max_abs_error = 0.0
    sample_count = 0

    for block_y in range(padded_height // 4):
        for block_x in range(blocks_x):
            lo = 255
            hi = 0
            for local_y in range(4):
                source_y = min(block_y * 4 + local_y, height - 1)
                for local_x in range(4):
                    source_x = min(block_x * 4 + local_x, width - 1)
                    value = int(field_u8[source_y, source_x])
                    lo = min(lo, value)
                    hi = max(hi, value)

            palette = np.empty(8, dtype=np.float64)
            palette[0] = hi
            palette[1] = lo
            for index in range(1, 7):
                palette[index + 1] = ((7 - index) * hi + index * lo) / 7.0

            packed_indices = np.uint64(0)
            texel = 0
            for local_y in range(4):
                source_y = min(block_y * 4 + local_y, height - 1)
                for local_x in range(4):
                    source_x = min(block_x * 4 + local_x, width - 1)
                    value = float(field_u8[source_y, source_x])
                    best_index = 0
                    best_distance = abs(value - palette[0])
                    for palette_index in range(1, 8):
                        distance = abs(value - palette[palette_index])
                        if distance < best_distance:
                            best_distance = distance
                            best_index = palette_index
                    packed_indices |= np.uint64(best_index) << np.uint64(3 * texel)
                    output_y = block_y * 4 + local_y
                    output_x = block_x * 4 + local_x
                    if output_y < valid_height and output_x < valid_width:
                        decoded_value = np.float32(palette[best_index] / 255.0)
                        error = abs(float(field_unit[output_y, output_x]) - decoded_value)
                        squared_error += error * error
                        max_abs_error = max(max_abs_error, error)
                        sample_count += 1
                    texel += 1

            offset = (block_y * blocks_x + block_x) * 8
            encoded[offset] = np.uint8(hi)
            encoded[offset + 1] = np.uint8(lo)
            for byte_index in range(6):
                encoded[offset + 2 + byte_index] = np.uint8(
                    (packed_indices >> np.uint64(8 * byte_index)) & np.uint64(0xFF)
                )

    return encoded, squared_error, max_abs_error, sample_count


def prewarm() -> None:
    """Compile the kernel once; safe to call concurrently from background work."""

    if _READY.is_set():
        return
    with _WARM_LOCK:
        if _READY.is_set():
            return
        field = np.zeros((4, 4), dtype=np.float32)
        _encode_u8(np.zeros((4, 4), dtype=np.uint8), field, 4, 4)
        _READY.set()


def ready() -> bool:
    return _READY.is_set()


def encode_if_ready(field_unit: np.ndarray) -> tuple[bytes, int, int] | None:
    """Encode without compiling, or return ``None`` while not prewarmed."""

    if not _READY.is_set():
        return None
    field = np.clip(np.ascontiguousarray(field_unit, dtype=np.float32), 0.0, 1.0)
    # Preserve the NumPy reference's half-way tie behavior exactly.
    field_u8 = np.rint(field * 255.0).astype(np.uint8)
    encoded, _squared_error, _max_abs_error, _sample_count = _encode_u8(
        field_u8, field, field.shape[0], field.shape[1]
    )
    height, width = field.shape
    return encoded.tobytes(), height, width


def encode_quality_if_ready(
    field_unit: np.ndarray,
    *,
    valid_shape: tuple[int, int] | None = None,
) -> tuple[bytes, int, int, float, float, int] | None:
    """Encode and return exact error accumulators without a decode pass."""

    if not _READY.is_set():
        return None
    field = np.clip(np.ascontiguousarray(field_unit, dtype=np.float32), 0.0, 1.0)
    field_u8 = np.rint(field * 255.0).astype(np.uint8)
    height, width = field.shape
    if valid_shape is None:
        valid_height, valid_width = height, width
    else:
        valid_height = max(0, min(height, int(valid_shape[0])))
        valid_width = max(0, min(width, int(valid_shape[1])))
    encoded, squared_error, max_abs_error, sample_count = _encode_u8(
        field_u8, field, valid_height, valid_width
    )
    return (
        encoded.tobytes(),
        height,
        width,
        float(squared_error),
        float(max_abs_error),
        int(sample_count),
    )
