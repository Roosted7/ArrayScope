"""Chunk-grid math: which chunks a view window needs, and window-shift deltas.

This module encodes the ADR 0055 slice-window example as testable geometry:
shifting a displayed-axis window from ``100:200`` to ``101:201`` must resolve
to the already-required chunks plus at most one boundary chunk per axis —
never a full re-request.

Windows are half-open ``(start, stop)`` pairs per axis, in the same axis
order as the grid's array shape. All math is integer and Qt/GL-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class WindowDelta:
    """Chunk-origin sets produced by moving a window over a grid."""

    kept: tuple[tuple[int, ...], ...]
    added: tuple[tuple[int, ...], ...]
    dropped: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ChunkGrid:
    """A fixed partition of an N-dimensional index space into chunks.

    The grid is anchored at the origin: chunk boundaries fall at integer
    multiples of ``chunk_shape``. Edge chunks are clipped to ``array_shape``,
    so every index belongs to exactly one chunk and chunk identity is stable
    under any window movement (the property the window-shift fast path needs).
    """

    array_shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        array_shape = tuple(int(value) for value in self.array_shape)
        chunk_shape = tuple(int(value) for value in self.chunk_shape)
        if len(array_shape) != len(chunk_shape):
            raise ValueError(
                f"array shape {array_shape} and chunk shape {chunk_shape} rank mismatch"
            )
        if any(value < 0 for value in array_shape):
            raise ValueError(f"array shape must be non-negative, got {array_shape}")
        if any(value <= 0 for value in chunk_shape):
            raise ValueError(f"chunk shape must be positive, got {chunk_shape}")
        object.__setattr__(self, "array_shape", array_shape)
        object.__setattr__(self, "chunk_shape", chunk_shape)

    @property
    def rank(self) -> int:
        return len(self.array_shape)

    def grid_shape(self) -> tuple[int, ...]:
        """Number of chunks along each axis."""

        return tuple(
            0 if extent == 0 else (extent + chunk - 1) // chunk
            for extent, chunk in zip(self.array_shape, self.chunk_shape)
        )

    def chunk_count(self) -> int:
        count = 1
        for chunks in self.grid_shape():
            count *= chunks
        return count

    def origin_for_index(self, index: tuple[int, ...]) -> tuple[int, ...]:
        """Origin of the unique chunk containing ``index``."""

        self._check_rank(index)
        for axis, (value, extent) in enumerate(zip(index, self.array_shape)):
            if not 0 <= int(value) < extent:
                raise IndexError(f"index {index} outside array shape {self.array_shape} on axis {axis}")
        return tuple(
            (int(value) // chunk) * chunk for value, chunk in zip(index, self.chunk_shape)
        )

    def shape_at(self, origin: tuple[int, ...]) -> tuple[int, ...]:
        """Actual (edge-clipped) shape of the chunk anchored at ``origin``."""

        self._check_rank(origin)
        shape = []
        for value, chunk, extent in zip(origin, self.chunk_shape, self.array_shape):
            value = int(value)
            if value % chunk or not 0 <= value < max(extent, 1):
                raise IndexError(f"{origin} is not a chunk origin of this grid")
            shape.append(min(chunk, extent - value))
        return tuple(shape)

    def origins(self) -> tuple[tuple[int, ...], ...]:
        """All chunk origins of the grid, in C order."""

        axes = [
            range(0, extent, chunk)
            for extent, chunk in zip(self.array_shape, self.chunk_shape)
        ]
        return tuple(product(*axes)) if all(self.array_shape) else ()

    def origins_for_window(self, window: tuple[tuple[int, int], ...]) -> tuple[tuple[int, ...], ...]:
        """Origins of every chunk intersecting the half-open ``window``."""

        self._check_rank(window)
        axes = []
        for axis, ((start, stop), chunk, extent) in enumerate(
            zip(window, self.chunk_shape, self.array_shape)
        ):
            start, stop = int(start), int(stop)
            if start < 0 or stop > extent:
                raise IndexError(
                    f"window {window} outside array shape {self.array_shape} on axis {axis}"
                )
            if stop <= start:
                return ()
            first = (start // chunk) * chunk
            axes.append(range(first, stop, chunk))
        return tuple(product(*axes))

    def window_delta(
        self,
        old_window: tuple[tuple[int, int], ...] | None,
        new_window: tuple[tuple[int, int], ...],
    ) -> WindowDelta:
        """Chunk difference between two windows (old may be ``None`` = cold).

        This is the planning primitive for the G3 fast path: only ``added``
        chunks need materialization/upload; ``dropped`` chunks become
        eviction candidates but stay resident until memory pressure claims
        them (scrolling back must be free).
        """

        new_origins = self.origins_for_window(new_window)
        if old_window is None:
            return WindowDelta(kept=(), added=new_origins, dropped=())
        old_origins = self.origins_for_window(old_window)
        old_set = set(old_origins)
        new_set = set(new_origins)
        return WindowDelta(
            kept=tuple(origin for origin in new_origins if origin in old_set),
            added=tuple(origin for origin in new_origins if origin not in old_set),
            dropped=tuple(origin for origin in old_origins if origin not in new_set),
        )

    def _check_rank(self, value: tuple) -> None:
        if len(value) != self.rank:
            raise ValueError(f"expected rank {self.rank}, got {len(value)}: {value!r}")
