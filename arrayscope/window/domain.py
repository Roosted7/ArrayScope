"""Window-local domain state for dimension transform indicators."""

from __future__ import annotations

from enum import Enum


class Domain(Enum):
    INV_FOURIER = -1
    NATIVE = 0
    FOURIER = 1
