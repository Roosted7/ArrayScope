"""Finite level/bounds helpers that avoid large temporary finite copies."""

from __future__ import annotations

import math

import numpy as np


def finite_bounds(data, *, exact_limit=4_000_000, max_samples=1_000_000):
    array = np.asarray(data)
    if array.size == 0:
        return None
    if array.size > int(exact_limit):
        step = max(1, int(math.ceil(math.sqrt(array.size / max(1, int(max_samples))))))
        sample = array[tuple(slice(None, None, step) for _axis in range(array.ndim))]
    else:
        sample = array
    try:
        minimum = np.nanmin(sample)
        maximum = np.nanmax(sample)
    except (TypeError, ValueError, FloatingPointError):
        return None
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return None
    return (float(minimum), float(maximum))
