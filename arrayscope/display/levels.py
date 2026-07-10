"""Finite level/bounds helpers that avoid large temporary finite copies."""

from __future__ import annotations

import math
import warnings

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            minimum = np.nanmin(sample)
            maximum = np.nanmax(sample)
            if not np.isfinite(minimum) or not np.isfinite(maximum):
                # ±Inf in the data must not blow the window to the float
                # range (it hides all finite structure). Mask only in this
                # rare path so the common case stays copy-free.
                finite = sample[np.isfinite(sample)]
                if finite.size == 0:
                    return None
                minimum = finite.min()
                maximum = finite.max()
    except (TypeError, ValueError, FloatingPointError):
        return None
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return None
    return (float(minimum), float(maximum))
