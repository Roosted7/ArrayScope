"""Pure helpers for saving displayed arrays as NumPy files."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def default_numpy_filename(source_filepath=None):
    if source_filepath is None:
        return "arrayscope.npy"

    source_path = Path(source_filepath)
    source_name = source_path.name
    if source_name.lower().endswith(".nii.gz"):
        return source_name[:-7] + ".npy"
    return f"{source_path.stem}.npy"


def selected_numpy_data(data, ranges, *, squeeze=True):
    sliced_data = data[tuple(slice(start, stop) for start, stop in ranges)]
    return np.squeeze(sliced_data) if squeeze else sliced_data
