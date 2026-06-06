"""NumPy save workflow for the current displayed array."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyqtgraph.Qt import QtWidgets

from arrayscope.ui.dialogs import SaveRangeDialog


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


def save_current_numpy_file(parent, data, source_filepath=None):
    range_dialog = SaveRangeDialog(parent, data.shape)
    if range_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None

    output_data = selected_numpy_data(
        data,
        range_dialog.get_ranges(),
        squeeze=range_dialog.should_squeeze(),
    )

    file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent,
        "Save current array as NumPy file",
        default_numpy_filename(source_filepath),
        "NumPy files (*.npy)",
    )
    if not file_path:
        return None

    if not file_path.lower().endswith(".npy"):
        file_path += ".npy"

    np.save(file_path, output_data)
    return file_path, output_data.shape
