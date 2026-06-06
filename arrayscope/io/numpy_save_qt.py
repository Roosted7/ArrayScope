"""Qt workflow for saving the current displayed array as a NumPy file."""

from __future__ import annotations

import numpy as np
from pyqtgraph.Qt import QtWidgets

from arrayscope.io.numpy_save import default_numpy_filename, selected_numpy_data
from arrayscope.ui.dialogs import SaveRangeDialog
from arrayscope.ui.file_dialogs import get_save_file_name


def save_current_numpy_file(parent, data, source_filepath=None):
    range_dialog = SaveRangeDialog(parent, data.shape)
    if range_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None

    output_data = selected_numpy_data(
        data,
        range_dialog.get_ranges(),
        squeeze=range_dialog.should_squeeze(),
    )

    file_path, _ = get_save_file_name(
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
