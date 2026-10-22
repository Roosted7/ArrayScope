"""Collect an exact NumPy export request before any array evaluation."""

from __future__ import annotations

from pyqtgraph.Qt import QtWidgets

from arrayscope.io.numpy_save import default_numpy_filename
from arrayscope.io.region_export import RegionExportRequest, capture_export_definitions
from arrayscope.ui.dialogs import SaveRangeDialog
from arrayscope.ui.file_dialogs import get_save_file_name


def collect_numpy_save_request(parent, document, source_filepath=None, *, dtype=None):
    """Ask for output ranges and path without reading source pixels.

    The document and definition identities are captured before either modal
    dialog; later view/operation edits cannot rewrite an accepted request.
    """
    definitions = capture_export_definitions(document)
    range_dialog = SaveRangeDialog(
        parent,
        document.current_shape,
        dtype=dtype,
        axes=document.current_axes,
        action_text="Next…",
    )
    if range_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None

    ranges = tuple(tuple(bounds) for bounds in range_dialog.get_ranges())
    squeeze = range_dialog.should_squeeze()

    file_path, _ = get_save_file_name(
        parent,
        "Export selected array as NumPy file",
        default_numpy_filename(source_filepath),
        "NumPy files (*.npy)",
    )
    if not file_path:
        return None

    if not file_path.lower().endswith(".npy"):
        file_path += ".npy"

    return RegionExportRequest(
        document,
        file_path,
        ranges,
        squeeze=squeeze,
        sidecar=False,
        definitions=definitions,
    )
