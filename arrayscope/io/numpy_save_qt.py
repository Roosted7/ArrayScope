"""Collect an exact NumPy export request before any array evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from pyqtgraph.Qt import QtWidgets

from arrayscope.core.array_source import LazySourceArray
from arrayscope.io.hdf5_source import HDF5ArraySource
from arrayscope.io.numpy_save import default_numpy_filename
from arrayscope.io.region_export import RegionExportRequest, capture_export_definitions
from arrayscope.operations.recipes import dumps_recipe
from arrayscope.ui.dialogs import SaveRangeDialog
from arrayscope.ui.file_dialogs import get_save_file_name


def _source_metadata(parent, document, source_filepath):
    source = document.base_data
    while isinstance(source, LazySourceArray):
        source = source.source
    path = source_filepath if source_filepath is not None else getattr(parent, "_filepath", None)
    dataset = getattr(parent, "_dataset_path", None)
    if isinstance(source, HDF5ArraySource):
        path = source.path
        dataset = source.descriptor.key
    metadata = {}
    if path is not None:
        metadata["source_path"] = str(Path(path).expanduser().absolute())
    if dataset is not None:
        metadata["dataset_path"] = str(dataset)
    return json.dumps(metadata) if metadata else None


def collect_numpy_save_request(parent, document, source_filepath=None, *, dtype=None):
    """Ask for output ranges and path without reading source pixels.

    The document and definition identities are captured before either modal
    dialog; later view/operation edits cannot rewrite an accepted request.
    """
    definitions = capture_export_definitions(document)
    steps = tuple(document.steps)
    source_metadata = _source_metadata(parent, document, source_filepath)
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

    file_path, selected_filter = get_save_file_name(
        parent,
        "Export selected array as NumPy file",
        default_numpy_filename(source_filepath),
        "NumPy array (*.npy);;NumPy archive with operation recipe (*.npz)",
    )
    if not file_path:
        return None

    if Path(file_path).suffix.lower() not in {".npy", ".npz"}:
        file_path += ".npz" if "*.npz" in selected_filter else ".npy"
    archive = Path(file_path).suffix.lower() == ".npz"

    return RegionExportRequest(
        document,
        file_path,
        ranges,
        squeeze=squeeze,
        recipe_json=dumps_recipe(steps) if archive else None,
        source_metadata_json=source_metadata if archive else None,
        sidecar=False,
        definitions=definitions,
    )
