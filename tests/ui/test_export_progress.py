"""Export feedback stays visible until its worker owns a terminal outcome."""

from types import SimpleNamespace

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.ui.export_progress import ExportProgressDialog


@pytest.mark.parametrize("action", ["cancel", "window_close", "escape"])
def test_cancel_requests_once_and_keeps_feedback_until_worker_finishes(qtbot, tmp_path, action):
    dialog = ExportProgressDialog(None, tmp_path / "selected.npy", (3, 4), np.float32)
    qtbot.addWidget(dialog)
    dialog.show()
    requests = []
    dialog.cancel_requested.connect(lambda: requests.append(True))

    if action == "cancel":
        qtbot.mouseClick(dialog.cancel_button, QtCore.Qt.MouseButton.LeftButton)
    elif action == "window_close":
        assert not dialog.close()
    else:
        qtbot.keyClick(dialog, QtCore.Qt.Key.Key_Escape)

    assert requests == [True]
    assert dialog.isVisible()
    assert not dialog.cancel_button.isEnabled()
    assert "current operation" in dialog.progress_detail_label.text()
    assert not dialog.close()
    dialog.reject()
    assert requests == [True]
    dialog.close_quietly()
    assert not dialog.isVisible()
    assert requests == [True]


def test_export_summary_and_byte_progress_handle_large_empty_and_scalar_arrays(qtbot, tmp_path):
    dialog = ExportProgressDialog(None, tmp_path / "selected.npy", (1 << 30,), np.complex64)
    qtbot.addWidget(dialog)
    dialog.show()
    assert not dialog.isModal()
    assert dialog.filename_label.text() == "selected.npy"
    assert "[1073741824]" in dialog.summary_label.text()
    assert "complex64" in dialog.summary_label.text()
    assert "8.0 GiB" in dialog.summary_label.text()

    dialog.apply_progress(
        SimpleNamespace(phase="writing", message="", completed_bytes=1 << 31, total_bytes=1 << 33)
    )
    assert dialog.progress_bar.maximum() == 1000
    assert dialog.progress_bar.value() == 250
    assert dialog.progress_detail_label.text() == "2.0 GiB of 8.0 GiB"
    dialog.apply_progress(
        SimpleNamespace(phase="writing", message="", completed_bytes=0, total_bytes=0)
    )
    assert dialog.progress_bar.value() == 1000
    assert dialog.progress_detail_label.text() == "0 B of 0 B"
    dialog.apply_progress(
        SimpleNamespace(
            phase="evaluating", message="Evaluating a crop…", completed_bytes=0, total_bytes=None
        )
    )
    assert dialog.progress_bar.maximum() == 0
    assert dialog.status_label.text() == "Evaluating a crop…"
    dialog.close_quietly()

    scalar = ExportProgressDialog(None, tmp_path / "scalar.npy", (), np.float32)
    qtbot.addWidget(scalar)
    assert scalar.summary_label.text() == "Selected output: Scalar · float32 · 4 B of array data"
    scalar.close_quietly()


def test_cancelling_and_error_feedback_survive_late_progress_and_close_without_cancel(
    qtbot, tmp_path
):
    dialog = ExportProgressDialog(None, tmp_path / "<selected>.npy", (3, 4))
    qtbot.addWidget(dialog)
    requests = []
    dialog.cancel_requested.connect(lambda: requests.append(True))
    dialog.show()
    dialog.set_cancelling("Stopping the selected operation.")
    explanation = dialog.progress_detail_label.text()
    progress = SimpleNamespace(
        phase="writing", message="Writing", completed_bytes=6, total_bytes=12
    )
    dialog.apply_progress(progress)
    assert dialog.status_label.text() == "Cancelling export…"
    assert dialog.progress_detail_label.text() == explanation

    dialog.set_error("Permission denied: <selected>.npy")
    dialog.apply_progress(progress)
    assert dialog.isVisible()
    assert dialog.status_label.text() == "Export failed"
    assert dialog.error_label.text() == "Permission denied: <selected>.npy"
    assert dialog.error_label.textFormat() == QtCore.Qt.TextFormat.PlainText
    accessible_error = QtGui.QAccessible.queryAccessibleInterface(dialog.error_label)
    assert accessible_error.text(QtGui.QAccessible.Text.Name) == dialog.error_label.text()
    assert dialog.cancel_button.isEnabled()
    assert "Close" in dialog.cancel_button.text()
    assert not dialog.progress_bar.isVisible()
    qtbot.mouseClick(dialog.cancel_button, QtCore.Qt.MouseButton.LeftButton)
    assert not dialog.isVisible()
    assert requests == []


def test_preflight_replaces_unknown_dtype_with_exact_output_metadata(qtbot, tmp_path):
    dialog = ExportProgressDialog(None, tmp_path / "selection.npy", (3, 4))
    qtbot.addWidget(dialog)
    assert "determined during evaluation" in dialog.summary_label.text()
    dialog.set_output_summary((3, 4), np.complex128)
    assert dialog.summary_label.text() == (
        "Selected output: [3 × 4] · complex128 · 192 B of array data"
    )
    assert dialog.summary_label.textFormat() == QtCore.Qt.TextFormat.PlainText
    dialog.close_quietly()


def test_export_dialog_inherits_application_appearance_without_mutating_it(qtbot, qapp, tmp_path):
    previous = (qapp.styleSheet(), qapp.palette(), qapp.organizationName(), qapp.applicationName())
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = ExportProgressDialog(parent, tmp_path / "selection.npy", (0, 4), np.float32)
    qtbot.addWidget(dialog)
    assert "0 B" in dialog.summary_label.text()
    assert dialog.palette() == parent.palette()
    assert (
        qapp.styleSheet(),
        qapp.palette(),
        qapp.organizationName(),
        qapp.applicationName(),
    ) == previous
    dialog.close_quietly()
