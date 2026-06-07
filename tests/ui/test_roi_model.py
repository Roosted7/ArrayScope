import os

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_roi_table_model_uses_selection_color_and_delete_path(qtbot):
    from arrayscope.core.roi import RoiGeometry, RoiKind, RoiSelection, RoiStatistics
    from arrayscope.ui.docks.inspection import InspectionDock

    deleted = []
    selected = []
    dock = InspectionDock(
        None,
        on_tool_changed=lambda _tool: None,
        on_add_roi=lambda _tool: None,
        on_delete_roi=deleted.append,
        on_clear_rois=lambda: None,
        on_select_roi=selected.append,
    )
    qtbot.addWidget(dock)
    selection = RoiSelection(
        id="roi-1",
        label="ROI 1",
        geometry=RoiGeometry(RoiKind.RECTANGLE, rect=(0, 0, 2, 2)),
        color=(40, 120, 210),
    )
    stats = RoiStatistics(4, 4, 1.0, 4.0, 2.5, 2.5, 1.0, 10.0, 5.0, 1.0, 4.0)

    dock.set_statistics({"roi-1": (selection, stats)})
    dock.stats_table.selectRow(0)
    dock._delete_clicked()

    assert dock.roi_model.rowCount() == 1
    assert dock.roi_model.roi_id_for_row(0) == "roi-1"
    assert deleted == ["roi-1"]
