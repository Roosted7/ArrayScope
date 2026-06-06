import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_material_icon_falls_back_for_unknown_names(qt_app):
    from arrayscope.ui.icons import material_icon, verify_icon_names

    icon = material_icon("definitely_not_a_real_material_icon_name")

    assert not icon.isNull()
    assert verify_icon_names(["save", "definitely_not_a_real_material_icon_name"]) == {
        "save": True,
        "definitely_not_a_real_material_icon_name": True,
    }
