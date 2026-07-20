from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.display.tile_truth_overlay import TileTruthOverlayLayer


def _row(tile: int, rect, *, drawable: bool):
    return {
        "tile": tile,
        "tile_rect": rect,
        "target_source": tile + 10,
        "acknowledged_source": tile + 10 if drawable else None,
        "drawable": drawable,
        "target_texture_kind": "complex_rg32f",
        "acknowledged_texture_kind": "complex_rg32f" if drawable else None,
        "physical_texture_kind": "complex_rg32f",
        "physical_storage_mode": "complex",
        "physical_mapping_mode": 4.0,
        "physical_component_mode": 2.0,
        "physical_levels": (0.0, 8.0),
        "physical_texture_shape": (16, 16, 2),
        "physical_texture_dtype": "float32",
        "target_channel": "complex",
        "target_complex_mapping": ("phase_color", "abs", "mapped"),
        "real_plane_identity": {
            "component": "real",
            "pointer": 4096 + tile,
            "shape": (16, 16),
            "strides": (64, 4),
            "dtype": "float32",
        },
        "imag_plane_identity": {
            "component": "imag",
            "pointer": 8192 + tile,
            "shape": (16, 16),
            "strides": (64, 4),
            "dtype": "float32",
        },
        "target_lod": {"level": 0},
        "acknowledged_lod": {"level": 0} if drawable else None,
        "target_semantic_generation": f"('sem', {tile})",
        "acknowledged_semantic_generation": f"('sem', {tile})" if drawable else None,
        "levels_generation": 4,
        "target_identity": ("target", tile),
        "acknowledged_identity": ("ack", tile) if drawable else None,
    }


def test_tile_truth_overlay_positions_one_label_inside_each_tile(qt_app):
    parent = QtWidgets.QWidget()
    parent.resize(400, 240)
    layer = TileTruthOverlayLayer(
        parent,
        lambda rect: QtCore.QRect(*(int(value) for value in rect)),
    )

    layer.set_rows(
        (
            _row(0, (10, 20, 170, 190), drawable=True),
            _row(1, (210, 20, 170, 190), drawable=False),
        )
    )

    assert len(layer.labels) == 2
    assert all(not label.isHidden() for label in layer.labels)
    assert layer.labels[0].geometry().left() == 12
    assert layer.labels[1].geometry().left() == 212
    assert layer.labels[0].property("truthState") == "draw"
    assert layer.labels[1].property("truthState") == "load"
    assert "slot 0  DRAW" in layer.labels[0].text()
    assert "slot 1  LOAD" in layer.labels[1].text()
    assert "planes r 0x1000:16x16:float32  i 0x2000:16x16:float32" in layer.labels[0].text()
    assert "phys complex_rg32f/complex map 4.0 comp 2.0" in layer.labels[0].text()
    assert layer.labels[0].testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    layer.clear()
    assert all(label.isHidden() for label in layer.labels)


def test_tile_truth_overlay_hides_tiles_outside_the_viewport(qt_app):
    parent = QtWidgets.QWidget()
    parent.resize(100, 100)
    layer = TileTruthOverlayLayer(
        parent,
        lambda rect: QtCore.QRect(*(int(value) for value in rect)),
    )

    layer.set_rows((_row(3, (200, 200, 50, 50), drawable=True),))

    assert layer.labels[0].isHidden()
    assert layer.visible_text() == ""
