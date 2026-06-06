import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[2]
PACKAGE = types.ModuleType("arrayscope")
PACKAGE.__path__ = [str(ROOT / "arrayscope")]
sys.modules.setdefault("arrayscope", PACKAGE)

ROI_PATH = ROOT / "arrayscope" / "core" / "roi.py"
SPEC = importlib.util.spec_from_file_location("arrayscope.core.roi", ROI_PATH)
roi = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = roi
SPEC.loader.exec_module(roi)


def test_polyline_sampling_across_horizontal_and_vertical_segments():
    image = np.arange(5 * 5, dtype=float).reshape(5, 5)

    samples = roi.polyline_roi_samples(image, ((0, 0), (4, 0), (4, 4)))

    np.testing.assert_allclose(samples, np.array([0, 1, 2, 3, 4, 9, 14, 19, 24], dtype=float))


def test_freehand_polygon_mask_includes_expected_pixels_for_box():
    mask = roi.polygon_roi_mask((5, 5), ((1, 1), (4, 1), (4, 4), (1, 4)))

    assert mask.shape == (5, 5)
    assert mask[2, 2]
    assert mask[1, 1]
    assert not mask[0, 0]
    assert not mask[4, 4]


def test_simplify_polyline_preserves_endpoints():
    points = ((0, 0), (1, 0.05), (2, -0.02), (3, 0), (4, 4))

    simplified = roi.simplify_polyline(points, tolerance=0.2)

    assert simplified[0] == (0.0, 0.0)
    assert simplified[-1] == (4.0, 4.0)
    assert len(simplified) < len(points)


def test_close_polygon_adds_first_point_when_needed():
    closed = roi.close_polygon(((0, 0), (1, 0), (1, 1)))

    assert closed[0] == closed[-1]
    assert len(closed) == 4


def test_roi_statistics_for_all_roi_kinds():
    image = np.arange(6 * 6, dtype=float).reshape(6, 6)
    geometries = (
        roi.RoiGeometry(roi.RoiKind.LINE, points=((0, 0), (5, 0))),
        roi.RoiGeometry(roi.RoiKind.RECTANGLE, rect=(1, 1, 3, 2)),
        roi.RoiGeometry(roi.RoiKind.POLYLINE, points=((0, 0), (5, 0), (5, 5))),
        roi.RoiGeometry(roi.RoiKind.FREEHAND_POLYGON, points=((1, 1), (5, 1), (5, 5), (1, 5))),
    )

    for geometry in geometries:
        values = roi.roi_values(image, geometry)
        stats = roi.roi_statistics(values)
        assert stats.count > 0
        assert stats.finite_count == stats.count
        assert stats.minimum is not None
        assert stats.maximum is not None
        assert stats.rss is not None


def test_roi_module_has_no_qt_or_pyqtgraph_imports():
    tree = ast.parse(ROI_PATH.read_text())

    imported_roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.append(node.module.split(".")[0])

    assert "pyqtgraph" not in imported_roots
    assert not any(name.startswith(("PyQt", "PySide")) for name in imported_roots)
