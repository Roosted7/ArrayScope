import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[2]
PACKAGE = types.ModuleType("arrayscope")
PACKAGE.__path__ = [str(ROOT / "arrayscope")]
sys.modules.setdefault("arrayscope", PACKAGE)

COMPARE_PATH = ROOT / "arrayscope" / "core" / "compare.py"
SPEC = importlib.util.spec_from_file_location("arrayscope.core.compare", COMPARE_PATH)
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


def test_compare_document_adds_compatible_layers():
    document = compare.CompareDocument.from_base(np.zeros((4, 5)), label="base")
    document = document.with_layer(np.ones((4, 5)), label="other")

    assert document.layers[0].label == "base"
    assert document.layers[1].label == "other"
    assert compare.compatible_roi_shape(document.layers[1].data, (4, 5))
    assert not compare.compatible_roi_shape(document.layers[1].data, (5, 4))
