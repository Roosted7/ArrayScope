import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[2]
PACKAGE = types.ModuleType("arrayscope")
PACKAGE.__path__ = [str(ROOT / "arrayscope")]
sys.modules.setdefault("arrayscope", PACKAGE)

HISTOGRAMS_PATH = ROOT / "arrayscope" / "core" / "histograms.py"
SPEC = importlib.util.spec_from_file_location("arrayscope.core.histograms", HISTOGRAMS_PATH)
histograms = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = histograms
SPEC.loader.exec_module(histograms)


def test_comparison_histograms_use_shared_range():
    spec = histograms.HistogramSpec(bins=4)

    results = histograms.comparison_histograms((("a", [0, 1]), ("b", [10, 11])), spec)

    assert len(results) == 2
    np.testing.assert_allclose(results[0].edges, results[1].edges)
    assert results[0].edges[0] == 0
    assert results[0].edges[-1] == 11


def test_empty_histogram_has_stable_edges_and_zero_counts():
    result = histograms.histogram([], histograms.HistogramSpec(bins=3), name="empty")

    assert result.name == "empty"
    np.testing.assert_array_equal(result.counts, np.zeros(3))
    np.testing.assert_allclose(result.edges, np.array([0.0, 1 / 3, 2 / 3, 1.0]))


def test_histogram_ignores_nonfinite_values():
    result = histograms.histogram([0, 1, np.nan, np.inf], histograms.HistogramSpec(bins=2))

    assert int(np.sum(result.counts)) == 2
