import numpy as np

from arrayscope.display.histogram_plot import HistogramPlotRequest, compute_histogram_plot


def _request(data, **kwargs):
    return HistogramPlotRequest(
        data=np.asarray(data),
        source_identity=("test", np.shape(data)),
        histogram_bounds=kwargs.pop("histogram_bounds", None),
        visible_value_span=kwargs.pop("visible_value_span", None),
        pixel_extent=kwargs.pop("pixel_extent", 200.0),
        **kwargs,
    )


def test_histogram_plot_ignores_nonfinite_values():
    result = compute_histogram_plot(_request([0.0, 1.0, np.nan, np.inf], histogram_bounds=(0.0, 1.0), bin_cap=8))

    assert result.has_data
    assert int(np.sum(result.y)) == 2


def test_histogram_plot_handles_complex_values_as_magnitude():
    data = np.array([3.0 + 4.0j, 0.0 + 0.0j, np.nan + 0.0j])

    result = compute_histogram_plot(_request(data, histogram_bounds=(0.0, 5.0), bin_cap=5))

    assert result.has_data
    assert int(np.sum(result.y)) == 2
    assert result.x[0] == 0.0


def test_histogram_plot_respects_bin_cap_and_visible_span():
    data = np.linspace(0.0, 100.0, 10_000, dtype=np.float32)

    result = compute_histogram_plot(
        _request(
            data,
            histogram_bounds=(0.0, 100.0),
            visible_value_span=10.0,
            pixel_extent=400.0,
            min_bin_screen_px=4,
            bin_cap=37,
        )
    )

    assert result.has_data
    assert len(result.x) == 37


def test_histogram_plot_empty_or_nonfinite_is_stable():
    result = compute_histogram_plot(_request([np.nan, np.inf]))

    assert not result.has_data
    assert result.x is None
    assert result.y is None
