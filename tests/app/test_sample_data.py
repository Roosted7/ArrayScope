"""Examples are meaningful scientific arrays; recent-data history stays bounded."""

from pathlib import Path

import numpy as np
import pytest

from arrayscope.app.recent_files import RECENT_FILES_KEY, recent_paths, remember_path
from arrayscope.app.sample_data import SAMPLES, make_sample


@pytest.mark.parametrize("sample_id", [sample.id for sample in SAMPLES])
def test_examples_are_fresh_bounded_arrays_with_correct_axis_metadata(sample_id):
    data, axes = make_sample(sample_id)
    again, _axes = make_sample(sample_id)
    assert data.nbytes <= 4 * 1024**2
    assert np.isfinite(data).all()
    assert data.dtype in (np.dtype("float32"), np.dtype("complex64"))
    assert tuple(axis.size for axis in axes) == data.shape
    assert tuple(axis.label for axis in axes[:2]) == ("Y", "X")
    assert np.ptp(np.abs(data)) > 0.5
    np.testing.assert_array_equal(data, again)
    assert not np.shares_memory(data, again)
    if sample_id == "time-series":
        assert axes[-1].label == "Time"
        assert axes[-1].unit == "s"
        assert not np.array_equal(data[..., 0], data[..., -1])
    elif sample_id == "complex":
        assert np.iscomplexobj(data)
        assert np.ptp(np.angle(data)) > 6


def test_unknown_example_is_rejected():
    with pytest.raises(ValueError, match="Unknown example"):
        make_sample("absent")


class _Settings(dict):
    def value(self, key, default=None):
        return self.get(key, default)

    def setValue(self, key, value):
        self[key] = value


def test_recent_data_history_deduplicates_and_caps_without_dropping_missing_paths(tmp_path):
    settings = _Settings()
    paths = [tmp_path / f"array-{i}.npy" for i in range(15)]
    for path in paths:
        remember_path(settings, path)
    assert recent_paths(settings) == tuple(reversed(paths[-10:]))
    remember_path(settings, paths[-4])
    assert recent_paths(settings)[0] == paths[-4]
    assert len(recent_paths(settings)) == 10
    assert len(set(recent_paths(settings))) == 10


@pytest.mark.parametrize("stored", [None, 5, {}, [None, "", "\0bad", 3]])
def test_recent_data_history_ignores_malformed_settings(stored):
    assert recent_paths(_Settings({RECENT_FILES_KEY: stored})) == ()


def test_recent_history_accepts_single_path_from_settings(tmp_path):
    path = tmp_path / "scan.nii.gz"
    assert recent_paths(_Settings({RECENT_FILES_KEY: str(path)})) == (Path(path),)
