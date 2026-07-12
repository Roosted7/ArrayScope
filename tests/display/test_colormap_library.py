import os

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from arrayscope.display import colormap_library as library


@pytest.fixture(autouse=True)
def _isolated_user_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "user_colormap_directory", lambda: str(tmp_path / "colormaps"))
    library.refresh_user_colormaps()
    yield
    library.refresh_user_colormaps()


def test_builtin_library_covers_all_kinds():
    kinds = {info.kind for info in library.builtin_colormaps()}
    assert kinds == {library.SEQUENTIAL, library.DIVERGING, library.CYCLIC}


def test_family_filtering_is_phase_safe():
    phase_names = {info.name for info in library.colormaps_for_family("phase")}
    scalar_names = {info.name for info in library.colormaps_for_family("scalar")}
    assert "viridis" not in phase_names
    assert "PAL-relaxed" in phase_names
    assert "CET-C1" in phase_names
    assert "viridis" in scalar_names
    assert "PAL-relaxed" not in scalar_names


def test_user_colormap_round_trip_and_shadowing():
    stops = ((0.0, (0, 0, 0)), (0.5, (255, 0, 0)), (1.0, (255, 255, 0)))
    info = library.save_user_colormap("test-hot", library.SEQUENTIAL, stops)
    assert info.source == "user"

    found = library.find_colormap("test-hot")
    assert found is not None and found.stops == info.stops
    lut = library.get_colormap("test-hot").getLookupTable(0.0, 1.0, 3, alpha=False)
    assert tuple(lut[0]) == (0, 0, 0)
    assert tuple(lut[-1]) == (255, 255, 0)

    # Reload from disk.
    library.refresh_user_colormaps()
    assert library.find_colormap("test-hot") is not None

    assert library.delete_user_colormap("test-hot")
    assert library.find_colormap("test-hot") is None


def test_import_matlab_colormap(tmp_path):
    from scipy.io import savemat

    table = np.column_stack(
        [np.linspace(0, 1, 16), np.zeros(16), np.linspace(1, 0, 16)]
    )
    path = tmp_path / "mymap.mat"
    savemat(path, {"mymap": table})

    info = library.import_colormap_file(str(path))
    assert info.name == "mymap"
    lut = library.get_colormap("mymap").getLookupTable(0.0, 1.0, 2, alpha=False)
    assert tuple(lut[0]) == (0, 0, 255)
    assert tuple(lut[-1]) == (255, 0, 0)


def test_import_csv_and_npy(tmp_path):
    table = np.column_stack([np.linspace(0, 255, 8)] * 3)
    csv_path = tmp_path / "gray8.csv"
    np.savetxt(csv_path, table, delimiter=",")
    info = library.import_colormap_file(str(csv_path), name="csv-gray")
    assert info.kind == library.SEQUENTIAL

    npy_path = tmp_path / "gray9.npy"
    np.save(npy_path, np.column_stack([np.linspace(0, 1, 9)] * 3))
    info = library.import_colormap_file(str(npy_path), name="npy-gray", kind=library.CYCLIC)
    assert info.kind == library.CYCLIC
    assert "npy-gray" in {i.name for i in library.colormaps_for_family("phase")}


def test_export_round_trip(tmp_path):
    library.save_user_colormap("exp", library.DIVERGING, ((0.0, (0, 0, 255)), (1.0, (255, 0, 0))))
    out = tmp_path / "exp.json"
    library.export_colormap("exp", str(out))
    library.delete_user_colormap("exp")

    info = library.import_colormap_file(str(out))
    assert info.name == "exp"
    assert info.kind == library.DIVERGING


def test_policy_family_mapping():
    from arrayscope.display.colormap_policy import colormap_family

    assert colormap_family("complex") == "phase"
    assert colormap_family("angle") == "phase"
    assert colormap_family("real") == "scalar"
    assert colormap_family("abs") == "scalar"
