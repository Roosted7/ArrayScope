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
    assert "CET-C2" in phase_names
    assert "RomaO" in phase_names
    assert "viridis" in scalar_names
    assert "PAL-relaxed" not in scalar_names


def test_user_colormap_round_trip_and_shadowing():
    stops = ((0.0, (0, 0, 0)), (0.5, (255, 0, 0)), (1.0, (255, 255, 0)))
    info = library.save_user_colormap("test-hot", library.SEQUENTIAL, stops)
    assert info.source == "user"

    found = library.find_colormap("test-hot")
    assert found is not None
    assert found.stops == info.stops
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

    table = np.column_stack([np.linspace(0, 1, 16), np.zeros(16), np.linspace(1, 0, 16)])
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


def test_grouped_colormaps_order_and_membership():
    groups = dict(library.grouped_colormaps())
    assert "Scientific" in groups
    names = {info.name for info in groups["Scientific"]}
    assert {"Lipari", "Navia"} <= names
    ordered = [group for group, _infos in library.grouped_colormaps()]
    # Favorites (the empty group) leads, before the named collections.
    assert ordered.index(library.FAVORITES_GROUP) < ordered.index("Perceptual")
    favorites = {info.name for info in groups[library.FAVORITES_GROUP]}
    assert {"gray", "viridis", "Batlow"} <= favorites


def test_layout_persistence_reorders_and_regroups(tmp_path):
    library.apply_library_layout(
        group_order=["Perceptual", library.FAVORITES_GROUP],
        map_groups={"turbo": library.FAVORITES_GROUP},
        map_order={"turbo": 0, "gray": 1},
    )
    groups = library.grouped_colormaps()
    ordered = [group for group, _infos in groups]
    assert ordered.index("Perceptual") < ordered.index(library.FAVORITES_GROUP)
    favorites = [info.name for info in dict(groups)[library.FAVORITES_GROUP]]
    assert favorites[0] == "turbo"
    assert favorites[1] == "gray"


def test_rename_group_moves_members(tmp_path):
    library.rename_group("Perceptual", "My Maps")
    groups = dict(library.grouped_colormaps())
    assert "Perceptual" not in groups
    assert "turbo" in {info.name for info in groups["My Maps"]}


def test_hidden_builtin_round_trip():
    assert library.find_colormap("turbo").hidden is False
    library.set_builtin_hidden("turbo", True)
    visible = {info.name for info in library.list_colormaps()}
    assert "turbo" not in visible
    assert library.find_colormap("turbo").hidden is True
    assert library.reset_builtin("turbo")
    assert library.find_colormap("turbo").hidden is False


def test_builtin_override_and_reset():
    stops = ((0.0, (255, 0, 0)), (1.0, (0, 0, 255)))
    library.save_user_colormap("viridis", library.SEQUENTIAL, stops)
    assert library.overrides_builtin("viridis")
    info = library.find_colormap("viridis")
    assert info.source == "user"
    assert info.group == library.builtin_group_for("viridis")
    assert library.reset_builtin("viridis")
    assert library.find_colormap("viridis").source == "builtin"


def test_kind_detection_on_real_maps():
    detect = library.detect_colormap_kind
    kind, confidence = detect(library.colormap_stops("RomaO", 33))
    assert kind == library.CYCLIC
    assert confidence > 0.9
    kind, confidence = detect(library.colormap_stops("Vik", 33))
    assert kind == library.DIVERGING
    assert confidence > 0.9
    kind, _confidence = detect(library.colormap_stops("Batlow", 33))
    assert kind == library.SEQUENTIAL


def test_import_auto_detects_cyclic(tmp_path):
    table = np.asarray([library.get_colormap("RomaO").getLookupTable(0.0, 1.0, 64, alpha=False)])[0]
    path = tmp_path / "wrapped.csv"
    np.savetxt(path, table, delimiter=",")
    info = library.import_colormap_file(str(path), name="wrapped")
    assert info.kind == library.CYCLIC


@pytest.fixture
def _isolated_listeners():
    """Run each listener test against an empty, restored global registry."""
    saved = list(library._listeners)
    library._listeners.clear()
    try:
        yield
    finally:
        library._listeners[:] = saved


@pytest.mark.usefixtures("_isolated_listeners")
def test_remove_library_listener_stops_notifications():
    calls = []

    def listener():
        calls.append(1)

    library.add_library_listener(listener)
    library.refresh_user_colormaps()
    assert calls == [1]

    library.remove_library_listener(listener)
    library.refresh_user_colormaps()
    assert calls == [1]  # no further notification after removal


@pytest.mark.usefixtures("_isolated_listeners")
def test_remove_library_listener_is_idempotent():
    def listener():
        pass

    # Removing before it is ever registered is a no-op...
    library.remove_library_listener(listener)
    library.add_library_listener(listener)
    library.remove_library_listener(listener)
    # ...and removing a second time must not raise.
    library.remove_library_listener(listener)
    assert listener not in library._listeners


@pytest.mark.usefixtures("_isolated_listeners")
def test_notify_prunes_listener_bound_to_deleted_widget():
    # A window that closed without unregistering leaves a listener bound to a
    # deleted C++ object; invoking it raises RuntimeError. The registry must
    # self-heal (drop it) so a colormap mutation neither crashes nor keeps
    # calling the dead wrapper on every future mutation.
    class _DeletedWidgetListener:
        def __call__(self):
            raise RuntimeError("Internal C++ object (ArrayScopeWindow) already deleted.")

    dead = _DeletedWidgetListener()
    live_calls = []

    def live_listener():
        live_calls.append(1)

    library.add_library_listener(dead)
    library.add_library_listener(live_listener)

    library.refresh_user_colormaps()  # must not raise
    assert dead not in library._listeners  # pruned on first failure
    assert live_calls == [1]  # a sibling listener still ran

    library.refresh_user_colormaps()
    assert live_calls == [1, 1]  # and keeps running once the corpse is gone
