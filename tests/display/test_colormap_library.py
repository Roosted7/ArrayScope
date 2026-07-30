import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from arrayscope.display import colormap_library as library

# Captured before the autouse isolation fixture replaces it, for the one test
# whose subject *is* where the library resolves its directory to.
_REAL_DIRECTORY_RESOLVER = library.user_colormap_directory


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


# ---------------------------------------------------------------------------
# Storage location, problems channel, unavailable built-ins (ADR 0063)
# ---------------------------------------------------------------------------


def test_storage_is_scoped_to_this_install_and_migrates_the_legacy_directory(tmp_path, monkeypatch):
    # The colormap library used to store in the per-*user* AppDataLocation, so a
    # scoped ``-n <name>`` run shared one library with every other scope while
    # its operations, settings and sessions were correctly separate. It now
    # lives beside the config, and the old files are copied across once.
    from arrayscope.app import user_dirs

    legacy = tmp_path / "legacy" / "colormaps"
    legacy.mkdir(parents=True)
    (legacy / "old.json").write_text(
        json.dumps({"name": "old", "kind": "sequential", "stops": [[0, [0, 0, 0]], [1, [9, 9, 9]]]})
    )
    monkeypatch.setattr(user_dirs, "user_config_directory", lambda: tmp_path / "scoped")
    monkeypatch.setattr(library, "legacy_user_colormap_directory", lambda: str(legacy))
    monkeypatch.setattr(library, "_migrated_legacy_directory", False)
    # The autouse isolation fixture replaces the resolver; this test is *about*
    # the resolver, so put the real one back for the duration.
    monkeypatch.setattr(library, "user_colormap_directory", _REAL_DIRECTORY_RESOLVER)

    directory = library.user_colormap_directory()
    assert directory == str(tmp_path / "scoped" / "colormaps")
    assert os.path.exists(os.path.join(directory, "old.json"))
    library.refresh_user_colormaps()
    assert "old" in {info.name for info in library.user_colormaps()}

    # One-way and once: an edit in the new location is never clobbered by a
    # second resolution of the directory.
    library.save_user_colormap("old", library.SEQUENTIAL, ((0.0, (1, 1, 1)), (1.0, (2, 2, 2))))
    monkeypatch.setattr(library, "_migrated_legacy_directory", False)
    library.user_colormap_directory()
    library.refresh_user_colormaps()
    assert library.find_colormap("old").stops[-1][1] == (2, 2, 2)


def test_an_unreadable_colormap_file_is_recorded_not_swallowed():
    directory = library.user_colormap_directory()
    os.makedirs(directory, exist_ok=True)
    library.save_user_colormap(
        "good", library.SEQUENTIAL, ((0.0, (0, 0, 0)), (1.0, (255, 255, 255)))
    )
    (Path(directory) / "garbled.json").write_text('{"name": "garbled", "stops": ')
    (Path(directory) / "one-stop.json").write_text(
        json.dumps({"name": "one-stop", "stops": [[0.0, [1, 2, 3]]]})
    )
    library.refresh_user_colormaps()

    # The good map still loads: one broken file never takes the library down.
    assert "good" in {info.name for info in library.user_colormaps()}
    problems = {
        os.path.basename(problem.path): problem for problem in library.user_colormap_problems()
    }
    assert set(problems) == {"garbled.json", "one-stop.json"}
    # Unparseable: no payload to drive a form from. Parseable but rejected: the
    # payload survives, keyed by the file path either way (ADR 0062).
    assert problems["garbled.json"].payload is None
    assert problems["one-stop.json"].payload["name"] == "one-stop"
    assert problems["one-stop.json"].item_id == "one-stop"

    # Stored means addressable: the repair primitives reach it by path.
    assert library.update_user_colormap_file(
        problems["one-stop.json"].path,
        {"name": "one-stop", "kind": "sequential", "stops": [[0.0, [1, 2, 3]], [1.0, [9, 9, 9]]]},
    )
    assert "one-stop" in {info.name for info in library.user_colormaps()}
    assert library.remove_user_colormap_file(problems["garbled.json"].path)
    assert library.user_colormap_problems() == []


def test_the_layout_and_hidden_files_are_never_mistaken_for_colormaps():
    library.set_builtin_hidden("turbo", True)
    library.apply_library_layout(["Perceptual"], {}, {})
    library.refresh_user_colormaps()
    assert library.user_colormap_problems() == []


def test_repair_primitives_refuse_paths_outside_the_library(tmp_path):
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}")
    assert not library.remove_user_colormap_file(str(outside))
    assert not library.update_user_colormap_file(str(outside), {"name": "x"})
    assert outside.exists()


def test_a_missing_optional_builtin_is_unavailable_not_absent(monkeypatch):
    # It used to be dropped from the listing entirely, so a curated name simply
    # vanished with no explanation anywhere.
    monkeypatch.setattr(library, "_crameri_stops_cache", {})
    monkeypatch.setattr(library, "_resolve_builtin_stops", lambda spec: None if spec else spec)

    listed = {info.name for info in library.list_colormaps()}
    assert "Batlow" not in listed  # pickers never offer a map they cannot draw
    manager_view = {info.name: info for info in library.list_colormaps(include_unavailable=True)}
    assert "cmcrameri" in manager_view["Batlow"].unavailable_reason
    # And it stays selectable in the manager, which is what makes it explainable.
    assert library.find_colormap("Batlow").unavailable_reason


def test_group_rename_moves_every_member_but_never_favorites():
    assert library.rename_group("Perceptual", "Mine")
    groups = dict(library.grouped_colormaps())
    assert "Mine" in groups
    assert "turbo" in {info.name for info in groups["Mine"]}
    assert not library.rename_group(library.FAVORITES_GROUP, "Starred")


def test_reset_layout_restores_the_curated_arrangement():
    library.apply_library_layout(["User"], {"turbo": "User"}, {"turbo": 0})
    assert library.effective_group(library.find_colormap("turbo")) == "User"
    library.reset_layout()
    assert library.effective_group(library.find_colormap("turbo")) == "Perceptual"
