import numpy as np


def test_file_view_session_round_trips_recipe_viewport_and_rois(tmp_path):
    from arrayscope.core.roi import RoiGeometry, RoiKind, RoiSelection
    from arrayscope.core.view_recipe import DisplaySettings, ViewRecipe
    from arrayscope.core.view_session import (
        FileViewSession,
        PanelSession,
        ViewportSession,
        dumps_session,
        loads_session,
        metadata_for_file,
    )
    from arrayscope.core.view_state import ViewState

    path = tmp_path / "scan.npy"
    data = np.zeros((4, 5, 6), dtype=np.float32)
    np.save(path, data)
    metadata = metadata_for_file(
        path, dataset_path="/data", selector_class_name="NpzDatasetSelector", data=data
    )
    session = FileViewSession(
        metadata=metadata,
        recipe=ViewRecipe(
            view_state=ViewState.from_shape(data.shape)
            .with_image_axes(0, 1)
            .with_montage_axis(2, indices=(0, 1)),
            display=DisplaySettings(
                channel="real",
                scale="linear",
                aspect_mode="square_pixels",
                window_mode="relative",
                levels=(1.0, 2.0),
            ),
        ),
        viewport=ViewportSession(
            mode="user", view_range=((1.0, 3.0), (2.0, 4.0)), viewport_shape=(240, 320)
        ),
        rois=(
            RoiSelection(
                "roi-7", "ROI 7", RoiGeometry(RoiKind.RECTANGLE, rect=(1.0, 2.0, 3.0, 4.0))
            ),
        ),
        selected_roi_id="roi-7",
        panels=PanelSession(
            operation_visible=False,
            inspection_visible=True,
            window_size=(1400, 900),
            window_maximized=False,
        ),
    )

    text = dumps_session(session)
    loaded = loads_session(text, data.shape)

    assert loaded.metadata == metadata
    assert loaded.viewport.view_range == ((1.0, 3.0), (2.0, 4.0))
    assert loaded.viewport.viewport_shape == (240, 320)
    assert loaded.rois[0].id == "roi-7"
    assert loaded.rois[0].geometry.rect == (1.0, 2.0, 3.0, 4.0)
    assert loaded.selected_roi_id == "roi-7"
    assert loaded.panels == PanelSession(
        operation_visible=False,
        inspection_visible=True,
        window_size=(1400, 900),
        window_maximized=False,
    )


def test_file_view_session_metadata_mismatch_rejects_shape(tmp_path):
    from arrayscope.core.view_session import metadata_for_file, metadata_matches

    path = tmp_path / "scan.npy"
    data = np.zeros((4, 5), dtype=np.float32)
    np.save(path, data)
    saved = metadata_for_file(path, data=data)
    current = dict(saved)
    current["shape"] = [4, 6]

    assert not metadata_matches(saved, current)


def test_file_view_session_can_load_from_indexed_json_filename(tmp_path):
    from arrayscope.core.view_recipe import DisplaySettings, ViewRecipe
    from arrayscope.core.view_session import (
        FileViewSession,
        load_session_file,
        metadata_for_file,
        save_session_file,
    )
    from arrayscope.core.view_state import ViewState

    path = tmp_path / "scan.npy"
    data = np.zeros((4, 5), dtype=np.float32)
    np.save(path, data)
    metadata = metadata_for_file(path, data=data)
    session = FileViewSession(
        metadata=metadata,
        recipe=ViewRecipe(
            view_state=ViewState.from_shape(data.shape),
            display=DisplaySettings(
                channel="real",
                scale="linear",
                aspect_mode="square_pixels",
                window_mode="relative",
            ),
        ),
    )

    stored_path = save_session_file(tmp_path / "config", session)
    loaded = load_session_file(tmp_path / "config", metadata, data.shape, filename=stored_path.name)

    assert loaded is not None
    assert loaded.metadata == metadata
    assert loaded.recipe.view_state.shape == data.shape


def test_file_view_session_filename_is_readable_bounded_and_hashed(tmp_path):
    from arrayscope.core.view_session import metadata_for_file, session_filename_for_metadata

    path = tmp_path / "scan with spaces and symbols !@#$ and a very very very very long name.npy"
    data = np.zeros((4, 5), dtype=np.float32)
    np.save(path, data)

    filename = session_filename_for_metadata(metadata_for_file(path, data=data))

    assert filename.startswith("scan-with-spaces-and-symbols-and-a-very-very-ver")
    assert filename.endswith(".json")
    assert "--" in filename
    assert len(filename) <= 67


def test_file_view_session_metadata_includes_saved_at_without_matching_on_it(tmp_path):
    from arrayscope.core.view_session import metadata_for_file, metadata_matches

    path = tmp_path / "scan.npy"
    data = np.zeros((4, 5), dtype=np.float32)
    np.save(path, data)

    saved = metadata_for_file(path, data=data)
    current = dict(saved)
    current["saved_at"] = "different-debug-timestamp"

    assert saved["saved_at"]
    assert metadata_matches(saved, current)


def test_viewport_session_rejects_malformed_view_range():
    import pytest

    from arrayscope.core.view_session import viewport_from_mapping

    with pytest.raises(ValueError, match=r"viewport\.view_range"):
        viewport_from_mapping({"mode": "user", "view_range": [[0, 1, 2], [0, 1]]})


def test_viewport_session_rejects_malformed_viewport_shape():
    import pytest

    from arrayscope.core.view_session import viewport_from_mapping

    with pytest.raises(ValueError, match=r"viewport\.viewport_shape"):
        viewport_from_mapping({"mode": "user", "view_range": None, "viewport_shape": [100]})


def test_roi_session_round_trip_normalizes_geometry_and_rgb_color():
    from arrayscope.core.view_session import roi_from_mapping, roi_to_mapping

    selection = roi_from_mapping(
        {
            "id": "roi-3",
            "label": "ROI 3",
            "enabled": True,
            "color": [1, 2, 3, 4],
            "geometry": {
                "kind": "rectangle",
                "points": [[1, 2], [3, 4]],
                "rect": [1, 2, 3, 4],
                "line_width": 2,
                "closed": False,
                "image_axes": [1, 0],
            },
        }
    )

    assert selection.color == (1, 2, 3)
    assert selection.geometry.points == ((1.0, 2.0), (3.0, 4.0))
    assert selection.geometry.rect == (1.0, 2.0, 3.0, 4.0)
    assert selection.geometry.image_axes == (1, 0)
    assert roi_to_mapping(selection)["color"] == [1, 2, 3]
