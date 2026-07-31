"""Keeps the README demo scenarios recordable as the UI evolves.

tools/demo_recorder.py resolves every interaction target from live widget
geometry at run time. When a widget or window entry point it scripts against
is renamed or removed, these smoke runs fail immediately — long before
anyone next re-renders the README media — so the demo pipeline stays
maintainable instead of rotting silently.

Each case records a scenario in ``--smoke`` mode (high speed factor, low
fps, no encoding), which keeps a full scenario run to a few seconds.
"""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RECORDER = ROOT / "tools" / "demo_recorder.py"


def _scenario_names() -> tuple[str, ...]:
    namespace = runpy.run_path(str(RECORDER), run_name="arrayscope_demo_recorder")
    return tuple(namespace["SCENARIOS"])


@pytest.mark.parametrize("name", _scenario_names())
def test_demo_scenario_records_in_smoke_mode(name, tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(RECORDER),
            "--smoke",
            "--only",
            name,
            "--frames",
            str(tmp_path),
            "--jobs",
            "1",
            "--keep-frames",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(ROOT),
        check=False,
    )
    assert proc.returncode == 0, f"{name} failed:\n{proc.stdout}\n{proc.stderr}"
    frames = sorted((tmp_path / name).glob("frame_*.png"))
    assert frames, f"{name} recorded no frames"
    manifest = json.loads((tmp_path / name / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["scenario"] == name
    assert manifest["frame_count"] == len(frames)
    assert manifest["window_size"][0] == 1100
    assert manifest["window_size"][1] > 0
    assert manifest["capture_size"] == [
        manifest["window_size"][0],
        manifest["window_size"][1] + 78,
    ]
    with Image.open(frames[0]) as first_frame:
        assert list(first_frame.size) == manifest["capture_size"]
    captions = [event for event in manifest["events"] if event["kind"] == "caption"]
    actions = [event for event in manifest["events"] if event["kind"] not in ("caption", "status")]
    undeclared = [
        event["text"]
        for event in manifest["events"]
        if event["kind"] == "status"
        and not any(allowed in event["text"] for allowed in manifest["expected_status"])
    ]
    assert not undeclared, f"{name} put an unplanned status message on screen: {undeclared}"
    assert captions, f"{name} recorded no caption beats"
    assert actions, f"{name} recorded no action beats"
    assert all(event["end_frame"] is not None for event in manifest["events"])
    if name == "fft":
        overlay_choices = [
            event
            for event in actions
            if event["kind"] == "click"
            and event["target"]
            and event["target"]["widget_type"] == "QMenuAction"
        ]
        labels = {event["label"] for event in overlay_choices}
        assert {
            "choose centered FFT on axis 0",
            "choose centered FFT on axis 1",
            "choose magnitude",
            "choose log scale",
            "choose phase",
        } <= labels
        for event in overlay_choices:
            x, y, width, height = event["target"]["rect"]
            cursor_x, cursor_y = event["cursor_at_press"]
            assert x <= cursor_x <= x + width
            assert y <= cursor_y <= y + height
            assert event["detail_inset"] is True
            assert event["detail_rect"] is not None
        ordered_labels = [event["label"] for event in overlay_choices]
        assert ordered_labels.index("choose log scale") < ordered_labels.index("choose magnitude")
        assert ordered_labels.index("choose magnitude") < ordered_labels.index(
            "choose centered FFT on axis 0"
        )
        expected_choice_text = {
            "choose log scale": "Logarithmic",
            "choose magnitude": "Magnitude",
            "choose phase": "Phase",
        }
        for label, display_text in expected_choice_text.items():
            event = next(item for item in overlay_choices if item["label"] == label)
            assert event["target"]["display_text"] == display_text
            assert event["target"]["rect"][2] >= 80
        assert {"disable first FFT", "enable first FFT"} <= {
            event["label"] for event in actions if event["kind"] == "click"
        }
    if name == "roi":
        labels = {event["label"] for event in actions}
        assert {
            "choose Rectangle ROI tool",
            "add selected Rectangle ROI",
            "resize added rectangle ROI",
            "open inspection panel",
            "search for polyline ROI",
            "choose Draw polyline ROI",
            "draw polyline profile",
            "close inspection panel",
            "enable live pointer profile",
            "track live profile with pointer",
        } <= labels
        assert not [
            event for event in actions if event["kind"] == "click" and event["target"] is None
        ], "ROI transitions must not imply a click at an unrelated cursor position"
        inspection = next(
            event for event in actions if event["label"] == "focus live ROI statistics"
        )
        assert inspection["detail_inset"] is True
        profile = next(
            event for event in actions if event["label"] == "track live profile with pointer"
        )
        assert profile["profile_updates"] > 3
        assert profile["profile_start"] != profile["profile_end"]
    if name == "showcase":
        labels = {event["label"] for event in actions}
        assert {
            "hover image values",
            "adjust histogram window and level",
            "middle-drag axis 2 slices",
            "crop axis 0 to 72:352",
            "restore full axis 0",
            "show image at true 1:1",
        } <= labels
        framed_labels = {event["label"] for event in actions if event.get("detail_inset")}
        assert {
            "adjust histogram window and level",
            "crop axis 0 to 72:352",
            "show image at true 1:1",
        } <= framed_labels
        hover = next(event for event in actions if event["label"] == "hover image values")
        assert hover["hud_visible_frames"] > 0
        assert any("=" in text for text in hover["hud_texts"])
        levels = next(
            event for event in actions if event["label"] == "adjust histogram window and level"
        )
        assert levels["button"] == "left"
        assert levels["end_levels"][0] == pytest.approx(levels["start_levels"][0])
        assert levels["end_levels"][1] == pytest.approx(levels["cursor_values"][-1])
        scrub = next(event for event in actions if event["label"] == "middle-drag axis 2 slices")
        assert scrub["button"] == "middle"
        assert scrub["gesture_axis"] == "horizontal"
        assert scrub["target_axis"] == 2
        assert scrub["after_index"] != scrub["before_index"]
    if name == "montage":
        montage_action = next(
            event for event in actions if event["label"] == "enter montage range on axis 2"
        )
        assert montage_action["kind"] == "type"
        assert montage_action["target"]["widget_type"] == "SliceIndexEdit"
        assert montage_action["caption"] == "Type ':' to montage the whole dimension"
        assert montage_action["detail_inset"] is True
        zoom = next(event for event in actions if event["label"] == "middle-drag montage zoom")
        assert zoom["button"] == "middle"
        assert zoom["gesture_axis"] == "vertical"
        before_span = zoom["before_view_range"][0][1] - zoom["before_view_range"][0][0]
        after_span = zoom["after_view_range"][0][1] - zoom["after_view_range"][0][0]
        assert after_span < before_span
    if name == "colormap":
        choices = {event["label"]: event for event in actions}
        assert {"open colormap choices", "choose viridis colormap"} <= choices.keys()
        chosen = choices["choose viridis colormap"]
        assert chosen["target"]["display_text"] == "viridis"
        assert chosen["detail_inset"] is True
    if name == "workflow":
        labels = {event["label"] for event in actions}
        assert {
            "choose crop axis 0",
            "show crop parameter popup",
            "set crop start to 16",
            "set crop stop to 80",
            "apply crop",
            "preview video and PNG export",
        } <= labels
        assert {
            "choose crop axis 0",
            "set crop start to 16",
            "set crop stop to 80",
            "apply crop",
            "preview video and PNG export",
        } <= {event["label"] for event in actions if event.get("detail_inset")}


def test_caption_wrapping_is_bounded_to_two_readable_lines(qt_app):
    from pyqtgraph.Qt import QtGui

    namespace = runpy.run_path(str(RECORDER), run_name="arrayscope_demo_caption_test")
    font = QtGui.QFont()
    font.setPixelSize(namespace["CAPTION_FONT_PX"])
    metrics = QtGui.QFontMetricsF(font)
    max_width = 260.0
    lines = namespace["_wrap_caption_lines"](
        metrics,
        "A deliberately long walkthrough caption that cannot fit on one narrow line",
        max_width,
    )

    assert len(lines) == 2
    assert all(metrics.horizontalAdvance(line) <= max_width for line in lines)


def test_detail_inset_uses_a_clear_corner_and_skips_full_size_duplicates():
    namespace = runpy.run_path(str(RECORDER), run_name="arrayscope_demo_detail_test")
    geometry = namespace["_detail_inset_geometry"]((650, 40, 150, 50), 1100, 780)

    assert geometry is not None
    assert geometry["scale"] >= namespace["DETAIL_INSET_MIN_SCALE"]
    assert namespace["_rect_intersection_area"]((650, 40, 150, 50), geometry["box"]) == 0
    assert namespace["_detail_inset_geometry"]((100, 100, 700, 300), 1100, 780) is None


def test_structured_demo_volume_has_edge_and_through_plane_evidence():
    namespace = runpy.run_path(str(RECORDER), run_name="arrayscope_demo_volume_test")

    volume = namespace["_structured_volume"](96, 80, 12)

    assert volume.shape == (96, 80, 12)
    assert str(volume.dtype) == "float32"
    assert volume[:7, 20:60, :].mean() > volume[10:17, 20:60, :].mean() + 0.2
    assert abs(float(volume[:, :, 0].mean()) - float(volume[:, :, -1].mean())) > 0.005
    assert float(volume[:, :, -1].std()) > 0.1
