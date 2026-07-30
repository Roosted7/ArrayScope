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
    actions = [event for event in manifest["events"] if event["kind"] != "caption"]
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
        ordered_labels = [event["label"] for event in overlay_choices]
        assert ordered_labels.index("choose log scale") < ordered_labels.index("choose magnitude")
        assert {"disable first FFT", "enable first FFT"} <= {
            event["label"] for event in actions if event["kind"] == "click"
        }
    if name == "roi":
        labels = {event["label"] for event in actions}
        assert {
            "draw rectangle ROI",
            "open inspection panel",
            "draw polyline profile",
            "close inspection panel",
            "enable live pointer profile",
        } <= labels
        assert not [
            event for event in actions if event["kind"] == "click" and event["target"] is None
        ], "ROI transitions must not imply a click at an unrelated cursor position"
    if name == "showcase":
        labels = {event["label"] for event in actions}
        assert {
            "scrub axis 2 slices",
            "crop axis 0 to 72:352",
            "restore full axis 0",
            "show image at true 1:1",
        } <= labels
    if name == "montage":
        montage_action = next(
            event for event in actions if event["label"] == "enter montage range on axis 2"
        )
        assert montage_action["kind"] == "type"
        assert montage_action["target"]["widget_type"] == "SliceIndexEdit"
        assert montage_action["caption"] == "Type ':' to montage the whole dimension"
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


def test_structured_demo_volume_has_edge_and_through_plane_evidence():
    namespace = runpy.run_path(str(RECORDER), run_name="arrayscope_demo_volume_test")

    volume = namespace["_structured_volume"](96, 80, 12)

    assert volume.shape == (96, 80, 12)
    assert str(volume.dtype) == "float32"
    assert volume[:7, 20:60, :].mean() > volume[10:17, 20:60, :].mean() + 0.2
    assert abs(float(volume[:, :, 0].mean()) - float(volume[:, :, -1].mean())) > 0.005
    assert float(volume[:, :, -1].std()) > 0.1
