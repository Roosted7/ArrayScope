"""End-to-end checks for agent-readable README walkthrough inspection."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
INSPECTOR = ROOT / "tools" / "demo_inspector.py"


def _ffmpeg(*args: str) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.fixture
def tiny_video(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("demo inspection requires ffmpeg and ffprobe")
    media = tmp_path / "media"
    media.mkdir()
    path = media / "tiny.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x180:rate=8:duration=1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    )
    return path


def _inspect(media_root: Path, out: Path, *extra: str):
    return subprocess.run(
        [
            sys.executable,
            str(INSPECTOR),
            "--media-root",
            str(media_root),
            "--out",
            str(out),
            "--expected-formats",
            "mp4",
            "--samples",
            "4",
            "--columns",
            "2",
            "--thumb-width",
            "160",
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=60,
        check=False,
    )


def test_inspector_decodes_final_media_to_storyboard_json_and_html(tiny_video, tmp_path):
    (tiny_video.parent / "tiny.timeline.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenario": "tiny",
                "record_fps": 8,
                "speed": 1.0,
                "frame_count": 8,
                "window_size": [320, 180],
                "events": [
                    {
                        "kind": "caption",
                        "text": "Inspect the action",
                        "start_frame": 0,
                        "end_frame": 8,
                    },
                    {
                        "kind": "click",
                        "label": "Visible target",
                        "start_frame": 1,
                        "press_frame": 2,
                        "activate_frame": 3,
                        "end_frame": 5,
                        "target": {
                            "widget_type": "QPushButton",
                            "object_name": "target",
                            "rect": [100, 50, 80, 30],
                            "point": [140, 65],
                        },
                        "cursor_at_press": [140, 65],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "inspection"
    proc = _inspect(tiny_video.parent, out)

    assert proc.returncode == 0, proc.stderr
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["expected_formats"] == ["mp4"]
    assert len(report["media"]) == 1
    facts = report["media"][0]
    assert facts["scenario"] == "tiny"
    assert facts["format"] == "mp4"
    assert (facts["width"], facts["height"]) == (320, 180)
    assert facts["duration_s"] == pytest.approx(1.0, abs=0.05)
    assert facts["fps"] == pytest.approx(8.0)
    board = out / "tiny-mp4-storyboard.png"
    assert board.exists()
    assert (out / "tiny-events.png").exists()
    with Image.open(board) as image:
        assert image.width > 2 * 160
        assert image.height > 2 * 90
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "tiny" in html
    assert "Action-centered evidence" in html
    assert tiny_video.resolve().as_uri() in html


def test_strict_inspection_fails_when_a_companion_format_is_missing(tiny_video, tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(INSPECTOR),
            "--media-root",
            str(tiny_video.parent),
            "--out",
            str(tmp_path / "inspection"),
            "--expected-formats",
            "mp4,gif",
            "--samples",
            "1",
            "--strict",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=60,
        check=False,
    )

    assert proc.returncode == 1
    assert "[error] tiny: missing expected formats: gif" in proc.stdout


def test_strict_inspection_rejects_a_click_without_a_visible_target(tiny_video, tmp_path):
    (tiny_video.parent / "tiny.timeline.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenario": "tiny",
                "record_fps": 8,
                "speed": 1.0,
                "frame_count": 8,
                "window_size": [320, 180],
                "events": [
                    {
                        "kind": "click",
                        "label": "disconnected action",
                        "start_frame": 1,
                        "press_frame": 2,
                        "activate_frame": 3,
                        "end_frame": 5,
                        "target": None,
                        "cursor_at_press": [140, 65],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    proc = _inspect(tiny_video.parent, tmp_path / "inspection", "--strict")

    assert proc.returncode == 1
    assert "[error] tiny: click 'disconnected action' has no visible target" in proc.stdout
