"""End-to-end checks for agent-readable README walkthrough inspection.

The subprocess cases below prove the command works. They cannot, however, put
``tools/demo_inspector.py`` in any test's coverage fingerprint — a child
process is invisible to selection — so a change to the gate would never
re-run the tests that guard it. The in-process cases load the module and
exercise its decisions directly, which is both the finer-grained coverage and
the reason an edit to the inspector is selected at all.
"""

from __future__ import annotations

import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
INSPECTOR = ROOT / "tools" / "demo_inspector.py"


@pytest.fixture(scope="module")
def inspector():
    return runpy.run_path(str(INSPECTOR), run_name="arrayscope_demo_inspector")


def _facts(
    inspector,
    scenario,
    fmt,
    *,
    width,
    height=600,
    duration=10.0,
    size_bytes=1000,
):
    return inspector["MediaFacts"](
        path=f"/tmp/{scenario}.{fmt}",
        scenario=scenario,
        format=fmt,
        width=width,
        height=height,
        fps=30.0,
        frames=300,
        stream_index=0,
        duration_s=duration,
        size_bytes=size_bytes,
        inline_scale=inspector["README_INLINE_WIDTH"] / width,
    )


def _timeline(events, *, frame_count=200, fps=30, speed=1.0):
    return {
        "schema_version": 1,
        "scenario": "probe",
        "record_fps": fps,
        "speed": speed,
        "frame_count": frame_count,
        "window_size": [1100, 780],
        "events": events,
    }


def _caption(text, start):
    return {"kind": "caption", "text": text, "start_frame": start, "end_frame": start + 100}


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


def test_strict_inspection_rejects_a_fallback_that_readme_must_enlarge(tiny_video, tmp_path):
    gif = tiny_video.with_suffix(".gif")
    _ffmpeg("-i", str(tiny_video), "-vf", "scale=760:-1", str(gif))

    proc = subprocess.run(
        [
            sys.executable,
            str(INSPECTOR),
            "--media-root",
            str(tiny_video.parent),
            "--out",
            str(tmp_path / "inspection"),
            "--expected-formats",
            "gif",
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
    assert "[error] tiny: GIF is enlarged 1.16x at README width" in proc.stdout


def test_every_inline_format_must_fill_the_readme_slot(inspector):
    """The AVIF is the primary inline source, not just the GIF fallback."""
    grouped = {
        "narrow": [
            _facts(inspector, "narrow", "mp4", width=1100),
            _facts(inspector, "narrow", "avif", width=760),
            _facts(inspector, "narrow", "gif", width=760),
        ]
    }

    messages = [
        finding.message for finding in inspector["parity_findings"](grouped, ("mp4", "avif", "gif"))
    ]

    assert "AVIF is enlarged 1.16x at README width" in messages
    assert "GIF is enlarged 1.16x at README width" in messages
    # The linked MP4 is never placed in the 880 px slot.
    assert not [message for message in messages if message.startswith("MP4")]


def test_every_inline_format_must_stay_under_the_size_budget(inspector):
    grouped = {
        "heavy": [
            _facts(inspector, "heavy", "mp4", width=1100, size_bytes=9_000_000),
            _facts(inspector, "heavy", "avif", width=880, size_bytes=3_500_001),
            _facts(inspector, "heavy", "gif", width=880, size_bytes=3_500_000),
        ]
    }

    messages = [
        finding.message for finding in inspector["parity_findings"](grouped, ("mp4", "avif", "gif"))
    ]

    assert "AVIF is 3.50 MB; inline budget is 3.50 MB" in messages
    assert not [message for message in messages if message.startswith("GIF")]
    assert not [message for message in messages if message.startswith("MP4")]


def test_reported_inline_scale_shows_enlargement_rather_than_clamping(inspector):
    facts = _facts(inspector, "narrow", "gif", width=760)
    assert facts.inline_scale > 1.0


def test_a_non_click_action_is_still_held_to_its_recorded_target(inspector):
    """An edit or preview names a control too; a cursor elsewhere is a lie."""
    events = [
        _caption("Edit the bounds", 0),
        {
            "kind": "edit",
            "label": "set crop start",
            "caption": "Edit the bounds",
            "start_frame": 20,
            "press_frame": 25,
            "activate_frame": 26,
            "end_frame": 40,
            "target": {
                "widget_type": "QSpinBox",
                "object_name": "",
                "rect": [100, 100, 60, 24],
                "point": [130, 112],
            },
            "cursor_at_press": [400, 400],
        },
    ]

    messages = [
        finding.message
        for finding in inspector["timeline_findings"](
            "probe",
            _timeline(events),
            _facts(inspector, "probe", "mp4", width=1100, duration=200 / 30),
        )
    ]

    assert "edit 'set crop start' cursor misses its target" in messages


def test_a_declared_detail_inset_must_name_an_in_frame_source(inspector):
    events = [
        _caption("Inspect the control", 0),
        {
            "kind": "preview",
            "label": "show export",
            "caption": "Inspect the control",
            "start_frame": 20,
            "press_frame": None,
            "activate_frame": 25,
            "end_frame": 40,
            "target": None,
            "cursor_at_press": None,
            "detail_inset": True,
            "detail_rect": [1050, 700, 200, 120],
        },
    ]

    messages = [
        finding.message
        for finding in inspector["timeline_findings"](
            "probe",
            _timeline(events),
            _facts(inspector, "probe", "mp4", width=1100, duration=200 / 30),
        )
    ]

    assert "preview 'show export' detail source lies outside the window" in messages


def test_an_action_must_be_introduced_by_a_readable_caption(inspector):
    events = [
        _caption("Choose crop", 18),
        {
            "kind": "click",
            "label": "choose crop",
            "caption": "Choose crop",
            "start_frame": 20,
            "press_frame": 22,
            "activate_frame": 23,
            "end_frame": 40,
            "target": {
                "widget_type": "QMenuAction",
                "object_name": "crop",
                "rect": [100, 100, 60, 24],
                "point": [130, 112],
            },
            "cursor_at_press": [130, 112],
        },
    ]

    messages = [
        finding.message
        for finding in inspector["timeline_findings"](
            "probe",
            _timeline(events),
            _facts(inspector, "probe", "mp4", width=1100, duration=200 / 30),
        )
    ]

    assert messages == [
        "click 'choose crop' starts 2 frames after its caption, before it can be read"
    ]


def test_a_caption_lead_is_judged_on_the_runs_own_time_base(inspector):
    """A smoke run divides every duration; its leads shrink with them."""
    events = [
        _caption("Choose crop", 18),
        {
            "kind": "transition",
            "label": "open the panel",
            "caption": "Choose crop",
            "start_frame": 20,
            "press_frame": None,
            "activate_frame": 21,
            "end_frame": 40,
            "target": None,
            "cursor_at_press": None,
        },
    ]

    assert not inspector["timeline_findings"](
        "probe",
        _timeline(events, speed=6.0),
        _facts(inspector, "probe", "mp4", width=1100, duration=200 / 30),
    )


def test_event_evidence_is_ordered_by_the_frame_it_shows(inspector):
    """A typed action opens before the focus click nested inside it."""
    typed = {
        "kind": "type",
        "label": "type a range",
        "start_frame": 89,
        "press_frame": 101,
        "activate_frame": 110,
        "end_frame": 115,
    }
    focus = {
        "kind": "click",
        "label": "focus the range",
        "start_frame": 89,
        "press_frame": 90,
        "activate_frame": 91,
        "end_frame": 100,
    }
    triplet = inspector["_event_frame_triplet"]

    assert triplet(focus)[1][1] < triplet(typed)[1][1]
