#!/usr/bin/env python
"""Render agent-readable inspection artifacts for the README walkthroughs.

The recorder's MP4/GIF/AVIF files are the product. A successful encode alone
does not show whether a click is visible, a caption is legible, or a feature
change reads clearly at README scale. This tool decodes those final files and
creates:

* timestamped contact sheets that can be inspected as ordinary PNGs;
* a machine-readable report with duration, frame, scale, and parity checks;
* a local HTML gallery for reviewing every format at its encoded size.

Usage:
    python tools/demo_inspector.py
    python tools/demo_inspector.py --only roi
    python tools/demo_inspector.py --strict
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEDIA_ROOT = REPO_ROOT / "docs" / "media"
DEFAULT_OUT = REPO_ROOT / "tests" / "artifacts" / "demo-inspection"
DEFAULT_FORMATS = ("mp4", "avif", "gif")
README_INLINE_WIDTH = 880
MAX_INLINE_BYTES = 3_500_000
# MP4 is linked, not embedded; every other format is rendered in the README
# slot and must therefore be at least as wide as it.
LINKED_FORMATS = frozenset({"mp4"})
# The recorder establishes a caption before the action it describes. Verify
# that here rather than trusting construction, and scale it by the run's
# speed factor so a fast smoke timeline is judged on its own time base.
CAPTION_LEAD_MIN_S = 0.25


@dataclass(frozen=True)
class MediaFacts:
    path: str
    scenario: str
    format: str
    width: int
    height: int
    fps: float
    frames: int | None
    stream_index: int
    duration_s: float
    size_bytes: int
    inline_scale: float


@dataclass(frozen=True)
class Finding:
    severity: str
    scenario: str
    message: str


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])}… failed:\n{proc.stderr[-2000:]}")
    return proc


def _fraction(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    return float(Fraction(value))


def probe(path: Path) -> MediaFacts:
    raw = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=index,width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ]
    ).stdout
    payload = json.loads(raw)
    candidates = [
        stream
        for stream in payload.get("streams", ())
        if stream.get("width") and stream.get("height")
    ]
    video = max(
        candidates,
        key=lambda stream: (
            int(stream.get("nb_frames") or 0)
            if stream.get("nb_frames") not in (None, "N/A")
            else 0,
            _fraction(stream.get("avg_frame_rate")),
        ),
        default=None,
    )
    if video is None:
        raise RuntimeError(f"{path} has no video stream")
    duration = float(payload["format"].get("duration") or 0.0)
    frames_text = video.get("nb_frames")
    frames = int(frames_text) if frames_text and frames_text != "N/A" else None
    return MediaFacts(
        path=str(path.resolve()),
        scenario=path.stem,
        format=path.suffix.removeprefix(".").lower(),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=_fraction(video.get("avg_frame_rate")),
        frames=frames,
        stream_index=int(video.get("index") or 0),
        duration_s=duration,
        size_bytes=int(payload["format"].get("size") or path.stat().st_size),
        # Report the true README scale. Clamping this to 1.0 would hide the
        # one direction that damages the artifact: a file narrower than its
        # slot, which the browser enlarges.
        inline_scale=README_INLINE_WIDTH / int(video["width"]),
    )


def sample_times(duration_s: float, count: int, *, safe_tail_s: float = 0.05) -> list[float]:
    if count < 1:
        raise ValueError("sample count must be positive")
    if duration_s <= 0:
        return [0.0] * count
    # Include the opening beat, sample the body evenly, and stay far enough
    # from EOF that inter-frame formats can return a decoded final sample.
    end = max(0.0, duration_s - min(safe_tail_s, duration_s / 2.0))
    if count == 1:
        return [end / 2.0]
    return [end * index / (count - 1) for index in range(count)]


def _extract_frame(
    media: Path,
    stream_index: int,
    at_s: float,
    output: Path,
    *,
    frame_index: int | None,
) -> None:
    selection = (
        ["-vf", f"select=eq(n\\,{frame_index})"]
        if frame_index is not None
        else ["-ss", f"{at_s:.6f}"]
    )
    _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(media),
            "-map",
            f"0:{stream_index}",
            *selection,
            "-frames:v",
            "1",
            "-fps_mode",
            "vfr",
            str(output),
        ]
    )


def render_storyboard(
    media: Path,
    facts: MediaFacts,
    output: Path,
    *,
    samples: int,
    columns: int,
    thumb_width: int,
) -> None:
    frame_period = 1.0 / facts.fps if facts.fps > 0 else 0.05
    # Prefer frame ordinals. Animated GIF container timestamps can extend well
    # beyond the final timestamp ffmpeg accepts for seeking; ordinal selection
    # still reaches the actual final frame.
    if facts.frames:
        if samples == 1:
            frame_indices = [(facts.frames - 1) // 2]
        else:
            frame_indices = [
                round((facts.frames - 1) * index / (samples - 1)) for index in range(samples)
            ]
        times = [
            min(facts.duration_s, index / facts.fps) if facts.fps > 0 else 0.0
            for index in frame_indices
        ]
    else:
        frame_indices = [None] * samples
        times = sample_times(facts.duration_s, samples, safe_tail_s=max(0.5, 3 * frame_period))
    rows = math.ceil(samples / columns)
    label_h = 30
    gap = 8
    thumb_height = max(1, round(facts.height * thumb_width / facts.width))
    sheet = Image.new(
        "RGB",
        (
            columns * thumb_width + (columns + 1) * gap,
            rows * (thumb_height + label_h) + (rows + 1) * gap,
        ),
        (242, 242, 242),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    with tempfile.TemporaryDirectory(prefix="arrayscope-demo-inspector-") as temp:
        temp_root = Path(temp)
        for index, (at_s, frame_index) in enumerate(zip(times, frame_indices, strict=True)):
            frame_path = temp_root / f"{index:03d}.png"
            _extract_frame(
                media,
                facts.stream_index,
                at_s,
                frame_path,
                frame_index=frame_index,
            )
            with Image.open(frame_path) as decoded:
                frame = decoded.convert("RGB")
                frame.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
                tile = Image.new("RGB", (thumb_width, thumb_height), (0, 0, 0))
                tile.paste(
                    frame,
                    ((thumb_width - frame.width) // 2, (thumb_height - frame.height) // 2),
                )
            column = index % columns
            row = index // columns
            x = gap + column * thumb_width
            y = gap + row * (thumb_height + label_h)
            sheet.paste(tile, (x, y))
            draw.text(
                (x + 8, y + thumb_height + 5),
                f"{at_s:05.1f}s  {facts.scenario}.{facts.format}",
                fill=(25, 25, 25),
                font=font,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "PNG", optimize=True)


def _event_frame_triplet(event: dict) -> list[tuple[str, int]]:
    start = int(event["start_frame"])
    press = event.get("press_frame")
    activate = event.get("activate_frame")
    end = max(start, int(event["end_frame"]) - 1)
    focus = int(press if press is not None else start)
    changed = int(activate if activate is not None else end)
    return [("before", max(0, focus - 1)), ("act", focus), ("after", max(changed, end))]


def render_event_sheet(
    media: Path,
    facts: MediaFacts,
    timeline: dict,
    output: Path,
    *,
    thumb_width: int,
) -> int:
    # Order by the frame each row is centered on, not by emission order: a
    # typed action opens before the focus click nested inside it, so the raw
    # list would show the later frames first.
    actions = sorted(
        (event for event in timeline["events"] if event["kind"] not in ("caption", "status")),
        key=lambda event: _event_frame_triplet(event)[1][1],
    )
    if not actions:
        return 0
    columns = 3
    label_h = 48
    gap = 8
    thumb_height = max(1, round(facts.height * thumb_width / facts.width))
    sheet = Image.new(
        "RGB",
        (
            columns * thumb_width + (columns + 1) * gap,
            len(actions) * (thumb_height + label_h) + (len(actions) + 1) * gap,
        ),
        (242, 242, 242),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    source_fps = float(timeline["record_fps"])
    with tempfile.TemporaryDirectory(prefix="arrayscope-demo-events-") as temp:
        temp_root = Path(temp)
        for row, event in enumerate(actions):
            for column, (stage, source_frame) in enumerate(_event_frame_triplet(event)):
                at_s = source_frame / source_fps
                media_frame = (
                    min(facts.frames - 1, round(at_s * facts.fps)) if facts.frames else None
                )
                frame_path = temp_root / f"{row:03d}-{column}.png"
                _extract_frame(
                    media,
                    facts.stream_index,
                    at_s,
                    frame_path,
                    frame_index=media_frame,
                )
                with Image.open(frame_path) as decoded:
                    frame = decoded.convert("RGB")
                    frame.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
                    tile = Image.new("RGB", (thumb_width, thumb_height), (0, 0, 0))
                    tile.paste(
                        frame,
                        ((thumb_width - frame.width) // 2, (thumb_height - frame.height) // 2),
                    )
                x = gap + column * thumb_width
                y = gap + row * (thumb_height + label_h)
                sheet.paste(tile, (x, y))
                label = f"{event['kind']} · {event.get('label', '')}"
                draw.text((x + 7, y + thumb_height + 4), label[:58], fill=(20, 20, 20), font=font)
                draw.text(
                    (x + 7, y + thumb_height + 24),
                    f"{stage} · frame {source_frame} · {at_s:.2f}s",
                    fill=(60, 60, 60),
                    font=font,
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "PNG", optimize=True)
    return len(actions)


def parity_findings(
    grouped: dict[str, list[MediaFacts]],
    expected_formats: tuple[str, ...],
    *,
    max_inline_bytes: int = MAX_INLINE_BYTES,
) -> list[Finding]:
    findings: list[Finding] = []
    for scenario, items in sorted(grouped.items()):
        by_format = {item.format: item for item in items}
        missing = sorted(set(expected_formats) - set(by_format))
        if missing:
            findings.append(
                Finding("error", scenario, f"missing expected formats: {', '.join(missing)}")
            )
        for item in sorted(items, key=lambda value: value.format):
            if item.format in LINKED_FORMATS:
                continue
            if item.width < README_INLINE_WIDTH:
                findings.append(
                    Finding(
                        "error",
                        scenario,
                        f"{item.format.upper()} is enlarged {item.inline_scale:.2f}x "
                        "at README width",
                    )
                )
            if item.size_bytes > max_inline_bytes:
                findings.append(
                    Finding(
                        "error",
                        scenario,
                        f"{item.format.upper()} is {item.size_bytes / 1e6:.2f} MB; "
                        f"inline budget is {max_inline_bytes / 1e6:.2f} MB",
                    )
                )
        if len(items) < 2:
            continue
        aspects = [item.width / item.height for item in items]
        # Encoders round scaled dimensions to whole pixels. Permit that
        # sub-pixel-scale drift while refusing a visibly different crop.
        if max(aspects) - min(aspects) > 0.005:
            findings.append(
                Finding("error", scenario, "encoded formats have different aspect ratios")
            )
        durations = [item.duration_s for item in items]
        if max(durations) - min(durations) > 0.12:
            findings.append(
                Finding(
                    "error",
                    scenario,
                    f"encoded durations differ by {max(durations) - min(durations):.3f}s",
                )
            )
    return findings


def timeline_findings(scenario: str, timeline: dict, media: MediaFacts) -> list[Finding]:
    findings: list[Finding] = []
    frame_count = int(timeline.get("frame_count") or 0)
    fps = float(timeline.get("record_fps") or 0)
    window_size = timeline.get("window_size")
    if frame_count < 1 or fps <= 0 or not window_size:
        return [Finding("error", scenario, "timeline lacks frame count, fps, or window size")]
    expected_duration = frame_count / fps
    if abs(expected_duration - media.duration_s) > max(0.12, 1.5 / fps):
        findings.append(
            Finding(
                "error",
                scenario,
                f"timeline duration {expected_duration:.3f}s does not match "
                f"{media.format} duration {media.duration_s:.3f}s",
            )
        )
    width, height = (float(value) for value in window_size)
    min_lead_frames = CAPTION_LEAD_MIN_S * fps / max(1.0, float(timeline.get("speed") or 1.0))
    expected_status = tuple(timeline.get("expected_status") or ())
    captions: list[dict] = []
    for index, event in enumerate(timeline.get("events", ())):
        start = event.get("start_frame")
        end = event.get("end_frame")
        if start is None or end is None or not (0 <= int(start) < int(end) <= frame_count):
            findings.append(Finding("error", scenario, f"event {index} has an invalid frame range"))
            continue
        kind = event.get("kind")
        if kind == "caption":
            captions.append(event)
            continue
        if kind == "status":
            # The viewer reads whatever the application says. A message the
            # scenario did not plan for is a fault it is putting on screen.
            text = str(event.get("text") or "")
            if not any(allowed in text for allowed in expected_status):
                seconds = (int(end) - int(start)) / fps
                findings.append(
                    Finding(
                        "error",
                        scenario,
                        f"undeclared status message on screen for {seconds:.2f}s: {text!r}",
                    )
                )
            continue
        name = f"{kind} {event.get('label', index)!r}"
        findings.extend(_caption_findings(scenario, name, event, captions, min_lead_frames))
        if kind == "drag" and event.get("button") not in {"left", "middle", "right"}:
            findings.append(Finding("error", scenario, f"{name} does not name its pressed button"))
        if event.get("detail_inset"):
            detail_rect = event.get("detail_rect")
            if detail_rect is None or len(detail_rect) != 4:
                findings.append(
                    Finding("error", scenario, f"{name} has no detail source rectangle")
                )
            else:
                detail_x, detail_y, detail_width, detail_height = (
                    float(value) for value in detail_rect
                )
                if (
                    detail_x < 0
                    or detail_y < 0
                    or detail_width <= 0
                    or detail_height <= 0
                    or detail_x + detail_width > width
                    or detail_y + detail_height > height
                ):
                    findings.append(
                        Finding("error", scenario, f"{name} detail source lies outside the window")
                    )
            detail_scale = float(event.get("detail_scale") or 0.0)
            if detail_scale < 1.15:
                findings.append(
                    Finding(
                        "error",
                        scenario,
                        f"{name} detail inset magnifies only {detail_scale:.2f}x",
                    )
                )
            detail_box = event.get("detail_box")
            if detail_box is None or len(detail_box) != 4:
                findings.append(Finding("error", scenario, f"{name} has no detail destination"))
            else:
                box_x, box_y, box_width, box_height = (float(value) for value in detail_box)
                if (
                    box_x < 0
                    or box_y < 0
                    or box_width <= 0
                    or box_height <= 0
                    or box_x + box_width > width
                    or box_y + box_height > height
                ):
                    findings.append(
                        Finding(
                            "error", scenario, f"{name} detail destination lies outside the window"
                        )
                    )
        target = event.get("target")
        cursor = event.get("cursor_at_press")
        if target is None:
            # Only a click asserts "the pointer caused this". A transition,
            # gesture, or viewport move is honest without a widget target.
            if kind == "click":
                findings.append(Finding("error", scenario, f"{name} has no visible target"))
            continue
        x, y, target_width, target_height = (float(value) for value in target["rect"])
        if x < 0 or y < 0 or x + target_width > width or y + target_height > height:
            findings.append(Finding("error", scenario, f"{name} target lies outside the window"))
        if cursor is None:
            if kind == "click":
                findings.append(Finding("error", scenario, f"{name} has no cursor"))
            continue
        cursor_x, cursor_y = (float(value) for value in cursor)
        if not (x <= cursor_x <= x + target_width and y <= cursor_y <= y + target_height):
            findings.append(Finding("error", scenario, f"{name} cursor misses its target"))
    return findings


def _caption_findings(
    scenario: str,
    name: str,
    event: dict,
    captions: list[dict],
    min_lead_frames: float,
) -> list[Finding]:
    """An action the viewer cannot read an explanation for is not evidence."""
    text = event.get("caption")
    if not text:
        return [Finding("error", scenario, f"{name} runs with no caption on screen")]
    owner = next(
        (
            caption
            for caption in reversed(captions)
            if caption.get("text") == text
            and int(caption["start_frame"]) <= int(event["start_frame"])
        ),
        None,
    )
    if owner is None:
        return [Finding("error", scenario, f"{name} claims a caption that never started before it")]
    lead = int(event["start_frame"]) - int(owner["start_frame"])
    if lead < min_lead_frames:
        return [
            Finding(
                "error",
                scenario,
                f"{name} starts {lead} frames after its caption, before it can be read",
            )
        ]
    return []


def _write_html(
    output: Path,
    grouped: dict[str, list[MediaFacts]],
    storyboards: dict[str, str],
    event_sheets: dict[str, str],
    findings: list[Finding],
) -> None:
    sections: list[str] = []
    for scenario, items in sorted(grouped.items()):
        cards: list[str] = []
        for item in sorted(items, key=lambda value: DEFAULT_FORMATS.index(value.format)):
            source = Path(item.path).resolve().as_uri()
            storyboard = storyboards[item.path]
            player = (
                f'<video controls loop muted src="{source}"></video>'
                if item.format == "mp4"
                else f'<img src="{source}" alt="{html.escape(item.scenario)} {item.format} animation">'
            )
            cards.append(
                "<article>"
                f"<h3>{html.escape(item.format.upper())}</h3>"
                f"<p>{item.width}×{item.height}; {item.duration_s:.2f}s; "
                f"{item.fps:.2f} fps; {item.size_bytes / 1e6:.2f} MB; "
                f"README scale {item.inline_scale:.2f}×</p>"
                f"{player}"
                f'<a href="{html.escape(storyboard)}">'
                f'<img src="{html.escape(storyboard)}" alt="{html.escape(scenario)} storyboard"></a>'
                "</article>"
            )
        event_sheet = event_sheets.get(scenario)
        event_html = (
            f"<h3>Action-centered evidence</h3>"
            f'<a href="{html.escape(event_sheet)}"><img src="{html.escape(event_sheet)}" '
            f'alt="{html.escape(scenario)} action-centered storyboard"></a>'
            if event_sheet
            else "<p>No timeline manifest was available for action-centered evidence.</p>"
        )
        sections.append(
            f"<section><h2>{html.escape(scenario)}</h2>{event_html}{''.join(cards)}</section>"
        )
    finding_items = (
        "".join(
            f'<li class="{item.severity}"><strong>{html.escape(item.scenario)}</strong>: '
            f"{html.escape(item.message)}</li>"
            for item in findings
        )
        or "<li>No structural findings.</li>"
    )
    output.write_text(
        """<!doctype html>
<html lang="en"><meta charset="utf-8">
<title>ArrayScope walkthrough inspection</title>
<style>
body{font:16px system-ui;margin:2rem;background:#181a1b;color:#eee}
section{margin:3rem 0}article{margin:1.5rem 0;padding:1rem;background:#242729}
video,img{display:block;max-width:880px;width:100%;margin:.75rem 0}
h1,h2,h3{color:#fff}.error{color:#ff9c9c}.note{color:#ffd479}
</style>
<h1>ArrayScope walkthrough inspection</h1>
<p>Review the decoded final formats below. Storyboards are timestamped and
can be opened directly by image-capable agents.</p>
<h2>Structural findings</h2><ul>"""
        + finding_items
        + "</ul>"
        + "".join(sections)
        + "</html>\n",
        encoding="utf-8",
    )


def inspect(
    media_root: Path,
    output: Path,
    *,
    only: str | None,
    expected_formats: tuple[str, ...],
    samples: int,
    columns: int,
    thumb_width: int,
    require_timelines: bool = False,
    max_inline_bytes: int = MAX_INLINE_BYTES,
) -> tuple[dict, list[Finding]]:
    media = sorted(
        path
        for suffix in expected_formats
        for path in media_root.glob(f"*.{suffix}")
        if not only or only in path.stem
    )
    if not media:
        raise RuntimeError(f"no matching media under {media_root}")
    output.mkdir(parents=True, exist_ok=True)
    facts = [probe(path) for path in media]
    grouped: dict[str, list[MediaFacts]] = {}
    for item in facts:
        grouped.setdefault(item.scenario, []).append(item)
    findings = parity_findings(
        grouped,
        expected_formats,
        max_inline_bytes=max_inline_bytes,
    )
    storyboards: dict[str, str] = {}
    timelines: dict[str, dict] = {}
    event_sheets: dict[str, str] = {}
    for item in facts:
        media_path = Path(item.path)
        board_name = f"{item.scenario}-{item.format}-storyboard.png"
        render_storyboard(
            media_path,
            item,
            output / board_name,
            samples=samples,
            columns=columns,
            thumb_width=thumb_width,
        )
        storyboards[item.path] = board_name
    for scenario, items in sorted(grouped.items()):
        timeline_path = media_root / f"{scenario}.timeline.json"
        if not timeline_path.exists():
            # Without a timeline nothing can adjudicate cursor targets or
            # caption timing, so a gate that accepts it is a gate that
            # cannot fail. Legacy media stays inspectable, never certified.
            findings.append(
                Finding(
                    "error" if require_timelines else "note",
                    scenario,
                    "no event timeline; uniform samples only",
                )
            )
            continue
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        timelines[scenario] = timeline
        preferred = next(
            (item for item in items if item.format == "mp4"),
            items[0],
        )
        findings.extend(timeline_findings(scenario, timeline, preferred))
        sheet_name = f"{scenario}-events.png"
        render_event_sheet(
            Path(preferred.path),
            preferred,
            timeline,
            output / sheet_name,
            thumb_width=thumb_width,
        )
        event_sheets[scenario] = sheet_name
    report = {
        "media_root": str(media_root.resolve()),
        "expected_formats": list(expected_formats),
        "max_inline_bytes": max_inline_bytes,
        "media": [asdict(item) for item in facts],
        "findings": [asdict(item) for item in findings],
        "timelines": timelines,
        "event_sheets": {
            scenario: str((output / name).resolve()) for scenario, name in event_sheets.items()
        },
        "storyboards": {
            str(Path(path).resolve()): str((output / name).resolve())
            for path, name in storyboards.items()
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_html(output / "index.html", grouped, storyboards, event_sheets, findings)
    return report, findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", help="substring filter on scenario names")
    parser.add_argument(
        "--expected-formats",
        default=",".join(DEFAULT_FORMATS),
        help="comma-separated companion formats to inspect and require",
    )
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=440)
    parser.add_argument(
        "--max-inline-bytes",
        type=int,
        default=MAX_INLINE_BYTES,
        help="maximum encoded size for each README-inline format",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "return non-zero for structural errors such as missing or mismatched "
            "companions, and require every scenario to ship an event timeline"
        ),
    )
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("demo inspection requires ffmpeg and ffprobe on PATH", file=sys.stderr)
        return 2
    expected_formats = tuple(
        value.strip().lower() for value in args.expected_formats.split(",") if value.strip()
    )
    if not expected_formats:
        print("--expected-formats must name at least one format", file=sys.stderr)
        return 2
    if args.samples < 1 or args.columns < 1 or args.thumb_width < 80 or args.max_inline_bytes < 1:
        print(
            "samples/columns/inline bytes must be positive and thumb width must be at least 80",
            file=sys.stderr,
        )
        return 2
    try:
        report, findings = inspect(
            args.media_root,
            args.out,
            only=args.only,
            expected_formats=expected_formats,
            samples=args.samples,
            columns=args.columns,
            thumb_width=args.thumb_width,
            require_timelines=args.strict,
            max_inline_bytes=args.max_inline_bytes,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"demo inspection failed: {exc}", file=sys.stderr)
        return 2
    print(f"inspected {len(report['media'])} files; open {(args.out / 'index.html').resolve()}")
    errors = [finding for finding in findings if finding.severity == "error"]
    for finding in findings:
        print(f"[{finding.severity}] {finding.scenario}: {finding.message}")
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    sys.exit(main())
