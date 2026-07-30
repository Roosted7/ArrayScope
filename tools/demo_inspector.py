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
        inline_scale=min(1.0, README_INLINE_WIDTH / int(video["width"])),
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


def parity_findings(
    grouped: dict[str, list[MediaFacts]], expected_formats: tuple[str, ...]
) -> list[Finding]:
    findings: list[Finding] = []
    for scenario, items in sorted(grouped.items()):
        by_format = {item.format: item for item in items}
        missing = sorted(set(expected_formats) - set(by_format))
        if missing:
            findings.append(
                Finding("error", scenario, f"missing expected formats: {', '.join(missing)}")
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
        gif = by_format.get("gif")
        if gif and gif.width < README_INLINE_WIDTH:
            findings.append(
                Finding(
                    "note",
                    scenario,
                    f"GIF is enlarged {README_INLINE_WIDTH / gif.width:.2f}x at README width",
                )
            )
    return findings


def _write_html(
    output: Path,
    grouped: dict[str, list[MediaFacts]],
    storyboards: dict[str, str],
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
        sections.append(f"<section><h2>{html.escape(scenario)}</h2>{''.join(cards)}</section>")
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
    findings = parity_findings(grouped, expected_formats)
    storyboards: dict[str, str] = {}
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
    report = {
        "media_root": str(media_root.resolve()),
        "expected_formats": list(expected_formats),
        "media": [asdict(item) for item in facts],
        "findings": [asdict(item) for item in findings],
        "storyboards": {
            str(Path(path).resolve()): str((output / name).resolve())
            for path, name in storyboards.items()
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_html(output / "index.html", grouped, storyboards, findings)
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
        "--strict",
        action="store_true",
        help="return non-zero for structural errors such as missing or mismatched companions",
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
    if args.samples < 1 or args.columns < 1 or args.thumb_width < 80:
        print(
            "samples/columns must be positive and thumb width must be at least 80", file=sys.stderr
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
