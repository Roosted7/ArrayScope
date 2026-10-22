#!/usr/bin/env python
"""Bound CI commands and publish their logs, JUnit failures, and timings.

Only the standard library is needed, so reporting also works after dependency
installation fails. The command's failure is preserved; reporting never turns
a failed test into a successful check.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import html
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path


def _stop_tree(process):
    notes = []
    if os.name == "nt":
        try:
            stopped = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=3,
            )
            if stopped.returncode:
                notes.append(f"taskkill exited {stopped.returncode}")
        except (OSError, subprocess.TimeoutExpired) as error:
            notes.append(f"taskkill could not finish: {error}")
            with suppress(OSError):
                process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=2)
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        notes.append("command could not be reaped within the cleanup deadline")
    return "; ".join(notes)


def _stream_log(path, finished):
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with path.open("rb") as handle:
        while True:
            block = handle.read(16384)
            if block:
                print(decoder.decode(block), end="", flush=True)
            elif finished.is_set():
                print(decoder.decode(b"", final=True), end="", flush=True)
                return
            else:
                finished.wait(0.1)


def run_command(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    junit = output / "junit.xml" if args.mode == "pytest" else None
    (output / "status.json").unlink(missing_ok=True)
    if junit is not None:
        junit.unlink(missing_ok=True)
    if junit is not None:
        overrides = []
        project = Path("pyproject.toml")
        if project.is_file():
            # Explicit -p junitxml cannot undo this repository's -p no:junitxml:
            # pytest re-registers its options and aborts before collection.
            # Override only that block, retaining the other configured defaults.
            try:
                import tomllib
            except ModuleNotFoundError:  # Python 3.10; included in the dev extra
                import tomli as tomllib
            with project.open("rb") as handle:
                defaults = (
                    tomllib.load(handle)
                    .get("tool", {})
                    .get("pytest", {})
                    .get("ini_options", {})
                    .get("addopts", "")
                )
            tokens = shlex.split(defaults)
            kept = []
            index = 0
            while index < len(tokens):
                if tokens[index : index + 2] == ["-p", "no:junitxml"]:
                    index += 2
                    continue
                if tokens[index] != "-pno:junitxml":
                    kept.append(tokens[index])
                index += 1
            if kept != tokens:
                overrides = ["-o", "addopts=" + shlex.join(kept)]
        command = [
            sys.executable,
            "-m",
            "pytest",
            *overrides,
            f"--junitxml={junit}",
            "--durations=20",
            *command,
        ]
    if not command:
        raise ValueError("a command is required after --")
    started = time.monotonic()
    record = {
        "name": args.name,
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "timeout_seconds": args.timeout,
        "timed_out": False,
        "exit_code": 1,
        "junit": str(junit) if junit is not None else None,
        "allow_empty": args.allow_empty,
    }
    log = output / "command.log"
    finished = threading.Event()
    reader = None
    process = None
    cleanup_attempted = False
    print(f"Running {args.name}; deadline {args.timeout:g}s; log: {log}", flush=True)
    try:
        with log.open("wb") as handle:
            process = subprocess.Popen(
                command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=os.name != "nt"
            )
            reader = threading.Thread(target=_stream_log, args=(log, finished), daemon=True)
            reader.start()
            try:
                record["exit_code"] = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                record["exit_code"] = 124
                cleanup_attempted = True
                record["cleanup_note"] = _stop_tree(process)
            except KeyboardInterrupt:
                record["exit_code"] = 130
                cleanup_attempted = True
                record["cleanup_note"] = _stop_tree(process)
    except OSError as error:
        record["error"] = str(error)
        print(f"Cannot run {args.name}: {error}", file=sys.stderr)
    finally:
        if process is not None and process.poll() is None and not cleanup_attempted:
            record["cleanup_note"] = _stop_tree(process)
        finished.set()
        if reader is not None:
            reader.join(timeout=3)
        record["duration_seconds"] = round(time.monotonic() - started, 3)
        record["junit_present"] = junit is not None and junit.is_file()
        if (
            junit is not None
            and record["exit_code"] == 0
            and not record["junit_present"]
            and not args.allow_empty
        ):
            record["error"] = "pytest returned success without a JUnit report"
            record["exit_code"] = 2
        (output / "status.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        f"Finished {args.name}: exit {record['exit_code']}, {record['duration_seconds']:g}s",
        flush=True,
    )
    return record["exit_code"] if record["exit_code"] >= 0 else 1


def _case_id(case):
    parts = case.get("classname", "").split(".")
    modules = [i for i, name in enumerate(parts) if name.startswith("test_")]
    if modules:
        module = modules[-1]
        return (
            "/".join(parts[: module + 1])
            + ".py::"
            + "::".join([*parts[module + 1 :], case.get("name", "")])
        )
    return "::".join(filter(None, (case.get("classname"), case.get("name"))))


def read_junit(path):
    root = ET.parse(path).getroot()
    cases = []
    for case in root.iter("testcase"):
        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")
        cases.append(
            {
                "id": _case_id(case),
                "seconds": float(case.get("time", "0")),
                "outcome": "failed"
                if problem is not None
                else "skipped"
                if case.find("skipped") is not None
                else "passed",
                "message": (
                    (problem.get("message", "") + "\n" + (problem.text or "")).strip()
                    if problem is not None
                    else ""
                ),
            }
        )
    return cases


def _inline(value):
    return html.escape(str(value)).replace("`", "'").replace("|", "\\|").replace("\n", " ")


def summarize(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    records, cases, problems = [], [], []
    for path in sorted(root.glob("*/status.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            records.append(record)
        except (OSError, ValueError) as error:
            problems.append(f"Cannot read {path}: {error}")
    for path in sorted(root.glob("*/junit.xml")):
        try:
            cases.extend(read_junit(path))
        except (OSError, ValueError, ET.ParseError) as error:
            problems.append(f"Cannot read {path}: {error}")
    failed = [case for case in cases if case["outcome"] == "failed"]
    counts = {
        name: sum(case["outcome"] == name for case in cases)
        for name in ("passed", "failed", "skipped")
    }
    lines = [f"## {_inline(args.title)}", ""]
    if args.artifact:
        if os.environ.get("GITHUB_REPOSITORY") and os.environ.get("GITHUB_RUN_ID"):
            run_url = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}#artifacts"
            artifact = f"[Download `{_inline(args.artifact)}`]({run_url})"
        else:
            artifact = f"Artifact `{_inline(args.artifact)}`"
        lines += [f"{artifact} contains full logs, JUnit XML, all timings, and command status.", ""]
    unsuccessful = sum(record["exit_code"] != 0 for record in records)
    if unsuccessful:
        lines += [
            f"**{unsuccessful} command(s) failed.** Inspect command status even when no JUnit cases were produced.",
            "",
        ]
    lines += [
        f"**{counts['passed']} passed, {counts['failed']} failed, {counts['skipped']} skipped** ({len(cases)} reported cases).",
        "",
    ]
    if not records:
        problems.append(
            "No command status was produced. Inspect setup steps or an interrupted job; no test pass is established."
        )
    if records:
        lines += ["| Command | Exit | Wall time | Report |", "| --- | ---: | ---: | --- |"]
        for record in records:
            report = (
                "JUnit"
                if record.get("junit_present")
                else "No JUnit"
                if record.get("junit")
                else "Log only"
            )
            if record.get("timed_out"):
                report += "; **deadline exceeded**"
            lines.append(
                f"| {_inline(record['name'])} | {record['exit_code']} | {record['duration_seconds']:g}s | {report} |"
            )
            if record.get("error"):
                problems.append(f"{record['name']}: {record['error']}")
            if record.get("cleanup_note"):
                problems.append(f"{record['name']}: {record['cleanup_note']}")
            if record.get("junit") and not record.get("junit_present"):
                problems.append(
                    f"{record['name']}: no JUnit cases are available; inspect command.log for collection, startup, or interruption details."
                )
        lines.append("")
    if problems:
        lines += ["### Report gaps", "", *[f"- {_inline(problem)}" for problem in problems], ""]
    if failed:
        lines += ["### Failed cases", ""]
        for case in failed[:20]:
            lines += [
                f"<details><summary>{_inline(case['id'])}</summary>",
                "",
                f"<pre>{html.escape(case['message'][:4000])}</pre>",
                "",
                "</details>",
                "",
            ]
        if len(failed) > 20:
            lines += [
                f"Showing 20 of {len(failed)} failures; the artifact contains every case.",
                "",
            ]
    if cases:
        lines += [
            "### Slowest cases",
            "",
            "Timings describe this runner and test regime; they are not hardware rendering acceptance.",
            "",
            "| Case | Time |",
            "| --- | ---: |",
        ]
        lines += [
            f"| `{_inline(case['id'])}` | {case['seconds']:g}s |"
            for case in sorted(cases, key=lambda item: item["seconds"], reverse=True)[:15]
        ]
        lines.append("")
    markdown = "\n".join(lines) + "\n"
    (root / "summary.md").write_text(markdown, encoding="utf-8")
    (root / "summary.json").write_text(
        json.dumps(
            {"counts": counts, "commands": records, "report_gaps": problems, "cases": cases},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (root / "timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "seconds", "outcome"))
        writer.writeheader()
        writer.writerows({name: case[name] for name in writer.fieldnames} for case in cases)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as handle:
            handle.write(markdown)
    print(markdown)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("run", "pytest"):
        command = subparsers.add_parser(mode)
        command.add_argument("--name", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--timeout", required=True, type=float)
        command.add_argument(
            "--allow-empty",
            action="store_true",
            help="Allow pytest's valid no-affected-tests shortcut without JUnit",
        )
        command.add_argument("command", nargs=argparse.REMAINDER)
    report = subparsers.add_parser("summarize")
    report.add_argument("root")
    report.add_argument("--title", default="CI test results")
    report.add_argument("--artifact")
    args = parser.parse_args(argv)
    if args.mode == "summarize":
        return summarize(args)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    return run_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
