"""File lifecycle helpers for diagnostics JSONL logging."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import platform as platform_module
import sys

from arrayscope.core.diagnostics_jsonl import (
    diagnostics_jsonl_line,
    diagnostics_snapshot_record,
    diagnostics_start_record,
)


def default_diagnostics_log_path() -> Path:
    return Path.cwd() / f"arrayscope-diagnostics-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"


def normalize_diagnostics_log_path(path) -> Path:
    normalized = Path(path)
    if str(normalized).lower().endswith(".jsonl"):
        return normalized
    return Path(str(normalized) + ".jsonl")


class DiagnosticsJsonlLogger:
    def __init__(self, path):
        self.path = normalize_diagnostics_log_path(path)
        self._file = None
        self._sequence = 0

    @property
    def active(self) -> bool:
        return self._file is not None

    @property
    def sequence(self) -> int:
        return int(self._sequence)

    def start(self, snapshot, *, app_version: str, interval_ms: int) -> None:
        try:
            self._file = self.path.open("a", encoding="utf-8")
            self._sequence = 0
            self._file.write(
                diagnostics_jsonl_line(
                    diagnostics_start_record(
                        snapshot,
                        recorded_at=_recorded_at(),
                        app_version=app_version,
                        cwd=str(Path.cwd()),
                        pid=os.getpid(),
                        python_version=sys.version,
                        platform=platform_module.platform(),
                        interval_ms=int(interval_ms),
                    )
                )
            )
            self.write_snapshot(snapshot)
        except Exception:
            self.close()
            raise

    def write_snapshot(self, snapshot) -> None:
        if self._file is None:
            return
        self._sequence += 1
        self._file.write(
            diagnostics_jsonl_line(
                diagnostics_snapshot_record(
                    snapshot,
                    sequence=self._sequence,
                    recorded_at=_recorded_at(),
                )
            )
        )
        self._file.flush()

    def close(self) -> None:
        log_file = self._file
        self._file = None
        self._sequence = 0
        if log_file is not None:
            log_file.close()


def _recorded_at() -> str:
    return datetime.now(timezone.utc).isoformat()
