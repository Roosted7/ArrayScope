"""Bounded structured event tracing for rendering correctness and latency."""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from time import perf_counter_ns

TRACE_SCHEMA_VERSION = 1
DEFAULT_RING_BYTES = 8 * 1024 * 1024


class TraceBus:
    """One process-wide flat-event stream with a bounded in-memory tail."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled = False
        self._sequence = 0
        self._ring_bytes_limit = 0
        self._ring_bytes = 0
        self._ring: deque[tuple[int, dict[str, object]]] = deque()
        self._handle = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def configure(
        self,
        path: str | Path | None = None,
        *,
        ring_bytes: int = DEFAULT_RING_BYTES,
        append: bool = False,
    ) -> None:
        self.close()
        handle = None
        if path is not None:
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            handle = output.open("a" if append else "w", encoding="utf-8", buffering=1)
        with self._lock:
            self._sequence = 0
            self._ring.clear()
            self._ring_bytes = 0
            self._ring_bytes_limit = max(0, int(ring_bytes))
            self._handle = handle
            self._enabled = bool(handle is not None or self._ring_bytes_limit > 0)

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            self._handle = None
            self._enabled = False
        if handle is not None:
            handle.close()

    def emit(self, kind: str, **fields) -> None:
        if not self._enabled:
            return
        event = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "sequence": 0,
            "ts_ns": perf_counter_ns(),
            "kind": str(kind),
            **{str(key): value for key, value in fields.items()},
        }
        with self._lock:
            if not self._enabled:
                return
            self._sequence += 1
            event["sequence"] = self._sequence
            line = json.dumps(event, sort_keys=True, separators=(",", ":"), default=repr)
            size = len(line.encode("utf-8")) + 1
            if self._ring_bytes_limit:
                self._ring.append((size, event))
                self._ring_bytes += size
                while self._ring and self._ring_bytes > self._ring_bytes_limit:
                    removed, _event = self._ring.popleft()
                    self._ring_bytes -= removed
            if self._handle is not None:
                self._handle.write(line + "\n")

    def snapshot(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(dict(event) for _size, event in self._ring)

    def dump(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        events = self.snapshot()
        with output.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(
                    json.dumps(event, sort_keys=True, separators=(",", ":"), default=repr) + "\n"
                )
        return output


TRACE = TraceBus()


def configure_trace(
    path: str | Path | None = None,
    *,
    ring_bytes: int = DEFAULT_RING_BYTES,
    append: bool = False,
) -> None:
    TRACE.configure(path, ring_bytes=ring_bytes, append=append)


def close_trace() -> None:
    TRACE.close()


def trace_enabled() -> bool:
    return TRACE.enabled


def emit_trace(kind: str, **fields) -> None:
    TRACE.emit(kind, **fields)


__all__ = [
    "DEFAULT_RING_BYTES",
    "TRACE",
    "TRACE_SCHEMA_VERSION",
    "TraceBus",
    "close_trace",
    "configure_trace",
    "emit_trace",
    "trace_enabled",
]
