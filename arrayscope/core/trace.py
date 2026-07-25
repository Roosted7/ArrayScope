"""Bounded structured event tracing for rendering correctness and latency."""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from time import perf_counter_ns

TRACE_SCHEMA_VERSION = 1
# The ring is bounded by event count, not encoded bytes.  The ring has always
# stored raw event dicts and re-encoded them in `dump`, so an emit-time encode
# bought nothing but the byte accounting — and it ran on every event of every
# montage session, because the stall watchdog arms a ring-only bus in normal
# production.  A count is exact, free, and holds the dump within the same
# envelope the byte budget actually held.  A live 100-tile montage scrub emits
# ~1 KiB/event (622 events and 607 KiB per scroll step, measured), so 8192
# events is the ~8 MiB the byte bound was buying there — about 13 scroll steps
# of history either way.  Leaner event mixes now keep *more* history for the
# same or less memory, which is the direction a flight recorder should err in.
# The approximation this accepts is that an unusually fat event no longer
# evicts more of its neighbours than a lean one, so the dump size varies with
# the workload; emit sites cap their own collection fields (see the `stall`
# event's `tile_rows`).
DEFAULT_RING_EVENTS = 8192


def _encode(event: dict[str, object]) -> str:
    """Canonical one-line JSON for one event, never raising.

    Encoding now happens where the bytes are consumed — the JSONL sink and
    `dump` — so a pathological field (a cycle, an exotic key) would surface at
    dump time, i.e. exactly when the trace is the evidence someone needs.  A
    row that cannot encode degrades to its identity plus a repr rather than
    taking the whole dump down with it.
    """

    try:
        return json.dumps(event, sort_keys=True, separators=(",", ":"), default=repr)
    except (TypeError, ValueError) as error:
        return json.dumps(
            {
                "schema_version": TRACE_SCHEMA_VERSION,
                "sequence": event.get("sequence"),
                "ts_ns": event.get("ts_ns"),
                "kind": event.get("kind"),
                "trace_encode_error": repr(error),
                "trace_event_repr": repr(event),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=repr,
        )


class TraceBus:
    """One process-wide flat-event stream with a bounded in-memory tail."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled = False
        self._sequence = 0
        self._ring: deque[dict[str, object]] = deque(maxlen=0)
        self._handle = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def configure(
        self,
        path: str | Path | None = None,
        *,
        ring_events: int = DEFAULT_RING_EVENTS,
        append: bool = False,
    ) -> None:
        self.close()
        handle = None
        if path is not None:
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            handle = output.open("a" if append else "w", encoding="utf-8", buffering=1)
        limit = max(0, int(ring_events))
        with self._lock:
            self._sequence = 0
            self._ring = deque(maxlen=limit)
            self._handle = handle
            self._enabled = bool(handle is not None or limit > 0)

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
            # The ring keeps the event; only a real sink pays for encoding it.
            self._ring.append(event)
            if self._handle is not None:
                self._handle.write(_encode(event) + "\n")

    def snapshot(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(dict(event) for event in self._ring)

    def dump(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        events = self.snapshot()
        with output.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(_encode(event) + "\n")
        return output


TRACE = TraceBus()


def configure_trace(
    path: str | Path | None = None,
    *,
    ring_events: int = DEFAULT_RING_EVENTS,
    append: bool = False,
) -> None:
    TRACE.configure(path, ring_events=ring_events, append=append)


def close_trace() -> None:
    TRACE.close()


def trace_enabled() -> bool:
    return TRACE.enabled


def emit_trace(kind: str, **fields) -> None:
    TRACE.emit(kind, **fields)


__all__ = [
    "DEFAULT_RING_EVENTS",
    "TRACE",
    "TRACE_SCHEMA_VERSION",
    "TraceBus",
    "close_trace",
    "configure_trace",
    "emit_trace",
    "trace_enabled",
]
