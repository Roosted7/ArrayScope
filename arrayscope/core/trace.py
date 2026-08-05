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
# Opt-in batching for a file-backed trace.  Batching turns hundreds of
# lifecycle edges per montage commit into a handful of writes, but it also
# makes the file lag the process and loses the unflushed tail if that process
# dies -- and the tail is the part worth having after a crash or hang.  The
# in-process ring does not substitute: it cannot be recovered post mortem.
#
# So this is NOT the default.  The only file-sink caller is the ``--trace``
# CLI flag, whose entire purpose is watching a live run; the profiler passes
# ``path=None`` and uses the ring, so it never paid the write cost this would
# save.  Callers that genuinely want throughput over durability ask for it.
LIVE_SINK_BUFFER_BYTES = 64 * 1024


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
        buffered: bool = False,
    ) -> None:
        self.close()
        handle = None
        if path is not None:
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            handle = output.open(
                "a" if append else "w",
                encoding="utf-8",
                buffering=LIVE_SINK_BUFFER_BYTES if buffered else 1,
            )
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
        self.emit_fields(kind, fields)

    def emit_fields(self, kind: str, fields: dict[str, object]) -> None:
        """Record one event from a field mapping the caller already holds.

        ``emit`` has to materialize a kwargs dict anyway, so it routes here.
        The point of the separate entry is the hot emitters that *build* a
        mapping — the per-tile lifecycle edge assembles one, and splatting it
        back apart made the same fields be re-hashed into a fresh dict twice
        more on the way to the ring.  ``fields`` is consumed, never retained:
        the event below is a fresh dict, so a caller may reuse or mutate its
        mapping afterwards.

        That fresh dict is a **copy plus four inserts**, not a
        ``{header, **fields}`` display.  The display re-hashed every field key
        on the way into a dict that already held four entries; ``dict.copy``
        duplicates the hash table wholesale and hashes nothing, so the merge
        cost stops scaling with the field count.  Measured on the two hot
        shapes with the ring armed: 790 → 703 ns for the 7-field
        ``rung_prepare`` record and 1165 → 952 ns for the 15-field
        ``lifecycle`` one.  The trade is that the four header keys are written
        *after* the caller's fields, so an event can no longer shadow its own
        identity by naming ``kind``/``ts_ns``/``sequence``/``schema_version``
        as a field — the header wins now.  No emit site does that, and a row
        that lied about its own kind was never the useful outcome.  Key
        *order* changes with it; nothing depends on it, because `_encode`
        sorts keys and every consumer reads by name.

        Keys must be strings.  Keyword emission guarantees that by
        construction, and there is deliberately no coercion pass here: it ran
        over every field of every event of every montage session (the stall
        watchdog arms a ring-only bus in normal production) to defend against
        a caller that cannot exist.  A key that is somehow not a string
        degrades at `_encode`, which is where this module already puts
        pathological fields.
        """

        if not self._enabled:
            return
        ts_ns = perf_counter_ns()
        event = fields.copy()
        event["schema_version"] = TRACE_SCHEMA_VERSION
        event["ts_ns"] = ts_ns
        event["kind"] = str(kind)
        with self._lock:
            if not self._enabled:
                return
            sequence = self._sequence + 1
            self._sequence = sequence
            event["sequence"] = sequence
            # The ring keeps the event; only a real sink pays for encoding it.
            self._ring.append(event)
            handle = self._handle
            if handle is not None:
                handle.write(_encode(event) + "\n")

    def snapshot(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(dict(event) for event in self._ring)

    @staticmethod
    def _write_events(
        path: str | Path,
        events,
        *,
        append: bool,
    ) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a" if append else "w", encoding="utf-8") as handle:
            for event in events:
                handle.write(_encode(event) + "\n")
        return output

    def dump(self, path: str | Path, *, append: bool = False) -> Path:
        return self._write_events(path, self.snapshot(), append=append)

    def drain(self, path: str | Path, *, append: bool = False) -> Path:
        """Write and clear the current ring without resetting event sequence."""

        with self._lock:
            events = tuple(dict(event) for event in self._ring)
            self._ring.clear()
        return self._write_events(path, events, append=append)


TRACE = TraceBus()


def configure_trace(
    path: str | Path | None = None,
    *,
    ring_events: int = DEFAULT_RING_EVENTS,
    append: bool = False,
    buffered: bool = False,
) -> None:
    TRACE.configure(path, ring_events=ring_events, append=append, buffered=buffered)


def close_trace() -> None:
    TRACE.close()


def trace_enabled() -> bool:
    return TRACE.enabled


def emit_trace(kind: str, **fields) -> None:
    """Emit by keyword — the door 84 of the app's 86 emit sites come through.

    It routes to ``emit_fields``, not to ``TraceBus.emit``.  ``**fields`` has
    already cost one dict; re-splatting it into ``emit(kind, **fields)`` built
    a *second* one with every key hashed again, for a mapping that was about
    to be handed straight back to ``emit_fields`` anyway.  Two dicts and three
    calls became one dict and two, with byte-identical output: measured
    1316 → 703 ns per keyword emit with the ring armed (the state the montage
    watchdog leaves every production session in), and 715 → 268 ns with it
    closed.
    """

    TRACE.emit_fields(kind, fields)


def emit_trace_fields(kind: str, fields: dict[str, object]) -> None:
    """``emit_trace`` for a caller that already assembled its fields."""

    TRACE.emit_fields(kind, fields)


__all__ = [
    "DEFAULT_RING_EVENTS",
    "TRACE",
    "TRACE_SCHEMA_VERSION",
    "TraceBus",
    "close_trace",
    "configure_trace",
    "emit_trace",
    "emit_trace_fields",
    "trace_enabled",
]
