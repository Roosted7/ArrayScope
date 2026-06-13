"""Singleflight coordination for expensive operation-stage materialization."""

from __future__ import annotations

from dataclasses import dataclass

from arrayscope.core.memory_budget import format_bytes
from arrayscope.operations.regions import StageCacheCandidate, StageKey, region_text
from arrayscope.operations.slabs import stage_key_for_candidate


@dataclass(frozen=True)
class StageMaterializationRequest:
    key: StageKey
    candidate: StageCacheCandidate
    document_key: object


@dataclass(frozen=True)
class StageMaterializationResult:
    decision: str
    key: StageKey
    value: object | None = None
    request: StageMaterializationRequest | None = None
    reason: str = ""

    @property
    def should_schedule(self) -> bool:
        return self.decision == "scheduled"


@dataclass(frozen=True)
class StageMaterializationDiagnostics:
    decision: str = ""
    candidate_bytes: int | None = None
    budget_bytes: int = 0
    consequence: str = ""
    key_summary: str = ""
    in_flight: int = 0
    scheduled: int = 0
    attached: int = 0
    completed: int = 0
    cancelled: int = 0
    failed: int = 0
    refused: int = 0


class StageMaterializationManager:
    def __init__(self, stage_cache):
        self.stage_cache = stage_cache
        self._in_flight: dict[StageKey, StageMaterializationRequest] = {}
        self._scheduled = 0
        self._attached = 0
        self._completed = 0
        self._cancelled = 0
        self._failed = 0
        self._refused = 0
        self._last = StageMaterializationDiagnostics(budget_bytes=int(getattr(stage_cache, "max_bytes", 0)))

    def key_for_candidate(self, document_key, candidate: StageCacheCandidate) -> StageKey:
        return stage_key_for_candidate(document_key, candidate)

    def request_stage(self, document_key, candidate: StageCacheCandidate) -> StageMaterializationResult:
        key = self.key_for_candidate(document_key, candidate)
        value = self.stage_cache.get_containing(key) if hasattr(self.stage_cache, "get_containing") else self.stage_cache.get(key)
        if value is not None:
            self._record("hit", candidate, key, "stage already cached")
            return StageMaterializationResult("hit", key, value=value)
        estimated = candidate.estimated_nbytes
        budget = int(getattr(self.stage_cache, "max_bytes", 0))
        if estimated is not None and int(estimated) > budget:
            self._refused += 1
            if hasattr(self.stage_cache, "note_refused"):
                self.stage_cache.note_refused(self._candidate_text(candidate))
            self._record("refused", candidate, key, "each tile may recompute FFT")
            return StageMaterializationResult("refused", key, reason="over budget")
        if key in self._in_flight:
            self._attached += 1
            self._record("in-flight", candidate, key, "tiles wait for shared stage")
            return StageMaterializationResult("attached", key, request=self._in_flight[key])
        request = StageMaterializationRequest(key=key, candidate=candidate, document_key=document_key)
        self._in_flight[key] = request
        self._scheduled += 1
        self._record("scheduled", candidate, key, "tiles wait for shared stage")
        return StageMaterializationResult("scheduled", key, request=request)

    def complete(self, key: StageKey, value) -> None:
        del value
        self._in_flight.pop(key, None)
        self._completed += 1
        self._record_key("completed", key, "shared stage ready")

    def cancel(self, key: StageKey) -> None:
        self._in_flight.pop(key, None)
        self._cancelled += 1
        self._record_key("cancelled", key, "stale stage ignored")

    def fail(self, key: StageKey, exc) -> None:
        self._in_flight.pop(key, None)
        self._failed += 1
        self._record_key("failed", key, str(exc))

    def invalidate_document(self, document_key) -> None:
        for key in tuple(self._in_flight):
            if key.document_key == document_key:
                self.cancel(key)

    def clear(self) -> None:
        for key in tuple(self._in_flight):
            self.cancel(key)

    def diagnostics(self) -> StageMaterializationDiagnostics:
        return StageMaterializationDiagnostics(
            decision=self._last.decision,
            candidate_bytes=self._last.candidate_bytes,
            budget_bytes=int(getattr(self.stage_cache, "max_bytes", 0)),
            consequence=self._last.consequence,
            key_summary=self._last.key_summary,
            in_flight=len(self._in_flight),
            scheduled=int(self._scheduled),
            attached=int(self._attached),
            completed=int(self._completed),
            cancelled=int(self._cancelled),
            failed=int(self._failed),
            refused=int(self._refused),
        )

    def _record(self, decision: str, candidate: StageCacheCandidate, key: StageKey, consequence: str) -> None:
        self._last = StageMaterializationDiagnostics(
            decision=str(decision),
            candidate_bytes=None if candidate.estimated_nbytes is None else int(candidate.estimated_nbytes),
            budget_bytes=int(getattr(self.stage_cache, "max_bytes", 0)),
            consequence=str(consequence),
            key_summary=self._key_text(key),
            in_flight=len(self._in_flight),
            scheduled=int(self._scheduled),
            attached=int(self._attached),
            completed=int(self._completed),
            cancelled=int(self._cancelled),
            failed=int(self._failed),
            refused=int(self._refused),
        )

    def _record_key(self, decision: str, key: StageKey, consequence: str) -> None:
        self._last = StageMaterializationDiagnostics(
            decision=str(decision),
            budget_bytes=int(getattr(self.stage_cache, "max_bytes", 0)),
            consequence=str(consequence),
            key_summary=self._key_text(key),
            in_flight=len(self._in_flight),
            scheduled=int(self._scheduled),
            attached=int(self._attached),
            completed=int(self._completed),
            cancelled=int(self._cancelled),
            failed=int(self._failed),
            refused=int(self._refused),
        )

    def _candidate_text(self, candidate: StageCacheCandidate) -> str:
        nbytes = "unknown" if candidate.estimated_nbytes is None else format_bytes(int(candidate.estimated_nbytes))
        return f"stage={candidate.stage_index}, region={region_text(candidate.region)}, bytes={nbytes}"

    def _key_text(self, key: StageKey) -> str:
        return f"stage={len(tuple(key.operation_prefix))}, region={region_text(key.region)}, dtype={key.dtype}, shape={tuple(key.shape)}"
