"""Shared GUI callback work budgets.

GUI-thread callbacks need item, byte, and elapsed limits.  This module keeps
that bookkeeping Qt-free so renderers, controllers, and tests can share one
definition of "make progress, then yield".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


INTERACTIVE_TARGET_MS = 4.0
IDLE_TARGET_MS = 8.0
WARNING_THRESHOLD_MS = 16.0
INTERACTIVE_BYTE_CAP = 8 * 1024 * 1024
IDLE_BYTE_CAP = 32 * 1024 * 1024


@dataclass(frozen=True)
class GuiCallbackBudgetDecision:
    channel: str
    batch_limit: int
    budget_ms: float
    interval_ms: int
    reason: str
    byte_cap: int = 0
    control_budget_ms: float = 0.0
    model: str = ""
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuiCallbackObservation:
    channel: str
    work_class: str
    backend: str
    target_ms: float
    warning_ms: float
    item_cap: int
    byte_cap: int
    elapsed_ms: float
    processed_items: int
    processed_bytes: int
    details: tuple[str, ...] = ()

    @property
    def over_warning(self) -> bool:
        return float(self.elapsed_ms) >= float(self.warning_ms)


@dataclass
class GuiCallbackBudget:
    channel: str
    work_class: str = ""
    backend: str = ""
    target_ms: float = IDLE_TARGET_MS
    warning_ms: float = WARNING_THRESHOLD_MS
    item_cap: int = 1
    byte_cap: int = IDLE_BYTE_CAP
    _started: float = field(default_factory=perf_counter, init=False, repr=False)
    processed_items: int = 0
    processed_bytes: int = 0

    def __post_init__(self) -> None:
        self.channel = str(self.channel)
        self.work_class = str(self.work_class or "")
        self.backend = str(self.backend or "")
        self.target_ms = max(0.0, float(self.target_ms))
        self.warning_ms = max(0.0, float(self.warning_ms))
        self.item_cap = max(1, int(self.item_cap))
        self.byte_cap = max(0, int(self.byte_cap))

    @classmethod
    def for_decision(
        cls,
        channel: str,
        decision,
        *,
        interactive: bool = False,
        work_class: str = "",
        backend: str = "",
        item_cap: int | None = None,
        byte_cap: int | None = None,
    ) -> "GuiCallbackBudget":
        return cls(
            channel=channel,
            work_class=work_class,
            backend=backend,
            target_ms=(
                INTERACTIVE_TARGET_MS
                if decision is None and interactive
                else IDLE_TARGET_MS
                if decision is None
                else float(getattr(decision, "budget_ms", IDLE_TARGET_MS))
            ),
            item_cap=(
                max(1, int(getattr(decision, "batch_limit", 1)))
                if item_cap is None
                else max(1, int(item_cap))
            ),
            byte_cap=(
                max(0, int(getattr(decision, "byte_cap", 0) or 0))
                if decision is not None and getattr(decision, "byte_cap", 0)
                else INTERACTIVE_BYTE_CAP
                if interactive
                else IDLE_BYTE_CAP
            )
            if byte_cap is None
            else max(0, int(byte_cap)),
        )

    @property
    def elapsed_ms(self) -> float:
        return max(0.0, (perf_counter() - self._started) * 1000.0)

    def record_item(self, *, byte_count: int = 0, item_count: int = 1) -> None:
        self.processed_items += max(0, int(item_count))
        self.processed_bytes += max(0, int(byte_count))

    def should_yield(self) -> bool:
        if self.processed_items <= 0:
            return False
        if self.processed_items >= self.item_cap:
            return True
        if self.byte_cap > 0 and self.processed_bytes >= self.byte_cap:
            return True
        return self.elapsed_ms >= self.target_ms

    def observation(self) -> GuiCallbackObservation:
        return GuiCallbackObservation(
            channel=self.channel,
            work_class=self.work_class,
            backend=self.backend,
            target_ms=float(self.target_ms),
            warning_ms=float(self.warning_ms),
            item_cap=int(self.item_cap),
            byte_cap=int(self.byte_cap),
            elapsed_ms=self.elapsed_ms,
            processed_items=int(self.processed_items),
            processed_bytes=int(self.processed_bytes),
            details=(),
        )


def should_yield_after_item(
    budget: GuiCallbackBudget,
    *,
    byte_count: int = 0,
    item_count: int = 1,
) -> bool:
    budget.record_item(byte_count=byte_count, item_count=item_count)
    return budget.should_yield()


def default_gui_callback_budget_decision(channel: str, *, interactive: bool = False) -> GuiCallbackBudgetDecision:
    """Return the static GUI drain-budget vocabulary outside governor knobs."""

    budget_ms = INTERACTIVE_TARGET_MS if interactive else IDLE_TARGET_MS
    return GuiCallbackBudgetDecision(
        channel=str(channel),
        batch_limit=1,
        budget_ms=budget_ms,
        interval_ms=0,
        reason="static GUI callback budget",
        byte_cap=INTERACTIVE_BYTE_CAP if interactive else IDLE_BYTE_CAP,
        control_budget_ms=budget_ms,
        model="static",
    )
