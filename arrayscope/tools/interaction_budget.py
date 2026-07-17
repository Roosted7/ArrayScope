"""Repository-wide user-visible interaction settlement budgets for gates.

These values are acceptance limits for tests and diagnostic tools, not
application scheduler tuning knobs.  A caller may ask for a shorter deadline,
but it may not make a slow interaction pass by requesting a longer one.
"""

from __future__ import annotations

import math


INTERACTION_SETTLE_TARGET_S = 2.0
INTERACTION_SETTLE_HARD_LIMIT_S = 5.0
INTERACTION_SETTLE_HARD_LIMIT_MS = int(INTERACTION_SETTLE_HARD_LIMIT_S * 1000.0)


def bounded_interaction_settle_timeout_s(requested_s: float | None = None) -> float:
    """Return a positive interaction deadline capped at the hard limit."""

    if requested_s is None:
        return INTERACTION_SETTLE_HARD_LIMIT_S
    requested = float(requested_s)
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("interaction settlement timeout must be positive and finite")
    return min(requested, INTERACTION_SETTLE_HARD_LIMIT_S)


def interaction_settle_timeout_ms(requested_s: float | None = None) -> int:
    """Qt millisecond form of :func:`bounded_interaction_settle_timeout_s`."""

    return int(round(bounded_interaction_settle_timeout_s(requested_s) * 1000.0))
