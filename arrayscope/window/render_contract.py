"""One render staleness/ordering vocabulary (roadmap Y1).

``RenderOrchestrator`` owns the state (the generation counter and the current
montage session); this module owns the predicates and token derivations.
Orchestration code must not reimplement a staleness comparison locally: it
either calls one of these predicates or captures one of these tokens.

The contract is anchored on ``(document_key, semantic_key, render_generation)``:

- ``RenderGeneration`` orders visible output. A deferred callback captured
  under an older generation must not commit visible state.
- A montage session binds one semantic key (document + view + viewport +
  colormap) to one monotonic ``session_id`` and the generation it was started
  under. ``session_is_current`` / ``session_token_is_current`` decide whether a
  session object (or a captured ``(session_id, key)`` pair) still owns the
  surface.
- ``montage_work_token`` derives per-kind work tokens from the session identity
  plus the session-local revisions relevant to that kind of work
  (payload/level/viewport), so a deferred callback can detect that its
  scheduled work was superseded without inventing another counter.

Intentionally *not* covered here: semantic-match checks that compare a session
key against a key re-derived from the live view state. Those are equality
checks between two freshly computed identities, not staleness of a captured
token.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RenderGeneration:
    """Visible-output render generation guard."""

    current: int = 0
    last_reason: str = ""

    def advance(self, reason: str = "") -> int:
        self.current += 1
        self.last_reason = str(reason or "")
        return self.current

    def capture(self) -> int:
        return int(self.current)

    def is_current(self, generation: int) -> bool:
        return int(generation) == int(self.current)


def generation_is_current(guard: RenderGeneration | None, generation: int) -> bool:
    """A missing guard admits everything (unit-test stubs without a window)."""

    return guard is None or guard.is_current(generation)


def session_token_is_current(current_session, session_id, key) -> bool:
    """Is a captured ``(session_id, key)`` pair still the current session?"""

    if current_session is None:
        return False
    return int(current_session.session_id) == int(session_id) and current_session.key == key


def session_is_current(current_session, session) -> bool:
    """Is a captured session object still the current session?"""

    if session is None:
        return False
    return session_token_is_current(current_session, session.session_id, session.key)


def montage_work_token(session, reason: str) -> tuple[object, ...]:
    """Derive the supersession token for one kind of deferred montage work.

    Every token carries the session identity and its render generation; kinds
    whose validity also depends on a session-local revision fold that revision
    in, so a rescheduled timer never applies work planned against older state.
    """

    base = (
        str(reason),
        int(getattr(session, "session_id", 0) or 0),
        getattr(session, "key", None),
        int(getattr(session, "render_generation", 0) or 0),
    )
    if reason == "commit":
        return (
            *base,
            int(getattr(session, "payload_revision", 0) or 0),
            int(getattr(session, "level_revision", 0) or 0),
        )
    if reason in ("priority_retarget", "viewport_update"):
        return (*base, int(getattr(session, "viewport_revision", 0) or 0))
    return base


def montage_work_token_is_current(session, token, reason: str) -> bool:
    """Is a captured work token still the latest for this session and kind?

    ``None`` means "no token was captured" and is treated as current so call
    sites can keep scheduling before their first token exists.
    """

    return token is None or token == montage_work_token(session, reason)


__all__ = [
    "RenderGeneration",
    "generation_is_current",
    "montage_work_token",
    "montage_work_token_is_current",
    "session_is_current",
    "session_token_is_current",
]
