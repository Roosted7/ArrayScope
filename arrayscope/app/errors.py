"""Shared GUI exception handling."""

from __future__ import annotations

import contextlib
import logging
import os
import traceback


def strict_ui_enabled() -> bool:
    return os.environ.get("ARRAYSCOPE_STRICT_UI", "").strip().lower() in {"1", "true", "yes", "on"}


def handle_ui_exception(context: str, exc: BaseException) -> None:
    logging.exception("%s failed", context, exc_info=exc)
    # Import outside the guard: a broken internal import is a real error the
    # import-health guard must see, not a silently missing trace. Only the
    # emit is allowed to fail -- this runs on the path that handles an
    # exception, so it must never raise a second one.
    from arrayscope.core.trace import emit_trace

    with contextlib.suppress(Exception):
        emit_trace(
            "ui_exception",
            context=str(context),
            error=repr(exc)[:300],
            trace_tail="".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[
                -1500:
            ],
        )
    if strict_ui_enabled():
        raise exc


def traceback_text(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
