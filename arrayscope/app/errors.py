"""Shared GUI exception handling."""

from __future__ import annotations

import logging
import os
import traceback


def strict_ui_enabled() -> bool:
    return os.environ.get("ARRAYSCOPE_STRICT_UI", "").strip().lower() in {"1", "true", "yes", "on"}


def handle_ui_exception(context: str, exc: BaseException) -> None:
    logging.exception("%s failed", context, exc_info=exc)
    if strict_ui_enabled():
        raise exc


def traceback_text(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

