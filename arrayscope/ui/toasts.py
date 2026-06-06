"""User-facing status message helpers."""

from __future__ import annotations


def show_status_message(window, message, timeout=4000):
    if hasattr(window, "statusBar"):
        window.statusBar().showMessage(str(message), int(timeout))
