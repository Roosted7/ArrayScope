"""Cancellation primitives shared by Qt and pure operation evaluation."""

from __future__ import annotations


class EvaluationCancelled(Exception):
    """Raised at cooperative cancellation points between bounded chunks."""

