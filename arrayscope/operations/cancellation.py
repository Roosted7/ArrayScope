"""Cancellation primitives shared by Qt and pure operation evaluation."""

from __future__ import annotations

from inspect import signature


class EvaluationCancelled(Exception):
    """Raised at cooperative cancellation points between bounded chunks."""


# Marks a *runtime-bound* callable as accepting ``cancellation_token=``.
#
# The token has to reach ``run_array_command``'s poll loop -- it is the only
# thing that can interrupt an external child -- without ever reaching the
# operation author's own ``fn(array) -> array``. A plugin spec's ``build()``
# returns the callable that wraps the runtime, so that wrapper (and only it)
# opts in by carrying this attribute. Introspection would be the obvious
# alternative and is wrong here: a third-party callable declaring ``**kwargs``
# would look willing and then fail, and one bound with ``functools.partial``
# would look unwilling and silently lose cancellation.
CANCELLABLE_ATTRIBUTE = "arrayscope_accepts_cancellation_token"


def cancellable(function):
    """Declare that ``function`` accepts a ``cancellation_token`` keyword."""

    setattr(function, CANCELLABLE_ATTRIBUTE, True)
    return function


def call_with_cancellation(function, data, cancellation_token=None):
    """Call ``function(data)``, passing the token only if it declared support."""

    if cancellation_token is not None and getattr(function, CANCELLABLE_ATTRIBUTE, False):
        return function(data, cancellation_token=cancellation_token)
    return function(data)


def apply_with_evaluation_context(operation, data, evaluation_context=None):
    """Apply one operation, handing over the evaluation context if it takes one.

    Built-in operations take ``apply(data)`` and always will; only the runtime
    -backed ones have anything to do with a token. The signature is inspected
    rather than probed with a ``TypeError`` retry, which would swallow a real
    ``TypeError`` raised from inside a working ``apply``.
    """

    if evaluation_context is None or not _accepts_evaluation_context(operation.apply):
        return operation.apply(data)
    return operation.apply(data, evaluation_context=evaluation_context)


def _accepts_evaluation_context(apply) -> bool:
    try:
        signature(apply).bind(None, evaluation_context=None)
    except (TypeError, ValueError):
        return False
    return True
