"""Temporary import surface for R1 kernel adoption.

The implementation moved to ``arrayscope.kernel.eval_adapter``. This module
exists only until R2 ports the remaining frame-renderer callers directly to the
kernel/pipeline.
"""

from arrayscope.core.scheduler import EvalPriority
from arrayscope.kernel.eval_adapter import KernelEvaluationController as EvaluationController
from arrayscope.kernel.task import CancellationToken

__all__ = ["CancellationToken", "EvalPriority", "EvaluationController"]
