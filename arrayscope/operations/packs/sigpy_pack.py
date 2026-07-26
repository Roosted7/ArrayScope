"""Optional SigPy availability seam; no operations are registered.

Bundle A demoted the pack's threshold, centered-resize, circular-shift,
downsample, and upsample wrappers because each paid SigPy's import/dtype cost
for work now owned by native ArrayScope operations.  The module and its lazy
availability probe intentionally remain: later operation-platform bundles can
build genuinely SigPy-shaped multi-input operations on this seam without
reintroducing eager imports.
"""

from __future__ import annotations

import importlib.util


def sigpy_available() -> bool:
    """Whether SigPy is importable, without importing it."""

    return importlib.util.find_spec("sigpy") is not None


def pack_specs() -> tuple:
    """Return no specs; Bundle A replaced every unary SigPy wrapper natively."""

    return ()


def register(register_fn=None) -> bool:
    """Keep the pack hook stable while contributing no registered operations."""

    del register_fn
    return False
