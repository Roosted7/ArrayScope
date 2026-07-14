"""Cyclic-GC pacing for the latency-sensitive Qt process."""

from __future__ import annotations

import gc


# CPython 3.14 incrementally samples the old generation according to
# threshold1. The default (10) repeatedly leads to a full old-generation
# collection during a five-second montage gesture; the live R8 trace measured
# a 38 ms stop-the-world pause. Reference counting and young-cycle collection
# remain unchanged. Old cycles are still sampled, but sufficiently gradually
# that a user input callback does not inherit a full long-lived-heap scan.
GUI_OLD_GENERATION_THRESHOLD = 1000


def configure_gui_gc_latency() -> tuple[int, int, int]:
    """Configure incremental old-generation scanning for a GUI process.

    Idempotent and deliberately process-wide: CPython's collector is global,
    so pretending this policy can belong to one window would be misleading.
    Returns the effective thresholds for diagnostics/tests.
    """

    young, old, legacy = (int(value) for value in gc.get_threshold())
    effective = (young, max(old, GUI_OLD_GENERATION_THRESHOLD), legacy)
    if effective != (young, old, legacy):
        gc.set_threshold(*effective)
    return tuple(int(value) for value in gc.get_threshold())
