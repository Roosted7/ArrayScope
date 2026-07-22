"""Tests for the shared optional-numba accelerator runtime."""

from __future__ import annotations

import threading

from arrayscope.core import numba_runtime


def test_register_is_idempotent_by_name():
    calls = []

    def builder():
        calls.append(1)
        return {"k": 42}

    g1 = numba_runtime.register("test-idem", builder)
    g2 = numba_runtime.register("test-idem", lambda: {"k": 0})  # ignored builder
    assert g1 is g2
    assert numba_runtime.get_group("test-idem") is g1


def test_prewarm_runs_builder_once_and_marks_ready():
    calls = []

    def builder():
        calls.append(1)
        return {"value": "compiled"}

    group = numba_runtime.register("test-once", builder)
    if not group.available:  # numba missing in this environment
        return
    assert group.ready() is False
    group.prewarm()
    group.prewarm()  # idempotent
    assert group.ready() is True
    assert group.kernels == {"value": "compiled"}
    assert group.get() == {"value": "compiled"}
    assert len(calls) == 1


def test_builder_failure_marks_group_unavailable_forever():
    def builder():
        raise RuntimeError("broken LLVM")

    group = numba_runtime.register("test-broken", builder)
    if not group.available:
        return
    group.prewarm()
    assert group.ready() is False
    assert group.available is False  # pinned off
    assert group.get() is None  # permanent numpy fallback


def test_get_returns_none_and_kicks_async_warm_before_ready():
    started = threading.Event()

    def builder():
        started.set()
        return {"ok": True}

    group = numba_runtime.register("test-async", builder)
    if not group.available:
        return
    # Before warm: get() returns None but triggers a background compile.
    assert group.get() is None
    started.wait(timeout=5.0)
    # The warm thread finishes; the group becomes ready and serves kernels.
    for _ in range(500):
        if group.ready():
            break
        threading.Event().wait(0.01)
    assert group.ready() is True
    assert group.get() == {"ok": True}


def test_selective_prewarm_only_warms_named_groups():
    a = numba_runtime.register("test-sel-a", lambda: {"a": 1})
    numba_runtime.register("test-sel-b", lambda: {"b": 1})
    if not a.available:
        return
    numba_runtime.prewarm("test-sel-a")  # blocking, only "a"
    assert numba_runtime.ready("test-sel-a") is True
    # "b" was not named, so it stays cold.
    assert numba_runtime.ready("test-sel-b") is False


def test_prewarm_async_unknown_name_is_a_noop():
    numba_runtime.prewarm_async("nope-not-here")  # must not raise
    assert numba_runtime.ready("nope-not-here") is False


def test_should_prewarm_predicate_gates_bulk_warm_but_not_get():
    built = []
    group = numba_runtime.register(
        "test-gated", lambda: built.append(1) or {"x": 1}, should_prewarm=lambda: False
    )
    if not group.available:
        return
    # The predicate says "not this session": a bulk-warm gate skips it.
    assert group.wanted() is False
    # ...but an on-demand caller still gets the kernel compiled and used.
    assert group.get() is None  # kicks a lazy warm regardless of the predicate
    for _ in range(500):
        if group.ready():
            break
        threading.Event().wait(0.01)
    assert group.ready() is True
    assert group.get() == {"x": 1}
    assert len(built) == 1


def test_wanted_defaults_true_and_is_fail_open_on_predicate_error():
    plain = numba_runtime.register("test-plain", dict)
    assert plain.wanted() is True

    def boom():
        raise RuntimeError("predicate exploded")

    guarded = numba_runtime.register("test-boom", dict, should_prewarm=boom)
    assert guarded.wanted() is True  # fail-open
