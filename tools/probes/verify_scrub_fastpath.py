"""Scrub fast-path probe: real-slider semantics (interaction noted per step).

Measures per-step synchronous cost + 10 ms heartbeat gaps during a scrub
burst, then verifies the deferred stage planning completes after the burst
(exact tiles arrive, no stall repairs, no orphan wedges).
"""

import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

from pyqtgraph.Qt import QtCore

from arrayscope.app.launch import _create_window
from arrayscope.io.file_interpreters import load_path
from arrayscope.operations.pipeline import CenteredFFT
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_S
from arrayscope.tools.presentation_settlement import (
    presentation_is_settled,
    presentation_settlement_diagnostic,
    presentation_target_token,
)

fp = REPO_ROOT / "data" / "_WIPDelRec-tT2_20260223150234_14.nii"
loaded = load_path(fp)
app, win = _create_window(
    loaded.data, title=fp.name, filepath=fp, axes=getattr(loaded, "axes", None)
)

BURST_STEPS = 60
state = {
    "n": 0,
    "steps": [],
    "hb_last": None,
    "gaps": [],
    "phase": "warmup",
    "deferred_at_burst_end": None,
    "settle_deadline": None,
    "reported": False,
    "startup_previous_target": None,
}

hb = QtCore.QTimer()
hb.setInterval(10)


def heartbeat():
    now = perf_counter()
    if state["hb_last"] is not None:
        state["gaps"].append((state["phase"], (now - state["hb_last"]) * 1000.0))
    state["hb_last"] = now


hb.timeout.connect(heartbeat)
hb.start()


def enable_fft():
    current = win._frame_session
    axis = current.montage_axis if current is not None else 2
    win.operation_coordinator.load_operations((CenteredFFT(axis=int(axis)),))
    win._set_document(win.operation_coordinator.document)
    win._coerce_channel_for_current_dtype()
    win.render(reason="probe-fft")


def frame_settled() -> bool:
    return presentation_is_settled(win)


def frame_progress() -> str:
    return presentation_settlement_diagnostic(win)


def scrub_step():
    state["phase"] = "burst"
    vs = win.view_state
    current = win._frame_session
    axis = current.montage_axis if current is not None else 2
    i = 20 + (state["n"] * 16) % 160
    state["n"] += 1
    if state["n"] <= 3:
        prev = win._frame_session
        print(
            f"STEP{state['n']} prev: committed={getattr(prev, 'display_committed', None)} "
            f"axis={getattr(prev, 'montage_axis', None)} interaction={win._viewport_interaction_active} "
            f"policy={win.renderer._montage_quality_policy_mode()}",
            flush=True,
        )
    t0 = perf_counter()
    # Real slider path: _apply_slice_state notes interaction before render.
    win._note_viewport_interaction("dimension-scrub")
    win._set_view_state(
        vs.with_montage_axis(
            int(axis),
            indices=tuple(range(i, i + 64)),
            text=f"{i}:{i + 64}",
        )
    )
    win.render(reason="probe-scrub")
    state["steps"].append((perf_counter() - t0) * 1000.0)
    if state["n"] >= BURST_STEPS:
        burst_timer.stop()
        burst_ended()
        QtCore.QTimer.singleShot(100, win, check_settled)


def burst_ended():
    state["phase"] = "settle"
    s = win._frame_session
    state["deferred_at_burst_end"] = (
        bool(getattr(s, "stage_planning_deferred", False)) if s else None
    )
    state["settle_deadline"] = perf_counter() + INTERACTION_SETTLE_HARD_LIMIT_S


def check_settled():
    if frame_settled():
        return report("SETTLED", success=True)
    if perf_counter() >= float(state["settle_deadline"]):
        return report(f"TIMEOUT {frame_progress()}", success=False)
    QtCore.QTimer.singleShot(100, win, check_settled)


def report(status, *, success):
    if state["reported"]:
        return
    state["reported"] = True
    steps = state["steps"]
    burst_gaps = sorted(g for p, g in state["gaps"] if p == "burst")

    def pct(a, q):
        return a[min(len(a) - 1, int(q * len(a)))] if a else -1

    print(f"RESULT {status}")
    print(
        f"steps={len(steps)} mean={sum(steps) / max(1, len(steps)):.1f}ms worst={max(steps or [0]):.1f}ms"
    )
    print(
        f"burst heartbeat: p50={pct(burst_gaps, 0.5):.1f} p95={pct(burst_gaps, 0.95):.1f} max={max(burst_gaps or [0]):.1f}ms"
    )
    print(f"deferred_at_burst_end={state['deferred_at_burst_end']}")
    over_hard_limit = tuple(
        elapsed_ms for elapsed_ms in steps if elapsed_ms > INTERACTION_SETTLE_HARD_LIMIT_S * 1000.0
    )
    print(f"synchronous_step_hard_failures={len(over_hard_limit)}")
    stall_assertions = int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0)
    print(f"stall_assertions={stall_assertions}")
    print(f"plans_deferred={getattr(win.renderer, '_montage_stage_plans_deferred', 0)}")
    s = win._frame_session
    if s is not None:
        audits = s.lifecycle.audit_counters() if hasattr(s.lifecycle, "audit_counters") else {}
        print(f"session={s.session_id} rendered={len(s.rendered_tiles)} audits={audits}")
    app.exit(0 if success and stall_assertions == 0 and not over_hard_limit else 1)


burst_timer = QtCore.QTimer(win)
burst_timer.setInterval(100)
burst_timer.timeout.connect(scrub_step)


startup_timer = QtCore.QTimer(win)
startup_timer.setInterval(100)
state["startup_deadline"] = perf_counter() + INTERACTION_SETTLE_HARD_LIMIT_S


def advance_startup():
    target = presentation_target_token(win)
    target_advanced = bool(
        state["startup_previous_target"] is None or target != state["startup_previous_target"]
    )
    if target_advanced and frame_settled():
        if state["phase"] == "warmup":
            state["phase"] = "fft"
            state["startup_previous_target"] = target
            enable_fft()
            state["startup_deadline"] = perf_counter() + INTERACTION_SETTLE_HARD_LIMIT_S
            return
        if state["phase"] == "fft":
            startup_timer.stop()
            state["phase"] = "burst"
            burst_timer.start()
            return
    if perf_counter() >= float(state["startup_deadline"]):
        report(
            f"STARTUP TIMEOUT phase={state['phase']} {frame_progress()}",
            success=False,
        )


startup_timer.timeout.connect(advance_startup)
startup_timer.start()
raise SystemExit(app.exec())
