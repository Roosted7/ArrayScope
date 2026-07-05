"""Scrub fast-path probe: real-slider semantics (interaction noted per step).

Measures per-step synchronous cost + 10 ms heartbeat gaps during a scrub
burst, then verifies the deferred stage planning completes after the burst
(exact tiles arrive, no stall repairs, no orphan wedges).
"""
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, "/home/thomas/projects/ArrayScope-lod-test")
from arrayscope.app.qt_binding import prefer_pyside6
prefer_pyside6()
from arrayscope.io.file_interpreters import load_path
from arrayscope.app.launch import _create_window
from arrayscope.operations.pipeline import CenteredFFT
from pyqtgraph.Qt import QtCore

fp = Path("/home/thomas/projects/ArrayScope-lod-test/data/_WIPDelRec-tT2_20260223150234_14.nii")
loaded = load_path(fp)
app, win = _create_window(loaded.data, title=fp.name, filepath=fp, axes=getattr(loaded, "axes", None))

state = {
    "n": 0, "steps": [], "hb_last": None, "gaps": [], "phase": "warmup",
    "deferred_at_burst_end": None, "settle_checks": 0,
}

hb = QtCore.QTimer(); hb.setInterval(10)

def heartbeat():
    now = perf_counter()
    if state["hb_last"] is not None:
        state["gaps"].append((state["phase"], (now - state["hb_last"]) * 1000.0))
    state["hb_last"] = now
hb.timeout.connect(heartbeat)
hb.start()

def enable_fft():
    axis = win.renderer._montage_session.montage_axis if win.renderer._montage_session else 2
    win.operation_coordinator.load_operations((CenteredFFT(axis=int(axis)),))
    win._set_document(win.operation_coordinator.document)
    win._coerce_channel_for_current_dtype()
    win.render(reason="probe-fft")

def scrub_step():
    state["phase"] = "burst"
    vs = win.view_state
    axis = win.renderer._montage_session.montage_axis if win.renderer._montage_session else 2
    i = 20 + (state["n"] * 16) % 160
    state["n"] += 1
    if state["n"] <= 3:
        prev = win.renderer._montage_session
        print(f"STEP{state['n']} prev: committed={getattr(prev, 'display_committed', None)} "
              f"axis={getattr(prev, 'montage_axis', None)} interaction={win._viewport_interaction_active} "
              f"policy={win.renderer._montage_lod_policy_mode()}", flush=True)
    t0 = perf_counter()
    # Real slider path: _apply_slice_state notes interaction before render.
    win._note_viewport_interaction("dimension-scrub")
    win._set_view_state(vs.with_montage_axis(int(axis), text=f"{i}:{i+64}"))
    win.render(reason="probe-scrub")
    state["steps"].append((perf_counter() - t0) * 1000.0)

def burst_ended():
    state["phase"] = "settle"
    s = win.renderer._montage_session
    state["deferred_at_burst_end"] = bool(getattr(s, "stage_planning_deferred", False)) if s else None

def check_settled():
    state["settle_checks"] += 1
    s = win.renderer._montage_session
    if s is None:
        return report("NO SESSION")
    deferred = bool(getattr(s, "stage_planning_deferred", False))
    loading = len(s.loading_tiles)
    pending = len(s.pending_tiles)
    evaluating = len(s.lifecycle.evaluating_tiles)
    if not deferred and not loading and not pending and not evaluating:
        return report("SETTLED")
    if state["settle_checks"] > 60:
        return report(f"TIMEOUT deferred={deferred} loading={loading} pending={pending} evaluating={evaluating}")
    QtCore.QTimer.singleShot(500, win, check_settled)

def report(status):
    steps = state["steps"]
    burst_gaps = sorted(g for p, g in state["gaps"] if p == "burst")
    def pct(a, q):
        return a[min(len(a) - 1, int(q * len(a)))] if a else -1
    print(f"RESULT {status}")
    print(f"steps={len(steps)} mean={sum(steps)/max(1,len(steps)):.1f}ms worst={max(steps or [0]):.1f}ms")
    print(f"burst heartbeat: p50={pct(burst_gaps,0.5):.1f} p95={pct(burst_gaps,0.95):.1f} max={max(burst_gaps or [0]):.1f}ms")
    print(f"deferred_at_burst_end={state['deferred_at_burst_end']}")
    print(f"stall_repairs={getattr(win.renderer, '_montage_stall_repairs', 0)}")
    print(f"plans_deferred={getattr(win.renderer, '_montage_stage_plans_deferred', 0)}")
    s = win.renderer._montage_session
    if s is not None:
        audits = s.lifecycle.audit_counters() if hasattr(s.lifecycle, "audit_counters") else {}
        print(f"session={s.session_id} rendered={len(s.rendered_tiles)} audits={audits}")
    app.quit()

QtCore.QTimer.singleShot(8000, win, enable_fft)
t = QtCore.QTimer(); t.setInterval(100); t.timeout.connect(scrub_step)
QtCore.QTimer.singleShot(22000, win, t.start)


def stop_burst():
    t.stop()
    burst_ended()
    QtCore.QTimer.singleShot(500, win, check_settled)


QtCore.QTimer.singleShot(28000, win, stop_burst)
app.exec()
