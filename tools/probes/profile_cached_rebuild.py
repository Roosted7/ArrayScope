"""Profile ONLY the cached-scrub rebuild (session-rebirth) cost.

Pass 1 scrubs a range cold (populates payload caches), settles, then pass 2
revisits the same indices with cProfile enabled per-step. Reports per-pass
step timings and the warm-pass hot spots.
"""
import cProfile
import io
import pstats
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

from arrayscope.io.file_interpreters import load_path
from arrayscope.app.launch import _create_window
from arrayscope.operations.pipeline import CenteredFFT
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_S
from arrayscope.tools.presentation_settlement import (
    presentation_is_settled,
    presentation_settlement_diagnostic,
    presentation_target_token,
)
from pyqtgraph.Qt import QtCore

fp = REPO_ROOT / "data" / "_WIPDelRec-tT2_20260223150234_14.nii"
loaded = load_path(fp)
app, win = _create_window(loaded.data, title=fp.name, filepath=fp, axes=getattr(loaded, "axes", None))

INDICES = [20 + k * 16 for k in range(10)]
prof = cProfile.Profile()
state = {
    "pass": 0,
    "i": 0,
    "p1": [],
    "p2": [],
    "settle_deadline": None,
    "startup_phase": "open",
    "startup_deadline": perf_counter() + INTERACTION_SETTLE_HARD_LIMIT_S,
    "startup_previous_target": None,
    "reported": False,
}
timer = QtCore.QTimer(win)
timer.setInterval(100)


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
    if state["i"] >= len(INDICES):
        timer.stop()
        state["settle_deadline"] = perf_counter() + INTERACTION_SETTLE_HARD_LIMIT_S
        QtCore.QTimer.singleShot(100, win, wait_settle)
        return
    vs = win.view_state
    current = win._frame_session
    axis = current.montage_axis if current is not None else 2
    i = INDICES[state["i"]]; state["i"] += 1
    warm = state["pass"] == 2
    t0 = perf_counter()
    if warm: prof.enable()
    win._note_viewport_interaction("dimension-scrub")
    win._set_view_state(vs.with_montage_axis(
        int(axis), indices=range(i, i + 64), text=f"{i}:{i+64}"))
    win.render(reason="probe-scrub")
    if warm: prof.disable()
    (state["p2"] if warm else state["p1"]).append((perf_counter() - t0) * 1000.0)
    if warm:
        r = win.renderer
        print("PHASES plan=%.1f cache=%.1f stage=%.1f setup=%.1f commit=%.1f payload_build=%.1f" % tuple(
            float(getattr(r, k, 0) or 0) for k in (
                "_last_montage_viewport_plan_ms", "_last_montage_cache_resolve_ms",
                "_last_montage_stage_plan_ms", "_last_frame_session_setup_ms",
                "_last_montage_initial_commit_ms", "_last_montage_tile_payload_build_ms"))
              + f" retargets={getattr(r, '_frame_session_retargets', 0)}"
              + f" rejects={dict(getattr(r, '_frame_session_retarget_rejects', {}) or {})}", flush=True)

def wait_settle():
    if not frame_settled():
        if perf_counter() >= float(state["settle_deadline"]):
            report(f"TIMEOUT {frame_progress()}", success=False)
            return
        QtCore.QTimer.singleShot(100, win, wait_settle)
        return
    if state["pass"] == 1:
        state["pass"] = 2
        state["i"] = 0
        QtCore.QTimer.singleShot(100, win, timer.start)
    else:
        report("SETTLED", success=True)

def start_pass1():
    state["pass"] = 1
    timer.start()

def report(status, *, success):
    if state["reported"]:
        return
    state["reported"] = True
    def stat(a):
        if not a:
            return "mean=n/a p50=n/a worst=n/a"
        b = sorted(a)
        return f"mean={sum(a)/len(a):.1f} p50={b[len(b)//2]:.1f} worst={max(a):.1f}ms"
    print(f"RESULT {status}")
    print(f"pass1 (cold, n={len(state['p1'])}): {stat(state['p1'])}")
    print(f"pass2 (warm, n={len(state['p2'])}): {stat(state['p2'])}")
    over_hard_limit = tuple(
        elapsed_ms
        for elapsed_ms in (*state["p1"], *state["p2"])
        if elapsed_ms > INTERACTION_SETTLE_HARD_LIMIT_S * 1000.0
    )
    print(f"synchronous_step_hard_failures={len(over_hard_limit)}")
    if prof.getstats():
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(30)
        print("\n".join(l for l in s.getvalue().splitlines() if l.strip())[:4500])
    else:
        print("profile: no warm-pass samples collected")
    stall_assertions = int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0)
    print(f"stall_assertions={stall_assertions}")
    app.exit(
        0
        if success and stall_assertions == 0 and not over_hard_limit
        else 1
    )


def advance_startup():
    target = presentation_target_token(win)
    target_advanced = bool(
        state["startup_previous_target"] is None
        or target != state["startup_previous_target"]
    )
    if target_advanced and frame_settled():
        if state["startup_phase"] == "open":
            state["startup_phase"] = "fft"
            state["startup_previous_target"] = target
            enable_fft()
            state["startup_deadline"] = perf_counter() + INTERACTION_SETTLE_HARD_LIMIT_S
            return
        if state["startup_phase"] == "fft":
            startup_timer.stop()
            start_pass1()
            return
    if perf_counter() >= float(state["startup_deadline"]):
        report(
            f"STARTUP TIMEOUT phase={state['startup_phase']} {frame_progress()}",
            success=False,
        )

timer.timeout.connect(scrub_step)
startup_timer = QtCore.QTimer(win)
startup_timer.setInterval(100)
startup_timer.timeout.connect(advance_startup)
startup_timer.start()
raise SystemExit(app.exec())
