"""Profile ONLY the cached-scrub rebuild (session-rebirth) cost.

Pass 1 scrubs a range cold (populates payload caches), settles, then pass 2
revisits the same indices with cProfile enabled per-step. Reports per-pass
step timings and the warm-pass hot spots.
"""
import sys, cProfile, pstats, io
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

INDICES = [20 + k * 16 for k in range(10)]
prof = cProfile.Profile()
state = {"pass": 0, "i": 0, "p1": [], "p2": [], "checks": 0}
timer = QtCore.QTimer(); timer.setInterval(100)


def enable_fft():
    axis = win.renderer._montage_session.montage_axis if win.renderer._montage_session else 2
    win.operation_coordinator.load_operations((CenteredFFT(axis=int(axis)),))
    win._set_document(win.operation_coordinator.document)
    win._coerce_channel_for_current_dtype()
    win.render(reason="probe-fft")

def scrub_step():
    if state["i"] >= len(INDICES):
        timer.stop()
        state["checks"] = 0
        QtCore.QTimer.singleShot(300, win, wait_settle)
        return
    vs = win.view_state
    axis = win.renderer._montage_session.montage_axis if win.renderer._montage_session else 2
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
                "_last_montage_stage_plan_ms", "_last_montage_session_setup_ms",
                "_last_montage_initial_commit_ms", "_last_montage_tile_payload_build_ms"))
              + f" retargets={getattr(r, '_montage_session_retargets', 0)}"
              + f" rejects={dict(getattr(r, '_montage_session_retarget_rejects', {}) or {})}", flush=True)

def wait_settle():
    state["checks"] += 1
    s = win.renderer._montage_session
    busy = s is None or getattr(s, "stage_planning_deferred", False) or s.loading_tiles or s.pending_tiles or s.lifecycle.evaluating_tiles
    if busy and state["checks"] <= 120:
        QtCore.QTimer.singleShot(500, win, wait_settle)
        return
    if state["pass"] == 1:
        state["pass"] = 2; state["i"] = 0
        QtCore.QTimer.singleShot(300, win, timer.start)
    else:
        report("SETTLED" if not busy else "TIMEOUT")

def start_pass1():
    state["pass"] = 1
    timer.start()

def report(status):
    def stat(a):
        b = sorted(a)
        return f"mean={sum(a)/max(1,len(a)):.1f} p50={b[len(b)//2]:.1f} worst={max(a or [0]):.1f}ms"
    print(f"RESULT {status}")
    print(f"pass1 (cold, n={len(state['p1'])}): {stat(state['p1'])}")
    print(f"pass2 (warm, n={len(state['p2'])}): {stat(state['p2'])}")
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(30)
    print("\n".join(l for l in s.getvalue().splitlines() if l.strip())[:4500])
    app.quit()

timer.timeout.connect(scrub_step)
QtCore.QTimer.singleShot(8000, win, enable_fft)
QtCore.QTimer.singleShot(22000, win, start_pass1)
QtCore.QTimer.singleShot(180000, win, app.quit)  # anti-hang
app.exec()
