"""Reproduce field report 2026-07-05: stale wrong-LOD tiles + floor misses.

Scenario: FFT->shift->iFFT pipeline on the real dataset, d2 index window
4:104, window scrubs and back. Detect:
  A) STUCK tiles: presented preview/mismatched level at settle with NO
     progress attached (not loading/pending/dirty, no pyramid claim).
  B) FLOOR MISSES: planned tiles presenting NOTHING while a resident
     floor level exists (should present instantly).
"""

import sys

sys.path.insert(0, "/home/thomas/projects/ArrayScope-lod-test")
from pathlib import Path

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

from arrayscope.app.launch import _create_window
from arrayscope.io.file_interpreters import load_path
from arrayscope.operations.pipeline import CenteredFFT, CenteredIFFT, FFTShift
from arrayscope.window import montage_lod
from pyqtgraph.Qt import QtCore

fp = Path("/home/thomas/projects/ArrayScope-lod-test/data/_WIPDelRec-tT2_20260223150234_14.nii")
loaded = load_path(fp)
app, win = _create_window(loaded.data, title=fp.name, filepath=fp, axes=getattr(loaded, "axes", None))

from time import perf_counter

floor_misses = {"samples": 0, "misses": 0, "worst": 0, "detail": {}}
miss_started: dict[tuple[int, int], float] = {}  # (session_id, tile) -> t0
miss_durations: list[float] = []


def session():
    return win.renderer._montage_session


def sample_floor_misses():
    s = session()
    if s is None or not montage_lod.resident_lod_active(s):
        return
    planned = {
        int(t.montage_index): t
        for t in tuple(s.visible_tiles)
        if int(t.montage_index) not in s.skipped_tiles
    }
    now = perf_counter()
    sid = int(s.session_id)
    misses = 0
    missing_now = set()
    for number, tile in planned.items():
        if number in s.display_tile_payloads or number in s.rendered_tiles:
            continue
        best = montage_lod.best_floor_key(s, int(tile.source_index))
        if best is not None:
            misses += 1
            missing_now.add((sid, number))
            miss_started.setdefault((sid, number), now)
            floor_misses["detail"][number] = (
                f"active_req={number in s.active_tile_requests} "
                f"loading={number in s.loading_tiles} level={best[1]}"
            )
    for key in tuple(miss_started):
        if key not in missing_now:
            miss_durations.append(now - miss_started.pop(key))
    floor_misses["samples"] += 1
    floor_misses["misses"] += misses
    floor_misses["worst"] = max(floor_misses["worst"], misses)


def stuck_scan(label):
    s = session()
    if s is None:
        print(f"[{label}] no session", flush=True)
        return
    demand = s.lod_policy_decision.demand
    desired = int(demand.desired_level)
    pyramid = s.lod_pyramid
    stuck = []
    for tile in tuple(s.visible_tiles):
        number = int(tile.montage_index)
        if number in s.skipped_tiles:
            continue
        payload = s.display_tile_payloads.get(number)
        quality = str(getattr(payload, "quality", "none") if payload is not None else "none")
        level = int(getattr(getattr(payload, "lod", None), "level", 0) or 0) if payload is not None else -1
        has_progress = (
            number in s.loading_tiles
            or number in s.active_tile_requests
            or number in s.dirty_payloads
            or number in s.pending_payload_upserts
            or any(int(r[0]) == number for r in tuple(s.pending_lod_requests))
        )
        rendered = number in s.rendered_tiles
        problem = None
        if payload is None and not rendered and not has_progress:
            # Thomas 2026-07-05: corner tiles stayed EMPTY though they had
            # been shown in other views — blank at settle with no work.
            problem = "blank-no-progress"
        elif payload is None and rendered:
            problem = "rendered-but-unpresented"
        elif quality == "preview" and not has_progress and rendered:
            problem = "preview-stuck-with-rendered"
        elif quality == "preview" and not has_progress and not rendered:
            problem = "preview-stuck-no-eval"
        elif quality == "exact" and desired > 0 and level != desired and not has_progress and rendered:
            key = montage_lod.pyramid_key_for(s, s.rendered_tiles[number], demand=demand, level=desired)
            if pyramid is not None and pyramid.peek(key) is not None:
                problem = f"exact-wrong-level({level}!={desired},resident)"
        if problem:
            stuck.append(f"  tile {number}: {problem} q={quality} lvl={level} parked={number in s.parked_dirty_payloads}")
    # Visible-set coverage: plan tiles inside the ACTUAL view rect that the
    # session's visible set excludes (Thomas's missing-corner-tiles class),
    # and view-rect tiles that are blank with no work attached.
    from arrayscope.display.montage import montage_rect_for_viewport

    rect = montage_rect_for_viewport(s.plan, view_range=s.view_range, viewport_shape=s.viewport_shape)
    (vx0, vx1), (vy0, vy1) = s.view_range
    in_view = {
        int(t.montage_index)
        for t in s.plan.tiles
        if t.x0 < vx1 and (t.x0 + t.width) > vx0 and t.y0 < vy1 and (t.y0 + t.height) > vy0
    }
    visible_set = {int(t.montage_index) for t in tuple(s.visible_tiles)}
    uncovered = sorted(in_view - visible_set - set(s.skipped_tiles))
    blank_in_view = sorted(
        n for n in in_view
        if n not in s.display_tile_payloads
        and n not in s.skipped_tiles
    )
    if uncovered:
        print(f"[{label}] VISIBLE-SET UNDER-COVERAGE: {len(uncovered)} in-view tiles not in visible_tiles: {uncovered[:15]}", flush=True)
    if blank_in_view:
        print(f"[{label}] BLANK-IN-VIEW: {len(blank_in_view)} tiles with no payload: {blank_in_view[:15]}", flush=True)
    counters = s.lifecycle.counters()
    layer = getattr(win.img_view, "_vispy_gpu_montage_layer", None)
    stats = getattr(layer, "last_stats", None)
    layer_level = int(getattr(stats, "lod_level", -1) or 0) if stats is not None else -1
    active_scope = {int(t) for t in s._last_active_tiles}
    session_level = max(
        (
            int(getattr(getattr(p, "lod", None), "level", 0) or 0)
            for number, p in s.display_tile_payloads.items()
            if int(number) in active_scope
        ),
        default=0,
    )
    desync = "DESYNC!" if (stats is not None and layer_level != session_level) else "ok"
    print(
        f"[{label}] visible={len(tuple(s.visible_tiles))} loaded={len(s.rendered_tiles)} "
        f"presented={len(s.presented_tiles)} loading={len(s.loading_tiles)} "
        f"pending={len(s.pending_tiles)} desired={desired} "
        f"pyr_pending={getattr(s.lod_pyramid, 'pending_count', '?')} "
        f"layer_lvl={layer_level} session_lvl={session_level} [{desync}] lifecycle={counters}",
        flush=True,
    )
    if stuck:
        print(f"[{label}] STUCK {len(stuck)} tiles:", flush=True)
        for line in stuck[:20]:
            print(line, flush=True)
    else:
        print(f"[{label}] no stuck tiles", flush=True)


def set_ops():
    win.operation_coordinator.load_operations((CenteredFFT(axis=2), FFTShift(axis=2), CenteredIFFT(axis=2)))
    win._set_document(win.operation_coordinator.document)
    win._coerce_channel_for_current_dtype()
    win.render(reason="verify-stale-ops")


def set_window(text):
    def apply():
        vs = win.view_state
        win._set_view_state(vs.with_montage_axis(2, text=text))
        win.render(reason=f"verify-stale-window-{text}")

    return apply


def zoom(scale):
    """Zoom around the montage center: crosses LOD thresholds, changes
    demanded level while presenting another — the field trigger."""

    def apply():
        s = session()
        if s is None:
            return
        (x0, x1), (y0, y1) = s.view_range
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hx, hy = (x1 - x0) / 2.0 * scale, (y1 - y0) / 2.0 * scale
        win.img_view.getView().setRange(
            xRange=(cx - hx, cx + hx), yRange=(cy - hy, cy + hy), padding=0
        )

    return apply


floor_timer = QtCore.QTimer()
floor_timer.setInterval(120)
floor_timer.timeout.connect(sample_floor_misses)
floor_timer.start()

def settled() -> bool:
    s = session()
    if s is None:
        return False
    # NOTE: legacy loading_tiles retains rendered-but-unpresented (out of
    # scope) tiles by design — the machine's evaluating set is the real
    # in-flight truth.
    return (
        not s.lifecycle.evaluating_tiles
        and not s.active_tile_requests
        and not len(s.pending_tiles)
        and not s.dirty_payloads
        and not s.pending_payload_upserts
        and not s.flush_pending
        and not s.final_commit_pending
    )


# Event-driven phase machine (no fixed waits): each phase applies its action
# and advances THE MOMENT the session settles (100 ms poll); the per-phase
# deadline is only the anti-hang fallback.
PHASES = [
    ("open", lambda: None, 30.0),
    ("after-ops-settle", set_ops, 60.0),
    ("after-4:104", set_window("4:104"), 45.0),
    ("after-100:200", set_window("100:200"), 45.0),
    ("after-scrub-back-4:104", set_window("4:104"), 45.0),
    ("zoomed-in", zoom(0.18), 20.0),
    ("zoomed-out", zoom(1.0 / 0.18), 20.0),
    ("zoom+window", lambda: (zoom(0.4)(), set_window("50:150")()), 45.0),
    ("zoom-back", zoom(1.0 / 0.4), 45.0),
]
machine = {"phase": -1, "deadline": 0.0, "grace": 0.0}


def advance():
    now = perf_counter()
    if machine["phase"] >= 0:
        name, _action, deadline = PHASES[machine["phase"]]
        if now < machine["grace"]:
            return
        if not settled():
            if now < machine["deadline"]:
                return
            print(f"[{name}] DEADLINE ({deadline:.0f}s) reached without settle", flush=True)
        stuck_scan(name)
    machine["phase"] += 1
    if machine["phase"] >= len(PHASES):
        driver.stop()
        finish()
        return
    _name, action, deadline = PHASES[machine["phase"]]
    action()
    now = perf_counter()
    # Short grace so the action's own render lands before settle sampling.
    machine["grace"] = now + 0.5
    machine["deadline"] = now + float(deadline)


driver = QtCore.QTimer()
driver.setInterval(100)
driver.timeout.connect(advance)


def finish():
    durations = sorted((*miss_durations, *(perf_counter() - t0 for t0 in miss_started.values())))
    if durations:
        p50 = durations[len(durations) // 2]
        p95 = durations[max(0, int(0.95 * len(durations)) - 1)]
        print(
            f"MISS-DURATIONS: n={len(durations)} p50={p50 * 1000:.0f}ms "
            f"p95={p95 * 1000:.0f}ms max={durations[-1] * 1000:.0f}ms",
            flush=True,
        )
    print(
        f"FLOOR-MISSES: samples={floor_misses['samples']} total={floor_misses['misses']} "
        f"worst_single_sample={floor_misses['worst']}",
        flush=True,
    )
    # ADR 0051 P2 machine-derived dispatch: stall assertion fires must be 0.
    print(
        "DISPATCH: stall_assertions="
        f"{int(getattr(win.renderer, '_montage_stall_assertions', 0) or 0)} "
        f"admission_declined={int(getattr(win.renderer, '_montage_tile_admission_declined', 0) or 0)} "
        f"orphans_requeued={int(getattr(win.renderer, '_montage_orphaned_tiles_repaired', 0) or 0)} "
        f"last_stall={getattr(win.renderer, '_montage_watchdog_last_stall', None)}",
        flush=True,
    )
    for number, info in list(floor_misses["detail"].items())[:15]:
        print(f"  tile {number}: {info}", flush=True)
    app.quit()


driver.start()
app.exec()
