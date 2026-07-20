"""Reproduce field report 2026-07-05: stale wrong-LOD tiles + floor misses.

Scenario: FFT->shift->iFFT pipeline on the real dataset, d2 index window
4:104, window scrubs and back. Detect:
  A) STUCK tiles: presented preview/mismatched level at settle with NO
     lifecycle task, target, commit, or page-claim progress attached.
  B) FLOOR MISSES: planned tiles presenting NOTHING while a resident
     floor level exists (should present instantly).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from arrayscope.app.qt_binding import prefer_pyside6
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_S
from arrayscope.tools.presentation_settlement import (
    presentation_is_settled,
    presentation_settlement_diagnostic,
    presentation_target_token,
)

prefer_pyside6()

from pyqtgraph.Qt import QtCore

from arrayscope.app.launch import _create_window
from arrayscope.io.file_interpreters import load_path
from arrayscope.operations.pipeline import CenteredFFT, CenteredIFFT, FFTShift
from arrayscope.render import lod as render_lod

fp = REPO_ROOT / "data" / "_WIPDelRec-tT2_20260223150234_14.nii"
loaded = load_path(fp)
app, win = _create_window(
    loaded.data, title=fp.name, filepath=fp, axes=getattr(loaded, "axes", None)
)

from time import perf_counter

floor_misses = {"samples": 0, "misses": 0, "worst": 0, "detail": {}}
miss_started: dict[tuple[int, int], float] = {}  # (session_id, tile) -> t0
miss_durations: list[float] = []
correctness_failures: list[str] = []


def session():
    return win._frame_session


def physical_rows() -> dict[int, dict[str, object]]:
    getter = getattr(win.img_view, "tileTruthPhysicalRows", None)
    if not callable(getter):
        return {}
    return {int(tile): dict(row) for tile, row in dict(getter() or {}).items()}


def sample_floor_misses():
    s = session()
    if s is None or not render_lod.resident_lod_active(s):
        return
    required = set(s.required_tile_numbers()) - set(s.skipped_tiles)
    planned = {
        int(t.montage_index): t for t in tuple(s.plan.tiles) if int(t.montage_index) in required
    }
    drawn = physical_rows()
    now = perf_counter()
    sid = int(s.session_id)
    misses = 0
    missing_now = set()
    for number, tile in planned.items():
        if number in drawn:
            continue
        best = render_lod.best_floor_key(
            s,
            int(tile.source_index),
            tile_number=int(number),
        )
        if best is not None:
            misses += 1
            missing_now.add((sid, number))
            miss_started.setdefault((sid, number), now)
            floor_misses["detail"][number] = (
                f"active_req={number in s.active_tile_requests} "
                f"loading={number in s.loading_tiles} actual_level={best[1]} "
                f"requested_level={best[0].level}"
            )
    for key in tuple(miss_started):
        if key not in missing_now:
            miss_durations.append(now - miss_started.pop(key))  # noqa: PERF401
    floor_misses["samples"] += 1
    floor_misses["misses"] += misses
    floor_misses["worst"] = max(floor_misses["worst"], misses)


def stuck_scan(label):
    s = session()
    if s is None:
        print(f"[{label}] no session", flush=True)
        return ("no-session",)
    demand = s.lod_policy_decision.demand
    desired = int(demand.desired_level)
    physical = physical_rows()
    semantic_rows = {
        int(row["tile"]): dict(row)
        for row in s.diagnostic_tile_identity_rows(
            limit=max(1, len(s.required_tile_numbers())),
            include_all_visible=True,
        )
    }
    pending_rung_tiles = {
        int(request.tile_number) for request in tuple(s.pending_rung_materializations)
    }
    stuck = []
    for number in s.required_tile_numbers():
        number = int(number)
        if number in s.skipped_tiles:
            continue
        row = {**semantic_rows.get(number, {}), **physical.get(number, {})}
        quality = str(row.get("desired_payload_quality", "") or "none")
        requested_level = row.get("desired_payload_lod")
        bindings = tuple(row.get("physical_page_bindings", ()) or ())
        actual_levels = tuple(
            int(getattr(binding.get("actual_lod"), "level", 0) or 0) for binding in bindings
        )
        if not actual_levels and row.get("physical_lod_level") is not None:
            actual_levels = (int(row["physical_lod_level"]),)
        has_progress = (
            bool(row.get("loading"))
            or bool(row.get("active"))
            or bool(row.get("target_unsettled"))
            or bool(row.get("dirty"))
            or bool(row.get("pending_upsert"))
            or row.get("evaluation_claim_source_index") is not None
            or bool(row.get("preview_claims"))
            or number in pending_rung_tiles
        )
        problem = None
        if number not in physical and not has_progress:
            problem = "physically-blank-no-progress"
        elif row.get("desired_payload_source_index") is not None and not bool(
            row.get("desired_matches_current_source")
        ):
            problem = "desired-payload-wrong-source"
        elif (
            row.get("desired_payload_source_index") is not None
            and not bool(row.get("backend_matches_desired"))
            and not has_progress
        ):
            problem = "backend-identity-stale"
        elif actual_levels and max(actual_levels) > desired and not has_progress:
            problem = f"physical-too-coarse({max(actual_levels)}>{desired})"
        elif (
            quality in {"preview", "fallback"}
            and actual_levels
            and max(actual_levels) >= desired
            and not has_progress
        ):
            problem = "fallback-stuck"
        if problem:
            stuck.append(
                f"  tile {number}: {problem} q={quality} requested={requested_level} "
                f"actual={actual_levels or None} state={row.get('presentation_state', '')}"
            )
    required = set(s.required_tile_numbers()) - set(s.skipped_tiles)
    physically_blank = sorted(required - set(physical))
    if physically_blank:
        print(
            f"[{label}] PHYSICALLY BLANK: {len(physically_blank)} required tiles: "
            f"{physically_blank[:15]}",
            flush=True,
        )
    counters = s.lifecycle.counters()
    presented_level, presented_factor, presented_factor_xy = s.presented_lod_summary()
    page_cache = s.lod_page_cache
    print(
        f"[{label}] visible={len(tuple(s.visible_tiles))} loaded={len(s.rendered_tiles)} "
        f"presented={len(s.lifecycle.presented_tiles)} loading={len(s.loading_tiles)} "
        f"target_unsettled={s.required_target_unsettled_tiles()} desired={desired} "
        f"rungs={len(s.pending_rung_materializations)} "
        f"page_claims={0 if page_cache is None else int(page_cache.pending_count)} "
        f"physical_lod={presented_level}/{presented_factor}/{presented_factor_xy} "
        f"draw_pending={win.img_view.presentationDrawPending()} lifecycle={counters}",
        flush=True,
    )
    if stuck:
        print(f"[{label}] STUCK {len(stuck)} tiles:", flush=True)
        for line in stuck[:20]:
            print(line, flush=True)
    else:
        print(f"[{label}] no stuck tiles", flush=True)
    return tuple(stuck)


def set_ops():
    win.operation_coordinator.load_operations(
        (CenteredFFT(axis=2), FFTShift(axis=2), CenteredIFFT(axis=2))
    )
    win._set_document(win.operation_coordinator.document)
    win._coerce_channel_for_current_dtype()
    win.render(reason="verify-stale-ops")


def set_window(text):
    start_text, stop_text = str(text).split(":", maxsplit=1)
    start, stop = int(start_text), int(stop_text)

    def apply():
        vs = win.view_state
        win._set_view_state(vs.with_montage_axis(2, indices=range(start, stop), text=text))
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


floor_timer = QtCore.QTimer(win)
floor_timer.setInterval(120)
floor_timer.timeout.connect(sample_floor_misses)
floor_timer.start()


def settled() -> bool:
    current = presentation_target_token(win)
    previous = machine.get("previous_target")
    if machine.get("require_advance") and (current is None or current == previous):
        return False
    return presentation_is_settled(win, expected_target=current)


# Event-driven phase machine (no fixed waits): each phase applies its action
# and advances THE MOMENT the session settles (100 ms poll); the per-phase
# deadline is only the anti-hang fallback.
PHASES = [
    ("open", lambda: None),
    ("after-ops-settle", set_ops),
    ("after-4:104", set_window("4:104")),
    ("after-100:200", set_window("100:200")),
    ("after-scrub-back-4:104", set_window("4:104")),
    ("zoomed-in", zoom(0.18)),
    ("zoomed-out", zoom(1.0 / 0.18)),
    ("zoom-0.4", zoom(0.4)),
    ("after-50:150", set_window("50:150")),
    ("zoom-back", zoom(1.0 / 0.4)),
]
machine = {
    "phase": -1,
    "deadline": 0.0,
    "grace": 0.0,
    "previous_target": None,
    "require_advance": False,
}


def advance():
    now = perf_counter()
    if machine["phase"] >= 0:
        name, _action = PHASES[machine["phase"]]
        if now < machine["grace"]:
            return
        if not settled():
            if now < machine["deadline"]:
                return
            print(
                f"[{name}] HARD FAIL: did not settle within {INTERACTION_SETTLE_HARD_LIMIT_S:.0f}s",
                flush=True,
            )
            print(presentation_settlement_diagnostic(win), flush=True)
            correctness_failures.extend(stuck_scan(name))
            floor_timer.stop()
            driver.stop()
            app.exit(1)
            return
        correctness_failures.extend(stuck_scan(name))
    machine["phase"] += 1
    if machine["phase"] >= len(PHASES):
        driver.stop()
        finish()
        return
    _name, action = PHASES[machine["phase"]]
    machine["previous_target"] = presentation_target_token(win)
    action()
    machine["require_advance"] = machine["phase"] > 0
    now = perf_counter()
    # Short grace so the action's own render lands before settle sampling.
    machine["grace"] = now + 0.5
    machine["deadline"] = now + INTERACTION_SETTLE_HARD_LIMIT_S


driver = QtCore.QTimer(win)
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
    stall_assertions = int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0)
    print(
        "DISPATCH: stall_assertions="
        f"{stall_assertions} "
        f"admission_declined={int(getattr(win.renderer, '_montage_tile_admission_declined', 0) or 0)} "
        f"last_stall={getattr(win.renderer, '_montage_watchdog_last_stall', None)}",
        flush=True,
    )
    for number, info in list(floor_misses["detail"].items())[:15]:
        print(f"  tile {number}: {info}", flush=True)
    failed = bool(
        correctness_failures or floor_misses["misses"] or miss_started or stall_assertions
    )
    if failed:
        print(
            "RESULT FAIL: "
            f"stuck={len(correctness_failures)} "
            f"floor_misses={floor_misses['misses']} "
            f"active_floor_misses={len(miss_started)} "
            f"stall_assertions={stall_assertions}",
            flush=True,
        )
    else:
        print("RESULT PASS", flush=True)
    app.exit(1 if failed else 0)


driver.start()
raise SystemExit(app.exec())
