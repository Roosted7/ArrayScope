# wgpu present decoupling — is drawing allowed while a fill is in progress? (2026-07-31)

**Question (Thomas's standard):** on WGPU the app should be 60 Hz smooth,
always. The architecture already retains everything a camera-only frame
needs (pages resident, `UpdateTileInstances` tuple-identity short-circuit,
camera latched at draw time), so the question is not "is there enough data
to draw" but **whether drawing is allowed to happen while other work is in
progress, and whether the present is paced to the display at all**.

**Verdict in one paragraph.** The draw is decoupled in *data* and in
*scheduling design*, on both present paths — but two things stood between a
camera move and a display-rate present. (1) **The default present path
(bitmap) was hard-capped at 30 fps** by rendercanvas's unpassed `max_fps`
default: on the shipped default configuration a pure pan could never exceed
30 fps on any display, before any commit or fill is even considered. Fixed
this session (`fix(display): pace the bitmap canvas to the display`), 30.5
→ 59.4 draws/s measured under real Wayland. (2) **Draw execution shares
the GUI thread with commits and is non-preemptible**: draws interleave
*between* GUI callbacks but never within one, so a slow tiled commit costs
`ceil(commit_ms / period)` frames every time it runs — ~5–6 frames per
90 ms whole-montage apply. Nothing re-draws mid-commit and nothing can;
this is exactly the event-loop-occupancy verdict of
`wgpu-frame-pacing-2026-07-21.md` phase 2, now confirmed with a controlled
occupancy experiment. Shrinking (or slicing) the commit is the only lever
for fill-time smoothness; the pacer, the swapchain, and the GPU are not
the constraint.

## 1. The mechanism, established from the code

One draw pipeline per present method, both driven by
`_request_wgpu_canvas_draw` → `canvas.request_draw()`:

- **screen** (`WgpuScreenCanvas`, live Wayland only): a coalesced
  `QTimer.singleShot` paced to the display's own refresh rate (cached,
  invalidated by `screenChanged`/`refreshRateChanged`), Mailbox present
  mode preferred (non-blocking acquire, ~0.13–0.18 ms), the present edge
  is `context.present()` inside the draw callback (~0.1 ms p50). Camera
  is latched when the paced draw *runs*, not when it is requested (pinned
  by `test_camera_is_latched_when_the_paced_draw_runs_not_when_it_is_requested`).
- **bitmap** (the default when this was written, and still what every
  non-Wayland session resolves to; rendercanvas `QRenderWidget`,
  `ondemand`): draw
  requests go to rendercanvas's scheduler, which caps requested draws at
  `max_fps` — **which defaulted to 30 because the constructor never passed
  it**. The draw presents through a GPU→CPU readback (~4–7 ms/frame,
  gate-B measurement) and Qt's backing store.

Commits (`FramePipelineEffects.request_presentation` →
`_on_presentation_gate` → `commit_pending_session` →
`setTiledPresentation` → `_apply_backend_tiled_presentation`) are **one
bounded transaction per event-loop turn, posted at LowEventPriority**, so
queued input/paint/timer events run between commits — the interleaving is
by design and it works. But a Qt callback is atomic: no timer can fire
inside a running commit, there is no `processEvents` on the commit path
(verified), and the commit itself never draws inline — it ends by
*requesting* a paced draw (`_finish_wgpu_tiled_presentation`). So:

- A commit can never land *inside* a draw, and a draw can never run
  *inside* a commit — same thread, atomic callbacks.
- A camera move whose paced draw comes due mid-commit slips by the
  commit's remaining duration. That slip is what
  `FrameTimingRecorder.schedule_slip_ms` measures.
- Consecutive commits do not starve the timer: the draw interleaves at
  commit boundaries (confirmed physically below — under a 90 ms
  occupancy hog, interval p50 stayed 16.8 ms; only frames due during the
  callback are lost).

## 2. Physical evidence (managed headless Weston, Intel iGPU, Mesa pin)

Conditions: load average 4.7–6.3 on 16 cores from other sessions during
all runs — absolute numbers carry that noise; the contrasts (30 vs 60,
free vs hog) are far outside it. Probe: continuous `translateBy` pans at
250 Hz against one committed 256×256 tile, 6 s per phase, draw timestamps
recorded inside the draw callback; arms interleaved and order-balanced,
one arm per process; each run records `arrayscope.__file__` (the
worktree-imports-main trap fired during setup and was closed by giving
the worktree its own `uv sync` venv and running every arm from its own
checkout cwd).

### 2a. The bitmap 30 fps cap (before/after) and the screen path when free

Medians of 3 runs per arm, draw-interval distributions from timestamps
inside the draw callback; `wt` = this branch (main + the fix only):

| arm | draws/s | interval p50 | p95 | max |
|---|---:|---:|---:|---:|
| main, bitmap, free-pan | **30.5** | 32.3 ms | 35.1 ms | 36.2 ms |
| wt, bitmap, free-pan | **59.4** | 16.8 ms | 17.6 ms | 18.0 ms |
| wt, screen, free-pan | 57.9 | 17.3 ms | 18.1 ms | 18.3 ms |

The screen arm's own present-edge recorder agrees and adds the split that
matters: interval p50/p95/max 17.3/18.2/19.0 ms, **schedule slip p50
0.6 ms / max 1.8 ms**, frame cost p50 0.96 ms, acquire p50 0.14 ms
(Mailbox). With a free event loop the screen path is display-paced with
sub-millisecond wakeup error — there is no pacer defect left to fix.

### 2b. Occupancy is the fill-time constraint (within-run contrast)

Same process, same config, a 5 Hz GUI-thread callback sleeping 90 ms (a
commit-shaped occupancy stand-in), pan stream unchanged:

| arm | draws/s | interval p50 | p95 | max |
|---|---:|---:|---:|---:|
| main, bitmap, hog | 19.9 | 33.3 ms | 104.5 ms | 113.9 ms |
| wt, bitmap, hog | 35.3 | 16.9 ms | 99.4 ms | 103.7 ms |
| wt, screen, hog | 35.2 | 17.4 ms | 97.4 ms | 103.7 ms |

Every path collapses identically: interval p50 **stays at the display
period** (draws interleave between callbacks — the LowEventPriority commit
gate design works), while p95/max jump to ≈ the callback duration + one
period. The screen recorder attributes it exactly: slip p95 80.9 ms / max
87.4 ms while frame cost stays 1.0 ms and acquire 0.14 ms. A 90 ms
GUI-thread callback eats ~5–6 frames per fire with the GPU idle the whole
time; nothing about the swapchain, present mode, or draw cost is involved.
This is the controlled-experiment form of the answer to "can a slow commit
hurt the frame at all?" — yes, by exactly its own duration, every time it
runs, and no present-path improvement can buy those frames back.

### 2c. The real fill (profiler, screen path, 3 runs)

`profile_montage_workflow --backend wgpu --wgpu-present-method screen
--stages load_data,raw_full_tiled_montage,montage_scroll_scalar,montage_zoompan_scalar`
under the same Weston recipe. Slip is the honest fill metric here — the
interval series of an on-demand canvas legitimately contains huge values
whenever nothing requests a draw (settle pauses, phase boundaries), while
slip is recorded once per *actual* draw and measures only loop lateness.

| run | phase | slip p50 | slip p95 | slip max | frame p50 | interval p50 |
|---|---|---:|---:|---:|---:|---:|
| 1 | scroll | 25.1 | 301.8 | 1084 | 2.6 | 50.9 |
| 2 | scroll | 36.7 | 440.3 | 1089 | 3.0 | 71.9 |
| 3 | scroll | 27.7 | 541.1 | 2400 | 3.6 | 60.1 |
| 1 | zoompan | 37.7 | 343.1 | 1084 | 2.5 | 81.3 |
| 2 | zoompan | 46.8 | 353.5 | 1089 | 2.9 | 89.9 |
| 3 | zoompan | 56.2 | 604.3 | 2400 | 3.6 | 109.1 |

(ms; loaded machine — spread across runs is real and reported rather than
averaged away.) The shape confirms the 2026-07-21 phase-2 verdict on
current main: during a fill the median paced wakeup is 1.5–3.5 refresh
periods late and the p95 is 20–35 periods late, while producing the frame
itself stays at ~3 ms and acquire at ~0.14 ms.

**The sharpening: the occupancy is mostly NOT the tiled-commit apply.**
The per-gesture ledgers put worst in-gesture event-loop gaps at 500–2000 ms
(`combined_zoomout_return` 843/711/1963 ms across the runs; `fast_scroll`
466–1062 ms) — and almost every one of those worst windows records **zero
tile commits** (the few nonzero ones total 15–94 ms of commit time against
a 300–700 ms gap). A ~90 ms whole-montage apply is real and costs ~5–6
frames whenever it runs (§2b generalizes to any callback), but in this
workload the frames are being eaten by *other* GUI-thread work between
commits — evaluation/admission/planning/drain activity that no current
counter attributes. The 07-21 dossier's "answer with a callback-attribution
profile" is still the open next step, and it should not assume the commit
is the main term.

Caveat: the profiler's in-process R1–R5/R7 invariant gate reports FAIL for
these stage-subset runs (it expects the full stage choreography). The
timing series above are valid observations of a real fill + interaction on
a real compositor, but the workload is not the canonical gallery pass.

## 3. What this means for the program

- The **only** remaining lever for 60 Hz-during-fill is per-callback
  occupancy: no single GUI-thread callback may exceed ~a refresh period.
  Shrinking `_apply_backend_tiled_presentation` (the sibling chip) buys
  frames one-for-one whenever a commit runs — but §2c shows it is not
  sufficient: the worst in-gesture gaps in a real fill happen in windows
  with zero commits. The next step is the callback-attribution profile the
  07-21 dossier already named, applied to the non-commit occupancy, with
  the budget stated per callback (≤ one refresh period), not per second.
  Anything that cannot get under the period must be sliced across turns
  (for the commit that is a contract-level change — the two-phase binding
  transaction is atomic by design).
- The pacer needs no further work (phase-2 verdict stands); present-mode
  and acquire costs are already negligible under Mailbox.
- **DONE (2026-07-31, Thomas's call):** `wgpu_present_method` now defaults
  to AUTO, so live-Wayland sessions get the screen path (no 4–7 ms
  readback, self-paced to the display) and bitmap — now display-rate
  capped rather than pinned at 30 — remains the everywhere-else
  resolution. Verified by building the view through the factory from a
  settings mapping that does not mention the key: under the managed
  Weston it resolves to `screen` with no fallback reason and no status
  message; under the offscreen QPA it resolves to `bitmap` with the
  reason recorded and still no message, because AUTO declining is the
  rule working rather than a fallback.

### 3a. The default is BITMAP, and all three owners agree

Worth stating because it is easy to assume otherwise: AUTO exists only as
a menu choice a user must pick. The default is BITMAP in the settings
dataclass (`settings_state.py`), in the menus' QSettings read fallback
(`menus.py`), and in the factory's `getattr` fallback
(`image_view_factory.py`) — plus `normalize_wgpu_present_method_choice`
returns BITMAP for anything unparseable. So the 30 fps cap above sat on
the **live default path**, not on a fallback nobody reaches.

### 3b. AUTO's Wayland-only scope is an evidence decision, not a wall

The enum comment says so outright ("Screen on other platforms (e.g. xcb)
would be a different, unmeasured path — AUTO widens only behind new
evidence"). Checked this session, because "unmeasured" is not "impossible":

- **wgpu-py 0.31.1 supports Xlib surfaces natively.**
  `get_surface_id_from_info` builds a `WGPUSurfaceSourceXlibWindow` for
  `platform: "x11"` — which is even the *default* platform value on
  Linux. (An `xcb` variant exists too, upstream-marked untested.)
- **PySide6 exposes `QNativeInterface.QX11Application`**, and under the
  xcb QPA `winId()` is the X11 Window XID — a documented, long-standing
  Qt contract, i.e. *more* stable than the undocumented Wayland
  `winId() == wl_surface*` behaviour the ring-4 test has to pin per Qt
  minor.
- **The recipe was run on X11 and it works.** Same shape as
  `WgpuScreenCanvas` (paint-less `QWindow` via `createWindowContainer`),
  `platform: "x11"` present-info, against a headless Weston started with
  `--xwayland` (its X server on `:2`), Intel iGPU: surface created,
  format `bgra8unorm`, present modes `immediate/mailbox/fifo/fiforelaxed`
  — **Mailbox is available exactly as on Wayland** — acquire p50 0.09 ms,
  present p50 0.02 ms over 120 frames. (Frame interval is meaningless
  there: the probe loop is unpaced.)
- **The historical blocker is plausibly already gone.** Gate B measured
  the screen path under xcb as "journey clean" but reproduced the
  overlay-stacking failure (backing store 5,416 px vs 0 on screen: Qt
  cannot composite above a native child). That was solved on Wayland by
  moving overlays *into* the wgpu frame — chip compositor, glyph atlas,
  overlay geometry — and every one of those keys on
  `_wgpu_present_method == "screen"`, not on the platform. Plausibly, not
  provably: nobody has captured X11 pixels through the current machinery.

On cost, the argument for X sessions is the same one that justified the
Wayland path: bitmap's price is a GPU→CPU readback per frame (gate B:
7.0 ms at 1300×650, 26.0 ms at 4K), and the screen path skips it on any
platform. A compositing X11 WM redirects the child window and composites
it GPU-side, which is not comparable to a CPU readback.

What a widening chip would still owe, and why this is not a one-liner:
(1) X11 pixel evidence for overlay compositing through the current
in-frame machinery; (2) native-child geometry/resize behaviour on X11 —
the Wayland subsurface pathology has an X11 analogue that has not been
characterised here; (3) a real X.org session, since the probe above is
XWayland only; (4) an X11 branch in `screen_present_unavailable_reason()`
plus an x11 present-info variant in `WgpuScreenCanvas`, and an X11
sibling for the ring-4 native-contract pin. Making AUTO the *default* on
Wayland and widening AUTO to X11 are separable steps; the first needs no
new evidence, the second does.
- `arrayscope/tools/wgpu_preview.py` still builds its own QRenderWidget
  without `max_fps` (dev tool; same 30 cap; not fixed here).

## 4. What was not verified

- **Bitmap's compositor edge.** The bitmap probe timestamps the draw
  callback (rendercanvas scheduler cadence). That cadence is an upper
  bound on presents and the 30-cap bound it — but the Qt-backing-store →
  compositor edge itself has no recorder on the bitmap path, so "59.4
  draws/s" has not been confirmed as 59.4 *compositor* frames/s. The
  screen path's recorder sits on the real `wgpuSurfacePresent` edge and
  needs no such caveat.
- **Machine load.** All runs shared the machine with other active
  sessions (load 4.7–6.3 / 16 cores). Contrasts quoted are 2× or bigger;
  absolute p95/max values carry noise.
- **Headless Weston only** — the repo's accepted real-compositor evidence
  path; no attached-session confirmation this session.
- **VRR and multi-monitor** remain unexercised (same caveat as the
  2026-07-21 dossier).
- The pan probe drives `ViewBox.translateBy` at 250 Hz, not real pointer
  input; input-delivery costs are outside what it measures (deliberately —
  the question was the present path).
- **Test-gate status.** `pytest --since` on the rebased tip surfaced
  UI-family failures that vary between identical runs (10 → 3 → 3,
  different names each time). Every named failure passed serially at
  moderate load. One
  (`test_data_change_serves_the_visible_set_before_any_out_of_view_removal[wgpu]`)
  then failed serially post-rebase — a single-variable interleaved A/B
  (only the `max_fps` argument toggled, same tip) failed in BOTH arms
  while another session drove the machine to load ~34, so the failure
  tracks machine load, not this change. A quiet-machine serial
  confirmation of that one test is still owed before merge.
