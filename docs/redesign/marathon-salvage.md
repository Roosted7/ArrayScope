# Marathon salvage plan (2026-07-14)

Source: full audit of `git diff 906e5c3c 77520886` — the single WIP
checkpoint on branch `redesign-r8-marathon` (worktree
`.worktrees/redesign-r8-marathon`, ~9,400 insertions / 96 files). This
supersedes the archived truth-only
[r8-marathon-salvage-matrix.md](archive/r8-marathon-salvage-matrix.md),
which deliberately excluded all scheduling/throughput/benchmark work; this
plan covers exactly that.

**Caveat that governs everything below:** the checkpoint self-declares
known-red: "wrong mixed complex rendering, stale tile transitions,
throughput/heartbeat regressions, incomplete full matrix." So even
"port-as-is" means rebase + suite green + a harness scenario, never a blind
checkout. The regression epicenter is `window/frame_effects.py` and the two
backend `tiles.py` diffs — **never port those wholesale**.

**Fact worth recording:** the marathon is *ahead* of `redesign` on the
profile tool (5,639 vs 2,788 lines). The scroll/zoompan phase work from
2026-07-09 lives there, not here — `redesign` only has
`montage_scroll_scrub`.

## Tier 1 — port now (self-contained, test-backed; lands as step T1)

1. **Profile-tool R8 certification harness** — new phases
   (`montage_scroll_fft/scalar`, `montage_zoompan_fft/scalar`),
   `--stages`/`--skip-stages`, `_EventLoopProbe` (1 ms heartbeat + GC-pause
   capture), `_PresentationContinuityProbe` (real first-pixel milestones,
   no-blackout sampling), ~30 named certification gates with the hard
   constants `R8_GUI_CALLBACK_MAX_MS=50`, `R8_HEARTBEAT_MAX_GAP_MS=16`,
   `R8_WARM_INPUT_MAX_MS=15`. Files: `arrayscope/tools/profile_montage_workflow.py`,
   `tests/app/test_profile_montage_workflow.py` (+550).
2. **Session fixture + PanelSession persistence** — the checked-in 60-tile
   `tests/fixtures/profile_montage_session.json` restored through the
   production restore path; requires the `PanelSession` work in
   `view_session.py`/`file_view_session.py`. Salvage together with #1.
3. **`gui_gc.py`** — raises CPython 3.14 old-gen GC threshold (measured
   38 ms stop-the-world mid-gesture). 29 lines, tested, standalone.
4. **Read-only instrumentation** — per-drain worst-event timing + bounded
   key identity (`kernel/qt_bridge.py`), governor UI-observation epochs,
   `render_sync` ~20 sub-phase timings, `graphics_view_paint` /
   viewport-bridge observations, `callback_cpu_ms` + `relocated_tiles` in
   `TileLayerUpdateStats`. Feeds the tracing pipeline directly.
5. **`reduce_nd_axis_mean` reshape fast path** (`render/effects.py`) —
   3–5× on power-of-two tiles, exact path kept for edges.
6. **Marathon's `r8-frame-certification.md`** — the gate spec the profile
   tool implements; adopt as the performance-bar reference document.

## Tier 2 — port after V1 lands (they touch the visible-set/levels seams V1 rewrites)

7. **Narrowed prefetch-busy** (`montage_prefetch._busy` on
   `onscreen_target_settled()` + onscreen-intersection) — must read V1's
   unified visible-set owner, not grow its own predicate.
8. **Committed-frame `level_source`** (`model/commit.py`, `model/frame.py`,
   `display_presenter.py`) — fixes histogram/levels stranding across
   frames; verify against V1's evidence-deadlock fix.
9. **`setViewportContentExtent()` changed-status +
   `refreshViewportContentExtentIntent()`** (`imageview2d.py`,
   `viewport.py`) — replay AUTO/FIT intent against the acknowledged
   successor extent; fixes fit-vs-user misclassification.
   **[Codex 2026-07-14 — re-derived and landed as P3]** The focused
   AUTO/FIT/USER contract and both real-display pixel gates pass. The
   canonical benchmark fixture is USER-owned and its existing 7/60 idle
   presentation stall did not change; P3 therefore carries correctness, not
   performance, credit. Do not retry it as a stall fix.
10. **Background histogram aggregation** (`level_stats.py`,
    `montage_levels.py`) — binning off the GUI thread behind
    `(key, session, level_key, revision)` guards, plus the arithmetic
    sample-selection fix (was 50–60 ms GUI stalls).
    **[Codex 2026-07-14 — re-derived and landed as P4]** Revision/source
    snapshots now feed explicit kernel tasks; arithmetic selection is 9.8×
    faster on the pinned 60-tile workload, and real aggregate jobs up to
    36.6 ms no longer run in Qt commits. The new wake also required atomic
    transaction validity to include level revision. The independent 7/60
    presentation stall remains open; do not attribute it to aggregation.

## Tier 3 — performance program (P-steps): valuable, re-derive with fresh measurements

Ordered by confidence:

11. **Coalesced kernel completion drain** (`completions.py`, `qt_bridge.py`)
    — empty→non-empty edge notification, coalesced QEvent + timer
    continuation so Qt gets real dispatcher edges between bounded chunks.
    Closest to port-as-is in this tier; the central "don't flood Qt ahead
    of paint/input" fix.
    **[Codex 2026-07-14 — rejected as P5]** Three independently paced
    variants passed the focused bridge model but stranded all 36 exact VisPy
    V2 targets at preview quality on real Wayland. The experiment was removed.
    Do not retry completion-edge coalescing without first redesigning and
    proving the pipeline capacity/refill contract that depends on completion
    turns.
12. **LOD-plan cadence throttle + synchronous `setTitle` removal**
    (`display_presenter.py`, `viewport_bridge.py`) — 16 ms replan cadence
    during interaction; setTitle was 20–35 ms on the wheel path.
    **[Codex 2026-07-14 — re-derived and landed as P6]** The accepted cadence
    is input-only and reads committed-frame truth. It has a dedicated timer;
    programmatic replay and pipeline continuation remain immediate after three
    broader variants broke V1 or VisPy V2. Two traces cut kernel submissions
    39–50% and bridge drains 61–65% with neutral two-run first-ack midpoint;
    the separate 7/60 presentation deadlock remains open.
13. **Stage-cache lock-free resident snapshot + cancellation tokens +
    `peek_many`** (`stage_cache.py`, `evaluator.py`, `effects.py`) — GUI
    planning probes never block behind worker mutation; superseded ROI/tile
    work aborts between projections.
    **[Codex 2026-07-14 — re-derived and landed as P7]** Mutation publishes a
    lock-free resident snapshot, floor residency uses one batch cache lock,
    and render cancellation is checked between expensive result-shaping
    boundaries. Deterministic concurrency coverage proves the GUI read can
    complete while mutation holds the cache lock; the frozen workflow did not
    expose a throughput improvement and retains the 7/60 deadlock.
14. **Governor lane policy** — 1 montage worker during gesture / park
    prefetch+inspection during visible work / seed first lane decision at
    desired value / bridge-drain clamp to one completion per visible-work
    turn / govern the real `montage_present_total` channel. Re-implement
    the *shape* with fresh benchmarks; don't trust the marathon constants.
15. **Pipeline admission batching** — `admission_chunk=2` in production,
    pending-queue preservation across same-target replans, per-completion
    refill, step-identity dedup, and killing the self-rearming no-op
    continuation (1000+ planning tasks). Re-derive the
    `reusable=True→False` rung-supersession change **separately** — prime
    suspect for the marathon's stale-transition regressions.
16. **Presentation-gate timer-edge pacing + GPU draw back-pressure**
    (`frame_effects.py`, `render_coordinator.py`) — replace self-reposting
    low-priority events with receiver-bound timers; don't admit a second
    cold VisPy batch until the canvas draw consumed the previous GL stream.
    Re-implement; the source diff sits in the regression epicenter.
17. **Source-keyed backend residency with atomic slot relocation**
    (both backends, +1,100 lines) — detached warm successors, atomic remap
    with zero uploads, `montage_layout_signature` excluding source indices
    so a scroll isn't a reflow, visual/page decoupling to keep GL program
    compiles off the interaction path. High value; re-implement the concept
    against V1/V2's unified owners, do not port the code.
18. **Viewport-continuity restore hardening** (`file_view_session.py`) —
    AUTO/FIT resolved against current layout + generation-guarded late
    layout-drift check. Good idea, fiddly; re-verify.
19. **`TileAdmissionQueue` max_free cap + retarget upsert limits** — tied
    to 17, same treatment.

## Do not port

- The full `frame_effects.py` / backend `tiles.py` diffs (known-red).
- Composite-raster `_DirectMontageCompositeItem` QPixmap experiment —
  legitimate future idea for PyQtGraph static-camera zoom/pan, but adds a
  whole caching state machine; separate experiment if PyQtGraph perf ever
  becomes the priority.

## Worktree policy

`.worktrees/redesign-r8-marathon` stays read-only until Tier 1 and Tier 2
are landed and at least items 11–13 of Tier 3 are re-derived or rejected
with a note here. Then delete it.
