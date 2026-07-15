# Current state

**Snapshot:** `codex/redesign-p8`, 2026-07-15, after the P8 correctness slice.
**[Codex 2026-07-15 — branch/state update]** P8 remains on a linear feature
branch rebased over `origin/main`; it has not been merged into `main`.
**[Codex 2026-07-14 — history correction]** No redesign integration merge
remains; completed queue and review commits are a single linear sequence.
**[Codex 2026-07-14 — post-V4 update]** The redesign execution record and
performance rules remain in [docs/redesign/README.md](redesign/README.md);
the live product ordering has returned to [docs/roadmap.md](roadmap.md).

## Architecture (what the rewrite delivered, R1–R7)

- One execution kernel (`arrayscope/kernel/`): priorities, dependencies,
  staleness, one worker→GUI bridge. WorkGraph and the eight FIFO pools are
  gone.
- One render pipeline (`arrayscope/render/`): typed stage contracts,
  `LodLadder` planner, kernel-backed `FramePipeline`; a normal image is a
  one-region plan, a montage a multi-region plan.
- One tile state machine (`presentation/tile_lifecycle.py`, ADR 0051).
- `frame_renderer.py` is deleted; orchestration lives on
  `RenderOrchestrator` (`window/render.py`) over
  `frame_controller/frame_session/frame_effects/frame_runtime`.

## Viewer-truth fixes through `codex/redesign-p8`

1. Required-tile admission, completion, evidence, and presentation consume
   the canonical `FrameSession.required_tile_numbers()` owner; the V1
   one-index/boundary scenario has correct pixels on both backends.
2. One canonical tile rank reaches kernel execution and progressive cold
   presentation; the V2 disjoint-scroll scenario paints center-out.
3. Dead prefetch imports and their silent broad-exception fallbacks are gone.
4. A stranded required tile is no longer silent: after two idle seconds it
   emits a `stall` owner-chain event, writes the bounded trace, and shows a
   persistent diagnostic.
5. Acknowledged montage extent changes replay AUTO/FIT camera intent against
   successor geometry; backend bounds changes no longer become USER input,
   and an actual USER camera is preserved.
6. Aggregate histogram sampling is derived on kernel workers behind tracker
   revision/source guards; prepared atomic tile transactions expire when the
   global level revision changes.
7. Real wheel/pan input paces committed-frame viewport replans at 16 ms without
   delaying camera motion, programmatic camera replay, or pipeline
   continuation; range signals no longer synchronously relayout the display
   group title.
8. GUI stage-reuse planning reads a published resident snapshot without the
   worker mutation lock; pyramid floor probes use one batch lookup, and
   superseded render results check cancellation between expensive shaping
   boundaries.
9. **[Codex 2026-07-15 — P8 update]** Interaction policy preserves one
   preview lane and parks exact/inspection work until the plan-wide preview
   pass closes. Recursive viewport continuation and synchronous VisPy draw
   acknowledgement now cross receiver-owned Qt turns; source/level
   convergence reaches the end of both backend workflows.
10. **[Codex 2026-07-15 — P9 pixel/order update]** Montage priority is owned
    by current semantic focus plus current layout, never by a backend cache.
    Ordered presentation commands survive VisPy upload and acknowledgement.
    Levels-only updates cannot replay stale mapping/component state, and every
    touched atlas page synchronizes its own physical shader uniforms. The exact
    60-tile real-display preview floor is coherent and its physical rows report
    one complex phase mapping with zero identity mismatches.
11. **[Codex 2026-07-15 — P9 level/admission update]** VisPy tiled no-op
    detection now consults public and physical layer levels as well as its
    completed-command cache, so an intervening automatic level update cannot
    make a later tiled command acknowledge stale physical uniforms. With that
    correctness blocker removed, the persistent four-item minimum cohort is
    safe under the byte cap and amortizes fixed transaction cost. The real
    FFT trace submits one shared full-volume transform, not one FFT per tile.

## Known open work

1. **Performance bars remain unmet.** The frozen T1 baseline exceeds the
   50 ms callback, 16 ms heartbeat, and 15 ms warm-input commitments; FFT
   scroll remains the primary throughput target. Follow the measured
   P-program in the roadmap.
2. **[Codex 2026-07-15 — superseded suite-risk update]** The stale
   deleted-owner assertions and the real coalescer/levels/viewport/ROI/
   cache-rebind/transition regressions from the pre-integration run have been
   repaired or migrated to canonical owners. The final parallel non-GPU run
   is **1,973 passed, 13 skipped** after the P9 level/admission checkpoint.
   Real-display GPU checks remain a separate
   mandatory gate and are not implied by that number.
3. Hardware evidence remains Linux-only; the histogram adapter remains
   sensitive to private PyQtGraph API.
4. **Completion-drain coalescing is rejected in isolation.** Empty-edge
   notification plus timer continuations passed its unit model but broke the
   real VisPy V2 pixel gate in three capacity-wake variants. The unchanged
   per-completion bridge remains production truth until the pipeline refill
   contract is redesigned with a real-display proof.
5. **[Codex 2026-07-15 — P8 recurrence update]** The former P6/P7 7/60
   presentation deadlock now converges on both backends. Performance remains
   unacceptable: complete workflow heartbeat maxima are 214-883 ms and FFT
   scroll takes about 29.0 s on VisPy / 17.3 s on PyQtGraph. The PyQtGraph
   run also emitted a stall-guard signature and its final whole-workflow trace
   scope was incomplete; preserve these as P9 evidence rather than reviving
   viewport-cadence or synchronous-continuation experiments.
6. **[Codex 2026-07-15 — P9 VisPy correctness veto]** Completion-owned
   admission refill is reverted despite PyQtGraph scroll improvements: VisPy
   scalar scroll regressed by about 94%, and a real FFT scroll displayed
   incompatible-looking tile groups while lifecycle acknowledgements still
   described complex float32 payloads. The defect may predate P9. Physical
   atlas/page identity, retarget rebinding, and backend-specific pixel tests
   are the current blocker before more performance work.
7. **[Codex 2026-07-15 — P9 shared-transform update]** The session-25
   100-presented/zero-work stall was a producer hole, not proof of atlas
   corruption: a finer acknowledged preview was excluded after coarsened
   demand, while shared target work had no plan-wide physical coverage gate.
   The shared path now uses unique required-tile coverage and re-arms from the
   backend acknowledgement edge.  Real PyQtGraph raw and VisPy FFT-full runs
   converge 272/272 with clean trace replay. The reported transient wrong
   pixels and initial FFT order are now closed for the exact real workflow:
   physical trace rows isolated stale scalar/page uniforms, and current-layout
   order now reaches upload and acknowledgement. General framebuffer readback
   remains an oracle gap, and convergence is still not accepted: one hard run
   stranded 58 target-ready tiles with only 2 target-presented and no work in
   flight.
8. **[Codex 2026-07-15 — duplicate key-owner risk]** Scalar montage cache-key
   derivation and its hoisted batch form remain separate implementations in
   `evaluator.py`. Runtime byte-parity fallback is intentionally load-bearing
   and observable as `montage_key_batch_fallbacks`; the next cache-owner slice
   must consolidate them without weakening that parity regression.
9. **[Codex 2026-07-15 — P9 performance remains open]** The accepted level-
   truth plus four-item admission slice cuts the real VisPy FFT refinement to
   8.39 s and scripted scroll to 13.55 s, with clean trace replay and no
   acknowledgement-churn violation. It still fails presentation continuity,
   the 16 ms heartbeat, and the 50 ms callback gate. Each scroll step still
   drains about 60 remaps through roughly seventeen commits; this is the next
   measured P9 surface, not permission to widen timeouts or hide continuity.

## Material risks

1. **Complexity debt is the top risk.** The renderer successor totals
   ~10,800 lines across six modules on one object; `FrameSession` has ~106
   fields; the same fact (residency, visibility, priority) lives in several
   owners. Every fix that doesn't reduce owner count tends to create the
   next bug.
2. **[Codex 2026-07-15 — suite/acceptance split update]** The broad offscreen
   suite is green again, but the visible-truth harness remains authoritative
   for pixels. Neither test layer substitutes for complete phase-scoped trace
   replay and real-display performance evidence.
3. **Performance work can regress truth.** Every P-step therefore needs
   before/after measurements plus the real-display pixel/trace gates.
4. **[Codex 2026-07-15 — oracle update]** The preview-floor harness now waits
   for required-target settlement, and verbose trace rows include physical
   page/slot, plane identities, texture dtype/shape, mapping/component, and
   typed acknowledgement identity. Those rows isolated the stale VisPy shader
   state. Backend framebuffer-to-CPU comparison with fault injection is still
   not implemented, so general pixel acceptance remains a real-display gate.

## What is working well

- The Qt-free semantic core (`core/`, `operations/`), operation pipeline,
  slicing, profiles, ROI, linked-window sync — untouched by the churn and
  solid.
- Kernel and ladder semantics are pinned by fast Qt-free tests (~0.5 s).
