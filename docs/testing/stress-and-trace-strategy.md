# Stress testing and trace-based correctness (2026-07-15)

How we stress every feature, prove correctness from recorded evidence, and
keep both properties while adding features and optimizing performance.

## The model: drivers × oracles

Keep the two roles separate and let them multiply, instead of writing a
bespoke assertion per scenario:

- **Drivers** exercise the app: the workflow harness
  (`tools/profile_montage_workflow.py` — cold load, FFT, refinement, scroll,
  zoom/pan), the GPU interaction harness (`tests/gpu_interaction/` — real
  display, real input, pixel readback), and the stress matrix
  (`tests/stress/` — the same workflow across synthetic input classes).
- **Oracles** judge the recorded evidence: `trace_verify` (final-scope
  acknowledgement, stall events, acknowledgement-churn livelock, vacuous-pass
  guards), `trace_latency` (queue/run/drain spans, input→first-ack, outcome
  breakdown), the R8 certification gates (canonical fixture only), and pixel
  assertions (real display only).

Every driver always records a trace. Any new correctness question becomes a
new oracle over traces we already know how to produce — never a new bespoke
harness.

## The rings (cost-ordered; each gates the next)

1. **Per-commit (seconds):** Qt-free kernel/render/trace suites + import
   health. Pure logic, no app.
2. **Per-change offscreen (minutes):** focused offscreen suites for the
   touched area, plus — for any rendering/scheduling change — one synthetic
   workflow run with `--trace` replayed through `trace_verify`.
3. **Stress matrix (opt-in, ~1–2 min serial):**
   `ARRAYSCOPE_STRESS=1 pytest tests/stress -n 0`. Datasets are input
   *classes* (dtype, anisotropy, tiny/deep axes); oracles are portable
   invariants (phases settle, no stall, no churn, trace replay clean). R8
   certification gates are advisory here — they are calibrated to the
   canonical fixture geometry.
4. **Real-display acceptance (the only gate that can say "done"):** the
   V1/V2-style Wayland scenarios and the canonical-fixture workflow with the
   full R8 gate set. Pixels + trace, both backends.

Every driver uses the repository interaction budget: 2 s target and 5 s hard
failure for each user-visible step. A scenario may contain many steps and take
longer overall, but no individual open, render, zoom, pan, scroll, scrub,
level, or refinement step may settle late and still pass. Shorter custom
budgets remain capped by `arrayscope.tools.interaction_budget`; timeout
widening is never a convergence fix.

A change is *suspected* good at ring 2, *confident* at ring 3, *accepted*
only at ring 4.

## Using it for new features

1. **New input class → new matrix row.** A feature that accepts new data
   (dtype, layout, source protocol) adds a `DATASETS` row before it merges.
   The complex64 row found a presentation deadlock on its first-ever run.
2. **New behavior → new trace events + one oracle rule.** A feature that adds
   state transitions must emit them on the bus (one `emit_trace` call at its
   choke point) and extend `verify_trace` with the invariant that makes its
   failure visible in replay. The rule of thumb from the retained-payload
   gap: **if a subsystem can satisfy an obligation without doing work, it
   must say so in the trace** (`target_satisfied_retained`), or replay cannot
   tell "satisfied cheaply" from "never satisfied".
3. **New scenario → drive, don't simulate.** Prefer a phase or GPU-harness
   step over a SimpleNamespace reconstruction; the R8 era showed simulated
   completion models drift from production and then pin the drift.

## Using it for performance work

1. **Freeze the question first.** Every P-step names one measured cause and
   carries before/after `trace_latency` output on the same fixture and
   geometry. (This is already the P-program discipline; keep it.)
2. **Correctness gates run on the *optimized* build.** Ring 3 + ring 4 after
   every performance change — P2 and P5 were both caught regressing pixels
   or priority by exactly these gates and were reverted. That is the system
   working; keep the reverts cheap and recorded.
3. **Watch the wasted-work counters, not just latency.**
   `kernel_finish_outcomes` (superseded/stale ratios), commit-batch counts,
   and identical-ack counts are the early-warning channel: churn shows up
   there before it shows up in frame time (the 5,521→398→1 identical-ack
   trajectory tracked the livelock fix precisely).
4. **The stall/churn pair is the safety net under load:** the V3 watchdog
   catches deadlock at runtime; `no_acknowledgement_churn` catches livelock
   in replay. Every stress or benchmark run gets both for free because the
   trace is always on.

## Known gaps (owners in the queue)

- ~~**`target_satisfied_retained` is not emitted yet.**~~ Closed 2026-07-17:
  the lifecycle emits it once per target requirement closed by retained
  pixels, `verify_trace` re-judges the edge with the production settlement
  rule, and `TOLERATED_INVARIANTS` is empty — the strongest invariant is
  enforced in whole-workflow replays.
- **The stress matrix is unstable** (see its docstring): synthetic-input
  convergence is nondeterministic at dfa53db3, complex64 raw input deadlocks
  deterministically, tiny-montage level settlement is racy. The matrix goes
  green by fixing those, never by loosening it.
- **Real-display evidence is manual and Linux-only.** The rings above cannot
  claim pixels; keep ring 4 mandatory for acceptance.

## Addendum (2026-07-15, review laws corrected by live validation)

**[Codex correction 2026-07-15]** The review identified real oracle gaps, but
its prescriptions are requirements, not evidence that those oracles already
exist.  In particular, the tile overlay and trace remain intent/identity
oracles; real framebuffer-to-CPU comparison is still missing.  The required
coverage set is `FrameSession.required_tile_numbers()`, not every active
lifecycle row: retained or near-viewport residency may legitimately be wider
than the current viewport obligation.

Three failure laws that session proved, now binding for oracle design:

1. **A count is not coverage.** The preview-floor trigger counted 272
   presentation *events* while only 158/272 unique tiles were physically
   present. This is the third appearance of the same bug class (R8 visible
   predicates, the harness floor counter, the P8 barrier). Any "complete"
   predicate must cover the set of unique current required identities, never
   an event counter or the wider active cache population — and its test must
   include repeats, replacements, and retained active rows outside the scope.
2. **Intent is not pixels.** The tile truth overlay and the trace both
   report upload-intent (CPU-side identity, intended uniforms). A frame can
   be visibly wrong while every label is truthful. Scenarios that accept a
   frame must include framebuffer readback compared against a CPU-rendered
   reference (generalize `assert_tile_identity_ramp` into
   `assert_tile_matches_cpu_reference`), and each oracle needs a
   fault-injection audit: deliberately break one uniform/mapping and assert
   the oracle catches it. This is a required next testing slice, not a gate
   satisfied by the current harness. An oracle that has never failed on an
   injected fault is unproven.

   **[Landed 2026-07-17]** The general oracle exists:
   `tests/oracles/framebuffer_reference.py`
   (`assert_frame_matches_cpu_reference`, surfaced on the GPU harness as
   `Harness.assert_tile_matches_cpu_reference`). It reads the live VisPy
   canvas framebuffer and compares every `required_tile_numbers()` tile
   interior against `cpu_display_rgba` of the committed payload values —
   component/scale/levels/LUT via `arrayscope.display.shader_mapping`,
   geometry via the real camera transform — tolerating only GPU rounding
   (healthy worst-case deviation measured at 1/255) with built-in vacuity
   guards (set-equality tile coverage, per-tile sample floor). The
   fault-injection audit lives in
   `tests/gpu_interaction/test_framebuffer_cpu_reference.py` (real GL): a
   wrong levels uniform, stale atlas-page texels behind fresh mapping keys,
   and swapped tile texcoords each fail the oracle, and restoring the state
   turns it green again. A default-ring smoke
   (`tests/ui/test_framebuffer_cpu_reference.py`, offscreen software GL)
   keeps the oracle honest per push but is never rendering acceptance.
   Bounds, stated loudly: RGB payload modes raise `NotImplementedError`
   rather than silently passing, and PyQtGraph has no equivalent physical
   readback gate yet (its complex modes are CPU-mapped, but scalar
   levels/LUT run in the Qt raster path and stay uncovered).
3. **Small fixtures skip regimes.** `preview_level = max(base, desired)`
   means 64×64 fixtures never enter the two-stage preview path that 336×336
   data exercises; a green 6×6 harness said nothing about the 272-tile
   failure. Parametrize scenarios by *regime* (preview vs native-only,
   shared-stage vs per-tile, reduced vs exact) and assert the regime was
   actually entered — a scenario that silently ran in the wrong regime must
   fail, not pass.

And one workflow rule: **manual observation must become replayable
evidence.** Run manual sessions with the trace on and periodic screenshots
(flight-recorder mode; the ring buffer + stall dumps already exist); a
human marks a timestamp when something looks wrong instead of describing
it. The human is the rarest oracle — spend them on glancing and marking,
never on hours of monitoring.

**[Codex rejected generalization 2026-07-15]** Applying the physical preview
barrier to every `FramePipeline` quality rung passed the focused suite and the
VisPy workflow, but stranded 45/272 ordinary PyQtGraph raw tiles with no work
in flight.  That implementation was removed.  The accepted barrier is scoped
to shared transforms, the path that bypasses the per-tile ladder and produced
the recorded 100-presented/zero-work stall.  Do not reintroduce a generic
pipeline barrier without a separate obligation and backend proof.

**[Codex physical-trace update 2026-07-15]** Verbose workflow rows now capture
physical atlas page/slot, real/imag plane identity, texture storage/dtype/shape,
mapping/component/levels/shader key, and the physical acknowledged identity.
Backend identity comparisons use the typed `tile_ack_identity(payload)` rather
than comparing unrelated source identifiers. This evidence isolated a real
failure to stale scalar and page-local uniforms while complex plane identities
remained correct. Fault-shaped tests now prove that levels-only updates cannot
replay a stale scalar mapping and that touching a stale atlas page repairs its
local mapping even when the global key is unchanged. This strengthens trace
diagnosis; it does not replace the still-missing general framebuffer-to-CPU
oracle.
