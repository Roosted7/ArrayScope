# R5 bulk render governor evidence — 2026-07-27

## Ownership

`ResourceGovernor.decide_render_pass()` owns the item cap, byte cap, fixed
32 ms control target, 50 ms R5 evidence threshold, and the preview/target
feedback channels. It measures
`elapsed = fixed + item*n + byte*MiB`, reports residual/R² evidence, and
separates item from byte cost only when those axes varied independently enough
to identify both. The samples are the measured callbacks, not the generic
latency controller's outlier-suppressed values.

The model is split at its physical owners. Fixed/submission and byte/transport
terms share one bank for a `(backend, commit path, montage geometry)` key, so
preview, target, LOD changes, and data reloads reuse them. Preview and target
keep distinct item terms. A dtype/representation change starts a new pass-local
item bank, warm-seeded from the old term with high uncertainty; it does not
discard fixed or byte knowledge. Changing backend, commit path, tile geometry,
grid, or tile count selects a different structural bank.

Steering minimizes predicted total fill time plus one continuous callback
latency price: zero through 18 ms, gently quadratic through 45 ms, and a
value/slope-continuous stronger quadratic above 45 ms. Fifty milliseconds is
therefore a reported violation point, not a cap or a controller mode switch.
Model extrapolation is also continuous: up to 1.1× the measured item/byte
range is free, then an exponential evidence-distance price makes 2×
noticeable, 10× expensive, and 100× practically unavailable without making
any point a hard yes/no boundary. That one-time price is scaled by fit
uncertainty derived from sample count, observed cohort span, and residual
variance. A narrow cold fit is optimistic enough to buy information; an
eight-sample, wide, low-residual fit recovers the full evidence-distance curve.
The 32/50 ms targets never adapt downward.

`responsiveness_weight` scales only the latency-price term. The exposed
setting has exactly three presets: Responsive 2.0, Balanced 1.0, and Throughput
0.3. Remote, offscreen, missing-GPU, and software-rendered sessions seed
Throughput when no value has been stored; an explicit user choice always wins.
The setting cannot change the R5 report line, budget, or any invariant.

`FramePipeline` produces preview tiles through its ordinary governed
continuation instead of one whole-round worker task. Presentation effects pass
the governor decision through unchanged. Backends may stop earlier at the
deadline; they cannot widen the cap or deadline.

PyQtGraph prefixes use ordinary hidden-per-item staging. The compact atlas is
still a complete hidden transaction and is never exposed as partial truth, but
production cannot enter that atomic branch unless the governor admits the
entire planned set in one measured chunk. WGPU has the same full-set admission
guard for its preview atlas.

Every backend completion emits `pass_kind`, `pass_chunk_items`,
`pass_chunk_budget_ms`, `pass_chunk_within_50ms`, and
`pass_completed_atomically`, plus payload-build, backend, acknowledgement,
state-publication, geometry, governor fixed/item/byte fit,
fit residual, predicted callback, latency price, and extrapolation price.
`presentation_build_timing` further attributes governed payload construction.
WGPU additionally measures structural executor initialization and texture-pool
growth at their owners. They remain inside the full callback/R5 evidence, but
the steady render-pass cost sample subtracts them: a one-time capacity
transition is not evidence that the next cohort's fixed, count, or byte cost
changed.

`python -m arrayscope.tools.render_pass_governor_probe` is the reusable
acceptance probe. It enters managed Weston itself, balances backend order when
`--backend all` is requested, reports true completed-tile throughput from
wall-clock milestones separately from callback work-item throughput, and
distills full and structural-compensated chunk distributions without writing
JSONL artifacts. It also runs a deterministic analytic cold-span A/B for
fixed-dominated, mixed, and per-item regimes and reports cold fill plus the
number of choices required to reach within 10% of the informed cohort.
The report labels the Weston compositor renderer separately from the
application renderer. Weston deliberately uses GL to composite Wayland
surfaces; ArrayScope's WGPU instance remains Vulkan-only and the report records
its adapter `backend_type`, so a compositor GLES log line cannot be mistaken
for a client fallback.

The optimizer no longer scans every cohort from 1 through `remaining`. It
finds the marginal-cost root of the continuous one-dimensional objective and
checks a bounded neighbourhood of integer and chunk-boundary candidates.

### Learning A/B (272 items)

Both arms use the same known fixed/item parameters and begin with an observed
cohort maximum of one; this isolates extrapolation learning from fit error.

| Regime | Pure exploitation fill / chunks / convergence | Uncertainty-aware fill / chunks / convergence |
|---|---:|---:|
| Fixed-dominated (49.9 ms + 0.1 ms/item) | 226.8 ms / 4 / 3 | 176.9 ms / 3 / 2 |
| Mixed (10 ms + 2 ms/item) | 724.0 ms / 18 / 2 | 724.0 ms / 18 / 2 |
| Per-item (2 ms + 8 ms/item) | 2358.0 ms / 91 / 1 | 2358.0 ms / 91 / 1 |

The informed first cohorts are 136, 16, and 3 respectively. Exploration buys
one decision in the fixed-dominated case and does not perturb the mixed or
per-item optima.

### Changing-load robustness

The fit uses an exponentially recency-weighted Cauchy regression initialized
by a repeated-median slope. Equal-valued callbacks count as separate
observations; tuple equality no longer makes a sustained plateau invisible.
A stable prior-round model is kept as the intrinsic cost surface. The median
residual of the latest three callbacks is a separate item-independent machine
load offset, so one arbitrary stall cannot rewrite item/byte cost, two
consistent late observations establish a changed regime, and two normal
observations establish recovery. The deterministic gate holds a 25-item
cohort through one +80 ms spike, moves to 68 when the +80 ms delay persists,
and returns to 25 after two clean callbacks while retaining the measured
1.00 ms/item slope. The fixed 32/50 ms report does not move.

### Quadratic-term check

The real-backend probe did not produce repeatable curvature. WGPU produced no
non-zero fit. PyQtGraph produced only isolated estimates of 0.01 and
1.23 ms/item², absent in the paired repeats and differing by 123× while those
runs were censored under elevated host load. That is noise-sensitive,
unfalsifiable curve fitting, not a transport property. The cohort-quadratic
term and its fitter were removed; the real model is fixed + item + byte.

## Weston measurements

All runs used the managed private Weston compositor, the deterministic geometry
scene, a 272-tile plan, and the final code in this branch. Baseline and current
were each measured with one in-process, order-balanced, interleaved
`--backend all --repeat 4` run. The product/test interaction gate remained
5 s.

| Backend/pass | Repeat | chunks | full min / p50 / p95 / max (ms) | >50 | pool / init total (ms) | atomic |
|---|---:|---:|---:|---:|---:|---:|
| PyQtGraph preview | 0 | 8 | 5.0 / 30.4 / 44.4 / 44.4 | 0 | 0 / 0 | 0 |
| PyQtGraph target | 0 | 19 | 5.5 / 27.1 / 49.3 / 98.4 | 1 | 0 / 0 | 0 |
| PyQtGraph preview | 1 | 8 | 5.2 / 26.2 / 45.3 / 45.3 | 0 | 0 / 0 | 0 |
| PyQtGraph target | 1 | 11 | 18.7 / 31.4 / 63.9 / 63.9 | 1 | 0 / 0 | 0 |
| PyQtGraph preview | 2 | 8 | 4.4 / 22.4 / 43.3 / 43.3 | 0 | 0 / 0 | 0 |
| PyQtGraph target | 2 | 10 | 13.8 / 25.6 / 148.6 / 148.6 | 2 | 0 / 0 | 0 |
| PyQtGraph preview | 3 | 9 | 4.9 / 26.4 / 42.4 / 42.4 | 0 | 0 / 0 | 0 |
| PyQtGraph target | 3 | 17 | 14.1 / 17.5 / 34.4 / 164.5 | 1 | 0 / 0 | 0 |
| WGPU preview | 0 | 10 | 10.4 / 38.8 / 569.1 / 569.1 | 2 | 0 / 384.4 | 0 |
| WGPU target | 0 | 18 | 5.1 / 38.9 / 51.5 / 138.6 | 4 | 28.3 / 0 | 0 |
| WGPU preview | 1 | 8 | 8.6 / 32.9 / 55.4 / 55.4 | 2 | 18.3 / 7.0 | 0 |
| WGPU target | 1 | 18 | 21.6 / 42.2 / 94.0 / 127.0 | 5 | 45.5 / 0 | 0 |
| WGPU preview | 2 | 8 | 9.2 / 32.6 / 55.5 / 55.5 | 1 | 19.2 / 6.8 | 0 |
| WGPU target | 2 | 12 | 28.1 / 51.0 / 56.7 / 95.0 | 7 | 40.8 / 0 | 0 |
| WGPU preview | 3 | 9 | 10.7 / 39.9 / 46.8 / 46.8 | 0 | 20.3 / 6.5 | 0 |
| WGPU target | 3 | 17 | 19.7 / 42.7 / 89.3 / 161.0 | 3 | 43.1 / 0 | 0 |

The 569.1 ms WGPU callback is cold executor initialization plus remaining
backend work, not an atomic pass; 384.4 ms is attributed to initialization.
Warm pool changes remain inside full R5 evidence but are subtracted from the
steady cost sample. First widened target commits still have 117–144 ms WGPU
backend-apply components even though later cohorts of similar size cost
25–32 ms. That is the main measured follow-up optimization debt.

### Wall time and throughput

Preview throughput is `272 / preview-complete`; target throughput is
`272 / (target-settle - preview-complete)`.

| Backend/arm | valid runs | preview median | settle median | target-phase median | target tiles/s median | censored |
|---|---:|---:|---:|---:|---:|---:|
| WGPU base `3970f5e4` | 3/4 | 1.289 s | 4.571 s | 3.216 s | 84.6 | 1/4 |
| WGPU current | 4/4 | 1.263 s | 4.221 s | 2.960 s | 92.0 | 0/4 |
| PyQtGraph base `3970f5e4` | 2/4 | 2.062 s | 4.843 s | 2.781 s | 98.0 | 2/4 |
| PyQtGraph current | 2/4 | 2.019 s | 4.538 s | 2.519 s | 108.0 | 2/4 |

Against the exact matched base, WGPU settlement improves 7.7% and its
target phase improves 8.0%; preview improves 2.0%. PyQtGraph's valid-settlement
median improves 6.3% and its target phase 9.4%, but the unchanged 2/4
censoring is still lifecycle/settlement debt. An earlier experimental
hard two-hit rebase produced a 3.94 s WGPU median once, but also produced
128–185 ms preview jumps and a later 4.83 s rerun. It is retained as an
optimization target, not presented as a stable baseline.

## Result and remaining red evidence

No preview or target pass completed atomically in these runs. The former
PyQtGraph compact-preview whole-atlas burst and FFT whole-active-set
republication are gone: the backend resolves only the admitted upserts and
retains already drawn items.

PyQtGraph preview is green in these runs, but target still has 64–165 ms
outliers and two censored settlements. WGPU no longer performs
whole-active-set binding publication on ordinary deltas: stable plane prefixes,
tile instances, level evidence, and committed state advance only for admitted
upserts. Its pool capacity is still derived from the physical byte working set;
incremental commits grow that existing executor in place instead of skipping
the capacity owner. Bound-page pins name committed coverage rather than every
resident LOD family.

R5 remains honestly red for the WGPU cold callback, first-widened backend
apply, warm pool-growth callbacks, PyQtGraph scalar-target outliers, and the
previously measured indivisible PyQtGraph FFT item updates. The governor
records and prices steady components, models a changing machine load
separately, excludes measured one-time structural transitions from its cost
fit, never lowers the target, and does not pretend item shrinkage can repair
fixed cost. No measured preview or target pass completed atomically.
