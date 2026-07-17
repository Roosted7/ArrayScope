# G5 source-grid sparse pyramid contract — 2026-07-16

**Status:** authoritative implementation contract and landing dossier for ADR
0056 G5. The canonical live page route, reducer families, renderer-shared
cache, producer migration, and both backend consumers are implemented on the
landing candidate; the remaining broad/stress and real-Wayland acceptance
matrix still gates queue row 1.

## One canonical reduction route

Ingest reduction, ladder materialization, cached derivation, and later
GPU-from-resident generation must all request the same source-grid page plan.
The plan is pure integer geometry and contains:

- document/operation generation, representation, dtype, reducer family, and
  anisotropic reduction vector;
- native-source footprint and valid input footprint for every logical page;
- stored-sample rectangle, global bin origin, and source samples per stored
  sample on every axis;
- clipped boundary coverage, gutter/sample expansion, and the exact transform
  from a target page's sample coordinates to the resolved resident page.

No caller may independently start reduction bins at its window-local zero.
`reduce_box_mean(array)` remains a low-level numeric primitive only; the
source-grid planner decides which samples each bin owns before any numeric
backend runs.

## Identity and boundary rules

Logical `DataChunkKey` page geometry is expressed in native source coordinates.
At any LOD, the physical page stores a uniform sample extent while its source
footprint grows by the reduction vector. Full aligned interior bins therefore
keep the same identity across overlapping windows.

Partial boundary bins are different values when their valid sample coverage
differs, even if they occupy the same nominal global bin. Their identity must
include the clipped valid footprint (directly or through a value-source key).
They are not reusable merely because a screen window moved by one pixel.

Gutters expand input coverage but do not move the global bin origin. Recursive
parent-to-child generation is legal only when grid origin, reducer lineage,
sufficient statistics, and valid coverage prove that the composition equals a
direct reduction. Otherwise the canonical route derives from the nearest valid
source, never from cache history.

## Page-table resolution and pins

A target virtual page resolves once on the CPU to either:

- the exact resident page; or
- the finest compatible resident ancestor whose source footprint covers the
  target, including actual key/LOD, physical slot, target-to-resident sample
  scale and offset, and that binding's generation.

Document/operation generation, representation, dtype, spatial coverage,
reducer family, and gutter must all match. Anisotropic reductions use
componentwise ancestry; a semantic-family mismatch never aliases. The shader
consumes the resolved binding and does not walk an ancestor ladder per
fragment.

Pins are owner-scoped sets, replaced atomically. Several active target pages
may share one coarse ancestor; one target leaving must not unpin coverage still
owned by another. Slot compaction/reuse refreshes binding generations so an old
resolution can never sample a new occupant. When all capacity is pinned,
refinement is denied and coarse coverage stays resident.

## Physical truth

A coarse ancestor drawn for a fine target is acknowledged as the actual coarse
LOD and fallback quality. It must never acknowledge the requested fine/exact
identity. Finer arrival changes only the page-table resolution; fine eviction
falls back to pinned coarse coverage without a black frame. Explicit missing
display is allowed only when no compatible resident ancestor exists.

## Ordered slices and gates

1. Pure `arrayscope/gpu` page geometry, ancestor resolution, binding
   generations, and owner-scoped pins.
2. VisPy consumption of resolved pages with actual-LOD physical truth and
   never-black fine-arrival/fine-eviction gates.
3. Ladder/cache migration from whole-plane `PyramidLevelKey` identity to
   logical `DataChunkKey` pages.
4. Source-grid reduction binning: origins 101 and 102 share aligned interior
   pages, keep clipped boundaries distinct, upload only boundary pages at
   factor > 1, and match the direct CPU source-grid oracle.
5. Reducer families and phase-cancellation correctness, followed by real
   Wayland GL certification.

G5 is row **1** of [`docs/queue.md`](../queue.md); older plan text referring
to queue step 2 is stale after the churn-convergence net landed. Each stage's
commit message records the ring, exact pass/skip counts, and artifact paths;
documentation evidence is not a substitute for commit-local evidence.

Before row 1 can move to Done, the real-Wayland ring includes both the
dedicated never-black `tests/gpu_interaction` coarse/fine arrival/eviction
scenario and a live, real-data run of
`tests/stress/test_interaction_convergence.py`. Performance evidence is valid
only on real data and is stored under `tests/artifacts/<gate>-<date>/`; the
synthetic registration charts diagnose placement, phase, and continuity but
do not establish a performance number.

All G5 gates also obey the repository-wide per-step interaction budget: 2 s
target and 5 s hard failure. A multi-step zoom/pan/scroll scenario receives a
fresh budget per step, but no stage may turn a 5+ s settlement into a pass by
widening its timeout. Each stage's commit message records the ring, exact
pass/skip counts, and artifact paths.

## Implementation progress

Release capture and interaction probes now share one read-only settlement
snapshot in `arrayscope.tools.presentation_settlement`. It combines the
current frame/viewport target token, `TileLifecycle` target completion, commit
debt, backend acknowledgement identity, exact required physical coverage,
backend-qualified geometry truth, and draw completion without scheduling or
mutating the live pipeline. Release diagnostics require a distinct target for
the initial image, changed slice, and montage captures; geometry-only or stale
physical rows fail loudly at the repository five-second cap.

The pure model and first VisPy consumption slice now stand without scheduler
coupling:

- `PageTable.resolve` returns exact or finest compatible covering residency,
  including actual key/slot, target-to-resident sample transform, and binding
  generation; unbinding fine content immediately exposes coarse fallback;
- slot remap/reuse mints new binding generations;
- `ChunkStore.replace_pins(owner, keys)` atomically owns coverage without one
  consumer unpinning another consumer's shared ancestor;
- `reduce_source_grid_mean` plans global anisotropic bins, reports native-source
  coverage/identity per sample, shares aligned interiors across origins 101/102,
  rejects clipped recursive inputs, and matches a direct CPU oracle;
- anchored atlas chunks now use canonical `DataChunkKey` identities instead of
  backend-private tuples; the mixed atlas page table deliberately excludes
  legacy whole-tile keys from logical ancestor lookup;
- `TextureAtlasPool.resolve_page_targets` performs one bounded CPU resolution
  pass, changes mappings/pins only, and never uploads or schedules. It rebinds
  fine arrival and fine removal in place, reports explicit missing only when no
  compatible page exists, and refreshes bindings after slot remap;
- owner pins are honored by every atlas eviction route, including speculative
  warm placement and superseded-page reclamation;
- physical presentation rows report target key, actual key/LOD, exact versus
  fallback quality, and binding generation; presented identity is the actual
  resident page, never the requested fine page;
- The initial seam landed with 111 focused GPU/pyramid/VisPy tests; the live
  ladder/cache migration and later evidence are recorded below.

The live page-backed VisPy cutover now also has a restored-session geometry
guard. A field reproduction using the saved 100-tile complex session found
that each per-tile resolver call treated its one-tile input as a complete
frame and cleared all earlier tiles' page mappings. Ninety-nine L2 tiles then
sampled a padded 336-by-336 slot instead of their valid 84-by-84 UV crop; only
the last committed tile kept the correct size. Multi-page resolution input is
now explicitly partial across tiles, while frame-boundary removal remains the
only owner that clears omitted tiles. The regression commits two page-backed
tiles in one frame and requires both complete binding sets and draw parts.
The real-Wayland saved-session artifact
`tests/artifacts/g5-wrong-tile-size-2026-07-16/repro-fixed/` records 100/100
bound tiles with one consistent UV span and uniform framebuffer geometry by
1.23 s. The accompanying zoom/pan artifact remains a red convergence gate:
it preserved page bindings but correctly hard-failed when an L6 target had
only complete L2/L4 fallback and no materialization work in flight. That
lost-wakeup is follow-on work, not a timeout exception or a reason to weaken
the never-black fallback.

The follow-on trace showed that no L6 work was actually missing: retained L2
and L4 pixels already exceeded the later L6 demand. `TileRecord.target_settled`
and the scoped `TileLifecycle` queries had drifted into two implementations;
the latter required the historical quality label `exact` and therefore called
finer retained fallback unsettled. `TileRecord` now owns first-pixel and target
settlement truth, all scoped queries delegate to it, and one shared quality/LOD
rule prevents demotion: an exact level satisfies its own or a coarser demand,
while a retained fallback satisfies only a *strictly* coarser later demand.
Equal-level and genuinely coarser fallback remain unsettled. The real-Wayland
artifact
`tests/artifacts/g5-wrong-tile-size-2026-07-16/zoompan-settlement-fixed/`
reaches `required_target_settled=True` without producing L6 replacement work.
The later real-data scrub gate exposed the complementary producer-side drift:
`FramePipelineEffects` asked the backend identity rule whether an equal-level
fallback was safe to draw, then treated that answer as proof that exact work
was already covered. The fallback correctly prevented black while tiles 0--7
were left with no exact claim or task. Producer suppression now delegates to
the same `TilePayloadRef.satisfies_target` lifecycle rule as settlement;
backend fallback drawability remains unchanged. A parsed 64-slice, 60-step
real-Wayland scrub changes from 56/64 exact acknowledgements plus a stall to
64/64, zero stalls, and zero identity rejections within the unchanged five-
second step cap. Before/after traces and verifier output are preserved under
`tests/artifacts/g5-probe-scrub-lost-wakeup-2026-07-17/`.

The saved-session zoom/pan/FFT trace then exposed a second correctness defect:
the complete predecessor surface collapsed to 16 or 69 drawn tiles when the
first bounded successor batch was acknowledged. Retention and atomic handoff
had drifted across a source-window flag, an independent completion generation,
derived dtype/RGB/geometry/ViewState vetoes, and a backend atomic query that no
backend implemented. One immutable presentation-transition decision now owns
retention, atomic obligation, reason, and trace detail. Its initial rule was too
broad: it allowed a common staged base to retain pixels across an operation
change. The corrected contract separates residency from visibility. Old pages
may remain resident for a later revert, but a document/operation source change
hides their mappings and shows black/placeholder until successor pixels present.
Never-black retention and coarse fallback are allowed only under the same full
semantic source identity; shader-only levels/LUT/mapping state crosses with the
compatible pixels atomically. The redundant
completion generation, lifecycle-based clear, and dead backend query are gone.
The real-Wayland artifact
`tests/artifacts/g5-wrong-tile-size-2026-07-16/zoompan-atomic-view-fixed/`
has no post-start sample with drawn coverage below the visible set and the
`presentation_continuity` gate is green. Its remaining failures are LOD
checkpoint coverage and GUI/event-loop latency, not tolerated continuity gaps.

A later 60-step real-Wayland scrub found the atomic obligation itself had
acquired a second owner: commit code re-decided the already-armed
`FrameSession.atomic_successor_pending` flag from the lagging committed semantic
frame. When that derived check disagreed, commits ran without the atomic guard
while the canonical flag could never be acknowledged, producing an idle
`pipeline_plan steps=[]` loop. The transition owner now arms the flag and the
backend acknowledgement alone clears it. The focused atomic ring passes 4/4,
the montage LOD residency ring passes 181/181, and the montage backend ring
passes 88/88. The replay at
`tests/artifacts/g5-atomic-successor-owner-2026-07-17/stdout.log` retains complete
64/64 physical coverage and clears all atomic commit debt. It remains a red
exact-convergence gate: 31 L1 tasks were still in flight at the unchanged
five-second cap, so this evidence does not claim the full scrub green.

The semantic-compatibility boundary is now explicit in that same transition
owner. A later axes/PyQtGraph gate caught the predecessor mappings remaining
visible after transposing the image axes. `plan_presentation_transition`
therefore normalizes only the source-selection field that retention is allowed
to bridge: `slice_indices` for a single image, or
`montage_indices`/`montage_text` for a montage. Axes, flips, channel/view mode,
and every other `ViewState` difference reject predecessor retention. Rejection
hides the old mapping through the existing surface invalidation while leaving
its CPU/GPU residency reusable for a later revert; it does not clear caches or
add a second freshness set. The full real-Wayland semantic transition matrix
(operation, real channel, complex mode, and axes on both backends) passes 8/8,
and the full montage LOD residency file passes 181/181.

A later 500 ms framebuffer sequence caught a stricter atomicity violation that
the tile-count gates missed: during raw-to-FFT replacement, canonical successor
pages could be uploaded and rebound while the layer still held predecessor
levels/mapping (or vice versa), briefly producing white-background grayscale or
psychedelic complex tiles. The real-Wayland artifact
`tests/artifacts/g5-wrong-tile-size-2026-07-16/zoompan-atomic-handoff-fixed/`
is retained as a **red** diagnostic: its scalar frames proved that
cross-operation predecessor retention could present stale complex values under
scalar target state. Page-backed warming remains residency-only, but operation
changes now hide the incompatible presentation; only same-source LOD and
shader-presentation handoffs may retain visible predecessor bindings. The
replacement real-Wayland saved-session sequence is retained at
`tests/artifacts/g5-wrong-tile-size-2026-07-16/scalar-freshness-storage-pressure-fixed/`.
Its periodic framebuffer/physical-truth record keeps one storage/mapping family
per sample across FFT-to-scalar replacement, ends with 60/60 exact scalar L0
bindings and settled target coverage, and was visually confirmed onscreen. Its
three red gates are explicitly performance-only (GUI callback, heartbeat, and
warm-input latency); they do not weaken this semantic correctness result.

The periodic screenshot/JSONL probe now separates lifecycle acknowledgement,
resident physical rows, scene-presented primitives, and primitives that
actually intersect the camera. It records the scripted action, live/session/
VisPy camera ranges, content intersection, projected tile sizes, exact/fallback
bindings, missing candidates, page-table occupancy, and atlas storage classes;
the generated contact sheet labels scene and onscreen counts. The earlier
stress path also derived an off-centre zoom from the maximum-out range, putting
the entire camera tens of thousands of samples outside the montage and making
correct black frames look like renderer failures. Non-limit gestures now start
from the content range. The real-Wayland evidence at
`tests/artifacts/g5-wrong-tile-size-2026-07-16/visual-truth-camera-fixed/`
contains 23 periodic frames: every sample has one 336×336 world-size class, no
layout-bound mismatch, no mixed scalar/complex storage/mapping/levels among
scene primitives, and a VisPy camera key matching the live range. The semantic
operation switch is explicitly hidden before scalar pixels arrive; the final
frame has 60/60 scalar scene bindings. Its three remaining reds are the same
performance-only gates and remain work, not widened timeouts.

The resident-LOD matrix now separates candidate construction from physical
truth. An unacknowledged native wrapper may be replaced by the first legitimate
reduced commit, while an acknowledged finer compatible presentation satisfies
later coarser demand and cannot be demoted by a logical payload/cache byte
estimate. Only the backend's physical-capacity owner may reclaim it, after
less-important residency and with complete same-source fallback retained.
Translated level-swap coverage still proves distinct page identity, per-tile
mixed availability, bounded deferral without removals, native semantic-stat
reuse, and stable histogram identity without manufacturing logical pressure.

Producer success is also exact now: ingest/worker materialization, retained
preview admission, prefetch, lifecycle residency, and terminal claim handling
require the precise planned `DataChunkKey` set. A coarse ancestor can keep a
target drawable but cannot suppress a finer claim or mark it resident. Floor
candidates are ranked by the actual physical page set returned by the same
resolver, so an L2 fallback can no longer win by masquerading as a hypothetical
L1 request. PyQtGraph's independent ancestry scan was deleted; both PyQtGraph
and VisPy use `PageTable.resolve` and report requested-to-actual identity,
anisotropic scale/offset, and exact/fallback quality. The focused serial ring
covering keys, chunk store, page-table resolution, source-grid geometry,
materialization, both backend routes, target planning, and the resident matrix
passes **283 tests** in 1.16 s on the current slice.

The page cache now attaches an owner to the whole requested set, not just its
new boundary claims. Before each admission it touches resident members of that
set, so a two-page-budget shift in either direction retains the shared interior
and evicts the outgoing boundary. Cache-ineligible exact sets are rejected
before worker scheduling and remembered until resize; the former completion →
eviction → GUI-admission race declines normally and balances every claim rather
than throwing or immediately recomputing an impossible set. Requested and
actual `LodInfo` source/stored shapes are checked against immutable plans and
resolved values, structurally rejecting the malformed payload class that made
tiles draw at mixed sizes. Native two-dimensional `uint8` is likewise a scalar
page family; only actual three/four-component `uint8` values may claim `RGB8`.

History identifies two earlier false-truth seams. Commit `6ffce57a` let
`build_tile_presentation` copy an unacknowledged lifecycle fallback directly
into `TilePresentationState`; that duplicate acknowledgement path is deleted.
The initial G5 cutover `56d1cc0a` then preserved a requested L2 identity over
physically sampled L4 values; floor payloads now keep requested geometry and
actual sampled LOD as separate typed facts.

Those two correctness gates are now closed at their existing owners. One
`tiledPayloadResident` seam reports page-backed ancestor resolution,
source-anchored native chunks, and classic residency; the session treats only
that physical proof as exempt from cold admission caps. A complete already-
resident coarse set therefore rebinds in one presentation transaction instead
of streaming tile by tile after pan/zoom. Hidden warming uses the same seam,
and first-display rough level evidence remains visible correctness work until
the first frame commits instead of parking on the optional histogram lane.
The five-node live window-shift ring passes in 4.54 s. An integrated
`FrameSession → FramePipelineEffects → DisplayCommitter → VisPy` gate
keeps interaction active, defeats a one-item/one-byte cold cap, and commits all
three L2 targets through resident L4 ancestors with no removals and zero
uploads before the stop edge. Its physically cold same-source control still
defers. This deleted a duplicate unconditional interaction-time LOD gate; the
backend-residency predicate now owns the decision once. The field failure is
preserved at
`tests/artifacts/g5-resident-fallback-field-2026-07-16/`: several transitions
reported 100 visible tiles but only 78, 92, 85, 97, or 80 presented while
compatible L0/L1/L2 residency existed. Prefetch page claims/admission are GUI-owned;
workers return checked pages only, and success, stale, cancellation, partial
fanout, and teardown all release claims and wake the standing replan path.

The follow-up saved-session pan/zoom reproduced two remaining ways that warm
coverage could still stream in. First, a mixed presentation transaction could
contain a complete set of physically free coarse rebinds plus one cold upload;
the interaction guard correctly deferred the cold member but necessarily
deferred the whole transaction with it. The admission owner now emits the
complete physically free cohort as its own transaction and leaves cold exact
work queued. Second, the obsolete `pending_tiles` queue still admitted
coverage-margin misses even though the frame pipeline, not that queue, owns
production scheduling. A dormant shell entry was then treated as already
known when it entered the required viewport, suppressing the immediate
cache/residency lookup. Viewport admission is now lifecycle-required only,
coverage warming remains prefetch-owned, and fossil queue membership has no
authority over retarget additions. Focused gates require all resident members
to cross before any cold member and require a dormant coverage hint to be
rechecked when it becomes visible. Full removal of the remaining legacy
`pending_tiles` state is still a one-owner cleanup before final acceptance.

The broader memory-stress gate initially appeared to show a roughly 216 MiB
G5 residency increase, but reproduced with a roughly 178 MiB increase on
`main`. Its baseline was taken before the one-time Qt/PyQtGraph window/backend
allocation. The corrected gate takes its RSS baseline after backend
initialization and separately asserts the deterministic owners: in the failing
scenario the logical page cache plateaued at 129 pages / 1,115,136 bytes with
zero pending claims and the display cache plateaued at 16 entries / 4,194,304
bytes; repeated viewport walks did not grow either owner. This correction does
not widen a cache budget or excuse unbounded RSS growth.

That live ring also caught one last route split rather than being weakened for
the new semantic-read contract. Cold reduced-target evaluation planned and
materialized at local `(0, 0)` while its payload advertised a shifted native
source anchor. The stored values could therefore be offset from their page
keys and draw geometry by the window origin. Cold-target evaluation now asks
the same session source-origin owner used by ladder materialization; the live
factor-two gate asserts shifted native coverage and samples recognizable bins
through the exact page plan.

Presentation sampling and scientific reads are now separate typed APIs.
`PageBackedPresentation.sample_presented_*` maps native coordinates through
clipped/nonuniform target bins and actual ancestor pages exactly, but those
reduced values do not thereby become native semantic data. `TiledValueSource`
admits only explicit native `semantic_data` (and optional native histogram
data); exact ROI/measurement/export region demand falls through display pages
to the evaluator/cache. Preview, fallback, or reduced display pages can no
longer silently satisfy an exact scientific read.

Runtime JSONL now reports resident page counts by anisotropic reduction vector
and reducer family. Together with requested-to-actual physical binding rows,
this distinguishes “already resident but not rebound” from “not resident”
without inferring state from logical acknowledgement.

`PageResolution.scale/offset` remains the nominal aligned-grid transform used
for ancestry diagnostics. One affine transform cannot represent a clipped
leading/trailing bin: the real factor-two target through a factor-eight
ancestor maps its first three-native-sample coarse bin to one third of a stored
sample, not the nominal one quarter. Both backends therefore consume the
canonical target and actual `SourceGridDrawBlock`s for exact boundary mapping;
the clipped VisPy gate proves the intentional divergence from the nominal
transform so a future simplification cannot reintroduce uniform stretching.

The dedicated real-Wayland gate
`tests/gpu_interaction/test_g5_page_fallback.py` drives missing fine → pinned
coarse → fine arrival → fine eviction → pinned coarse on a real VisPy
surface and samples the framebuffer at every state. Fine resolution and
eviction fallback take 35.5 ms and 23.3 ms with zero uploads; binding
generations are coarse 1, fine 2, fallback 1. Its fault injection hides the
actual GL visual and produces a genuinely black framebuffer, proving the
never-black oracle can fail. The green run and PNG/JSON evidence are under
`tests/artifacts/g5-never-black-real-gl-2026-07-17/`.

The ladder-side target planner is also now explicit:

- `render.lod.plan_lod_page_targets` is a Qt-free deterministic transform
  from content identity, native source rect, anisotropic reduction, and
  uniform stored-page shape to canonical `DataChunkKey` targets;
- factor-2 windows starting at 101 and 102 share the aligned interior page
  and keep both clipped boundaries distinct; desired mean-family identity
  stays separate from a coarser physical resolution;
- the planner is attached to `DisplayTilePayload`, ladder materialization,
  retained/floor queries, and backend resolution. Requested target geometry
  stays distinct from actual sampled residency, so a coarse fallback cannot be
  relabeled as its fine target.

The pure boundary-geometry half of that atomic slice is now implemented:

- `partition_source_grid_pages` groups globally reduced samples into uniform
  stored-page classes while retaining the exact native-source rectangle of
  every sample for draw construction;
- clipped first/last factor-2 bins retain width one, aligned interiors retain
  width two, and the flattened draw spans cover every valid source coordinate
  exactly once;
- shifted windows share only a complete aligned interior page identity and
  attach byte-identical values to it. Boundary page identities remain distinct.
- per-sample spans coalesce into at most a 3-by-3 Cartesian set of uniform
  `SourceGridDrawBlock`s; aligned interior pages are one block, while stored
  and native-source coverage both remain exactly once.

These values and spans now travel through the page-backed ladder/cache and
payload contract. The stale live name `RungMaterializationRequest` is removed;
`LodPageMaterializationRequest` carries the source coverage, canonical plans,
and only the claims newly owned by one request. Named asymmetric gates now
compare `(1, 2)` with `(2, 1)` across conversion, bin footprints, page shapes,
and componentwise ancestry. PyQtGraph and VisPy share direct-oracle parity for
clipped pages, anisotropy, `mean_abs` magnitude, and phase cancellation; a
complex-mean page cannot resolve a magnitude target. Current focused evidence
includes 70 pure geometry/key/page-table tests, 93 route/reducer/model tests,
236 backend/offscreen page/physical/window/floor tests, and 129
producer/model/semantic tests, the 18-node prefetch file, and the dedicated
real-GL node above. The former order-dependent prefetch red was a circular test
fixture: forged visible-busy also set the governor's speculative quota to zero
while the test waited for that parked body before releasing busy. It now uses
real bounded visible work and its actual drain; production ownership was not
changed and no timeout was widened. Remaining G5 work is broader suite/stress
cleanup, the required live real-data interaction-convergence run, full
real-Wayland `gpu_interaction` baseline comparison, and the onscreen workflow
on both backends. The queue row does not move to Done until every exit gate
above is green.

## Rejected shortcuts

- backend-private tuple keys as the permanent pyramid identity;
- per-fragment ancestor walks or sampled-zero missingness;
- active exact chunks as a substitute for owner-scoped coarse pins;
- pinning every coarse page forever;
- acknowledging a coarse physical fallback as the fine target;
- wrapping a `DataChunkKey` in the legacy whole-tile residency tuple;
- treating `max_texture_size` as a total VRAM/page-count budget (pin-pressure
  denial gates use an explicit byte budget; without one the atlas may grow);
- treating previous-screen retention as data-level coverage;
- renaming window-origin reductions without changing their bins;
- stamping source-grid target keys onto window-origin reduced values;
- drawing a clipped boundary page as one uniformly stretched quad when its
  first/last partial bins have different native widths;
- reusing clipped boundary values across different valid footprints;
- preserving `PyramidLevelKey` behind a compatibility shim;
- deferring anisotropy or collapsing complex modes to one reducer family;
- changing scheduler or `prepare_rung` ownership to land an identity,
  reduction-grid, and residency feature.
