# Operations as a platform — current state, vision, and the road there

**Status:** Implemented (2026-07-26). Successor to the "custom operations
first-class" program that landed 2026-07-24 (`eaaea1cf`). That program built the
*plumbing* — a library, a manager, anchored popups, a user-op wrapper format.
This proposal is about the part it did not do: making the operation set itself
worth having, and making every operation something a user can read, copy, and
change.

Decisions that need to be pinned before code are in
[ADR 0060](../decisions/0060-operation-definitions-runtimes-and-discovered-shapes.md).

## Why this document exists

The first program was scoped as "custom operations should be first-class". It
delivered that literally: you *can* add a Python function, it *does* appear in
the menus, and it *does* round-trip through recipes. Then we looked at the
result as a user rather than as an implementer, and three things were obvious:

1. **The default set is thin, and what we did add came from the wrong places.**
   Every operation we exposed through sigpy and most of what we exposed through
   BART is arithmetic NumPy does in one line — while the reasons anyone installs
   sigpy or BART remain unexposed.
2. **System operations are opaque.** A user cannot see how `mean` is defined,
   cannot copy `sigpy:soft_thresh` as a starting point, cannot adapt a BART
   invocation to a different BART tool. The manager can hide and regroup system
   ops; it cannot teach with them.
3. **We built a general subprocess bridge and then hid it.** `run_bart(argv,
   ...)` is already "array in → argv → array out". Nothing exposes that.

Everything below follows from those three, plus a fourth we imposed on
ourselves: user operations may not change shape, because a wrapper cannot
declare an output-shape adapter. That restriction was a choice, and it is the
wrong one — see phase 4.

## Where we are (2026-07-26)

### What exists and is good

- **Three registration tiers, one execution path.** Built-ins
  (`operations/pipeline.py` dataclasses in `OPERATION_REGISTRY`), in-tree packs
  (`operations/packs/`), entry-point plugins, and user ops (`user:` namespace)
  all materialize through `PluginOperation`/the built-in dataclass and share
  recipe round-tripping.
- **A real capability model.** `OperationCapabilities` (kind, blocking axes,
  chunkable axes, `cache_stage`, `temp_multiplier`, `lod_commuting`) is what
  lets the evaluator plan tiles, stages, and LOD. Tier-2 "windowable" claims are
  *adjudicated*, not trusted: `plugin_conformance.verify_region_conformance`
  property-tests `fn(whole)[region] == fn(whole[region])` and silently demotes a
  false claim to OPAQUE.
- **Qt-free parameter forms.** `operations/parameter_forms.py` models fields,
  bounds, read-only derived values, and cross-field adjustment (crop's
  start/stop nudge each other) with no widget dependency.
- **A configuration layer mirroring the colormap library.** `operations/library.py`
  persists group layout, per-op order, hidden ops, and the "Common" pin set
  next to the session config, notifies listeners, and owns user-op loading with
  a no-crash guarantee plus a retrievable problem list.
- **Anchored, non-modal add/edit flows** and a manager dialog with drag
  regrouping, hide-not-delete, and reset-to-defaults.
- **An all-ops smoke harness.** `tests/operations/test_all_operations_smoke.py`
  applies every registered op with its default form on float32 and complex64,
  so a declared-but-unhandled parameter cannot ship again.

### What is wrong

#### The default set is thin

Nineteen operations are registered today. The everyday array/MRI toolbox is
mostly absent: **magnitude, phase, real, imag, log-magnitude** (arguably the
single most-used display transform in MRI), normalize, scale/offset, power,
clip, std/var, median, percentile, pad, transpose/permute, squeeze, difference/
gradient, cumulative sum, masking. A user's first ten minutes hit these, not
`combine_real_imag`.

#### What we did add is mis-sourced, and the sourcing has a measured cost

Every sigpy op we expose is NumPy-trivial, and the wrapper pays for the detour:

| Exposed | What it is | Cost of the detour |
|---|---|---|
| `sigpy:soft_thresh`, `sigpy:hard_thresh` | `sign(x)·max(|x|−λ, 0)` | sigpy promotes to **complex128 unconditionally** and we narrow back, so a float32 input round-trips float32 → complex128 → float32 (8× the working set for a pointwise map) |
| `sigpy:resize` | centred zero-pad / centre-crop | `np.pad` / slicing |
| `sigpy:circshift` | `np.roll` | an import and a copy |
| `sigpy:downsample`, `sigpy:upsample` | strided decimation / zero-insertion | slicing; and both are **restricted to integer factors**, which is not what "resample" means to a user |

BART is worse, because the handoff is a filesystem and a process:

| Exposed | What it is | Cost of the detour |
|---|---|---|
| `bart:fft`, `bart:ifft` | what the built-in centred FFT already does | subprocess + 2 cfl temp files |
| `bart:scale`, `bart:spow` | `x·k`, `x**p` | subprocess + 2 cfl temp files |
| `bart:cabs`, `bart:carg` | `np.abs`, `np.angle` | ditto |
| `bart:normalize` | `x / ‖x‖_axis` | ditto |
| `bart:std`, `bart:var` | `np.std/var(ddof=1)` | ditto |

cfl is a complex64 container, so **every** BART op promotes real input on write
and returns complex64 — an honest cost signal for a real recon, an absurd one
for `x·2`. On a 4 GiB volume these ops write the entire array to disk twice to
perform a multiplication.

Meanwhile the tools people actually want a BART/sigpy bridge *for* — `pics`,
`ecalib`, `nlinv`, `pocsense`, NUFFT with a trajectory, sigpy's iterative
`App`s, `EspiritCalib`, wavelet pairs — are all unexposed, because they need
more than one input array or non-array metadata.

**Caution for whoever picks this up:** the commit that added the BART ops
reported verifying semantics against a runnable binary, but no retained output
made that claim independently reproducible; its guarded tests concerned
now-removed unary wrappers, not today's `ecalib`, `walsh`, and `pics`
definitions. The fake-`bart` tests remain load-bearing integration oracles only.
Current numeric evidence comes from
[`tools/validate_bart_numerics.py`](../../tools/validate_bart_numerics.py) and
the [2026-07-26 real-toolbox review](../reviews/2026-07-26-bart-numeric-validation.md).
That first real run found and fixed two semantic defects rather than weakening
the gate: PICS needed `-S` to restore output scale, and BART `walsh` returns
packed calibration covariance for `ecaltwo`, not sensitivity maps.

#### System operations cannot teach

There is no "show me how this works" anywhere. A user who wants
`sigpy:soft_thresh` with a different threshold rule, or `bart:fft` pointed at
`bart nufft`, has to leave the app, read our source, and hand-author a wrapper.
The colormap system already solved the analogous problem: a built-in colormap
can be edited, which saves a user override that shadows it, and Reset restores
the original. Operations have hide/regroup but no edit-by-copy.

#### The import dialog is a dead end

`_OperationImportDialog` is a separate window with:

- a parameter table of **4 columns** (name/kind/default/description) against the
  manager's **6** (name/kind/default/min/max/step) — so a user op created
  through the import path cannot express the bounds a built-in has, and the two
  tables disagree about what a parameter *is*;
- a default height of 460 px against a 430 px minimum-size hint, which leaves
  room for **two** parameter rows and clips the third mid-glyph;
- **no UI-gallery scenario at all**, which is exactly why these defects
  survived a screenshot review pass;
- a `Function` picker that was visually indistinguishable from a read-only text
  field — because the app stylesheet styled `QComboBox::drop-down`, which makes
  Qt stop drawing the native arrow. That was app-wide (every colormap-kind,
  group, and axis picker) and is **fixed** as of `682cf9ef`.

The structural problem is not the widget list, it is that a second editor exists
at all. Two editors for one concept guarantees drift.

#### We hid our own extensibility

`run_bart(argv, ...)` already accepts an arbitrary argument vector, writes the
input as cfl, runs a binary with concurrent pipe draining and a
SIGTERM→SIGKILL cancellation path, and reads the output back. That is a
general external-tool bridge with one hardcoded binary and one hardcoded array
format. Nothing lets a user say "run `mytool --sigma {sigma} in out`".

#### There is no execution-environment concept

Python user ops are imported into *ArrayScope's* interpreter. That fails the
moment a user's script needs a package we do not ship, which is the common case
for research code — their function lives in a conda env or venv with its own
NumPy/torch/sigpy. External binaries need environment variables
(`BART_TOOLBOX_PATH` being the obvious one) and sometimes a working directory.
None of this is modelled.

#### Shape-changing user ops are refused

`_spec_from_wrapper_file` raises on `changes_shape: true`, because a wrapper
JSON has no way to express an output-shape function, and the default identity
adapter would lie to `evaluate_shape`. The reasoning was sound; the conclusion
was not. See phase 4.

#### Single input, no ROIs, no acceleration story

Ops are `f(array) -> array`. There is no second input, no ROI/mask input, and no
measured native acceleration anywhere (numba is installed and unused by the op
layer).

## Where we want to be

A user should be able to say, in the app, without reading our source:

- *"Show me the log-magnitude of this."* — present by default, instant.
- *"Threshold this at 0.3."* — present by default, with a real slider range.
- *"Resample this axis to 0.7× and pad the other to 256."* — no integer-factor
  restriction, no "shape-changing operations are unsupported".
- *"How is the built-in mean defined? Give me a copy I can change."* — every op
  is readable; any op duplicates into an editable user op.
- *"Run my `denoise.py` — it only works in my `recon` conda env."* — named
  environments, chosen per op, advanced settings folded away by default.
- *"Run `bart pics -R W:7:0:0.01 -i 50` on this."* — the command line is visible
  and editable; the system-provided BART ops are examples to copy, not black
  boxes.
- *"Use this ROI as the mask argument."* / *"Take the coil maps from that other
  dataset."* — inputs beyond the primary array.

And throughout: the app should figure out what it can (signature, parameters,
shapes, dtypes, whether an op is windowable) and still show every inferred
value as an editable field — the colormap-manager principle, applied to
computation.

## The road there

Six phases. Each is a bundle of related work that lands as several bounded
commits, not one mega-commit. Phases 1–2 are independent of each other; 3 builds
on 2's editor; 4 is engine-side and unblocks 5.

### Phase 1 — A native toolbox, and demote the packs

**Goal:** the everyday set exists, is NumPy-native, dtype-honest, and
region-capable where it truly is pointwise; external backends keep only what
genuinely needs them.

**Bundle A implemented 2026-07-26 on `claude/native-ops-bundle-a`.** The Numba
verdict and exact measurements are recorded in the
[dated review](../reviews/2026-07-26-native-operations-numba.md): normalize
landed behind the shared lazy runtime; faster log-magnitude and soft-threshold
kernels were rejected because mixed NumPy/Numba region paths failed the exact
ELEMENTWISE oracle by 1–2 ULP.

- Implement natively (built-in tier, `pipeline.py` + registry): magnitude, phase,
  real, imag, log-magnitude (with an epsilon parameter), scale, offset, power,
  normalize (axis or global), clip, soft/hard threshold, roll/circshift, pad
  (centred and asymmetric), resample (**fractional**, order-selectable), median,
  percentile, std, var, transpose/permute, squeeze, difference/gradient,
  cumulative sum.
- Pointwise ones declare and *pass* Tier-2 region conformance; reductions and
  resamplers stay OPAQUE with exact shape adapters.
- Remove the redundant pack ops (all six BART arithmetic ops, `bart:fft`,
  `bart:ifft`, and the five NumPy-trivial sigpy ops), each with a
  [graveyard](../graveyard.md) row naming the native replacement.
- **Numba, measured only.** Prior work found numba is mostly *not* the answer in
  this codebase (strided-copy and scheduling bound), with a real win on
  window+LUT. So: for each candidate (fused magnitude+log, complex soft-threshold,
  normalize) show a before/after on a representative array; land the
  acceleration only where the number justifies it, always behind a NumPy
  fallback, never as a blanket policy.

**Exit gate:** the all-ops smoke harness covers the new set on float32/complex64;
value-correctness tests against NumPy references; a written note recording which
numba candidates measured a win and which did not.

### Phase 2 — One definition format, one editor, every op a template

**Goal:** any operation can be read; any operation can be duplicated into an
editable copy; there is exactly one parameter editor in the product.

- **Definition export.** Every registered op — built-in, pack, plugin — can
  render its definition into the user-op wrapper schema (label, description,
  group, icon, requires-axis, parameters with bounds, runtime, source
  reference). System ops display it read-only.
- **Duplicate to edit.** "Duplicate" on any op writes a `user:` copy pre-filled
  from that definition, including a copy of the code where the source is ours to
  copy, or a command template where it is a command. This is how a user learns
  the format, and how they retarget `bart:fft` at `bart nufft`.
- **Fold the import dialog into the manager.** Delete `_OperationImportDialog`.
  "New" and "Duplicate" create an entry and select it; the manager's own editor
  is the single editor, extended with the fields the import panel had (source
  file + callable picker, copy-vs-link storage mode) and the parameter columns it
  lacked. Parity with built-ins by construction: one table, one schema.
- Parameter editing gains the full metadata set (min/max/step/description) plus
  read-only derived rows, so a user op's popup can look exactly like crop's.
- **Gallery coverage for every state** of the manager editor: system read-only,
  user editable, new-empty, duplicate-prefilled, source picker, problems.

**Exit gate:** no second parameter table anywhere; a duplicated built-in produces
a working user op whose popup is field-identical to the original; gallery shots
for each editor state reviewed.

### Phase 3 — Runtimes and environments

**Goal:** the subprocess bridge we already have becomes a first-class,
user-editable runtime, and ops can name the environment they need.

- **Command-template runtime.** Generalize `run_bart` into a runtime that takes
  a command template (`bart pics -S -i {iters} {in} {out}`), an array-handoff
  format (cfl / npy / nifti / raw), a timeout, and the existing concurrent-drain
  and cancellation behaviour. Placeholders cover inputs, outputs, and every
  declared parameter. Quoting/escaping is explicit and tested; a template is
  never passed through a shell unless the user opts in.
- **Execution environments** as their own small persisted record (name,
  interpreter or conda env or venv path, working directory, environment
  variables), reusable across ops, edited in the manager, and surfaced under an
  *Advanced* disclosure so the common case stays one line. `BART_TOOLBOX_PATH`
  becomes an environment record instead of a special case.
- **Out-of-process Python.** A python op may run in a named environment instead
  of ArrayScope's interpreter, using the same array handoff as the command
  runtime. Same wrapper schema, one extra field.
- Re-express the surviving BART ops as command-template definitions, and ship a
  small set of *genuinely BART-shaped* examples users can copy.
- Reserved `julia` / `matlab` runtimes become concrete: both are command
  templates with a known interpreter, so they cost schema, not machinery.

**Exit gate:** an op defined purely by a command template runs end-to-end with a
fake binary in tests; a python op runs in a non-default interpreter; cancellation
and timeout are proven for both; no secrets or paths leak into recipes that
should not travel.

### Phase 4 — Discovered shapes and dtypes — DONE 2026-07-26

**Goal:** stop asking authors to declare what we can observe, and lift the
shape-changing restriction.

The evaluator needs `output_shape`/`output_dtype` *before* running an op, which
is why declaration exists. But we can discover the relation instead of trusting
it:

- **Probe once, cache the relation.** Run the op on a small representative slab
  (or, where that is unsafe, on the first real call) and record
  `(input shape, dtype, params) → (output shape, dtype)`, keyed also by source
  mtime / command template so an edit invalidates it. Persist nothing that a
  code change could stale.
- **Infer the shape *rule*, conservatively.** Probing two or three slab sizes
  distinguishes the common cases (identity, axis-reduced, axis-scaled,
  fixed-size) and lets us extrapolate; anything that does not fit a known rule
  is treated as unpredictable, which means whole-array evaluation with
  `cache_stage` — honest, not refused.
- **Fold in region conformance.** The Tier-2 harness already decides windowability
  empirically. Discovery and conformance become one adjudication step with one
  cache and one honest downgrade path.
- Then delete the `changes_shape: true` rejection, and let a user op pad,
  resample, or permute.

**Exit gate:** a shape-changing user op works through tiles/LOD without lying to
the planner; probe cost is measured and bounded; an edited linked file
re-probes; a deliberately inconsistent op is caught and downgraded rather than
corrupting a view.

Landed in `2e2c3a7c`, `9208cf94`, `36a2c8fa`, and `58e7f3aa`; measurements and the
deliberate whole-array misfit cost are recorded in the
[characterization cost review](../reviews/2026-07-26-operation-characterization-cost.md).

### Phase 5 — More than one input

**Goal:** the ops that need a second array (coil maps, masks, references) become
expressible; start simple.

- **Input slots.** An op may declare extra inputs, each bound to one of: another
  dimension set of the same array, another open document, an ROI (as mask or as
  coordinates), or a saved array file. Slots appear in the parameter popup and
  the manager.
- **Engine shape.** The operation chain stays linear for the primary input;
  extra inputs are resolved as *sources*, and multi-input ops are OPAQUE with
  whole-array demand at first. Region-aware multi-input is a later refinement,
  justified by a real workload rather than by symmetry.
- **ROIs as inputs** deserve their own small design: one ROI vs many, mask vs
  bounding box vs coordinates, and what happens when the ROI moves (an ROI edit
  must invalidate exactly what it should).
- This unlocks the BART/sigpy tools that motivated the bridge in the first
  place: `pics` with sensitivity maps, `ecalib`, NUFFT with a trajectory.

**Exit gate:** one real two-input recon runs end-to-end from the UI; ROI-driven
invalidation is proven to be neither stale nor over-eager.

Landed in Bundle E. Definitions and the single manager/form model now carry
dimension-set, Compare-document, one-ROI mask/coordinates, and saved-array
bindings. Multi-input characterization keys include binding plus resolved
shape/dtype/source identity and are always OPAQUE. `bart:pics` binds sensitivity
maps and runs from the UI through the shared cfl runtime; the fake BART oracle
pins primary/sensitivity argv order, cancellation, timeout, and cleanup.
Evaluation-counter tests prove a referenced ROI geometry edit recomputes once,
while unrelated geometry and referenced label/color edits reuse the incumbent
result. Missing sources restore as disabled/unavailable recipe steps.

### Phase 6 — Discoverability and polish

- Search/filter in the add popup once the set is large; the "More…" partition
  reflects the richer taxonomy.
- A "write your own operation" walkthrough in the docs, built from the manager's
  own duplicate flow.
- Gallery scenarios for every surface introduced by phases 3–5 (command
  template editor, environments, slots), reviewed as screenshots — the import
  dialog's defects existing *through* a screenshot pass is the reason this is a
  standing requirement, not a closing chore.

Bundle E supplied reviewed dark/light captures for the slot-bearing parameters
popup, manager slot editor, and unresolved-slot state. The remaining phase-6
work is the walkthrough and the already-noted add-popup search/fold redesign;
the slot surfaces themselves no longer owe gallery coverage.

## How the work is bundled

Five bundles, each landing as a series of bounded commits with real messages.
Bundle order respects the dependencies above; A and B may run in parallel, C
after B, D after A, E last.

| Bundle | Phase | Scope |
|---|---|---|
| **A — Native toolbox** | 1 | native ops + pack demotion + graveyard rows + measured numba verdict |
| **B — One editor** | 2 | definition export, duplicate-to-edit, fold in the import dialog, parameter parity, gallery states |
| **C — Runtimes** | 3 | command-template runtime, execution environments, out-of-process python, BART re-expression |
| **D — Discovery** | 4 | probe-based shape/dtype + unified conformance, lift the shape restriction |
| **E — Inputs** | 5 | slot model design + one real multi-input op + ROI input slice |

Phase 6 rides along: every bundle that adds a surface adds its gallery scenario
in the same bundle.

## Standing rules for this program

1. **Native first.** An external backend is justified only by what it uniquely
   provides. "It exists in that library" is not a reason to shell out.
2. **Every inferred value is shown and editable.** Introspection fills the form;
   the user overrides it. Never infer silently.
3. **One editor per concept.** A second parameter table is a bug.
4. **Adjudicate, do not trust.** Shape, dtype, and windowability claims are
   verified empirically and downgraded honestly, never accepted on assertion.
5. **Measure accelerations.** No numba, no vectorization rewrite, no caching
   without a before/after on a representative array.
6. **Screenshot every new surface.** A UI change that no gallery scenario can
   show is not reviewable.
