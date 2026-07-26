# ADR 0060: One operation definition, pluggable runtimes, discovered shapes

- **Status:** Accepted and implemented (2026-07-26); Bundles A–E and the
  closing discoverability/authoring phase are complete. Refines the
  operation-extensibility model
  established by the plugin-ops work (ADR-less, `docs/plugin-operations.md`) and
  the custom-operations program that landed 2026-07-24 (`eaaea1cf`). Supersedes
  that program's decision to *refuse* shape-changing user operations.
- **Number:** 0060 was free at `6ad55232`. Parallel worktrees have collided on
  ADR numbers before; renumber on integration if 0060 is taken.
- **Roadmap:** [operations-as-a-platform](../proposals/operations-as-a-platform.md).

## Context

Four properties of the current operation layer are in tension with the product
we want, and each of them is a decision we made rather than a constraint we
found.

1. **A built-in operation is a Python dataclass; a user operation is a JSON
   wrapper.** Two formats mean a user cannot start from a built-in, and we
   cannot show them how anything works.
2. **A "runtime" is implied, not modelled.** In-process Python is assumed for
   user ops; a subprocess bridge exists but is welded to one binary
   (`run_bart`) and one array format (cfl).
3. **The environment an operation needs is unmodelled.** We import user code
   into ArrayScope's interpreter and we read `BART_TOOLBOX_PATH` as a special
   case.
4. **Output shape and dtype must be declared before execution**, so
   `changes_shape: true` in a user wrapper is rejected: the only adapter a
   wrapper could get is the identity, and an identity adapter would lie to
   `evaluate_shape`, which the tile/LOD planner trusts.

Point 4 is the load-bearing one. The planner genuinely needs a shape *before* it
runs anything — that is not negotiable. What *was* negotiable, and wrong, is the
conclusion that the author must therefore supply it.

## Decision

### 1. One definition format for every operation

Every operation — built-in, in-tree pack, entry-point plugin, user — is
describable by a single declarative *definition*: identity (id, label,
description, group, icon), an interface (requires-axis, parameters with kind,
default, bounds, step, description, and input slots), and a *body* that
names a runtime plus its runtime-specific fields.

Built-ins keep their Python dataclass implementations — this is not a rewrite of
the engine. What changes is that each one can **render** its definition in the
shared format, and that any definition can be **duplicated into a user
operation**. The definition is the lingua franca for reading and copying, not a
new execution mechanism.

Consequence: the manager shows one editor, over one schema, for everything. A
second parameter editor (today's import dialog) is deleted rather than aligned.

### 2. Runtimes are a named, extensible axis of the definition

A definition's body names a runtime:

- `python-inprocess` — import a module, call a callable (today's user-op path).
- `python-environment` — the same callable, executed by a named interpreter
  outside our process, over an array handoff.
- `command` — a command template with placeholders for inputs, outputs, and each
  declared parameter, plus an array-handoff format (cfl / npy / nifti / raw).
  This is `run_bart` generalized: the concurrent pipe draining, timeout, and
  SIGTERM→SIGKILL cancellation are runtime behaviour, not BART behaviour.
- `julia`, `matlab` — command runtimes with a known interpreter shape. They cost
  schema, not machinery.

A command template is **never** handed to a shell unless the definition opts in
explicitly; argument vectors are built by explicit tokenization, and quoting is
tested.

### 3. Execution environments are their own record, referenced by id

An *environment* (name, interpreter / conda env / venv path, working directory,
environment variables) is persisted alongside operations and referenced by
definitions. It is edited in the manager behind an *Advanced* disclosure so the
common case stays invisible.

`BART_TOOLBOX_PATH` stops being special-cased and becomes an environment record
like any other. An operation that cannot resolve its environment is reported as
a *problem* (the existing `user_operation_problems` channel) and is offered in
the UI as unavailable — never as a silent failure at apply time.

### 4. Shape, dtype, and windowability are discovered and adjudicated, not declared

The planner keeps asking for a shape before execution. We answer it from a
**probe-and-cache** step instead of from an author's promise:

- Probe the operation on small representative slabs, recording
  `(input shape, dtype, parameters) → (output shape, dtype)`.
- Fit the result against a small set of known **shape rules** (identity,
  axis-reduced, axis-scaled, fixed-size, per-axis pad/crop, permutation). A fit lets us
  predict; no fit means *unpredictable*, which means whole-array evaluation with
  `cache_stage` — an honest cost class, not a refusal.
- Cache the verdict keyed by operation id, parameters, input signature, and
  source identity (file mtime, or the command template text), so editing a
  linked file or a command re-probes.
- **Unify with region conformance.** `plugin_conformance` already decides
  windowability empirically and demotes false claims to OPAQUE. Discovery and
  conformance become one adjudication with one cache and one downgrade path: an
  operation is characterized, not interrogated.

Therefore the `changes_shape: true` rejection is removed. A user operation may
pad, resample, or permute.

### 5. Native first

An external backend is justified only by capability it uniquely provides.
Arithmetic that NumPy expresses directly is implemented natively, in-process,
dtype-preserving. Accelerations (numba or otherwise) are landed only with a
before/after measurement on a representative array, always behind a NumPy
fallback.

### 6. Auxiliary inputs are declarative sources, not a graph rewrite

An operation definition may declare named `input_slots` with a label,
description, and accepted source kinds. The primary operation chain remains
linear. Each slot binding is recipe data and resolves at the window boundary to
an immutable source snapshot:

- `dimension-set` records one fixed-index selection from the current base
  array;
- `open-document` resolves through the existing Compare group;
- `roi-mask` rasterizes one ROI to a 2-D boolean image-plane mask;
- `roi-coordinates` supplies one ROI as an `N×2` float64 `(x, y)` array; and
- `saved-array` memory-maps one `.npy` array.

One slot binds exactly one source. A recipe may bind several slots, but a slot
does not implicitly collect multiple ROIs. Bounding boxes are not a separate
representation: a rectangle ROI is either its boolean mask or its four corner
coordinates.

Multi-input operations begin OPAQUE and demand the whole primary array and
whole resolved slot arrays. A region capability is not inherited from the
single-input path. It must be earned later by a workload-specific empirical
mapping.

The characterization cache key includes every slot's serialized binding,
resolved shape, dtype, and source identity. This prevents different
documents/ROIs/files from sharing a shape verdict. ROI source identity includes
geometry but excludes label and color: moving or editing the referenced ROI
invalidates its dependent operation, while renaming, recoloring, or editing an
unrelated ROI does not. Deleting an ROI, closing a bound document, or removing
a saved file rebuilds the step as disabled with an `unavailable_reason`.
Recipe load follows the same disabled/quarantine path rather than failing the
load or deferring a crash until apply.

Resolved slots are process-local snapshots and are deliberately not serialized.
The binding/resolution split is permanent ownership, not temporary scaffolding.
The current `python-environment` runtime does not yet transport extra arrays and
therefore advertises a definition-level unavailable reason when slots are
declared; adding multi-file transport there would delete that explicit
limitation.

## Consequences

**Good**

- A user can read any operation, duplicate it, and adapt it — including
  retargeting a BART invocation at a different BART tool, which is the actual
  reason to want a BART bridge.
- Research code that only runs in its own environment becomes usable without us
  vendoring anyone's dependencies.
- Shape-changing operations work, so the toolbox can contain pad/resample/permute
  and users are not told "unsupported" for something obviously reasonable.
- Cost signals become honest per operation instead of per backend: a native
  pointwise op stops paying a complex128 or complex64 promotion it never needed.

**Costs and risks**

- **Probing is real work.** It runs the operation, possibly out of process, before
  the planner can proceed. It must be small, cached, and bounded, and the first
  use of an operation is measurably slower than a declared one. If a probe cannot
  be made cheap for a given runtime, that runtime's operations start as
  unpredictable/whole-array rather than blocking on a probe.
- **A misfit shape rule is a correctness hazard.** Extrapolating from two slabs
  can be wrong. Mitigation: only a small set of well-understood rules is
  extrapolated, a mismatch found later invalidates the cache and demotes to
  unpredictable, and the smoke harness asserts predicted-vs-actual on every
  registered operation.
- **Subprocess runtimes are an execution surface.** Command templates run
  binaries with user-supplied arguments. They are user-authored and
  user-triggered, like a shell alias, but recipes travel: a recipe referencing a
  command operation must not silently execute an unfamiliar command on load. The
  loading rule is that a command operation arriving from an imported recipe is
  registered as *unavailable until reviewed* in the manager, never auto-run.
- **Environments drift.** A named conda env can vanish or change. That surfaces
  through the problems channel, not as a crash.
- More definition surface means more to keep coherent; mitigated by there being
  exactly one editor and one schema.
- Open-document and dimension-set bindings use session-local identities. A
  recipe restored in a different session loads safely but disabled until the
  user chooses replacement sources.

## Alternatives considered

- **Keep declaration, add a shape-expression DSL to wrappers.** Rejected: it
  pushes a second language onto users to state something we can observe, and a
  wrong expression is as damaging as a wrong adapter.
- **Always evaluate whole-array for anything non-built-in.** Rejected as the
  default: it discards tiling/LOD for every user op, which is the difference
  between usable and unusable on large data. It remains the honest *fallback*
  for unpredictable ops.
- **Vendor sigpy/BART-style functionality instead of bridging.** Partially
  adopted: we implement natively what is trivial (phase 1) and bridge only what
  is not. Vendoring a full recon toolbox is out of scope.
- **Run user Python in a sandbox.** Out of scope: user operations are the user's
  own code, like a plugin. The reviewed-before-run rule for *imported recipes* is
  the boundary that matters.
