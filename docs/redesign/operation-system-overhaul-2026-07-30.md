# Operation system overhaul — program dossier (2026-07-30)

Successor scrutiny pass over the operations platform that closed as queue
row 11 (ADR 0060). Four research sweeps (lifecycle, manager parity,
dimension semantics, external toolboxes) established that the platform's
core — one definition format, pluggable runtimes, discovered shapes — is
sound, and that the reported gaps are defects and missing rungs *on top of*
it, several of them verified live crashes. This dossier is the program's
single design anchor; each bundle below records its verdict in place when
it lands.

## Verified defects (all reproduced in code, not inferred)

1. **Bricked definitions (Class A).** `library.duplicate_operation`'s
   command branch writes no `source` block; switching the runtime combo to
   Python then persists a wrapper that `_spec_from_wrapper_file` refuses
   (`library.py:513-514`). The manager's tree is built from the registry,
   so the file falls into a virtual "Problems" row whose
   `selected_operation_id()` is `None` — editor blanked, delete an early
   return. 18 distinct triggers land in the same dead end (bad JSON,
   deleted source file, renamed callable, schema drift, …). The raw-file
   primitives (`user_operation_wrapper`, `remove_user_operation`) work the
   whole time; only the UI cannot address them.
2. **Whole-listing failure (Class C).** A command op with a conda
   environment plus shell execution — all reachable from the manager —
   makes `availability()` raise uncaught through
   `current_unavailable_reason` (`plugins.py:209-213`), which takes down
   `all_operations()`: the manager cannot open, the palette raises, no
   in-app repair path exists.
3. **Axis-space conflation at the stage-cache boundary.**
   `PluginOperation.capabilities` declares `expands_request_axes` in
   *input* space; `planner.py:172` applies them to an *output*-space
   region. A rank-reducing plugin op (`fn=lambda a: a.mean(axis=1)` — an
   ADR-0060-supported `axis_reduced` rule) crashes `plan_slab`; a
   rank-increasing one silently under-expands, so scrubbing the new axis
   re-runs the whole opaque op per step (measured 6/6 invocations vs 0).
   Built-ins escape only because no rank-changing built-in sets
   `cache_stage`.
4. **Probe failure escapes the characterization funnel.** A
   shape-constrained op (NUFFT with a coord slot, fixed-input model)
   raises its own error on the 5×5×5 synthetic probes;
   `plugin_conformance.py:224-225` re-raises it verbatim, and
   `evaluate_shape` catches only `CharacterizationUnavailable`, so the
   exact-input fallback never engages. Invisible on small arrays (the
   probe *is* the real shape), fatal on real data.
5. **Unknown rank-changing op = `ValueError` at document construction.**
   `axis_info.output_axes_for_operation` falls through to `return
   tuple(axes)`; `axes_for_shape` then raises "axis metadata length must
   match shape length". `output_axes` is de-facto mandatory for rank
   change and only `PluginOperation` implements it (with duplicate
   "Dim N" labels).
6. **Cancellation built, never wired.** `run_array_command` implements
   token-polled SIGTERM→SIGKILL; `PluginOperation.apply_to_region`
   deletes the evaluation context that carries the token; no call site
   passes it. No external child is interruptible.
7. **cfl handoff rank drop.** `read_cfl` pops trailing singleton dims —
   undocumented, and it breaks characterization for any cfl op probed
   with a size-1 axis (probes preserve size-1 axes).
8. **`conda_env` locator tax.** Resolving to `conda run` costs a
   measured ~1.55 s per call (43× a direct interpreter) — paid per probe
   (7–9× per new signature) and per apply. Same class of cost
   `eb86e159` removed from tooling.
9. **Colormap manager silently swallows broken entries** (parse errors
   dropped, missing cmcrameri built-ins vanish), stores in an unscoped
   directory (ignores `-n <name>` scoping that operations honor), and
   has ~1/6 the gallery review surface of the operation manager.

## Capability gaps

- **Gap A (rank-increasing ops)** is *not* an engine limitation: the
  planner, stage cache, and slab evaluator already evaluate
  rank-increase byte-identically (measured), and characterization
  already fits `fixed_and_mapped` rules for it. What blocks it: defects
  3/4/5 above, no authoring affordance creates one, and the Qt window's
  per-dimension widget arrays are rank-immutable after construction
  (`file_reload.py:118-130` says so itself). The sibling-window
  precedent (`compare_launcher.open_difference_window`) sidesteps the
  widget problem entirely.
- **Gap B (apply over a subset of dims)** is a genuine absence at four
  layers: declaration (`requires_axis: bool` → one int; param kinds are
  int/float only), serialization (`item["axis"] = int(...)`), UI (one
  combo), execution (no broadcast concept). The engine's per-axis
  vocabulary (`blocking_axes`/`expands_request_axes`/`chunkable_axes`)
  is already sufficient to make it *cheap*: `applied_axes` maps directly
  onto it. The `dimension-set` slot already computes exactly this slab
  shape for auxiliary inputs.
- **Parameter kinds** are the single most repeated integration blocker:
  no string/enum/bool/path kind means no `mode="reflect"`, no wavelet
  name, no model path, no device selection — blocking scipy.ndimage,
  scikit-image, SigPy wavelets, and any torch pack.
- **The env-Python runtime** — the one you need for heavy toolboxes —
  is the one runtime that cannot take a second array (slots
  hard-blocked), cold-spawns per call and per probe, and has no
  progress/cancel transport. The highest-leverage long-term change is a
  warm worker speaking a multi-array in/out envelope with
  progress/cancel events; it simultaneously serves SigPy, torch,
  registration, and a future Jupyter bridge.
- **No entry-point group for sources/loaders**: adding a file format
  today means forking `arrayscope/io/` — contrary to the program goal.

## Manager parity (colormaps ↔ operations)

Neither side is uniformly better: operations win 13 rows (Problems
surface, read-only built-ins with visible affordance, delete
confirmation, discrete Restore, reset-all, open-folder, scoped storage,
live listener, group combo), colormaps win 8 (session Revert, kind
filter, inline group rename, Import/Export, Apply-and-use,
disable-with-reason, dirty-snapshot autosave). Concrete traps: the same
`folder_open` glyph means Import on one side and Open-folder on the
other; "Favorites" (drag target) vs "Common" (checkbox) for the same
concept; four different names for the colormap dialog across surfaces;
no colormap ever appears in the command palette. Three literal
copy-paste blocks (tree widget, listener block, layout persistence)
already exist — the convergence target is a shared library protocol +
one `UserLibraryManagerDialog` base, with each side keeping only its
editor pane. Dialog tests consolidate onto the base (net fewer tests).

## Bundles

- **R1 — Never bricked** (ADR 0062): guard `availability()`; problems
  channel carries `(path, message, raw payload, id?)` keyed by *path*;
  Problems rows selectable/editable/deletable in the manager; duplicate
  and runtime-toggle stop manufacturing sourceless wrappers; environment
  reference never silently cleared. Invariant: *every stored definition
  is always listable, editable, and deletable, no matter how broken.*
- **R2 — Engine axis-space + characterization correctness**: defects
  3–7 above; capability axes declared/translated at the
  `expand_region_axes` boundary in output space; probe failures funnel
  to `CharacterizationUnavailable`; `output_axes` synthesized safely for
  unknown ops; cancellation token threaded end-to-end; cfl rank guard.
- **R3 — Parameter kinds**: `string`/`enum`/`bool`/`path` kinds +
  `file` slot kind through registry, wrapper schema, recipes, forms,
  palette; direct-interpreter resolution for `conda_env` environments.
- **R4 — Manager convergence** (ADR 0063): **LANDED.** `app/user_library.py`
  is the one storage protocol (listeners, layout with a format header,
  hidden set, path-keyed problems, atomic write) and
  `ui/user_library_manager.py` the one dialog grammar; the button
  vocabulary is a tested data structure rather than a convention, which
  is what the previous "deliberately mirrors the colormap designer"
  docstring was and was not enough. Colormap storage is install-scoped
  with a one-time migration; a colormap file the loader rejects is
  recorded and repairable; a built-in whose package is missing is
  `(unavailable)`, not absent; built-ins are read-only behind an explicit
  Override. Operations gained Revert, a filter, inline group rename,
  Import/Export and Apply. Two dead buttons ("Add a parameter", "Add an
  input slot", measured rowCount 0 → 0) fixed. Colormaps are palette
  commands; the dialog has one name on every surface. The recurring trap:
  a dialog that listens to a library it also writes re-enters its own
  autosave through the rebuild.
- **R5 — Subset-of-dims (gap B)**: axis-set parameter kind,
  `applied_axes` → blocking/expands/chunkable mapping, recipe field,
  multi-axis form widget; generalize the dimension-set slot to a named
  axis selection.
- **R6 — Rank-increase enablement (gap A)**: **LANDED.** Gap A was never
  one gap: the engine was already right (R2), and what remained was a
  *presentation* rule plus two serialization defects. The rule: a
  rank-increasing operation applied from the UI routes its result to a
  compare-linked sibling window whose document is the origin's stack plus
  the new step, handed to the constructor — which is the only moment the
  per-dimension widget arrays are sized. No rank-growing step ever enters
  a live window's own stack, and the origin is untouched. `_commit_operation`
  turned out to be the single choke point all three add surfaces already
  funnelled through, so the routing has exactly one site; it predicts the
  output rank with `evaluate_shape` *before* mutating anything, declared
  shapes first and the exact input only for an op that needs discovering,
  and treats a failed prediction as "unknown" so no failure mode is
  swallowed. The reload handoff at `file_reload.py:118-130` generalized to
  `window/rank_handoff.py` for the same rank arriving any other way
  (re-enabled inherited step, loaded recipe), keyed on the widget arrays'
  *construction* capacity rather than the current rank — so a document that
  shrank and grew back stays in place. Two seams were live bugs
  independently of any of this: `view_state_from_mapping` built the state at
  the target rank before `for_shape` could migrate it (a `ValueError` in
  **both** directions, not just growth), and `recipe_from_mapping` walked
  `output_shape` directly, which is precisely what a discovered-shape op
  cannot answer. The proof op is `split_axis` — one axis into blocks plus a
  new interleave axis — and it ships **exact** region algebra rather than
  the sanctioned `expands_request` fallback, because the two requests the
  interactive engine actually makes (scrub the new axis, scrub the block
  axis) collapse to a stride and a run. Surprise worth keeping: linking the
  sibling with the full compare facet set was *wrong* — synced dims dragged
  the new axis back to whatever the origin's axis of the same number sat on,
  i.e. it broke the one thing the sibling exists for. Only `levels` starts
  enabled; axis-indexed facets are not meaningful across a rank change.
- **R7 — Warm worker exchange protocol** (proposal first): long-lived
  env worker, multi-array envelope, progress/cancel transport; deletes
  the env-slot limitation and the probe/conda tax.
- **R8 — Proof packs + sources entry-point group**: skimage/scipy
  filter pack (exercises R3), minimal torch `apply_model` op
  (exercises R3+R2 exact-input + R7), `arrayscope.sources` entry-point
  group over `ArraySource`.

R1+R2 are independent and land first; R3+R4 follow; R5/R6 build on
R2+R3; R7/R8 close.

## Test-surface intent

Every bundle extends existing test files in place rather than opening
new ones unless a genuinely new surface appears. R4 dedupes the two
dialog test files onto the shared base. R2's regression tests go into
the existing planner/conformance files. The all-ops smoke harness stays
the single predicted-vs-actual gate; per-op shape assertions fold into
its table rather than standalone tests.
