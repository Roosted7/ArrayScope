# Plugin operations (Tier-1 and Tier-2)

ArrayScope discovers external operations contributed by third-party pip
packages through a Python entry-point group. This page is the contract a
plugin author writes against. Most of it describes **Tier-1** (opaque,
whole-array) ops; the final section covers the **Tier-2** windowable claim and
the conformance test that gates it.

## What a Tier-1 plugin op is

Tier-1 semantics are deliberately narrow:

- **OPAQUE** — the operation materializes the whole array on CPU. It is not a
  coordinate remap, a shader-on-read, or a chunkable reduction.
- **Whole-array** — it makes no region/partial claims. Producing any output
  requires the whole input. (Region-aware ops are Tier-2, out of scope.)
- **Cache-stage-able** — its output is a legitimate stage boundary, so the
  engine may cache it.

A plugin contributes a pure `fn(ndarray) -> ndarray` plus a shape/dtype
adapter. ArrayScope wraps that into a pipeline step
(`arrayscope.operations.plugins.PluginOperation`) that satisfies the same
interface the built-in operations use, so it flows through the existing opaque
materialization path — there is no separate execution engine for plugins.

## The entry point

Advertise each operation in the `arrayscope.operations` group. The entry-point
**name is the operation's stable, namespaced id** and the value points at a
zero-argument factory:

```toml
# in the plugin package's pyproject.toml
[project.entry-points."arrayscope.operations"]
"mypkg:reverse_rows" = "mypkg.arrayscope_ops:make_reverse_rows"
"mypkg:roll"         = "mypkg.arrayscope_ops:make_roll"
```

Rules enforced at discovery time:

- The id **must be namespaced** — it must contain a `:` separator (e.g.
  `mypkg:reverse_rows`). Un-namespaced ids are rejected and logged.
- The id **must not collide with a built-in id** (`crop`, `mean`,
  `centered_fft`, …). A collision is rejected and logged, never silently
  honored.

Discovery is **lazy**: ArrayScope enumerates the entry-point *names* without
importing your package. Your module is imported (`entry_point.load()`) only on
the first actual use of one of its ops — constructing it, applying it, or
reconstructing it from a recipe. Keep import side effects out of module scope.

## The factory and the spec

The factory returns a `PluginOperationSpec`. The common stateless case only
needs `id`, `label`, and `fn`:

```python
import numpy as np
from arrayscope.operations.plugins import OperationParameter, PluginOperationSpec


def make_reverse_rows():
    def fn(array):
        return array[::-1]

    return PluginOperationSpec(
        id="mypkg:reverse_rows",
        label="Reverse rows",
        fn=fn,
    )
```

Parametric ops declare `parameters` (and optionally `requires_axis`) and supply
`build(axis, params) -> fn` instead of a bare `fn`:

```python
def make_roll():
    def build(axis, params):
        shift = int(params["shift"])
        target_axis = 0 if axis is None else int(axis)
        return lambda array: np.roll(array, shift, axis=target_axis)

    return PluginOperationSpec(
        id="mypkg:roll",
        label="Roll",
        build=build,
        parameters=(OperationParameter("shift", "Shift"),),
        requires_axis=True,
    )
```

`PluginOperationSpec` fields:

| field | default | meaning |
| --- | --- | --- |
| `id` | — | Namespaced stable id; must match the entry-point name. |
| `label` | — | Human-readable menu label. |
| `fn` | `None` | Pure `fn(ndarray) -> ndarray`. Provide this **or** `build`. |
| `build` | `None` | `build(axis, params) -> fn` for parametric ops. |
| `output_shape` | identity | Adapter `output_shape(shape, axis, params) -> shape`. |
| `output_dtype` | identity | Adapter `output_dtype(input_dtype) -> dtype`. |
| `parameters` | `()` | Tuple of `OperationParameter` (see below). |
| `requires_axis` | `False` | Whether the op takes an axis. |
| `changes_shape` | `False` | Whether output shape differs from input. |
| `group` | `"Other"` | Taxonomy group for menus/palettes (see `DEFAULT_GROUP_ORDER`). |
| `description` | `""` | One-line summary shown in tooltips / the operation manager. |
| `icon` | `"data_array"` | Material icon name the UI renders for this op. |

The shape/dtype adapter must be honest: if `fn` drops a row, `output_shape`
must return the reduced shape. ArrayScope predicts the derived-view shape from
the adapter without running `fn`.

### Parameter metadata

`OperationParameter` carries the metadata a UI form renders from and that
`create_operation` / `create_plugin_operation` coerce against:

| field | default | meaning |
| --- | --- | --- |
| `name` | — | Keyword the op's `build`/constructor receives. |
| `label` | — | Human-readable field label. |
| `kind` | `"int"` | `"int"` or `"float"`; the value is coerced to it. |
| `default` | `None` | Seed value. **A missing parameter that declares a default is filled from it**; a missing parameter *without* a default still raises, so recipes/CLI never silently drop a required value. |
| `minimum` / `maximum` | `None` | Inclusive bounds used to seed and validate the form field. |
| `step` | `None` | Suggested increment for a spinbox. |
| `description` | `""` | One-line help text shown beside the field. |

The Qt-free form model (`arrayscope.operations.parameter_forms`) turns an
`OperationEntry` plus the current array context (shape + axis) into a
`ParameterForm` of bounded, typed fields with read-only derived info lines
(e.g. crop's *Output length*) and cross-field interdependence (editing crop
`start` nudges `stop` to keep `start < stop`). Ops that need context-awareness
register a small provider keyed by op id; every other parameterized op gets a
default form derived purely from this metadata. Because the form is headless it
is unit-tested without a display, and any UI surface renders the same fields
from the same source of truth.

### Non-crash smoke guarantee

`tests/operations/test_all_operations_smoke.py` iterates **every** operation
`all_operations()` exposes — built-ins and installed packs — builds each from
its parameter form, and round-trips it through `output_shape` / `output_dtype`
/ `capabilities` / `apply` on a real float32 and complex64 array, asserting the
produced shape and dtype match the predictions. A new op that declares a
parameter it never handles, whose `apply` crashes, or whose adapter lies about
the output fails CI here without a hand-written per-op test, and the failure
message names the offending op id.

## Recipes round-trip

A recipe references a plugin op by its namespaced id plus its axis/parameters,
so a saved recipe reconstructs the same step:

```json
{
  "version": 2,
  "operations": [
    {"id": "mypkg:reverse_rows", "enabled": true},
    {"id": "mypkg:roll", "axis": 0, "parameters": {"shift": 2}, "enabled": true}
  ]
}
```

Loading a recipe that references an **uninstalled** plugin id fails with a
clear error (`unknown or uninstalled plugin operation: …`) rather than
crashing. A mathematically invertible op plus its inverse round-trips the data
(e.g. `roll` +k then −k), and the recipe→steps→recipe cycle is stable.

## Tier-2 (windowable ops): a claim that is conformance-tested

A Tier-1 op is **OPAQUE**: producing any output requires the whole input, so
the engine always materializes the whole array on CPU. A **Tier-2** op is an op
that additionally *claims* it is **windowable** — that it commutes with
sub-region reads:

```
fn(whole)[region] == fn(whole[region])   for a window `region` on any axis
```

When that holds the engine may run the op **per-region** and never materialize
the whole array — a real performance win for large volumes. This is the same
shape the built-in pointwise ops (`Conjugate`, scalar arithmetic) already have:
each output element depends only on the co-located input element.

Declare the claim with a single opt-in flag on the spec:

```python
PluginOperationSpec(
    id="mypkg:scale",
    label="Scale",
    fn=lambda a: a * 2 + 1,   # elementwise -> genuinely commutes with windowing
    region_capable=True,      # Tier-2 claim
)
```

A Tier-2 op **must be shape-preserving** (windowing an axis only makes sense if
that axis still exists and lines up in the output). Tier-2 v1 covers the
**elementwise-windowable** class — the op reads exactly the output sub-region
(identity input-region map). Ops that need a wider input than they emit (halo /
neighborhood kernels, declaring a non-identity `required_input_region`) are a
future extension: honoring them needs input-shape context at region-apply time,
which is engine surface beyond the plugin contract.

### The claim is never trusted on your word

A **false** windowable claim yields plausible-but-wrong pixels at interactive
speed — the exact silent-corruption class ArrayScope guards hardest against
(e.g. a global roll, a global normalization `x - x.mean()`, or an FFT dressed up
to look elementwise). So a Tier-2 claim is **honored only after it passes a
conformance property test**
(`arrayscope.operations.plugin_conformance.verify_region_conformance`):

1. build a deterministic seeded test array;
2. compute the whole-array result `fn(whole)` once;
3. for many sampled sub-regions (a partial slice and a point on **every** axis,
   plus random windows), compare the region path `fn(whole[region])` against the
   oracle `fn(whole)[region]`;
4. equality is **exact** by default (a truthful per-element op is bit-exact
   whether or not the input was windowed); an explicit tolerance is available
   only for float ops that legitimately reorder arithmetic.

If every sampled region matches, the op is honored: it runs per-region and
reports `ELEMENTWISE` capabilities with a `SHADER_ON_READ` execution class. If
**any** region disagrees, the claim is **downgraded to the OPAQUE whole-array
path** — the op still produces correct pixels (it just runs whole-array, not the
fast path) — and the downgrade is made observable: a loud `WARNING` is logged and
a `region_conformance_stats()` tally (`verified` / `honored` / `rejected`) is
incremented. Downgrade-not-refuse is deliberate: the underlying `fn` is still a
valid Tier-1 op, so we never break a user's pipeline over a performance
annotation — we only refuse to trust the fast path.

## First-party in-process packs

Third-party packages contribute ops through the entry-point group above. A
**first-party pack** is the in-tree counterpart: a module under
`arrayscope/operations/packs/` that ships *inside* ArrayScope and registers its
`PluginOperationSpec`s directly (via
`arrayscope.operations.registry.register_pack_operation`) instead of through an
entry point. A pack reuses the exact same `PluginOperation` machinery — opaque
materialization, the Tier-2 conformance gate, recipe round-trip — so a pack op is
indistinguishable from an entry-point plugin op once registered.

A pack is **optional and lazy**. Each pack module self-guards on its backend and
is enumerated through `registry.all_operations()` (which the operation dock,
command palette, and export menu use). `operation_entries()` stays built-ins-only
for callers that assume concrete dataclass operation types.

### `sigpy_pack` — availability seam, zero registered ops

Bundle A demoted every unary SigPy wrapper to a native built-in. The module
keeps its lazy `find_spec("sigpy")` availability probe for later genuinely
SigPy-shaped operations, but `pack_specs()` is empty and importing or
enumerating operations never imports SigPy.

| removed id | native replacement | reason |
|---|---|---|
| `sigpy:soft_thresh` | `soft_threshold` | the wrapper promoted float32 to complex128 before narrowing |
| `sigpy:hard_thresh` | `hard_threshold` | the wrapper promoted float32 to complex128 before narrowing |
| `sigpy:resize` | `pad`, `crop`, `resample` | native structure ops cover centred growth/crop and fractional interpolation |
| `sigpy:circshift` | `roll` | the wrapper was `np.roll` plus an optional-package import |
| `sigpy:downsample` | `resample` | integer-only decimation was not an honest general resampler |
| `sigpy:upsample` | `resample` | integer-only zero insertion was not an honest general resampler |

NUFFT, ESPIRiT, iterative apps, and wavelet pairs remain deferred until the
definition/input-slot bundles can carry their extra arrays and structural
metadata. Bundle A does not invent a unary approximation.

### `bart_pack` — shared command runtime plus readable BART-native examples

Bundle A likewise removed every BART operation that merely performed built-in
arithmetic. Bundle C reimplemented `run_bart` over the shared cfl command
runtime, preserving its argv API, fake-binary coverage, timeout, cancellation,
concurrent draining, and cleanup. `bart_executable` now resolves from the
effective environment `PATH`; it does not interpret `BART_TOOLBOX_PATH`.

| removed id | native replacement | reason |
|---|---|---|
| `bart:fft` | `centered_fft` | avoided a subprocess and two cfl temp files |
| `bart:ifft` | `centered_ifft` | avoided a subprocess and two cfl temp files |
| `bart:scale` | `scale` | avoided a subprocess, temp files, and forced complex64 |
| `bart:spow` | `power` | avoided a subprocess, temp files, and forced complex64 |
| `bart:cabs` | `magnitude` | avoided a subprocess, temp files, and forced complex64 |
| `bart:carg` | `phase` | avoided a subprocess, temp files, and forced complex64 |
| `bart:normalize` | `normalize` | native per-axis L2 normalization preserves input precision |
| `bart:std` | `std` | native sample std (`ddof=1`) returns the honest real dtype |
| `bart:var` | `var` | native sample variance (`ddof=1`) returns the honest real dtype |

The pack registers two genuinely BART-native, readable command definitions:
`bart:ecalib` and `bart:walsh`. They are intentionally unavailable until Bundle
D can characterize their output shapes; users can still inspect and duplicate
their `bart … {in} {out}` templates. The cfl handoff remains complex64 because
that is BART's format. Multi-input reconstruction commands such as `pics`
remain for Bundle E rather than being misrepresented as unary operations.

## User-defined operations (no packaging required)

A plugin op (above) ships in a pip package with an entry point. That is the
right vehicle for a *distributable* operation, but it is heavy for the common
case: "I have a one-off Python function on disk and I want to run it as a view
stage." **User-defined operations** are that lighter path. They live entirely in
the user's config directory — no `pip install`, no entry point — and are managed
at runtime through `arrayscope.operations.library`.

### Where they live

User ops are stored next to the user's session config, in the `operations/`
subdirectory of `arrayscope.app.user_dirs.user_config_directory()` (resolved via
`arrayscope.operations.library.user_operations_directory()`). Each op is one
wrapper JSON `<slug>.json`; an *imported* op also has its copied code file
`<slug>.py` alongside it.

### Wrapper schema

```json
{
  "format": "arrayscope-operation",
  "version": 1,
  "id": "user:<slug>",
  "label": "Human label",
  "description": "One-line summary (from the docstring by default).",
  "group": "User",
  "icon": "extension",
  "runtime": "python",
  "source": {
    "mode": "import",
    "path": "<slug>.py",
    "callable": "my_function"
  },
  "requires_axis": true,
  "changes_shape": false,
  "parameters": [
    {"name": "amount", "label": "Amount", "kind": "int", "default": 1}
  ]
}
```

The id **must** be namespaced `user:<slug>`, so a user op can never shadow a
built-in (`crop`) or a third-party plugin op. Everything else is
auto-filled on import from an `ast` introspection of the target function, so a
user rarely writes this JSON by hand.

### Runtime bodies

`runtime` selects one of four concrete bodies:

- `"python"` with no `environment` imports the source into ArrayScope's
  interpreter, as above.
- `"python"` with an `environment` id executes the same source/callable in that
  environment's interpreter. The array crosses the process boundary using the
  wrapper's `handoff` (`"npy"` or `"cfl"`) and `timeout_s`.
- `"command"` executes `command_template` directly as an argument vector.
- `"julia"` and `"matlab"` use the command machinery but prepend the selected
  environment's interpreter, or `julia` / `matlab` from `PATH`.

A command wrapper is:

```json
{
  "format": "arrayscope-operation",
  "version": 1,
  "id": "user:external-recon",
  "label": "External reconstruction",
  "runtime": "command",
  "command_template": "recon-tool --iterations {iterations} {in} {out}",
  "handoff": "npy",
  "timeout_s": 600,
  "shell": false,
  "environment": "recon",
  "parameters": [
    {"name": "iterations", "label": "Iterations", "kind": "int", "default": 30}
  ]
}
```

`{in}`, `{out}`, and every declared parameter must occur in the template.
ArrayScope tokenizes the authored template first and substitutes values second:
a path containing spaces remains one argument, and a value such as
`"--looks-like-a-flag"` remains a literal value. The template is never sent to
a shell unless `"shell": true` is explicitly set. Shell execution is an
advanced, user-authored opt-in and is unavailable for the Julia/Matlab prefix
runtimes.

`handoff` currently accepts `"npy"` (dtype-preserving NumPy files) and `"cfl"`
(BART's complex64, Fortran-ordered `.hdr`/`.cfl` pair). The runtime dispatches
through a small handoff registry so NIfTI/raw can be added without another
subprocess implementation. Both formats use the same concurrent stdout/stderr
draining, overall timeout, and SIGTERM-to-SIGKILL cancellation path.

### Named execution environments

Reusable environments live beside wrappers in `operations/environments.json`:

```json
{
  "format": "arrayscope-operation-environments",
  "version": 1,
  "environments": [
    {
      "id": "recon",
      "name": "Recon tools",
      "interpreter": "/opt/recon/bin/python",
      "conda_env": "",
      "venv_path": "",
      "working_directory": "/data/reconstruction",
      "variables": {
        "BART_TOOLBOX_PATH": "/opt/bart",
        "OMP_NUM_THREADS": "4"
      }
    }
  ]
}
```

At most one locator is set: an executable `interpreter`, a named `conda_env`,
or a `venv_path`. A variables/working-directory-only record is valid for a
general command. Resolution is lazy. Missing interpreters, vanished conda
environments, invalid working directories, and commands absent from the
effective `PATH` make referencing operations unavailable; they do not defer a
crash until Apply. `BART_TOOLBOX_PATH` has no special resolver anymore—it is an
ordinary variable on a named environment.

`changes_shape` is **reserved and must be `false`**. A wrapper cannot supply an
`output_shape` adapter, so a shape-changing op could not predict its output
shape — its `evaluate_shape` would diverge from `apply` and lie to the
evaluator. A wrapper that sets `changes_shape: true` is skipped with a recorded
problem, and `import_custom_operation(..., changes_shape=True)` raises. A user op
is therefore shape- and dtype-preserving today.

### Import vs. link

- **import** (`mode: "import"`, the default): the `.py` file is *copied* into the
  ops directory. The op is self-contained — deleting or editing the original file
  has no effect. This is the safe default for "capture this function".
- **link** (`mode: "link"`): the wrapper stores the **absolute** path to the
  original `.py`, which is imported live. Editing that file is picked up
  automatically: the import is cached by `(path, mtime)`, so a saved edit (bumped
  mtime) triggers a fresh import on the next use. This is the "I'm actively
  iterating on this function" mode.

### The call contract (how your function is invoked)

Your function receives the array as its **first positional argument**. Beyond
that the library adapts to your signature — introspected from the live function,
so a link-mode edit that changes the signature is honored:

- `axis` is passed **only if** your function declares an `axis` parameter (or
  `**kwargs`). A function with an `axis` param sets `requires_axis` on import.
- each declared parameter is passed **by name**, only if your function accepts
  that name (or `**kwargs`).

So `f(data)`, `f(data, axis)`, `f(data, **params)` and `f(data, axis, **params)`
all work without you writing any glue. Parameter `kind` (`int`/`float`) is guessed
from the annotation, then the default value, falling back to `float`.

A user op is **Tier-1 OPAQUE** (whole-array). It makes no region/windowable
claim, so it never touches the Tier-2 conformance gate — and its output must be
shape- and dtype-preserving (a shape-changing op is rejected; see
`changes_shape` above).

### Managing them (the public API the manager UI builds on)

`arrayscope.operations.library` is Qt-free and is the whole surface the operation
manager UI drives:

- `introspect_python_source(path) -> list[CallableInfo]` — list a file's
  top-level functions **without importing or executing** it (pure `ast`). It
  works even on a file that would fail to import (a top-level `raise`, a missing
  dependency), so the manager can still offer its callables.
- `create_empty_user_operation() -> str` — write a deliberately unfinished,
  loud template and return its new `user:<slug>` id for in-manager editing.
- `duplicate_operation(id) -> str` — write an editable user copy. Native
  shape-preserving code is copied into a function; pack/entry-point operations
  get a working adapter that names the dependency; shape-changing operations
  become an explicit blocked template until discovered shapes land (never a
  false shape-preserving claim).
- `update_user_operation_source(id, path, callable, *, link, infer=True)` —
  retarget the existing entry, copy or link its code, and expose AST-inferred
  label/description/axis/parameters through the ordinary editable wrapper
  fields.
- `import_custom_operation(py_path, callable_name, *, link=False, label=None, …)
  -> str` remains the Qt-free convenience API for programmatic import.
- `remove_user_operation(id, *, delete_files=True)`,
  `update_user_operation(id, **wrapper_fields)`,
  `user_operation_wrapper(id)`, and `user_operation_source_path(id)`.
- `execution_environments()`, `update_execution_environment(**fields)`,
  `remove_execution_environment(id)`, and
  `resolve_execution_environment(id)` own the reusable environment records.
- `quarantine_imported_command(id)` and `review_user_operation(id)` own the
  imported-recipe trust transition.
- `refresh_user_operations()` — re-scan the directory and re-register. The app
  calls this at startup (as it does for the colormap library) so recipes that
  reference a `user:` op resolve.
- `grouped_operations(*, include_hidden=False)`, `set_operation_hidden`,
  `hidden_operations`, `reset_operation`, `apply_library_layout`, `reset_layout`,
  `effective_common_ids`, `effective_more_groups`, and `add_library_listener` /
  `remove_library_listener` (listeners are notified on every mutation and
  self-heal dead Qt listeners).

### The no-crash guarantee

A broken user op **never** breaks startup or any registry enumeration. A wrapper
with bad JSON, a missing/`syntax-error` code file, or a missing callable is
caught, logged, and **skipped** — the rest of the library loads normally. The
reason it was skipped is retrievable via
`user_operation_problems() -> list[(file, message)]`, so the manager UI can show
the failure instead of the op silently vanishing. Because registry code never
scans the ops directory itself (the library owns the scan and drives
`register_user_operation`), one user's broken file can never fail an unrelated
machine's `all_operations()` or the non-crash smoke harness.

A structurally valid but non-runnable wrapper is different: it remains
registered with `unavailable_reason`. This covers New's empty template, a
shape-changing duplicate pending Bundle D, an incomplete command template, a
missing environment/executable, and an imported command awaiting review.
Registered unavailable operations remain visible for diagnosis and editing but
are never offered as runnable work.

### Imported recipes and command trust

A command operation referenced by an imported operation or view recipe is
persistently marked `review.required` and loaded as a disabled pipeline step.
It cannot execute during the recipe-triggered render. The manager shows the
review reason and a **Mark imported command reviewed** action; reviewing clears
that flag and immediately rechecks its template, environment, and executable.
Locally authored command operations are not quarantined merely for existing.

## Managing operations (the manager UI)

The **operation manager** dialog (`arrayscope.ui.operation_manager`) is the
graphical front end over the library above — the operations analogue of the
colormap manager. Open it from **View ▸ Operation manager…**, from the **Manage
operations** command in the command palette (`Ctrl+K`), or from the **tune**
button at the top of the operations dock.

The left column is a drag-reorderable tree of groups and operations. Dragging
ops within/between groups or reordering the groups persists through
`apply_library_layout`, so the arrangement drives every surface that lists ops
(the dock add popup, the axis context menu, the command palette). Hidden ops are
shown greyed with a `(hidden)` marker so they can be restored; user ops carry a
`(user)` marker; a wrapper that failed to load appears under a virtual
**Problems** group with the loader's message as its tooltip.

- **Hide vs. remove** — a **system** op (built-in or pack) can only be *hidden*
  (`set_operation_hidden`), which removes it from the listings but keeps it
  restorable; its definition is read-only. A **user** op is *removed* outright
  (`remove_user_operation`), deleting its wrapper and — for import-mode — its
  copied code file, after a confirmation.
- **Restore** un-hides the selected hidden op; **Reset layout and unhide all**
  discards the persisted arrangement (`reset_layout`) and clears every hidden
  flag.

The right column is the product's **one operation editor**. It renders every
operation through the Qt-free declarative definition exporter
(`operations.operation_definitions`): identity, shape/axis interface, full
parameter metadata, and a source body. System definitions are visibly
read-only except for group (a layout override) and the **Common** toggle.
**Duplicate** creates and selects an editable `user:` copy. User ops expose
label, description, group, icon (with a live preview), *requires axis*, source
file, callable, copy-vs-link storage, and the full parameter metadata table
(name, kind, default, min, max, step, description). Changes auto-save; there is
no separate Save and no second parameter editor.

The runtime selector keeps the ordinary Python case compact. Command, Julia,
and Matlab definitions replace the source picker with the editable command
template. The **Advanced** disclosure contains per-operation environment,
handoff, timeout, and explicit shell settings plus the reusable environment
editor (interpreter/conda/venv locator, working directory, and variables).
Unavailable operations carry an `(unavailable)` marker and the same reason
shown by the add popup, axis-chip menu, and command palette; those three
surfaces render the entry disabled with the reason as tooltip.

### Connecting up a custom function (import vs. link)

**New** creates an empty user entry and selects it without leaving the manager.
Choose a `.py` source file in the right-hand implementation section; the
callable picker is populated by `introspect_python_source` (pure `ast`, never
executing user code). Choosing a callable fills label, description,
*requires axis*, parameter names, kinds, and defaults into the same ordinary
editable controls. Inference is visible initial data, never hidden policy. The
storage choice sits directly below it:

- **Import a copy (recommended)** copies the file into the operations directory,
  so the op keeps working if you move or edit the original.
- **Link to the file** keeps a live link to the original path; edits you make to
  it are picked up automatically (mtime-keyed re-import).

Retargeting keeps the same selected `user:` id through
`update_user_operation_source`; there is no confirm dialog or second editor.
The **Open the code file** button opens a user op's `.py` in your default
editor, and **Open the operations folder** opens the directory itself.
