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

### `sigpy_pack` — threshold + centered resize (no FFT)

`arrayscope/operations/packs/sigpy_pack.py` ships the sigpy operations that are
**additive** over the 13 built-ins. It deliberately ships **no FFT**: an earlier
design shipped `sigpy:fft` / `sigpy:ifft`, and they were removed as redundant
(see docs/graveyard.md). ArrayScope already covers a centered FFT two ways a
`sigpy:fft` op only duplicated — the built-in `centered_fft` / `centered_ifft`
**operations** (`arrayscope/operations/pipeline.py`, in `OPERATION_REGISTRY`),
and the **FFT backend** setting (`FFTBackendChoice {AUTO, NUMPY, PYFFTW, SCIPY}`
in `arrayscope/app/settings_state.py`, resolved in
`arrayscope/operations/fft_backend.py`) which selects the FFT *implementation*
underneath those built-in ops. `sigpy.fft` is `numpy.fft` under the hood, so a
sigpy FFT adds nothing as an op or as a backend, and is **not** added as a fourth
`FFTBackendChoice`.

What the pack does ship:

- `sigpy:soft_thresh` — pointwise complex **soft** thresholding (magnitude
  shrinkage, `sign(x)·max(|x|−λ, 0)`; the L1 proximal operator). The workhorse MRI
  sparsity/denoising primitive, offered as a view stage.
- `sigpy:hard_thresh` — pointwise complex **hard** thresholding (keep samples with
  `|x| > λ`, zero the rest). A sparsifying / support-view companion.
- `sigpy:resize` — **centered** zero-pad / center-crop of one axis to a target
  length (`sigpy.resize`): the canonical k-space *zero-fill* interpolation and its
  inverse center-crop. Additive over the built-in `crop` (which only *shrinks* by
  an explicit `[start:stop]` window and does not center).

Both threshold ops are strictly pointwise, so `fn(whole)[region] ==
fn(whole[region])` on every axis. They therefore declare **Tier-2
`region_capable=True`** — and are the first pack ops whose windowable claim the
conformance harness actually *honors* (the BART ops are all OPAQUE). `sigpy:resize`
is shape-changing and re-indexes the whole axis, so it is **Tier-1 OPAQUE**.

Numeric precision: sigpy's `soft_thresh` / `hard_thresh` always return
`complex128`. The ops **narrow the result back** to the input dtype (complex stays
complex by width, real floats keep their width, other real inputs → `float32`), so
they respect the repo's float32 discipline; the narrowing cast is pointwise and
does not disturb the windowable property. The `λ` threshold is a `float`
parameter, which is why a `"float"` parameter kind sits beside `"int"` in the
`create_operation` / `create_plugin_operation` paths.

**Still deferred** (they do not fit the `fn(ndarray) -> ndarray` +
scalar-parameter unary contract without engine changes):

- `sigpy.nufft` / `sigpy.nufft_adjoint(input, coord, ...)` — needs a k-space
  *coordinate array* as a second argument; the plugin parameter model carries
  scalars, not a companion ndarray.
- `sigpy.mri.app.EspiritCalib(ksp, ...)` — needs coil-axis + calibration
  semantics and a compute device, and it *changes dimensionality* (produces
  sensitivity maps) in a way the scalar-param shape adapter cannot predict; it is
  an iterative app object, not a pure `fn(ndarray) -> ndarray`.
- `sigpy.fwt` / `sigpy.iwt` (wavelet transform pair) — a natural reversible view
  stage, but `iwt` needs the *original* `oshape` **and** the `coeff_slices`
  structure `fwt` produced to invert. The scalar-parameter model cannot carry that
  structural metadata between two independent unary steps, so the forward/inverse
  pair cannot be expressed honestly. Deferred for the same reason as nufft/espirit.

### `bart_pack` — out-of-process BART ops (subprocess + cfl handoff)

`arrayscope/operations/packs/bart_pack.py` ships operations that run the external
[BART](https://mrirecon.github.io/bart/) `bart` binary **as a subprocess**, handing
data across in BART's native **cfl** temp-file format. Unlike the sigpy pack (an
in-process library call), the compute happens in a child process, so the pack owns
the cfl handoff, a working child environment, and cancellation of the child.

| id | label | axis | capability |
|----|-------|------|------------|
| `bart:fft`  | Centered FFT (BART)             | required | **OPAQUE / Tier-1** |
| `bart:ifft` | Centered iFFT (BART, unnormalized) | required | **OPAQUE / Tier-1** |
| `bart:cabs` | Complex magnitude (BART)        | none     | **OPAQUE / Tier-1** |

**BART's FFT convention.** `bart fft <bitmask>` is **centered but unnormalized**:
it equals `fftshift(fft(ifftshift(x, ax), ax), ax)` (verified against NumPy in the
tests). An axis maps to a dimension bitmask (`1 << axis`); the cfl handoff preserves
axis order, so numpy axis *a* is BART dim *a*. `bart:ifft` (`bart fft -i`) is likewise
unnormalized, so `ifft(fft(x)) == N·x` along the axis — BART's convention, **not**
NumPy's 1/N. `bart:cabs` is pointwise `|z|`.

**Everything is complex64.** cfl is a complex64 container, so every op takes and
returns complex64 (real/integer inputs are promoted on write). The ops declare
`output_dtype = complex64` unconditionally.

**cfl handoff.** The pack rolls its own minimal cfl reader/writer (`write_cfl` /
`read_cfl`) rather than importing `$BART_TOOLBOX_PATH/python/cfl.py` — this keeps the
pack self-contained (no `sys.path` mutation into the BART source tree) and cfl is
trivially simple: a `.hdr` text file listing dimensions plus a `.cfl` blob of raw
complex64 in column-major (Fortran) order. The format is byte-for-byte what BART
reads/writes (proven end-to-end by the fft-correctness test). Each op writes its input
to `in.cfl` in a `tempfile.TemporaryDirectory`, runs `bart <cmd> in out`, reads
`out.cfl`, and the temp dir is **always** cleaned up — on success, error, or cancel.

**Cancellation (SIGTERM → SIGKILL, `<1 s`).** `run_bart` starts the child in its own
session (`start_new_session=True`) and polls the operation's `cancellation_token`
while it runs. On cancel it `SIGTERM`s the child's process group, waits a short grace
(0.25 s), then `SIGKILL`s, and raises `EvaluationCancelled` (the same signal the rest
of the operations engine uses). A mid-op cancel therefore kills `bart` in well under a
second with **no orphaned process** and the temp dir cleaned; the test proves this
deterministically with a fake-`bart` shim and a startup barrier (the `<1 s` is the
assertion, not a fixed sleep). This subprocess cancellation is **independent of** the
kernel cooperative-cancellation item (queue item 10 notes item 10 "requires the
shutdown/cancellation item closed first"): the SIGTERM machinery does not depend on the
kernel work that threads a token into the plugin `fn` call path. Until that lands, the
engine plugin path applies the op with no token (the sync whole-array path); the runner
is ready to forward a token the moment the engine supplies one.

**Admission cost (honest).** Every BART op is **OPAQUE** — a per-region execution would
mean one out-of-process cfl round-trip *per tile*, which is never the right plan for an
expensive subprocess op, so even the pointwise `bart:cabs` stays OPAQUE (here the *cost
model*, not correctness, forbids windowing). The plugin path classifies each pack op as
a whole-array `TRANSFORM` (blocks and expands every axis, not chunkable, not fusable, a
cache-stage boundary) — the heaviest class the admission cost model has — and the forced
complex64 output raises the estimated bytes for real inputs. That OPAQUE/TRANSFORM
classification is the admission cost hint. (A dedicated per-op *out-of-process* cost
multiplier would need a new field on the frozen `PluginOperationSpec`, i.e. a plugin
contract change, which is out of scope here.)

**Optionality.** Availability is decided by `bart_available()` — a cheap, lazy
filesystem check that an executable `bart` exists on `PATH` or in the
`BART_TOOLBOX_PATH` toolbox, with that env var set. It never runs `bart version`.
When `bart` is not runnable the pack registers nothing; importing ArrayScope, building
the registry, and enumerating operations never spawn `bart`, so import-health stays
green.

**Deferred BART op.**

- `bart pics` (parallel-imaging compressed sensing) is **multi-input**: it needs a
  k-space array *and* a coil-sensitivity map array (`bart pics kspace sens out`), which
  does not fit the unary `fn(ndarray) -> ndarray` + scalar-parameter contract — there is
  no honest way to bind the second ndarray through the recipe/dock parameter model. A
  self-contained `ecalib`→`pics` variant would have to hard-code a coil axis and
  calibration semantics and would *change dimensionality*. It is deferred (mirroring the
  sigpy ESPIRiT deferral) rather than forced through the unary pipeline — correctness over
  coverage. The cfl-handoff + subprocess + cancellation mechanism is proven via `bart:fft`.
