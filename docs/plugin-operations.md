# Plugin operations (Tier-1)

ArrayScope discovers external operations contributed by third-party pip
packages through a Python entry-point group. This page is the contract a
plugin author writes against.

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
| `parameters` | `()` | Tuple of `OperationParameter(name, label, kind="int")`. |
| `requires_axis` | `False` | Whether the op takes an axis. |
| `changes_shape` | `False` | Whether output shape differs from input. |

The shape/dtype adapter must be honest: if `fn` drops a row, `output_shape`
must return the reduced shape. ArrayScope predicts the derived-view shape from
the adapter without running `fn`.

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
