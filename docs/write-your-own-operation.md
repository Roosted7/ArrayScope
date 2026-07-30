# Write your own operation

ArrayScope can turn a top-level Python function into an operation without a
package or entry point. The operation manager owns the complete workflow: start
from an empty operation or a working copy, connect code, edit the public form,
and repair anything that is unavailable.

This page is the practical walkthrough. The persisted JSON fields and callable
contract remain defined in the
[plugin-operations schema reference](plugin-operations.md#user-defined-operations-no-packaging-required);
do not hand-write a wrapper unless the manager cannot express the change you
need.

## Start in the operation manager

Open **View ▸ Operation manager…**. You can also choose **Manage operations**
from the command palette (`Ctrl+K`) or use the **Manage operations…** tune
button in the Operations dock.

Choose how to begin:

- **New** creates an unfinished `user:` operation and selects it in the same
  dialog. This is the shortest route when you already have a function on disk.
- Select any existing operation and choose **Duplicate** to begin with a
  working, editable copy. A duplicated native operation contains editable
  Python source. A duplicated pack or entry-point operation is a working
  adapter that still names its provider dependency. A duplicated command
  operation keeps its command runtime settings.

The manager saves operation edits as you make them; there is no separate Save
button. The stable **Id** is assigned when the user operation is created and
does not change when you edit its label or implementation.

For example, a source file can contain:

```python
import numpy as np


def threshold(data, level: float = 0.25):
    """Set samples below a level to zero."""
    return np.where(data < level, 0, data)
```

The array must be the first positional argument. Keep the function at module
scope: the callable picker lists top-level functions, not methods or nested
functions.

## Connect the source and callable

For a **Python** runtime, use **Source file** in the **Implementation** section
to choose a `.py` file, then choose **Callable**. ArrayScope parses the file
without importing or executing it. From the selected function it fills ordinary
editable fields:

- the function name becomes **Label**;
- the first docstring line becomes **Description**;
- an `axis` argument enables **Requires an axis**; and
- the remaining positional-or-keyword and keyword-only arguments become rows
  under **Parameters**.

Parameters may be `int`, `float`, `string`, `enum`, `bool`, or `path`. An
annotation is used first to infer the kind, then a literal default; otherwise
the inferred kind is `float`. `str` infers `string`, `bool` infers `bool`,
`Path` infers `path`, and a `Literal["reflect", "edge"]` infers an `enum` with
those choices — so a function that takes `mode="reflect"` arrives with a combo
box already filled in. Inference seeds the visible controls once. You can edit
them afterwards.

Choose the storage mode before or after selecting the source:

- **Copy into ArrayScope** keeps an independent copy in ArrayScope's operations
  directory. Moving, deleting, or later editing the original file has no
  effect. This is the default.
- **Link to original** stores the original file's absolute path. A change to
  that file is imported on the next use, and its modification time invalidates
  the prior output characterization.

Use **Open the code file** to edit the selected user operation's effective
source and **Open the operations folder** to inspect its files. After replacing
the body of a **New** operation's generated stub, choose that **Source file**
again so the manager retargets the source, refreshes inferred fields, and
clears the unfinished-template state.

At execution time, ArrayScope passes the array positionally. It passes `axis`,
declared parameters, and declared input slots by keyword only when the live
function accepts that name or `**kwargs`.

## Define the parameter form

Each **Parameters** row has these columns:

**Name**, **Kind**, **Default**, **Min**, **Max**, **Step**, and
**Description**.

The name must match the function keyword. **Kind** controls conversion and which
control the parameter popup renders: a spinbox for `int`/`float`, a combo box
for `enum`, a check box for `bool`, a text field for `string`, and a text field
with a browse button for `path`. **Default** supplies the initial value; if it
is blank, the lower bound or the kind's empty value seeds the form. **Min**,
**Max**, and **Step** apply to `int` and `float` only, and **Description**
becomes field help. An `enum` additionally carries its list of admissible
values; a value outside that list is refused rather than snapped to a
neighbour.

For the example above, keep `level` as `float`, use `0.25` as **Default**, and
add bounds appropriate to your data, such as `0.0` and `1.0`. When a user adds
the operation, the same metadata builds the parameter popup and prevents an
out-of-bounds value from being applied.

## Apply it to a subset of the dimensions

Most functions you already have are written for one image, not for a whole
5-D acquisition. You do not have to widen them.

An operation is told which axes to work on in one of three ways: none, exactly
one (**Requires axis**, a single dropdown at add time), or an axis **set**
(`"requires_axes": true`, which renders a checkbox per dimension in the
parameter popup). A set is normalized on the way in — sorted, deduplicated,
non-empty — so picking dimensions 2 and 0 is the same operation as picking 0
and 2.

On top of the set, `"maps_over_slabs": true` changes what your function
receives. It is handed each *slab* spanning exactly the selected axes, once per
position of every other axis, and ArrayScope reassembles the results:

```python
def unsharp_image(data, amount: float = 1.0):
    """Sharpen one image. `data` is a 2-D slab, not the volume."""
    import numpy as np
    from scipy.ndimage import gaussian_filter

    blurred = gaussian_filter(np.asarray(data, dtype=float), sigma=2.0)
    return data + amount * (data - blurred)
```

Import it, then set `"maps_over_slabs": true` and `"axis_arity": [2, 2]` in the
operation's wrapper JSON — `<slug>.json`, next to your code file in the user
operations folder, reachable from the manager's **Open folder**; the fields are
listed in the [wrapper schema](plugin-operations.md#wrapper-schema). The arity
says this function only makes sense on an image. Add the operation to a 5-D
dataset, tick the two displayed dimensions, and it runs on every slice of the
remaining three without a loop anywhere in your code.

The declaration buys more than convenience. Because ArrayScope knows the
operation only couples the selected axes, scrubbing an *un-selected* dimension
recomputes a single slab instead of re-running your function over the whole
volume. Two rules follow from that, and both are enforced rather than assumed:

- Return a slab of the shape you were given and you keep the per-slab
  behaviour. Resize the slab and the operation still runs correctly, but falls
  back to whole-array evaluation — the cheap path would otherwise have to place
  chunks using a fitted shape rule, and a misplaced chunk is a wrong picture
  shown at interactive speed.
- Return a slab with a different number of dimensions and the operation is
  refused: there is no position in the array for a slab that lost an axis.

If you want the axes themselves rather than a slab — you are writing the N-D
code yourself — declare an `axes` argument instead:

```python
def sum_over(data, axes):
    """Sum over the chosen dimensions, keeping them as length 1."""
    import numpy as np

    return np.sum(data, axis=tuple(axes), keepdims=True)
```

`axes` is recognized on import the way `axis` is, so it becomes the axis
picker rather than one more number to type.

## Add auxiliary array inputs

The primary array remains the function's first argument. Declare every
additional array under **Input slots**, with:

**Name**, **Label**, **Accepts**, and **Description**.

**Name** is a Python-style identifier and must match the function keyword.
**Accepts** is a comma-separated list of source kinds. The available kinds are
current dimension set, another open Compare document, one ROI as a mask or
coordinates, a saved `.npy` array, and a plain file; use the exact persisted
names from the [input-slot schema](plugin-operations.md#wrapper-schema). The
`file` kind is the one that delivers a **path** rather than an array — use it
for a model checkpoint or a database the operation opens itself.

Suppose the function is:

```python
def subtract_reference(data, reference, gain: float = 1.0):
    """Subtract a scaled reference array."""
    return data - gain * reference
```

Source inference initially treats both `reference` and `gain` as numeric
parameters. Remove the `reference` row from **Parameters**, add an **Input
slots** row named `reference`, and leave `gain` as the bounded numeric
parameter. Parameter and slot names cannot overlap.

When the operation is added to a view, its parameter popup requires the user to
choose one source for every slot. The resolved arrays are passed by keyword.
In-process Python and command runtimes support slots. Python running in a
separate environment does not yet transport auxiliary arrays and is shown as
unavailable when slots are declared.

## Use another runtime or environment

Leave **Runtime** set to **Python** for a function imported into ArrayScope's
process. To run it under another interpreter, expand **Advanced**, create a
record under **Named environments**, and select that record under
**Environment** in **Runtime settings**.

A named environment has an **Id**, **Name**, one **Locator** and **Value**, an
optional **Working directory**, and **Environment variables** entered one
`NAME=value` per line. Locator choices are:

- **Variables / working directory only**;
- **Interpreter path**;
- **Conda environment**; or
- **Virtualenv path**.

Choose **Save environment** before selecting the record for the operation. A
Python operation with an environment runs out of process. **Array handoff**
selects `npy` or `cfl`, and **Timeout** uses seconds; **No timeout** is the
special zero value.

For an external tool, choose **Command**, **Julia**, or **MATLAB** as the
runtime. The **Implementation** source picker is replaced by **Command
template** and its **Arguments** field. The template must include `{in}`,
`{out}`, every parameter name, and every input-slot name. ArrayScope tokenizes
the template into an argument vector and substitutes values as literal tokens.
Only **Command** can enable **Run through the system shell (unsafe; explicit
opt-in)**. Julia and MATLAB use the selected environment's interpreter, or
their executable from `PATH`.

The complete runtime fields, placeholders, handoff formats, and persisted
environment record are documented once in the
[runtime schema reference](plugin-operations.md#runtime-bodies).

## Understand and repair “unavailable”

An unavailable operation is still a valid, registered definition. It stays in
the manager with an `(unavailable)` marker and the exact reason in the status
line, but add surfaces disable it so it cannot fail later during Apply.

Common repairs are:

- finish a **New** operation and retarget its source/callable;
- supply every required command placeholder;
- select or repair a named environment whose interpreter, conda environment,
  virtualenv, working directory, or executable no longer resolves;
- use in-process Python or a command runtime when the definition has input
  slots; or
- for a command operation encountered through an imported recipe, inspect the
  definition and choose **Mark imported command reviewed**.

A **Problems** group means something different. It contains stored definitions
that could not be registered at all — bad JSON, a syntax error, a missing
source, a missing callable, an unsupported parameter kind. They are ordinary
rows: select one and the editor becomes a **repair editor** with the loader's
message at the top.

- If the wrapper still parses, the usual form is filled in from what is stored.
  Fix the field the message names — retarget the source file, correct the
  callable, change a parameter kind — and it saves as you edit. When the fix
  works the row leaves **Problems** and the operation appears in its group.
- If the file is not valid JSON at all, there is nothing to fill a form from,
  so you get the wrapper text with a **Save wrapper JSON** button. Saving text
  that is still not a JSON object tells you so and writes nothing.
- **Remove** deletes the stored definition and its files, whatever state it is
  in, and the code-file button opens the wrapper itself.

One broken operation does not stop the rest of the library from loading, and it
never has to be repaired from outside the application.[^folder]

[^folder]: **Open the operations folder** is still there, and an external
    editor is still the fastest way to fix a wrapper whose `id` or `format` is
    wrong — those two fields have no widget in the form. Everything else is
    reachable in the manager.

## Let ArrayScope discover shape and dtype

Do not write a shape adapter or dtype promise for a user operation. On first use
for an input signature, ArrayScope runs bounded representative probes, observes
the result dtype, and tries to fit a conservative shape rule. Shape-changing
functions are supported.

If the observations fit, the planner can predict the output before running the
full operation. If they do not, the operation stays an honest whole-array stage
with an exact result for that input signature. The characterization is cached
by operation, parameters, input shape and dtype, input-slot identities, and
source identity. Editing a linked Python file or a command template therefore
causes a fresh characterization.

Your function must consequently work on small representative arrays as well as
the real input. It should return something NumPy can view as an array, and it
must not depend on hidden mutable state that makes shape or dtype change between
equivalent calls.

## Write an operation that adds a dimension

An operation may return an array with **more** axes than it was given — a
sliding window over a time axis, an unpacked interleave, a per-coil expansion.
Nothing about the engine forbids it: the region planner, the stage cache and the
slab evaluator evaluate a rank-increasing operation the same way they evaluate
any other. Two things are different for you.

**Say where the new axes came from.** For a rank change, implement
`output_axes(axes)` so the result carries axis metadata of the right length; a
document whose axis metadata does not match its shape is a construction error,
not a display glitch. If you also implement `output_axis_sources(input_shape)`
— one entry per *output* axis naming the *input* axis it derives from, or `None`
for an axis you created — the planner can translate your per-axis capability
declarations across the rank change instead of failing closed. A user operation
whose shape is discovered gets both of these fitted from the discovered shape
rule, so this only concerns hand-written operations. The built-in **Split axis
into two…** is a worked example of all of it.

**Expect a new window.** ArrayScope's per-dimension controls — the Y/X/P
buttons, the slice spinboxes, the dimension chips — are built once, for the rank
the window was opened at, and cannot grow. So when you apply a rank-increasing
operation, the result opens in a **sibling window** and a toast says so
(“*<operation> adds a dimension — opened in a new window*”). The sibling shows
the same base array through your window's whole stack plus the new step; every
step is an ordinary entry in its operations dock, so you can edit, reorder or
disable any of them there. The window you applied from is left untouched — no
rank-growing step ever enters a live window's own stack. The sibling shares this
window's levels (the split preserves the value population) but deliberately not
its dimension positions, since its axes are no longer the same axes; the Sync
toggles turn the rest on if you want them.

If a rank-increasing step reaches a live window some other way — you re-enable
an inherited one, or load a recipe containing one — the same rule applies: the
document moves to a fresh window at the new rank and the old one closes. That
carries the base array, the stack, the title and the file identity; it does not
carry the view state, display settings, ROIs, or the dock layout, because every
one of those is indexed by axis.

Rank *decrease* needs none of this and stays in place, down to and including a
1-D line plot. Only a 0-D result is refused.

## Save and restore it in a recipe

After adding the operation to a view, use **File ▸ Save Operation Recipe**.
The recipe records the operation's stable id, axis, parameter values, enabled
state, and serializable input-slot bindings. It does not embed the Python source
or the operation wrapper, so another machine or profile must already have the
same user operation installed under that id.

Use **File ▸ Load Operation Recipe** to restore the steps. Session-local slot
bindings are resolved against the current window. A closed document, missing
ROI, or missing saved array restores the affected step disabled with an
unavailable reason instead of deferring a crash. Imported command operations
are likewise disabled until reviewed in the operation manager.

Once repaired or rebound, the operation uses the same parameter, slot,
shape/dtype discovery, and execution path as it did before saving.
