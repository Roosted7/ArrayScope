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

Only `int` and `float` parameters are supported. An annotation is used first to
infer the kind, then a literal default; otherwise the inferred kind is
`float`. Inference seeds the visible controls once. You can edit them
afterwards.

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

The name must match the function keyword. **Kind** controls conversion and the
spinbox type. **Default** supplies the initial value; if it is blank, the lower
bound or zero seeds the form. **Min** and **Max** are inclusive bounds, **Step**
controls the spinbox increment, and **Description** becomes field help.

For the example above, keep `level` as `float`, use `0.25` as **Default**, and
add bounds appropriate to your data, such as `0.0` and `1.0`. When a user adds
the operation, the same metadata builds the parameter popup and prevents an
out-of-bounds value from being applied.

## Add auxiliary array inputs

The primary array remains the function's first argument. Declare every
additional array under **Input slots**, with:

**Name**, **Label**, **Accepts**, and **Description**.

**Name** is a Python-style identifier and must match the function keyword.
**Accepts** is a comma-separated list of source kinds. The available kinds are
current dimension set, another open Compare document, one ROI as a mask or
coordinates, and a saved `.npy` array; use the exact persisted names from the
[input-slot schema](plugin-operations.md#wrapper-schema).

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

A **Problems** group means something different. It contains wrapper files that
could not be registered at all, for example bad JSON, a syntax error, a missing
source, a missing callable, or an unsupported parameter kind. Hover the item
for the loader's message, use **Open the operations folder**, and repair the
source or wrapper against the
[wrapper schema](plugin-operations.md#wrapper-schema). One broken operation
does not stop the rest of the library from loading.

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
