# ADR 0062: A stored operation definition is always addressable

- **Status:** Accepted and implemented (2026-07-30). Bundle R1 ("Never
  bricked") of the operation-system overhaul. Refines ADR 0060, which
  established the problems channel but left it a read-only list.
- **Number:** 0062 was free when this branch was cut (0061 is the VisPy
  retirement). Parallel worktrees have collided on ADR numbers before;
  renumber on integration if 0062 is taken.
- **Program:** operation-system overhaul dossier,
  `docs/redesign/operation-system-overhaul-2026-07-30.md`.

## Context

ADR 0060 made the operation manager the single editor over a single schema,
and it made a broken wrapper *visible* rather than silently missing: the
loader catches per file, records the failure in `user_operation_problems`,
and skips. That was the right half of the answer, and for two years the other
half was missing.

Visible is not the same as addressable. A wrapper the loader rejects is not
registered, and every action in the manager is keyed on a registered
operation:

- the tree is built from `library.grouped_operations`, i.e. from the
  *registry*, so rejected wrappers can only be appended afterwards as a
  virtual "Problems" group;
- `selected_operation_id()` answers `None` for those rows, because there is
  no id to answer with;
- `_load_editor` needs `get_operation_entry` plus
  `export_operation_definition` and bails when either fails;
- `_sync_button_states` disables every button on `None`, and `_remove_or_hide`
  early-returns.

The result is a row that can be seen, hovered for a message, and *nothing
else*. Not edited, not deleted. The one escape hatch documented in
`docs/write-your-own-operation.md` was to leave the application, open the
operations folder, and hand-edit or `rm` the file.

Eighteen distinct failure classes reach that dead end, and the product walks
users into it. `duplicate_operation`'s command branch wrote a wrapper with no
`source` block; `source` is what the *Python* runtime requires; the Runtime
combo box is one click away in the editor the duplicate opens in. Switching it
persisted a wrapper the loader refuses, and the operation the user had just
created disappeared from the tree into an inert Problems row.

One class was worse. `availability()` is a probe of the outside world: it
resolves environments, renders command templates, and stats executables. A
command operation with a conda environment *and* "Run through the system
shell" — both reachable in the manager, one checkbox and one combo apart —
makes `command_runtime.build_command` raise `ValueError`. That call is made,
uncaught, from `PluginOperationSpec.current_unavailable_reason`, which
`registry._pack_operation_entry` calls for every spec. So the exception did
not disable one operation: it took down `all_operations()`, and with it the
operation manager, the add popup, and the command palette — every surface
from which the definition could have been repaired.

## Decision

**Every stored user-operation definition is always listable, editable, and
deletable, no matter how broken.** Four changes hold that invariant up.

### 1. The availability boundary is guarded

`current_unavailable_reason` wraps the `availability()` call and returns the
exception text as the unavailable reason. A definition that cannot answer
"can I run?" is an ordinary `(unavailable)` row that explains itself, not a
crash in its own enumeration.

This is deliberately a boundary, not a fix for any particular composition. The
shell/prefix conflict is one reachable combination; the general property is
that an availability probe touches the filesystem, `PATH`, and user-authored
templates, and none of those are things the listing of *other* operations may
depend on.

### 2. The problems channel carries the payload, keyed by file path

`user_operation_problems()` returns `OperationProblem(path, message, payload,
operation_id)` records instead of `(file, message)` pairs.

**The key is the path, and that is the load-bearing choice.** The obvious key
is the operation id, and it is wrong: the id lives *inside* the payload. Every
failure class that stops the payload parsing has no id at all, and the classes
that reject the id itself — a non-`user:` namespace, a shadowed built-in —
have one that cannot be looked up. `_wrapper_path_for_id` returns `None` for
exactly those files, so an id-keyed delete fails on the definitions that most
need deleting. The file path is the one identity a stored definition always
has.

`payload` is the raw document when it parsed as an object. It is what lets the
repair surface reuse the ordinary editor rather than grow a second one: the
loader rejected the definition, it did not make the stored `label` or
`command_template` unreadable.

Two path-keyed primitives join the id-keyed ones:
`remove_user_operation_file(path)` and `update_user_operation_file(path,
payload)`. Both refuse paths outside the operations directory. The second is a
whole replacement rather than a merge — a repair surface owns the entire
document, and merging into a payload the loader could not understand would
preserve exactly the keys the user is removing. Wrapper writes became atomic
(temp file plus `os.replace`), because a half-written save is a *new* broken
wrapper.

### 3. Problems rows are first-class rows

The manager's problem rows carry a path role instead of the dead virtual role.
`selected_problem_path()` is the sibling of `selected_operation_id()`, and
every action that used to early-return on `None` now asks the second question
before deciding the selection is unaddressable: Remove deletes by path behind
the same confirmation as any user operation, and Open-file opens the wrapper
itself.

Selecting a problem row loads a **repair editor** in the same column, in one
of two shapes:

- **payload parses** — the ordinary form is driven from the raw dict, fully
  editable, with the loader's message shown prominently above it. Edits save
  back to the file by path and the library rescans. This is not a second
  editor: it is the same widgets, fed from a dict instead of a registry entry.
- **payload does not parse** — there is no dict to fill a form from, so a
  plain-text JSON pane is the editor and Save is explicit. Save refuses text
  that is still not a JSON object and says why, rather than writing a
  knowingly-broken file.

A repair that works removes the row, so the selection follows the definition
to wherever it now lives in the tree.

### 4. The product stops manufacturing bricks

- `duplicate_operation`'s command branch emits a `source` block and a stub
  `.py`, exactly like the Python branch, so the runtime toggle is
  non-destructive by construction.
- The manager's runtime toggle calls `ensure_user_operation_source` before
  persisting `runtime: "python"`, so a wrapper from any other origin gets a
  stub instead of becoming a Problems row.
- The environment combo shows an unresolvable environment id as a
  `<id> (missing)` entry and keeps it selected. Falling back to "none" looked
  harmless and was then persisted by the next runtime autosave, silently
  destroying the reference the user needed in order to recreate the
  environment. ADR 0060 already said an unresolvable environment is a reported
  problem; it must not also be a discarded field.

## Consequences

**Good**

- The 18 failure classes stop being 18 dead ends. Whatever the loader
  objected to, the definition is in the tree, selectable, editable, and
  removable.
- The repair surface is the product's existing editor, so a repaired
  definition is stored in the same shape an ordinary one is, and there is one
  editor to keep coherent rather than two.
- The whole-listing failure class is closed at the boundary rather than at
  each composition that can reach it.
- The documented repair for a broken wrapper stops being "leave the
  application and use `rm`".

**Costs and risks**

- The manager now writes to files it did not load. `update_user_operation_file`
  is a whole replacement, so a repair surface bug overwrites the stored
  document rather than corrupting one field of it. Mitigation: the write is
  atomic, the primitives refuse paths outside the operations directory, and
  the rescan immediately re-adjudicates the result.
- The form-driven repair covers the fields the form has widgets for. A payload
  whose defect is in `format` or `id` still parses, so it gets the form — and
  the form cannot express those two fields. Such a definition remains listable
  and deletable, and Open-file reaches the wrapper in an external editor, but
  its in-manager repair is incomplete. Exposing `id` as an editable field is
  the obvious next rung and is deliberately not taken here: it interacts with
  layout keys, hidden-operation sets, and recipe references, none of which
  this bundle touches.
- Guarding `availability()` converts a programming error in a *third-party*
  spec's availability callable into a quiet "(unavailable)" row instead of a
  loud crash. That is the correct trade for the user, and the exception text
  is carried verbatim into the reason so the failure is still legible.

## Alternatives considered

- **Refuse the runtime toggle when the wrapper has no source.** Rejected: it
  strands the user on the runtime they are trying to leave, and it only
  addresses one of eighteen classes. Synthesizing the missing piece is both
  smaller and more general.
- **Key the problems channel on the operation id and repair
  `_wrapper_path_for_id`.** Rejected: unparseable payloads have no id to
  repair the lookup *with*. Any id-keyed scheme is defined only over the
  definitions that already load.
- **A separate "broken definitions" dialog.** Rejected: it is the second
  editor over the same schema that ADR 0060 deleted, and a definition being
  broken is not a different kind of definition.
- **Validate wrappers on write and refuse invalid saves.** Rejected as the
  primary mechanism: it cannot help the wrappers already on disk, wrappers
  arriving from another machine, or wrappers hand-edited in the folder. Write
  validation is a useful addition, not a substitute for being able to fix what
  is already stored.
