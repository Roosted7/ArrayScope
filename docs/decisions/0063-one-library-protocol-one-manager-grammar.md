# ADR 0063: One library protocol, one manager grammar

- **Status:** Accepted and implemented (2026-07-31). Bundle R4 ("Manager
  convergence") of the operation-system overhaul. Builds on ADR 0062, whose
  always-addressable invariant this ADR extends from operations to colormaps.
- **Number:** 0063 was free when this branch was cut (0062 is R1's
  always-addressable invariant). Parallel worktrees have collided on ADR
  numbers before; renumber on integration if 0063 is taken.
- **Program:** operation-system overhaul dossier,
  `docs/redesign/operation-system-overhaul-2026-07-30.md`.

## Context

ArrayScope has two libraries a user browses, rearranges, extends and breaks:
colormaps and operations. The colormap designer shipped first; the operation
manager was written eighteen months later and its module docstring says, in as
many words, that it "deliberately mirrors the colormap designer's interaction
model".

It mirrored it by copying it. Three blocks were literal copy-paste — the tree
widget, the listener registry with its dead-Qt self-heal comment, and the
layout load/save/`effective_group`/grouped-listing quartet — and from there the
two drifted, in both directions, on exactly the details a person notices.

The parity sweep counted it: operations win thirteen rows, colormaps win eight.
Neither side is the better manager; each is missing half of one.

- Operations have a Problems surface, read-only system definitions with a
  visible affordance, confirmation before anything is destroyed, a discrete
  Restore, Reset-all, Open-folder, install-scoped storage, a live library
  listener, and a layout file with a format header.
- Colormaps have a session Revert (the best affordance in the repo: snapshot,
  one-step undo, a button that only exists when it leads somewhere), a kind
  filter, inline group rename, Import, Export, an Apply that applies and
  closes, disable-with-a-written-reason, and a dirty-snapshot autosave.

And the vocabulary had collided outright. The same `folder_open` glyph meant
*Import* in one manager and *Open the folder* in the other. `data_object` — a
JSON-ish glyph that says nothing about duplication — was Duplicate on both.
Duplicating produced `"<name>-2"` here and `"<name> copy"` there. The colormap
dialog answered to four names across four surfaces ("Colormap manager…" in the
menu bar, "Customize…" in the toolbar, "Edit colormaps" in the palette,
`ColormapDesignerDialog` in the code). Every operation was a palette command
and no colormap ever was.

Two of the divergences were not cosmetic at all. Colormaps stored under
`AppDataLocation`, which is per-*user*, so a scoped `-n <name>` run shared one
colormap library with every other scope while its operations, settings and
sessions were correctly separate. And a colormap file the loader could not read
was dropped by a bare `except: continue` — a map the user had saved simply
stopped existing, with no message on any surface.

## Decision

**One library protocol and one manager grammar, with each side keeping only
its editor.**

### 1. `arrayscope/app/user_library.py` is the storage protocol

The listener registry, the layout store, the hidden set, the problems record
and the atomic write live there once, parameterized by the key names each
library already writes to disk (`map_groups`/`map_order` versus
`op_groups`/`op_order`). Neither on-disk file changes shape in order to share
the code.

Three consequences follow immediately, and they are the point:

- **Colormap storage is install-scoped.** It resolves through
  `user_dirs.user_config_directory()`, beside the operations directory. The
  first resolution per process copies an old unscoped library across when the
  new location is still empty — a *copy*, not a move, so an older ArrayScope on
  the same machine keeps working and a migration that goes wrong has not
  destroyed the only copy.
- **The colormap layout gains the format+version header** operations already
  wrote. A header-less file still loads; the next save adds the header. The
  upgrade is one-way and invisible.
- **Colormaps get a problems channel** in the same `LibraryProblem(path,
  message, payload, item_id)` shape ADR 0062 established, keyed by path for the
  same reason: the id lives inside the payload, so a file that will not parse
  has no id to key on.

A built-in whose optional package is missing is no longer dropped from the
listing either. It is `(unavailable)`, the word operations have used since ADR
0060: listed, greyed, and able to explain itself when selected. A curated name
vanishing silently is the same failure class as a swallowed parse error.

### 2. `arrayscope/ui/user_library_manager.py` is the dialog grammar

The tree (one class, with the Problems-row guard), the button bar, the filter
row, the marker vocabulary and its italic/grey treatment, group rename, layout
persistence, the Problems group, the session Revert, listener registration and
teardown, and selection preservation. Each manager subclasses it and
contributes its editor pane and what Apply means.

**The vocabulary is a data structure, not a convention.** `LIBRARY_BUTTONS`
fixes the order, the attribute name, the icon and the tooltip of all ten
buttons, and `tests/ui/test_user_library_manager.py` asserts both dialogs
expose exactly it. Drift fails a test rather than accumulating for eighteen
months.

The collisions are settled repo-wide: `folder_open` means *open the folder*
everywhere, Import is `upload`, Export is `download`, Duplicate is
`content_copy`, and a duplicate is named `"<base> copy"` on both sides. One
marker vocabulary — `(hidden)`, `(modified)`, `(user)`, `(unavailable)` — with
the operations side's italic + grey treatment for the two that mean "not
currently in play". One verb for the surface: **Manage colormaps** / **Manage
operations**, on the menu bar, in the toolbar and in the palette, with the
module renamed to match so the name in the code is the name on screen.

### 3. Each side adopts what the other had

Colormaps gain the Problems group and its repair pane, confirmation before a
delete, Restore, Open-file, Open-folder, Reset-all, a name search, and a
library listener so two open managers cannot show two different libraries.

Operations gain Revert, a filter row (text over label/id/group, plus a
user/system/hidden/unavailable narrowing that is the analogue of the colormap
*kind* filter), inline group rename, Import and Export of a definition JSON,
and an Apply that adds the selected operation to the window the manager was
opened from.

"Favorites" (a drag-target group) and "Common" (an editor checkbox) were the
same concept reached two different ways. Both are a checkbox now, in the same
place in the editor, over each side's existing mechanism — the colormap one
moves the map into the unnamed top section the pickers already render first, so
it is a second route to the same arrangement rather than a new concept. The two
*words* still differ, because "Common" is `registry.COMMON_OPERATION_IDS` and
`DEFAULT_GROUP_ORDER`, which this bundle does not own.

Import stopped being destructive on the colormap side. Importing a definition
whose name already existed silently overwrote the stored one, while operations
have always re-minted the id into a free slug; both pick a free name now, and an
explicit name is how a deliberate replacement is expressed.

**A built-in colormap becomes read-only, with an explicit "Override built-in".**
This is the one place where adopting the operations model *removes* an ability,
and it is deliberate: editing a built-in used to store an override silently on
the first keystroke, so a user could modify a built-in without ever deciding
to, and the only clue was a `(modified)` marker appearing later. A system
operation has been read-only-with-Duplicate since it shipped. `Restore` keeps
undoing the override.

### 4. Two dead buttons and one hazard

"Add a parameter" and "Add an input slot" appended a blank row, the autosave
skipped it for having no name, and the library refresh refilled the table —
wiping the row the click had just created. Measured rowCount 0 → 0, on main,
for as long as the buttons have existed. They append a *named* row now, with
the table's signals blocked while it is filled: every `setItem` emits
`cellChanged`, which saves, which rebuilds the table, so an unblocked append
persisted a half-written row and then wrote into cells whose C++ objects were
already gone.

The same shape is the load-bearing trap of this bundle in the large. A save
notifies the library, the library notifies this dialog, and the tree rebuild
lands back in "selection changed" while the save is still on the stack. Without
a re-entry guard the second pass compares the editor — already showing the
*next* definition — against the snapshot of the one being saved, decides it is
dirty, and saves again, forever. Registering the listener is what made the
colormap manager consistent across windows; `save_pending_edits` is what keeps
that from eating itself.

With R3's parameter kinds landed, one more hazard is closed here: the editor
offers the registry's `PARAMETER_KINDS` rather than a private copy, a kind this
build does not recognize is shown as stored rather than coerced to `float`, the
range and choice cells are disabled for the kinds that reject them, and a row
the registry would refuse (an `enum` whose choices are not typed yet) is *not
written* — it stays on screen uncommitted with the reason, because writing it
would move the definition into Problems mid-edit.

## Consequences

**Good**

- Learning one manager is learning the other. The tree behaves the same, the
  buttons are in the same order with the same glyphs, the markers read the
  same, Revert works the same, and a broken stored definition is repairable
  where it is listed on both sides.
- Two real data defects close: a scoped install no longer shares one colormap
  library with every other scope, and a colormap file the loader rejects is
  recorded and repairable instead of gone.
- The dialog tests consolidate: the shared behavior is exercised once, against
  *both* dialogs, and each manager's own file keeps only what is genuinely its
  own. `tests/ui/test_colormap_manager.py` shrank while its coverage grew.
- The colormap manager's gallery surface goes from one shot of an untouched
  dialog to the same states the operation manager is reviewed in.

**Costs and risks**

- Two dialogs now share a base class, so a change to the base is a change to
  both. That is the intent, and the parity test is what makes it visible; it is
  still a coupling that did not exist before.
- The colormap manager gained a library listener, which is what exposed the
  autosave re-entry hazard above. The guard is narrow and tested, but the
  general shape — a dialog that is a listener on a library it also writes — now
  exists on both sides.
- Making built-ins read-only is a behavior *removal* for anyone who was editing
  one directly. The path is one click longer.
- Operations' editor still saves per signal rather than through the colormap
  side's dirty-snapshot comparison. The seam is shared
  (`autosave_pending_edits` is called at the same moments on both sides) but
  the implementations differ; converging them means rewriting a forty-field
  editor, which is a larger change than this bundle's remit and buys
  consistency of mechanism rather than of behavior.
- `import_operation_definition` re-mints the id into a free `user:` slug rather
  than honoring the stored one. Importing the same definition twice gives two
  operations. That is the safe default — the alternative silently overwrites
  whatever already held that id — but it means an export/import round trip is
  not an identity.

## Alternatives considered

- **Leave the two managers alone and only fix the storage bugs.** Rejected: the
  storage bugs are symptoms. The colormap library swallowed parse errors
  *because* nobody had asked it to answer the question the operations library
  answers, and it would drift again the moment either side changed.
- **One dialog with a mode switch instead of a base class.** Rejected: the two
  editor panes have nothing in common (a gradient with draggable stops versus a
  runtime/source/parameter/slot form), and merging them produces a dialog whose
  every widget is conditional.
- **Make the shared vocabulary a documented convention.** Rejected on the
  evidence: it *was* a documented convention — the operation manager's docstring
  says it mirrors the colormap designer — and it still drifted into four names,
  two meanings for one glyph, and two duplicate-suffix conventions. A tested
  data structure is the only version of this that holds.
- **Keep `(user)` and `(modified)` separate from `(unavailable)` per side.**
  Rejected: they are one vocabulary answering one question ("why does this row
  look different?"), and splitting it is how it drifted in the first place.
- **Let the manager write an invalid parameter row and rely on the Problems
  surface to catch it.** Rejected: it is technically safe now that ADR 0062
  makes a broken definition repairable, but a definition leaving the tree
  mid-keystroke because the user has not finished typing is a shock the product
  should not deliver. Holding the row uncommitted keeps it where the user is
  looking.
