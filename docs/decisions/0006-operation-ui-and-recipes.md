# 0006 — Operation UI and recipes

ArrayScope exposes operation-pipeline actions through a registry-driven
dimension menu and an operation-stack dock.

The registry lives outside Qt and defines the operation label, operation type,
user parameters, and whether the operation changes shape. GUI code builds menus
from the registry and only collects parameter values, appends operations,
refreshes the view, or reports validation errors. Dimension actions include
fftshift and real/complex axis conversion, so those transformations are
undoable and recipe-backed instead of being one-off data mutations.

The operation dock is a `QDockWidget` because operation history should be visible
without taking over the viewer surface. It lists operations in order and provides
undo, clear, recipe save/load, and materialize controls. Undo and clear modify
the operation list, not the base data.

Recipes are JSON documents containing a version field and the operation stack
only. They do not contain array data. Loading validates every operation against
the current base shape by replaying shape prediction before replacing the active
document. Invalid recipes produce a user-facing error instead of crashing the
viewer.

The first evaluator keeps a small cache rather than a full compute graph. It
caches the last derived array plus the last image and line display results keyed
by base data identity, operation stack, and `ViewState`. Crop, reverse, and
conjugate use NumPy view-like operations where NumPy can provide them; FFTs and
reductions may materialize. This is enough to avoid repeated full-array
evaluation during unchanged refreshes while leaving room for future lazy
backends.

The existing viewer requires at least one array dimension. The GUI therefore
rejects operation stacks that would produce a scalar. The pure pipeline can still
predict such shapes for future support.
