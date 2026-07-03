# 0048 — Linked-window sync over per-user local sockets

**Status:** Accepted (2026-07).

## Context

ArrayShow demonstrated that synchronized viewer windows (shared dimension
selection, levels, and ROIs) are a genuinely useful inspection workflow, but
its implementation only worked inside one MATLAB session. ArrayScope windows
are frequently started as separate processes (`arrayscope(...)` forks a child
per window, and users launch independent sessions from different scripts or
shells), so an in-process registry cannot cover the real usage pattern. The
roadmap item "Linked windows and inspection groups" fixed the constraints:
explicit group objects and typed messages, never a global workspace registry;
independent links per facet; feedback-loop prevention with origin/revision
ids.

## Decision

Sync is carried by newline-delimited JSON envelopes over Qt local sockets
(`QLocalServer`/`QLocalSocket`): named pipes on Windows, Unix domain sockets
on Linux and macOS. This keeps one implementation across all three platforms,
works between separately started processes, never touches the network, and
scopes the group to one machine and one user (the server name embeds the
username; `ARRAYSCOPE_SYNC_NAME` overrides it for isolation or multiple
groups).

Topology is broker-relay with dynamic election: the first participating
process listens as broker, later ones connect as clients, and the broker
relays every message to all other participants. When the broker process
exits, surviving clients retry after a random jitter delay; the first to bind
becomes the new broker (stale Unix socket files from crashed brokers are
reclaimed with `QLocalServer.removeServer` after a failed connect). There is
no persistent daemon and no state on disk.

Participation is per window and per facet — window/level, dimension indexing,
operations, ROIs — via toggle buttons (levels in the display toolbar, dims
next to the dimension strip, operations and ROIs in their docks). All facets
share the default group; named groups remain open for later without protocol
changes (envelopes already carry a group field).

Message payloads reuse the existing serialization vocabulary instead of
inventing a parallel one: `view_state_to_mapping` fields for dimension
indexing, operation recipes (`recipe_from_steps`/`steps_from_recipe`), and
ROI session mappings (`roi_to_mapping`/`roi_from_mapping`). Apply paths reuse
the existing semantic entry points (`with_slice_indices` + state sync,
`_apply_display_level_override`, `operation_coordinator.load_steps` +
`_set_document`, `_restore_roi_session`), so a synced change is
indistinguishable from a local one downstream.

Loop prevention is layered: the bus never returns a window's own messages;
every state envelope carries `(origin, revision)` and receivers drop already-
applied duplicates; a facet being applied suppresses republish; and publishes
are coalesced (120 ms) and skipped when the payload equals the last payload
sent or applied.

Mismatched receivers degrade instead of erroring: dimension indices are
matched by position and clamped per axis (extra sender axes ignored, missing
axes keep local values); an operation recipe that is incompatible with the
receiver's base shape is skipped with a status toast. Joining a facet pulls
the group's current state (a request message answered by enabled peers)
rather than pushing the joiner's state onto the group.

## Consequences

- Windows from any mix of processes sync as one group; one process acting as
  broker is invisible to users and survives arbitrary exit orders.
- Sync is opt-in per facet and costs nothing until the first toggle (the bus
  starts lazily and stops when the last facet is disabled).
- Levels-only sync cannot restart array evaluation (it goes through the level
  override path, preserving the levels-are-presentation invariant), while
  operations sync re-renders because source pixels legitimately change.
- The broker relays blindly; malformed lines are dropped by the codec, and
  versioned envelopes (`v`, `group`, `facet`) leave room for named groups,
  cursor links, and viewport links without breaking older builds.
- Same-machine, same-user scope is a deliberate limit: remote multi-user
  collaboration stays out of scope (roadmap "Explicitly not now").
