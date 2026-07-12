# R5 — Canonical frame lifecycle

**Goal:** one writable record answers what every frame region targets, what
work owns it, what payloads are usable, and what the backend acknowledged.

## Landed

- `TileLifecycle` now owns target, task/dependency claim, payload candidates,
  emission, backend identity, acknowledgement, and settlement on one record.
- The parallel `TileLedger` module and session-owned ledger were deleted.
- Source retarget preserves physical backend truth for diagnosis without
  treating stale pixels as semantically presented.
- Target invalidation retains acknowledged pixels as fallback while reopening
  exact settlement.
- Transaction tests cover arrival ordering, wrong metadata, partial backend
  acknowledgement, source retarget, and fallback-to-target monotonicity.

## Exit gate

- No `tile_ledger` production path or second per-region transaction map.
- Watchdog and diagnostics are read-only.
- Lifecycle/render/backend focused suites stay green.
