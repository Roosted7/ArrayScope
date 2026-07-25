# Montage cold-fill cost — the plane-warm cohort clamp (2026-07-25)

**Status:** root cause found and fixed; the enabling second-order cost
(per-commit whole-montage work) is characterised below and deliberately not
taken further, with the reason recorded.

Field report: loading a 272-tile montage over the full third dimension of a
`(336, 336, 272)` float32 NIfTI, wgpu screen present, LOD 4 `(4, 4)`, "took
ages — tiles came in relatively smoothly, but in slow batches, with a
significant delay in between, and no workers active for most of the time."
The trace is a diagnostics JSONL taken across a data-reload (caches cleared,
forced re-load and re-render).

## What the field trace already proved

Presentation resets to 0 and climbs back to 272 over ~15 s while
`kernel/completed_keys` advances only 134 — so the fill was not compute-bound.
Every one of the 143 `tile_layer_commit` observations reads `->2`:

```
backlog=0/270/2->2    payload=68.072  apply=11.747   elapsed=82.3
backlog=0/172/100->2  payload=48.709  apply=34.688   elapsed=91.5
backlog=0/2/270->2    payload= 6.834  apply=74.522   elapsed=103.0
```

Fitting all 143:

| term | fit | r |
|---|---|---|
| payload build | `0.228 ms × pending + 7.4` | 0.991 |
| tile-layer apply | `0.232 ms × presented + 12.5` | 0.984 |

`pending + presented` is the constant montage, so the two slopes cancel:
**every commit costs ~93 ms regardless of how far along the fill is**, and
each one advances two tiles. 143 commits × 93 ms = **13.4 s of GUI callbacks**
— the whole load, for pixels that were already computed.

## Root cause

`_persistent_tile_upsert_limits` (`window/frame_effects.py`) clamped
`max_upserts` to 2 whenever *any* session payload was a native-plane warm:

```python
wgpu_native_prefetch = bool(
    capabilities.name == "wgpu"
    and any(wgpu_native_plane_warm_payload(payload) for payload in ...)
)
if wgpu_native_prefetch:
    batch_limit = min(2, max(1, int(batch_limit)))
    byte_cap = min(int(byte_cap), 3 * 1024 * 1024)
```

Two things make that fire on an ordinary cold fill:

1. `wgpu_native_plane_warm_payload` qualifies on `lod.level > 0` **alone**. On
   any zoomed-out montage that is every tile — not just the cropped-scroll
   case the clamp was written for (`e266260`, 2026-07-23).
2. It runs *after* `_idle_backlog_cohort`, overriding the very cohort whose
   docstring names this workload: "the 272-tile cold fill at 4 items per turn
   outran its settlement budget."

The clamp is also redundant on its own axis. `upsert_cost_fn` is
`wgpu_payload_upload_nbytes`, which already charges a warm payload its
**whole plane**, so the byte cap bounds hidden source-page warming without
help from an item count.

**Fix** (`f450660`): the item clamp keeps the interactive arm, where a
mid-gesture callback that warms 20–40 MiB delays the next frame's pixels.
Idle commits keep their backlog cohort and their byte cap.

## Measured

`profile_montage_workflow --backend wgpu --wgpu-present-method screen
--stages load_data,raw_full_tiled_montage,montage_scroll_scalar,
montage_zoompan_scalar --session-fixture ""`, headless Weston, same NIfTI.

| stage | before | after |
|---|---:|---:|
| `raw_full_tiled_montage` full-refined | 15.4 / 15.6 / 15.9 s | **4.05 / 4.07 s** |
| commits over that fill | 145 | ~14–19 |
| `montage_scroll_scalar` | 9.8 s | 9.2 / 9.4 s |
| `montage_zoompan_scalar` | 12.3 s | 8.4 / 10.9 s |

No oracle regresses. The standing FAILs on the scroll/zoompan stages
(`gui_callbacks_below_50ms`, `event_loop_heartbeat`, `warm_input_dispatch`)
and the intermittent `window_level_flicker_free` and ADR 0051 stall probe
reproduce on **both** arms — see the benchmarking note about rerunning
against a `git stash` baseline of the same command before blaming a diff.

## The second-order cost, and why it stops here

The ~82 ms whole-montage fixed cost per commit is what made a small cohort
catastrophic rather than merely suboptimal. With the cohort fixed it is paid
~14 times instead of 143, and the remaining fill divides roughly as: kernel
workers wall-busy 2.07 s (real tile evaluation, from `kernel_start`/
`kernel_finish` spans), commit CPU ~1.9 s.

One genuine quadratic was found inside that and fixed (`0292344`):
`LodPageCache._resolver_snapshot` rebuilt its whole passive `PageTable` per
residency revision, and during a fill every admit is a revision — 544
republishes, 147 696 binds, **0.508 s**. Republishing by diff makes that 543
binds and 0.053 s. It still publishes a *fresh* table, because
`resolved_page_set` resolves against it outside the cache lock while workers
land new revisions.

**Three hypotheses were measured and refuted; do not re-derive them:**

- *`_wgpu_payload_binding` copies the resident-key set per tile.* It does, but
  removing the copy changed nothing — the cost is `plane_chunk_key`
  construction (~5 keys/tile), not the copy.
- *`resident_by_plane` regrouping dominates the commit.* It does not. Grouping
  2048 keys is ~0.2 ms; an earlier attribution of 1.2 s to it was a
  mis-anchored timer window that actually spanned the executor-ensure prep. A
  page-table-maintained plane index was written, measured flat, and dropped.
- *GC pauses.* 0.13 s total with a 4.7 ms max across a whole fill.

What is left in the wgpu commit path is per-payload derivation over the full
montage (`_wgpu_payload_binding`, `_wgpu_reusable_native_texture`). It is real
and it is O(tiles) per commit, but each candidate saving is **below this
machine's run-to-run spread on the fill (4.0–4.9 s)**, so it cannot be landed
on evidence from this harness. Taking it needs either a lower-variance
offscreen repro (as the index-window dossier built) or a directly measured
work counter of the kind that justified the resolver fix.

`gui_callbacks_below_50ms` remains red on the fill (worst callback 551 ms,
down from 571–686 ms). That bar belongs to queue row 1.

## Follow-ups

- **The session fixture's window-size gate is stale, and it is not a headless
  artifact.** Every fixture-based profile run dies with
  `window_size [1400, 948]` against `session_window_size_target [1400, 940]`.
  This reproduces **identically on a live Wayland session**, and a headless
  probe resizes the same window to exactly 1400×940 on demand (output is
  1920×1200, dpr 1.0), so the compositor is not the constraint.

  `window_size` is a *derived* quantity — viewport plus chrome — asserted as
  if it were an input. Measured today, vertical chrome is a constant 209 px,
  so the fixture's recorded viewport of 739 needs a 948-tall window; when the
  fixture was captured (`0f11a22`) the same viewport came with 940, i.e.
  chrome was 201. Chrome has grown 8 px since. The restore is *correct*: it
  reproduces `viewport_shape [739, 1247]` exactly, which is what determines
  aspect, montage layout, and LOD. The gate fails it anyway, because
  `session_viewport_shape_matches` allows ±1 while
  `session_window_size_matches` demands exact equality.

  The same stale number sizes the screenshot compositor:
  `_managed_weston_output_size` reads `panels.window_size` so that "the window
  fills the output and one capture is the window". At 940 against a 948-tall
  window that identity is broken by 8 px — a likely source of previously
  observed full-window pixel mismatches attributed to Qt chrome.

- With `--session-fixture ""` the crop stages (`display_x_axis_slice`,
  `display_y_axis_slice`) crash in `_wgpu_payload_lod_geometry`
  (`source=(100, 336)`, `declared=(51, 168)`, `expected=(50, 168)`) — an
  edge-bin geometry mismatch that **reproduces unchanged on `65a9540`** and is
  therefore pre-existing and unrelated to this work. Untriaged.
