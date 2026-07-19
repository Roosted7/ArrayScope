# Areas — parallel work without merge hell

How to split work across people/agents so branches integrate cheaply. The
2026-07 experience: parallel feature branches collided on ADR numbers,
shared index files, and — worse — *semantic seams* (two branches extending
the same function signature differently). These conventions exist to make
those collisions boring.

## Ownership areas

Pick one area per branch/agent. Within an area, work rarely conflicts; the
chokepoints below are where areas meet.

| Area | Code | Tests | Docs anchor |
|---|---|---|---|
| **GPU engine / residency** | `arrayscope/gpu/`, `display/vispy*`, atlas/pool, `display/source_anchoring.py` | `tests/gpu`, `tests/display` | ADR 0055/0056, [gpu-engine-plan](proposals/gpu-engine-plan.md) |
| **Kernel / scheduling** | `arrayscope/kernel/`, governor, lanes, `latency_feedback.py` | `tests/kernel`, `tests/presentation` | ADR 0053 (one scheduler — hard rule) |
| **Render pipeline / LOD** | `arrayscope/render/` (ladder, pipeline, lod) | `tests/render` | [g5 contract](redesign/g5-source-grid-pyramid-2026-07-16.md) |
| **Operations / stages** | `arrayscope/operations/` (capabilities, stage cache, slabs) | `tests/operations` | ADR on operation capabilities |
| **Window / frame orchestration** | `arrayscope/window/` (`render.py`, `frame_*`), presenter | `tests/window`, `tests/ui` | [architecture/rendering.md](architecture/rendering.md) |
| **Core / IO / sources** | `arrayscope/core/`, `io/`, lazy sources | `tests/core`, `tests/io` | ADR 0049 |
| **Sync / multi-window** | `arrayscope/sync/` | `tests/sync` | ADR 0048 |
| **Tools / tracing / benchmarks** | `arrayscope/tools/`, diagnostics | `tests/app`, `tests/stress` | [testing/README.md](testing/README.md) |

## Chokepoints (expect conflicts; touch deliberately)

- `window/frame_effects.py`, `display_presenter`, `frame_session.py` — the
  commit path; nearly every lane ends here. If two branches must touch it,
  assign disjoint functions explicitly up front.
- `core/view_state.py` / `ViewState` — semantic identity. Canonicalization
  rules (e.g. full-coverage ranges → `None`) live here; never add a second
  spelling of the same state.
- `operations/capabilities.py` and `io/file_interpreters.py::load_path` —
  known **semantic seams**: multiple past branches extended them with
  different-but-overlapping parameters. When your branch touches a shared
  extension point, state the intended merged signature in the PR/commit so
  the integrator merges meanings, not just text.
- Doc index files (`docs/queue.md`, `decisions/README.md`, `CHANGELOG.md`,
  `reviews/README.md`) — append-only sections conflict textually; keep all
  entries from both sides, then dedupe.

## Conventions

1. **ADR numbers are assigned at integration, not on the branch.** On a
   branch, name the file `XXXX-slug.md` (or your best-guess number, knowing
   it will move) and let whoever integrates renumber: file name, `# NNNN`
   header, README row, and every cross-reference (grep for both `ADR 00NN`
   and `00NN-slug`). Note the renumber in the integration commit.
2. **One queue.** Branch work items come from [`queue.md`](queue.md) or a
   dossier it links; a branch must not grow its own queue doc. Handoff
   briefs (like the GPU-port one) are fine but say at the top which queue
   they serve, and die into the archive when consumed.
3. **Disjoint file ownership for concurrent agents.** When running several
   agents at once, assign explicit disjoint file sets; two agents in one
   chokepoint file is a merge you will lose time on.
4. **Integration checklist** (from the 2026-07-03 cowork round): renumber
   ADRs; union the doc-index sections; check semantic seams for merged
   signatures; then run ring 1 plus the owning area's focused tests, and
   ring 3/4 if the area is display/scheduling
   ([testing/README.md](testing/README.md)).

## Integration workflow rules (earned 2026-07-18/19)

- **Queue conflicts**: resolve per-row taking the newest row-truth, then
  graft unique passages from the other side — never drop a recorded
  fact. After ANY resolution, rescan the WHOLE file for conflict
  markers (two committed-marker incidents; `git diff --check` cannot see
  markers that are already committed).
- **Resolution scripts**: write→verify→stage as ONE guarded step; never
  `git add` after a script that may have failed before writing.
- **Shell verification chains**: `pytest | tail` hands the pipe's exit
  code to `&&` — redirect to a log and test `$?` explicitly.
- **Scheduler-adjacent branches get a codex second-opinion review before
  the fast-forward**: it caught two real defects (a coverage-evidence
  bypass and lost-wakeup family member #7, reproduced) that survived
  deep human review, red-first gates, two full suites, and a green
  journey matrix.
- **Parallel landings compose**: three semantic-seam breaks in two days
  came from textually-clean rebases (arity change, oracle-vs-behavior,
  dispatch-edge move). The full suite after EVERY fast-forward is the
  net — focused gates never substitute (an architecture-guard red hid
  from three focused runs).
- Background watchers: `pgrep -f` patterns must not match the watcher's
  own command line (bracket idiom: `[j]ourney`).
