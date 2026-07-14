# R8 marathon salvage matrix

The abandoned `redesign-r8-marathon` worktree is read-only evidence.  This
matrix records the narrow comparison against commit
`775208866bc658dd415364cd1315e34138aead7e`; it is not a second implementation
plan.

| Concern | Marathon evidence | Clean-branch decision |
| --- | --- | --- |
| First-pass level and histogram phasing | The tile commit still publishes from `first_display_commit` or later metadata improvement. Semantic/refined evidence can be scheduled independently of final display settlement. | Do not port. Seed the landed `LevelStatsService` tracker from first-pass payload evidence, gate refinement until the final display pass settles, and certify the ordering directly. |
| Histogram derivation | Histogram sample aggregation is moved to a guarded background task with revision and session checks. | Keep as a possible later bounded-work refinement only. It does not establish the required first-pass publication boundary. |
| Stale evidence | Worker callbacks compare session, semantic level key, and evidence generation before installation. | Preserve the clean branch's equivalent typed-generation checks and extend the phasing tests for supersession. |
| Source-based slot remapping | Frame-session and backend paths contain source-identity reuse and placement acknowledgement machinery. | Compare only after the source-remap tests are red. Port a minimal missing transaction mechanism, not surrounding scheduling or pacing. |
| Physical residency identity | Backend residency is keyed separately from montage slot placement. | Reuse the compatible identity rule if the clean branch tests expose a gap; keep exiting residents inactive. |
| Scheduling, pacing, batching, composite raster, governors | The marathon contains broad experimental changes in all five areas. | Out of scope. Do not transplant. |

