# R8 marathon salvage matrix

The abandoned `redesign-r8-marathon` worktree is read-only evidence.  This
matrix records the narrow comparison against commit
`e23ea7a5c6ab4473338e639df0012222397d479f`; it is not a second implementation
plan.

| Concern | Marathon evidence | Clean-branch decision |
| --- | --- | --- |
| First-pass level and histogram phasing | The tile commit still publishes from `first_display_commit` or later metadata improvement. Semantic/refined evidence can be scheduled independently of final display settlement. | Do not port. Seed the landed `LevelStatsService` tracker from first-pass payload evidence, gate refinement until the final display pass settles, and certify the ordering directly. |
| Histogram derivation | Histogram sample aggregation is moved to a guarded background task with revision and session checks. | Keep as a possible later bounded-work refinement only. It does not establish the required first-pass publication boundary. |
| Stale evidence | Worker callbacks compare session, semantic level key, and evidence generation before installation. | Preserve the clean branch's equivalent typed-generation checks and extend the phasing tests for supersession. |
| Source-based slot remapping | `FrameSession.retarget_index_window` derives the predecessor slot from semantic source identity and rebinds resident payloads to the successor slot. The clean branch already has this mechanism. The material difference is that the clean controller invalidates the entire physical presentation before calling it, while the marathon leaves compatible predecessor residency active. | Keep the clean source-identity remap. Remove the unconditional index-window invalidation; the successor transaction must explicitly re-acknowledge compatible placements. |
| Partial source-continuity transaction | The marathon builds a narrow delta containing every current compatible payload and keeps the full target active scope. | Comparison only. The clean transaction now preserves all compatible VisPy/PyQtGraph identities in the deterministic one-index trace, so do not port another builder unless a later transition test proves a missing case. |
| Cold entering source | The marathon can leave predecessor physical fallback content in a changed slot while the successor is missing. | Do not port. A predecessor source may never survive under successor geometry/labels. The clean path must expose a truthful placeholder until a current-source payload is acknowledged. |
| Physical residency identity | Backend residency is keyed separately from montage slot placement; the VisPy path can resolve a payload by resident source key and relocate its slot without texture upload. PyQtGraph retains source-keyed item state. | Reuse the existing clean-branch source-keyed backend paths first. Port backend code only if the deterministic remap test still reports uploads after the placement transaction lands. |
| Session generation and acknowledgement | The successor delta carries the new session transaction generation. Lifecycle acceptance still compares emitted identity, backend identity, and active target scope; retained predecessor pixels do not commit successor semantics. | Preserve the clean typed identity/generation machinery. No slot-number acknowledgement shortcuts. |
| Scheduling, pacing, batching, composite raster, governors | The marathon contains broad experimental changes in all five areas. | Out of scope. Do not transplant. |
