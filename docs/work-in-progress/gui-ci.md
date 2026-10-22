# GUI CI follow-up at winddown

The completed camera observer correction is commit `4768ad39`,
`test(montage): observe visible camera targets in covering scopes`. It is
separate from the remaining investigation below. It changes only test code:
a fully resident published tile scope may legally contain the current camera
scope without shrinking to exactly that scope. Seven focused Qt cases passed,
including the original failure, a deterministic real suppressed-retarget
reproduction, and negative controls for missing, stale, coarse, hidden and
uncovered camera content. No renderer code or interaction deadline changed.

The separate fixture-only follow-up is commit `d939e64f`,
`test(ui): declare codec capability and cover atlas edges`.

## Evidence and narrow fixture fixes

The actual CI reports inspected are retained outside git:

- `/workspace/work/arrayscope-ci/work/ci-logs/120054892724.log`
- `/workspace/work/arrayscope-ci/work/ci-logs/120054892798.log`

A serial local baseline of the following four cases produced two failures and
two passes in 44.52 seconds:

- `tests/ui/test_performance_settings.py::test_host_cache_compression_menu_switches_and_persists`
  passed locally, where optional codecs are installed. Both CI reports fail
  waiting for ZFP after triggering an intentionally disabled action when the
  optional `zfpy` library is absent. The test now declares codec availability
  at the existing menu capability seam; the adjacent tests still verify that
  unavailable choices stay disabled. This tests menu intent and persistence,
  not actual ZFP compression.
- `tests/ui/test_montage_data_change_invalidation.py::test_cpu_item_preview_atlas_stops_drawing_what_a_data_change_superseded`
  failed locally at the same `away_members` precondition as CI. Its 3x3
  comparison-box search used non-overlapping starts and omitted the last two
  columns and last row of the 17x16 fixture. A metadata-only probe found 47 of
  272 possible singleton member positions could never be selected. The search
  now considers all 3x3 positions. One pure helper regression checks every
  possible last member and disjoint comparison boxes without more Qt windows.
  The original pixel, stale-content, refill and completion assertions remain.
- `tests/ui/test_montage_data_change_invalidation.py::test_data_change_serves_the_visible_set_before_any_out_of_view_removal[pyqtgraph]`
  passed locally. CI failed its fixture precondition with all 272 tiles in
  `session.visible_tile_numbers` and none out of view. A plausible explanation
  is legal covering-scope retention after a camera move, as independently
  proven for `4768ad39`; that is not yet established for this test's subsequent
  data-change/removal ordering. No change was made to this case.
- `tests/ui/test_montage_scroll_settling.py::test_cropped_display_axis_scroll_keeps_complete_montage[pyqtgraph]`
  failed its initial `_window_settled` wait, matching CI. An isolated rerun
  passed in 43.12 seconds overall. The cause remains unresolved; a standalone
  pass does not establish that the initial session cannot strand work.

Scratch output lives under this checkout's untracked `work/` directory:
`ci-ui-baseline.log`, `ci-ui-probe.log`, and `ci-ui-fixture-final.log`.
These logs and diagnostic scripts are deliberately not committed.

The two narrow fixture corrections passed five focused serial cases in 27.58
seconds: menu switching, both unavailable-codec controls, exhaustive atlas-box
coverage, and the unchanged real atlas stale-pixel/refill interaction. This is
scoped evidence, not a whole-suite or GPU acceptance result.

## Remaining investigation

Keep these as open CI failures rather than weakening their assertions:

1. For the data-change ordering case, capture the actual camera's canonical
   `viewport_tiles` rings immediately before and after the source retarget,
   alongside the retained published scope, pending removals and deferred
   invalidation counters. Establish which scope the ordering contract names
   before changing the fixture. Preserve cold successor data, non-vacuous
   removal deferral, current visible pixels and eventual completion.
2. For the cropped-scroll timeout, record actual camera/viewport, source
   indices, required target identities and acknowledgements, semantic-level
   readiness, tile-controller work, atomic-warm state, queued owned timers and
   watchdog counters *before window teardown*. Determine whether the predicate
   names irrelevant background work or a real stalled visible target. Keep the
   existing interaction limit and NumPy/pixel checks.
3. CI also reports
   `tests/window/test_frame_paced_commits.py::test_a_montage_fill_interleaves_its_draws_with_its_commit_batches`
   with no commit deferrals. This lane has only read that test: its WGPU fill
   assumes a real owed-draw overlap occurs. Inspect whether the overlap is an
   actual fixture precondition on CI's software surface, preserving the
   interleaved draw/commit and eventual-fill obligations. No local reproduction
   or patch was attempted before winddown.
4. The WGPU scroll-settling and physical pixel failures belong to the separate
   GPU/driver investigation. Do not treat offscreen PyQtGraph passes as evidence
   about those paths.

Run focused cases serially with an absolute private `TESTMON_DATAFILE` and
`--testmon-noselect` while coordinating shared worker capacity. No broad suite,
production rendering change, deadline increase or hardware acceptance claim is
part of this follow-up.
