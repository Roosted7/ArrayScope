"""Inputs read only by tests that launch child processes.

Keep declarations separate from the executable selection policy. A declaration
edit must invalidate its child-process tests, not every test whose report uses
that policy. The injector records this module as a whole-file dependency for
all declared tests, including empty entries and declaration guards.
"""

#: Tests whose real work happens in a child process, mapped to the entry points
#: that child runs. Coverage never sees those files, so they are re-attached to
#: the test's recorded dependencies by hand (whole-file granularity) — which
#: means these tests are selected *exactly* when their entry point changes, and
#: skipped otherwise. That precision is the point: the set below costs ~46 s to
#: run, far too much to attach to every iteration.
#:
#: Only genuine spawns belong here. A test that monkeypatches ``subprocess.run``
#: (``test_qt_platform``, ``test_free_threading``, ``test_headless_display``,
#: ``test_profile_montage_workflow``, ``test_bart_pack``,
#: ``test_execution_environments``) runs entirely in-process and is already
#: traced; adding it would only slow the loop down.
#: ``tests/app/test_test_selection.py`` fails when a new real spawn appears
#: without an entry here, when a declared path has moved, and when a declared
#: path is one testmon cannot fingerprint — the last of which used to be a
#: mid-suite ``INTERNALERROR``; see
#: :func:`install_extensionless_dependency_support`.
OUT_OF_PROCESS_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    # A fresh interpreter reproduces Windows' missing Unix resource module
    # before importing the shared pytest diagnostics helper.
    "tests/app/test_load_report.py": ("tests/load_report.py",),
    # A fresh interpreter spawns real attachers for all three transports, then
    # exits so resource-tracker cleanup and stderr belong to the oracle too.
    "tests/core/test_shared_memory_compat.py": (
        "arrayscope/core/shared_memory.py",
        "arrayscope/kernel/shm_slots.py",
        "arrayscope/operations/source_transport.py",
        "arrayscope/operations/stage_segments.py",
        "arrayscope/operations/regions.py",
    ),
    # Named HDF5 replay is also exercised in a fresh interpreter that refuses
    # Qt imports and payload reads during planning; the parent cannot trace it.
    "tests/app/test_hdf5_cli.py": (
        "arrayscope/__main__.py",
        "arrayscope/app/batch_cli.py",
        "arrayscope/io/recipe_replay.py",
        "arrayscope/io/hdf5_catalog.py",
        "arrayscope/io/hdf5_source.py",
        "arrayscope/io/region_export.py",
        "arrayscope/io/numpy_save.py",
        "arrayscope/operations/recipes.py",
        "arrayscope/operations/planner.py",
        "arrayscope/operations/source_read.py",
    ),
    # The descriptor roundtrip reconstructs the actual HDF5 source in a child,
    # rather than merely proving that its metadata can be pickled.
    "tests/operations/test_source_transport.py": (
        "arrayscope/core/array_source.py",
        "arrayscope/io/hdf5_catalog.py",
        "arrayscope/io/hdf5_source.py",
        "arrayscope/operations/source_transport.py",
        "arrayscope/operations/source_read.py",
        "arrayscope/operations/regions.py",
    ),
    # Real terminal invocations exercise early headless dispatch, replay, and
    # atomic publication outside the parent's trace, including SIGINT races.
    "tests/app/test_batch_cli.py": (
        "arrayscope/__main__.py",
        "arrayscope/app/batch_cli.py",
        "arrayscope/io/recipe_replay.py",
        "arrayscope/io/region_export.py",
        "arrayscope/io/numpy_save.py",
        "arrayscope/operations/recipes.py",
    ),
    "tests/app/test_batch_process_integration.py": (
        "arrayscope/__main__.py",
        "arrayscope/app/batch_cli.py",
        "arrayscope/io/recipe_replay.py",
        "arrayscope/io/region_export.py",
        "arrayscope/io/numpy_save.py",
        "arrayscope/operations/recipes.py",
    ),
    # Fresh gw0-labeled interpreters exercise the import-time filesystem
    # setup, which cannot be re-run inside the parent pytest process.
    "tests/app/test_qsettings_isolation.py": (
        "tests/conftest.py",
        "tests/filesystem_isolation.py",
    ),
    # `python tools/demo_recorder.py --smoke ...`
    "tests/app/test_demo_recorder_smoke.py": ("tools/demo_recorder.py",),
    # `python tools/demo_inspector.py --strict ...`. The findings tests in the
    # same file load the inspector in-process and trace normally; this covers
    # what only the child reaches — argument handling, exit codes, and the
    # ffmpeg decode of a real encoded file.
    "tests/app/test_demo_inspector.py": ("tools/demo_inspector.py",),
    # `python -c "import arrayscope, pyqtgraph.Qt as Qt"` — the binding
    # preference has to hold on a *fresh* interpreter, which is the whole point
    # of the test and the reason it cannot be traced.
    "tests/app/test_qt_binding.py": (
        "arrayscope/__init__.py",
        "arrayscope/app/qt_binding.py",
        "arrayscope/app/launch.py",
    ),
    # `python -c "... from arrayscope.app.launch import _prepare_qt_environment"`
    # — the Qt logging-category rules are process-global mutable state, so the
    # only honest place to assert on them is a fresh interpreter. Both files are
    # named because both are only *half* covered in-process: the two in-process
    # tests here call `install_wgpu_log_noise_filter`, but
    # `apply_qt_logging_noise_rules` and the `_prepare_qt_environment` wiring
    # that reaches it run nowhere except this child.
    # wgpu_imageview2d joins them for the deferred-install tests: the hook's
    # ordering contract against `import wgpu` is only reproducible in a fresh
    # interpreter (the damage is done at first import), so the child is the
    # ONLY place that guard runs.
    "tests/app/test_log_noise.py": (
        "arrayscope/app/launch.py",
        "arrayscope/app/log_noise.py",
        "arrayscope/display/wgpu_imageview2d.py",
    ),
    # `python experiments/wgpu_gate_b/probe_native_x11.py <out>` — the tier-0
    # native-child probe needs its own Qt + wgpu instance, so it cannot share
    # this interpreter. The probe is the thing under test; nothing in-process
    # covers it.
    "tests/gpu_interaction/test_wgpu_native_x11_pin.py": (
        "experiments/wgpu_gate_b/probe_native_x11.py",
    ),
    # `git commit` / `git merge` / `git worktree add` in throwaway repos, which
    # run the hooks as shell scripts. They are the subject, they are not Python,
    # and nothing in-process executes them — so editing a hook has to select
    # this file or the hook ships unguarded.
    "tests/app/test_git_hooks.py": (
        ".githooks/pre-commit",
        ".githooks/post-merge",
        ".githooks/post-checkout",
    ),
    # `bash tools/load_fragility ...` in throwaway sandboxes. The script is the
    # subject, it is not Python, and the two tests that read its text read it as
    # data — which testmon does not record — so only this declaration ties the
    # co-operation rules to the file that implements them.
    "tests/tools/test_load_fragility.py": ("tools/load_fragility",),
    # `python -c "import arrayscope.io.numpy_save"` (asserts pyqtgraph stays out
    # of sys.modules).
    "tests/io/test_numpy_save.py": ("arrayscope/io/numpy_save.py",),
    # `python -m arrayscope.tools.profile_montage_workflow ...` (ring 3).
    "tests/stress/test_synthetic_stress_matrix.py": (
        "arrayscope/tools/profile_montage_workflow.py",
    ),
    # One opt-in test in this file runs the profiler under py-spy
    # (ARRAYSCOPE_RUN_PY_SPY_SMOKE=1); every other subprocess here is
    # monkeypatched and traced normally.
    "tests/app/test_profile_montage_workflow.py": ("arrayscope/tools/profile_montage_workflow.py",),
    # Spawns, but the child is a client script generated from a template inside
    # this same test file, and the compositor launcher it drives runs
    # in-process. Nothing to re-attach — declared so the guard stays honest
    # rather than silent.
    "tests/gpu_interaction/test_headless_exact_window.py": (),
    # `python experiments/wgpu_gate_b/probe_native_wayland.py` (ring 4).
    "tests/gpu_interaction/test_wgpu_native_wayland_pin.py": (
        "experiments/wgpu_gate_b/probe_native_wayland.py",
    ),
    # The per-test-batch and skipped-record oracles run a real pytest+testmon
    # session over files generated in tmp_path: a second live testmon inside
    # this process would fight the one recording the oracle itself.
    # `tests/testmon_policy.py` needs no entry — the tests here already execute
    # it in-process. `tests/skip_ledger.py` does: the child's conftest is the
    # only thing that ever calls `SkipLedger.observe` and `save`, and the
    # in-process half of these tests reaches no further than `read_skips`. A
    # change to the outcome rule would otherwise not select the oracle that
    # exists to check it — which is this file's own failure mode, one level in.
    "tests/app/test_test_selection.py": ("tests/skip_ledger.py",),
}
