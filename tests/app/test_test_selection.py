"""Guards for change-driven test selection (``tests/testmon_policy.py``).

Ring 0 — no Qt, no rendering. These pin the seams where selection can fail
*silently*, which is the only way it can hurt: a test that is never selected
reads exactly like a test that keeps passing.

Two of them are the loud channel for testmon's blind spot. Coverage cannot see
into a child process, so a test that spawns one must declare what that child
runs; if the declaration is missing or stale, the test stops being selected by
anything and nothing else in the suite would notice.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import testmon_policy

REPO_ROOT = testmon_policy.REPO_ROOT

#: Calls that start a real child process. A test that only monkeypatches
#: ``subprocess.run`` — by string target or by attribute — produces no such
#: call node and is fully traced, so it must not be dragged in here.
_SPAWNING_CALLS = {
    ("subprocess", "run"),
    ("subprocess", "Popen"),
    ("subprocess", "call"),
    ("subprocess", "check_call"),
    ("subprocess", "check_output"),
    ("multiprocessing", "Process"),
    ("multiprocessing", "Pool"),
}


def _spawns_a_child_process(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and (function.value.id, function.attr) in _SPAWNING_CALLS
        ):
            return True
    return False


def test_every_test_that_spawns_a_child_process_declares_what_it_runs():
    """A child process is invisible to coverage; the declaration puts it back.

    Without an entry in ``OUT_OF_PROCESS_DEPENDENCIES`` the file the child
    executes is not part of any test's recorded dependencies, so editing it
    selects nothing and the test silently stops guarding it.
    """

    spawning = {
        path.resolve().relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "tests").rglob("test_*.py")
        if _spawns_a_child_process(path)
    }
    undeclared = sorted(spawning - set(testmon_policy.OUT_OF_PROCESS_DEPENDENCIES))
    assert not undeclared, (
        "these tests start a child process that coverage cannot trace, so they "
        "need an entry in tests/testmon_policy.py OUT_OF_PROCESS_DEPENDENCIES "
        "naming the file the child runs (an empty tuple is fine when the child "
        f"only runs code this test file already contains): {undeclared}"
    )


def test_declared_entry_points_still_exist():
    """A declaration that points at a moved file is worse than none.

    It looks like coverage and provides none: the test is then selected by a
    path that can never change again.
    """

    missing = [
        (test_file, entry_point)
        for test_file, declared in testmon_policy.OUT_OF_PROCESS_DEPENDENCIES.items()
        for entry_point in declared
        if not (REPO_ROOT / entry_point).exists()
    ]
    assert not missing, f"declared child-process entry points have moved: {missing}"


def test_declarations_name_real_test_files():
    missing = [
        test_file
        for test_file in testmon_policy.OUT_OF_PROCESS_DEPENDENCIES
        if not (REPO_ROOT / test_file).exists()
    ]
    assert not missing, f"declarations name test files that no longer exist: {missing}"


def test_injection_records_the_declared_entry_point_as_a_whole_file():
    """The declared file must arrive as *every* line, not a token line.

    testmon fingerprints a dependency from the AST blocks covering the lines it
    was given. One line would pin one block, so a change anywhere else in the
    entry point would read as unaffected — the exact silent failure this
    mechanism exists to prevent.
    """

    test_file, declared = next(
        (name, files) for name, files in testmon_policy.OUT_OF_PROCESS_DEPENDENCIES.items() if files
    )
    entry_point = declared[0]
    node_id = f"{test_file}::test_something"
    report = SimpleNamespace(nodes_files_lines={node_id: {test_file: {1, 2}}})

    testmon_policy.inject_out_of_process_dependencies(report, REPO_ROOT)

    recorded = report.nodes_files_lines[node_id]
    assert recorded[test_file] == {1, 2}, "existing coverage must survive untouched"
    expected = len((REPO_ROOT / entry_point).read_bytes().splitlines())
    assert recorded[entry_point] == set(range(1, expected + 1))


def test_injection_leaves_undeclared_tests_alone():
    report = SimpleNamespace(
        nodes_files_lines={"tests/core/test_nothing.py::test_x": {"arrayscope/__init__.py": {1}}}
    )
    testmon_policy.inject_out_of_process_dependencies(report, REPO_ROOT)
    assert report.nodes_files_lines == {
        "tests/core/test_nothing.py::test_x": {"arrayscope/__init__.py": {1}}
    }


def _config(**options):
    return SimpleNamespace(option=SimpleNamespace(**options), args=[])


def test_selection_is_on_by_default():
    decision = testmon_policy.decide(_config())
    assert decision.active
    assert decision.reason == "default"


def test_selection_refuses_to_truncate_a_coverage_run():
    """A coverage number that describes a subset of the suite is a false number."""

    decision = testmon_policy.decide(_config(cov_source=["arrayscope"]))
    assert not decision.active


def test_selection_is_off_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    assert not testmon_policy.decide(_config()).active


def test_ci_can_opt_one_job_back_in(monkeypatch):
    """The override must configure testmon, not merely permit it.

    ``explicit`` means "the developer set the testmon options by hand, do not
    touch them". Marking the environment override explicit would leave the
    options unset and silently run the whole suite instead — a job that looks
    selective and is not.
    """

    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("ARRAYSCOPE_TESTMON", "1")
    decision = testmon_policy.decide(_config())
    assert decision.active
    assert not decision.explicit


def test_no_testmon_wins_over_the_default():
    config = _config()
    vars(config.option)["no-testmon"] = True
    assert not testmon_policy.decide(config).active


def test_an_explicit_flag_is_left_exactly_as_given():
    """The default must never rewrite a deliberate ``--testmon-noselect``."""

    decision = testmon_policy.decide(_config(testmon_noselect=True))
    assert decision.active
    assert decision.explicit


@pytest.mark.parametrize(
    ("options", "arguments"),
    [
        ({"keyword": "montage"}, []),
        ({"markexpr": "gpu_interaction"}, []),
        ({"lf": True}, []),
        ({}, ["tests/ui/test_prefetch.py::test_one"]),
    ],
)
def test_a_manual_selector_stops_the_map_from_narrowing_the_run(options, arguments):
    """Worker sizing must not assume a narrow run when testmon will not narrow it."""

    config = _config(**options)
    config.args = arguments
    assert not testmon_policy.selects(config)


class _Item:
    def __init__(self, nodeid):
        self.nodeid = nodeid


def _regroup(sorted_ids, declaration_ids, groups):
    order = {nodeid: index for index, nodeid in enumerate(declaration_ids)}
    items = [_Item(nodeid) for nodeid in sorted_ids]
    return [item.nodeid for item in testmon_policy.regroup_coupled_items(items, order, groups)]


def test_coupled_tests_survive_the_duration_sort():
    """The whole point: a sort that separated the pair reported a phantom bug."""

    declaration = ["f.py::test_01", "f.py::test_02", "g.py::test_fast"]
    groups = {"f.py::test_01": "identity", "f.py::test_02": "identity"}
    # What the duration sort produces: the cheap unrelated test lands between.
    reordered = ["f.py::test_02", "g.py::test_fast", "f.py::test_01"]

    assert _regroup(reordered, declaration, groups) == [
        "f.py::test_01",
        "f.py::test_02",
        "g.py::test_fast",
    ]


def test_regrouping_leaves_the_rest_of_the_sort_alone():
    """A coupled group must not drag unrelated tests back to declaration order."""

    declaration = ["a.py::test_slow", "b.py::test_quick", "c.py::test_solo"]
    reordered = ["b.py::test_quick", "c.py::test_solo", "a.py::test_slow"]

    assert _regroup(reordered, declaration, {}) == reordered
    assert _regroup(reordered, declaration, {"a.py::test_slow": "lonely"}) == reordered


def test_a_group_stays_where_its_earliest_member_landed():
    """An expensive group must not promote itself to the front of the run."""

    declaration = ["f.py::test_01", "f.py::test_02", "g.py::test_fast"]
    groups = {"f.py::test_01": "identity", "f.py::test_02": "identity"}
    reordered = ["g.py::test_fast", "f.py::test_02", "f.py::test_01"]

    assert _regroup(reordered, declaration, groups)[0] == "g.py::test_fast"


def test_regrouping_survives_a_partly_deselected_group():
    """Selection can leave one member behind; the survivor must still run."""

    declaration = ["f.py::test_01", "f.py::test_02"]
    groups = {"f.py::test_01": "identity", "f.py::test_02": "identity"}

    assert _regroup(["f.py::test_02"], declaration, groups) == ["f.py::test_02"]


def _order(sorted_ids, declaration_ids, seconds):
    order = {nodeid: index for index, nodeid in enumerate(declaration_ids)}
    items = [_Item(nodeid) for nodeid in sorted_ids]
    return [item.nodeid for item in testmon_policy.order_for_makespan(items, seconds, order)]


def test_the_heavy_files_go_longest_first():
    """Shortest-first ends the run with one worker holding the longest file.

    Simulated on this suite's own durations that costs 100 s of makespan
    (296.5 s against a 196 s floor) and leaves seven workers idle at the end.
    """

    declaration = ["a.py::t1", "b.py::t1", "c.py::t1"]
    seconds = {"a.py": 3.0, "b.py": 40.0, "c.py": 5.0}

    assert _order(declaration, declaration, seconds) == ["b.py::t1", "c.py::t1", "a.py::t1"]


def test_the_quick_files_still_report_first():
    """Pure longest-first says nothing for 49 s; the sub-500 ms prefix is 0.3%."""

    declaration = ["heavy.py::t1", "quick.py::t1", "medium.py::t1"]
    seconds = {"heavy.py": 40.0, "quick.py": 0.02, "medium.py": 5.0}

    assert _order(declaration, declaration, seconds) == [
        "quick.py::t1",
        "heavy.py::t1",
        "medium.py::t1",
    ]


def test_an_untimed_file_leads():
    """Unknown cost is more safely assumed large than small."""

    declaration = ["slow.py::t1", "new.py::t1"]
    seconds = {"slow.py": 40.0}

    assert _order(declaration, declaration, seconds)[0] == "new.py::t1"


def test_declaration_order_is_restored_inside_a_file():
    """A run order that differs every run makes every red ambiguous.

    Recorded durations change on every run, so testmon's sort reshuffles each
    file differently each time — which turned latent order dependencies into
    failures that moved between files from run to run.
    """

    declaration = ["f.py::t1", "f.py::t2", "f.py::t3"]
    reshuffled = ["f.py::t3", "f.py::t1", "f.py::t2"]

    assert _order(reshuffled, declaration, {"f.py": 3.0}) == declaration


def test_a_file_is_never_split_across_the_run():
    """--dist loadfile hands out whole files; interleaving them means nothing."""

    declaration = ["a.py::t1", "b.py::t1", "a.py::t2"]
    ordered = _order(declaration, declaration, {"a.py": 5.0, "b.py": 9.0})

    assert ordered == ["b.py::t1", "a.py::t1", "a.py::t2"]


class _FakeTestmonSelect:
    def __init__(self, deselected_tests, deselected_files):
        self.deselected_tests = list(deselected_tests)
        self.deselected_files = list(deselected_files)


class _FakePluginManager:
    def __init__(self, plugin):
        self._plugin = plugin

    def get_plugin(self, name):
        return self._plugin if name == "TestmonSelect" else None


_RED = "tests/b/test_b.py::test_red"


def _config_with_known_red():
    """testmon's view after a run that left one test red: it stays selected."""

    plugin = _FakeTestmonSelect(
        deselected_tests=["tests/a/test_a.py::test_green"],
        deselected_files=["tests/a/test_a.py"],
    )
    data = SimpleNamespace(
        stable_test_names={"tests/a/test_a.py::test_green", _RED},
        stable_files={"tests/a/test_a.py", "tests/b/test_b.py"},
        all_tests={
            "tests/a/test_a.py::test_green": {"failed": False},
            _RED: {"failed": True},
        },
    )
    config = SimpleNamespace(
        pluginmanager=_FakePluginManager(plugin),
        testmon_data=data,
        testmon_config=SimpleNamespace(select=True, collect=True),
    )
    return config, plugin


def test_an_inherited_red_is_not_re_run(monkeypatch):
    """A red that was already failing here tells you nothing new.

    Re-running all of them measured 124 s on an otherwise clean tree — the
    whole inner loop, spent re-confirming what everybody already knew.
    """

    monkeypatch.delenv("ARRAYSCOPE_TESTMON_RERUN_FAILING", raising=False)
    config, plugin = _config_with_known_red()

    assert testmon_policy.apply_known_red_policy(config, new_reds=set()) == 1
    assert _RED in plugin.deselected_tests
    assert "tests/b/test_b.py" in plugin.deselected_files


def test_a_red_this_checkout_broke_always_runs(monkeypatch):
    """The hazard this exists for: the map cannot vouch for a red it recorded.

    Break T, then edit something unrelated. T's dependencies are unchanged
    *since the map was written* -- but the map was written by the run that broke
    it, so "unchanged" is worthless evidence and skipping T would hide the
    regression minutes after introducing it.
    """

    monkeypatch.delenv("ARRAYSCOPE_TESTMON_RERUN_FAILING", raising=False)
    config, plugin = _config_with_known_red()

    assert testmon_policy.apply_known_red_policy(config, new_reds={_RED}) == 0
    assert _RED not in plugin.deselected_tests
    assert "tests/b/test_b.py" not in plugin.deselected_files, (
        "the file has to stay collectable or the red inside it cannot run"
    )


def test_the_flag_re_runs_the_inherited_ones_too(monkeypatch):
    monkeypatch.setenv("ARRAYSCOPE_TESTMON_RERUN_FAILING", "1")
    config, plugin = _config_with_known_red()

    assert testmon_policy.apply_known_red_policy(config, new_reds=set()) == 0
    assert _RED not in plugin.deselected_tests


def test_an_exhaustive_run_is_left_untouched(monkeypatch):
    """Under --testmon-noselect the lists only decide order, so do not touch them.

    testmon runs its "deselected" group last rather than dropping it, so
    rewriting those lists on an exhaustive run reshuffles it for no benefit —
    and this suite has enough latent order sensitivity that a reshuffle costs
    roughly one spurious failure per run.
    """

    monkeypatch.delenv("ARRAYSCOPE_TESTMON_RERUN_FAILING", raising=False)
    config, plugin = _config_with_known_red()
    config.testmon_config = SimpleNamespace(select=False, collect=True)
    before = list(plugin.deselected_tests)

    assert testmon_policy.apply_known_red_policy(config, new_reds=set()) == 0
    assert plugin.deselected_tests == before


def test_a_red_test_whose_dependencies_changed_still_runs(monkeypatch):
    """Skipping known reds must never reach a test the change actually touches."""

    monkeypatch.delenv("ARRAYSCOPE_TESTMON_RERUN_FAILING", raising=False)
    plugin = _FakeTestmonSelect(deselected_tests=[], deselected_files=[])
    config = SimpleNamespace(
        pluginmanager=_FakePluginManager(plugin),
        # An affected test is absent from the stable sets by construction.
        testmon_data=SimpleNamespace(stable_test_names=set(), stable_files=set(), all_tests={}),
        testmon_config=SimpleNamespace(select=True, collect=True),
    )

    testmon_policy.apply_known_red_policy(config, new_reds=set())
    assert not plugin.deselected_tests


def test_the_baseline_prefers_the_upstream_over_main(monkeypatch):
    """A branch stacked on another branch measures itself against its parent.

    Against ``main`` it would inherit the parent branch's entire diff, and
    ``--since`` would report the parent's work as this branch's.
    """

    asked = []

    def fake_git(rootdir, *arguments, binary=False):
        asked.append(arguments)
        return "abc123\n" if arguments[0] == "merge-base" else None

    monkeypatch.delenv("ARRAYSCOPE_BASELINE_REF", raising=False)
    monkeypatch.setattr(testmon_policy, "_git", fake_git)
    ref, merge_base = testmon_policy.resolve_baseline(REPO_ROOT, None)

    assert ref == "@{upstream}"
    assert merge_base == "abc123"


def test_an_explicit_baseline_wins(monkeypatch):
    """A branch split in two has no inferable parent; naming one is the answer."""

    monkeypatch.setattr(
        testmon_policy,
        "_git",
        lambda rootdir, *arguments, binary=False: "deadbeef\n",
    )
    assert testmon_policy.resolve_baseline(REPO_ROOT, "origin/release")[0] == "origin/release"


def test_the_baseline_can_be_pinned_per_worktree(monkeypatch):
    monkeypatch.setenv("ARRAYSCOPE_BASELINE_REF", "feature/parent")
    monkeypatch.setattr(
        testmon_policy,
        "_git",
        lambda rootdir, *arguments, binary=False: "cafe\n",
    )
    assert testmon_policy.resolve_baseline(REPO_ROOT, None)[0] == "feature/parent"


def test_an_unresolvable_baseline_is_an_error_not_a_silent_default(monkeypatch):
    """Silently falling back would report the wrong branch's work as yours."""

    monkeypatch.delenv("ARRAYSCOPE_BASELINE_REF", raising=False)
    monkeypatch.setattr(testmon_policy, "_git", lambda rootdir, *a, binary=False: None)
    with pytest.raises(testmon_policy.UsageError, match="no baseline"):
        testmon_policy.resolve_baseline(REPO_ROOT, None)


def test_since_compares_methods_not_whole_files(monkeypatch):
    """File-level would drag in every test that merely recorded the file.

    Measured on a four-file diff including tests/conftest.py: 41 tests
    method-level against 55 file-level, and the gap grows with how widely the
    touched code is executed.
    """

    from testmon.db import checksums_to_blob

    unchanged, changed = 111, 222

    class _Row(dict):
        def __getitem__(self, key):
            return dict.__getitem__(self, key)

    data = SimpleNamespace(
        exec_id=1,
        db=SimpleNamespace(
            _test_execution_fk_column=lambda: "environment_id",
            con=SimpleNamespace(
                execute=lambda *args: SimpleNamespace(
                    fetchall=lambda: [
                        _Row(
                            test_name="t.py::untouched",
                            filename="src.py",
                            method_checksums=checksums_to_blob([unchanged]),
                        ),
                        _Row(
                            test_name="t.py::touched",
                            filename="src.py",
                            method_checksums=checksums_to_blob([changed]),
                        ),
                    ]
                )
            ),
        ),
    )

    reached = testmon_policy.tests_reached_since_baseline(data, {"src.py": [unchanged]})
    assert reached == {"t.py::touched"}


def test_a_non_python_change_reaches_everything_that_uses_it(monkeypatch):
    """A fixture array or an icon has no method structure to compare.

    ``None`` checksums are what testmon's comparison reads as "all of it
    changed", so the file-level fallback falls out of the same code path
    instead of needing a second one.
    """

    def fake_git(rootdir, *arguments, binary=False):
        if arguments[0] == "diff":
            return "tests/fixtures/trace.jsonl\narrayscope/render/lod.py\n"
        if arguments[0] == "ls-files":
            return ""
        return None  # `git show` is never reached for a non-Python path

    monkeypatch.setattr(testmon_policy, "_git", fake_git)
    checksums = testmon_policy.baseline_method_checksums(REPO_ROOT, "abc123")

    assert checksums["tests/fixtures/trace.jsonl"] is None
    assert checksums["arrayscope/render/lod.py"] is None, "unreadable at the baseline == changed"


def _ledger(tmp_path, previous, populated=True):
    from tests import red_ledger

    return red_ledger.RedLedger.load(
        tmp_path / ".testmondata", "offscreen", previous, map_was_populated=populated
    )


def test_a_test_that_used_to_pass_here_is_recorded_as_broken(tmp_path):
    ledger = _ledger(tmp_path, {"t.py::a": False})
    ledger.record("t.py::a", failed=True)
    assert ledger.new_reds == {"t.py::a"}


def test_a_test_that_was_already_failing_here_is_not(tmp_path):
    """The incumbent reds are the whole reason the exemption exists."""

    ledger = _ledger(tmp_path, {"t.py::a": True})
    ledger.record("t.py::a", failed=True)
    assert ledger.new_reds == set()


def test_a_new_test_born_failing_counts_as_broken(tmp_path):
    """Otherwise a test you just wrote and left red goes quiet after one run."""

    ledger = _ledger(tmp_path, {"other.py::x": False})
    ledger.record("t.py::brand_new", failed=True)
    assert ledger.new_reds == {"t.py::brand_new"}


def test_the_first_recording_pass_is_an_inventory_not_a_regression(tmp_path):
    """With an empty map every test is unseen; none of them just broke."""

    ledger = _ledger(tmp_path, {}, populated=False)
    ledger.record("t.py::a", failed=True)
    assert ledger.new_reds == set()


def test_a_fixed_test_leaves_the_ledger(tmp_path):
    ledger = _ledger(tmp_path, {"t.py::a": False})
    ledger.record("t.py::a", failed=True)
    ledger.save()

    later = _ledger(tmp_path, {"t.py::a": True})
    assert later.new_reds == {"t.py::a"}
    later.record("t.py::a", failed=False)
    later.save()

    assert _ledger(tmp_path, {"t.py::a": False}).new_reds == set()


def test_the_ledger_is_kept_per_environment(tmp_path):
    """A wayland red is not an offscreen red; the map keys them apart too."""

    from tests import red_ledger

    offscreen = _ledger(tmp_path, {"t.py::a": False})
    offscreen.record("t.py::a", failed=True)
    offscreen.save()

    wayland = red_ledger.RedLedger.load(
        tmp_path / ".testmondata", "wayland+gpu", {"t.py::a": False}, map_was_populated=True
    )
    assert wayland.new_reds == set()
    assert red_ledger.read_new_reds(tmp_path / ".testmondata", "offscreen") == {"t.py::a"}


def test_an_unreadable_ledger_reads_as_empty(tmp_path):
    from tests import red_ledger

    red_ledger.ledger_path(tmp_path / ".testmondata").write_text("{ not json", encoding="utf-8")
    assert red_ledger.read_new_reds(tmp_path / ".testmondata", "offscreen") == set()


def test_seeding_never_overwrites_an_existing_map(tmp_path, monkeypatch):
    """A live map outranks any donor; overwriting one would discard real work."""

    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    existing = tmp_path / "themap"
    existing.write_bytes(b"mine")
    assert testmon_policy.seed_map(tmp_path) is None
    assert existing.read_bytes() == b"mine"


def test_seeding_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    monkeypatch.setenv("ARRAYSCOPE_TESTMON_SEED", "0")
    assert testmon_policy.seed_map(tmp_path) is None
    assert not (tmp_path / "themap").exists()


def test_seeding_copies_rather_than_links(tmp_path, monkeypatch):
    """Two checkouts must never share one file.

    Each run records its own tree's fingerprints and vacuums what the other
    still references, so a shared map degrades to "everything changed" in both
    checkouts — the opposite of what selection is for.
    """

    donor_checkout = tmp_path / "main"
    worktree = tmp_path / "worktree"
    donor_checkout.mkdir()
    worktree.mkdir()
    donor = donor_checkout / "themap"
    donor.write_bytes(b"donor-map")
    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    # Seeding is what this asserts, so the opt-out must not be inherited from the
    # shell. Without this the test fails under `ARRAYSCOPE_TESTMON_SEED=0`, which
    # is exactly what you set while recording a map from scratch.
    monkeypatch.delenv("ARRAYSCOPE_TESTMON_SEED", raising=False)
    monkeypatch.setattr(
        testmon_policy,
        "_best_donor_map",
        lambda rootdir, filename: donor,
    )

    assert testmon_policy.seed_map(worktree) == str(donor)
    seeded = worktree / "themap"
    assert seeded.read_bytes() == b"donor-map"
    assert not seeded.is_symlink()
    assert seeded.stat().st_ino != donor.stat().st_ino, "a hard link is a shared map too"
    assert not list(worktree.glob(".themap.*")), "the staging file must not survive"


def test_testmon_still_exposes_what_the_policy_reaches_into():
    """Fail loudly on a testmon upgrade that renames what we depend on.

    These are internals, deliberately: recording a dependency coverage cannot
    see has no public API. A rename would otherwise degrade selection quietly.
    """

    from testmon.pytest_testmon import TestmonCollect, TestmonSelect
    from testmon.testmon_core import TestmonData

    assert hasattr(TestmonData, "for_local_run")
    assert hasattr(TestmonData, "determine_stable")
    collect_source = inspect.getsource(TestmonCollect)
    assert "nodes_files_lines" in collect_source, (
        "testmon no longer carries per-test coverage on the report; "
        "tests/testmon_policy.py injects declared child-process dependencies there"
    )
    assert "deselected_tests" in inspect.getsource(TestmonSelect), (
        "tests/conftest.py reports the unaffected count from this attribute"
    )


# --------------------------------------------------------------------------- #
# The scope guard: a run that looked at part of the suite must not delete the rest
# --------------------------------------------------------------------------- #


class _FakeTestmonData:
    """Just enough of ``TestmonData`` to observe what gets retained."""

    def __init__(self, recorded):
        self.all_tests = {name: {"failed": False} for name in recorded}
        self.retained = None

    def sync_db_fs_tests(self, retain):
        self.retained = set(retain)


def _scoped_config(arguments, recorded, testpaths=("tests",)):
    return SimpleNamespace(
        rootdir=str(REPO_ROOT),
        args=list(arguments),
        invocation_params=SimpleNamespace(dir=str(REPO_ROOT)),
        testmon_data=_FakeTestmonData(recorded),
        testmon_config=SimpleNamespace(select=True, collect=True),
        getini=lambda name: list(testpaths) if name == "testpaths" else None,
    )


#: Real paths, because ``path_arguments`` resolves against the filesystem and
#: silently drops an argument that does not exist — which would make every
#: assertion below pass for the wrong reason.
_IN_SCOPE = "tests/kernel/test_kernel.py::test_in_scope"
_OUT_OF_SCOPE = "tests/ui/test_window_sync.py::test_out_of_scope"


def test_a_scoped_run_keeps_the_map_entries_it_never_looked_at():
    """The defect this closes, at unit scale.

    testmon deletes every recorded test a run neither collected nor called
    unaffected. On ``pytest tests/render`` that silently drops the tests in
    *other* directories that the working tree affects — and because its whole
    file then looks unaffected, collection never rediscovers it and nothing
    re-adds it. Measured before the guard: an edit to ``arrayscope/core/roi.py``
    affecting 14 tests outside ``tests/kernel``, then ``pytest tests/kernel``,
    then ``pytest`` — 14 map entries deleted and 13 of the 14 tests never run,
    with `pytest`, `pytest --since` and `tools/test_selection.py` all reporting
    nothing affected.
    """

    config = _scoped_config(["tests/kernel"], [_IN_SCOPE, _OUT_OF_SCOPE])
    assert testmon_policy.protect_map_outside_the_scope(config) == 1
    config.testmon_data.sync_db_fs_tests(retain={_IN_SCOPE})
    assert _OUT_OF_SCOPE in config.testmon_data.retained


def test_the_scoped_run_still_cleans_up_inside_its_own_scope():
    """The deletion has a real job; only its reach was wrong.

    A renamed or deleted test inside the scope *was* collected, so judging it
    absent is sound. Protecting everything would leave those entries forever.
    """

    config = _scoped_config(["tests/kernel"], [_IN_SCOPE, _OUT_OF_SCOPE])
    testmon_policy.protect_map_outside_the_scope(config)
    config.testmon_data.sync_db_fs_tests(retain=set())
    assert _IN_SCOPE not in config.testmon_data.retained


def test_a_node_id_run_counts_as_narrow():
    """The sharpest form of the bug, not an exception to it.

    ``pytest file.py::test`` collects that file and nothing else. Measured with
    node ids treated as "no path arguments", which is what
    ``path_arguments`` reports for them: one such run deleted 49 map entries —
    the edit's entire affected set — and the following ``pytest`` ran nothing.
    """

    config = _scoped_config([_IN_SCOPE], [_IN_SCOPE, _OUT_OF_SCOPE])
    assert not testmon_policy.collected_the_whole_suite(config)
    assert testmon_policy.protect_map_outside_the_scope(config) == 1


def test_a_whole_suite_run_leaves_testmons_own_bookkeeping_alone():
    """A bare ``pytest`` carries ``testpaths`` as an argument and is not scoped.

    Reading that as a scope would arm the guard on every ordinary run and stop
    the map ever shedding a renamed test.
    """

    config = _scoped_config(["tests"], [_IN_SCOPE, _OUT_OF_SCOPE])
    assert testmon_policy.collected_the_whole_suite(config)
    assert testmon_policy.protect_map_outside_the_scope(config) == 0
    config.testmon_data.sync_db_fs_tests(retain={_IN_SCOPE})
    assert config.testmon_data.retained == {_IN_SCOPE}


def test_the_guard_is_installed_once():
    """Both the controller and the workers install it; wrapping twice would not
    break the result, but it would make the retained set impossible to reason
    about. The second call is a no-op."""

    config = _scoped_config(["tests/kernel"], [_IN_SCOPE, _OUT_OF_SCOPE])
    assert testmon_policy.protect_map_outside_the_scope(config) == 1
    assert testmon_policy.protect_map_outside_the_scope(config) == 0


# --------------------------------------------------------------------------- #
# Executed-function coverage read out of the map (tests/coverage_map.py)
# --------------------------------------------------------------------------- #

_SAMPLE = '''\
"""A module docstring."""

CONSTANT = 1


def covered(value):
    if value:
        return value
    return 0


class Holder:
    attribute = 2

    def method(self):
        return self.attribute
'''


def _sample_tree(tmp_path, source=_SAMPLE, name="pkg/module.py"):
    from tests import coverage_map

    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return coverage_map, path.relative_to(tmp_path).as_posix()


def test_the_module_block_is_reported_as_reach_never_as_a_function():
    """It spans line one to EOF, so "covered" only means "some test touched the
    file". Counting it as a function is what makes a merely-imported module read
    as covered — measured at up to 36 points of disagreement with coverage.py on
    a single file, in that direction."""

    from tests import coverage_map

    functions, module_checksum = coverage_map.functions_of(REPO_ROOT, "tests/coverage_map.py")
    assert module_checksum is not None
    assert functions, "the module has functions"
    assert all(function.name != "unknown" for function in functions), (
        "testmon names the module body 'unknown'; it must not be in the function count"
    )


def test_class_bodies_are_module_level_and_methods_are_not(tmp_path):
    """The shape the metric depends on: only ``def`` bodies are blocks, so a
    class statement lives in the module block while its methods do not."""

    coverage_map, relative = _sample_tree(tmp_path)
    functions, _ = coverage_map.functions_of(tmp_path, relative)
    assert sorted(function.name for function in functions) == ["covered", "method"]


def test_a_function_counts_as_covered_when_any_one_of_its_lines_ran(tmp_path):
    """The metric's upward bias, pinned rather than hidden: a function abandoned
    at its first guard is indistinguishable from one that ran to completion."""

    from testmon.process_code import Module, create_fingerprint

    coverage_map, relative = _sample_tree(tmp_path)
    module = Module(source_code=_SAMPLE, filename=relative, rootdir=str(tmp_path))
    functions, _ = coverage_map.functions_of(tmp_path, relative)
    entered = next(function for function in functions if function.name == "covered")
    # Only the `if value:` line of `covered` — nothing else in the file.
    hit = set(create_fingerprint(module, [entered.line]))
    assert entered.checksum in hit


def test_a_functions_identity_survives_an_insertion_above_it(tmp_path):
    """Why the sidecar stores a body digest and not testmon's checksum.

    ``Block.code`` carries the block's index within the file, so inserting a
    function at the top changes every checksum below it. Comparing baselines on
    that would report a whole file as changed after a one-line insertion — and
    would churn the "newly covered / no longer" delta into noise.
    """

    coverage_map, relative = _sample_tree(tmp_path)
    before = {
        function.name: (function.checksum, function.body)
        for function in coverage_map.functions_of(tmp_path, relative)[0]
    }
    (tmp_path / relative).write_text(
        "def inserted():\n    return 1\n\n\n" + _SAMPLE, encoding="utf-8"
    )
    after = {
        function.name: (function.checksum, function.body)
        for function in coverage_map.functions_of(tmp_path, relative)[0]
    }
    assert after["covered"][0] != before["covered"][0], "testmon's checksum does shift"
    assert after["covered"][1] == before["covered"][1], "the body digest must not"


def test_two_identical_bodies_in_one_file_stay_two_functions(tmp_path):
    """Otherwise the identities collapse and the baseline arithmetic stops adding
    up — a file could lose a function and report no change."""

    coverage_map, relative = _sample_tree(
        tmp_path, "def a():\n    return 1\n\n\ndef b():\n    return 1\n"
    )
    functions, _ = coverage_map.functions_of(tmp_path, relative)
    assert len({function.body for function in functions}) == 1, "same body, same digest"
    assert len({function.identity for function in functions}) == 2


def test_the_structure_cache_only_reparses_what_changed(tmp_path):
    """Parsing the tree costs ~2.3 s, more than a small selected run takes end to
    end, so a warm run must parse nothing."""

    coverage_map, relative = _sample_tree(tmp_path)
    cold = coverage_map.Structure()
    entry = cold.for_file(tmp_path, relative)
    assert cold.parsed == 1

    warm = coverage_map.Structure(cold.payload())
    assert warm.for_file(tmp_path, relative) == entry
    assert warm.parsed == 0, "unchanged content must not be re-parsed"

    (tmp_path / relative).write_text(_SAMPLE + "\n\ndef added():\n    return 2\n")
    again = coverage_map.Structure(cold.payload())
    again.for_file(tmp_path, relative)
    assert again.parsed == 1, "changed content must be"


def test_the_cache_forgets_a_deleted_file(tmp_path):
    """The payload is what this pass saw, so a removed file expires rather than
    propping the denominator up forever."""

    coverage_map, relative = _sample_tree(tmp_path)
    cold = coverage_map.Structure()
    cold.for_file(tmp_path, relative)
    second = coverage_map.Structure(cold.payload())
    assert second.payload() == {}


def test_drift_names_what_was_gained_and_lost():
    from tests import coverage_map

    reach = coverage_map.Reach(
        environment="offscreen",
        covered=2,
        total=3,
        files_reached=1,
        files_total=1,
        identities=frozenset({"a", "b"}),
    )
    drift = coverage_map.drift(reach, {"covered": 2, "total": 3, "identities": ["a", "c"]})
    assert drift.gained == ("b",)
    assert drift.lost == ("c",)
    assert drift.moved


def test_no_baseline_means_no_delta_rather_than_a_zero():
    """A fresh clone has nothing to compare against, and inventing a comparison
    is worse than not having one."""

    from tests import coverage_map

    reach = coverage_map.Reach(
        environment="offscreen", covered=1, total=2, files_reached=1, files_total=1
    )
    assert coverage_map.drift(reach, None) is None
    assert coverage_map.drift(reach, {}) is None


def test_an_unreadable_map_measures_as_nothing(tmp_path):
    """Every degradation here has to be "no number", never a wrong one."""

    from tests import coverage_map

    assert coverage_map.read_map(tmp_path / "absent", "offscreen") is None
    assert coverage_map.measure(REPO_ROOT, tmp_path / "absent", "offscreen") is None


def test_the_denominator_is_the_one_the_coverage_job_measures():
    """``[tool.coverage.run] source`` declared once, so the map-derived figure and
    CI's cannot silently describe different trees."""

    from tests import coverage_map

    assert coverage_map.source_roots(REPO_ROOT) == ("arrayscope",)


def test_the_sidecar_round_trips_and_a_broken_one_reads_as_empty(tmp_path):
    from tests import coverage_map

    map_path = tmp_path / ".testmondata"
    coverage_map.write_sidecar(
        map_path, structure={"a.py": {"fsha": "x"}}, baselines={"offscreen": {"covered": 1}}
    )
    payload = coverage_map.read_sidecar(map_path)
    assert coverage_map.cached_structure(payload) == {"a.py": {"fsha": "x"}}
    assert coverage_map.baselines(payload)["offscreen"]["covered"] == 1

    coverage_map.sidecar_path(map_path).write_text("{not json", encoding="utf-8")
    assert coverage_map.read_sidecar(map_path) == {}
    assert coverage_map.cached_structure({}) == {}
    assert coverage_map.baselines({}) == {}


def test_seeding_carries_the_coverage_sidecar_across(tmp_path, monkeypatch):
    """The donor's baseline is the right baseline for a checkout cut from it.

    That is what makes "since this checkout's baseline" mean "since main" in a
    worktree with nothing to bootstrap — the same reasoning that makes a seeded
    map's reds the inherited ones.
    """

    from tests import coverage_map

    donor_checkout = tmp_path / "main"
    worktree = tmp_path / "worktree"
    donor_checkout.mkdir()
    worktree.mkdir()
    donor = donor_checkout / "themap"
    donor.write_bytes(b"donor-map")
    coverage_map.write_sidecar(
        donor, structure={"a.py": {"fsha": "x"}}, baselines={"offscreen": {"covered": 7}}
    )
    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    monkeypatch.delenv("ARRAYSCOPE_TESTMON_SEED", raising=False)
    monkeypatch.setattr(testmon_policy, "_best_donor_map", lambda rootdir, filename: donor)

    assert testmon_policy.seed_map(worktree) == str(donor)
    seeded = coverage_map.read_sidecar(worktree / "themap")
    assert coverage_map.baselines(seeded)["offscreen"]["covered"] == 7


def test_seeding_without_a_donor_sidecar_is_not_an_error(tmp_path, monkeypatch):
    """A donor recorded before this existed still seeds its map."""

    donor_checkout = tmp_path / "main"
    worktree = tmp_path / "worktree"
    donor_checkout.mkdir()
    worktree.mkdir()
    donor = donor_checkout / "themap"
    donor.write_bytes(b"donor-map")
    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    monkeypatch.delenv("ARRAYSCOPE_TESTMON_SEED", raising=False)
    monkeypatch.setattr(testmon_policy, "_best_donor_map", lambda rootdir, filename: donor)

    assert testmon_policy.seed_map(worktree) == str(donor)
    assert (worktree / "themap").read_bytes() == b"donor-map"


# --- Reaching the inherited reds without leaving selection ------------------


def test_rerun_reds_is_reachable_as_a_flag_and_as_the_variable(monkeypatch):
    """The flag exists because the variable was not findable when it mattered.

    Fixing an inherited red is the one everyday task selection actively works
    against — it does not re-run those, so even a targeted node id reports
    "deselected" — and the escape everybody reached for instead was
    ``--no-testmon``, trading a few-second loop for the whole suite.
    """

    monkeypatch.delenv("ARRAYSCOPE_TESTMON_RERUN_FAILING", raising=False)

    assert testmon_policy.rerun_known_red_tests() is False
    assert testmon_policy.rerun_known_red_tests(_option_config(rerun_reds=False)) is False
    assert testmon_policy.rerun_known_red_tests(_option_config(rerun_reds=True)) is True

    # The variable still answers on its own, for scripts, CI steps, and any
    # xdist worker handed no command line of its own.
    monkeypatch.setenv("ARRAYSCOPE_TESTMON_RERUN_FAILING", "1")
    assert testmon_policy.rerun_known_red_tests() is True
    assert testmon_policy.rerun_known_red_tests(_option_config(rerun_reds=False)) is True


def test_a_config_without_the_option_falls_back_to_the_variable(monkeypatch):
    """`tools/test_selection.py` builds configs that never registered it."""

    monkeypatch.delenv("ARRAYSCOPE_TESTMON_RERUN_FAILING", raising=False)

    class _NoOptions:
        def getoption(self, name, default=None):
            raise ValueError(f"no option named {name!r}")

    assert testmon_policy.rerun_known_red_tests(_NoOptions()) is False
    monkeypatch.setenv("ARRAYSCOPE_TESTMON_RERUN_FAILING", "yes")
    assert testmon_policy.rerun_known_red_tests(_NoOptions()) is True


def _option_config(**options):
    return SimpleNamespace(getoption=lambda name, default=None: options.get(name, default))
