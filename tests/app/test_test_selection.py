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
