"""Guards for change-driven test selection (``tests/testmon_policy.py``).

Ring 0 — no Qt, no rendering. These pin the seams where selection can fail
*silently*, which is the only way it can hurt: a test that is never selected
reads exactly like a test that keeps passing.

A group of them is the loud channel for testmon's blind spot. Coverage cannot
see into a child process, so a test that spawns one must declare what that child
runs; if the declaration is missing or stale, the test stops being selected by
anything and nothing else in the suite would notice. A declaration is a static
fact, so every way one can be wrong is answered here, by reading it — including
the way that used to be answered by testmon raising ``IndexError`` mid-suite.
"""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import testmon_policy
from tests.app.conftest import break_a_page, read_map, write_map

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


def _source_tree():
    """testmon's own file reader, with the policy's patch installed."""

    from testmon.testmon_core import SourceTree

    assert testmon_policy.install_extensionless_dependency_support()
    return SourceTree(rootdir=str(REPO_ROOT))


def test_every_declared_entry_point_is_one_testmon_can_key():
    """A declaration testmon cannot fingerprint must fail *here*, not mid-suite.

    ``SourceTree.get_file`` is where a declared path becomes a fingerprint, and
    it used to raise ``IndexError`` on any filename with no dot in it — from
    inside testmon, after the whole suite had run, as ``INTERNALERROR``. Nothing
    in that traceback names the declaration that caused it.

    So the question is asked of the real reader, over the real declarations, in
    a ring-0 test that costs milliseconds: whatever a new entry does to testmon,
    it does it to this test first, with the offending path in the message.
    """

    source_tree = _source_tree()
    unkeyable = []
    for test_file, declared in testmon_policy.OUT_OF_PROCESS_DEPENDENCIES.items():
        for entry_point in declared:
            try:
                module = source_tree.get_file(entry_point)
            except Exception as error:  # the whole point: name it, do not raise it
                unkeyable.append((test_file, entry_point, f"{type(error).__name__}: {error}"))
                continue
            if module is None or not module.blocks:
                unkeyable.append((test_file, entry_point, "no fingerprint"))
    assert not unkeyable, (
        "these declared child-process entry points cannot be fingerprinted by "
        "testmon, so a run that records one dies inside testmon rather than "
        f"failing here: {unkeyable}"
    )


def test_a_non_python_entry_point_is_fingerprinted_as_its_whole_content():
    """Whole file, one block — so any byte of it selects the declaring test.

    The three ``.githooks/*`` entries and ``tools/load_fragility`` are shell,
    and shell is what the declaration mechanism has to be able to hold: nothing
    parses it into functions, so the honest fingerprint is the file itself. This
    pins that, because the failure would be quiet — a script whose fingerprint
    covered less than the file would stop selecting its test for edits outside
    the covered part.
    """

    source_tree = _source_tree()
    script = source_tree.get_file("tools/load_fragility")

    assert script.ext != "py"
    assert len(script.blocks) == 1
    block = script.blocks[0]
    assert block.start == 1
    assert block.end == len((REPO_ROOT / "tools" / "load_fragility").read_bytes().splitlines())

    edited = script.blocks[0].checksum
    from testmon.process_code import Module

    changed = Module(source_code=script.source_code + "\n# one more line\n", ext="")
    assert changed.blocks[0].checksum != edited


def _testmon_extension(path: str) -> str | None:
    """testmon's own unpatched derivation, or ``None`` where it raises."""

    parts = path.rsplit(".", 1)
    return parts[1] if len(parts) > 1 else None


def test_deriving_the_extension_from_the_basename_changes_no_existing_answer():
    """The patch may only stop the raise — never move a file in or out of Python.

    ``Module`` compares ``ext`` against ``"py"`` and nothing else reads it, so
    that comparison is the entire observable behaviour. Checked over every real
    path in this tree rather than argued: wherever testmon answers at all, the
    basename derivation agrees on ``== "py"``.
    """

    roots = ("arrayscope", "tests", "tools", ".githooks", "experiments", "packaging", "wrappers")
    paths = [
        path.relative_to(REPO_ROOT).as_posix()
        for root in roots
        if (REPO_ROOT / root).is_dir()
        for path in (REPO_ROOT / root).rglob("*")
        if path.is_file()
    ]
    # Shapes this tree does not happen to contain, where the two derivations
    # look at different dots.
    paths += ["pkg.d/run", "a.py/b", "x.y/mod.py", ".hidden", "plain"]
    assert len(paths) > 500, "the census is meant to be the tree, not a sample"

    disagreements = [
        path
        for path in paths
        if (theirs := _testmon_extension(path)) is not None
        and (theirs == "py") != (os.path.basename(path).rpartition(".")[2] == "py")
    ]
    assert not disagreements, (
        f"the basename derivation disagrees with testmon on these paths: {disagreements}"
    )


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


def test_selection_is_on_by_default(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
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


def test_ci_can_opt_one_job_back_in_with_an_explicit_flag(monkeypatch):
    """`--testmon` beats the CI refusal; no environment variable does.

    The variable that used to do this was removed on 2026-07-30 as a duplicate.
    A second spelling of "bypass selection" is exactly what escapes a refusal
    built around the first, and the flag already says it.
    """

    monkeypatch.setenv("CI", "true")
    assert not testmon_policy.decide(_config()).active

    config = _config()
    vars(config.option)["testmon"] = True
    decision = testmon_policy.decide(config)
    assert decision.active
    assert decision.explicit


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


def test_an_inherited_red_is_not_re_run():
    """A red that was already failing here tells you nothing new.

    Re-running all of them measured 124 s on an otherwise clean tree — the
    whole inner loop, spent re-confirming what everybody already knew.
    """

    config, plugin = _config_with_known_red()

    assert testmon_policy.apply_known_red_policy(config, new_reds=set()) == 1
    assert _RED in plugin.deselected_tests
    assert "tests/b/test_b.py" in plugin.deselected_files


def test_a_red_this_checkout_broke_always_runs():
    """The hazard this exists for: the map cannot vouch for a red it recorded.

    Break T, then edit something unrelated. T's dependencies are unchanged
    *since the map was written* -- but the map was written by the run that broke
    it, so "unchanged" is worthless evidence and skipping T would hide the
    regression minutes after introducing it.
    """

    config, plugin = _config_with_known_red()

    assert testmon_policy.apply_known_red_policy(config, new_reds={_RED}) == 0
    assert _RED not in plugin.deselected_tests
    assert "tests/b/test_b.py" not in plugin.deselected_files, (
        "the file has to stay collectable or the red inside it cannot run"
    )


def test_the_flag_re_runs_the_inherited_ones_too():
    config, plugin = _config_with_known_red()
    config.getoption = lambda name, default=None: True if name == "rerun_reds" else default

    assert testmon_policy.apply_known_red_policy(config, new_reds=set()) == 0
    assert _RED not in plugin.deselected_tests


def test_an_exhaustive_run_is_left_untouched():
    """Under --testmon-noselect the lists only decide order, so do not touch them.

    testmon runs its "deselected" group last rather than dropping it, so
    rewriting those lists on an exhaustive run reshuffles it for no benefit —
    and this suite has enough latent order sensitivity that a reshuffle costs
    roughly one spurious failure per run.
    """

    config, plugin = _config_with_known_red()
    config.testmon_config = SimpleNamespace(select=False, collect=True)
    before = list(plugin.deselected_tests)

    assert testmon_policy.apply_known_red_policy(config, new_reds=set()) == 0
    assert plugin.deselected_tests == before


def test_a_red_test_whose_dependencies_changed_still_runs():
    """Skipping known reds must never reach a test the change actually touches."""

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
    assert testmon_policy.seed_map(tmp_path, seeding=False) is None
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
    donor = write_map(donor_checkout / "themap", "donor-map")
    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    monkeypatch.setattr(
        testmon_policy,
        "_best_donor_map",
        lambda rootdir, filename: donor,
    )

    assert testmon_policy.seed_map(worktree) == str(donor)
    seeded = worktree / "themap"
    assert read_map(seeded) == "donor-map"
    assert not seeded.is_symlink()
    assert seeded.stat().st_ino != donor.stat().st_ino, "a hard link is a shared map too"
    assert not list(worktree.glob(".themap.*")), "the staging file must not survive"


# --- The map is SQLite, and a copy of SQLite is not a copy of a file --------


def test_the_copy_takes_what_lives_only_in_the_wal(tmp_path, monkeypatch):
    """A plain file copy of the map copies the map's *past*, or worse.

    ``.testmondata`` runs in ``journal_mode=wal``. Recently committed
    transactions live in the ``-wal`` sidecar until a checkpoint folds them
    back, so a copy of the main file alone arrives without them — and that is
    the *benign* half. The other half is what actually bit: a checkpoint
    writing pages into the main file while the copy reads it yields a mixture
    of two page images that is a valid-looking file and a malformed database,
    which then kills an unrelated suite four minutes in with ``INTERNALERROR>
    sqlite3.DatabaseError: database disk image is malformed``.

    Both halves have the same cause — reading the file behind SQLite's back —
    and one primitive fixes both. This asserts the cheap, deterministic half:
    a donor whose latest commit is still in the WAL. The torn-page half cannot
    be raced for reliably in a unit test, so it is stated here rather than
    reproduced; see ``testmon_policy._snapshot_sqlite`` for the measurement.
    """

    import sqlite3

    donor_checkout, worktree = tmp_path / "main", tmp_path / "worktree"
    donor_checkout.mkdir()
    worktree.mkdir()
    donor = write_map(donor_checkout / "themap", "checkpointed")
    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    monkeypatch.setattr(testmon_policy, "_best_donor_map", lambda rootdir, filename: donor)

    # A run holding its map open, exactly as every pytest on this box does:
    # committed, and not yet checkpointed into the file a `cp` would read.
    live = sqlite3.connect(str(donor))
    try:
        live.execute("PRAGMA wal_autocheckpoint=0")
        live.execute("UPDATE marker SET mark = 'only-in-the-wal'")
        live.commit()
        assert testmon_policy.seed_map(worktree) == str(donor)
    finally:
        live.close()

    assert read_map(worktree / "themap") == "only-in-the-wal"


def test_a_donor_that_is_not_a_sound_database_is_not_copied(tmp_path, monkeypatch):
    """Seeding may not launder a broken map into a second checkout.

    This is how one corrupt map became three: the donor was copied on to every
    worktree created after it broke, and each arrived looking like an ordinary
    seeded map. Refusing costs one traced run in a fresh checkout, which is the
    price of having no map at all — the price the whole design already pays
    whenever there is no donor.
    """

    donor_checkout, worktree = tmp_path / "main", tmp_path / "worktree"
    donor_checkout.mkdir()
    worktree.mkdir()
    donor = write_map(donor_checkout / "themap", "poisoned", pages=8)
    break_a_page(donor)
    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    monkeypatch.setattr(testmon_policy, "_best_donor_map", lambda rootdir, filename: donor)

    assert testmon_policy.seed_map(worktree) is None
    assert not (worktree / "themap").exists()
    assert not list(worktree.glob(".themap.*")), "the staging file must not survive"


def test_a_map_that_is_not_a_database_at_all_is_refused(tmp_path, monkeypatch):
    """The other shape a killed copy leaves: a file that is not SQLite."""

    donor_checkout, worktree = tmp_path / "main", tmp_path / "worktree"
    donor_checkout.mkdir()
    worktree.mkdir()
    donor = donor_checkout / "themap"
    donor.write_bytes(b"half a copy, and then the box rebooted")
    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    monkeypatch.setattr(testmon_policy, "_best_donor_map", lambda rootdir, filename: donor)

    assert testmon_policy.seed_map(worktree) is None
    assert not (worktree / "themap").exists()


# --- A corrupt map costs time, never a wall --------------------------------


def test_a_sound_map_is_reported_sound(tmp_path, monkeypatch):
    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    write_map(tmp_path / "themap", "fine", pages=4)
    assert testmon_policy.map_health(tmp_path) is None
    assert testmon_policy.quarantine_broken_map(tmp_path) is None
    assert (tmp_path / "themap").exists()


def test_a_missing_or_empty_map_is_not_a_corrupt_one(tmp_path, monkeypatch):
    """Both are states every reader already handles, and neither is damage."""

    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    assert testmon_policy.map_health(tmp_path) is None
    (tmp_path / "themap").write_bytes(b"")
    assert testmon_policy.map_health(tmp_path) is None
    assert testmon_policy.quarantine_broken_map(tmp_path) is None


def test_a_broken_map_is_moved_aside_rather_than_opened(tmp_path, monkeypatch):
    """The whole point: a corrupt map costs a rebuild, never the session.

    Every reader in this module degrades politely on a malformed map — and then
    testmon's own ``init_testmon_data`` opens the same file seconds later and
    takes the run down from inside ``initiate_execution``, out of a traceback
    that names neither the map nor the checkout that copied it in.

    Moved, not deleted: 12 MB is a cheap price for being able to ask *which*
    pages later, and the question cannot be asked twice.
    """

    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    map_file = write_map(tmp_path / "themap", "poisoned", pages=8)
    break_a_page(map_file)
    (tmp_path / "themap-wal").write_bytes(b"frames describing a file that is leaving")

    complaint = testmon_policy.map_health(tmp_path)
    assert complaint, "a scribbled-over b-tree page must not read as healthy"

    quarantined = testmon_policy.quarantine_broken_map(tmp_path)
    assert quarantined is not None
    said, where = quarantined
    assert said == complaint
    assert not map_file.exists(), "the run must find no map, which every path handles"
    assert where.exists()
    assert where.name.endswith(testmon_policy.QUARANTINE_SUFFIX)
    assert not (tmp_path / "themap-wal").exists(), (
        "WAL frames outlive the file they describe and would be adopted by the rebuild"
    )


def test_the_run_says_the_map_was_corrupt_and_carries_on(tmp_path, monkeypatch, capsys):
    """Loudly, early, and to stderr, because there is no reporter yet.

    ``_ensure_map_seeded`` runs from ``pytest_cmdline_main`` as well as
    ``pytest_configure``, and the earlier of those is before pytest has a
    terminal writer. The sentence still has to arrive: a run that silently
    rebuilds looks identical to a run that inexplicably stopped selecting.
    """

    from tests import conftest as harness

    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    monkeypatch.setattr(harness, "_XDIST_WORKER", None)
    monkeypatch.setattr(testmon_policy, "_best_donor_map", lambda rootdir, filename: None)
    break_a_page(write_map(tmp_path / "themap", "poisoned", pages=8))

    config = SimpleNamespace(rootdir=tmp_path, getoption=lambda name, default=None: default)
    harness._ensure_map_seeded(config)

    said = capsys.readouterr().err
    assert "map is corrupt" in said
    assert "rebuilding" in said
    assert config.arrayscope_map_rebuilt
    assert not (tmp_path / "themap").exists(), "the rebuild starts from no map"


def test_a_corrupt_map_is_replaced_by_a_donor_when_there_is_one(tmp_path, monkeypatch):
    """Quarantine leaves the checkout looking fresh, so seeding is the rebuild.

    That ordering is the whole reason the health check lives where it does: a
    corrupt map in a worktree costs a copy from main rather than a four-minute
    traced suite.
    """

    from tests import conftest as harness

    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    monkeypatch.setattr(harness, "_XDIST_WORKER", None)
    donor_checkout, worktree = tmp_path / "main", tmp_path / "worktree"
    donor_checkout.mkdir()
    worktree.mkdir()
    donor = write_map(donor_checkout / "themap", "the-donors-records")
    monkeypatch.setattr(testmon_policy, "_best_donor_map", lambda rootdir, filename: donor)
    break_a_page(write_map(worktree / "themap", "poisoned", pages=8))

    config = SimpleNamespace(rootdir=worktree, getoption=lambda name, default=None: default)
    harness._ensure_map_seeded(config)

    assert config.arrayscope_map_rebuilt
    assert config.arrayscope_selection_seed == str(donor)
    assert read_map(worktree / "themap") == "the-donors-records"


def test_a_health_check_that_fails_may_not_fail_the_run(tmp_path, monkeypatch):
    """The backstop. This runs inside ``pytest_configure``; anything that
    escapes it takes the whole session, which is the outcome it exists to
    prevent — so an unforeseen failure here must cost the check, not the run.
    """

    from tests import conftest as harness

    monkeypatch.setattr(harness, "_XDIST_WORKER", None)

    def explode(rootdir):
        raise ValueError("a shape of database nobody planned for")

    monkeypatch.setattr(testmon_policy, "quarantine_broken_map", explode)
    monkeypatch.setattr(testmon_policy, "seed_map", lambda rootdir, seeding=True: None)

    config = SimpleNamespace(rootdir=tmp_path, getoption=lambda name, default=None: default)
    harness._ensure_map_seeded(config)
    assert config.arrayscope_selection_seed is None


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
    select_source = inspect.getsource(TestmonSelect)
    assert "deselected_tests" in select_source, (
        "tests/conftest.py reports the unaffected count from this attribute"
    )
    assert "deselected_files" in select_source, (
        "testmon_policy.apply_known_red_policy and apply_since_baseline rewrite this "
        "file-level list; losing it means a red this checkout broke stops being "
        "collected at all, and --since stops widening"
    )
    data_source = inspect.getsource(TestmonData)
    assert "sync_db_fs_tests" in data_source, (
        "testmon_policy.protect_map_outside_the_scope wraps this to stop a scoped "
        "run deleting the map entries it never looked at — the erosion defect"
    )
    assert "stable_files" in data_source, (
        "apply_known_red_policy reads it to decide which files may be skipped wholesale"
    )
    from testmon import pytest_testmon

    assert hasattr(pytest_testmon, "init_testmon_data"), (
        "testmon_policy.reuse_map_read replaces this module global so testmon adopts "
        "the map reading peek() already took instead of building a second one"
    )
    configure_source = inspect.getsource(pytest_testmon.pytest_configure)
    assert "init_testmon_data(config)" in configure_source, (
        "testmon no longer reaches its data through the module global, so replacing "
        "that name no longer intercepts it — the map would be read twice again"
    )
    # Through the marker, because this very run has already replaced the name:
    # ``arrayscope_reuse`` holds whatever it displaced.
    upstream = getattr(
        pytest_testmon.init_testmon_data, "arrayscope_reuse", pytest_testmon.init_testmon_data
    )
    assert "for_worker" in inspect.getsource(upstream), (
        "an xdist worker's data used to come from a different constructor with the "
        "controller's exec_id; reuse_map_read refuses on workers because of it"
    )


# --------------------------------------------------------------------------- #
# Reading the map once per run instead of twice
# --------------------------------------------------------------------------- #


class _FakeMapData:
    """Just enough of ``TestmonData`` to be identified."""

    def __init__(self, environment="offscreen"):
        self.environment = environment


def _reuse_config(tmp_path, environment="offscreen", **extra):
    return SimpleNamespace(
        rootdir=tmp_path,
        getini={
            "environment_expression": f"'{environment}'",
            "testmon_ignore_dependencies": [],
        }.get,
        getoption=lambda name, default=None: None,
        **extra,
    )


def test_a_reading_of_the_same_map_is_offered_to_testmon(tmp_path):
    """The whole point: testmon gets the object peek already stability-checked."""

    data = _FakeMapData()
    config = _reuse_config(tmp_path)
    config.arrayscope_map_read = (
        testmon_policy.map_read_identity(tmp_path, "offscreen", []),
        data,
    )
    assert testmon_policy.adopted_map_read(config) is data


def test_a_reading_of_a_different_environment_is_refused(tmp_path):
    """Environment is part of the map's identity, and testmon *replaces* an
    environment row whose inputs moved — so adopting across one would not merely
    mis-measure, it would discard the map."""

    config = _reuse_config(tmp_path, environment="offscreen")
    config.arrayscope_map_read = (
        testmon_policy.map_read_identity(tmp_path, "wayland+gpu", []),
        _FakeMapData("wayland+gpu"),
    )
    assert testmon_policy.adopted_map_read(config) is None


def test_a_reading_from_another_checkout_is_refused(tmp_path):
    config = _reuse_config(tmp_path)
    config.arrayscope_map_read = (
        testmon_policy.map_read_identity(tmp_path / "elsewhere", "offscreen", []),
        _FakeMapData(),
    )
    assert testmon_policy.adopted_map_read(config) is None


def test_an_xdist_worker_never_adopts_the_controllers_reading(tmp_path):
    """A worker's data is built by ``TestmonData.for_worker`` from the
    controller's ``exec_id``. Handing it a local-run reading would have it record
    against the wrong execution — the one failure here that is not merely slow."""

    config = _reuse_config(tmp_path, workerinput={"testmon_exec_id": 7})
    config.arrayscope_map_read = (
        testmon_policy.map_read_identity(tmp_path, "offscreen", []),
        _FakeMapData(),
    )
    assert testmon_policy.adopted_map_read(config) is None


def test_a_run_that_offered_nothing_gets_testmons_own_behaviour(tmp_path):
    """The interception is a module global and outlives any one run, so a config
    that made no offer must fall straight through rather than inherit one."""

    assert testmon_policy.adopted_map_read(_reuse_config(tmp_path)) is None


def test_the_system_package_walk_is_paid_once_per_process():
    """``importlib.metadata`` over the whole environment, 0.045 s, asked for by
    both readers. Installed distributions cannot change inside one process."""

    assert testmon_policy.memoise_system_packages()
    from testmon import common, pytest_testmon, testmon_core

    assert getattr(common.get_system_packages, "arrayscope_memoised", False)
    # Both importers bound the name at import time, so patching only the owner
    # would leave testmon's own reader on the original.
    assert pytest_testmon.get_system_packages is common.get_system_packages
    assert testmon_core.get_system_packages is common.get_system_packages
    assert common.get_system_packages(ignore=[]) == common.get_system_packages(ignore=[])


def test_the_option_keys_testmon_subscripts_all_still_exist():
    """The tripwire that would have documented the cacheprovider crash.

    ``testmon/configure.py`` reads ``vars(config.option)`` with **subscripts,
    not ``.get``**. Every key it indexes is therefore a hard dependency on some
    other plugin having registered that option — and ``lf`` belongs to pytest's
    cache plugin, which is why ``-p no:cacheprovider`` kills the run with a
    ``KeyError`` inside ``pytest_configure``, before a single test.

    Pinned in both directions. A key disappearing from testmon is a behaviour
    change we want to know about; a *new* subscript is a new plugin this suite
    silently cannot run without, and finding that out from an INTERNALERROR in
    somebody's terminal is how this was found the first time.
    """

    import re

    from testmon import configure

    subscripted = set(re.findall(r'options\[["\']([^"\']+)["\']\]', inspect.getsource(configure)))
    expected = {
        "no-testmon",
        "testmon_nocollect",
        "testmon_noselect",
        "testmon_forceselect",
        "keyword",
        "markexpr",
        "lf",  # _pytest.cacheprovider — see tests/conftest.py _HARMFUL_PLUGIN_DISABLES
    }
    assert subscripted == expected, (
        "testmon's hard option dependencies moved. Anything added here is a plugin "
        "that can no longer be disabled without crashing the run, and belongs in "
        "tests/conftest.py _HARMFUL_PLUGIN_DISABLES with a sentence explaining it"
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


def test_the_cached_blocks_are_testmons_own_method_checksums(tmp_path):
    """The equivalence the mhash cache rests on, asserted against testmon itself.

    The sidecar keeps a file's blocks for the coverage figure. They are the same
    blocks testmon fingerprints with, in the same order — both come from one
    ``dump_and_block`` walk — so the list testmon would re-derive by re-parsing
    is already on disk. If this ever stops holding, selection would be handed
    checksums that are not testmon's, which is the one way this cache could be
    wrong rather than merely slow.
    """

    from testmon.testmon_core import SourceTree

    coverage_map, relative = _sample_tree(tmp_path)
    structure = coverage_map.Structure()
    entry = structure.for_file(tmp_path, relative)

    module = SourceTree(rootdir=str(tmp_path)).get_file(relative)
    assert coverage_map.method_checksums_of(entry) == list(module.method_checksums)


def test_an_empty_file_has_no_blocks_rather_than_one_empty_one(tmp_path):
    """testmon reports ``[]`` for a file with nothing in it — every package
    ``__init__.py`` in this tree — and ``[None]`` would be a checksum list
    nothing can match."""

    coverage_map, relative = _sample_tree(tmp_path, source="", name="pkg/__init__.py")
    entry = coverage_map.Structure().for_file(tmp_path, relative)
    assert coverage_map.method_checksums_of(entry) == []


def test_a_cache_entry_from_another_revision_is_a_miss_not_an_answer(tmp_path):
    """The safety property, stated as a test: the key is the file's content, so
    a stale entry cannot be served. The worst this cache can do is a re-parse."""

    coverage_map, relative = _sample_tree(tmp_path)
    structure = coverage_map.Structure()
    structure.for_file(tmp_path, relative)
    payload = {"structure": structure.payload()}

    assert coverage_map.revision_at(payload, tmp_path, relative) is not None
    (tmp_path / relative).write_text(_SAMPLE + "\n\ndef added():\n    return 2\n")
    assert coverage_map.revision_at(payload, tmp_path, relative) is None


def test_a_sidecar_from_before_this_cache_existed_is_a_miss(tmp_path):
    coverage_map, relative = _sample_tree(tmp_path)
    assert coverage_map.method_checksums_of({"fsha": "x", "module": 1}) is None
    assert coverage_map.revision_at({}, tmp_path, relative) is None


def test_forgiveness_answers_the_same_from_the_cache_as_from_a_parse(tmp_path):
    """The cached decomposition carries the body digests and the element digests,
    which is everything the first forgiveness rule needs — so handing it over
    must not change a single forgiven checksum."""

    from tests import coverage_map as module

    relative = "pkg/module.py"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_SAMPLE, encoding="utf-8")

    before = module.Structure()
    previous = before.for_file(tmp_path, relative)
    path.write_text(_SAMPLE + "\n\ndef inert():\n    return 3\n", encoding="utf-8")
    after = module.Structure(before.payload())
    current = after.for_file(tmp_path, relative)

    revisions = [previous]
    parsed = testmon_policy.forgivable_checksums(tmp_path, relative, revisions, parsed=current)
    unparsed = testmon_policy.forgivable_checksums(tmp_path, relative, revisions)
    assert parsed == unparsed
    assert previous["module"] in parsed, "a purely inert addition is forgivable"


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
    donor = write_map(donor_checkout / "themap", "donor-map")
    coverage_map.write_sidecar(
        donor, structure={"a.py": {"fsha": "x"}}, baselines={"offscreen": {"covered": 7}}
    )
    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
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
    donor = write_map(donor_checkout / "themap", "donor-map")
    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    monkeypatch.setattr(testmon_policy, "_best_donor_map", lambda rootdir, filename: donor)

    assert testmon_policy.seed_map(worktree) == str(donor)
    assert read_map(worktree / "themap") == "donor-map"


# --- Reaching the inherited reds without leaving selection ------------------


def test_rerun_reds_is_a_flag_and_nothing_else():
    """One spelling. The environment variable that shadowed it is gone.

    It was kept on the claim that an xdist worker is "handed no command line of
    its own". Measured false on 2026-07-30: a worker reads ``rerun_reds=True``
    straight from the controller's command line, so the variable was a pure
    duplicate — and a second spelling of a selection bypass is exactly what
    slips past a refusal built around the first.
    """

    assert testmon_policy.rerun_known_red_tests() is False
    assert testmon_policy.rerun_known_red_tests(_option_config(rerun_reds=False)) is False
    assert testmon_policy.rerun_known_red_tests(_option_config(rerun_reds=True)) is True


def test_since_re_runs_inherited_reds_without_being_asked():
    """The merge gate owns the reds it would otherwise only mention.

    `--since` claims "everything this branch changed, ready to merge", and an
    inherited red it declines to run is a hole in that claim -- the moment the
    branch lands, someone else's red is main's red. It used to only print the
    skip: one grey line under a bold `900 passed`, which reads as noise, and
    did, for an entire session.

    The inner loop keeps the cheap behaviour: re-running every red on a clean
    tree measured 124 s, and an inner loop that costs two minutes to
    re-confirm what everybody already knows is one nobody uses.
    """

    assert testmon_policy.rerun_known_red_tests(_option_config(since="")) is True
    assert testmon_policy.rerun_known_red_tests(_option_config(since="main")) is True
    # The plain loop still skips them.
    assert testmon_policy.rerun_known_red_tests(_option_config(since=None)) is False
    assert testmon_policy.rerun_known_red_tests(_option_config()) is False


def test_a_config_without_the_option_reads_as_off():
    """`tools/test_selection.py` builds configs that never registered it."""

    class _NoOptions:
        def getoption(self, name, default=None):
            raise ValueError(f"no option named {name!r}")

    assert testmon_policy.rerun_known_red_tests(_NoOptions()) is False


def _option_config(**options):
    return SimpleNamespace(getoption=lambda name, default=None: options.get(name, default))


def test_a_dead_plugin_disable_is_detected_and_a_real_one_is_not():
    """`-p no:randomly` aborts the run; the rule cannot go stale.

    It is refused rather than noted afterwards because of who types it: nothing
    in this repository suggests the flag, so it arrives from another codebase's
    habit — and whoever brought it has not read rule 5, which makes a line
    printed *after* the run the wrong moment. It is also worse than inert: it
    reads as "the ordering is pinned", so a flake it never had a chance of
    preventing gets blamed elsewhere.
    """

    from tests.conftest import pointless_plugin_disables

    absent = lambda name: False  # noqa: E731 - one-expression stub reads better inline
    present = lambda name: True  # noqa: E731

    assert pointless_plugin_disables(["no:randomly"], absent) == [
        ("no:randomly", "pytest-randomly is not installed")
    ]
    # Installing the plugin retires the entry by itself — no list to prune.
    assert pointless_plugin_disables(["no:randomly"], present) == []
    # Untouched: plugins with no entry, and enabling rather than disabling.
    assert pointless_plugin_disables(["no:cacheprovider"], absent) == []
    assert pointless_plugin_disables(["randomly"], absent) == []
    assert pointless_plugin_disables([], absent) == []


def test_a_selection_bypass_is_refused_on_cost_not_on_spelling():
    """The gate measures time, because time is what it exists to protect.

    `--no-testmon` and `--testmon-noselect` are the same act — running tests
    the change provably cannot reach — so the rule is written about the cost
    rather than about either flag. A run that still selects is never refused
    however long it takes: the map already narrowed it to what the change
    touches, so that time is earned.
    """

    from tests.conftest import unselected_run_refusal

    # Cheap enough that retyping the command would cost more than the run.
    assert unselected_run_refusal(4.0, "") == ""
    # Expensive, and nothing legitimises it.
    assert "skips selection" in unselected_run_refusal(222.0, "")
    # Legitimate exhaustive contexts are never refused, at any price.
    assert unselected_run_refusal(222.0, "CI, where every job is a gate") == ""
    assert unselected_run_refusal(222.0, "explicitly forced") == ""


def test_a_bypass_is_priced_from_the_map_before_anything_runs():
    """Pricing has to precede collection — the only moment refusing saves time."""

    from tests.conftest import unselected_run_seconds

    recorded = {
        "tests/ui/test_a.py::test_one": 10.0,
        "tests/ui/test_a.py::test_two": 5.0,
        "tests/core/test_b.py::test_three": 1.0,
    }
    under = lambda name, scope: any(name.startswith(entry) for entry in scope)  # noqa: E731

    # No scope: the whole recorded suite.
    assert unselected_run_seconds(recorded, (), under) == 16.0
    # Scoped: only the tests whose file sits under it.
    assert unselected_run_seconds(recorded, ("tests/ui",), under) == 15.0
    assert unselected_run_seconds(recorded, ("tests/core",), under) == 1.0
    # An empty map cannot price anything, and must not invent a refusal.
    assert unselected_run_seconds({}, (), under) == 0.0


def _fake_worktrees(monkeypatch, entries):
    monkeypatch.setattr(testmon_policy, "_worktree_heads", lambda rootdir: entries)


def test_a_merge_adopts_the_map_of_the_worktree_that_did_the_work(tmp_path, monkeypatch):
    """The map has to follow the work, and the work happens in worktrees.

    Almost nothing is done in the main checkout, so its map predates every
    merge: it reports reds inherited from whatever seeded it and re-records
    what was traced minutes earlier elsewhere. Landing a branch used to cost a
    --rerun-reds and a --since there purely to quiet a map nobody updated.

    After a fast-forward both HEADs are the same commit, which is stronger than
    the "close enough donor" seeding settles for -- identical commits mean
    identical fingerprints, so the adopted map needs no invalidation pass.
    """

    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    root, branch = tmp_path / "main", tmp_path / "wt"
    root.mkdir()
    branch.mkdir()
    write_map(root / "themap", "stale")
    write_map(branch / "themap", "recorded-the-merge")
    os.utime(root / "themap", (1, 1))
    os.utime(branch / "themap", (2, 2))

    _fake_worktrees(monkeypatch, [(root, "abc123"), (branch, "abc123")])

    assert testmon_policy.adopt_map_from_identical_worktree(root) == str(branch / "themap")
    assert read_map(root / "themap") == "recorded-the-merge"


def test_adoption_refuses_a_different_commit_or_an_older_map(tmp_path, monkeypatch):
    """Both halves of the safety argument, since only equality makes it free."""

    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    root, branch = tmp_path / "main", tmp_path / "wt"
    root.mkdir()
    branch.mkdir()
    write_map(root / "themap", "ours")
    write_map(branch / "themap", "theirs")
    os.utime(root / "themap", (1, 1))
    os.utime(branch / "themap", (2, 2))

    # A worktree on another commit is not this tree, so its map is not free.
    _fake_worktrees(monkeypatch, [(root, "abc123"), (branch, "def456")])
    assert testmon_policy.adopt_map_from_identical_worktree(root) is None
    assert read_map(root / "themap") == "ours"

    # Same commit, but ours ran later: adopting would throw that run away.
    os.utime(root / "themap", (3, 3))
    _fake_worktrees(monkeypatch, [(root, "abc123"), (branch, "abc123")])
    assert testmon_policy.adopt_map_from_identical_worktree(root) is None
    assert read_map(root / "themap") == "ours"


def test_a_bare_since_says_what_it_decided_to_compare_against(monkeypatch):
    """Which of the three baselines it found changes the answer completely.

    A bare `--since` tries ARRAYSCOPE_BASELINE_REF, then the upstream, then
    `main`. Standing on `main` here it resolves to `origin/main`, which nothing
    has been pushed to for 261 commits, so the branch-sized question quietly
    becomes a 2000-test one. The ref name alone does not show that -- the
    distance does, so it is counted and printed before the run.
    """

    monkeypatch.setattr(
        testmon_policy,
        "resolve_baseline",
        lambda rootdir, explicit: ("@{upstream}", "41a21f64" * 5),
    )
    counts = {"41a21f64" * 5 + "..HEAD": "261", "HEAD..@{upstream}": "0"}
    monkeypatch.setattr(
        testmon_policy,
        "_git",
        lambda rootdir, *args: counts.get(args[-1], ""),
    )

    line = testmon_policy.describe_baseline("/repo", None)

    assert "--since @{upstream} (resolved)" in line
    assert "branch point 41a21f64" in line
    assert "261 commit(s) of yours since it" in line


def test_an_explicitly_named_baseline_is_reported_as_named(monkeypatch):
    monkeypatch.setattr(
        testmon_policy, "resolve_baseline", lambda rootdir, explicit: (explicit, "abcdef12" * 5)
    )
    monkeypatch.setattr(testmon_policy, "_git", lambda rootdir, *args: "3")

    assert "--since origin/main (named)" in testmon_policy.describe_baseline("/repo", "origin/main")


# --- Per-test dependency batches ---------------------------------------------
# testmon reads a test's dependency edges out of coverage dynamic contexts,
# batching up to 250 tests into one live Coverage object. The sysmon core
# (coverage's default on CPython 3.14+) reports each line location once per
# Coverage lifetime, so inside a batch only the FIRST test to execute a line
# is credited with it — every later test records a subset of what it ran, and
# a missing edge means the test is silently never selected again for changes
# to that file. tests/conftest.py gives every test its own batch at import
# time; these tests hold both the pin and the behaviour it exists to buy.
# (Pinning COVERAGE_CORE=ctrace instead would also attribute correctly, but
# measured 388 s against sysmon's 267 s for the full suite and flipped 13
# timing-sensitive tests/ui settle tests red — see
# testmon_policy.pin_per_test_dependency_batches.)


def test_every_test_gets_its_own_dependency_batch():
    """The live invariant: by the time any test executes, the pin has run."""

    from testmon import testmon_core

    assert testmon_core.TEST_BATCH_SIZE == 1


def test_the_pin_survives_without_testmon(monkeypatch):
    """Absence is decide()'s problem, not an import error at conftest time."""

    import builtins

    real_import = builtins.__import__

    def refuse_testmon(name, *args, **kwargs):
        if name.startswith("testmon"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_testmon)
    assert testmon_policy.pin_per_test_dependency_batches() is False


def test_a_later_test_still_records_the_dependency_an_earlier_one_executed(tmp_path):
    """The oracle behind the pin, through a real testmon recording.

    Two tests in one process execute the same helper; the map must credit
    *both* with the helper file. With testmon's stock 250-test batches under
    the sysmon core the second loses the edge — exactly the failure that
    slimmed ``test_session_fixture_shape_mismatch_raises_actionable_error``'s
    record from 69 dependency files to 8 and let the 2026-07-31
    ``view_recipe`` regression through every ``--since`` run. In a subprocess
    because a second live testmon session inside this pytest process would
    fight the one recording this very test.
    """

    import sqlite3
    import subprocess
    import sys
    import textwrap

    (tmp_path / "shared_mod.py").write_text("def shared():\n    return 42\n", encoding="utf-8")
    (tmp_path / "test_two_share.py").write_text(
        textwrap.dedent(
            """
            import shared_mod

            def test_first():
                assert shared_mod.shared() == 42

            def test_second():
                assert shared_mod.shared() == 42
            """
        ),
        encoding="utf-8",
    )
    # The pin, applied the way production applies it: from a conftest, at
    # import time, before testmon builds its first Coverage.
    (tmp_path / "conftest.py").write_text(
        textwrap.dedent(
            f"""
            import sys

            sys.path.insert(0, {str(testmon_policy.REPO_ROOT)!r})
            from tests import testmon_policy

            testmon_policy.pin_per_test_dependency_batches()
            """
        ),
        encoding="utf-8",
    )
    # A clean pytest environment: the outer run's PYTEST_* variables describe
    # *this* session (xdist worker identity, addopts) and must not leak in.
    child_environment = {
        key: value for key, value in os.environ.items() if not key.startswith("PYTEST_")
    }
    child_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        (sys.executable, "-m", "pytest", "--testmon", "-q"),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=child_environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    with sqlite3.connect(tmp_path / ".testmondata") as connection:
        rows = connection.execute(
            """
            SELECT te.test_name, f.filename
            FROM test_execution te
            JOIN test_execution_file_fp tef ON tef.test_execution_id = te.id
            JOIN file_fp f ON f.id = tef.fingerprint_id
            """
        ).fetchall()
    edges = {}
    for test_name, filename in rows:
        edges.setdefault(test_name.split("::")[-1], set()).add(filename)
    assert "shared_mod.py" in edges["test_first"]
    assert "shared_mod.py" in edges["test_second"], (
        "the second test lost the dependency the first already executed — "
        "per-test batching is no longer protecting the map"
    )


# --------------------------------------------------------------------------- #
# A record made by a test that skipped
# --------------------------------------------------------------------------- #
#
# The map keeps `failed` and `duration` and nothing else, so a test that skipped
# lands in it as a fast green whose dependency set is whatever the import and
# the gate touched. Selection then reads that set as proof the test does not
# execute your change, which is a claim the record cannot support -- it means
# only that the test did not run. See `tests/skip_ledger.py` for the branch this
# cost.


def _gated_project(tmp_path):
    """A one-test project whose test is gated on an environment variable.

    The production shape exactly: a module-level flag read from the environment,
    a ``skipif`` on it, and a body that calls into a second file. When the gate
    is shut the test never enters ``gated_mod``, and the map has no way to say
    so.
    """

    import textwrap

    (tmp_path / "gated_mod.py").write_text("def work():\n    return 'original'\n", encoding="utf-8")
    (tmp_path / "test_gated.py").write_text(
        textwrap.dedent(
            """
            import os

            import pytest

            import gated_mod

            ENABLED = os.environ.get("GATE_OPEN") == "1"


            @pytest.mark.skipif(not ENABLED, reason="the gate is shut")
            def test_gated():
                assert gated_mod.work() == "original"
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "conftest.py").write_text(
        textwrap.dedent(
            f"""
            import sys

            sys.path.insert(0, {str(REPO_ROOT)!r})
            from tests import skip_ledger, testmon_policy

            testmon_policy.pin_per_test_dependency_batches()

            _ledger = None


            def pytest_sessionstart(session):
                global _ledger
                _ledger = skip_ledger.SkipLedger.load(session.config.rootdir / ".testmondata", "e")


            def pytest_runtest_logreport(report):
                _ledger.observe(report)


            def pytest_sessionfinish(session):
                _ledger.save()
            """
        ),
        encoding="utf-8",
    )


def _run_child_pytest(tmp_path, *arguments, **environment):
    import subprocess
    import sys

    child = {key: value for key, value in os.environ.items() if not key.startswith("PYTEST_")}
    child["PYTHONDONTWRITEBYTECODE"] = "1"
    child.update(environment)
    result = subprocess.run(
        (sys.executable, "-m", "pytest", "--testmon", "-q", *arguments),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=child,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _recorded_dependencies(tmp_path) -> dict[str, set[str]]:
    import sqlite3

    with sqlite3.connect(tmp_path / ".testmondata") as connection:
        rows = connection.execute(
            """
            SELECT te.test_name, f.filename
            FROM test_execution te
            JOIN test_execution_file_fp tef ON tef.test_execution_id = te.id
            JOIN file_fp f ON f.id = tef.fingerprint_id
            """
        ).fetchall()
    edges: dict[str, set[str]] = {}
    for test_name, filename in rows:
        edges.setdefault(test_name, set()).add(filename)
    return edges


def test_a_skipped_test_records_a_dependency_set_that_omits_the_code_it_would_run(tmp_path):
    """The defect itself, through a real testmon recording. Not our bug to fix.

    Pinned because everything else here is a compensation for it, and a testmon
    release that started distinguishing the two outcomes would make the
    compensation dead weight rather than wrong. If this ever goes green the
    other way round, the ledger can go.
    """

    from tests import skip_ledger

    _gated_project(tmp_path)
    _run_child_pytest(tmp_path, GATE_OPEN="0")

    recorded = _recorded_dependencies(tmp_path)
    name = next(key for key in recorded if key.endswith("test_gated"))
    assert "test_gated.py" in recorded[name], "the skip still recorded the module it imported"
    assert "gated_mod.py" not in recorded[name], (
        "testmon now records what a skipped test *would* have run — the skip "
        "ledger is no longer needed"
    )
    assert skip_ledger.read_skips(tmp_path / ".testmondata", "e") == {name}


def test_a_test_recorded_as_skipped_is_not_evidence_that_a_change_misses_it(tmp_path):
    """The regression this whole file section exists for.

    A branch rewrites ``gated_mod.py``. The map says the one test that would
    execute it does not depend on it — truthfully, because when it was recorded
    it skipped. Before the fix that answer flowed straight through
    :func:`partition_reached` into ``verified``, and ``--since`` reported the
    branch fully proven with its only real coverage never collected. That is the
    2026-07-31 ``screen_canvas.py`` miss reproduced in twenty lines.
    """

    from tests import skip_ledger

    _gated_project(tmp_path)
    _run_child_pytest(tmp_path, GATE_OPEN="0")
    (tmp_path / "gated_mod.py").write_text(
        "def work():\n    return 'rewritten'\n", encoding="utf-8"
    )

    name = next(key for key in _recorded_dependencies(tmp_path) if key.endswith("test_gated"))
    skipped = skip_ledger.read_skips(tmp_path / ".testmondata", "e")
    # The map's own answer, unchanged and still wrong: the test is stable
    # against a rewritten `gated_mod.py`, because its record never saw it.
    partition = testmon_policy.partition_reached(
        {name},
        recorded={name},
        stable={name},
        failed=set(),
        broken_here=set(),
        flaky=set(),
        skipped=skipped,
    )
    assert partition[testmon_policy.VERIFIED] == frozenset(), (
        "a test that skipped when the map recorded it was counted as verified "
        "green against this tree"
    )
    assert partition["skipped when recorded"] == frozenset({name})


def test_the_gate_reaches_a_skip_recorded_test_the_branch_diff_cannot_find():
    """Widening ``reached``, not just relabelling it.

    ``--since`` derives the reached set from recorded dependencies, so a
    skip-recorded test is absent from it however completely the branch rewrites
    what it executes. Partitioning alone would therefore fix nothing: there is
    nothing in the set to partition. Both callers go through
    :func:`since_verification`, which is where the widening lives.
    """

    name = "tests/gpu_interaction/test_wgpu_screen_present.py::test_auto_present_method"
    config = SimpleNamespace(
        testmon_data=SimpleNamespace(
            environment="offscreen",
            all_tests={name: {"failed": False, "duration": 0.001}},
            stable_test_names={name},
        ),
        rootdir=REPO_ROOT,
    )
    reached_without_it = frozenset()

    def fake_skips(map_path, environment):
        assert environment == "offscreen"
        return {name}

    from tests import skip_ledger

    original = skip_ledger.read_skips
    skip_ledger.read_skips = fake_skips
    try:
        report = testmon_policy.since_verification("main", reached_without_it, config)
    finally:
        skip_ledger.read_skips = original

    assert report.reached == 1, "the gate never reached the one test that covers the change"
    assert report.verified == 0
    assert report.run_tests == frozenset({name})
    assert ("skipped when recorded", 1) in report.running


def test_a_mark_clears_the_moment_the_test_actually_runs(tmp_path):
    """The only event that retires a mark, and the only one that should.

    A skip mark says "this record describes a gate". Running the test for real
    is precisely when that stops being true: the record becomes a statement
    about code, and the test goes back to being ordinary evidence. Nothing
    expires, nothing is tuned, and no mark can outlive the condition that
    justified it — which is what keeps the remainder from becoming a permanent
    line everybody reads past.

    The first half of this test pins the *limitation* that follows from it, and
    it is deliberately not a bug: opening the gate is invisible to selection,
    because the gate's inputs are the environment rather than the code, so the
    ordinary inner loop deselects the marked test and the mark survives a run
    that would have cleared it. Widening the inner loop to re-attempt every mark
    was the alternative and was rejected — it would drag the whole opt-in ring
    into every offscreen run and cost the "nothing to run" fast path. So the
    escape hatch is named instead, on the remainder line, on every run that has
    one: ``pytest --since`` reaches the marks, and a node-id run (which is how
    the branch that found this got its tests to run at all) selects past the map
    entirely. Either clears the mark.
    """

    from tests import skip_ledger

    _gated_project(tmp_path)
    _run_child_pytest(tmp_path, GATE_OPEN="0")
    marked = skip_ledger.read_skips(tmp_path / ".testmondata", "e")
    assert marked, "nothing was marked to clear"

    # The gate opens, but nothing the map can see has changed, so the inner loop
    # deselects the test and learns nothing new about it.
    _run_child_pytest(tmp_path, GATE_OPEN="1")
    assert skip_ledger.read_skips(tmp_path / ".testmondata", "e") == marked

    # A run that actually selects it does clear it, and re-records the real
    # dependency the skip could never have shown.
    _run_child_pytest(tmp_path, "test_gated.py::test_gated", GATE_OPEN="1")

    name = next(key for key in _recorded_dependencies(tmp_path) if key.endswith("test_gated"))
    assert "gated_mod.py" in _recorded_dependencies(tmp_path)[name], (
        "the run that opened the gate did not re-record the real dependency"
    )
    assert skip_ledger.read_skips(tmp_path / ".testmondata", "e") == set()


def test_clearing_a_mark_leaves_a_durable_record_that_it_was_ever_set(tmp_path):
    """The self-healing above is right, and it used to destroy the evidence.

    Within one environment a gate should answer the same way every time, so a
    test that skips on one run and runs on the next is describing something the
    environment does not explain. That is worth knowing and nothing recorded it:
    the mark cleared on the very run that proved it had been intermittent, which
    is how ``make_backend_window`` skipped three different tests across eleven
    passes and left nothing behind to find.
    """

    from tests import skip_ledger

    _gated_project(tmp_path)
    _run_child_pytest(tmp_path, GATE_OPEN="0")
    _run_child_pytest(tmp_path, "test_gated.py::test_gated", GATE_OPEN="1")

    intermittent = skip_ledger.read_intermittent(tmp_path / ".testmondata", "e")
    name = next(key for key in _recorded_dependencies(tmp_path) if key.endswith("test_gated"))
    assert name in intermittent, "the transition that cleared the mark was not written down"
    assert intermittent[name]["count"] == 1


def test_an_intermittent_skip_record_widens_no_selection(tmp_path):
    """It is a thing a person reads, never a thing the gate reads.

    A test that ran recorded real dependencies, and those are evidence whatever
    its previous run did -- so this record must not put a single node back into
    the ``--since`` set. The mark above is what selection consults; this is
    strictly commentary, and the two live in different keys so that staying true
    does not depend on anyone remembering the distinction.
    """

    from tests import skip_ledger

    _gated_project(tmp_path)
    _run_child_pytest(tmp_path, GATE_OPEN="0")
    _run_child_pytest(tmp_path, "test_gated.py::test_gated", GATE_OPEN="1")

    assert skip_ledger.read_intermittent(tmp_path / ".testmondata", "e")
    assert skip_ledger.read_skips(tmp_path / ".testmondata", "e") == set(), (
        "the durable record leaked into the set that widens the merge gate"
    )


def test_a_test_that_simply_runs_is_never_called_intermittent(tmp_path):
    """No transition, no record. The obvious half, and the expensive one to get
    wrong: a mark on every green would bury the handful that mean something."""

    from tests import skip_ledger

    _gated_project(tmp_path)
    _run_child_pytest(tmp_path, GATE_OPEN="1")
    _run_child_pytest(tmp_path, "test_gated.py::test_gated", GATE_OPEN="1")

    assert skip_ledger.read_intermittent(tmp_path / ".testmondata", "e") == {}


def test_an_xfail_is_not_a_skip_however_it_is_reported(tmp_path):
    """xfail arrives as ``skipped`` with ``wasxfail``, and its body ran.

    Worth its own test because the distinction is invisible in the outcome
    string and getting it wrong is silent in the expensive direction: every
    xfail in the suite would carry a permanent mark, widening the merge gate
    forever and burying the remainder line under tests that were never the
    problem.
    """

    import textwrap

    from tests import skip_ledger

    (tmp_path / "test_x.py").write_text(
        textwrap.dedent(
            """
            import pytest

            @pytest.mark.xfail(reason="known")
            def test_expected_to_fail():
                assert False

            @pytest.mark.skip(reason="gated")
            def test_really_skipped():
                pass
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "gated_mod.py").write_text("def work():\n    return 1\n", encoding="utf-8")
    _gated_project(tmp_path)  # for the conftest that wires the ledger
    (tmp_path / "test_gated.py").unlink()
    _run_child_pytest(tmp_path)

    marks = skip_ledger.read_skips(tmp_path / ".testmondata", "e")
    assert not any("test_expected_to_fail" in name for name in marks), (
        "an xfail was marked unmeasured — its body ran and its record is evidence"
    )
    assert any("test_really_skipped" in name for name in marks)


def test_a_function_only_another_environment_runs_is_not_reported_untested(tmp_path):
    """The same hole, read from the coverage side.

    ``new_code`` compares against one environment's map, so a function only ring
    4 enters reads as "executed by no test" under ``offscreen`` however
    thoroughly ``xcb+gpu`` covers it — which is how ``screen_canvas.py``'s
    ``ensure_context`` and ``_on_surface_resized`` were reported as new
    functions with no test while ten recorded ring-4 tests were executing both.

    The split is what the reader acts on, so it is the split that is pinned
    here; the cross-environment lookup itself is a query and is exercised
    against a real multi-environment map below.
    """

    from tests.coverage_map import Function, NewCode

    def function(name, checksum):
        return Function(path="a.py", name=name, line=1, checksum=checksum, body=name)

    ring_four_only = function("ensure_context", 11)
    nobody = function("dead_helper", 22)
    fresh = NewCode(
        ref="main",
        covered=(),
        uncovered=(ring_four_only, nobody),
        covered_elsewhere=((ring_four_only, ("xcb+gpu", "wayland+gpu")),),
    )

    assert fresh.total == 2
    assert fresh.genuinely_uncovered == (nobody,), (
        "a function another environment executes was reported as untested"
    )


def test_the_map_reports_every_environment_it_holds():
    """The lookup above is only as good as the list of environments to ask.

    Against this checkout's own map, because a hand-built fixture would pin the
    query's shape rather than that it answers — and the environment table is
    testmon's, so a schema move must surface here rather than as a silently
    empty answer that reads exactly like "no other environment covers it".
    """

    from tests import coverage_map

    map_path = testmon_policy.map_path(REPO_ROOT)
    if not map_path.exists():
        pytest.skip("no recorded map in this checkout")
    names = coverage_map._environment_names(map_path)
    assert names, "the map holds no environment rows at all — has the schema moved?"
    assert all(isinstance(name, str) for name in names)
