"""Guards for module-block forgiveness (``tests/testmon_policy.py``).

Ring 0 — no Qt, no rendering, no map on disk. Every test here drives testmon's
*own* comparison, ``db.check_fingerprint_db``, against a fingerprint built by
testmon's *own* ``create_fingerprint``, so each one carries its own control: the
``selects_today`` arm is what selection does without this change and the
``still_selects`` arm is what it does with it. A test whose control arm already
answered "unaffected" would be proving nothing, so every one of them asserts
that it does not.

The defect being closed: a module's whole top-level body is a single AST block
spanning line 1 to EOF, so it sits in 100% of a file's fingerprints, and
``check_fingerprint_db`` is a subset test against the file's current checksum
list. Appending one inert function to ``arrayscope/window/frame_session.py``
therefore selected 258 of 3933 tests and cost a measured 123.16 s of wall clock,
for zero tests whose executed code had changed.

The direction of the risk is fixed by construction: forgiveness only ever
*appends* to the right-hand side of a subset test, so a mistake can turn
"affected" into "unaffected" and never the reverse. Which is why most of what
follows is refusals.
"""

from __future__ import annotations

import ast

import pytest

from tests import coverage_map, testmon_policy

RELATIVE = "pkg/sample.py"

#: One import, one constant, one function, one class with one method — the four
#: top-level statement kinds the gate reasons about, in one file small enough to
#: read.
BEFORE = """\
import os

TILE = 8


def helper(value):
    return value + TILE


class Thing:
    def method(self):
        return helper(os.getpid())
"""

#: The line inside ``helper`` a recorded test is pretended to have executed.
COVERED_LINE = BEFORE.splitlines().index("    return value + TILE") + 1


def _record(source: str, lines=(COVERED_LINE,)):
    """What testmon records for a test that executed ``lines`` of ``source``."""

    from testmon.process_code import Module, create_fingerprint

    return list(create_fingerprint(Module(source_code=source, filename=RELATIVE), list(lines)))


def _whole_file(source: str):
    """The fingerprint ``inject_out_of_process_dependencies`` produces: every block."""

    return _record(source, range(1, len(source.splitlines()) + 1))


def _checksums(source: str):
    from testmon.process_code import Module

    return list(Module(source_code=source, filename=RELATIVE).method_checksums)


def _revision(tmp_path, source: str, relative: str = RELATIVE) -> dict:
    """The sidecar entry a run would have retained for ``source``."""

    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return coverage_map.Structure().for_file(tmp_path, relative)


def _arms(tmp_path, before: str, after: str, *, relative: str = RELATIVE, fingerprint=None):
    """``(selects_today, selects_with_forgiveness)`` for one edit.

    Both arms run testmon's real subset test against a real fingerprint. The
    only difference between them is the checksums this module appends.
    """

    from testmon.db import check_fingerprint_db

    recorded = _record(before) if fingerprint is None else fingerprint
    revisions = [_revision(tmp_path, before, relative)]
    (tmp_path / relative).write_text(after, encoding="utf-8")

    current = _checksums(after)
    extra = testmon_policy.forgivable_checksums(tmp_path, relative, revisions)
    selects_today = not check_fingerprint_db({relative: current}, relative, recorded)
    still_selects = not check_fingerprint_db({relative: current + extra}, relative, recorded)
    return selects_today, still_selects


def _refuses(tmp_path, after: str, before: str = BEFORE):
    selects_today, still_selects = _arms(tmp_path, before, after)
    assert selects_today, "the control arm must show the edit selecting today"
    return still_selects


# --------------------------------------------------------------------------- #
# What is forgiven
# --------------------------------------------------------------------------- #


def test_appending_a_function_stops_retiring_every_fingerprint(tmp_path):
    """The headline case: 258 tests and 123.16 s of wall clock for a no-op."""

    selects_today, still_selects = _arms(
        tmp_path, BEFORE, BEFORE + "\n\ndef added(value):\n    return value\n"
    )
    assert selects_today, "today one inert append invalidates the fingerprint"
    assert not still_selects, "the edit must be forgiven"


def test_inserting_a_function_needs_the_index_free_digest_as_well(tmp_path):
    """Module-block forgiveness alone does not open this one.

    ``Block.code`` is ``f"{index}:{ast_repr}"``, so a definition inserted near
    the top renumbers every block below it: the recorded ``helper`` block is
    retired too, and only the body-digest rule brings it back.
    """

    after = "def inserted():\n    return 0\n\n\n" + BEFORE
    recorded = _record(BEFORE)
    assert set(recorded) - set(_checksums(after)) != {_checksums(BEFORE)[-1]}, (
        "more than the module block must have been retired, or this proves nothing"
    )

    selects_today, still_selects = _arms(tmp_path, BEFORE, after)
    assert selects_today
    assert not still_selects, "the edit must be forgiven"


def test_adding_a_method_to_an_existing_class_is_forgiven(tmp_path):
    """Measured at 157 tests for ``ViewState``. ``ClassDef`` is not in testmon's
    transform tuple, so a class body is dumped inline into the module block —
    which is why the decomposition has to recurse into it."""

    after = BEFORE + "\n    def extra(self):\n        return 2\n"
    selects_today, still_selects = _arms(tmp_path, BEFORE, after)
    assert selects_today
    assert not still_selects, "the edit must be forgiven"


def test_adding_a_test_function_to_a_test_file_is_forgiven(tmp_path):
    """The largest blast radius measured: one new test in
    ``tests/display/test_odd_origin_crop_geometry.py`` selects all 297 tests in
    it. Undecorated, so no ``@pytest.mark`` — that exemption is deliberately not
    taken."""

    before = "def test_one():\n    assert True\n"
    after = before + "\n\ndef test_two():\n    assert True\n"
    selects_today, still_selects = _arms(
        tmp_path, before, after, relative="tests/test_thing.py", fingerprint=_record(before, [2])
    )
    assert selects_today
    assert not still_selects, "the edit must be forgiven"


def test_an_added_definition_may_carry_inert_default_arguments(tmp_path):
    """Default arguments evaluate at definition time, so they are gated — but to
    a whitelist wide enough to be useful: constants, names, attributes,
    containers of those, and a negated constant."""

    after = (
        BEFORE + "\n\ndef added(axis=-1, mode=None, sep=os.sep, shape=(1, 2)):\n    return axis\n"
    )
    selects_today, still_selects = _arms(tmp_path, BEFORE, after)
    assert selects_today
    assert not still_selects, "the edit must be forgiven"


def test_an_added_class_with_an_inert_body_is_forgiven(tmp_path):
    after = (
        BEFORE
        + '\n\nclass Added:\n    """Doc."""\n\n    LIMIT = 4\n\n    def run(self):\n        return 1\n'
    )
    selects_today, still_selects = _arms(tmp_path, BEFORE, after)
    assert selects_today
    assert not still_selects, "the edit must be forgiven"


# --------------------------------------------------------------------------- #
# What is never forgiven
# --------------------------------------------------------------------------- #


def test_a_changed_constant_is_never_forgiven(tmp_path):
    """The case the whole design has to get right by construction: changing
    ``TILE = 8`` to ``= 16`` *modifies* an element, so the decomposition is not
    a pure insertion and nothing is forgiven."""

    assert _refuses(tmp_path, BEFORE.replace("TILE = 8", "TILE = 16"))


def test_an_added_import_is_never_forgiven(tmp_path):
    """The module's import-time behaviour changed. No recorded body did, but the
    honest answer is still "re-run" — an import can execute anything."""

    assert _refuses(tmp_path, BEFORE + "\nimport sys\n")


def test_an_added_constant_is_never_forgiven(tmp_path):
    assert _refuses(tmp_path, BEFORE + "\nEXTRA = 3\n")


def test_a_changed_function_body_is_never_forgiven(tmp_path):
    """testmon is already exact here — 32 of 166 on ``render/lod.py`` — and this
    must not disturb it. The module block does not even move; the function's own
    block does, and it is not forgiven."""

    assert _refuses(tmp_path, BEFORE.replace("return value + TILE", "return value - TILE"))


def test_a_decorated_definition_is_never_forgiven(tmp_path):
    """Decorators run at import time and mutate shared registries — this tree
    has them, in ``arrayscope/operations/library.py``. There is no exemption for
    ``@pytest.*`` either: it was measured at 11 file-touches and 19 recorded
    seconds, which does not buy a second class of reasoning."""

    assert _refuses(tmp_path, BEFORE + "\n\n@os.wrap\ndef added():\n    return 1\n")
    assert _refuses(tmp_path, BEFORE + "\n\n@pytest.fixture\ndef added():\n    return 1\n")


def test_a_default_argument_that_calls_something_is_never_forgiven(tmp_path):
    """It evaluates once, at import, and can mutate anything."""

    assert _refuses(tmp_path, BEFORE + "\n\ndef added(seed=os.getpid()):\n    return seed\n")
    assert _refuses(
        tmp_path, BEFORE + "\n\ndef added(seen=[x for x in (1, 2)]):\n    return seen\n"
    )
    assert _refuses(tmp_path, BEFORE + "\n\ndef added(hook=lambda: 1):\n    return hook\n")
    assert _refuses(
        tmp_path, BEFORE + "\n\ndef added(limit={'a': os.getpid()}):\n    return limit\n"
    )


def test_a_class_with_a_base_or_a_metaclass_is_never_forgiven(tmp_path):
    """Defining a subclass calls ``__init_subclass__`` on the base, and a
    ``metaclass=`` keyword calls the metaclass. Both are calls into code that
    was already there, however inert the body looks."""

    assert _refuses(tmp_path, BEFORE + "\n\nclass Added(Thing):\n    pass\n")
    assert _refuses(tmp_path, BEFORE + "\n\nclass Added(metaclass=type):\n    pass\n")
    assert _refuses(tmp_path, BEFORE + "\n\n@os.wrap\nclass Added:\n    pass\n")


def test_a_class_body_that_is_not_inert_is_never_forgiven(tmp_path):
    assert _refuses(tmp_path, BEFORE + "\n\nclass Added:\n    LIMIT = os.getpid()\n")
    assert _refuses(tmp_path, BEFORE + "\n\nclass Added:\n    import sys\n")
    assert _refuses(tmp_path, BEFORE + "\n\nclass Added:\n    class Inner:\n        pass\n")


def test_rebinding_a_name_the_previous_revision_bound_is_never_forgiven(tmp_path):
    """``def helper(): a`` … ``def helper(): b`` appended reads as a pure
    addition — the old element survives — while ``helper`` now resolves to
    something else. The name check closes it, for every kind of binding the
    previous revision made, not just ``def``."""

    assert _refuses(tmp_path, BEFORE + "\n\ndef helper(value):\n    return 0\n")
    assert _refuses(tmp_path, BEFORE + "\n\ndef os():\n    return 0\n")
    assert _refuses(tmp_path, BEFORE + "\n\ndef TILE():\n    return 0\n")
    assert _refuses(tmp_path, BEFORE + "\n\nclass Thing:\n    pass\n")


def test_an_added_dunder_is_never_forgiven(tmp_path):
    """What a dunder overrides has no block to compare against: ``__repr__``
    replaces ``object.__repr__``, and a module-level ``__getattr__`` (PEP 562)
    changes every attribute lookup on the module. Both read as pure additions
    and are behaviour changes in fact."""

    assert _refuses(tmp_path, BEFORE + "\n\ndef __getattr__(name):\n    return None\n")
    assert _refuses(tmp_path, BEFORE + "\n    def __repr__(self):\n        return 'Thing()'\n")


def test_a_method_added_to_a_decorated_class_is_forgiven_only_for_dataclass(tmp_path):
    """A class decorator receives the finished class and can read its body, so
    adding a method re-runs it over different input. Census of this tree: 415
    decorated classes, 411 ``@dataclass`` and 4 ``@runtime_checkable`` — and
    adding a method to a runtime-checkable Protocol changes what ``isinstance``
    answers for code that was already there."""

    frozen = (
        "from dataclasses import dataclass\n\n\n@dataclass(frozen=True)\n"
        "class Point:\n    x: int\n\n    def moved(self):\n        return self.x\n"
    )
    selects_today, still_selects = _arms(
        tmp_path,
        frozen,
        frozen + "\n    def doubled(self):\n        return self.x * 2\n",
        fingerprint=_record(frozen, [8]),
    )
    assert selects_today
    assert not still_selects, "the edit must be forgiven"

    protocol = (
        "from typing import Protocol, runtime_checkable\n\n\n@runtime_checkable\n"
        "class Sink(Protocol):\n    def write(self):\n        ...\n"
    )
    selects_today, still_selects = _arms(
        tmp_path,
        protocol,
        protocol + "\n    def flush(self):\n        ...\n",
        fingerprint=_record(protocol, [6]),
    )
    assert selects_today
    assert still_selects

    # The name is recognised syntactically, so it is honoured only where the
    # module actually imports ``dataclasses``.
    shadowed = frozen.replace(
        "from dataclasses import dataclass", "from elsewhere import dataclass"
    )
    selects_today, still_selects = _arms(
        tmp_path,
        shadowed,
        shadowed + "\n    def doubled(self):\n        return self.x * 2\n",
        fingerprint=_record(shadowed, [8]),
    )
    assert selects_today
    assert still_selects


def test_reordering_two_statements_is_never_forgiven(tmp_path):
    """Matching is a subsequence test, not a multiset one. Swapping two
    top-level statements leaves the multiset untouched and changes what the
    module computes."""

    before = "A = 1\nB = A\n\n\ndef read():\n    return B\n"
    after = "B = A\nA = 1\n\n\ndef read():\n    return B\n"
    selects_today, still_selects = _arms(tmp_path, before, after, fingerprint=_record(before, [6]))
    assert selects_today
    assert still_selects


def test_moving_a_function_within_a_file_is_never_forgiven(tmp_path):
    """The honest reading of a move: one removal and one insertion, because the
    statements around it now run in a different order. The function's own block
    is still forgiven by its body digest; the module block is not."""

    after = "class Thing:\n    def method(self):\n        return helper(os.getpid())\n\n\nimport os\n\nTILE = 8\n\n\ndef helper(value):\n    return value + TILE\n"
    assert _refuses(tmp_path, after)


def test_a_module_that_declares_dunder_all_is_never_forgiven(tmp_path):
    """Adding a name there changes an ``import *`` surface. Textual, because it
    costs nothing and errs toward selecting."""

    before = BEFORE + '\n__all__ = ["helper"]\n'
    selects_today, still_selects = _arms(
        tmp_path, before, before + "\n\ndef added():\n    return 1\n"
    )
    assert selects_today
    assert still_selects


def test_a_declared_child_process_entry_point_is_never_forgiven(tmp_path):
    """``OUT_OF_PROCESS_DEPENDENCIES`` re-attaches these at whole-file
    granularity because nothing about the child is traced, and forgiveness rests
    on "no execution we recorded can observe the new name".

    This exclusion is load-bearing rather than decorative, and the second half
    of the test is what proves it: the argument that a whole-file fingerprint
    "contains every block, so any real change removes one" does **not** hold for
    a pure addition — every old block survives, and every one of them would be
    forgiven.
    """

    declared = sorted(testmon_policy.declared_whole_file_paths())
    assert declared, "the declaration set is what this test is about"
    entry_point = declared[0]

    after = BEFORE + "\n\ndef added():\n    return 1\n"
    selects_today, still_selects = _arms(
        tmp_path, BEFORE, after, relative=entry_point, fingerprint=_whole_file(BEFORE)
    )
    assert selects_today
    assert still_selects, "a declared entry point must stay selected"

    # Same edit, same whole-file fingerprint, undeclared path: forgiven. So the
    # exclusion above is doing the work, not the shape of the fingerprint.
    selects_today, still_selects = _arms(
        tmp_path, BEFORE, after, relative="pkg/undeclared.py", fingerprint=_whole_file(BEFORE)
    )
    assert selects_today
    assert not still_selects, "the edit must be forgiven"


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #


def test_no_retained_revision_forgives_nothing_and_raises_nothing(tmp_path):
    """A fresh sidecar, or more than five revisions of drift. The failure mode
    has to be over-selection, and it has to be silent."""

    (tmp_path / RELATIVE).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / RELATIVE).write_text(BEFORE, encoding="utf-8")
    assert testmon_policy.forgivable_checksums(tmp_path, RELATIVE, []) == []
    assert testmon_policy.forgivable_checksums(tmp_path, RELATIVE, [{}]) == []
    assert testmon_policy.forgivable_checksums(tmp_path, RELATIVE, [{"module": 1}]) == []
    assert testmon_policy.forgivable_checksums(tmp_path, "gone.py", [{"module": 1}]) == []
    assert testmon_policy.forgivable_checksums(tmp_path, "icon.png", [{"module": 1}]) == []


def test_a_sidecar_from_an_older_schema_forgives_nothing(tmp_path):
    """An entry without ``elements`` predates this and cannot be reasoned about;
    one whose ``functions`` rows have a different shape is skipped row by row."""

    entry = _revision(tmp_path, BEFORE)
    (tmp_path / RELATIVE).write_text(BEFORE + "\n\ndef added():\n    return 1\n", encoding="utf-8")
    stale = {key: value for key, value in entry.items() if key != "elements"}
    assert entry["module"] not in testmon_policy.forgivable_checksums(tmp_path, RELATIVE, [stale])
    assert testmon_policy.forgivable_checksums(tmp_path, RELATIVE, [{"functions": [[1]]}]) == []


def test_a_retained_revision_is_stripped_but_still_forgives(tmp_path):
    """A superseded revision is only ever compared against, so it keeps the two
    ends of each block row and drops the line and name it would be reported by —
    1.20 MB per generation against 0.79 MB over this tree's 572 tracked files."""

    entry = _revision(tmp_path, BEFORE)
    slim = coverage_map._slim(entry)
    assert [len(record) for record in slim["functions"]] == [2] * len(entry["functions"])
    assert slim["elements"] == entry["elements"]

    (tmp_path / RELATIVE).write_text(BEFORE + "\n\ndef added():\n    return 1\n", encoding="utf-8")
    assert testmon_policy.forgivable_checksums(
        tmp_path, RELATIVE, [slim]
    ) == testmon_policy.forgivable_checksums(tmp_path, RELATIVE, [entry])


def test_an_unparseable_revision_is_no_opinion_rather_than_no_change(tmp_path):
    """A syntax error must not read as "the decomposition is empty", which would
    forgive everything."""

    assert coverage_map.elements_of("def (\n", RELATIVE) is None
    assert coverage_map.revision_of("def (\n", RELATIVE) == ([], None, None)
    entry = _revision(tmp_path, BEFORE)
    (tmp_path / RELATIVE).write_text("def (\n", encoding="utf-8")
    assert testmon_policy.forgivable_checksums(tmp_path, RELATIVE, [entry]) == []


@pytest.fixture
def unpatched():
    """testmon's own ``collect_mhashes``, and the module restored afterwards.

    The run executing these tests has already installed the patch for the real
    checkout, so without this the "before" arm below would be the patch itself.
    The marker attribute carries what it replaced, which is exactly what is
    needed to undo it.
    """

    from testmon import testmon_core

    installed = testmon_core.collect_mhashes
    original = getattr(installed, "arrayscope_forgiveness", None) or installed
    testmon_core.collect_mhashes = original
    try:
        yield original
    finally:
        testmon_core.collect_mhashes = installed


def test_installing_forgiveness_without_a_sidecar_leaves_selection_alone(tmp_path, unpatched):
    """No sidecar means no retained revision, so the patch is not installed at
    all and the next call retries — a fresh checkout seeds one with its map."""

    from testmon import testmon_core

    assert testmon_policy.install_forgiveness(tmp_path) is False
    assert testmon_core.collect_mhashes is unpatched


def test_the_patch_can_only_ever_add_checksums(tmp_path, unpatched):
    """The safety floor, asserted rather than argued: ``check_fingerprint_db``
    is a subset test, so as long as the patched ``collect_mhashes`` returns a
    superset of the original for every file, no test can start running that was
    not running before."""

    from testmon import testmon_core

    coverage_map.write_sidecar(
        tmp_path / ".testmondata",
        structure={RELATIVE: _revision(tmp_path, BEFORE)},
        baselines={},
        revisions={},
    )
    (tmp_path / RELATIVE).write_text(BEFORE + "\n\ndef added():\n    return 1\n", encoding="utf-8")

    assert testmon_policy.install_forgiveness(tmp_path) is True
    assert testmon_policy.install_forgiveness(tmp_path) is True, "idempotent"
    patched = testmon_core.collect_mhashes
    assert patched is not unpatched

    class _Tree:
        @staticmethod
        def get_file(name):
            from testmon.process_code import Module

            return Module(source_code=(tmp_path / name).read_text(encoding="utf-8"), filename=name)

    plain = unpatched(_Tree, [RELATIVE])
    forgiving = patched(_Tree, [RELATIVE])
    assert set(plain[RELATIVE]) < set(forgiving[RELATIVE])


# --------------------------------------------------------------------------- #
# The decomposition itself
# --------------------------------------------------------------------------- #


def test_the_elements_are_the_substrings_testmon_joins(tmp_path):
    """The load-bearing structural claim, against testmon's own output.

    The module block's code is ``f"{index}:"`` followed by the top-level
    statement representations joined with ``", "``. If a testmon upgrade changed
    its dump format or its transform tuple, this is what says so — and the
    forgiveness gate would be reasoning about pieces that are no longer what it
    compares.
    """

    from testmon.process_code import Module

    source = "import os\n\nTILE = 8\n\n\ndef helper():\n    return os, TILE\n"
    module = Module(source_code=source, filename=RELATIVE)
    elements = coverage_map.elements_of(source, RELATIVE)
    joined = ", ".join(element.code for element in elements)
    assert joined == coverage_map._stripped(module.blocks[-1].code)


def test_a_class_shell_and_its_body_move_independently(tmp_path):
    """Which is the entire reason a class is split: the shell carries the bases
    and decorators the gate has to see, and the body carries the methods whose
    addition must read as an addition rather than a modified class."""

    before = coverage_map.elements_of(BEFORE, RELATIVE)
    shells = [element for element in before if element.kind == "class"]
    assert len(shells) == 1
    assert [element.scope for element in before].count(0) == 0, "module scope is -1"
    assert before[shells[0].stop - 1].scope == before.index(shells[0])

    renamed = coverage_map.elements_of(BEFORE.replace("class Thing:", "class Other:"), RELATIVE)
    assert renamed[before.index(shells[0])].digest != shells[0].digest
    assert renamed[-1].digest == before[-1].digest, "the method element must not move"


def test_a_decomposition_of_every_source_file_stays_cheap_and_total():
    """Every Python file this tree ships has to decompose, or forgiveness is
    silently off for it — and "silently off" is the state this whole change
    exists to leave behind."""

    root = testmon_policy.REPO_ROOT
    failures = []
    for relative in coverage_map.source_files(root, ("arrayscope",)):
        source = (root / relative).read_text(encoding="utf-8")
        if coverage_map.elements_of(source, relative, root) is None:
            failures.append(relative)
    assert not failures


# --------------------------------------------------------------------------- #
# --since (the same two rules, mirrored)
# --------------------------------------------------------------------------- #


def _since_arms(tmp_path, baseline: str, current: str, *, relative: str = RELATIVE, lines=None):
    """``(reached_today, reached_with_forgiveness)`` for one branch change.

    Mirror image of the map case: here the fingerprint is *current* and the
    checksum list is the *baseline*'s, so the equivalent checksums to append are
    the current ones.
    """

    from testmon.db import check_fingerprint_db

    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(current, encoding="utf-8")
    recorded = _record(current, lines or (COVERED_LINE,))

    baseline_checksums = _checksums(baseline)
    extra = testmon_policy.since_forgivable_checksums(tmp_path, relative, baseline.encode("utf-8"))
    selects_today = not check_fingerprint_db({relative: baseline_checksums}, relative, recorded)
    still_selects = not check_fingerprint_db(
        {relative: baseline_checksums + extra}, relative, recorded
    )
    return selects_today, still_selects


def test_since_no_longer_reaches_a_file_the_branch_only_added_a_function_to(tmp_path):
    """Measured, translating the settled map forward: 166 → 32 on ``lod.py``,
    258 → 0 on ``frame_session.py``, 157 → 16 on ``view_state.py``, 297 → 0 on a
    test file. ``--since`` is the pre-merge command, so this is the case that
    made it a flag nobody could afford."""

    selects_today, still_selects = _since_arms(
        tmp_path, BEFORE, BEFORE + "\n\ndef added():\n    return 1\n"
    )
    assert selects_today
    assert not still_selects, "the edit must be forgiven"


def test_since_still_reaches_a_body_the_branch_changed(tmp_path):
    """The measurement that says the mirrored rules did not simply switch the
    flag off: a changed body is still reached."""

    current = BEFORE.replace("return value + TILE", "return value - TILE")
    selects_today, still_selects = _since_arms(tmp_path, BEFORE, current)
    assert selects_today
    assert still_selects


def test_since_still_reaches_a_changed_constant_and_a_new_import(tmp_path):
    for current in (BEFORE.replace("TILE = 8", "TILE = 16"), BEFORE + "\nimport sys\n"):
        selects_today, still_selects = _since_arms(tmp_path, BEFORE, current)
        assert selects_today
        assert still_selects, current


def test_since_forgives_a_block_the_branch_only_moved(tmp_path):
    """The index-free digest rule on its own: the branch inserted a definition
    above ``helper``, so ``helper``'s checksum shifted while its code did not."""

    current = "def inserted():\n    return 0\n\n\n" + BEFORE
    line = current.splitlines().index("    return value + TILE") + 1
    selects_today, still_selects = _since_arms(tmp_path, BEFORE, current, lines=(line,))
    assert selects_today
    assert not still_selects, "the edit must be forgiven"


def test_since_refuses_the_same_things_the_map_does(tmp_path):
    """One gate, two callers. Spot-checked here rather than restated: a decorated
    definition, a rebinding, ``__all__`` and a declared child-process entry
    point are refused on this path too."""

    for current in (
        BEFORE + "\n\n@os.wrap\ndef added():\n    return 1\n",
        BEFORE + "\n\ndef helper(value):\n    return 0\n",
    ):
        selects_today, still_selects = _since_arms(tmp_path, BEFORE, current)
        assert selects_today
        assert still_selects, current

    baseline = BEFORE + '\n__all__ = ["helper"]\n'
    selects_today, still_selects = _since_arms(
        tmp_path, baseline, baseline + "\n\ndef added():\n    return 1\n"
    )
    assert selects_today
    assert still_selects

    entry_point = sorted(testmon_policy.declared_whole_file_paths())[0]
    selects_today, still_selects = _since_arms(
        tmp_path, BEFORE, BEFORE + "\n\ndef added():\n    return 1\n", relative=entry_point
    )
    assert selects_today
    assert still_selects


def test_since_forgiveness_never_raises_on_a_revision_it_cannot_read(tmp_path):
    """``--since`` is a merge gate; it degrades to selecting more, never to
    failing."""

    assert testmon_policy.since_forgivable_checksums(tmp_path, "gone.py", b"x = 1\n") == []
    assert testmon_policy.since_forgivable_checksums(tmp_path, "icon.png", b"x = 1\n") == []
    (tmp_path / RELATIVE).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / RELATIVE).write_text(BEFORE, encoding="utf-8")
    assert testmon_policy.since_forgivable_checksums(tmp_path, RELATIVE, b"def (\n") == []


@pytest.mark.parametrize(
    ("expression", "inert"),
    [
        ("1", True),
        ("None", True),
        ("-1", True),
        ("os.sep", True),
        ("(1, 2)", True),
        ("{'a': 1}", True),
        ("os.getpid()", False),
        ("[x for x in y]", False),
        ("lambda: 1", False),
        ("A + B", False),
        ("f'{value}'", False),
        ("(1, os.getpid())", False),
        ("{'a': os.sep}", True),
    ],
)
def test_the_default_argument_whitelist(expression, inert):
    """A whitelist, not a blacklist: an unlisted form costs one over-selection,
    which is the only failure mode this is allowed to have. ``BinOp`` is
    deliberately out — ``A + B`` over two module-level names calls
    ``A.__add__``."""

    node = ast.parse(expression, mode="eval").body
    assert coverage_map._inert_expression(node) is inert


# --------------------------------------------------------------------------- #
# The sidecar carries more than one feature's state
# --------------------------------------------------------------------------- #


def test_a_sidecar_write_preserves_the_fields_it_was_not_given(tmp_path):
    """``None`` means "keep what is on disk", never "clear".

    The sidecar holds state for independent features — retained revisions for
    forgiveness, the erosion canary's high-water mark — and each call site
    updates only the slice it owns. When the two features were merged from
    separate branches, every writer defaulted the other's field to empty, so
    whichever ran last silently disabled the other. This pins the repaired
    contract: an omitted field survives a rewrite, an explicit empty dict
    still clears."""

    map_path = tmp_path / ".testmondata"
    coverage_map.write_sidecar(
        map_path,
        structure={},
        baselines={},
        revisions={"pkg/mod.py": [{"fsha": "a", "module": 1, "elements": [], "functions": []}]},
        map_sizes={"offscreen": 3933},
    )

    coverage_map.write_sidecar(map_path, structure={}, baselines={})
    payload = coverage_map.read_sidecar(map_path)
    assert coverage_map.retained_revisions(payload) == {
        "pkg/mod.py": [{"fsha": "a", "module": 1, "elements": [], "functions": []}]
    }
    assert coverage_map.map_sizes(payload) == {"offscreen": 3933}

    coverage_map.write_sidecar(map_path, structure={}, baselines={}, revisions={}, map_sizes={})
    payload = coverage_map.read_sidecar(map_path)
    assert coverage_map.retained_revisions(payload) == {}
    assert coverage_map.map_sizes(payload) == {}
