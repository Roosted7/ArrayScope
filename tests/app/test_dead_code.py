"""Proof that the dead-code rules in :mod:`tests.dead_code` can actually fail.

Ring 0 — no Qt, no imports of ``arrayscope``. The whole-tree scan is NOT run
here: it is a lint, gated at commit time by ``.githooks/pre-commit`` through
``tools/dead_code.py``, and running it again under pytest would cost seconds on
every suite run to re-answer a question the hook already refused to let past.

What is left is the part pytest is actually for: a synthetic package where
every rule has a case that would otherwise pass silently, plus the shape of
the two excuse lists. Both are instant — no real tree is read.
"""

from __future__ import annotations

from tests.dead_code import (
    _ALLOWLIST,
    _PENDING_ADJUDICATION,
    _PENDING_CEILING,
    _PENDING_METHOD_CEILING,
    _PENDING_METHODS,
    dangling_module_attributes,
    unreferenced_definitions,
    unreferenced_methods,
)


def test_allowlist_entries_each_carry_a_reason_and_are_not_also_pending():
    """The allowlist is the rule's only widening, so it stays legible."""

    missing_reason = [f"{path}::{name}" for path, name, reason in _ALLOWLIST if not reason.strip()]
    overlap = [
        f"{path}::{name} is both allowlisted and pending; pick one"
        for path, name, _ in _ALLOWLIST
        if (path, name) in _PENDING_ADJUDICATION
    ]
    duplicated = [
        f"{path}::{name} is listed twice"
        for path, name, _ in _ALLOWLIST
        if sum(1 for other, other_name, _ in _ALLOWLIST if (other, other_name) == (path, name)) > 1
    ]

    assert missing_reason + overlap + duplicated == []


def test_the_pending_backlog_declares_a_ceiling_it_is_within():
    """A backlog whose ceiling is set to its own size can only shrink."""

    assert len(_PENDING_ADJUDICATION) <= _PENDING_CEILING
    assert len(set(_PENDING_ADJUDICATION)) == len(_PENDING_ADJUDICATION)
    assert len(_PENDING_METHODS) <= _PENDING_METHOD_CEILING
    assert len(set(_PENDING_METHODS)) == len(_PENDING_METHODS)


def test_method_guard_reports_rot_without_accusing_dispatched_classes(tmp_path):
    """Falsifiability for the method rule, including what it must NOT report.

    The rule exists because ``PageTable.eviction_candidates`` kept only test
    callers after its production caller was replaced, and the module-level
    scan could not see it. Its whole risk is the opposite error — accusing a
    method something outside the tree dispatches into — so the cases that must
    stay silent are pinned here next to the ones that must fire.
    """

    package = tmp_path / "arrayscope"
    package.mkdir()
    (tmp_path / "tests").mkdir()

    (package / "live.py").write_text(
        "from arrayscope.shapes import Engine\n"
        "\n"
        "def drive():\n"
        "    engine = Engine()\n"
        "    engine.public_entry()\n"
        "    return engine.value_property\n",
        encoding="utf-8",
    )
    (package / "shapes.py").write_text(
        "class Engine:\n"
        "    def public_entry(self):\n"
        "        return self._helper()\n"
        "\n"
        "    def _helper(self):\n"
        "        return 1\n"
        "\n"
        "    @property\n"
        "    def value_property(self):\n"
        "        return 2\n"
        "\n"
        "    def only_a_test_calls_me(self):\n"
        "        return 3\n"
        "\n"
        "    def nobody_calls_me(self):\n"
        "        return 4\n"
        "\n"
        "    def recurses(self, n):\n"
        "        return 0 if n <= 0 else self.recurses(n - 1)\n"
        "\n"
        "    def __len__(self):\n"
        "        return 0\n"
        "\n"
        "\n"
        "class Painted(QtGraphicsItem):\n"
        "    def paint(self, painter):\n"
        "        return None\n"
        "\n"
        "\n"
        "class Base:\n"
        "    def overridable(self):\n"
        "        return 1\n"
        "\n"
        "\n"
        "class Derived(Base):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (package / "tools").mkdir()
    (package / "tools" / "harness.py").write_text(
        "class Oracle:\n    def read_for_tests(self):\n        return 1\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_shapes.py").write_text(
        "def test_it(engine, oracle):\n"
        "    assert engine.only_a_test_calls_me() == 3\n"
        "    assert oracle.read_for_tests() == 1\n",
        encoding="utf-8",
    )

    unreachable, test_only = unreferenced_methods(tmp_path)

    assert {definition.name for definition in unreachable} == {
        "Engine.nobody_calls_me",
        # Recursion is not a caller, exactly as at module level.
        "Engine.recurses",
    }
    assert {definition.name for definition in test_only} == {"Engine.only_a_test_calls_me"}
    # Silent by construction, and each for its own reason:
    #   public_entry   — named by live.py
    #   _helper        — a sibling method calls it
    #   value_property — reached as an attribute, not a call
    #   __len__        — Python calls it
    #   Painted.paint  — the class has a base, so Qt may dispatch into it
    #   Base.overridable / Derived — the class is subclassed in-tree
    #   Oracle.read_for_tests — arrayscope/tools/ is the test harness package


def test_guard_separates_dead_code_from_every_exempt_entry_point(tmp_path):
    """Falsifiability: each rule is driven by a case that would otherwise fail."""

    package = tmp_path / "arrayscope"
    package.mkdir()
    (tmp_path / "tests").mkdir()

    (package / "live.py").write_text(
        "from arrayscope.rot import kept_by_import\n"
        "import arrayscope.rot as rot\n"
        "\n"
        "def called_by_sibling():\n"
        "    return 1\n"
        "\n"
        "def reached_dynamically():\n"
        "    return 2\n"
        "\n"
        "def entry():\n"
        "    kept_by_import()\n"
        "    rot.kept_by_attribute()\n"
        "    called_by_sibling()\n"
        "    return getattr(rot, 'reached_dynamically')\n",
        encoding="utf-8",
    )
    (package / "rot.py").write_text(
        "from typing import Protocol, runtime_checkable\n"
        "\n"
        '__all__ = ["kept_by_import", "listed_but_dead"]\n'
        "\n"
        "def kept_by_import():\n"
        "    return 1\n"
        "\n"
        "def kept_by_attribute():\n"
        "    return 2\n"
        "\n"
        "def listed_but_dead():\n"
        "    return 3\n"
        "\n"
        "def only_a_test_calls_me():\n"
        "    return 4\n"
        "\n"
        "def recurses_but_nobody_calls_it(n):\n"
        "    return 0 if n <= 0 else recurses_but_nobody_calls_it(n - 1)\n"
        "\n"
        "def __getattr__(name):\n"
        "    return None\n"
        "\n"
        "def pytest_configure(config):\n"
        "    return None\n"
        "\n"
        "class StructuralContract(Protocol):\n"
        "    def read(self) -> int: ...\n"
        "\n"
        "@runtime_checkable\n"
        "class CheckableContract(Protocol):\n"
        "    def read(self) -> int: ...\n",
        encoding="utf-8",
    )
    (package / "tools").mkdir()
    (package / "tools" / "harness.py").write_text(
        "def oracle_for_tests():\n    return 1\n\ndef harness_rot():\n    return 2\n",
        encoding="utf-8",
    )
    (package / "__main__.py").write_text(
        "from arrayscope.live import entry\n\ndef main():\n    return entry()\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_rot.py").write_text(
        "from arrayscope.rot import only_a_test_calls_me\n"
        "from arrayscope.tools.harness import oracle_for_tests\n"
        "\n"
        "def test_it():\n"
        "    assert only_a_test_calls_me() == 4\n"
        "    assert oracle_for_tests() == 1\n",
        encoding="utf-8",
    )

    unreachable, test_only = unreferenced_definitions(tmp_path)

    assert {definition.name for definition in unreachable} == {
        # ``__all__`` is a declaration, not a caller.
        "listed_but_dead",
        # Recursion is not a caller.
        "recurses_but_nobody_calls_it",
        # Harness rot still fails; only the *test-only* class is exempt there.
        "harness_rot",
    }
    assert {definition.name for definition in test_only} == {"only_a_test_calls_me"}
    # oracle_for_tests is under arrayscope/tools/ and a test names it: exempt.
    # entry/main/__getattr__/pytest_configure/Protocols are never reported;
    # kept_by_* and reached_dynamically prove import, attribute, sibling-call
    # and getattr-string references all count.


def test_a_reference_to_a_missing_module_attribute_is_found(tmp_path):
    """The mirror of the dead-code scan: a reference with nothing behind it.

    This is the shape a rebase produces with no textual conflict — the caller
    and the deleted definition live in different files, so git merges cleanly
    and Python says nothing until the line runs. It cost five `tests/ui`
    failures to find that way once.
    """

    package = tmp_path / "arrayscope"
    package.mkdir()
    (package / "target.py").write_text(
        "CONSTANT = 1\n\n\ndef exists():\n    return 2\n\n\nclass Shape:\n    pass\n",
        encoding="utf-8",
    )
    (package / "lazy.py").write_text("def __getattr__(name):\n    return None\n", encoding="utf-8")
    (package / "caller.py").write_text(
        "from arrayscope import lazy\n"
        "from arrayscope import target as aliased\n"
        "\n"
        "def use():\n"
        "    aliased.exists()\n"
        "    aliased.Shape()\n"
        "    print(aliased.CONSTANT)\n"
        "    lazy.anything_at_all()\n"
        "    return aliased.was_deleted()\n",
        encoding="utf-8",
    )

    found = dangling_module_attributes(tmp_path)

    # Only the one that resolves to nothing. Functions, classes and plain
    # module-level assignments all count as definitions, and a PEP 562
    # __getattr__ module resolves anything.
    assert [(entry.alias, entry.name) for entry in found] == [("aliased", "was_deleted")]
    assert found[0].module == "arrayscope.target"
    assert found[0].relative == "arrayscope/caller.py"
