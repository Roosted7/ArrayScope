"""Definitions that no caller on the owning side reaches.

Three classes, all failures:

* **Unreachable** — nothing names it. Not ``arrayscope/``, not ``tests/``, not
  the repo-root product trees.
* **Test-only** — the only files naming it live under ``tests/``. This is the
  class that rots silently: the function still *executes*, so a coverage
  number reports it healthy, while no shipped code path reaches it.
* **Dead test helper** — the same question asked of the suite's own support
  modules (anything under ``tests/`` that is not a ``test_*.py`` or a
  ``conftest.py``). Without it, "delete the method, keep a shim" would be a way
  to launder dead code: neither production scan looks at ``tests/``, so a seam
  moved there to live beside its assertions could lose its last caller and
  nothing would say so.

There is no allowlist and no backlog, and there is deliberately nowhere to put
one. A definition the product cannot reach is either the product's, the
suite's, or nobody's; the answers are to give it a real caller, to move it to
the side that actually owns it, or to delete it with its assertions. The
harness package ``arrayscope/tools/`` is where an oracle or a benchmark ring
belongs — being called only from tests is *that* package's design, so it is
exempt from the test-only class alone (a definition nothing there reaches at
all still fails as unreachable).

This is a lint, not a test, and it is wired the way ruff is: the gate is
``.githooks/pre-commit`` via ``tools/dead_code.py``. Running the tree scan
again under pytest would buy nothing and cost several seconds on every suite run,
``tests/app/test_dead_code.py`` keeps only the synthetic proof that these
rules can actually fail.

``tests/coverage_map.py`` cannot answer either half, so this does not build on
it. Its ``uncovered`` set means "no recorded test entered this function": a
live function nobody tests is uncovered, and a test-only function is
*covered*. Reachability is a static question about who **names** a definition,
which is why it is asked statically here.

Two scopes, because one of them was a blind spot. Module-level ``def``/``class``
is the original rule. **Methods** are the second, added after a commit replaced
``PageTable.eviction_candidates`` with a cheaper query, left the old method
behind with only test callers, and this gate printed "dead code: none" — methods
were simply never candidates.

Methods are adjudicated only where a name scan honestly can: a method of a
top-level class that has **no base classes** and is **never used as a base**
in-tree. A class with a base can be dispatched into from outside the tree
(Qt calls ``paint``, ``logging`` calls ``filter``, a Protocol seam is called
through the interface), and a subclass can override; neither leaves a name a
scan can find. Measured on this tree, dropping that restriction takes the
report from 116 entries to 200 and every one of the extra 84 examined was
framework dispatch — the "long, ignored report" this rule is meant not to be.
Nested functions stay out of scope entirely.

A *reference* is any ``Name``, attribute name, import alias, or **string
constant** anywhere in the tree. Counting bare strings is what keeps dynamic
access honest: ``getattr(module, "name")``, the PEP 562 lazy re-export maps,
Qt ``connect()`` by slot name, and entry-point values all reach their target
through a string, and every one of them is credited as a caller here.

Two references deliberately do *not* count:

* A name's own definition — recursion is not a caller.
* An ``__all__`` entry — a declaration is not a caller. This is the rule that
  exposes rot: ``display_output_is_composited_rgb`` kept an ``__all__`` line
  and four test assertions for the whole time it had no production caller.

AST over ``arrayscope/`` and the repo-root product trees; a plain word scan
over ``tests/``, where over-approximating can only relabel a finding and never
hide one. Nothing is imported and no Qt object is built. All three scans share
one cached parse of the package, which is why adding the method rule made the
whole gate *faster* than the two scans that each parsed it themselves: 5.3-5.7 s
against 6.2-8.8 s before, on a machine with about half its cores busy.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

ROOT = Path(__file__).parents[1]

#: Repo-root trees that ship or drive the product rather than test it. The
#: gallery in ``tools/`` is a review surface, so a name it reaches is live.
_NON_TEST_ROOTS = ("tools", "wrappers", "packaging", "experiments")

_DEFINITION = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)

_STRUCTURAL_BASES = {"Protocol", "ABC"}

_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

_PACKAGE = "arrayscope"


#: ``arrayscope/tools/`` is the tracing/benchmark/oracle harness (docs/areas.md
#: routes it to ``tests/app`` + ``tests/stress``). Being called only from tests
#: is that package's design, so it is exempt from the *test-only* class alone --
#: a definition nothing there reaches at all still fails as unreachable.
_HARNESS_PACKAGE = "arrayscope/tools/"


@dataclass(frozen=True)
class Definition:
    relative: str
    name: str
    lineno: int

    def __str__(self) -> str:
        return f"{self.relative}:{self.lineno} {self.name}"


def _dunder_all_strings(tree: ast.Module) -> set[int]:
    """``id()`` of every string inside a module-level ``__all__`` assignment."""

    marked: set[int] = set()
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        if node.value is None:
            continue
        marked.update(
            id(child)
            for child in ast.walk(node.value)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        )
    return marked


def _names_referenced(node: ast.AST, declared: set[int]) -> set[str]:
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Attribute):
            found.add(child.attr)
        elif isinstance(child, ast.Constant):
            if isinstance(child.value, str) and id(child) not in declared:
                found.add(child.value)
        elif isinstance(child, ast.ImportFrom):
            for alias in child.names:
                found.add(alias.name)
                found.add(alias.asname or alias.name)
        elif isinstance(child, ast.Import):
            for alias in child.names:
                found.add(alias.name.split(".")[0])
                if alias.asname:
                    found.add(alias.asname)
    return found


def _structurally_exempt(node: ast.AST, relative: str) -> bool:
    """Entry points Python itself reaches, so no in-tree name can exist."""

    name = node.name
    # PEP 562 module ``__getattr__`` and every other dunder: called by protocol.
    if name.startswith("__") and name.endswith("__"):
        return True
    # pytest calls its hooks by name.
    if name.startswith("pytest_"):
        return True
    # ``[project.scripts] arrayscope = "arrayscope.__main__:main"``.
    if relative.endswith("__main__.py"):
        return True
    if isinstance(node, ast.ClassDef):
        # Structural types: an implementation never names its Protocol.
        if any(label in _STRUCTURAL_BASES for label in _base_labels(node)):
            return True
        for decorator in node.decorator_list:
            decorated = decorator.func if isinstance(decorator, ast.Call) else decorator
            label = (
                decorated.attr
                if isinstance(decorated, ast.Attribute)
                else getattr(decorated, "id", "")
            )
            if label == "runtime_checkable":
                return True
    return False


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py"))


@cache
def _package_trees(root: Path) -> tuple[tuple[str, ast.Module], ...]:
    """``(relative path, parsed module)`` for every file under ``root/arrayscope``.

    Cached because both scans below want the same parse of the same tree, and
    parsing the package twice is most of what this gate costs.
    """

    return tuple(
        (path.relative_to(root).as_posix(), ast.parse(path.read_text(encoding="utf-8")))
        for path in _python_files(root / _PACKAGE)
    )


@cache
def _product_root_names(root: Path) -> frozenset[str]:
    """Every name the shipping repo-root trees reference.

    A product-root reference SUPPRESSES a finding, so ``tools/dead_code.py``
    — this module's own CLI — is skipped: if it ever grew a mention of a
    listed name, it would silently hide that entry.
    """

    names: set[str] = set()
    for name in _NON_TEST_ROOTS:
        directory = root / name
        if not directory.is_dir():
            continue
        for path in _python_files(directory):
            if path.name == Path(__file__).name:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names |= _names_referenced(tree, _dunder_all_strings(tree))
    return frozenset(names)


@cache
def _words_under(directory: Path) -> frozenset[str]:
    """Every identifier-shaped word in a tree, without parsing it.

    Cached because both scans ask for the same ``tests/`` sweep, and reading
    that tree twice is most of what a second scan would cost.
    """

    words: set[str] = set()
    if not directory.is_dir():
        return words
    for path in _python_files(directory):
        # This module names every allowlisted and pending definition. Counting
        # itself would let an entry keep its own subject looking
        # test-referenced forever, which is what the staleness check must see.
        if path.name == Path(__file__).name:
            continue
        words.update(_WORD.findall(path.read_text(encoding="utf-8")))
    return words


@cache
def _words_per_test_file(root: Path) -> tuple[tuple[str, frozenset[str]], ...]:
    """``(relative path, words)`` for every file under ``tests/``.

    Per file rather than per tree, because a helper module naming its own
    definition must not count as that definition's caller -- the same rule the
    package scan applies, asked of the suite's own support code.
    """

    directory = root / "tests"
    if not directory.is_dir():
        return ()
    return tuple(
        (
            path.relative_to(root).as_posix(),
            frozenset(_WORD.findall(path.read_text(encoding="utf-8"))),
        )
        for path in _python_files(directory)
    )


def _is_test_module(relative: str) -> bool:
    """Whether pytest collects this file, rather than it being support code."""

    name = Path(relative).name
    return name.startswith("test_") or name == "conftest.py"


@cache
def unreferenced_test_helpers(root: Path) -> tuple[Definition, ...]:
    """Definitions in ``tests/`` support modules that no other file names.

    The suite's own helpers are the blind spot the package scan cannot see: a
    seam moved out of ``arrayscope/`` to live beside its assertions is still
    dead the moment its last caller goes, and nothing here is production, so
    neither the unreachable nor the test-only rule ever looks at it. This is
    the same question asked of ``tests/`` itself -- and it is what stops
    "delete the method, keep a shim" from being a way to launder dead code.

    Only *support* modules are in scope. A ``test_*.py`` file's own top-level
    definitions are its fixtures and helpers, and pytest reaches its tests by
    collection rather than by name.
    """

    files = _words_per_test_file(root)
    this_module = Path(__file__).resolve()
    found: list[Definition] = []
    for relative, _words in files:
        if _is_test_module(relative) or (root / relative).resolve() == this_module:
            continue
        text = (root / relative).read_text(encoding="utf-8")
        tree = ast.parse(text)
        # A sibling helper in the same module is a real caller, so count the
        # module's own words too -- but a name that appears exactly once there
        # is only its own ``def`` line.
        own = Counter(_WORD.findall(text))
        elsewhere: set[str] = set()
        for other, words in files:
            if other != relative:
                elsewhere |= words
        elsewhere |= _product_root_names(root)
        for node in tree.body:
            if not isinstance(node, _DEFINITION):
                continue
            if _structurally_exempt(node, relative) or node.name.startswith("_"):
                continue
            if node.name in elsewhere or own[node.name] > 1:
                continue
            found.append(Definition(relative, node.name, node.lineno))
    return tuple(found)


@cache
def unreferenced_definitions(root: Path) -> tuple[tuple[Definition, ...], tuple[Definition, ...]]:
    """``(unreachable, test_only)`` module-level definitions under ``root``.

    ``root`` is the repository root; the package scanned is ``root/arrayscope``.
    Taking the repo root (rather than the package) is what lets the synthetic
    fixture below drive the whole rule set, allowlist excluded.

    Cached because both guards below ask the same question of the same tree,
    and one parse of ``arrayscope/`` plus ``tests/`` is the whole cost here.
    """

    definitions: list[tuple[Definition, ast.AST]] = []
    # name -> number of distinct owners naming it; an owner is one top-level
    # definition, or one file's module level.
    package_references: dict[str, int] = {}
    owner_references: dict[Definition, set[str]] = {}

    def credit(names: set[str]) -> None:
        for name in names:
            package_references[name] = package_references.get(name, 0) + 1

    for relative, tree in _package_trees(root):
        declared = _dunder_all_strings(tree)
        module_level: set[str] = set()
        for node in tree.body:
            if isinstance(node, _DEFINITION):
                definition = Definition(relative, node.name, node.lineno)
                definitions.append((definition, node))
                owned = _names_referenced(node, declared) - {node.name}
                owner_references[definition] = owned
                credit(owned)
                # A decorator sits outside the body it decorates, so it is
                # module-level evidence -- a registration decorator naming a
                # sibling really does reach it.
                for decorator in node.decorator_list:
                    module_level |= _names_referenced(decorator, declared)
            else:
                module_level |= _names_referenced(node, declared)
        credit(module_level)

    # ``tests/`` is read with a word scan rather than a parse, which is most of
    # this guard's wall clock. It over-approximates -- a name in a test comment
    # counts -- and that is safe in exactly one direction: the test side only
    # ever decides which *label* a finding gets, never whether it is reported.
    # ``product_names`` does suppress findings, so those roots stay parsed.
    test_names = _words_under(root / "tests")
    product_names = _product_root_names(root)

    unreachable: list[Definition] = []
    test_only: list[Definition] = []
    for definition, node in definitions:
        if _structurally_exempt(node, definition.relative):
            continue
        owners = package_references.get(definition.name, 0)
        if definition.name in owner_references[definition]:
            owners -= 1
        if owners or definition.name in product_names:
            continue
        if definition.name in test_names:
            if not definition.relative.startswith(_HARNESS_PACKAGE):
                test_only.append(definition)
        else:
            unreachable.append(definition)
    return tuple(unreachable), tuple(test_only)


def _base_labels(node: ast.ClassDef) -> list[str]:
    return [
        base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
        for base in node.bases
    ]


@cache
def _dispatchable_classes(root: Path) -> frozenset[str]:
    """Class names something outside a name scan can call methods on.

    A class with a base may be dispatched into from outside the tree — Qt
    calls ``paint``, ``logging`` calls ``filter``, a Protocol seam is called
    through the interface — and a class used as a base may have any of its
    methods overridden or called on a subclass instance. Neither leaves a
    name in this tree, so neither is adjudicable and both are skipped whole.
    """

    dispatchable: set[str] = set()
    for _relative, tree in _package_trees(root):
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.bases:
                dispatchable.add(node.name)
            dispatchable.update(label for label in _base_labels(node) if label)
    return frozenset(dispatchable)


@cache
def unreferenced_methods(root: Path) -> tuple[tuple[Definition, ...], tuple[Definition, ...]]:
    """``(unreachable, test_only)`` methods of name-reachable classes.

    ``Definition.name`` is qualified (``PageTable.eviction_candidates``) so a
    report names the owner; references are still resolved on the bare method
    name, because that is all an attribute access carries. Two classes
    therefore share a verdict when they share a method name — a live ``A.foo``
    hides a dead ``B.foo``. That is the safe direction: this scan under-reports
    rather than accusing live code.

    An owner is one method body, one class body, one top-level function, or
    one file's module level. A method's own body is not an owner of itself
    (recursion is not a caller), but a *sibling* method is: ``self.foo()``
    from elsewhere in the class is a real caller.
    """

    references: dict[str, set[tuple[str, str]]] = {}
    candidates: list[tuple[Definition, str]] = []
    dispatchable = _dispatchable_classes(root)

    def credit(names: set[str], site: tuple[str, str]) -> None:
        for name in names:
            references.setdefault(name, set()).add(site)

    for relative, tree in _package_trees(root):
        declared = _dunder_all_strings(tree)
        module_level: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                adjudicable = node.name not in dispatchable
                class_level: set[str] = set()
                for item in node.body:
                    if not isinstance(item, _FUNCTION):
                        class_level |= _names_referenced(item, declared)
                        continue
                    qualified = f"{node.name}.{item.name}"
                    credit(_names_referenced(item, declared), (relative, qualified))
                    # ``@x.setter`` names the property it extends, so a
                    # decorator belongs to the class, not to the method.
                    for decorator in item.decorator_list:
                        class_level |= _names_referenced(decorator, declared)
                    if adjudicable and not (
                        item.name.startswith("__") and item.name.endswith("__")
                    ):
                        candidates.append((Definition(relative, qualified, item.lineno), item.name))
                credit(class_level, (relative, node.name))
                for decorator in node.decorator_list:
                    module_level |= _names_referenced(decorator, declared)
            elif isinstance(node, _FUNCTION):
                credit(_names_referenced(node, declared), (relative, node.name))
                for decorator in node.decorator_list:
                    module_level |= _names_referenced(decorator, declared)
            else:
                module_level |= _names_referenced(node, declared)
        credit(module_level, (relative, "<module>"))

    test_names = _words_under(root / "tests")
    product_names = _product_root_names(root)

    unreachable: list[Definition] = []
    test_only: list[Definition] = []
    for definition, bare in candidates:
        sites = references.get(bare, set()) - {(definition.relative, definition.name)}
        if sites or bare in product_names:
            continue
        if bare in test_names:
            if not definition.relative.startswith(_HARNESS_PACKAGE):
                test_only.append(definition)
        else:
            unreachable.append(definition)
    return tuple(unreachable), tuple(test_only)


@dataclass(frozen=True)
class Dangling:
    relative: str
    lineno: int
    alias: str
    module: str
    name: str

    def __str__(self) -> str:
        return f"{self.relative}:{self.lineno} {self.alias}.{self.name} ({self.module})"


def _module_of(relative: str) -> str:
    parts = relative[: -len(".py")].split("/")
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _module_level_bindings(tree: ast.Module) -> set[str]:
    """Every name a module binds at import time, over-approximated.

    Over-approximating is the safe direction here: a name this misses would be
    reported as dangling when it is not, and a guard that is wrong once gets
    deleted. Anything bound by any module-level statement counts, however it
    was bound.
    """

    bound: set[str] = set()
    for node in tree.body:
        if isinstance(node, _DEFINITION):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        else:
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    bound.add(child.id)
    return bound


def _module_aliases(tree: ast.Module) -> dict[str, str]:
    """``local name -> arrayscope module`` for every module imported by a file.

    Collected from anywhere, not just module level: lazy imports inside
    functions are load-bearing in this codebase (``PLC0415`` is ignored for
    exactly that reason), and a call through one dangles just as hard.
    """

    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(_PACKAGE):
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(_PACKAGE) and alias.asname:
                    aliases[alias.asname] = alias.name
    return aliases


def dangling_module_attributes(root: Path) -> list[Dangling]:
    """``module.name`` references inside ``arrayscope/`` that resolve to nothing.

    The mirror image of the dead-code scan: that one finds a definition with no
    reference, this one finds a reference with no definition. Python does not
    catch it until the line executes, so a call through a module alias can sit
    green in an untouched file indefinitely.

    It exists because of a failure git cannot see. A branch deleted
    ``render/effects.py::presented_first_pixel_payload``, correctly: nothing
    named it at that branch's base. Meanwhile ``7d50f5b0`` landed a caller in
    ``window/frame_effects.py``. The rebase touched different files, so it
    merged clean, and five ``tests/ui`` tests died on ``AttributeError`` --
    found by running the suite, which is the expensive way. This is the cheap
    way, and it is exactly as useful after a rebase as after an edit.

    A module with a PEP 562 ``__getattr__`` is skipped: anything resolves there.
    """

    files = _package_trees(root)
    modules = {_module_of(relative): _module_level_bindings(tree) for relative, tree in files}

    lazy = {name for name, bound in modules.items() if "__getattr__" in bound}

    found: list[Dangling] = []
    for relative, tree in files:
        aliases = _module_aliases(tree)
        if not aliases:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
                continue
            target = aliases.get(node.value.id)
            if target is None or target in lazy or target not in modules:
                continue
            if node.attr not in modules[target]:
                found.append(Dangling(relative, node.lineno, node.value.id, target, node.attr))
    return found


def problems(root: Path = ROOT) -> list[str]:
    """Every reason this tree should be refused, most actionable first."""

    unreachable, test_only = unreferenced_definitions(root)
    dead_methods = sum(unreferenced_methods(root), ())

    dangling = [
        f"dangling reference (nothing defines it): {entry}"
        for entry in dangling_module_attributes(root)
    ]
    if dangling:
        dangling.append("  -> a rebase can do this with no textual conflict: the caller and")
        dangling.append("     the deleted definition live in different files.")

    found = [f"unreachable (nothing names it):   {definition}" for definition in unreachable]
    found += [f"test-only (only tests/ names it): {definition}" for definition in test_only]
    if found:
        found.append("  -> delete it with its assertions, or restore a real caller.")
        found.append("     There is no allowlist: a definition the product cannot reach")
        found.append("     is either the product's, or the suite's, or nobody's.")

    helpers = [
        f"dead test helper (no test names it): {definition}"
        for definition in unreferenced_test_helpers(root)
    ]
    if helpers:
        helpers.append("  -> a seam that moved out of arrayscope/ to live beside its")
        helpers.append("     assertions is dead the moment its last caller goes too.")

    methods = [f"dead method (no production caller): {definition}" for definition in dead_methods]
    if methods:
        methods.append("  -> the caller you replaced was the last one; delete the method with")
        methods.append("     its assertions, or point the new seam's tests at the survivor.")

    return dangling + found + helpers + methods
