"""Read executed-function coverage out of the selection map, for free.

The map already holds, per test, the AST blocks that test executed. Unioned
across every test it has recorded, it answers a question nobody was asking it:
**which functions in ``arrayscope/`` does the suite execute?** No coverage run,
no extra tracing, no wall clock — one SQL query plus the block structure of the
source tree.

It is legitimate on a *selected* run, which is the whole point: a test the map
deselected keeps its recorded fingerprint, and that fingerprint is still valid by
construction — the same invariant selection itself rests on. It self-heals, too:
when a file changes, every test that recorded it is affected, so it re-runs and
re-records in the same breath, and stale checksums stop matching the current tree
and drop out of the numerator.

What it is **not** is `coverage.py`'s line percentage, and the two must never be
quoted as if they were. Measured on identical execution data (``tests/core`` plus
``tests/kernel``, coverage.py's own ``executed_lines`` folded through testmon's
own ``create_fingerprint``):

    coverage.py lines   14.8%
    testmon    blocks   13.1%
    per file            median +0.0, but the spread runs -36.1 to +17.4 points

The aggregate agreement is two large biases cancelling, not accuracy. Per file
they diverge violently, in both directions, for one structural reason each:

* **Upward** — a block counts as covered when *any* line in it ran, so a
  function entered and abandoned at its first guard reads exactly like one run
  to completion.
* **Downward** — a module's top-level statements (imports, class bodies, every
  ``def``, every constant) are a *single* block spanning the whole file, so
  merely importing a module makes coverage.py call most of its statements
  covered while this counts one block.

The second is severe enough that the module block is excluded from the metric
entirely: it spans line one to EOF, so "the module block is covered" only ever
means "some test touched this file at all". That is reported separately, as
reach, and never mixed into the function figure.

So this measures **executed-function coverage**, a different metric from CI's
Codecov line figure, and a *floor* rather than an estimate — a test the map has
never recorded contributes nothing, so :attr:`Reach.recorded_tests` is the figure
that says how tight the floor is, and is printed with it.

Validated against a full ``pytest --cov`` run on the same tree, with a completely
recorded map: 5312 of 6204 functions from the map against 5271 ground truth,
**0.66 points apart**. Of the difference, 99 were functions the map knew about and
that day's coverage run skipped, and 58 were seen by coverage.py and invisible
here — mostly child processes, testmon's first blind spot.

Cost: one SQL query (~6 ms over 5.7k fingerprint rows) plus block structure for
the tree. Parsing all 295 files costs ~2.3 s, more than a small selected run
takes end to end, so the structure is cached beside the map keyed by each file's
git blob sha — the same content addressing testmon uses, so it is valid across
checkouts and invalidated by exactly the files that changed. Steady state is a
read-and-hash of the tree, ~20 ms.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

#: Sits next to the map, shares its lifetime, gitignored like it. Holds the parse
#: cache *and* this checkout's inherited baseline, so seeding a worktree's map
#: alongside this file makes the default comparison "since where I branched
#: from" with nothing to bootstrap. See ``testmon_policy.seed_map``.
COVERAGE_SUFFIX = "-coverage.json"

_VERSION = 1

#: Revisions of each file kept beside the current one, so an edited file still
#: has the revision the map was recorded against to be compared with. Five is
#: what an edit loop burns through between two runs of the suite; past that the
#: fingerprints have usually moved on as well, and a missing predecessor costs
#: one over-selection — never a wrong answer. Measured cost of the retention:
#: see ``docs/testing/test-selection.md``.
RETAINED_REVISIONS = 5


def sidecar_path(map_path) -> Path:
    map_path = Path(map_path)
    return map_path.with_name(map_path.name + COVERAGE_SUFFIX)


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


@dataclass(frozen=True)
class Function:
    """One function or method body, as the map identifies it."""

    path: str
    line: int
    name: str
    #: testmon's own block checksum — what the map stores, and therefore the only
    #: thing that can be looked up *in* it. It includes the block's index within
    #: the file, so inserting a function above this one changes it.
    checksum: int
    #: Digest of the block's representation with that index removed. Stable
    #: across edits elsewhere in the file, which is what comparing two
    #: revisions needs: otherwise one insertion at the top of a file reports
    #: every function below it as changed.
    body: str
    #: Occurrence index, to keep two byte-identical same-named bodies in one
    #: file from collapsing into a single entry and unbalancing the totals.
    occurrence: int = 0

    @property
    def identity(self) -> str:
        """Cross-run identity: this body, in this file, this many times over."""

        return f"{self.path}#{self.body}#{self.occurrence}"

    def __str__(self) -> str:
        return f"{self.path}:{self.line}  {self.name}"


@dataclass(frozen=True)
class Reach:
    """What the map says the suite executes, as of now."""

    environment: str
    covered: int
    total: int
    files_reached: int
    files_total: int
    #: Tests the map holds for this environment. The figure above is a floor of
    #: exactly this tightness: a test the map has never recorded contributes
    #: nothing, and there are two ordinary ways for that to happen — a ring
    #: never run here, and testmon's own housekeeping, which drops a map entry
    #: for an affected test that a *scoped* run did not collect.
    recorded_tests: int = 0
    #: ``{"arrayscope/render": (covered, total)}``, two path components deep.
    per_package: dict[str, tuple[int, int]] = field(default_factory=dict)
    uncovered: tuple[Function, ...] = ()
    #: Files no recorded test touched at all. Note the systematic false
    #: positive: a package ``__init__`` is imported at *collection* time, before
    #: any test's tracing starts, so it never appears under a test. 19 of this
    #: tree's 31 unreached files are that, not dead code.
    unreached_files: tuple[str, ...] = ()
    #: Files whose content still differs from the map when this was measured, so
    #: their functions read as uncovered until the tests using them re-run.
    #: Non-zero means the figure is provisional — normal mid-edit, and after a
    #: run scoped by path or ``-k``, which leaves deselected tests the map would
    #: otherwise have counted.
    provisional_files: int = 0
    #: Identities of the covered functions, for comparison against a baseline.
    identities: frozenset[str] = frozenset()

    @property
    def percent(self) -> float:
        return 100.0 * self.covered / self.total if self.total else 0.0


def _stripped(code: str) -> str:
    """A block's representation without testmon's per-file block index.

    ``Block.code`` is ``f"{index}:{ast_repr}"``. The index is what makes an
    inserted function invalidate every block below it — right for selection,
    where over-running is safe, and wrong for "what did this branch change",
    which would otherwise name a whole file after a one-line insertion.
    """

    _, separator, rest = code.partition(":")
    return rest if separator else code


def functions_of(rootdir, relative: str) -> tuple[list[Function], int | None]:
    """``(functions, module_block_checksum)`` for one file at its current revision."""

    from testmon.process_code import read_source_sha

    source, _ = read_source_sha(Path(rootdir) / relative)
    if source is None:
        return [], None
    functions, module_checksum, _ = revision_of(source, relative, rootdir)
    return functions, module_checksum


def revision_of(
    source_code: str, relative: str, rootdir=None
) -> tuple[list[Function], int | None, list[Element] | None]:
    """Blocks *and* top-level elements of one revision, from a single parse.

    Both halves come from one AST: selection asks for them together for every
    file it has to adjudicate, and parsing this tree twice measured 13 s against
    7 s for the sidecar's cold pass.

    testmon appends a body's block *after* the bodies nested inside it, so the
    module block — which encloses everything — is always the last one. Taken
    positionally rather than by span: a file holding a single function starting
    at line one would tie on width.

    An unparseable revision yields ``([], None, None)``. ``None`` elements are
    "no opinion", never "no top-level statements": testmon records no
    fingerprint for a file it cannot parse, and a caller that read a syntax
    error as an empty decomposition would forgive everything.
    """

    from testmon.process_code import Module, methods_to_checksums

    try:
        module = Module(
            source_code=source_code,
            filename=relative,
            rootdir=None if rootdir is None else str(rootdir),
        )
        text = module.source_code
        tree = ast.parse(text)
    except (SyntaxError, ValueError, TypeError, UnicodeDecodeError):
        return [], None, None

    end = len(text.splitlines())
    try:
        module._blocks = []
        module.dump_and_block(tree, end, name="<module>")
        blocks = list(module._blocks)
        checksums = methods_to_checksums([block.checksum for block in blocks])
        # A fresh counter and block list: the decomposition re-walks the same
        # tree and must not append to the run above.
        module._blocks = []
        module.counter = 0
        elements: list[Element] = []
        _decompose(module, tree.body, -1, end, elements)
    except Exception:  # a testmon upgrade moved dump_and_block: no opinion
        return [], None, None
    if not blocks:
        return [], None, elements

    functions: list[Function] = []
    seen: dict[str, int] = {}
    # Not strict: ``checksums`` carries the module block's too, and it is the one
    # entry deliberately left out of the function list.
    for block, checksum in zip(blocks[:-1], checksums, strict=False):
        body = _digest(_stripped(block.code))
        occurrence = seen.get(body, 0)
        seen[body] = occurrence + 1
        functions.append(
            Function(
                path=relative,
                line=block.start,
                name=block.name,
                checksum=checksum,
                body=body,
                occurrence=occurrence,
            )
        )
    return functions, checksums[-1], elements


def bodies_at(rootdir, relative: str, revision_source: bytes) -> set[str]:
    """Body digests of one file as it stood at another revision.

    Same digest as :attr:`Function.body`, so the two are directly comparable.
    An unparseable revision yields the empty set, which reads as "all of it is
    new" — the safe direction for a report about uncovered new code.
    """

    from testmon.process_code import Module

    try:
        module = Module(
            source_code=revision_source.decode("utf-8", "replace"),
            filename=relative,
            rootdir=str(rootdir),
        )
        return {_digest(_stripped(block.code)) for block in module.blocks}
    except Exception:
        return set()


# --------------------------------------------------------------------------- #
# Decomposing the module block
# --------------------------------------------------------------------------- #
#
# testmon puts a module's *whole* top-level body in one block spanning line 1 to
# EOF (`process_code.Module.dump_and_block` lists `Module` among the classes
# whose `body` field is transformed into a block). That block is therefore in
# 100% of the file's fingerprints -- measured 166/166 for `render/lod.py`,
# 258/258 for `window/frame_session.py`, 297/297 for a test file -- and
# `db.check_fingerprint_db` is a subset test against the file's *current*
# checksum list, so retiring that one 4-byte checksum retires every test that
# ever touched the file. Appending one inert function to `frame_session.py`
# selected 258 of 3933 tests and cost a measured 123.16 s of wall clock, for
# zero tests whose executed code had changed.
#
# What follows makes that single checksum adjudicable. The module block's code
# is the concatenation of one representation per top-level statement, so
# decomposing it into those pieces -- with testmon's own `dump_and_block`, so
# they are byte-for-byte the substrings testmon joined -- turns "the module
# block changed" into "these statements were added, those were removed".
#
# Class bodies are decomposed too, and have to be: `ClassDef` is *not* in
# testmon's transform tuple, so a class's body is dumped inline into the module
# block and adding one method would otherwise read as a modified class rather
# than an addition. Each method's *body* is still its own block, which is why
# editing a method body is precise today and adding one is not.


@dataclass(frozen=True)
class Element:
    """One statement of a module's top-level body, or of a class body inside it."""

    #: The statement's ``dump_and_block`` representation — byte-for-byte one of
    #: the substrings testmon joins to make the module block's code, so a guard
    #: can assert the decomposition against testmon's own output. Carries no
    #: block index, so it is stable against insertions elsewhere in the file.
    code: str
    #: ``"class"`` for a class *shell* — its name, bases, keywords, decorators
    #: and type parameters with the body emptied — whose body statements follow
    #: as elements of their own. ``"statement"`` for everything else.
    kind: str
    #: The AST node, so a gate can ask what an added element would execute.
    node: ast.stmt
    #: Index of the enclosing class shell, or -1 at module scope.
    scope: int
    #: One past this shell's last descendant. Only meaningful for a shell.
    stop: int

    @property
    def digest(self) -> str:
        return _digest(self.code)


def elements_of(
    source_code: str, filename: str = "<unknown>", rootdir=None
) -> list[Element] | None:
    """Decompose one revision's module block. ``None`` if it does not parse."""

    return revision_of(source_code, filename, rootdir)[2]


def _decompose(module, statements, scope: int, end: int, found: list[Element]) -> None:
    for statement in statements:
        if isinstance(statement, ast.ClassDef):
            body = statement.body
            statement.body = []  # the shell alone; the body follows as elements
            try:
                shell = module.dump_and_block(statement, end)
            finally:
                statement.body = body
            position = len(found)
            found.append(Element(shell, "class", statement, scope, position + 1))
            _decompose(module, body, position, end, found)
            found[position] = replace(found[position], stop=len(found))
        else:
            found.append(
                Element(module.dump_and_block(statement, end), "statement", statement, scope, 0)
            )


def digests_of(elements: Iterable[Element]) -> list[str]:
    """What the sidecar retains of a revision: the digests, in document order."""

    return [element.digest for element in elements]


def forgives_module_block(previous: Sequence[str], current: Sequence[Element]) -> bool:
    """Whether ``previous``'s module-block checksum still describes ``current``.

    True only when ``current`` is ``previous`` with statements *inserted*, and
    every inserted statement is one whose execution at import time cannot be
    observed by code that was already there.

    **Why insertion and not merely a superset.** Matching is a subsequence test,
    not a multiset one: swapping two top-level statements leaves the multiset
    untouched while changing what the module computes (``A = 1`` before
    ``B = A + 1`` is not the same file as the reverse). Order-preserving
    matching refuses that, at the price of also refusing a function *moved*
    within a file — which reads as one removal plus one addition, and is the
    honest reading, because the statements around it now run in a different
    order.

    **Soundness.** Every element of ``previous`` survives, byte-for-byte, at the
    same relative position; function bodies are already elided to
    ``transformed_into_block`` in these representations, and each has its own
    checksum which testmon compares independently. So no code a recorded test
    executed has moved or changed. The only difference the module object can
    show is names that did not exist when those tests ran, which is why the gate
    below refuses anything that could be *reached* rather than merely defined —
    a decorator, an import, a constant, a base class.

    Nothing here can make more tests run: the caller appends to the right-hand
    side of a subset test, so a wrong answer can only ever be an over-forgiveness
    of one checksum, and every clause below is a reason one cannot happen.
    """

    if not previous:
        return False  # nothing to forgive, and an empty list is not evidence
    index = 0
    added: list[int] = []
    for position, element in enumerate(current):
        if index < len(previous) and previous[index] == element.digest:
            index += 1
        else:
            added.append(position)
    if index != len(previous):
        return False  # something was removed, modified or reordered

    # Names the previous revision bound, per scope, read off the elements that
    # matched — which is exactly the previous revision, since nothing was
    # removed. An addition that rebinds one of them changes what an existing
    # name resolves to, and is refused.
    bound: dict[int, set[str]] = {}
    surplus = set(added)
    for position, element in enumerate(current):
        if position not in surplus:
            bound.setdefault(element.scope, set()).update(_binds(element.node))

    dataclasses_imported = any(
        element.scope == -1 and _imports_dataclasses(element.node) for element in current
    )
    consumed: set[int] = set()
    for position in added:
        if position in consumed:
            continue
        element = current[position]
        already = bound.get(element.scope, set())
        if element.kind == "class":
            descendants = range(position + 1, element.stop)
            if any(index not in surplus for index in descendants):
                return False  # a class whose shell moved but whose body did not
            if not _inert_class(element.node, already):
                return False
            consumed.update(descendants)
        else:
            host = current[element.scope].node if element.scope >= 0 else None
            if host is not None and not _extendable_class(host, dataclasses_imported):
                return False
            if not _inert_function(element.node, already):
                return False
    return True


def _extendable_class(node, dataclasses_imported: bool) -> bool:
    """Whether a method may be added to this *existing* class without notice.

    A class decorator receives the finished class and can read its body, so
    adding a method to a decorated class re-runs the decorator over different
    input. Census of this tree: 415 decorated classes, of which 411 are
    ``@dataclass`` and 4 are ``@runtime_checkable``.

    ``@runtime_checkable`` is refused because adding a method to such a Protocol
    changes what ``isinstance`` answers for code that was already there — a
    behaviour change with no block to notice it.

    ``@dataclass`` is allowed, and the reasoning is the whole of its contract:
    it derives fields from ``__annotations__`` (which a method does not touch,
    and an added class-body assignment is refused anyway) and otherwise only
    checks whether the body defines particular dunders — which
    :func:`_inert_function` refuses. The name is recognised syntactically, so it
    is honoured only in a module that actually imports ``dataclasses``.
    """

    if not node.decorator_list:
        return True
    return dataclasses_imported and all(
        _is_dataclass_decorator(decorator) for decorator in node.decorator_list
    )


def _is_dataclass_decorator(node) -> bool:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id == "dataclass"
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "dataclass"
        and isinstance(node.value, ast.Name)
        and node.value.id == "dataclasses"
    )


def _imports_dataclasses(node) -> bool:
    if isinstance(node, ast.ImportFrom):
        return node.module == "dataclasses"
    return isinstance(node, ast.Import) and any(alias.name == "dataclasses" for alias in node.names)


def _inert_function(node, already_bound: set[str], *, allow_dunder: bool = False) -> bool:
    """Whether adding this ``def`` can change what the module already did.

    Undecorated is not caution, it is the measured case: this tree registers
    plugin operations with import-time decorators
    (``arrayscope/operations/library.py``), so ``@register("thing")`` mutates a
    registry a test that never calls the function can read. There is deliberately
    no exemption for ``@pytest.*`` either — it would buy 11 file-touches worth
    901 test-selections and only 19 recorded seconds, which is not worth a second
    class of reasoning.

    A dunder name is refused wherever something could already be calling it,
    because what it overrides has no block to compare: ``def __repr__`` in a
    class body replaces ``object.__repr__`` (or the one ``@dataclass``
    generates), and a module-level ``def __getattr__`` (PEP 562) changes every
    attribute lookup on the module. Both are pure additions by the
    decomposition and behaviour changes in fact. Inside a *newly added* class
    the refusal is lifted: nothing can refer to a class that did not exist.

    Default arguments *do* evaluate at definition time, so they are gated to a
    whitelist of forms that cannot run project code. Annotations are not gated:
    PEP 649 (Python 3.14) evaluates them lazily, and this tree also carries
    ``from __future__ import annotations`` throughout.
    """

    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return False
    if node.decorator_list:
        return False
    if node.name.startswith("__") and not allow_dunder:
        return False
    if node.name in already_bound:
        return False
    arguments = node.args
    defaults = list(arguments.defaults) + [
        default for default in arguments.kw_defaults if default is not None
    ]
    return all(_inert_expression(default) for default in defaults)


def _inert_class(node, already_bound: set[str]) -> bool:
    """Whether adding this ``class`` can change what the module already did.

    A class body executes at definition time, and a base class gets
    ``__init_subclass__`` called on it, so a subclass of an existing class is a
    call into existing code however inert the body looks. Hence: no decorators,
    no bases, no keywords (which is where ``metaclass=`` lives), and a body of
    nothing but methods, docstrings, ``pass`` and literal assignments.
    """

    if not isinstance(node, ast.ClassDef):
        return False
    if node.decorator_list or node.bases or node.keywords:
        return False
    if node.name in already_bound:
        return False
    for statement in node.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            if not _inert_function(statement, set(), allow_dunder=True):
                return False
        elif isinstance(statement, ast.Pass):
            continue
        elif isinstance(statement, ast.Expr):
            if not isinstance(statement.value, ast.Constant):
                return False
        elif isinstance(statement, ast.Assign | ast.AnnAssign):
            if statement.value is not None and not _inert_expression(statement.value):
                return False
        else:
            return False
    return True


def _inert_expression(node) -> bool:
    """Whether evaluating this at import time cannot run project code.

    A whitelist, not a blacklist: an unlisted form costs one over-selection,
    which is the failure mode this whole file is allowed to have. ``BinOp`` is
    deliberately absent — ``A + B`` over two module-level names calls
    ``A.__add__``, which is a call however little it looks like one.
    """

    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return _inert_expression(node.value)
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return all(_inert_expression(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(key is not None and _inert_expression(key) for key in node.keys) and all(
            _inert_expression(value) for value in node.values
        )
    if isinstance(node, ast.UnaryOp):
        # `axis=-1` is the common one; restricted to constants so no operand can
        # carry a `__neg__` of its own.
        return isinstance(node.operand, ast.Constant)
    return False


def _binds(statement) -> set[str]:
    """Names one statement binds in the scope containing it.

    Descends through ``if`` / ``try`` / ``for`` / ``while`` / ``with`` bodies,
    which share the enclosing scope, and stops at a ``def`` or ``class``, which
    own theirs — so a local variable inside a function cannot masquerade as a
    module-level name and refuse an otherwise safe addition. Anything this does
    not model precisely falls back to walking the whole statement, which
    over-collects and therefore only ever refuses more.
    """

    if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return {statement.name}
    if isinstance(statement, ast.Import | ast.ImportFrom):
        return {
            alias.asname or alias.name.split(".", 1)[0]
            for alias in statement.names
            if alias.name != "*"
        }
    if isinstance(statement, ast.Assign):
        return set().union(*(_targets(target) for target in statement.targets))
    if isinstance(statement, ast.AnnAssign | ast.AugAssign):
        return _targets(statement.target)
    if isinstance(statement, ast.If | ast.Try | ast.While):
        return _binds_all(_bodies(statement))
    if isinstance(statement, ast.For | ast.AsyncFor):
        return _targets(statement.target) | _binds_all(_bodies(statement))
    if isinstance(statement, ast.With | ast.AsyncWith):
        names = _binds_all([statement.body])
        for item in statement.items:
            if item.optional_vars is not None:
                names |= _targets(item.optional_vars)
        return names
    if isinstance(statement, ast.Expr | ast.Pass | ast.Return | ast.Raise | ast.Delete):
        return set()
    return {
        node.id
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def _bodies(statement) -> list[list]:
    return [
        body
        for body in (
            getattr(statement, "body", None),
            getattr(statement, "orelse", None),
            getattr(statement, "finalbody", None),
        )
        if body
    ] + [handler.body for handler in getattr(statement, "handlers", ()) or ()]


def _binds_all(bodies) -> set[str]:
    names: set[str] = set()
    for body in bodies:
        for statement in body:
            names |= _binds(statement)
    return names


def _targets(node) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Tuple | ast.List):
        return set().union(*(_targets(item) for item in node.elts)) if node.elts else set()
    if isinstance(node, ast.Starred):
        return _targets(node.value)
    return set()  # an attribute or subscript target binds nothing new by name


def source_roots(rootdir) -> tuple[str, ...]:
    """Whatever ``[tool.coverage.run] source`` names, so the two agree.

    One denominator, declared once: if CI's coverage job measures a package,
    this measures the same package.
    """

    import tomllib

    try:
        with (Path(rootdir) / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)
    except (OSError, ValueError):
        return ("arrayscope",)
    source = config.get("tool", {}).get("coverage", {}).get("run", {}).get("source")
    if isinstance(source, list) and source:
        return tuple(str(entry) for entry in source)
    return ("arrayscope",)


def source_files(rootdir, roots: tuple[str, ...]) -> list[str]:
    rootdir = Path(rootdir)
    files: list[str] = []
    for root in roots:
        base = rootdir / root
        if base.is_file():
            files.append(base.relative_to(rootdir).as_posix())
        elif base.is_dir():
            files.extend(path.relative_to(rootdir).as_posix() for path in base.rglob("*.py"))
    return sorted(files)


def _read_only(map_path):
    """Our own read-only connection, or ``None``.

    Deliberately not testmon's: this runs at terminal-summary time, after xdist
    workers have written their fingerprints to the same file, and testmon's
    connection may still be holding an older snapshot.
    """

    map_path = Path(map_path)
    if not map_path.exists():
        return None
    try:
        # ``as_uri`` rather than an f-string: a checkout path containing ``?`` or
        # ``#`` would otherwise be parsed as query or fragment and read as a
        # missing database.
        connection = sqlite3.connect(f"{map_path.as_uri()}?mode=ro", uri=True)
    except (sqlite3.Error, ValueError):
        return None
    connection.row_factory = sqlite3.Row
    return connection


def read_map(map_path, environment: str) -> tuple[dict[str, set[int]], int] | None:
    """``({path: checksums some recorded test executed}, recorded test count)``.

    ``None`` if the map is missing or its schema has moved;
    ``tests/app/test_test_selection.py`` fails on the latter deliberately.
    """

    from testmon.process_code import blob_to_checksums

    connection = _read_only(map_path)
    if connection is None:
        return None
    try:
        rows = connection.execute(
            """
            SELECT f.filename, f.method_checksums
            FROM environment e, test_execution te, test_execution_file_fp tf, file_fp f
            WHERE e.environment_name = ?
              AND te.environment_id = e.id
              AND te.id = tf.test_execution_id
              AND tf.fingerprint_id = f.id
            """,
            (environment,),
        ).fetchall()
        tests = connection.execute(
            """
            SELECT count(*) FROM environment e, test_execution te
            WHERE e.environment_name = ? AND te.environment_id = e.id
            """,
            (environment,),
        ).fetchone()[0]
    except (sqlite3.Error, IndexError, TypeError):
        return None
    finally:
        with contextlib.suppress(sqlite3.Error):
            connection.close()

    covered: dict[str, set[int]] = {}
    for row in rows:
        blob = row["method_checksums"]
        if blob is not None:
            covered.setdefault(row["filename"], set()).update(blob_to_checksums(blob))
    return covered, int(tests)


def track_revisions(rootdir, map_path, environment: str, structure: Structure) -> int:
    """Parse every file the map depends on, so each keeps a revision history.

    :func:`measure` only walks the coverage source roots — ``arrayscope/`` — but
    selection is invalidated by any file a test recorded, and in this tree that
    is 578 files across ``arrayscope/``, ``tests/``, ``tools/`` and
    ``experiments/``. Test files are not a footnote here: adding one test to
    ``tests/display/test_odd_origin_crop_geometry.py`` selects all 297 tests in
    it, the largest single blast radius measured.

    Reads the map for filenames only, so the cost is one ``DISTINCT`` query plus
    a read-and-hash of files that have not changed. Returns how many paths were
    tracked, or 0 if the map could not be read — in which case the files simply
    have no retained predecessor and forgive nothing.
    """

    connection = _read_only(map_path)
    if connection is None:
        return 0
    try:
        rows = connection.execute(
            """
            SELECT DISTINCT f.filename
            FROM environment e, test_execution te, test_execution_file_fp tf, file_fp f
            WHERE e.environment_name = ?
              AND te.environment_id = e.id
              AND te.id = tf.test_execution_id
              AND tf.fingerprint_id = f.id
            """,
            (environment,),
        ).fetchall()
    except (sqlite3.Error, IndexError, TypeError):
        return 0
    finally:
        with contextlib.suppress(sqlite3.Error):
            connection.close()

    tracked = 0
    for row in rows:
        relative = row["filename"]
        if relative.endswith(".py") and structure.for_file(rootdir, relative) is not None:
            tracked += 1
    return tracked


class Structure:
    """Block structure for the source tree, cached by content.

    Parsing 295 files costs ~2.3 s, which is more than a small selected run
    takes end to end, so it cannot be paid every time. Keyed by git blob sha, so
    the cost after the first run is proportional to the diff — the same shape as
    everything else here.
    """

    def __init__(self, cached: dict | None = None, revisions: dict | None = None) -> None:
        self._cached = cached if isinstance(cached, dict) else {}
        self._revisions = revisions if isinstance(revisions, dict) else {}
        self._fresh: dict[str, dict] = {}
        self._retained: dict[str, list] = {}
        self.parsed = 0

    def for_file(self, rootdir, relative: str) -> dict | None:
        from testmon.process_code import read_source_sha

        source, fsha = read_source_sha(Path(rootdir) / relative)
        if source is None:
            return None
        entry = self._cached.get(relative)
        # "elements" missing means a sidecar written before module-block
        # forgiveness existed. Re-parsing once is what upgrades it; treating it
        # as fresh would leave forgiveness permanently off for every file that
        # never changes again.
        if isinstance(entry, dict) and entry.get("fsha") == fsha and "elements" in entry:
            self._fresh[relative] = entry
            self._retained[relative] = list(_revisions_for(self._revisions, relative))
            return entry
        functions, module_checksum, elements = revision_of(source, relative, rootdir)
        self.parsed += 1
        fresh = {
            "fsha": fsha,
            "module": module_checksum,
            "functions": [
                [function.checksum, function.line, function.name, function.body]
                for function in functions
            ],
            "elements": digests_of(elements) if elements is not None else None,
        }
        self._fresh[relative] = fresh
        # The revision just displaced is the one selection needs: the map's
        # fingerprints were recorded against it, so it is what an edited file
        # has to be compared with.
        history = [_slim(entry)] if isinstance(entry, dict) and entry.get("fsha") != fsha else []
        history.extend(_revisions_for(self._revisions, relative))
        self._retained[relative] = history[:RETAINED_REVISIONS]
        return fresh

    def payload(self) -> dict:
        """Only the files seen this time, so deletions expire the cache entry."""

        return self._fresh

    def retained(self) -> dict:
        """Previous revisions of each file seen this time, newest first."""

        return {relative: entries for relative, entries in self._retained.items() if entries}


def _slim(entry: dict) -> dict:
    """A retained revision, stripped to what forgiveness reads.

    A superseded revision is never reported on, only compared against, so the
    line and name of each block go. Measured on this tree's 572 tracked files:
    1.20 MB per generation against 0.79 MB, and the file grows by one generation
    per file this checkout edits between two runs of the suite.
    """

    return {
        "fsha": entry.get("fsha"),
        "module": entry.get("module"),
        "elements": entry.get("elements"),
        "functions": [
            [record[0], record[-1]] for record in entry.get("functions") or () if len(record) >= 2
        ],
    }


def _revisions_for(revisions: dict, relative: str) -> list:
    entries = revisions.get(relative)
    return (
        [entry for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []
    )


def previous_revisions(payload: dict, relative: str) -> list[dict]:
    """Every retained revision of ``relative``, newest first.

    The sidecar's current entry leads, because for the files selection asks
    about — the ones that differ from the map — it *is* the previous revision:
    the sidecar was last written by a run that had already recorded them. When
    it happens to match the working tree instead, forgiving its checksums is a
    no-op, since they are the ones already being compared against. Nothing is
    verified here: a revision is identified by its own content, and one no
    fingerprint refers to forgives nothing.
    """

    current = cached_structure(payload).get(relative)
    entries = [current] if isinstance(current, dict) else []
    entries.extend(_revisions_for(retained_revisions(payload), relative))
    return entries[:RETAINED_REVISIONS]


def functions_in(entry: dict, relative: str) -> list[Function]:
    functions: list[Function] = []
    seen: dict[str, int] = {}
    for checksum, line, name, body in entry.get("functions", ()):
        occurrence = seen.get(body, 0)
        seen[body] = occurrence + 1
        functions.append(
            Function(
                path=relative,
                line=line,
                name=name,
                checksum=checksum,
                body=body,
                occurrence=occurrence,
            )
        )
    return functions


def measure(
    rootdir,
    map_path,
    environment: str,
    *,
    structure: Structure | None = None,
    provisional_files: int = 0,
) -> Reach | None:
    """Executed-function coverage for one environment. ``None`` if unreadable.

    Per environment, never unioned across them: the map is keyed by ring (see
    ``environment_expression`` in ``pyproject.toml``), and merging an offscreen
    recording with a compositor one would credit code to a ring that has never
    executed it. A ring never run here reads as zero, which is the truth.
    """

    read = read_map(map_path, environment)
    if read is None:
        return None
    covered, recorded = read
    structure = structure if structure is not None else Structure()

    total = hit = files_total = files_reached = 0
    per_package: dict[str, list[int]] = {}
    uncovered: list[Function] = []
    unreached: list[str] = []
    identities: set[str] = set()

    for relative in source_files(rootdir, source_roots(rootdir)):
        entry = structure.for_file(rootdir, relative)
        if entry is None:
            continue
        seen = covered.get(relative, set())
        files_total += 1
        covered_here = 0
        functions = functions_in(entry, relative)
        # Reach has to be checksum-matched too, not merely "the map has rows for
        # this path". A file whose every recorded fingerprint is from an older
        # revision would otherwise read as reached while nothing in it matches.
        if entry.get("module") in seen or any(function.checksum in seen for function in functions):
            files_reached += 1
        else:
            unreached.append(relative)
        for function in functions:
            if function.checksum in seen:
                covered_here += 1
                identities.add(function.identity)
            else:
                uncovered.append(function)
        total += len(functions)
        hit += covered_here
        bucket = per_package.setdefault("/".join(relative.split("/")[:2]), [0, 0])
        bucket[0] += covered_here
        bucket[1] += len(functions)

    return Reach(
        environment=environment,
        covered=hit,
        total=total,
        files_reached=files_reached,
        files_total=files_total,
        recorded_tests=recorded,
        per_package={name: (values[0], values[1]) for name, values in per_package.items()},
        uncovered=tuple(uncovered),
        unreached_files=tuple(unreached),
        provisional_files=provisional_files,
        identities=frozenset(identities),
    )


# --------------------------------------------------------------------------- #
# What the figure is compared against
# --------------------------------------------------------------------------- #
#
# The map answers "what does the suite execute *now*". A delta needs a second
# point, and there are only three honest candidates:
#
#   1. The branch point, for the functions this branch added or changed. Exact,
#      and the one that matters before merging -- see `new_code` below.
#   2. What this checkout inherited when its map arrived. Observed rather than
#      inferred, exactly like tests/red_ledger.py and for the same reason: after
#      a worktree seeds its map from `main`, "what did I add or lose" is the
#      question, and it has a recorded answer.
#   3. Nothing -- a fresh clone gets the absolute figure and no delta, because
#      inventing a comparison is worse than not having one.
#
# What is NOT a candidate: the whole tree as it stood at the branch point. For a
# file this branch edited, the map holds fingerprints of the *current* revision
# only, because the tests re-recorded when they re-ran, so the baseline
# revision's coverage is not recoverable from it. A global "percent since main"
# would silently read every edited file as newly uncovered. That measurement is
# refused rather than approximated.


def read_sidecar(map_path) -> dict:
    try:
        payload = json.loads(sidecar_path(map_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != _VERSION:
        return {}
    return payload


def write_sidecar(
    map_path,
    *,
    structure: dict,
    baselines: dict,
    revisions: dict | None = None,
    map_sizes: dict | None = None,
) -> None:
    payload = {
        "version": _VERSION,
        "structure": structure,
        "baselines": baselines,
        "revisions": revisions or {},
        "map_sizes": dict(map_sizes or {}),
    }
    # A lost sidecar costs one re-parse and one baseline, never a wrong result.
    # ``revisions`` is optional for the same reason: a sidecar written before it
    # existed reads back with none, and selection is then exactly what it was.
    with contextlib.suppress(OSError, TypeError, ValueError):
        sidecar_path(map_path).write_text(json.dumps(payload), encoding="utf-8")


def cached_structure(payload: dict) -> dict:
    structure = payload.get("structure")
    return structure if isinstance(structure, dict) else {}


def retained_revisions(payload: dict) -> dict:
    revisions = payload.get("revisions")
    return revisions if isinstance(revisions, dict) else {}


def baselines(payload: dict) -> dict:
    recorded = payload.get("baselines")
    return dict(recorded) if isinstance(recorded, dict) else {}


def map_sizes(payload: dict) -> dict:
    """Per-environment high-water mark of the map's recorded test count.

    Kept beside the coverage baseline because it answers the same shape of
    question — "has this checkout's ground moved under it?" — and because the
    number it guards is the denominator of the coverage figure itself: an eroded
    map once read as a 15-point phantom regression.
    """

    recorded = payload.get("map_sizes")
    return (
        {str(key): int(value) for key, value in recorded.items()}
        if isinstance(recorded, dict)
        else {}
    )


def baseline_from(reach: Reach) -> dict:
    return {
        "covered": reach.covered,
        "total": reach.total,
        "identities": sorted(reach.identities),
    }


@dataclass(frozen=True)
class Drift:
    """How this checkout's coverage moved since the baseline it inherited."""

    gained: tuple[str, ...]
    lost: tuple[str, ...]
    baseline_covered: int
    baseline_total: int

    @property
    def moved(self) -> bool:
        return bool(self.gained or self.lost)


def drift(reach: Reach, baseline: dict | None) -> Drift | None:
    if not baseline:
        return None
    was = frozenset(baseline.get("identities", ()))
    return Drift(
        gained=tuple(sorted(reach.identities - was)),
        lost=tuple(sorted(was - reach.identities)),
        baseline_covered=int(baseline.get("covered", 0)),
        baseline_total=int(baseline.get("total", 0)),
    )


@dataclass(frozen=True)
class NewCode:
    """Functions this branch added or changed, split by whether a test runs them."""

    ref: str
    covered: tuple[Function, ...]
    uncovered: tuple[Function, ...]

    @property
    def total(self) -> int:
        return len(self.covered) + len(self.uncovered)


def new_code(
    rootdir,
    map_path,
    environment: str,
    ref: str,
    merge_base: str,
    *,
    structure: Structure | None = None,
) -> NewCode | None:
    """What this branch added or changed, and whether the suite executes it.

    A percentage is the wrong answer here (see the note above); this is the
    computable one, and it is exact in the direction that matters: a function
    whose body this branch touched necessarily re-recorded in the run that just
    happened, so "no test executes it" cannot be an artefact of a stale map.

    Comparison is on the index-free body digest, so inserting a function at the
    top of a file does not report every function below it as changed — which the
    raw block checksum would, and which is why testmon over-selects there.
    """

    from tests import testmon_policy

    read = read_map(map_path, environment)
    if read is None:
        return None
    covered, _ = read
    structure = structure if structure is not None else Structure()
    roots = source_roots(rootdir)

    tracked = testmon_policy._git(rootdir, "diff", "--name-only", merge_base)
    untracked = testmon_policy._git(rootdir, "ls-files", "--others", "--exclude-standard")
    if tracked is None or untracked is None:
        return None
    paths = sorted(
        {
            candidate
            for line in (tracked + untracked).splitlines()
            if (candidate := line.strip()).endswith(".py")
            and any(candidate.startswith(f"{root}/") or candidate == root for root in roots)
        }
    )

    fresh: list[Function] = []
    for relative in paths:
        entry = structure.for_file(rootdir, relative)
        if entry is None:
            continue  # deleted in this tree; nothing of it left to cover
        source = testmon_policy._git(rootdir, "show", f"{merge_base}:{relative}", binary=True)
        before = bodies_at(rootdir, relative, source) if source is not None else set()
        fresh.extend(
            function for function in functions_in(entry, relative) if function.body not in before
        )

    executed, missing = [], []
    for function in fresh:
        seen = covered.get(function.path, set())
        (executed if function.checksum in seen else missing).append(function)
    return NewCode(ref=ref, covered=tuple(executed), uncovered=tuple(missing))
