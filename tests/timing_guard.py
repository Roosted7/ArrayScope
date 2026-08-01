"""Pauses and stopwatches under ``tests/``, and which of them named a reason.

This week the suite stopped asking the wall clock questions it cannot answer.
672 fixed pauses became waits denominated in the product's own progress
(:mod:`tests.settle`), the measurements moved to the benchmark harness, and
``ba53fbc8`` restated the last wall-clock assertion on counters. Nothing
protects any of that. The next hurried commit re-adds ``qtbot.wait(300)``, it
passes on a quiet box, and weeks later somebody loses a session bisecting a
phantom regression that is really a busy machine.

So this module asks two questions of the tree, statically.

**A clock reading may not reach an assertion.** Not through a threshold, not
through a ratio, and not through a comparison between two measurements either:
``docs/testing/strategy.md`` records what that last form cost. The final
wall-clock assertion in the suite was ``empty_ms <= working_ms`` -- no constant
anywhere in it, machine-independent by construction, and still both flaky *and*
vacuous, because forcing a real defect back left the regained fixed cost on
both arms of the ``<=`` where it largely cancelled. **"Ratio-free and
threshold-free" is not the same as "not a stopwatch."** The rule here is
therefore about the reading and not about what is done with it: the suite
records durations freely (``tests/conftest.py``, ``tests/load_report.py`` and
``tests/seams.py`` all do) and adjudicates none.

**A fixed pause must name the throttle that bounds it.** A blanket ban is wrong
and has already been refused on evidence.
``tests/ui/test_interactive_render_coalescing.py`` is *about* a time-based
throttle, so a burst followed by a short pause is the only honest way to say
"only the latest render was requested within the interval" -- waiting for the
coalesced render to arrive asserts the opposite thing. Several
``process_events`` sites are equally real: a live event loop never idles, so
low-priority planning continuations only run under sustained queue pressure,
and an idle loop hides the field scheduling entirely.

The reason must be *structural*, and this is the part worth arguing. A comment
marker (``# fixed pause: needed because ...``) would work exactly once. It is
satisfiable by typing a sentence, it survives the deletion of the throttle it
names, and at fifty-odd sites it is an excuse list with the entries scattered
through the tree instead of gathered at the top of a file -- the shape
``819789f8`` deleted three of. What a pause is asked for here is the thing a
sentence cannot fake: **the duration has to be read from something rather than
typed**, from a constant the product or the toolkit publishes or from the live
object that owns the timer. ``qtbot.wait(2 * PUBLISH_COALESCE_MS)`` and
``qtbot.wait(win.render_coordinator.interactive_interval_ms)`` say which
throttle bounds them in code, and deleting that throttle breaks them rather
than leaving a comment describing a mechanism that no longer exists. There is
no allowlist and nowhere to put one.

``arrayscope.tools.interaction_budget`` is deliberately *not* such a throttle.
It owns the 5 s settlement deadline, and a pause bounded by it is a
five-second sleep with a respectable import in front of it.

Four shapes are not fixed pauses at all, and each is recognised by structure:

* **The poll cadence inside a predicate wait.** ``while not marker.exists() and
  monotonic() < deadline: sleep(0.005)`` waits on the marker, not on the clock;
  the deadline is the interaction budget's business, and
  ``test_interaction_gates_have_one_bounded_timeout_owner`` already adjudicates
  it. A loop qualifies when it can end on something other than its own clock --
  a non-clock term in the ``while`` test, or a ``break``/``return`` in the body.
  A loop that can only end on the clock is a fixed pause wearing a loop's
  clothes, and ``while perf_counter() < stop_at: qtbot.wait(1)`` is reported as
  one.
* **Simulated slow work.** ``time.sleep(0.15)`` inside a function handed to the
  product -- a monkeypatched loader, a ``TaskSpec`` payload, a lambda -- makes
  the *product* slow. That is the test's material, not the test's wait. It is
  detected as a function whose name is passed somewhere as a value and never
  called locally, or as any lambda body -- but never one holding ``qtbot``,
  which is a fixture nothing but a test can have to hand down. An
  immediately-invoked lambda is not simulated work either, and is reported.
* **A bounded wait on a real event.** ``event.wait(timeout=5.0)`` and
  ``proc.wait(timeout=5)`` return when the thing happens. Only ``qtbot.wait``,
  ``QTest.qWait`` and ``sleep`` burn their argument, so only those are pauses.
* **The body of a pause primitive.** ``process_events`` is a loop of
  ``qtbot.wait(10)``; reporting its one line instead of its call sites would
  name the wrong place, and would go quiet the moment somebody wrote a second
  wrapper. Any function whose body does nothing but pause is *derived* as a
  primitive, so its callers are what gets reported -- which is also why moving
  a pause behind a fresh helper cannot launder it.

Over-approximation is the safe direction: clock-tainted names are tracked per
module rather than per scope, which can only make a site reported that need not
have been, and never the reverse.

Cost: a file is parsed only once a pause or clock token appears in its text,
which is 112 of the suite's 313 modules, and every rule reads one pre-collected
:class:`Module` rather than reaching for ``ast.walk`` itself. 1.9-2.3 s on a box
with nine of sixteen cores busy, against 4.4 s for the interaction-timeout guard
that has lived in the suite all along. Both numbers are why this is a pytest
gate and not a commit hook like ``tools/dead_code.py``, whose whole-tree scan
costs seconds it would spend again on every run.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Functions returning a duration reading. ``time`` counts only as
#: ``time.time()``: the bare name is far more often a module or a parameter.
_CLOCK_READERS = frozenset(
    {
        "perf_counter",
        "perf_counter_ns",
        "monotonic",
        "monotonic_ns",
        "process_time",
        "thread_time",
        "default_timer",
    }
)

#: Calls that burn their argument no matter what the product does.
_SLEEP_NAMES = frozenset({"sleep", "qWait", "msleep"})

#: Text that has to appear in a file before it is worth parsing.
_SEED_TOKENS = frozenset(_SLEEP_NAMES | _CLOCK_READERS | {"wait", "time.time"})

#: The one settlement-deadline owner. A pause bounded by it is not bounded by a
#: throttle; it is a five-second sleep with a respectable import in front of it.
_BUDGET_OWNER = "arrayscope.tools.interaction_budget"

#: Roots publishing intervals a pause may legitimately be bounded by.
_PUBLISHER_PREFIXES = ("arrayscope", "pyqtgraph", "PySide6", "PyQt")

#: Casts, so that ``int(250)`` is still the literal 250.
_NUMERIC_CASTS = frozenset({"int", "float", "round"})


@dataclass(frozen=True)
class Finding:
    """One site, named the way a failure message has to name it."""

    path: str
    line: int
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.detail}"


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _root_name(node: ast.AST) -> str:
    while isinstance(node, (ast.Attribute, ast.Call, ast.Subscript)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else ""


def _receiver(node: ast.Call) -> str:
    return ast.unparse(node.func.value) if isinstance(node.func, ast.Attribute) else ""


def _is_clock_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = _call_name(node)
    if name in _CLOCK_READERS:
        return True
    return name == "time" and _receiver(node) == "time"


def _labels(target: ast.AST) -> set[str]:
    """What an assignment target binds.

    ``config.started = monotonic()`` binds ``config.started`` and emphatically
    not ``config``: taking the root would make every later ``x =
    config.getoption(...)`` a clock reading, and the guard would report the
    whole conftest.
    """

    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Attribute):
        return {ast.unparse(target)}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {label for element in target.elts for label in _labels(element)}
    return set()


def _named(nodes) -> set[str]:
    """Every name and dotted path in an already-walked expression."""

    found = set()
    for child in nodes:
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Attribute):
            found.add(ast.unparse(child))
    return found


def _references(node: ast.AST) -> set[str]:
    """Every name and dotted path this expression reads."""

    return _named(ast.walk(node))


@dataclass
class Module:
    """Everything the two rules need, collected in one walk of the tree.

    Assembled once per file because it used to be assembled eight times: each
    rule reaching for ``ast.walk`` on its own turned a 0.3 s scan into a 12 s
    one, which is the difference between a pytest gate and a commit hook.
    """

    path: str
    tree: ast.AST
    functions: list[ast.FunctionDef]
    #: ``(bound labels, value, names the value reads, does it read a clock)``
    assignments: list[tuple[set[str], ast.expr, set[str], bool]]
    imports: list[tuple[str, str, str]]  #: ``(source module, original, bound)``
    called: set[str]
    valued: set[str]
    reads_a_clock: bool

    def imported_from(self, prefixes: tuple[str, ...]) -> set[str]:
        return {
            bound
            for source, _, bound in self.imports
            if source.split(".")[0] in prefixes or source in prefixes
        }

    def _spread(self, seeded: set[str]) -> set[str]:
        tainted = set(seeded)
        changed = True
        while changed:
            changed = False
            for labels, _, references, _ in self.assignments:
                if labels <= tainted or not (references & tainted):
                    continue
                tainted |= labels
                changed = True
        return tainted

    def clock_names(self) -> set[str]:
        """Names and dotted paths holding a clock reading, or one derived."""

        direct = {
            label
            for labels, _, _, reads_clock in self.assignments
            if reads_clock
            for label in labels
        }
        return self._spread(direct)

    def budget_names(self) -> set[str]:
        """Names for the 5 s settlement deadline, which bounds no throttle."""

        return self._spread(self.imported_from((_BUDGET_OWNER,)))

    def published_names(self) -> set[str]:
        """Names for an interval the product or the toolkit publishes.

        The settlement-budget owner is excluded: see the module docstring. A
        module constant counts only when its own value reaches one of these
        imports, so ``INTERVAL = 40`` stays the literal it is.
        """

        return self._spread(self.imported_from(_PUBLISHER_PREFIXES)) - self.budget_names()


def read_module(path: str, tree: ast.AST) -> Module:
    """One walk, every fact."""

    functions: list[ast.FunctionDef] = []
    assignments: list[tuple[set[str], ast.expr, set[str], bool]] = []
    imports: list[tuple[str, str, str]] = []
    called: set[str] = set()
    valued: set[str] = set()
    reads_a_clock = False
    for node in ast.walk(tree):
        if _is_clock_call(node):
            reads_a_clock = True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            labels = {label for target in targets for label in _labels(target)}
            if labels:
                children = list(ast.walk(node.value))
                assignments.append(
                    (
                        labels,
                        node.value,
                        _named(children),
                        any(_is_clock_call(child) for child in children),
                    )
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.extend(
                (node.module, alias.name, alias.asname or alias.name) for alias in node.names
            )
        elif isinstance(node, ast.Import):
            imports.extend(
                (alias.name, alias.name, alias.asname or alias.name.split(".")[0])
                for alias in node.names
            )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
        elif isinstance(node, ast.Name):
            valued.add(node.id)
    return Module(
        path,
        tree,
        functions,
        assignments,
        imports,
        called,
        # Every name read anywhere, minus the ones read in order to call them:
        # what is left is the set of names handed somewhere as a value.
        valued - called,
        reads_a_clock,
    )


def _is_typed_number(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float))
    if isinstance(node, ast.UnaryOp):
        return _is_typed_number(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_typed_number(node.left) and _is_typed_number(node.right)
    if isinstance(node, ast.Call) and _call_name(node) in _NUMERIC_CASTS:
        return all(_is_typed_number(argument) for argument in node.args)
    return False


def _reads_a_published_duration(node: ast.Call, published: set[str], budgets: set[str]) -> bool:
    """Did this pause ask something for its duration, or was it typed?

    Only the arguments are examined. ``qtbot.wait`` is itself an attribute
    access, so judging the whole call would make every pause in the suite
    look like it had consulted the product.
    """

    arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
    for argument in arguments:
        if _is_typed_number(argument):
            continue
        for child in ast.walk(argument):
            if isinstance(child, ast.Attribute):
                if _root_name(child) not in budgets and ast.unparse(child) not in budgets:
                    return True
            elif isinstance(child, ast.Call):
                if _call_name(child) not in _NUMERIC_CASTS and _root_name(child) not in budgets:
                    return True
            elif isinstance(child, ast.Name) and child.id in published:
                return True
    return False


def _is_seed_pause_call(node: ast.Call) -> bool:
    name = _call_name(node)
    if name in _SLEEP_NAMES:
        return True
    return name == "wait" and _receiver(node).endswith("qtbot")


def _pause_call_name(node: ast.Call, primitives: set[str]) -> str:
    """The primitive this call burns time through, or ``""``."""

    if _is_seed_pause_call(node):
        return _call_name(node)
    name = _call_name(node)
    return name if name in primitives else ""


def _only_clock_calls(node: ast.AST) -> bool:
    """True when evaluating this asks nothing of the product but the time."""

    return all(_is_clock_call(child) for child in ast.walk(node) if isinstance(child, ast.Call))


def _body_only_pauses(body: list[ast.stmt], primitives: set[str]) -> bool:
    """True when every statement is a pause, a loop of pauses, or nothing."""

    found = False
    for statement in body:
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.Return) and statement.value is None:
            continue
        if isinstance(statement, ast.Expr):
            if isinstance(statement.value, ast.Constant):  # a docstring
                continue
            if isinstance(statement.value, ast.Call) and _pause_call_name(
                statement.value, primitives
            ):
                found = True
                continue
            return False
        if isinstance(statement, (ast.For, ast.While)):
            if statement.orelse or not _body_only_pauses(statement.body, primitives):
                return False
            found = True
            continue
        if isinstance(statement, ast.Assign) and _only_clock_calls(statement.value):
            continue  # the deadline computed for the loop below it
        return False
    return found


def pause_primitives(functions: list[ast.FunctionDef], imported: set[str]) -> set[str]:
    """Functions whose body does nothing but pause, plus the imported ones.

    Derived rather than listed, so hiding a pause behind a fresh helper makes
    the helper a primitive and reports its *callers* -- the place that chose to
    pause.

    A test is never one, whatever its body looks like. A test that does nothing
    but pause is the finding, not a wrapper for one, and without this exclusion
    it exempts itself: ``def test_thing(qtbot): qtbot.wait(250)`` was silently
    accepted until the proofs below asked for it by name.
    """

    primitives = set(imported)
    changed = True
    while changed:
        changed = False
        for function in functions:
            if function.name in primitives or function.name.startswith("test_"):
                continue
            if _body_only_pauses(function.body, primitives):
                primitives.add(function.name)
                changed = True
    return primitives


def workload_functions(module: Module) -> set[str]:
    """Functions handed to the product as a value rather than called here.

    A monkeypatched loader, a ``TaskSpec`` payload, a fake renderer: a sleep in
    one of those makes the *product* slow, which is the test's material and not
    its wait.

    Taking ``qtbot`` disqualifies a function whatever else it looks like. The
    bot is a pytest fixture, so nothing but a test can have one to hand down;
    a callable the product calls is never given it. Without that, the exemption
    is reachable on purpose -- define the pause in a helper, name the helper
    once as a value so it is never seen being called, and the sleep inside it
    stops being reported.
    """

    return {
        function.name
        for function in module.functions
        if function.name in module.valued and "qtbot" not in _parameters(function)
    }


def _parameters(function: ast.FunctionDef) -> set[str]:
    arguments = function.args
    named = {
        argument.arg
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    }
    for extra in (arguments.vararg, arguments.kwarg):
        if extra is not None:
            named.add(extra.arg)
    return named


def _own_body(node: ast.AST):
    """Every node under ``node`` that a nested callable does not own.

    A ``return`` belonging to a function defined inside the loop says nothing
    about whether the loop can end.
    """

    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield child
        yield from _own_body(child)


def _only_clock_comparisons(test: ast.expr, clock_names: set[str]) -> bool:
    if isinstance(test, ast.BoolOp):
        return all(_only_clock_comparisons(value, clock_names) for value in test.values)
    if not isinstance(test, ast.Compare):
        return False
    return any(
        _references(side) & clock_names or any(_is_clock_call(c) for c in ast.walk(side))
        for side in (test.left, *test.comparators)
    )


def _loop_can_end_on_the_product(loop: ast.For | ast.While, clock_names: set[str]) -> bool:
    """Can this loop stop for a reason other than its own clock?

    That is the whole difference between a poll cadence and a fixed pause with
    a loop drawn around it.
    """

    if any(isinstance(node, (ast.Break, ast.Return)) for node in _own_body(loop)):
        return True
    if isinstance(loop, ast.For):
        return False
    return not _only_clock_comparisons(loop.test, clock_names)


class _PauseVisitor(ast.NodeVisitor):
    """Walks a module carrying the context each pause site is judged in."""

    def __init__(self, module: Module, primitives: set[str]):
        self.path = module.path
        self.primitives = primitives
        self.workloads = workload_functions(module)
        self.published = module.published_names()
        self.budgets = module.budget_names()
        self.clock_names = module.clock_names()
        self.findings: list[Finding] = []
        self._names: list[str] = []
        self._loops: list[ast.For | ast.While | None] = []
        self._lambdas = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._names.append(node.name)
        self._loops.append(None)  # a loop does not reach across a function boundary
        self.generic_visit(node)
        self._loops.pop()
        self._names.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._lambdas += 1
        self._loops.append(None)
        self.generic_visit(node)
        self._loops.pop()
        self._lambdas -= 1

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_loop(node)

    def _visit_loop(self, node: ast.For | ast.While) -> None:
        self._loops.append(node)
        self.generic_visit(node)
        self._loops.pop()

    def _enclosing_loops(self) -> list[ast.For | ast.While]:
        enclosing = []
        for entry in reversed(self._loops):
            if entry is None:
                break
            enclosing.append(entry)
        return enclosing

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Lambda):
            # Immediately invoked: this is the test's own path whatever it looks
            # like, so the workload exemption must not reach inside it.
            for argument in (*node.args, *(k.value for k in node.keywords)):
                self.visit(argument)
            lambdas, self._lambdas = self._lambdas, 0
            self.visit(node.func.body)
            self._lambdas = lambdas
            return
        self.generic_visit(node)
        if not _pause_call_name(node, self.primitives):
            return
        if any(name in self.primitives for name in self._names):
            return
        if self._lambdas or any(name in self.workloads for name in self._names):
            return
        if any(
            _loop_can_end_on_the_product(loop, self.clock_names) for loop in self._enclosing_loops()
        ):
            return
        if _reads_a_published_duration(node, self.published, self.budgets):
            return
        self.findings.append(
            Finding(
                path=self.path,
                line=node.lineno,
                kind="pause",
                detail=(
                    f"{ast.unparse(node)[:64]} pauses for a duration that was typed, "
                    "not read from the throttle it is bounded by"
                ),
            )
        )


def unjustified_pauses(module: Module, primitives: set[str]) -> list[Finding]:
    """Fixed pauses in this module whose duration names no throttle."""

    visitor = _PauseVisitor(module, primitives)
    visitor.visit(module.tree)
    return visitor.findings


def _body_fails(body: list[ast.stmt]) -> bool:
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Raise):
                return True
            if isinstance(node, ast.Call) and _call_name(node) in {"fail", "xfail"}:
                return True
    return False


def clock_backed_assertions(module: Module) -> list[Finding]:
    """Every place a duration reading reaches a verdict.

    Both forms: the ``assert``, and the ``if elapsed > ...: pytest.fail()`` an
    assert-only rule would step straight over. A deadline check bounded by the
    interaction-budget owner is the sanctioned exception and the only one -- an
    ``assert`` on a duration is a stopwatch however it is bounded.
    """

    if not module.reads_a_clock:
        return []  # nothing in the file can carry a duration, so nothing can adjudicate one
    path = module.path
    clock_names = module.clock_names()
    budgets = module.budget_names()
    findings = []
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Assert):
            if _reads_a_clock(node.test, clock_names):
                findings.append(
                    Finding(
                        path=path,
                        line=node.lineno,
                        kind="stopwatch",
                        detail=(
                            f"assert {ast.unparse(node.test)[:64]} adjudicates a clock "
                            "reading, and pytest is not where a duration is adjudicated"
                        ),
                    )
                )
            continue
        if not isinstance(node, ast.If):
            continue
        if not _reads_a_clock(node.test, clock_names):
            continue
        if _references(node.test) & budgets:
            continue
        if not _body_fails(node.body):
            continue
        findings.append(
            Finding(
                path=path,
                line=node.lineno,
                kind="stopwatch",
                detail=(
                    f"if {ast.unparse(node.test)[:56]} ... fails on a clock reading, "
                    "which is an assert on a duration with the assert spelled out"
                ),
            )
        )
    return findings


def _reads_a_clock(node: ast.AST, clock_names: set[str]) -> bool:
    if _references(node) & clock_names:
        return True
    return any(_is_clock_call(child) for child in ast.walk(node))


def _module_name(relative: Path) -> str:
    return ".".join(relative.with_suffix("").parts)


def _imports_of(module: Module, exported: dict[str, set[str]]) -> set[str]:
    return {
        bound for source, original, bound in module.imports if original in exported.get(source, ())
    }


def findings(root: Path = REPO_ROOT) -> list[Finding]:
    """Both classes, over every module under ``tests/`` that can hold one.

    A file is parsed only once a pause or clock token appears in its text, and
    the token set grows as derived primitives are discovered, so a wrapper
    imported under a new name still pulls its callers into the scan.
    """

    sources = {
        path.relative_to(root): path.read_text() for path in sorted((root / "tests").rglob("*.py"))
    }
    tokens = set(_SEED_TOKENS)
    modules: dict[Path, Module] = {}
    primitives: dict[str, set[str]] = {}
    growing = True
    while growing:
        growing = False
        for relative, text in sources.items():
            if relative not in modules and any(token in text for token in tokens):
                modules[relative] = read_module(str(relative), ast.parse(text))
        for relative, module in modules.items():
            name = _module_name(relative)
            found = pause_primitives(module.functions, _imports_of(module, primitives))
            if found != primitives.get(name):
                primitives[name] = found
                growing = growing or bool(found - tokens)
                tokens |= found

    collected: list[Finding] = []
    for relative, module in modules.items():
        collected.extend(clock_backed_assertions(module))
        collected.extend(unjustified_pauses(module, primitives[_module_name(relative)]))
    return sorted(collected, key=lambda finding: (finding.path, finding.line))


def scan_source(text: str, path: str = "tests/ui/test_probe.py") -> list[Finding]:
    """Both rules over one module of source, for the falsifiability proofs."""

    module = read_module(path, ast.parse(text))
    primitives = pause_primitives(module.functions, set())
    return sorted(
        clock_backed_assertions(module) + unjustified_pauses(module, primitives),
        key=lambda finding: finding.line,
    )


def report(collected: list[Finding]) -> str:
    """The findings as a failure message: every one of them by path and line."""

    by_kind: dict[str, list[Finding]] = {}
    for finding in collected:
        by_kind.setdefault(finding.kind, []).append(finding)
    lines = []
    for kind in sorted(by_kind):
        lines.append(f"{len(by_kind[kind])} {kind}:")
        lines.extend(f"  {finding}" for finding in by_kind[kind])
    return "\n".join(lines)
