"""Guards for the gates that refuse a *run shape* (``tests/conftest.py``).

Ring 0 — no Qt, no rendering. The sibling file
``tests/app/test_test_selection.py`` guards which tests selection picks; this
one guards the commands the harness declines to obey at all, and the wording it
declines them with.

Both halves matter here, and the second one is not decoration. Every refusal in
this file exists because a specific command was typed by somebody who believed
the harness was broken, and the refusal text is the only thing guaranteed to be
read at that moment. A gate that fires with an unhelpful sentence teaches the
next, wider sweep.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests import testmon_policy

REPO_ROOT = testmon_policy.REPO_ROOT


def _config(args=(), **options):
    return SimpleNamespace(option=SimpleNamespace(**options), args=list(args))


# --------------------------------------------------------------------------- #
# -p no:X — the disables that crash or silently defeat the harness
# --------------------------------------------------------------------------- #


def test_a_disable_that_crashes_the_run_is_refused_before_it_can():
    """`-p no:cacheprovider` is an INTERNALERROR, not a preference.

    testmon subscripts ``config.option['lf']``, which only exists while pytest's
    cache plugin is registered, so blocking it kills the run during
    ``pytest_configure`` with a 40-line pluggy traceback and exit 3 — reliably,
    100% of the time. That traceback reads as "this repository is broken", which
    is exactly the belief that produces the next, wider sweep. The gate has to
    intercept it before testmon, and say what it is.
    """

    from tests.conftest import harmful_plugin_disables

    present = lambda module: True  # noqa: E731 - one-expression stub reads better inline

    ((flag, what, instead),) = harmful_plugin_disables(["no:cacheprovider"], present)
    assert flag == "no:cacheprovider"
    assert "crashes the run" in what
    assert "'lf'" in what, "the message must name the mechanism, not just the verdict"
    # The counter-intuitive half: here the cached state is the feature.
    assert ".testmondata" in instead


def test_the_disable_that_silently_turns_selection_off_is_refused():
    """`-p no:pytest-testmon` is the worst hole, because nothing looks wrong.

    It disables selection *and* recording while the header keeps printing
    ``test selection: on``. Priced as a bypass it is refused anyway (see the
    fail-closed gate below); this entry exists so the message names the habit
    rather than quoting a number.
    """

    from tests.conftest import harmful_plugin_disables

    present = lambda module: True  # noqa: E731

    ((flag, what, _instead),) = harmful_plugin_disables(["no:pytest-testmon"], present)
    assert flag == "no:pytest-testmon"
    assert "recording" in what


def test_the_harmful_table_fires_on_the_opposite_predicate_to_the_pointless_one():
    """Inert is refused while absent; harmful is refused while present.

    ``-p no:randomly`` is refused because the plugin is *not* installed, so the
    flag makes a run look controlled when it changed nothing. ``-p
    no:pytest-testmon`` is the mirror image: it is refused precisely when the
    disable would take effect, because the effect is the damage. Getting the two
    predicates the same way round would make each table fire for the runs the
    other one is about.
    """

    from tests.conftest import harmful_plugin_disables, pointless_plugin_disables

    absent = lambda name: False  # noqa: E731
    present = lambda name: True  # noqa: E731

    assert pointless_plugin_disables(["no:randomly"], absent)
    assert pointless_plugin_disables(["no:randomly"], present) == []
    assert harmful_plugin_disables(["no:pytest-testmon"], present)
    assert harmful_plugin_disables(["no:pytest-testmon"], absent) == []
    # A pytest builtin has no module to probe: it is always there to be broken.
    assert harmful_plugin_disables(["no:cacheprovider"], absent)


def test_the_self_punishing_disables_are_left_alone():
    """Only the silent ones need a gate.

    ``-p no:testmon`` misses (the entry point is ``pytest-testmon``) and does
    nothing; ``-p no:xdist`` makes ``addopts`` a usage error; ``-p no:mark``
    is refused by pytest itself. Each answers instantly and unambiguously, so
    adding them here would only be a list to keep in sync.
    """

    from tests.conftest import harmful_plugin_disables, pointless_plugin_disables

    present = lambda name: True  # noqa: E731
    for flag in ("no:testmon", "no:xdist", "no:mark", "no:python", "no:qt", "no:cov"):
        assert harmful_plugin_disables([flag], present) == [], flag
        assert pointless_plugin_disables([flag], present) == [], flag


# --------------------------------------------------------------------------- #
# The bypass gate's own reading of its inputs
# --------------------------------------------------------------------------- #


def test_a_coverage_run_is_permitted_under_the_dest_pytest_cov_actually_uses(monkeypatch):
    """``--cov`` registers ``cov_source``; there is no ``cov`` option at all.

    ``testmon_policy.decide`` reads the right key and turns selection off, so
    the run is classified as a bypass — and then the permission check read
    ``options["cov"]``, found nothing, and priced it. A documented, legitimate
    command was therefore refused: measured, ``pytest --cov <scope>`` exits 4.
    The two readings have to be the same reading.
    """

    from tests.conftest import _bypass_permitted

    monkeypatch.delenv("CI", raising=False)
    assert not testmon_policy.decide(_config(cov_source=["arrayscope"])).active
    assert "coverage" in _bypass_permitted(_config(cov_source=["arrayscope"]))
    assert "coverage" in _bypass_permitted(_config(cov_report={"term": None}))
    assert _bypass_permitted(_config()) == ""


def test_the_permission_check_and_the_policy_agree_on_every_key_they_share():
    """A dest read in one place and not the other is how bug-by-typo returns.

    The classification (``decide``) and the permission (``_bypass_permitted``)
    are two separate readings of the same option namespace, and they disagreed
    for as long as one of them spelled it ``cov``. Pinned against the source so
    that renaming the key in one place fails here rather than in a terminal.
    """

    import inspect

    from tests import conftest

    permitted = inspect.getsource(conftest._bypass_permitted)
    policy = inspect.getsource(testmon_policy.decide)
    for key in ("cov_source", "cov_report"):
        assert key in permitted, f"{key} is read by decide() and must be read here too"
        assert key in policy


def test_ci_and_the_override_are_still_permitted_at_any_price(monkeypatch):
    from tests.conftest import BYPASS_FLAG, _bypass_permitted

    monkeypatch.setenv("CI", "1")
    assert "CI" in _bypass_permitted(_config())
    monkeypatch.delenv("CI", raising=False)
    forced = {BYPASS_FLAG.lstrip("-").replace("-", "_"): True}
    assert _bypass_permitted(_config(**forced)) == "explicitly forced"


# --------------------------------------------------------------------------- #
# Fail closed: an unpriceable bypass is a large one
# --------------------------------------------------------------------------- #


def test_a_bypass_the_map_cannot_price_is_refused_rather_than_priced_at_zero():
    """Unknown cost is assumed large — for a bypass, and only for a bypass.

    The price gate reads a missing map as 0 s, and 0 s opens it. Four separate
    routes land there and every one of them was a working bypass: a fresh clone,
    a worktree that has not run yet, ``TESTMON_DATAFILE`` pointing at nothing
    (measured — price 0, allowed), and a map recorded under another environment
    key. Each turns the gate into a no-op at exactly the moment there is no
    evidence to defend it with.
    """

    from tests.conftest import unpriceable_bypass_refusal

    assert "cannot be priced" in unpriceable_bypass_refusal(0, "", node_ids_only=False)
    assert "cannot be priced" in unpriceable_bypass_refusal(None, "", node_ids_only=False)
    # A map that holds something is priced by the ordinary rule instead.
    assert unpriceable_bypass_refusal(3941, "", node_ids_only=False) == ""
    # Legitimate exhaustive contexts are never refused, here either.
    assert unpriceable_bypass_refusal(0, "CI, where every job is a gate", node_ids_only=False) == ""


def test_named_tests_stay_runnable_with_no_map_at_all():
    """The exemption that keeps the fail-closed rule from being a wall.

    ``pytest --no-testmon file.py::test`` cannot run more than it names,
    whatever the map does or does not hold — so it needs no price to be safe,
    and refusing it would take the one question ``--no-testmon`` uniquely
    answers away from a fresh checkout.
    """

    from tests.conftest import node_ids_only, unpriceable_bypass_refusal

    assert unpriceable_bypass_refusal(0, "", node_ids_only=True) == ""
    assert node_ids_only(["tests/ui/test_a.py::test_one"])
    assert node_ids_only(["tests/ui/test_a.py::test_one", "tests/ui/test_b.py::test_two"])
    assert not node_ids_only(["tests/ui/test_a.py"])
    assert not node_ids_only(["tests"])
    assert not node_ids_only([])  # a bare run names nothing, which is the widest shape


# --------------------------------------------------------------------------- #
# The canary: a run may only claim selection is on if it can prove it
# --------------------------------------------------------------------------- #


def test_blocking_the_plugin_turns_selection_off_instead_of_faking_it():
    """`-p no:pytest-testmon` must reach the price gate, not the header.

    ``decide`` asked ``find_spec("testmon")``, which answers "the package is
    installed" — a different question from "the plugin is registered for this
    run". Blocking the entry point left the package importable, so selection
    reported itself active while nothing selected and nothing recorded. The
    check has to be about the pluginmanager, which is the thing that would
    actually have done the work.
    """

    registered = SimpleNamespace(pluginmanager=SimpleNamespace(hasplugin=lambda name: True))
    blocked = SimpleNamespace(pluginmanager=SimpleNamespace(hasplugin=lambda name: False))

    assert testmon_policy.testmon_installed(registered)
    assert not testmon_policy.testmon_installed(blocked)
    # No config to ask: fall back to the package, which is what the callers
    # outside a pytest run (tools/test_selection.py, peek) have to use.
    assert testmon_policy.testmon_installed() == testmon_policy.testmon_installed(None)

    blocked_config = SimpleNamespace(
        option=SimpleNamespace(), args=[], pluginmanager=blocked.pluginmanager
    )
    decision = testmon_policy.decide(blocked_config)
    assert not decision.active
    assert "registered" in decision.reason


def test_a_run_only_claims_selection_when_the_plugin_actually_materialised():
    """The single invariant behind the fail-open holes.

    Every degradation path in this layer fails *open* — an unreadable map, a
    renamed testmon attribute, a blocked entry point — and each one leaves the
    header printing ``test selection: on``. So the header asks for proof: the
    plugin is registered, and it built the config object it was supposed to.
    """

    from tests.conftest import selection_materialised

    manager = SimpleNamespace(hasplugin=lambda name: True)
    assert selection_materialised(
        SimpleNamespace(pluginmanager=manager, testmon_config=SimpleNamespace(select=True))
    )
    assert not selection_materialised(SimpleNamespace(pluginmanager=manager, testmon_config=None))
    assert not selection_materialised(
        SimpleNamespace(
            pluginmanager=SimpleNamespace(hasplugin=lambda name: False),
            testmon_config=SimpleNamespace(select=True),
        )
    )


def test_the_map_is_seeded_before_anything_is_priced_against_it():
    """Order, because the fail-closed rule is only as good as the map it reads.

    Seeding used to happen *after* the bypass gate and only when
    ``decision.active`` — and a bypass run is never active, so a fresh worktree
    had no map at the one moment a price mattered. Every bypass in it was
    therefore priced at zero and allowed. Seeding is a file copy from a sibling
    checkout; there is no reason for the gate to run ahead of it.

    Asserted against the source because the defect is a *sequence*, and the
    only other way to observe it is a full pytest run in a fresh checkout.
    """

    import inspect

    from tests import conftest

    body = inspect.getsource(conftest.pytest_configure)
    assert body.index("_ensure_map_seeded(config)") < body.index("_refuse_unselected_run(config)")
    seeding = inspect.getsource(conftest._ensure_map_seeded)
    assert "decision" not in seeding, "seeding must not depend on selection being active"


# --------------------------------------------------------------------------- #
# `-n 0`: serial is for one thing at a time
# --------------------------------------------------------------------------- #


def test_only_a_typed_dash_n_zero_trips_the_serial_gate():
    """The harness sizes itself to zero constantly; that must never be caught.

    ``pytest_xdist_auto_num_workers`` returns 0 for any selection under 2.5 s or
    in one file, and xdist writes that straight into
    ``config.option.numprocesses`` before ``pytest_configure`` — so by the time
    any gate can look, the option cannot tell a person's ``-n 0`` from the
    harness's own. The command line can, and that is the only place it can be
    read from.
    """

    from tests.conftest import explicit_worker_count

    assert explicit_worker_count(["-n", "0", "tests/ui"]) == 0
    assert explicit_worker_count(["-n0"]) == 0
    assert explicit_worker_count(["--numprocesses=0"]) == 0
    assert explicit_worker_count(["--numprocesses", "0"]) == 0
    assert explicit_worker_count(["-n", "16"]) == 16
    # `auto` is the harness deciding, not a person, and so is an empty command
    # line -- `addopts` carries `-n auto` for every run in this repository.
    assert explicit_worker_count(["-n", "auto"]) is None
    assert explicit_worker_count(["-q", "tests/ui"]) is None
    assert explicit_worker_count([]) is None


def test_a_wide_serial_run_is_refused_and_a_named_one_is_not():
    """Priced on what the run will execute, with 60 s in an empty 3.8x gap.

    The threshold is calibrated rather than chosen: the most expensive command
    any ring documents as serial is the ring-2 artifact file at 27.5 s, and the
    cheapest shape that has ever hurt anybody is `tests/app` at 105.9 s
    (`tests/ui` is 1164.8 s, the whole suite 1320 s). Nothing lives in between.
    """

    from tests.conftest import serial_run_refusal

    assert "one process" in serial_run_refusal(1165.0, True, whole_suite=True, permitted_because="")
    assert serial_run_refusal(27.5, True, whole_suite=False, permitted_because="") == ""
    assert serial_run_refusal(5.6, True, whole_suite=False, permitted_because="") == ""


def test_an_unpriceable_serial_run_is_refused_only_for_the_whole_suite():
    """The safe direction is inverted here, and getting it wrong breaks ring 4.

    Rings 3 and 4 have **zero** recorded seconds by construction: the map is
    keyed by regime, and it has never held `wayland+gpu` or `offscreen+stress`.
    They are also precisely the runs that must be serial. So "unknown cost =
    assume large", which is right for a bypass, would refuse the two rings
    ground rule 1 depends on. Scope is the backstop instead: an unpriceable run
    is refused only when it names the whole suite, which is the fresh-clone
    `pytest -n 0` — 1320 s of first-run tracing in one process.
    """

    from tests.conftest import serial_run_refusal

    assert serial_run_refusal(0.0, False, whole_suite=True, permitted_because="")
    assert serial_run_refusal(0.0, False, whole_suite=False, permitted_because="") == ""
    assert serial_run_refusal(0.0, True, whole_suite=True, permitted_because="") == ""


def test_everything_that_legitimately_needs_serial_is_permitted(monkeypatch):
    """Every documented serial use, checked against the gate that could break it.

    Listed from `docs/testing/README.md` rings 0-4 plus the debugging shapes.
    `-s` is deliberately absent: "I want to see prints" is a reflex, not a
    reason to spend ten minutes.
    """

    from tests.conftest import _serial_permitted

    monkeypatch.delenv("CI", raising=False)
    for name in ("ARRAYSCOPE_GPU_TESTS", "ARRAYSCOPE_STRESS", "ARRAYSCOPE_STRICT_UI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    def config(arguments=(), **options):
        return SimpleNamespace(
            option=SimpleNamespace(**options),
            args=list(arguments),
            rootdir=str(REPO_ROOT),
            invocation_params=SimpleNamespace(dir=str(REPO_ROOT)),
            getini=lambda name: ["tests"] if name == "testpaths" else None,
        )

    assert _serial_permitted(config(["tests/ui"])) == "", "a wide serial run has no exemption"

    assert "debugger" in _serial_permitted(config(["tests/ui"], usepdb=True))
    assert "debugger" in _serial_permitted(config(["tests/ui"], trace=True))
    assert _serial_permitted(config(["tests/ui"], capture="no")) == "", "-s is not a reason"
    assert "collect" in _serial_permitted(config(["tests/ui"], collectonly=True))

    assert "serial by design" in _serial_permitted(config(["tests/gpu_interaction"]))
    assert "serial by design" in _serial_permitted(config(["tests/stress"]))
    assert "serial by design" in _serial_permitted(config(["tests/ui/test_qt_smoke_artifacts.py"]))
    # One ring target plus one ordinary directory is an ordinary wide run.
    assert _serial_permitted(config(["tests/stress", "tests/ui"])) == ""

    monkeypatch.setenv("ARRAYSCOPE_GPU_TESTS", "1")
    assert "offscreen" in _serial_permitted(config(["tests/ui"]))
    monkeypatch.delenv("ARRAYSCOPE_GPU_TESTS")
    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland")
    assert "offscreen" in _serial_permitted(config(["tests/ui"]))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    monkeypatch.setenv("CI", "1")
    assert "CI" in _serial_permitted(config(["tests/ui"]))
    monkeypatch.delenv("CI")

    from tests.conftest import SERIAL_FLAG

    forced = {SERIAL_FLAG.lstrip("-").replace("-", "_"): True}
    assert _serial_permitted(config(["tests/ui"], **forced)) == "explicitly forced"


def test_forcing_one_gate_does_not_force_the_other():
    """Two overrides on purpose: they are claims about different things.

    "I need the tracer out of the way" and "I need this in one process" are
    unrelated, and one flag that granted both would make each exemption
    reachable by somebody who only meant the other.
    """

    from tests.conftest import BYPASS_FLAG, SERIAL_FLAG

    assert BYPASS_FLAG != SERIAL_FLAG
    assert SERIAL_FLAG == "--serial-force-please-i-really-consider-it-needed"


def test_an_oversized_worker_count_is_capped_rather_than_refused():
    """`-n 16` silently defeated the GL-stability cap; the intent is still benign.

    ``pytest_xdist_auto_num_workers`` owns every worker-count decision and is
    only consulted for `-n auto`, so an explicit count walked straight past the
    half-cores cap that keeps concurrent GL context creation from segfaulting
    the driver. "More parallel" is not a mistake worth refusing, so it is
    clamped and said out loud.
    """

    from tests.conftest import worker_ceiling

    assert worker_ceiling(16) == 8
    assert worker_ceiling(4) == 2
    assert worker_ceiling(2) == 2, "a 2-core CI runner must keep its parallelism"
    assert worker_ceiling(1) == 2
