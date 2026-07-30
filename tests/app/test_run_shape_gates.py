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
