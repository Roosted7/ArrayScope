"""Ring 1. The two co-operation rules in ``tools/load_fragility``.

Both rules fail SILENTLY when they break, which is why they are pinned here
rather than left to the next person noticing an odd number.

The target clamp fails loudly enough to see; the loop-detection pattern does
not. If that pattern stops matching real loops, every instance believes another
one already owns the box and runs zero loops -- while still printing
``holding 15 runnable``. A run that generated no load at all then reads exactly
like a run that did, and any "it passed under load" conclusion drawn from it is
worthless. That is the failure this file exists to catch.

Nothing here spins a CPU, and nothing here starts the supervisor: a test for the
load tool must not itself become the load that poisons a concurrent measurement.
The decoys sleep, and the clamp is read through ``-p``, which resolves and
announces the plan and stops before any load exists.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "load_fragility"


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf8")


def _any_loop_pattern() -> str:
    """The pattern the supervisor uses to find OTHER instances' loops.

    Read out of the script rather than restated here.  A copy of the pattern
    would keep passing after the real one was changed, which is the whole
    failure this test is for.
    """

    match = re.search(r'any_pattern="\$\{marker\}-loop-(.+?)"', _script_text())
    assert match, "supervisor no longer defines any_pattern; update this test"
    # The script escapes `$` for the outer heredoc-ish quoting; undo that.
    return "arrayscope-load-fragility-loop-" + match.group(1).replace("\\$", "$")


def _spawn(argv0: str, *, body: str) -> subprocess.Popen:
    return subprocess.Popen(["bash", "-c", body, argv0], start_new_session=True)


def _pgrep_count(pattern: str) -> int:
    result = subprocess.run(["pgrep", "-fc", pattern], capture_output=True, text=True, check=False)
    return int(result.stdout.strip() or 0)


@pytest.mark.skipif(shutil.which("pgrep") is None, reason="pgrep is required")
def test_loop_detection_matches_real_loops_and_not_processes_that_merely_name_them():
    """A mention of the marker is not a loop.

    Observed while verifying the co-operative supervisor: an unanchored
    ``pgrep -fc arrayscope-load-fragility-loop`` reported 2 on a box whose
    /proc argv held zero real loops -- it had matched the agent harness's own
    command lines, which quoted the marker. Unanchored, that reads as "another
    instance owns the box" and this instance defers forever.
    """

    pattern = _any_loop_pattern()
    real = _spawn("arrayscope-load-fragility-loop-424242", body="while :; do sleep 0.2; done")
    decoy_untagged = _spawn(
        "arrayscope-load-fragility-loop-not-a-pid", body="while :; do sleep 0.2; done"
    )
    decoy_mention = subprocess.Popen(
        ["bash", "-c", "sleep 30 # arrayscope-load-fragility-loop-1"], start_new_session=True
    )
    try:
        time.sleep(1.0)
        assert _pgrep_count(pattern) == 1, (
            "the loop pattern must match exactly the one correctly-tagged loop; "
            "matching a decoy makes every instance defer and load nothing"
        )
        # And it must still find a real loop by its own tag, or cleanup cannot
        # reap a straggler whose process group died under it.
        assert _pgrep_count(r"arrayscope-load-fragility-loop-424242$") == 1
        assert _pgrep_count(r"arrayscope-load-fragility-loop-424243$") == 0
    finally:
        for proc in (real, decoy_untagged, decoy_mention):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)


def _plan(*argv: str) -> subprocess.CompletedProcess[str]:
    """Resolve the load plan without starting the load or running any test.

    The clamp is pure arithmetic over ``nproc`` and the flags, but reading it
    used to cost ~10 s per call: the script started the supervisor (a fixed 2 s
    sleep) and then booted a whole pytest on ``tests/_no_such_test.py`` purely
    so the run would fail fast and let the supervisor die. ``-p`` stops between
    the line that announces the resolved target and ``start_load``, so the
    assertions below read the same two printfs a real run emits, for ~20 ms.
    """

    return subprocess.run(
        [str(SCRIPT), "-p", *argv],
        capture_output=True,
        text=True,
        check=False,
        cwd=SCRIPT.parents[1],
        timeout=60,
    )


@pytest.mark.parametrize("requested", ["999", "48"])
def test_target_is_clamped_to_one_below_nproc(requested):
    """Asking for more runnable than the box has is rounded down, and said so.

    Past saturation an extra spinner takes a core from the test under
    observation without adding contention it can feel, and a target well above
    nproc silently imposes that regime on every other session on the machine.
    """

    cap = max(1, (os.cpu_count() or 2) - 1)
    result = _plan("-t", requested)
    assert f"target {requested} exceeds nproc-1; holding {cap} instead" in result.stdout
    assert f"holding {cap} runnable" in result.stdout


def test_a_target_under_the_cap_is_left_alone():
    cap = max(1, (os.cpu_count() or 2) - 1)
    if cap < 2:
        pytest.skip("needs at least two cores to have a value under the cap")
    result = _plan("-t", "2")
    assert "exceeds nproc-1" not in result.stdout
    assert "holding 2 runnable" in result.stdout


def test_the_plan_reports_the_real_path_and_stops_before_it_costs_anything():
    """``-p`` is a short-circuit of the real path, not a second implementation.

    Both facts the two tests above assert are printed by lines the expensive
    path runs as well; the flag only decides whether execution continues past
    them. So those tests stay honest exactly as long as the gate sits after
    both printfs and before ``start_load`` -- above the first printf it would
    report nothing, below ``start_load`` it would stop being cheap. Assert that
    ordering, and assert the stop itself: a ``-p`` that was accepted and then
    ignored leaves both tests above green and slow, which is the whole reason
    they were rewritten.
    """

    result = _plan("-t", "2")
    assert result.returncode == 0
    assert "plan only (-p): no load started, no tests run" in result.stdout
    assert "--- run " not in result.stdout, "-p must not reach the pytest invocation"

    text = _script_text()
    announced = text.index("load: holding %s runnable on %s cores")
    gate = text.index('if [ "$plan_only" -eq 1 ]; then', announced)
    started = text.index("\nstart_load\n", announced)
    assert announced < gate < started, (
        "the plan gate must sit between the announcement it reports and the "
        "first thing that costs; anywhere else it reports the wrong numbers "
        "or reports them just as slowly"
    )


def test_cleanup_is_scoped_to_this_instance():
    """One instance's exit must not cull another's loops.

    Now that a second instance legitimately defers to a first, the old
    belt-and-braces ``pkill -f arrayscope-load-fragility-loop`` would cut the
    owner's load out from under it on our way out -- and the owner would ramp
    straight back up. That oscillation is what the co-operative supervisor
    exists to prevent, so the reaping pattern must carry our own tag.
    """

    text = _script_text()
    reap = re.search(r"pkill -9 -f \"(.+?)\"", text)
    assert reap, "the cleanup trap no longer reaps stragglers; keep the trap"
    assert "${load_tag}" in reap.group(1), (
        "cleanup must reap only THIS instance's loops, or exiting kills a concurrent session's load"
    )


def _verdict_block() -> str:
    """The script's own verdict chain, lifted out of the script.

    Read rather than restated, for the reason :func:`_any_loop_pattern` gives:
    a second copy of this logic would keep passing after the real one changed,
    and this particular logic has already been wrong three different ways.
    """

    text = _script_text()
    start = text.index("  rc=0\n")
    end = text.index("\ndone\n", start)
    return text[start:end]


def _verdict(output: str, returncode: int) -> subprocess.CompletedProcess:
    """Run that chain against a canned pytest result.

    No supervisor, no pytest, no spinning core -- this file's opening rule. The
    only edit to the lifted text is the invocation itself, which is replaced by
    the canned pair; if that substitution stops matching, the test says so
    rather than silently asserting against an unedited block that would run a
    real suite.
    """

    block = _verdict_block()
    canned = 'out="$CANNED_OUT"\n  rc="$CANNED_RC"'
    patched, count = re.subn(r'out="\$\(.*?\)" \|\| rc=\$\?', canned, block)
    assert count == 1, "the pytest invocation in the verdict block moved; update this test"
    script = (
        "set -euo pipefail\n"
        "declare -a results\n"
        "status=0\n"
        "run=1\n"
        f"{patched}\n"
        'printf "%s\\n" "${results[@]}"\n'
        "exit $status\n"
    )
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "CANNED_OUT": output, "CANNED_RC": str(returncode)},
        timeout=60,
    )


def test_a_clean_run_is_verdicted_and_does_not_kill_the_script():
    """The good case must reach a verdict, and it used not to.

    ``skipped=$(... | grep -oE '[0-9]+ skipped' | ...)`` finds nothing when
    nothing skipped, and under the script's own ``set -euo pipefail`` that
    ended the run right there -- at repeat 1, with no summary and a non-zero
    exit. So the tool for re-checking fragile tests under load reported failure
    on precisely the runs where every target survived.
    """

    result = _verdict("22 passed in 32.16s", 0)
    assert result.returncode == 0
    assert "run 1: PASS" in result.stdout
    assert "SKIPPED" not in result.stdout


def test_a_failing_run_is_verdicted_rather_than_aborting_the_script():
    """A red is the observation, not an error in the observer.

    ``out="$(pytest ...)"`` under ``set -e`` aborted on a failing pytest before
    ``rc=$?`` was ever read, taking the remaining repeats and the summary with
    it -- the one outcome this script exists to catch was the one it could not
    report.
    """

    result = _verdict("1 failed, 21 passed in 34.02s", 1)
    assert result.returncode == 1
    assert "run 1: FAIL" in result.stdout


def test_a_skipped_target_is_not_read_as_a_survival():
    """A target that skipped did not face the load. Fixed in 57471d26; kept."""

    result = _verdict("20 passed, 2 skipped in 30.11s", 0)
    assert result.returncode == 1
    assert "2 SKIPPED" in result.stdout
    assert "did not face the load" in result.stdout


def test_a_target_that_matches_nothing_is_not_read_as_a_survival():
    """The quietest member of the family: pytest exits 0 for a missing nodeid.

    So a target renamed out from under the default list verdicts PASS forever,
    with no test having run at all -- the same false survival as the skip, and
    harder to notice because the count it would have shown is simply absent.
    """

    result = _verdict("no tests ran in 4.39s", 0)
    assert result.returncode == 1
    assert "NO TESTS RAN" in result.stdout
