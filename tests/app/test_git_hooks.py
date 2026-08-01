"""The git hooks recover their own environment, exactly once.

Ring 0 — no Qt, no imports of ``arrayscope``. The hooks are shell, and the
part worth pinning is the wiring rather than the gates: whether a hook that
cannot see ``ruff``/``python`` hands itself to ``tools/env`` and continues,
and whether the guard that makes that a *single* handoff really holds. A
recursion bug in a commit hook is not a thing to discover interactively.

So the real gates are never run here. Each test builds a throwaway repository
containing a stub ``tools/env`` and stub tools that only record that they were
called, which is what lets these stay instant — the real ``pre-commit`` costs
~2.1 s, and paying that per suite run is the same trade
``tests/app/test_dead_code.py`` already refused.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[2] / ".githooks"


def _stub(path: Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755)


def _repo(tmp_path: Path, hook: str, *, env_supplies_tools: bool) -> Path:
    """A repository whose only tools are the ones ``tools/env`` hands out."""

    root = tmp_path / "repo"
    (root / ".githooks").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "envbin").mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)

    shutil.copy(HOOKS / hook, root / ".githooks" / hook)
    (root / ".githooks" / hook).chmod(0o755)

    # The stub environment: prepend a bin directory, then exec, which is the
    # only part of `tools/env` this wiring depends on.
    _stub(
        root / "tools" / "env",
        f'export PATH="{root}/envbin:$PATH"\nexec "$@"',
    )
    if env_supplies_tools:
        for tool in ("ruff", "python"):
            _stub(root / "envbin" / tool, f'echo "{tool} $*" >> "{root}/calls"\nexit 0')

    # Reached only if the hook gets as far as its real work.
    (root / "tools" / "dead_code.py").write_text("")
    return root


def _run(root: Path, hook: str) -> subprocess.CompletedProcess[str]:
    """Run the hook with a PATH holding git and nothing else it needs.

    Building the PATH rather than trimming the caller's is what keeps this
    honest on a machine where ``ruff`` happens to be installed globally.
    """

    hostbin = root / "hostbin"
    hostbin.mkdir(exist_ok=True)
    # `bash` because the hooks are bash and `env` resolves it through PATH;
    # `git` because a hook's first act is asking git where it is. Neither is
    # a tool the hook is meant to recover, so both are always present.
    for name in ("bash", "git"):
        found = shutil.which(name)
        assert found, f"{name} is required to run a git hook test"
        (hostbin / name).symlink_to(found)

    return subprocess.run(
        [str(root / ".githooks" / hook)],
        cwd=root,
        env={"PATH": str(hostbin), "HOME": str(root)},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_pre_commit_recovers_its_tools_through_tools_env(tmp_path):
    """The retry the old hook asked the caller to perform, performed."""

    root = _repo(tmp_path, "pre-commit", env_supplies_tools=True)

    result = _run(root, "pre-commit")

    assert result.returncode == 0, result.stderr
    calls = (root / "calls").read_text().splitlines()
    assert calls == ["ruff check .", "ruff format --check .", f"python {root}/tools/dead_code.py"]


def test_pre_commit_hands_off_once_and_then_reports_the_missing_tool(tmp_path):
    """The guard, which is the whole reason the handoff is safe.

    An environment that does not supply ``ruff`` must produce one report, not
    a hook that re-enters itself until something else stops it. The timeout in
    ``_run`` is the backstop; the assertion is that the handoff happened once.
    """

    root = _repo(tmp_path, "pre-commit", env_supplies_tools=False)

    result = _run(root, "pre-commit")

    assert result.returncode == 1
    assert "ruff is not on PATH" in result.stderr
    assert "uv sync" in result.stderr
    assert "--no-verify" in result.stderr
    # The second pass knows it is the second one, and says so rather than
    # repeating the advice that has already been taken.
    assert "does not have it either" in result.stderr
    assert "tools/env" not in result.stderr


def test_pre_commit_without_tools_env_still_refuses_rather_than_skipping(tmp_path):
    """Missing is a hard failure, never a skip — recovery does not soften it."""

    root = _repo(tmp_path, "pre-commit", env_supplies_tools=False)
    (root / "tools" / "env").unlink()

    result = _run(root, "pre-commit")

    assert result.returncode == 1
    assert "no tools/env here" in result.stderr


@pytest.mark.parametrize("env_supplies_tools", [True, False])
def test_post_merge_never_fails_the_merge_and_stays_silent(tmp_path, env_supplies_tools):
    """The adoption is an optimisation, so neither arm may report anything."""

    root = _repo(tmp_path, "post-merge", env_supplies_tools=env_supplies_tools)

    result = _run(root, "post-merge")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


def test_post_merge_reaches_the_adoption_through_tools_env(tmp_path):
    """Silence must not be the hook skipping the work it exists to do."""

    root = _repo(tmp_path, "post-merge", env_supplies_tools=True)

    result = _run(root, "post-merge")

    assert result.returncode == 0, result.stderr
    calls = (root / "calls").read_text().splitlines()
    assert len(calls) == 1
    assert calls[0].startswith("python ")


def test_hooks_are_executable_and_tracked_with_that_mode():
    """A hook checked out without +x is a gate that silently never runs."""

    for hook in ("pre-commit", "post-merge"):
        assert os.access(HOOKS / hook, os.X_OK), f"{hook} is not executable"
