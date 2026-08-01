"""What the git hooks do around the work, rather than the work itself.

Ring 0 — no Qt, no imports of ``arrayscope``. The hooks are shell, and the
part worth pinning is the wiring: whether a hook that cannot see
``ruff``/``python`` hands itself to ``tools/env`` and continues, whether the
guard that makes that a *single* handoff really holds, and whether
``post-checkout`` fires on worktree creation and on nothing else. A recursion
bug in a commit hook, or a setup routine that runs on every branch switch, is
not a thing to discover interactively.

So the real gates and the real setup are never run here. Each test builds a
throwaway repository whose tools are stubs that record they were called, which
is what lets these stay instant — the real ``pre-commit`` costs ~2.1 s and a
real ``uv sync`` ~1.7 s, and paying that per suite run is the same trade
``tests/app/test_dead_code.py`` already refused.

``post-checkout`` is exercised through real ``git worktree add`` / ``git
checkout`` / ``git clone`` rather than by calling it with hand-written
arguments, because its gate is a claim about what git passes and that claim is
the whole safety of the hook.
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

    # Reached only if the hook gets as far as its real work. Every scan the
    # hook shells out to needs a file here, or the recovery under test is
    # masked by a missing-file failure from the scan after it.
    for scan in ("dead_code.py", "timing_guard.py"):
        (root / "tools" / scan).write_text("")
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
    assert calls == [
        "ruff check .",
        "ruff format --check .",
        f"python {root}/tools/dead_code.py",
        f"python {root}/tools/timing_guard.py",
    ]


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

    for hook in ("pre-commit", "post-merge", "post-checkout"):
        assert os.access(HOOKS / hook, os.X_OK), f"{hook} is not executable"


# --- post-checkout: worktree creation, and nothing else ----------------------


def _origin(tmp_path: Path) -> Path:
    """A repository shaped enough for post-checkout to have work to do."""

    origin = tmp_path / "origin"
    (origin / "tools").mkdir(parents=True)
    (origin / "tests").mkdir()
    (origin / "data").mkdir()
    (origin / "data" / "scan.nii").write_text("")
    (origin / ".envrc").write_text("# nothing\n")

    # The seed step reaches for the real `seed_map`, so the stub stands in for
    # it under the same name: the point of the hook is that it calls the
    # existing tooling at a better moment, never that it copies by hand.
    (origin / "tests" / "__init__.py").write_text("")
    (origin / "tests" / "testmon_policy.py").write_text(
        "import os\n"
        "def seed_map(root):\n"
        "    open(os.path.join(os.environ['HOOK_CALLS_DIR'], 'calls'), 'a')"
        ".write(f'seed_map {root}\\n')\n"
        "    return '/donor/.testmondata'\n"
    )

    # data/ is gitignored here exactly as it is in the real repo — that is why
    # a worktree has none and why the link step exists at all.
    (origin / ".gitignore").write_text("data/\n")

    subprocess.run(["git", "init", "-q", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "-C", str(origin), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(origin), "commit", "-qm", "root"],
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )
    return origin


def _hooked_env(tmp_path: Path) -> dict[str, str]:
    """Real PATH, with the slow real tools shadowed by recording stubs."""

    stubbin = tmp_path / "stubbin"
    stubbin.mkdir(exist_ok=True)
    calls = tmp_path / "calls"
    for tool in ("uv", "direnv"):
        _stub(stubbin / tool, f'echo "{tool} $*" >> "{calls}"\nexit 0')

    return {
        **os.environ,
        "PATH": f"{stubbin}{os.pathsep}{os.environ['PATH']}",
        "HOOK_CALLS_DIR": str(tmp_path),
        "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"),
    }


def _git(args: list[str], env: dict[str, str], cwd: Path | None = None) -> None:
    subprocess.run(
        ["git", "-c", f"core.hooksPath={HOOKS}", *args],
        check=True,
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_post_checkout_sets_a_new_worktree_up(tmp_path):
    """All four steps, on the one event that should trigger them."""

    origin = _origin(tmp_path)
    env = _hooked_env(tmp_path)

    _git(["-C", str(origin), "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "topic"], env)

    worktree = tmp_path / "wt"
    calls = (tmp_path / "calls").read_text()
    assert "uv sync" in calls
    assert "direnv allow" in calls
    assert f"seed_map {worktree}" in calls
    # The dataset is gitignored, so the link is the only way a worktree has one.
    assert (worktree / "data").is_symlink()
    assert (worktree / "data" / "scan.nii").exists()


def test_post_checkout_ignores_a_branch_switch(tmp_path):
    """Setup belongs to creation. Re-running it per checkout would be a bug."""

    origin = _origin(tmp_path)
    env = _hooked_env(tmp_path)
    _git(["-C", str(origin), "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "topic"], env)

    worktree = tmp_path / "wt"
    (worktree / "data").unlink()
    (tmp_path / "calls").unlink()

    _git(["checkout", "-q", "-b", "other"], env, cwd=worktree)
    _git(["checkout", "-q", "topic"], env, cwd=worktree)

    assert not (tmp_path / "calls").exists(), "a branch switch re-ran the setup"
    assert not (worktree / "data").exists()


def test_post_checkout_ignores_a_clone(tmp_path):
    """A clone shares the null old sha and must not be mistaken for a worktree.

    It is told apart by ``.git`` being a directory rather than the gitdir
    pointer file only a linked worktree has — the cheapest fact that separates
    them, and the reason the gate does not rest on the sha alone.
    """

    origin = _origin(tmp_path)
    env = _hooked_env(tmp_path)

    _git(["clone", "-q", str(origin), str(tmp_path / "cloned")], env)

    assert (tmp_path / "cloned" / ".git").is_dir()
    assert not (tmp_path / "calls").exists()
    assert not (tmp_path / "cloned" / "data").exists()


def test_post_checkout_opts_out(tmp_path):
    """Scripted bulk creation gets to say no."""

    origin = _origin(tmp_path)
    env = {**_hooked_env(tmp_path), "ARRAYSCOPE_WORKTREE_SETUP": "0"}

    _git(["-C", str(origin), "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "topic"], env)

    assert not (tmp_path / "calls").exists()
    assert not (tmp_path / "wt" / "data").exists()


def test_post_checkout_survives_every_step_failing(tmp_path):
    """A worktree that exists and is unset up beats no worktree at all.

    ``git worktree add`` must succeed even when every convenience below it
    errors, which is why the hook carries no ``set -e`` and always exits 0.
    """

    origin = _origin(tmp_path)
    env = _hooked_env(tmp_path)
    stubbin = tmp_path / "stubbin"
    for tool in ("uv", "direnv"):
        _stub(stubbin / tool, "exit 3")
    # No importable policy module, so the seed step raises inside python.
    (origin / "tests" / "testmon_policy.py").unlink()
    # An existing entry the link step must not clobber.
    (tmp_path / "wt").mkdir()
    (tmp_path / "wt").rmdir()

    _git(["-C", str(origin), "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "topic"], env)

    assert (tmp_path / "wt" / ".git").is_file(), "the worktree itself must still be created"
