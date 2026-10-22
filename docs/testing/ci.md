# Read and reproduce CI results

Open the repository's **Actions → CI → run** page. Each test job writes a
summary with command exits, reported pass/fail/skip counts, failed case details,
and the slowest cases. Its artifact link opens the run's download section.
Failed commands remain failed checks; later reporting and artifact uploads run
after ordinary failures.

Each test artifact contains:

- `*/junit.xml`: every reported case, including failures and timings.
- `*/command.log`: complete combined output, also streamed to the job log.
- `*/status.json`: exact command, exit code, elapsed time, and timeout status.
- `summary.md`, `summary.json`, and `timings.csv`: readable and machine-readable
  results across that job's commands.

The summary shows at most 20 detailed failures; the artifact retains the rest.
Collection failures, missing or malformed reports, and interrupted commands
are reported as gaps. Zero reported cases never establish a test pass. If
checkout or dependency installation fails, inspect those steps first. A lost
runner or forcibly cancelled job may not reach artifact upload.

## Which job owns each check

| Job | Scope and limits |
| --- | --- |
| Python compatibility | Core, operations, public API, architecture, and a shared-memory/process subset on Python 3.10–3.14. At most three matrix jobs run together. |
| GUI fast signal | A small strict UI smoke on a cold map or broad change; cached affected GUI tests for a small mapped change. Its name and command report say which ran. |
| Exhaustive strict coverage | One complete offscreen sweep, with strict callback handling and coverage. This replaces the previous separate strict-UI sweep. Optional maintained codecs and numerical backends come from the locked development environment. |
| Platform validation | Outside-checkout wheel import/version smoke, native platform screenshots, PyQtGraph benchmark smoke, and display/window tests on Linux, macOS, and Windows. The pytest checks run from the source checkout; the outside-checkout import is the explicit wheel check. |
| Hosted diagnostics | Offscreen PyQtGraph/WGPU traces and benchmark artifacts. Hosted software-adapter results are correctness and diagnostic evidence, not hardware timing or presentation acceptance. |
| Real GPU/compositor | Explicit manual opt-in, requiring an available self-hosted runner labelled `arrayscope-gpu`, real GPU access, and Weston. Adapter metadata and compositor logs accompany interaction results. |

Dependency environments use `uv.lock` with `uv sync --frozen`; compatibility
and smoke jobs select their explicit extras. Coverage owns the full maintained
development group. No timing or pixel assertion is disabled to accommodate a
hosted runner. A pixel mismatch, incomplete histogram, stale frame, or timeout
still needs diagnosis.

Commands have wall deadlines, and every job also has a bounded Actions timeout.
The command wrapper records exit `124` on its deadline and terminates its owned
process group on Unix or process tree on Windows. These deadlock bounds do not
change the product's five-second interaction limit. They do not make software
rendering equivalent to a real compositor or turn runner timings into a
performance baseline.

## Seed and reuse the GUI map

Actions sets `CI`, which disables implicit change selection. The cached GUI
path deliberately uses `--testmon --rerun-reds`; exhaustive coverage does not.
A valid map must represent the complete strict/offscreen GUI scope, including
its attribution metadata and red/skip ledgers. A cold smoke is never labelled
or saved as that complete map.

To seed it, choose **Run workflow**, select the branch, and enable
`seed_gui_map`. That manual run executes the complete GUI scope and replaces
the exhaustive coverage sweep for that run. It is a map-maintenance run, not
an exhaustive coverage verdict. A normally completed pytest run may seed the
map even with failing cases; timed-out or collection-aborted runs cannot.

Subsequent CI runs restore that map for the same interpreter, strict/offscreen
regime, and dependency lock. Explicit cache-save steps preserve it after
ordinary test failures, so a red run does not force every future run cold.
When a change is broad or the map has unknown GUI files, the fast job reports
its bounded smoke; the exhaustive owner still checks correctness.

The separate `hardware_gpu` dispatch option requires a provisioned runner.
Its interaction harness does not replace change-specific journey, profiler,
or stress evidence described in [the testing rings](README.md#the-rings).
An unrun or queued hardware lane establishes no hardware acceptance.

## Reproduce a reported command

Use `status.json` to recover the exact command and its test scope. For example:

```bash
uv sync --frozen
CI=true tools/env python tools/ci_report.py pytest \
  --name "Focused CI reproduction" --output work/ci-reproduction/unit \
  --timeout 120 -- -q -n 2 tests/app/test_ci_report.py
tools/env python tools/ci_report.py summarize work/ci-reproduction
```

`pytest` mode enables JUnit while preserving the repository's other configured
defaults; supplying `-p junitxml` alone conflicts with its default plugin block.
`summarize` needs only the standard library, so it can report a setup failure
without importing Qt, NumPy, or the application.
