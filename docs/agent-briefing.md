# Briefing for delegated coding agents

Written for the agents we hand bounded bundles of work to, and for whoever
writes their prompts. It is a record of what *actually* went wrong across the
operations program (four bundles, 2026-07-24 → 2026-07-26), not a generic style
guide. Every item below is something an agent did that passed its own tests and
still had to be caught in review.

Read [ground-rules.md](ground-rules.md) first — that is the repo's contract.
This file is about failure modes that survive the contract.

## The recurring failure modes

### 1. A default that was policy for the old world, silently retired by your change

Bundle A demoted the sigpy/BART packs. `DEFAULT_MORE_GROUPS` — the list of
groups the add popup folds away — happened to name exactly those backend
groups. Removing the packs therefore emptied the fold-out and dropped all 37
operations into one flat scroll: a defect with no failing test, produced by a
change that never touched the popup.

**Ask:** what constants, defaults, or thresholds were *tuned for the world I am
changing*? Grep for the names you are removing. A policy that now selects
nothing, or everything, is a bug even when nothing asserts on it.

### 2. Test tolerances that hide the defect they were written to catch

The Numba normalize kernel accumulated in float32 and summed sequentially where
NumPy sums pairwise. Its agreement test used an 8-element axis and
`rtol=2e-5` — both loose enough to hide error that grows with axis length
(2.6e-4 relative at 1e6 samples). Worse, the kernel only engages once the JIT is
warm, so the same operation on the same data returned different values
depending on compilation timing.

**Ask:** what is the failure *mode* of this code, and does the test vary the
axis it depends on? An accumulation test must vary length. A cache test must
vary key. A tolerance you picked to make the test pass is not evidence.

### 3. Edge cases the original handles and the replacement does not

Discovered shape rules rejected any predicted extent below 1. But the built-in
crop returns `(0, 3, 4)` for `start == stop` — a legal array. So a *duplicate*
of crop raised where the operation it copied succeeded.

**Ask:** when you re-implement, generalize, or wrap something, run the original
and the replacement side by side over the boundary inputs — empty, single
element, degenerate parameters, extreme dtype. Equality with the incumbent is
the cheapest oracle in the repo and it is almost never used.

### 4. Tests that encode an intermediate step as if it were the destination

Bundle B blocked shape-changing duplicates and wrote tests asserting they raise,
including the string "Bundle D owns shape discovery". Bundle D then delivered
shape discovery, and those tests had to be rewritten. Similarly, tests asserting
"a julia runtime is skipped" outlived the commit that made julia concrete.

**Ask:** does this test pin the *product's* behaviour, or my bundle's temporary
scaffolding? Scaffolding tests are fine — name them so, and delete them in the
bundle that removes the scaffold. When you supersede another bundle's
behaviour, grep the test suite for its error strings.

### 5. Reporting failures without localizing them

Bundles C and D both reported "25 failures, in untouched rendering code". Both
were right, and neither found that the cluster reproduces on pristine `main` at
`f2e7e985` — a five-minute bisect that would have converted an alarming number
into a one-line, actionable finding.

**Ask:** before reporting a failure as pre-existing, prove it. Check out the
merge-base, run the narrowest failing node, and name the first bad commit. "Not
mine" is a claim; a commit hash is evidence.

### 6. Surfaces with no screenshot

The custom-operation import dialog shipped with a 4-column parameter table
against the manager's 6, a default height leaving room for two rows, and a
combobox that rendered pixel-identically to a read-only line edit — through a
review pass that looked at screenshots, because no gallery scenario rendered
*that* dialog.

**Ask:** does every surface I touched appear in `tools/ui_gallery.py`? Add the
scenario in the same bundle, run it, and **look at the PNG** — you can read
images. Do not describe a screenshot you have not opened.

### 7. Test volume standing in for test coverage

Every bundle produced overlapping tests: three that assert the same section
appears in a listing, several parametrizations of one construction path — while
the boundary cases in §3 went untested.

**Ask:** would one parametrized test replace these five? Prefer few sharp tests.
A merged test with a comment explaining the hazard is worth more than four that
restate the happy path.

### 8. Silent failure in your own tooling

While integrating, a splice script run under `conda run` failed and printed
nothing; its output was assumed and conflict markers were committed. (That one
was the orchestrator's, not an agent's — the lesson is general.)

**Ask:** did I *verify* the effect, or read the exit code? After scripted edits,
grep for the thing you expect to be gone.

## What consistently goes well — keep doing it

- **Honest negative results.** Bundle A measured Numba wins of 3.4× and 11.7×
  and still declined to land them, because mixing a Numba whole-array path with
  a NumPy region path broke exact `ELEMENTWISE` region conformance by 1–2 ULP,
  which would put seams at tile boundaries. It named the condition under which
  the decision should be revisited. That is exactly the standard.
- **Refusing to fix someone else's owner.** Both C and D left the rendering
  failures alone rather than "fixing" a subsystem they had not measured.
- **Loud, explicit blocking over silently-wrong behaviour.** Bundle B emitted a
  blocked template with a clear message rather than a wrapper whose identity
  shape adapter would lie to the planner.
- **Measuring the cost you introduce.** Bundle D published probe cost (cold
  median 202–663 µs, warm ~5 µs) *and* flagged that its unbounded real-call
  fallback may not be worth it for an expensive command.

## Prompt checklist

When writing the next bundle's prompt, state explicitly:

1. **Ownership boundaries** — which files this bundle owns, which belong to a
   concurrent bundle, and who owns the seam between them.
2. **The incumbent oracle** — what existing behaviour the new code must match,
   and the boundary inputs to compare on.
3. **Which tests are scaffolding**, and which bundle deletes them.
4. **Localize-before-reporting**: pre-existing failures need a first-bad-commit,
   not an assertion.
5. **Gallery scenario in the same bundle**, with "open the PNG and describe what
   you see" as an explicit step.
6. **Test budget** — "prefer few sharp tests; parametrize rather than duplicate"
   is worth saying every time, because the default drifts the other way.
