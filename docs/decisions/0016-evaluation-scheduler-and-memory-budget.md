# 0016 - Evaluation scheduler and memory budgets

## Problem

Generation checks prevented stale background results from committing, but old queued work could still
consume CPU and memory during rapid interaction.

## Decision

Visible image, profile, ROI, and prefetch work use local per-window `EvaluationController` instances.
Controllers support `start_latest()` with a replace group, explicit `EvalPriority`, local
`QThreadPool`, queue clearing, and cancellation tokens. Visible/profile/ROI pools run with one worker.

Render memory estimates are explicit. Visible images, montage, and prefetch have fixed default budgets
of 512 MiB, 1 GiB, and 256 MiB. Requests over budget are skipped with user-facing status text.

## Consequences

Rapid scrolling drops queued old visible requests instead of letting them pile up. Running NumPy calls
are still allowed to finish, but late results cannot commit.

## Rejected alternatives

Using the global Qt thread pool was rejected because unrelated work can compete with visible rendering.
Killing NumPy worker threads was rejected because Python/NumPy work is not safely interruptible.

## Tests required

Scheduler tests cover latest-only commit behavior, max worker count, prefetch dedupe/limits, and close
shutdown. UI tests cover default-off prefetch and montage prefetch skipping.

## Manual checks required

Rapidly scroll operation-backed 3D/4D data and confirm the previous image remains visible while only
the newest render commits.
