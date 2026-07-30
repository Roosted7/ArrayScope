# Warm-worker array exchange protocol (Bundle R7 proposal)

- **Status:** Proposed (2026-07-31). Bundle R7 of the
  [operation-system overhaul](../redesign/operation-system-overhaul-2026-07-30.md).
  Not yet scheduled; the queue owns ordering.
- **Depends on:** ADR 0060 (runtimes), R2 (cancellation threading, probe
  funnel), R3 (parameter kinds, `file` slot, direct-prefix conda launch).

## What problem is left

R3 already deleted the measured 1.55 s `conda run` activation tax by
launching `<prefix>/bin/python` directly, and R2 threaded cancellation
into every child. Three structural limits remain, and they are all
consequences of the same design fact: `python_environment_worker.py` is
a **one-shot** script with a **single-array** contract.

1. **The env-Python runtime cannot take a second array.** Declaring
   input slots on an environment-backed operation sets a definition-level
   unavailable reason (`library.py`, slots-with-environment check), and
   the builder raises if slots arrive anyway. The runtime you need for
   heavy toolboxes (torch, SigPy-on-GPU) is exactly the one that cannot
   receive a sensitivity map, a coordinate array, or a model file.
2. **Every call pays the import bill.** A cold interpreter per apply is
   ~40 ms for numpy — and seconds for `import torch` or a model load.
   Characterization multiplies it: 5–9 probe children per new signature
   (R2 reduced the *cancellation* cost of that, not the warm cost).
   Import cost is inherent to one-shot processes; no launcher flag fixes
   it.
3. **No progress or event transport.** Child stdout is drained and
   discarded; a five-minute iterative recon is a frozen progress story.
   Multi-output (registration → warped image + transform; `fwt` →
   coeffs + slices) has no representation either.

## Proposal

Promote the worker to a **long-lived child** speaking a line-delimited
JSON request/response protocol over stdin/stdout, with arrays exchanged
as **files in a per-request scratch directory** (npy envelope; `.npz`
for multi-array), keeping today's cfl/npy handoff registry as the
serialization layer.

- **Requests:** `{id, op: "run", source, callable, parameters, axis,
  inputs: {name: path}, outputs_dir}` plus `{op: "probe"}` (same body,
  used by characterization) and `{op: "shutdown"}`.
- **Responses:** terminal `{id, ok, outputs: {name: path}, error?}`,
  interleaved with event lines `{id, event: "progress", fraction?,
  message?}` that the runtime forwards to the existing trace/status
  channels. The stdout drain thread already exists; it starts parsing
  instead of discarding.
- **Cancellation:** unchanged semantics — the token that R2 threaded
  through now sends `{op: "cancel", id}` first and falls back to the
  existing SIGTERM→SIGKILL group kill. A worker that honors the message
  can checkpoint; one that ignores it dies exactly as today.
- **Lifecycle:** one worker per (environment id, source file mtime),
  started on first use, reused across applies AND across
  characterization probes, retired on idle timeout, on environment
  edit, and on source-file change (the existing mtime identity).
  Everything stays behind the current builder seam
  (`_make_python_environment_build`), so definitions do not change.
- **Slots:** with multi-file inputs in the envelope, the
  slots-with-environment refusal is deleted — its removal is the
  acceptance test. `file` slots pass paths through untouched.
- **The worker stays ArrayScope-free** (imports nothing from the app),
  exactly like today's one-shot script, so any Python ≥3.9 environment
  can host it.

## What this is not

- Not a new runtime: `command` keeps its one-shot file-per-call
  semantics untouched (BART is stateless and fast to spawn; a protocol
  would buy it nothing).
- Not shared memory. `docs/ideas.md` parks an shm handoff; it composes
  with this protocol later as a faster envelope, and nothing in the
  protocol assumes file transport in its schema.
- Not a Jupyter bridge — but the request/response framing is the piece
  a future bridge reuses pointed the other way.

## Costs and risks

- A resident child is state: source edits, env edits, and crashes must
  all retire it (mtime identity + a dead-process check per request; a
  request to a dead worker restarts it once, then reports).
- Probe-vs-apply interleaving across threads needs one lock per worker;
  the kernel already serializes per-stage evaluation, so contention is
  bounded.
- Protocol versioning: the handshake line carries a version; a mismatch
  falls back to one-shot mode, which remains implemented as the
  degraded path.

## Follow-on it unlocks (Bundle R8)

- A `torch:apply_model` pack: model `file` slot (R3), device/precision
  enum (R3), exact-input characterization (R2), warm model cache (R7),
  progress (R7) — per-slab batching arrives with R5's slab mapping.
- SigPy `nufft`/`espirit` with coord/calib arrays as slots over the
  same envelope; iterative apps stream progress events.
- An `arrayscope.sources` entry-point group over the ADR 0049
  `ArraySource` protocol is independent of this protocol and can land
  any time; it is listed with R8 because the same integrations motivate
  both.
