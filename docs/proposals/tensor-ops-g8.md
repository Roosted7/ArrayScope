# Tensor ops on the GPU — the G8 candidate (T0–T4 ladder)

Status: **proposal, explicitly not now.** This names the "GPU op kernels —
late, evidence-gated experiment" that [roadmap.md](../roadmap.md) has always
deferred, so that when its trigger fires the program is designed instead of
improvised. Direction context:
[tensor-engine-endpoint.md](tensor-engine-endpoint.md) (operation classes,
precision tiers); course rationale:
[reviews/2026-07-19-course-review.md](../reviews/2026-07-19-course-review.md).

**Trigger (all three, in order):** wgpu AUTO promotion flip complete → the
presentation clock landed (queue/Program A close-out) → the
shutdown/cancellation capability contract landed. T0 may ride earlier (it
is display-mapping work, not compute residency).

## The inversion

Today the CPU evaluates image planes (NumPy/pyfftw) and uploads finished
pixels; every scroll step through an FFT'd montage pays CPU FFT **plus**
PCIe upload. Tensor-mode inverts the flow: **upload the source once,
derive on the GPU.** The k-space slab becomes the resident tensor;
scrolling echoes/frames/slices dispatches per-plane FFTs against chunks
already on the device. Steady-state PCIe traffic goes to ~zero, and G7
compressed transport compounds (compress the source upload once, not every
derived frame). Ops become semantic commands in the ADR 0057 protocol —
`DispatchFFT(chunk, axis)`, `DispatchReduce(rss, axis)` — beside
`EnsureChunkResident` and `DispatchHistogram`, which already prove the
compute-command pattern (G6).

## Why the architecture is already positioned

1. **Trust is solved in law, not by this proposal.** Ground rule 8 already
   separates presentation truth from value exactness: presentation-
   qualified samples may explain drawn pixels, but exact hover, histogram,
   ROI, measurement, and export reads "require explicit native semantic
   data or must fall through to exact evaluation." Therefore: **GPU
   derives pixels; the CPU remains the value oracle.** A GPU f32 FFT that
   differs from pyfftw in ulps is a presentation fact, never an answered
   probe. No new philosophy is required — only enforcement of existing law
   on a new producer.
2. **Derived chunks unify with content-keyed residency.** A derived chunk
   is keyed `(source DataChunkKey, op-chain hash)` — the StageCache
   concept moved on-device. Evictable and recomputable (the CPU can always
   rematerialize), so it obeys "residency is a cache, not truth," and it
   enriches chunk-level structural diff (two recon variants share source
   residency; only derived chunks differ).
3. **The parity harness exists.** Framebuffer-to-CPU oracles with fault
   injection are exactly the machinery to certify a GPU compute path
   against the CPU reference at stated tolerances — the same
   differential-oracle pattern that certifies display backends.
4. **The capability taxonomy reserved the slots.** `OperationClass.
   SHADER_ON_READ` and `DERIVED_CHUNK_COMPUTE`
   (`operations/capabilities.py`) are declared and consumed nowhere; this
   proposal is their consumer.

## The ladder

- **T0 — shader-on-read elementwise.** Consume `SHADER_ON_READ`: global
  phase multiply (the "phase spinner"), conjugate, scale — one uniform
  each, zero new residency concepts, fused into the existing display
  mapping. Exit gate: ops render identically (tolerance-stated) to their
  CPU twins under the pixel oracles; exact reads provably fall through.
- **T1 — per-plane 2D FFT as `DispatchFFT`.** One WGSL Stockham FFT
  (radix-2/4, workgroup shared-memory staging, f32), per-plane only
  (display axes; each montage plane independent — the easy, bounded
  shape). Derived-chunk keys; presentation-only. **Exit gate: the
  FFT-scroll headline on real Wayland decisively beats the ~17 fps scalar
  rate. If it does not, the program stops here and keeps T0.**
- **T2 — chains.** fftshift as pure index addressing (no data movement);
  elementwise fused into the FFT epilogue; op-chain hashing; the
  histogram-frontier eviction shield generalized to compute frontiers.
- **T3 — cross-axis reductions.** RSS/mean over a non-display axis (e.g.
  32 coils resident per output tile) — the hardest residency case,
  deliberately last. Requires dependency-aware pinning beyond T2's shield.
- **T4 — endpoint.** The pipeline is expressible as compute commands; the
  kernel brokers CPU and GPU as resources; the CPU is oracle and fallback.
  This fulfills the tensor-engine-endpoint direction record.

## Hard parts, stated up front

- **FFT in WGSL is real work.** No library ecosystem (cuFFT/VkFFT
  unreachable through wgpu); WGSL has no f64. Per-plane 2D is tractable;
  chunked FFT along huge axes stays CPU — declared, not implied.
- **Residency frontiers get more dangerous.** The `EnsureChunkResident`
  self-eviction crash (2026-07-19, queue row 3) is the prototype failure;
  its scoped-pin shield is the prototype fix. Compute makes frontier
  pinning a first-class design element, not a patch.
- **No preemption on GPU.** Cancellation = bounded dispatch sizes the
  scheduler can decline to enqueue (chunk-granular submissions). Aligned
  with "bounded by default," but must be designed in; interacts with the
  shutdown contract (hence the trigger ordering).
- **wgpu-only compute deepens backend asymmetry.** PyQtGraph-headless
  keeps the CPU path, so every GPU op has a CPU twin by construction. The
  standing mitigation: **CPU is the reference semantics; the GPU path is a
  certified accelerator; a GPU op without a continuously-tested CPU twin
  may not exist.** (This is the archaeology's P3 lesson applied forward.)
- **Memory doubling (source + derived resident).** Bounded by making
  derived chunks the most evictable class; they are pure cache.

## Why not an ADR yet

The trigger has not fired; T1's exit gate is a measurement that does not
exist yet; and the WGSL FFT effort estimate deserves a spike before any
durable commitment. When T1's gate is green, T0+T1 convert to an ADR and
T2–T4 move to the queue with per-step gates.
