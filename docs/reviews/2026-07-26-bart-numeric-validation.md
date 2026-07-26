# BART numeric validation — 2026-07-26

## Verdict

The standing external numeric gap is closed for the default exposed
`bart:ecalib`, `bart:walsh`, and `bart:pics` paths on one small, deterministic,
fully sampled four-coil problem. The real-toolbox run passed independent
references or mathematically required invariants; it did not reuse BART to
grade BART.

The first run was usefully red:

- `bart:pics` reconstructed the right spatial image at scale `1.3143405`
  because the definition omitted `-S`. Relative L2 was `0.314340` and maximum
  absolute error was `0.346394`, far outside the preselected limits. The
  definition now requests BART's output rescaling; no tolerance changed.
- `bart:walsh` returned shape `(24, 24, 1, 10)` for four coils and an output-axis
  RSS range of `14914.5` to `47561.4`, disproving the advertised
  sensitivity-map contract. BART documents and implements this command as
  packed Hermitian covariance for a subsequent `ecaltwo`. The operation is now
  named and validated as covariance.
- The system pack declared execution environment `bart` but bypassed named
  environment resolution. The pack now consumes that record, with an ambient
  `PATH` fallback only when no record is configured.

## Environment and command

- ArrayScope: `220470fde36f9a88b6edd3619b1b54f15d2c76ca`
- BART executable: `/home/thomas/projects/bart/bart`
- BART source: `731bfd3ec4f09ffa59b497ded1274a2c0357c8af`
- Reported version: `v1.0.00-1-g731bfd3-dirty`
- Dirty BART files at the evidence boundary: `Makefile`, `matlab/bart.m`, and
  untracked `matlab/bartORIG.m`
- Runtime library path:
  `/home/thomas/miniconda3/envs/precon311/lib` (required for the local MKL-linked
  binary)

The exact command was:

```bash
conda run -n arrayscope python tools/validate_bart_numerics.py \
    --bart-toolbox-path /home/thomas/projects/bart \
    --library-path /home/thomas/miniconda3/envs/precon311/lib
```

The dirty version warning makes this evidence exact for the named local build,
not a claim about every upstream BART release. The harness is the reproducible
gate for another installation.

## Method

The harness creates a `24 x 24` complex64 Gaussian-mixture image and a constant,
normalized four-coil vector. It forms fully sampled k-space with NumPy's
centered unitary two-dimensional FFT. The construction is rank one in coil
space and the SENSE operator is isometric, which supplies three sharp oracles:

| operation | reference or invariant | why it is independent |
|---|---|---|
| `bart:ecalib -m 1` | foreground coil vectors have unit RSS norm and span the known rank-one coil subspace | the coil vector is analytic; absolute correlation discards only the arbitrary map phase |
| `bart:walsh -r 5` | unpacked lower-triangle storage forms a Hermitian PSD covariance whose dominant eigenspace is the known conjugated coil vector | matrix unpacking and `numpy.linalg.eigh` grade BART's output; no BART calibration result is reused |
| `bart:pics -S -i 30` | centered NumPy inverse FFT followed by direct SENSE combination | normalized maps make the fully sampled noiseless inverse analytic |

## Results

| operation | measured result | acceptance | result |
|---|---|---|---|
| `bart:ecalib` | max foreground norm error `1.192e-7`; minimum subspace correlation `0.999999940` | norm error `<= 2e-5`; correlation `>= 0.999` | PASS |
| `bart:walsh` | relative minimum-eigenvalue floor `-2.697e-8`; minimum dominant fraction `1.000000000`; minimum subspace correlation `1.000000000` | floor `>= -1e-6`; fraction and correlation `>= 0.999` | PASS |
| `bart:pics` | relative L2 `1.521e-7`; maximum absolute error `2.417e-7` | relative L2 `<= 1e-5`; max absolute `<= 5e-5` | PASS |

The thresholds were chosen from the complex64 error model and the exact
rank-one construction before the corrected run. They retain roughly two
orders of magnitude over the observed floating-point residue while rejecting
the original PICS amplitude defect by four orders of magnitude.

## Scope and refusal boundary

This proves the shipped default paths on fully sampled, noiseless, two-
dimensional data with one ESPIRiT map and normalized single-map sensitivities.
It does not claim:

- accuracy for `ecalib` with multiple map sets or noisy/rank-higher coil data;
- PICS behavior for undersampling, regularization, trajectories, multiple map
  sets, or three-dimensional reconstruction, none of which the shipped
  definition currently exposes beyond its input arrays and iteration count;
- that `bart:walsh` returns sensitivity maps. It does not. The validated product
  is packed covariance; producing maps would require an explicit `ecaltwo`
  stage and its own contract.

The optional pytest gate calls the same harness wherever `bart` is on `PATH`.
Without a toolbox it skips with an explicit reason. The fake-BART tests remain
necessary for process mechanics and make no numeric claim.

The CLI's absent-toolbox state was also exercised with a nonexistent
`--bart-toolbox-path`: it printed `BART numeric validation: UNAVAILABLE` and
returned status `2`, distinct from numeric failure status `1`.
