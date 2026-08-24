# Resumable evidence cache

Heavy evidence jobs use two distinct storage layers.

- `experimental/data/evidence_jobs/<timestamped-run>/` is an immutable execution/provenance record.
- `experimental/data/evidence_cache/` contains validated reusable numerical payloads.
- `experimental/data/evidence_registry/` contains small logical pointers to adopted/authoritative runs.

The timestamp remains useful for distinguishing attempts, but downstream jobs should not need to
repeat a week-scale eigensolve merely because a new attempt has a new timestamp.

## Compatibility rule

A checkpoint is reused from its **scientific problem signature**, not from its producing git commit.
For the QDM folded-spectrum lane the signature contains the resolved-sector Hamiltonian fingerprint,
sector dimension, target energy, matrix dtype/shape, and folded-operator schema. The producing commit,
run id/timestamp, backend, solver tolerance, and solver statistics remain provenance metadata.

Before a cached spectrum is accepted as final evidence, the loader checks its metadata and array
shapes, finite values, stored residuals, and a deterministic sample of physical eigenpair residuals
and orthogonality. A scientifically compatible checkpoint that is numerically sane but does not meet
the requested final tolerance may be used only as a warm start.

The cache states are:

- `VALID_FINAL` — may replace the expensive solve.
- `VALID_WARM_START` — may seed a later solve but is not accepted as final evidence.
- `INCOMPATIBLE` — must not be reused for this scientific problem.

## Square-QDM large strip

Each folded-spectrum budget is persisted independently:

```text
experimental/data/evidence_cache/
└── qdm/checkerboard_large_strip/<problem-signature>/
    ├── arpack/
    │   ├── budget_00000512/
    │   ├── budget_00001024/
    │   └── ...
    └── primme/
        ├── budget_00000512/
        └── ...
```

A completed budget contains uncompressed `energies.npy`, `eigenvectors.npy`, `residuals.npy`, an
optional `transformed_residuals.npy`, and `metadata.json`. The metadata file is written last and is
the completion marker. The next timestamped run validates and reuses the largest applicable
completed budget before computing missing work.

The normal QDM job runner enables this cache automatically. The folded backend can be selected with

```bash
--large-strip-folded-backend auto|arpack|primme
```

`auto` uses PRIMME when it is installed and otherwise retains SciPy/ARPACK. This preserves the
standard notebook image while allowing a dedicated PRIMME production image.

### PRIMME image

Build the dedicated Python-3.13 evidence image with:

```bash
bash scripts/docker/build_primme_evidence_image.sh
```

The ordinary evidence launcher currently pulls its image tag unconditionally. For a locally built
PRIMME image, use the dedicated launcher below; it defaults to Docker pull policy `never`, so the
local image cannot be replaced by a registry pull.

```bash
QLINKS_EVIDENCE_RUN_ID=qdm_checkerboard_primme \
QLINKS_NUM_THREADS=16 \
QLINKS_DOCKER_MEMORY_LIMIT=400g \
bash scripts/docker/docker_run_qdm_primme_evidence.sh \
  --stage compute \
  --profile production \
  --transport-repeats 1,2,3 \
  --ed-repeats 1,2 \
  --thermal-protocol finite-beta \
  --dark-classification-repeats 1,2 \
  --run-large-strip \
  --large-strip-repeats 3 \
  --large-strip-spectral-method folded \
  --large-strip-subspace-budgets 512,1024,2048,4096,8192 \
  --large-strip-extra-convergence-step \
  --finite-beta-samples 8 \
  --finite-beta-beta-max 0.25 \
  --finite-beta-beta-points 41 \
  --transfer-max-length 256 \
  --window-prefactors 0.50,0.75,1.00 \
  --timeout -1
```

PRIMME receives up to 256 vectors from the largest validated lower-budget checkpoint as an initial
subspace by default. Use `--primme-warm-start-vectors 0` to disable this or change the cap explicitly.
The initial PRIMME comparison should keep the same folded operator, tolerance, budget ladder, and
scientific acceptance checks as ARPACK; backend performance is not itself evidence of convergence.

The already-running August 11 ARPACK call cannot be checkpointed in the middle of its current
`eigsh` invocation. It should be allowed to finish if it remains numerically active. New jobs created
from this code checkpoint immediately after each folded solve returns.

## Spin-1 Sec. VI

The Sec. VI provisioning helper already writes a spectral payload immediately after an expensive
sparse solve. New attempts now default that checkpoint directory to

```text
experimental/data/evidence_cache/spin1/sec6_sparse/
```

instead of the timestamped output directory. The existing checkpoint metadata compatibility checks
remain in force. `--checkpoint-source-dir` is retained only for legacy/external sources.

## Adopt existing timestamped runs

Existing runs are not discarded. Register a completed run with:

```bash
python experimental/jobs/adopt_evidence_run.py \
  experimental/data/evidence_jobs/<run-id>
```

For an otherwise failed or still-running parent run that already contains a completed independent
checkpoint:

```bash
python experimental/jobs/adopt_evidence_run.py \
  experimental/data/evidence_jobs/<run-id> \
  --allow-incomplete-run
```

Spin-1 checkpoint arrays are validated and hard-linked into the stable cache when the source and
cache are on the same filesystem; copying is used as a fallback. The timestamped source remains
untouched. QDM/other runs are still recorded in the logical evidence registry even when they do not
contain an adoptable spectral payload.

Use `--dry-run` to inspect the validation/adoption plan without writing anything.

## Operational rule

For a heavy scientific run, the intended order is:

```text
validate existing evidence
    -> reuse every compatible completed checkpoint
    -> compute only missing stages
    -> validate the new result
    -> checkpoint immediately
    -> continue with derived observables/figures
```

A timestamped job directory records *an attempt*. It is no longer the only home of an expensive
scientific result.
