# Benchmark scripts

These scripts are for local performance profiling and are not part of the unit-test suite.

## Basis generation

```bash
python scripts/benchmark_basis.py
python scripts/benchmark_basis.py --only spin_one
python scripts/benchmark_basis.py --json basis_benchmark.json
python scripts/benchmark_basis.py --markdown basis_benchmark.md
```

Use `--markdown` to write a compact GitHub-ready timing table.

## Hamiltonian construction

By default, the Hamiltonian benchmark uses each case's recommended builder.
Use `--builder sparse`, `--builder optimized`, or `--builder bitmask` to force
one builder where supported, and `--builder all` to compare all supported
builders per case.

```bash
python scripts/benchmark_hamiltonian.py
python scripts/benchmark_hamiltonian.py --split-basis-timing
python scripts/benchmark_hamiltonian.py --builder all --split-basis-timing
python scripts/benchmark_hamiltonian.py --builder sparse --only qdm --split-basis-timing
python scripts/benchmark_hamiltonian.py --list-cases
python scripts/benchmark_hamiltonian.py --json hamiltonian_benchmark.json
python scripts/benchmark_hamiltonian.py --builder all --split-basis-timing --markdown hamiltonian_benchmark.md
```

Use `--markdown` to write a compact GitHub-ready report containing both the raw
benchmark table and a fastest-observed-builder summary. This is useful for
copying benchmark results into GitHub issues without including the large model
parameter payload from `--json`.

## Cage search

```bash
python scripts/benchmark_cage_search.py
python scripts/benchmark_cage_search.py --only qdm --split-basis-timing
python scripts/benchmark_cage_search.py \
  --only qlm \
  --degenerate-basis-strategy ipr \
  --ipr-n-restarts 32
python scripts/benchmark_cage_search.py --json cage_search_benchmark.json
python scripts/benchmark_cage_search.py --markdown cage_search_benchmark.md
```

The cage-search benchmark reports separate timings for candidate generation,
candidate solving, rank deduplication, and total search time. Use `--markdown`
to write a compact GitHub-ready stage-timing table.

## Cage-Lindblad construction

```bash
python scripts/benchmark_cage_lindblad.py
python scripts/benchmark_cage_lindblad.py --only qdm
python scripts/benchmark_cage_lindblad.py --only qlm --builder bitmask
python scripts/benchmark_cage_lindblad.py --monitor-source reduced_iz_operators
python scripts/benchmark_cage_lindblad.py   --monitor-source reduced_iz_operators   --reduced-iz-monitor-decomposition connected_support
python scripts/benchmark_cage_lindblad.py --check-liouvillian
python scripts/benchmark_cage_lindblad.py --skip-jump-residuals
python scripts/benchmark_cage_lindblad.py --json cage_lindblad_benchmark.json
python scripts/benchmark_cage_lindblad.py --markdown cage_lindblad_benchmark.md
```

The Cage-Lindblad benchmark separates model build, cage search, cage
classification, and open-system construction time. It also reports construction
sub-stages such as monitor assembly, jump assembly, diagnostics, and optional
Liouvillian checks. Use `--skip-jump-residuals` to separate jump materialization
from the cost of computing `||J psi||` diagnostics. The default local-term
builder is `sparse`, but encoded bitmask build results are promoted to the
bitmask local-term path internally, matching the construction API behavior.

## Open-system solvers and MCWF

```bash
python scripts/benchmark_open_system.py
python scripts/benchmark_open_system.py --operation single_trajectory --n-times 201
python scripts/benchmark_open_system.py --operation mcwf --n-trajectories 512
python scripts/benchmark_open_system.py --only qubit --operation all
python scripts/benchmark_open_system.py --json open_system_benchmark.json
python scripts/benchmark_open_system.py --markdown open_system_benchmark.md
python scripts/benchmark_open_system.py \
  --operation all \
  --n-trajectories 512 \
  --json open_system_benchmark.json \
  --markdown open_system_benchmark.md
```

The open-system benchmark separates dense/sparse operator preparation,
Liouvillian construction, deterministic Lindblad solvers, single MCWF trajectory
evolution, and MCWF ensemble sampling. Use `--operation single_trajectory`
with a larger `--n-times` value to profile animation-oriented runs.

---

# Grid search for cages

Small dry run:
```bash
python scripts/run_cage_sweep.py \
  --output-root ./data/qlinks_cage_sweep_test \
  --backend serial \
  --dry-run
```

Small real test:
```bash
python scripts/run_cage_sweep.py \
  --output-root ./data/qlinks_cage_sweep_test \
  --backend serial \
  --models qdm \
  --geometries square \
  --square-sizes 2x2,3x2 \
  --max-states 65536
```

Ray run:
```bash
python scripts/run_cage_sweep.py \
  --output-root ./data/qlinks_cage_sweep_full \
  --backend ray \
  --num-cpus-per-task 1 \
  --max-states 262144 \
  --ipr-n-restarts 128 \
  --ipr-candidate-count 64
```

## Status inspection

Declare the path to output folder
```bash
OUTPUT_ROOT=/path/to/cage_sweep_output
```
Note that the commands below require to have `jq` installed.

Then, you can monitor with:
```bash
find "$OUTPUT_ROOT/jobs" -name status.json \
  -exec jq -r '[.status, .job_id, (.n_states // ""), (.n_records // "")] | @tsv' {} \;
```

Counts:
```bash
find "$OUTPUT_ROOT/jobs" -name status.json \
  -exec jq -r '.status' {} \; | sort | uniq -c
```

### Find jobs that contain regional candidates

```bash
find "$OUTPUT_ROOT/jobs" -name summary.json -print0 \
  | xargs -0 jq -r '
      select((.classification_counts.regional_candidate // 0) > 0)
      | "job=\(.job_id)  n_states=\(.n_states)  regional=\(.classification_counts.regional_candidate)  h5=\(.hdf5_path)"
    '
```

### Count total regional candidates across the whole sweep

```bash
find "$OUTPUT_ROOT/jobs" -name summary.json -print0 \
  | xargs -0 jq -s '
      map(.classification_counts.regional_candidate // 0)
      | add
    '
```

---

# Docker evidence jobs

The model-evidence notebooks can be run as detached Docker jobs through:

```bash
scripts/docker_run_evidence_job.sh spin1 [job options]
scripts/docker_run_evidence_job.sh qdm [job options]
```

The wrapper forwards model-specific flags unchanged and understands the common
`--stage compute|render|all` workflow. Path-bearing options
`--data-dir`, `--source-data-dir`, and `--export-dir` may be supplied as paths
relative to the repository root, absolute host paths under a mounted root, or
container paths under `/workspace/qlinks` and `/workspace/output`. The wrapper
translates them before starting Docker and prints both host and container paths.

Numerical pass:

```bash
QLINKS_EVIDENCE_RUN_ID=spin1_production \
QLINKS_NUM_THREADS=16 \
QLINKS_DOCKER_MEMORY_LIMIT=400g \
scripts/docker_run_evidence_job.sh spin1 \
  --stage compute \
  --profile production \
  --microcanonical-sizes 6,8,10,12 \
  --deformation-sizes 6,8,10,12 \
  --large-size-sizes 14 \
  --large-size-eigenpairs 8192 \
  --representative-kappa 0.1 \
  --principal-kappa-values 0.05,0.10,0.15,0.20 \
  --kappa-values 0,0.05,0.10,0.15,0.20 \
  --large-size-concentration \
  --window-exponents 0.5,0.25,0 \
  --window-prefactors 0.75,1.0,1.25 \
  --fit-bootstrap-repeats 1000 \
  --counting-max-length 60 \
  --timeout -1
```

The `L=14` point uses a sparse shift-invert partial spectrum and sparse projected
observables; it is deliberately not added to the complete deformation grid.
The primary large-size job is run at the interior representative point
`kappa/J=0.1`. Inspect `spin1_xy_large_size_memory_feasibility.csv` and the
`window_coverage_complete` columns before accepting it. If the primary window
is not covered, increase `--large-size-eigenpairs`. Production enables the
complete 19-operator covariance diagnostic by default; use
`--no-large-size-concentration` for a cheaper preflight run, then rerun with the
diagnostic once the partial-spectrum window is known to be covered.


Square-QDM checkerboard finite-temperature production pass:

```bash
QLINKS_EVIDENCE_RUN_ID=qdm_checkerboard_finite_beta \
QLINKS_NUM_THREADS=16 \
QLINKS_DOCKER_MEMORY_LIMIT=400g \
scripts/docker_run_evidence_job.sh qdm \
  --stage compute \
  --profile production \
  --transport-repeats 1,2,3 \
  --ed-repeats 1,2 \
  --phase-values 0,0.025,0.05,0.075,0.10 \
  --positive-phase-values 0.025,0.05,0.075,0.10 \
  --representative-phase 0.05 \
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

The beta-zero energy-density gate has already failed for `lambda_star=1`, so
the primary protocol is now explicitly finite temperature.  Full spectra remain
restricted to `4x4` and `8x4`.  The `12x4` lane constructs the fully resolved
checkerboard translation irrep sparsely and checkpoints the energy-matched
canonical typicality result before attempting an interior spectrum.  The default
interior solver is folded-spectrum Lanczos on `(H-E_psi)^2`; it uses sparse
Hamiltonian products only and does not construct SuperLU factors.  Direct-LU
shift-invert is retained only as an explicit diagnostic fallback.  A partial-
spectrum row is accepted only when its reported `window_coverage_complete` flag
is true and the staged budget comparison is available.  The complete stripe-local
covariance calculation is then evaluated on the same covered window.  The
1024-eigenpair setting is a production starting point, not an assumed sufficient
window size; increase it only if the coverage export requires it.

The job also classifies the translated `A,Z` joint-dark subspace against the
directly projected Type-I compact-cage span at the exact-ED sizes.  The physical
finite-beta canonical target is kept distinct from joint-dark cleaning, which is
reported as a finite-size diagnostic.

Render the completed timestamped folder with:

```bash
scripts/docker_run_evidence_job.sh qdm \
  --stage render \
  --source-data-dir experimental/data/evidence_jobs/qdm_checkerboard_production_YYYYMMDDTHHMMSSZ \
  --use-tex \
  --figure-formats pdf,svg \
  --export-dir output/qdm_checkerboard_production
```

TeX-backed render pass using a repository-relative host path:

```bash
scripts/docker_run_evidence_job.sh spin1 \
  --stage render \
  --source-data-dir experimental/data/evidence_jobs/spin1_production \
  --use-tex \
  --figure-formats pdf,svg \
  --export-dir output/spin1_production
```

Use custom host storage by setting the mount roots before invoking the wrapper:

```bash
QLINKS_DATA_DIR=/large-volume/qlinks-data \
QLINKS_OUTPUT_DIR=/large-volume/qlinks-output \
scripts/docker_run_evidence_job.sh qdm \
  --stage compute \
  --data-dir /large-volume/qlinks-data/qdm-production \
  --profile production
```

To inspect path translation and the complete Docker command without starting a
container:

```bash
QLINKS_DOCKER_DRY_RUN=1 \
scripts/docker_run_evidence_job.sh spin1 --stage compute --profile smoke
```
