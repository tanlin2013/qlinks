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
