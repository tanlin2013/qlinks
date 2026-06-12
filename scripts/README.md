# Benchmark scripts

These scripts are for local performance profiling and are not part of the unit-test suite.

## Basis generation

```bash
python scripts/benchmark_basis.py
python scripts/benchmark_basis.py --only spin_one
python scripts/benchmark_basis.py --json basis_benchmark.json
```

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
```

## Cage search

```bash
python scripts/benchmark_cage_search.py
python scripts/benchmark_cage_search.py --only qdm --split-basis-timing
python scripts/benchmark_cage_search.py \
  --only qlm \
  --degenerate-basis-strategy ipr \
  --ipr-n-restarts 32
python scripts/benchmark_cage_search.py --json cage_search_benchmark.json
```

The cage-search benchmark reports separate timings for candidate generation,
candidate solving, rank deduplication, and total search time.

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
