# Benchmark scripts

These scripts are for local performance profiling and are not part of the unit-test suite.

## Basis generation

```bash
python scripts/benchmark_basis.py
python scripts/benchmark_basis.py --only spin_one
python scripts/benchmark_basis.py --json basis_benchmark.json
```

## Hamiltonian construction

```bash
python scripts/benchmark_hamiltonian.py
python scripts/benchmark_hamiltonian.py --split-basis-timing
python scripts/benchmark_hamiltonian.py --only qdm --split-basis-timing
python scripts/benchmark_hamiltonian.py --json hamiltonian_benchmark.json
```

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

You can monitor with:
```bash
find ./data/qlinks_cage_sweep_full/jobs -name status.json \
  -exec jq -r '[.status, .job_id, (.n_states // ""), (.n_records // "")] | @tsv' {} \;
```

Counts:
```bash
find ./data/qlinks_cage_sweep_full/jobs -name status.json \
  -exec jq -r '.status' {} \; | sort | uniq -c
```
