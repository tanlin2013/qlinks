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
