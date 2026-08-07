# Caging-layer agent rules

This file supplements the repository-root `AGENTS.md`.

- Treat cage classification, local search, stability, tensor-network construction, and
  thermodynamic diagnostics as separate responsibilities even when they share data.
- Do not add new responsibilities to `local_search.py`, `stability.py`, or other existing
  large modules. Prefer a focused module and a compatibility import when necessary.
- Do not import from `qlinks.open_system`. Extract genuinely shared local algebra or operator
  primitives into a neutral lower layer.
- Keep exact algebraic certification separate from heuristic search ranking and from
  presentation/report generation.
- Every new cage certificate must define its residuals, tolerance semantics, and failure
  modes. A low objective value is not automatically an exact certificate.
- Finite-cluster, cross-size, PEPS optimization, and thermodynamic evidence belong in
  integration or scientific tests, not unit tests.
- Avoid expanding `qlinks.caging.__init__` unless the interface is intentionally supported.
