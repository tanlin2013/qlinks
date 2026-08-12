# Caging-layer agent rules

This file supplements the repository-root `AGENTS.md`.

- Treat cage classification, local search, stability, tensor-network construction, and
  thermodynamic diagnostics as separate responsibilities even when they share data.
- Do not add new responsibilities to `local_search.py` or other existing large modules. Prefer
  a focused module and a compatibility import only while callers migrate.
- `stability.py` is a temporary refactor facade. New implementation code must import the
  focused `stability_core`, `stability_topology`, `stability_boundary`, `stability_qdm`,
  `stability_laurent`, or `stability_types` module directly.
- Local-search data contracts and pure region geometry live in `local_search_types.py` and
  `local_search_geometry.py`; do not move those responsibilities back into `local_search.py`.
- Do not import from `qlinks.open_system`. Extract genuinely shared local algebra or operator
  primitives into a neutral lower layer.
- Keep exact algebraic certification separate from heuristic search ranking and from
  presentation/report generation.
- Every new cage certificate must define its residuals, tolerance semantics, and failure
  modes. A low objective value is not automatically an exact certificate.
- Finite-cluster, cross-size, PEPS optimization, and thermodynamic evidence belong in
  integration or scientific tests, not unit tests.
- Avoid expanding `qlinks.caging.__init__` unless the interface is intentionally supported.
