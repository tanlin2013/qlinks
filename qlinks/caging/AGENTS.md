# Caging-layer agent rules

This file supplements the repository-root `AGENTS.md`.

- Treat cage classification, local search, stability, tensor-network construction, and
  thermodynamic diagnostics as separate responsibilities even when they share data.
- `local_search/` is the local-cage-search sublayer. Preserve its dependency DAG:
  `types` is the passive contract leaf; `geometry` contains pure region geometry; `core` owns
  generic cage-search algebra; `qdm` adapts that algebra to QDM local regions; `global_ops` owns
  explicit global-QDM actions; `padding` owns exterior-padding search; `factorized` owns exact
  factorized-product certification; `certification` owns residual certification/result assembly;
  `proposals` generates regions; `scan` executes proposal streams; and `workflows` orchestrates
  high-level robust search. Do not add reverse imports that recreate a static or eager cycle.
- `stability/` is the cage-stability sublayer. Use `core`, `topology`, `boundary`, `qdm`,
  `laurent`, `symmetry`, and `types` according to scientific responsibility rather than adding
  another broad stability facade.
- `qlinks.caging.local_search` and `qlinks.caging.stability` are curated subpackage APIs. New
  first-party implementation code should still import the defining child module directly; use the
  subpackage API at workflow/user boundaries.
- Do not import from `qlinks.open_system`. Extract genuinely shared local algebra or operator
  primitives into a neutral lower layer.
- Keep exact algebraic certification separate from heuristic search ranking and from
  presentation/report generation.
- Every new cage certificate must define its residuals, tolerance semantics, and failure
  modes. A low objective value is not automatically an exact certificate.
- Finite-cluster, cross-size, PEPS optimization, and thermodynamic evidence belong in
  integration or scientific tests, not unit tests.
- Avoid expanding `qlinks.caging.__init__` unless the interface is intentionally supported.
