# Caging-layer agent rules

This file supplements the repository-root `AGENTS.md`.

- Treat environment reduction, local-structure analysis, local search, stability, tensor-network
  construction, and thermodynamic diagnostics as separate responsibilities even when they share
  transition data. Environment reduction is not a classification of the caged eigenstate.
- `analysis/` owns post-search scientific analysis. `transitions` is the shared leaf for
  support-aware local transition patterns; `environment/` determines whether exterior degrees of
  freedom can be removed and is itself split by contracts, support keys, monitor planning, reduced
  operators, zero discovery, mechanism annotation, reporting, and orchestration; `local_structure`,
  `support`, and `support_morphology` analyze the resulting local object; `spectral`,
  `thermodynamic`, and `evidence` own ensemble/spectrum diagnostics. Do not introduce reverse
  imports from `environment` into higher analysis layers or reverse edges inside the environment
  subpackage.
- An exterior probe is safely removable only through one of three physical routes: no exterior
  wavefunction weight, projective annihilation, or the same support-aware local cancellation
  pattern. Merely reaching another known interference zero is insufficient. Pattern comparison
  must include both the local support and the weighted local transition signature.
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
