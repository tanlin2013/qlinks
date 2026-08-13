# Caging-layer agent rules

This file supplements the repository-root `AGENTS.md`.

- Treat cage classification, local search, stability, tensor-network construction, and
  thermodynamic diagnostics as separate responsibilities even when they share data.
- `local_search.py` is now a temporary compatibility facade and must not receive implementation
  logic. First-party code must import the focused local-search modules directly.
- Preserve the local-search dependency DAG rather than a single monolithic chain.
  `local_search_types` is the passive contract leaf; `local_search_geometry` contains pure region
  geometry; `local_search_core` owns generic cage-search algebra; `local_search_qdm` adapts that
  algebra to QDM local regions; `local_search_global` owns explicit global-QDM action primitives;
  `local_search_padding` owns exterior-padding search and structural block validation;
  `local_search_factorized` owns exact factorized-product certification;
  `local_search_certification` owns residual certification/result assembly;
  `local_search_proposals` generates regions; `local_search_scan` executes proposal streams; and
  `local_search_workflows` orchestrates the high-level robust search. Do not add reverse imports
  that recreate a static or eager dependency cycle.
- `stability.py` is a temporary refactor facade. New implementation code must import the
  focused `stability_core`, `stability_topology`, `stability_boundary`, `stability_qdm`,
  `stability_laurent`, or `stability_types` module directly.
- Local-search data/result contracts and pure region geometry live in `local_search_types.py` and
  `local_search_geometry.py`; generic search algebra/adapter registration lives in
  `local_search_core.py`; QDM local-region algebra lives in `local_search_qdm.py`; explicit global
  QDM actions live in `local_search_global.py`; exterior-padding enumeration lives in
  `local_search_padding.py`; exact factorized-product contraction lives in
  `local_search_factorized.py`; residual certification lives in `local_search_certification.py`;
  proposal generation lives in `local_search_proposals.py`; proposal execution lives in
  `local_search_scan.py`; robust portfolio orchestration lives in `local_search_workflows.py`.
- Do not import from `qlinks.open_system`. Extract genuinely shared local algebra or operator
  primitives into a neutral lower layer.
- Keep exact algebraic certification separate from heuristic search ranking and from
  presentation/report generation.
- Every new cage certificate must define its residuals, tolerance semantics, and failure
  modes. A low objective value is not automatically an exact certificate.
- Finite-cluster, cross-size, PEPS optimization, and thermodynamic evidence belong in
  integration or scientific tests, not unit tests.
- Avoid expanding `qlinks.caging.__init__` unless the interface is intentionally supported.
