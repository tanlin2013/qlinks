# Caging-layer agent rules

This file supplements the repository-root `AGENTS.md`.

- Treat cage classification, local search, stability, tensor-network construction, and
  thermodynamic diagnostics as separate responsibilities even when they share data.
- `local_search.py` is now a temporary compatibility facade and must not receive implementation
  logic. First-party code must import the focused local-search modules directly.
- Preserve the local-search import direction: `local_search_core` <- `local_search_qdm` <-
  `local_search_certification` <- `local_search_proposals` <- `local_search_workflows`, with
  `local_search_types`/`local_search_geometry` serving as lower-level contracts/helpers. A higher
  layer may depend on lower layers; do not introduce reverse imports that recreate the monolith.
- `stability.py` is a temporary refactor facade. New implementation code must import the
  focused `stability_core`, `stability_topology`, `stability_boundary`, `stability_qdm`,
  `stability_laurent`, or `stability_types` module directly.
- Local-search data contracts and pure region geometry live in `local_search_types.py` and
  `local_search_geometry.py`; generic search algebra/adapter registration lives in
  `local_search_core.py`; QDM local-region algebra lives in `local_search_qdm.py`; padding and
  exact certification live in `local_search_certification.py`; proposal generation/scan logic
  lives in `local_search_proposals.py`; robust portfolio orchestration lives in
  `local_search_workflows.py`.
- Do not import from `qlinks.open_system`. Extract genuinely shared local algebra or operator
  primitives into a neutral lower layer.
- Keep exact algebraic certification separate from heuristic search ranking and from
  presentation/report generation.
- Every new cage certificate must define its residuals, tolerance semantics, and failure
  modes. A low objective value is not automatically an exact certificate.
- Finite-cluster, cross-size, PEPS optimization, and thermodynamic evidence belong in
  integration or scientific tests, not unit tests.
- Avoid expanding `qlinks.caging.__init__` unless the interface is intentionally supported.
