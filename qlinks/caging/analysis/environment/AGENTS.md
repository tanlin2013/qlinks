# Environment-reduction sublayer agent rules

This file supplements the repository-root and `qlinks/caging/AGENTS.md` rules.

- This package diagnoses whether the exterior environment can be removed when constructing a
  bounded local caging operator. It is not a taxonomy of caged eigenstates.
- Preserve the ownership DAG: `contracts` owns passive configuration/probe/monitor contracts;
  `support` owns support-key extraction; `monitor` owns reduced-IZ monitor selection/grouping;
  `operator` owns reduced local-operator application; `discovery` owns zero/probe discovery;
  `mechanisms` owns per-probe and collective removal-mechanism annotation; `summary` owns compact
  count/norm summaries; `report` owns the user-facing report/readout; and `diagnosis` orchestrates
  the full analysis.
- Do not move transition-pattern semantics out of `qlinks.caging.analysis.transitions`. The
  environment layer consumes those support-aware weighted signatures; it must not redefine them.
- A non-projective exterior target is safely removable through local cancellation only when its
  support-aware weighted local transition pattern matches the source probe. Reaching a known
  interference zero alone is insufficient.
- `qlinks.caging.analysis.environment` is a curated workflow API. Internal first-party code should
  import defining child modules when it needs implementation-specific helpers.
- Keep collective-cancellation search separate from reduced local-operator application. Numerical
  optimization or heuristic ranking must not leak into the exact per-probe mechanism contracts.
- New mechanism labels or changes to tolerances/closure criteria are scientific changes and require
  dedicated tests plus an explicit manuscript/evidence rationale; they are not structural refactors.
