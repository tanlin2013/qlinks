# Spin-1 Sec. VI optional P1 claim upgrades

**Date:** 2026-08-26
**Frozen P0 source:** `data/evidence_jobs/spin1_sec6_integration_20260825T073925Z/`

P0 is scientifically and numerically closed. This lane contains only optional
claim upgrades selected because they are substantially cheaper than another
`L=14` sparse shift-invert production point. The frozen P0 CSVs and figures are
inputs only and must not be overwritten.

## P1-A: generic-size Jacobian cleanup

Recompute the cage-conditioning illustration at `L=8`, `M=-2`,
`J3/J=0.1`, `kappa/J=0.1`. The job also records the periodic range-three pair
count at `L=8` and contrasts it with the exceptional `L=6` half-ring geometry.
This is an editorial/scientific cleanup, not a new compatibility claim.

Output:

- `spin1_xy_sec6_p1_L8_cage_jacobian_conditioning.csv`
- `spin1_xy_sec6_p1_L8_geometry_audit.json`

No eigensolver is used.

## P1-B: denser sampled positive-kappa grid

Retain the frozen P0 points

`kappa/J = {0.05, 0.10, 0.15, 0.20}`

and add only the midpoint samples

`kappa/J = {0.075, 0.125, 0.175}`

at `L=8,10,12`. Each missing midpoint is one full dense diagonalization and is
checkpointed independently under
`evidence_cache/spin1/sec6_p1_kappa_refinement/`.

The job exports the combined seven-point sampled grid plus adjacent finite
differences for `w_L`, `tau_A`, `tau_Z`, and `tau_Y`. These finite differences
are continuity diagnostics only; they do not establish a literal open
thermodynamic interval.

No sparse solver and no `L=14` route exist in this job.

## P1-C: complete three-site local algebra

At the representative `kappa/J=0.1`, test the complete magnetization-preserving
Hermitian algebra on the contiguous region `(0,1,2)`. The local fixed-charge
block dimensions are `(1,3,6,7,6,3,1)`, so the Hermitian algebra has dimension

`1^2 + 3^2 + 6^2 + 7^2 + 6^2 + 3^2 + 1^2 = 141`.

The stage is deliberately cache-only with respect to spectra. It searches and
validates the existing full dense `L=8,10,12` representative checkpoints, then
constructs/project the 141 local operators and evaluates the same
`W_L = L^(1/4)` block-invariant covariance diagnostic. Missing or incompatible
spectral checkpoints fail explicitly; no eigensolve fallback exists.

Outputs:

- `spin1_xy_sec6_p1_three_site_concentration.csv`
- `spin1_xy_sec6_p1_three_site_worst_eigenoperator.csv`
- `spin1_xy_sec6_p1_three_site_progress.json`

A successful decreasing three-size sequence strengthens the locality evidence
beyond two sites but still does not prove concentration for every bounded
region.

## Docker handoff

Use one explicit P1 run id so every stage writes to the same timestamped output
folder:

```bash
export QLINKS_EVIDENCE_RUN_ID=spin1_sec6_p1_claim_upgrades_20260826T000000Z

scripts/docker/docker_run_spin1_sec6_p1.sh --stage jacobian-l8
scripts/docker/docker_run_spin1_sec6_p1.sh --stage kappa-refinement-status
scripts/docker/docker_run_spin1_sec6_p1.sh --stage kappa-refinement
scripts/docker/docker_run_spin1_sec6_p1.sh --stage three-site-status
scripts/docker/docker_run_spin1_sec6_p1.sh --stage three-site
```

Replace the example timestamp with the actual run timestamp before execution.
The status stages do not start solves. The midpoint refinement stage may start
only the nine declared dense `L<=12` points. The three-site stage contains no
eigensolver and reuses existing spectra.

## Explicitly deferred

Do not include these in the cheap P1 batch:

- nonrepresentative `L=14, kappa/J=0.20` sparse shift-invert point;
- finite-beta deformation grid;
- a forced four-size thermodynamic exponent or lower-bound fit.

Those remain optional future strengthening and do not affect the approved
finite-size ICQMBS claim.
