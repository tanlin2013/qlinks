# Spin-1 Sec. VI optional P1 claim upgrades

**Updated:** 2026-09-03
**Frozen historical P0 source:** `data/evidence_jobs/spin1_sec6_integration_20260825T073925Z/`
**Permanent convention:** `J_over_2_ladder_v1`

P0 is scientifically and numerically closed. This lane contains only optional claim
upgrades selected because they are much cheaper than another `L=14` sparse
shift-invert production point.

The August P0/P1 folders are immutable historical inputs. Current reuse must proceed
through the explicit exchange-convention migration layer. The permanent kinetic
normalization is

\[
H_{XY}=J\sum(S^xS^x+S^yS^y)
=\frac J2\sum(S^+S^-+S^-S^+).
\]

For `h=D=0`, historical P1 observables are mapped with `E_current=E_legacy/2`,
unchanged eigenvectors and normalized covariance/witness quantities, and halved energy
windows.

## P1-A: generic-size Jacobian calibration

Recompute the cage-conditioning illustration at `L=8`, `M=-2`, `J3/J=0.1`,
`kappa/J=0.1` using the current Hamiltonian. The job also records the periodic
range-three pair count at `L=8` and contrasts it with the exceptional `L=6` half-ring
geometry.

Outputs:

- `spin1_xy_sec6_p1_L8_cage_jacobian_conditioning.csv`
- `spin1_xy_sec6_p1_L8_geometry_audit.json`

No eigensolver is used. The purely energy-dimensional/interference gap is expected to
reflect the factor-one-half Hamiltonian change, but the full mixed-coordinate Jacobian
conditioning is **remeasured**, not obtained by blindly dividing the historical value
by two.

## P1-B: denser sampled positive-kappa grid

Retain the mapped P0 points

`kappa/J = {0.05, 0.10, 0.15, 0.20}`

and add only the midpoint samples

`kappa/J = {0.075, 0.125, 0.175}`

at `L=8,10,12` if a current checkpoint is genuinely missing.

The permanent primary window is

\[
\Delta E=(J/2)L^{1/4},
\]

with protocol `quarter_power_c0p5` and prefactor `0.5` at displayed `J=1`.

New midpoint checkpoints are schema-v2/current-convention. A legacy midpoint
checkpoint is rejected and must be handled through explicit convention migration;
it is not treated as a cache miss that automatically authorizes a new solve.

The job exports the combined seven-point sampled grid plus adjacent finite differences
for `w_L`, `tau_A`, `tau_Z`, and `tau_Y`. These finite differences are continuity
diagnostics only; they do not establish a literal open thermodynamic interval.

No sparse solver and no `L=14` route exist in this job.

## P1-C: complete three-site local algebra

At representative `kappa/J=0.1`, test the complete magnetization-preserving Hermitian
algebra on contiguous sites `(0,1,2)`. The local fixed-charge block dimensions are
`(1,3,6,7,6,3,1)`, hence the Hermitian algebra dimension is

`1^2 + 3^2 + 6^2 + 7^2 + 6^2 + 3^2 + 1^2 = 141`.

This stage is solver-free with respect to spectra. It validates/reuses existing full
dense `L=8,10,12` representative checkpoints and evaluates the same complete local
covariance diagnostic in the current primary window

\[
\Delta E=(J/2)L^{1/4}.
\]

A completed historical three-site row may be mapped exactly in memory: normalized
`w_L` is invariant, while its window and energy-dimensional residual diagnostics are
divided by two. Missing or incompatible spectral checkpoints fail explicitly; no
eigensolve fallback exists.

Outputs:

- `spin1_xy_sec6_p1_three_site_concentration.csv`
- `spin1_xy_sec6_p1_three_site_worst_eigenoperator.csv`
- `spin1_xy_sec6_p1_three_site_progress.json`

A successful decreasing three-size sequence strengthens the locality evidence beyond
two sites but still does not prove concentration for every bounded region.

## Convention-migration handoff

The preferred first step is the dedicated one-time migration runner:

```bash
export QLINKS_EVIDENCE_RUN_ID=spin1_exchange_convention_migration_20260903T000000Z

scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage migrate-p0
scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage migrate-p1
scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage validate
scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage jacobian-l8
```

The default historical P1 source is
`spin1_sec6_p1_claim_upgrades_20260827T055013Z`. The mapped P1 products are written
under the new migration run and are never written back into the historical folder.

If future P1 computations are needed after migration, use the existing P1 runner only
with current/mapped inputs and convention-stamped caches. The status stages remain
no-solve; midpoint refinement may start only the declared dense `L<=12` points; the
three-site stage contains no eigensolver.

## Explicitly deferred

Do not turn the normalization migration into any of these optional projects:

- nonrepresentative `L=14, kappa/J=0.20` sparse shift-invert point;
- finite-beta deformation grid;
- a forced four-size thermodynamic exponent or lower-bound fit.

They remain optional future strengthening and do not affect the approved finite-size
ICQMBS claim or the exchange-normalization correction.
