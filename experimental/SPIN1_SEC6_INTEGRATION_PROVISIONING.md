# Spin-1 Sec. VI integration provisioning

**Updated:** 2026-09-03
**Permanent convention:** `J_over_2_ladder_v1`

This lane now integrates **convention-mapped derived evidence**, not the historical
August folders directly. The August evidence remains immutable and is converted by the
explicit Spin-1 exchange-normalization migration runner.

## Exchange and window contract

Use

\[
H_{XY}=J\sum_r(S_r^xS_{r+1}^x+S_r^yS_{r+1}^y)
=\frac J2\sum_r(S_r^+S_{r+1}^-+S_r^-S_{r+1}^+).
\]

The permanent homogeneous windows at displayed `J=1` are:

- `quarter_power_c0p5`: `Delta E=0.5 L^(1/4)`;
- `fixed_width_0p5`: `Delta E=0.5`.

These select exactly the same `h=D=0` eigenstate sets as the historical
`quarter_power_c1` / `fixed_width_1` products after the exact mapping
`E_current=E_legacy/2`.

## Execution rule

The policy is **map -> validate -> reuse -> derive**.

- Never overwrite the historical timestamped P0/P1 evidence.
- Require explicit `spin1_xy_exchange_convention=J_over_2_ladder_v1` provenance on
  current figure-data/common-window products.
- Reuse historical spectral eigenvectors only through the explicit factor-one-half
  energy mapping and physical residual validation.
- Do not invoke an eigensolver as an implicit fallback.
- Never launch another `L=14` solve merely to rebuild Sec. VI integration products.
- The renderer consumes already-mapped energy-density values and does not rescale them
  again.

## Preferred one-time migration workflow

After the migration PR is merged, use the dedicated Docker runner with one timestamped
run ID:

```bash
export QLINKS_EVIDENCE_RUN_ID=spin1_exchange_convention_migration_20260903T000000Z
export QLINKS_NUM_THREADS=16

scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage status
scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage migrate-p0
scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage migrate-p1
scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage validate
scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage jacobian-l8
scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage render-p0
```

The defaults are the frozen historical sources

```text
P0: spin1_sec6_integration_20260825T073925Z
P1: spin1_sec6_p1_claim_upgrades_20260827T055013Z
```

and derived products are written under the new migration run as `p0/`, `p1/`, and
`validation/`.

The `validate` stage contains only cheap matrix/shell checks and, by default, a dense
`L=8` scaling spot check. An optional `L=10` spot check is requested with

```bash
QLINKS_SPIN1_CONVENTION_DENSE_SIZES=8,10
```

for that stage. No `L=14` solver route exists in this runner.

## Current integration adapter

`experimental/jobs/spin1_sec6_integration.py` is the active current-convention entry
point. It refuses an unstamped historical source directory. A source directory must
contain `spin1_exchange_convention_migration_manifest.json` declaring the permanent
convention.

The adapter uses the preserved August integration formatter internally but replaces
its primary-window selection by the current `c=1/2` contract and stamps every newly
written figure-data CSV.

Representative figure products include:

```text
spin1_xy_figure6_panel_a_scatter.csv
spin1_xy_figure6_panel_b_witness_sequence.csv
spin1_xy_figure6_panel_c_deformation.csv
spin1_xy_figure6_panel_d_family_band.csv
spin1_xy_kappa0p1_concentration_common_windows.csv
spin1_xy_appendix_beta0_bridges_data.csv
spin1_xy_appendix_complex_t2_obstruction_data.csv
```

A legacy/untagged deformation or concentration table is reported as pending rather
than silently reused.

## Common-window reducer

`experimental/jobs/spin1_sec6_common_windows.py` is cache-only. It targets both current
protocols at `kappa/J=0.1` and `L=8,10,12,14`.

For a historical checkpoint the reducer:

1. validates the scientific metadata and array shapes;
2. maps stored eigenvalues and energy-dimensional metadata by `1/2`;
3. keeps eigenvectors unchanged;
4. reconstructs the current resolved-sector Hamiltonian;
5. validates mapped in-window eigenpair residuals and coverage;
6. evaluates/reuses the complete 19-operator covariance diagnostics.

A missing or incompatible reusable spectrum yields an explicit
`MISSING_REUSABLE_SPECTRUM`-type failure. No eigensolver fallback exists in this lane.

Primary output:

```text
spin1_xy_kappa0p1_concentration_common_windows.csv
```

Supporting outputs:

```text
spin1_xy_kappa0p1_common_window_checkpoint_audit.csv
spin1_xy_kappa0p1_common_window_worst_eigenoperator.csv
spin1_xy_kappa0p1_common_window_tolerance_audit.csv
spin1_xy_kappa0p1_common_window_summary.json
```

All current outputs are convention-stamped. No thermodynamic power-law exponent is
fitted by this lane.

## Positive-kappa deformation grid

The active deformation-grid adapter keeps the existing solver boundary:

- only `L=8,10,12`;
- only the declared positive-kappa grid;
- full dense diagonalization only for explicitly missing points;
- no sparse solver and no `L=14` route.

New point checkpoints are schema-v2/current-convention. A legacy point checkpoint is
an explicit migration/provenance error rather than a silent cache miss.

## Rendering

The current renderer accepts only convention-stamped figure data and labels the
window controls as

\[
\Delta E=(J/2)L^{1/4},\qquad \Delta E=J/2.
\]

Run directly on the mapped P0 directory, or use the migration runner's `render-p0`
stage. Expected outputs remain the Fig. 6 PRX files and Appendix-D support figures,
plus the JSON/Markdown rendering audit.

The audit records that energy density was **not** rescaled inside the renderer, which
prevents accidental double application of the factor `1/2`.

## Numerical/claim boundary

P0 remains closed. The normalization migration should not be turned into a new
production campaign. Optional future work remains optional scientific strengthening,
not a prerequisite for the normalization fix:

- nonrepresentative `L=14, kappa/J=0.20`;
- finite-beta deformation grid;
- stronger bounded-region locality tests;
- any additional thermodynamic fitting.

This separation keeps the approved finite-size Sec. VI claim unchanged while making
its Hamiltonian normalization conventional and mechanically auditable.
