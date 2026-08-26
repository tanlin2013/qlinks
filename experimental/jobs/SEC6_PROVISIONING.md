# Spin-1 XY Sec. VI remaining P0 provisioning

This companion runbook implements the 2026-08-26 Sec. VI handoff. The
representative `kappa/J=0.1` common-window evidence is already complete in
`data/evidence_jobs/spin1_sec6_integration_20260825T073925Z/`; do not rerun it.

## Closed evidence

The integration cache already contains and validates:

- representative `W_L(gamma=1/4,c=1)` and fixed `Delta E=1` complete two-site
  covariance for `L=8,10,12,14`;
- Fig. 6(a) representative raw ETH scatter;
- Fig. 6(b) representative raw witness sequence;
- Appendix-D beta-zero bridge data;
- Appendix-D complex-`t2` obstruction data;
- the certified `L=14` sparse-budget and exact-energy tolerance audits.

The old integration-audit JSON predates the successful common-window pass. The
first remaining action is therefore bookkeeping, not another eigensolve.

## Remaining P0 numerical scope

The only new numerical grid is

- `L in {8,10,12}`;
- `kappa/J in {0.05,0.10,0.15,0.20}`;
- primary window `W_L(gamma=1/4,c=1)`.

The `kappa/J=0.1` rows are reused from the completed representative cache. Thus
at most nine nonrepresentative dense points remain. Each new point uses one
full dense diagonalization and derives both the raw `A/Z/Y` microcanonical row
and the complete 19-operator raw covariance from the same eigensystem.

This P0 lane contains no sparse solver and no `L=14` solve. A nonrepresentative
`L=14` point remains P1.

## Resume discipline

The deformation-grid job writes a validated checkpoint after every `(L,kappa)`
point under

`experimental/data/evidence_cache/spin1/sec6_deformation_grid/`.

It also rewrites the aggregate row cache, Fig. 6(c) source table, Fig. 6(d)
family-band table, worst-eigenoperator table, and progress manifest atomically
after every completed point. A rendering failure cannot invalidate numerical
checkpoints.

The status stage never starts an eigensolver. The compute stage is the only
explicit opt-in to the nine possible `L<=12` dense solves.

## Server sequence

Reuse the successful integration run id throughout:

```bash
export QLINKS_EVIDENCE_RUN_ID=spin1_sec6_integration_20260825T073925Z
```

Refresh the stale audit without solving:

```bash
QLINKS_NUM_THREADS=16 \
  scripts/docker/docker_run_spin1_sec6_integration.sh --stage audit
```

Optionally inventory reusable/pending grid points without solving:

```bash
QLINKS_NUM_THREADS=16 \
  scripts/docker/docker_run_spin1_sec6_integration.sh \
  --stage deformation-grid-status
```

Compute only the missing nonrepresentative dense `L<=12` points:

```bash
QLINKS_NUM_THREADS=16 \
  scripts/docker/docker_run_spin1_sec6_integration.sh \
  --stage deformation-grid
```

The same command is resumable: validated point checkpoints are reused on a
rerun. After it completes, refresh the audit again:

```bash
QLINKS_NUM_THREADS=16 \
  scripts/docker/docker_run_spin1_sec6_integration.sh --stage audit
```

Then render the final main and Appendix-D figures from frozen CSVs only:

```bash
QLINKS_NUM_THREADS=16 \
  scripts/docker/docker_run_spin1_sec6_integration.sh --stage render-final
```

## Expected remaining-P0 exports

Numerical/grid products:

- `spin1_xy_sec6_deformation_grid_rows.csv`;
- `spin1_xy_sec6_deformation_grid_worst_eigenoperators.csv`;
- `spin1_xy_figure6_panel_c_deformation.csv`;
- `spin1_xy_figure6_panel_d_family_band.csv`;
- `spin1_xy_sec6_deformation_grid_progress.json`;
- refreshed `spin1_xy_sec6_integration_audit.json`.

Figure products:

- `spin1_xy_figure6_prx.svg`;
- `spin1_xy_figure6_prx.pdf`;
- `spin1_xy_figure6_prx_preview.png`;
- `spin1_xy_figure6_prx_audit.json` and `.md`;
- `spin1_xy_appendix_concentration_windows.svg/pdf`;
- `spin1_xy_appendix_beta0_bridges.svg/pdf`;
- `spin1_xy_appendix_complex_t2_obstruction.svg/pdf`.

## Claim discipline

P0 supports representative complete two-site concentration through `L=14` and
the sampled positive-`kappa` family band through `L=12`. It does not establish a
concentration exponent, a full `L=14` deformation-wide envelope, concentration
for every bounded region, or an open `kappa` interval without further
continuity/grid control.
