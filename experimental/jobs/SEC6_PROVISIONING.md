# Spin-1 XY Sec. VI P0 provisioning

This companion workflow implements the 2026-08-19 Sec. VI handoff without
repeating the completed 10000-eigenpair convergence solve.

## Scientific roles

- **P0.0** repairs the exported sparse-convergence bookkeeping and writes a
  spectral checkpoint immediately after each newly required shift-invert solve.
- **P0.1** computes the complete block-invariant 19-operator covariance at
  `L=14`, `kappa/J=0.1`, using the contained `Delta E=1` window. Raw covariance
  is primary; the joint-dark-cleaned covariance and removed fraction are
  exported as finite-size diagnostics.
- **P0.2** keeps raw microcanonical expectation values primary and resolves the
  two local ensemble bridges
  `rho_mc^(M,k) <-> rho_beta0^(M,k) <-> rho_beta0^M`. The reduced-density-matrix
  residual is expanded in the same Hilbert--Schmidt-orthonormal 19-operator
  basis used by the covariance test.
- **P0.3** is an explicit follow-up at `L=14`, `kappa/J=0.20`. It is not enabled
  by default, so the family solve cannot be started accidentally before the
  representative concentration output has been inspected.

No workflow in this companion schedules `L=16`.

## Production run

The Docker wrapper pins the authoritative dense baseline and sparse-convergence
addendum by default:

```bash
QLINKS_NUM_THREADS=16 QLINKS_DOCKER_MEMORY_LIMIT=400g \
  scripts/docker/docker_run_spin1_sec6_provisioning.sh \
    --stage compute --timeout -1
```

Follow the printed `docker logs -f ...` command. Newly computed `L=14`
eigenvectors are stored uncompressed under the run's `checkpoints/` directory
before covariance, fitting, or rendering starts. A matching checkpoint is
reused automatically on a rerun.

After P0.1 is secure, rerun the **same data directory** with the family flag:

```bash
QLINKS_NUM_THREADS=16 QLINKS_DOCKER_MEMORY_LIMIT=400g \
  scripts/docker/docker_run_spin1_sec6_provisioning.sh \
    --stage compute \
    --data-dir experimental/data/evidence_jobs/<P0_RUN_ID> \
    --run-family-l14 --timeout -1
```

Because the data directory is unchanged, the representative 8192-eigenpair
checkpoint is reused; only the new `kappa/J=0.20` sparse solve is required.

Render only after the numerical products are complete:

```bash
scripts/docker/docker_run_spin1_sec6_provisioning.sh \
  --stage render \
  --source-data-dir experimental/data/evidence_jobs/<P0_RUN_ID> \
  --use-tex --figure-formats pdf,svg
```

## Main exports

The workflow writes, among other provenance tables:

- `spin1_xy_kappa0p1_L14_sparse_convergence.csv` (repaired/normalized copy);
- `spin1_xy_kappa0p1_concentration_L14.csv`;
- `spin1_xy_kappa0p1_concentration_L14_tolerance_audit.csv`;
- `spin1_xy_kappa0p1_worst_eigenoperator_L14.csv`;
- `spin1_xy_kappa0p1_concentration.csv` and its fit file;
- `spin1_xy_kappa0p1_microcanonical_windows_sec6.csv` and raw-MC fit table;
- `spin1_xy_kappa0p1_two_bridge_rdm_distance.csv`;
- `spin1_xy_kappa0p1_residual_operator_spectrum.csv`;
- `spin1_xy_kappa0p1_residual_operator_coefficients.csv`;
- `spin1_xy_kappa_matching_large_size_safe_window.csv`;
- `spin1_xy_large_size_family_concentration.csv` after the P0.3 follow-up.

The existing shared Spin-1 renderer consumes these tables together with the
copied baseline products. It can therefore add the solver-certified `L=14`
point to panel (b), and only adds it to panel (d) when the concentration table
is present and the sparse-budget certification passes.
