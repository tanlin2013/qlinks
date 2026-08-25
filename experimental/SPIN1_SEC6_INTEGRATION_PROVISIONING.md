# Spin-1 Sec. VI integration provisioning

This lane integrates the completed `spin1_sec6_provisioning_20260820T052954Z` evidence into the Sec. VI figure contract without repeating established numerical work.

## Execution rule

The default policy is **validate -> reuse -> derive**.

- Validate the completed representative `L=14`, `kappa/J=0.1` concentration result, exact-energy tolerance audit, sparse-budget certification, and the two beta-zero bridges.
- Reuse already exported observables for figure panels whenever their window protocol is explicitly identifiable.
- Reuse spectral checkpoints for any new complete 19-operator covariance calculation.
- Do not invoke an eigensolver as an implicit fallback. Missing reusable spectra are reported as provisioning gaps.
- In particular, a missing homogeneous `L^{1/4}` concentration sequence must not cause a new `L=14` shift-invert solve.

The integration code intentionally lives outside the already-large `spin1_sec6_provisioning.py` module so manuscript/figure assembly does not add another responsibility to the numerical kernel.

## Step 1: validate and export reusable figure data

```bash
python experimental/jobs/spin1_sec6_integration.py \
  --source-data-dir experimental/data/evidence_jobs/spin1_sec6_provisioning_20260820T052954Z \
  --output-dir experimental/data/evidence_jobs/spin1_sec6_integration_$(date -u +%Y%m%dT%H%M%SZ)
```

This checks the already-established representative evidence against the Sec. VI handoff values and writes `spin1_xy_sec6_integration_audit.json`. It also exports stable panel/appendix CSVs that can be obtained purely by validated post-processing.

A legacy deformation or concentration grid is **not** accepted for Fig. 6(c)/(d) unless the table itself certifies the primary `W_L(gamma=1/4,c=1)` window. Such products are recorded as pending rather than silently mixing window conventions.

## Step 2: compute P0-A from reusable spectra only

```bash
python experimental/jobs/spin1_sec6_common_windows.py \
  --checkpoint-root experimental/data/evidence_cache/spin1 \
  --checkpoint-root experimental/data/evidence_jobs/spin1_sec6_provisioning_20260820T052954Z \
  --existing-data-dir experimental/data/evidence_jobs/<previous-integration-run> \
  --output-dir experimental/data/evidence_jobs/<integration-run>
```

Before inspecting eigensystems, the reducer checks `--existing-data-dir` (or its own output directory) for a complete common-window export. A completed export is reused only after validating its protocol half-widths, residual/coverage fields, companion audits, and the independently established `L=14`, `Delta E=1` raw/clean widths.

The cache-only calculation targets, at `kappa/J=0.1`, both homogeneous protocols:

- `quarter_power_c1`: `Delta E=L^(1/4)`;
- `fixed_width_1`: `Delta E=1`.

For each `L=8,10,12,14` it exports raw and cleaned complete 19-operator covariance widths, median nonidentity width, worst eigenoperator coefficients, state/rank counts, exact-energy block counts, and residual/checkpoint audit fields.

Before any covariance is computed, the selected eigensystem must pass:

1. scientific metadata compatibility (`L`, fixed `M`, `J3/J`, `kappa/J`);
2. array shape and finiteness checks;
3. deterministic sample orthogonality;
4. deterministic sampled physical eigenpair residuals against the reconstructed resolved-sector Hamiltonian;
5. spectral coverage of the requested window.

If a size has no reusable validated eigensystem, the script reports `MISSING_REUSABLE_SPECTRUM`; it does not solve it.

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

No thermodynamic power-law exponent is fitted by this lane. The summary only records whether the raw widths narrow qualitatively under each homogeneous window protocol.

## Step 3: render figures from stable CSVs

Once the common-window concentration table and the same-window deformation/family products are present:

```bash
python experimental/jobs/render_spin1_xy_sec6_integration_figures.py \
  --data-dir experimental/data/evidence_jobs/<integration-run> \
  --use-tex
```

The strict renderer expects the sampled positive-`kappa` concentration band for Fig. 6(d). `--allow-incomplete` may be used for a preview only; it must not be treated as the final figure.

Expected outputs are:

```text
figures/spin1_xy_figure6_prx.svg
figures/spin1_xy_figure6_prx.pdf
figures/spin1_xy_figure6_prx_preview.png
figures/spin1_xy_appendix_concentration_windows.svg
figures/spin1_xy_appendix_concentration_windows.pdf
figures/spin1_xy_appendix_beta0_bridges.svg
figures/spin1_xy_appendix_beta0_bridges.pdf
figures/spin1_xy_appendix_complex_t2_obstruction.svg
figures/spin1_xy_appendix_complex_t2_obstruction.pdf
figures/spin1_xy_figure6_prx_audit.json
figures/spin1_xy_figure6_prx_audit.md
```

## Remaining numerical gaps

The code distinguishes scientific evidence already established from genuinely new products.

- Representative `L=14`, fixed `Delta E=1` concentration: validate only.
- `L=14` beta-zero bridge decomposition: validate/use only.
- Homogeneous primary-window concentration: derive only if absent, from reusable spectra.
- Positive-`kappa` `L=12` witness curve under the same primary window: pending unless the source table explicitly certifies that protocol.
- Positive-`kappa` concentration band through `L=12`: pending unless same-window covariance products are available.
- Nonrepresentative `L=14`, preferably `kappa/J=0.20`: remains P1 and is not started by this lane.

This keeps the manuscript claim boundary unchanged while preventing an integration/plotting rerun from becoming an accidental week-scale spectral calculation.
