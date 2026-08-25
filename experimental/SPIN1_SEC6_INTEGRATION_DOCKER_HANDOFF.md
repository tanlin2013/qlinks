# Spin-1 Sec. VI integration Docker handoff

Use this launcher after the completed Sec. VI production evidence is available on the server:

```bash
scripts/docker/docker_run_spin1_sec6_integration.sh --stage audit
```

The default source is `experimental/data/evidence_jobs/spin1_sec6_provisioning_20260820T052954Z`. The launcher creates a timestamped `spin1_sec6_integration_*` evidence directory and keeps all later stages under the same run id.

Read the resolved run id printed by the first command and reuse it:

```bash
export QLINKS_EVIDENCE_RUN_ID=spin1_sec6_integration_<timestamp>
```

## Audit and cache-only reduction

The normal path is:

```bash
scripts/docker/docker_run_spin1_sec6_integration.sh --stage common-windows
```

`common-windows` first reuses a completed derived export when valid. Otherwise it searches the stable Spin-1 evidence cache and the source evidence run for compatible eigensystems, validates them, and performs only the complete two-site covariance reduction. It never solves implicitly.

The established Aug-25 audit showed that the representative `L=14` sparse checkpoint is reusable, but the old Sec. VI workflow did not persist its dense `L=8,10,12` eigensystems. Those sizes were historically recomputed with dense `scipy.linalg.eigh` on every run. Therefore the cache-only reducer correctly reports:

```text
missing validated reusable spectra for L=8,10,12; no eigensolve was started
```

## One-time dense-cache seed

For that specific gap, run:

```bash
scripts/docker/docker_run_spin1_sec6_integration.sh --stage seed-dense-cache
```

This stage is deliberately narrow:

- fixed `kappa/J=0.1`;
- exactly `L=8,10,12`;
- full dense diagonalization with the same resolved-sector Hamiltonian used by the established Sec. VI workflow;
- atomic `energies.npy`, `vectors.npy`, and `metadata.json` writes under `experimental/data/evidence_cache/spin1/sec6_dense/`;
- immediate cached-spectrum validation using sampled orthogonality and physical eigenpair residuals;
- incremental audit output so completed smaller sizes survive a later-size failure;
- no sparse solver path and no `L=14` solve.

The stage writes:

```text
spin1_xy_sec6_dense_cache_seed_audit.csv
spin1_xy_sec6_dense_cache_seed_summary.json
```

If a compatible validated checkpoint already exists with enough spectral coverage, it is reused instead of recomputed.

After the seed completes, rerun:

```bash
scripts/docker/docker_run_spin1_sec6_integration.sh --stage common-windows
```

The reducer should then combine the newly persisted dense `L=8,10,12` spectra with the already-established reusable `L=14` sparse spectrum and export the homogeneous common-window concentration products.

## Rendering

Rendering remains separate:

```bash
scripts/docker/docker_run_spin1_sec6_integration.sh --stage render-preview
scripts/docker/docker_run_spin1_sec6_integration.sh --stage render-final
```

`render-preview` relaxes only the positive-kappa family-band requirement. Other required panel inputs must still exist. Use `render-final` only after the integration audit and derived-data products show that all strict Fig. 6 inputs are present.

Each stage gets its own Docker container name while all stages with the same `QLINKS_EVIDENCE_RUN_ID` share one evidence output directory. The repository mount is read-only; only `experimental/data` is writable.

To point at a different completed source evidence run, set:

```bash
QLINKS_SEC6_SOURCE_RUN_ID=<source-run-id> \
  scripts/docker/docker_run_spin1_sec6_integration.sh --stage audit
```

The heavy production provisioning entry point remains absent from this integration launcher. The only solve-capable integration stage is the explicitly named, fixed-size `seed-dense-cache` step described above.
