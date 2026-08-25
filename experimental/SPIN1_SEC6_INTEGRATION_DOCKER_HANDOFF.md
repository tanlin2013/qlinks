# Spin-1 Sec. VI cache-only Docker handoff

Use this launcher after the completed Sec. VI production evidence is available on the server:

```bash
scripts/docker/docker_run_spin1_sec6_integration.sh --stage audit
```

The default source is `experimental/data/evidence_jobs/spin1_sec6_provisioning_20260820T052954Z`. The launcher creates a timestamped `spin1_sec6_integration_*` evidence directory and invokes only the post-processing integration code. It does not invoke `run_spin1_xy_sec6_provisioning.py` or any eigensolver entry point.

Read the resolved run id printed by the first command and reuse it for later stages:

```bash
export QLINKS_EVIDENCE_RUN_ID=spin1_sec6_integration_<timestamp>

scripts/docker/docker_run_spin1_sec6_integration.sh --stage common-windows
```

`common-windows` first reuses a completed derived export when valid. Otherwise it searches the stable Spin-1 evidence cache and the source evidence run for compatible eigensystems, validates them, and performs only the complete two-site covariance reduction. If a required eigensystem is missing or does not cover the requested window, the stage fails with a cache gap; no solve is started.

Rendering is deliberately separate:

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

For the first server pass, run only `--stage audit` and inspect `spin1_xy_sec6_integration_audit.json`. That audit determines whether P0-A can be derived entirely from the existing cache and which Fig. 6/Appendix-D products are genuinely still missing. Do not run `common-windows` until that audit has been inspected.
