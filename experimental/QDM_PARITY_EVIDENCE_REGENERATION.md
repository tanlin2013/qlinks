# Parity-aware QDM evidence regeneration

The PR151 server bundle reconciles the old Fig. 9 spectrum and energy-block
A/Z witness traces. Individual diagonals in degenerate blocks can rotate. The
new workflow therefore regenerates the population independently and measures
the compact target's exact shifted-reflection irrep before selecting an ensemble.
The provenance verdict now separates invariant agreement from row equality;
invariant agreement does not certify the original eigenvectors or other operators.

## The target parity depends on strip length

All rows use electric winding (0,0), lambda=1, Tdiag=i, Ty2=+1, and Ly=4.

| Lx | Translation dimension | Sy=+1 | Sy=-1 | Measured target parity | Target dimension |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 15 | 14 | 1 | -1 | 1 |
| 8 | 1125 | 875 | 250 | +1 | 875 |
| 12 | 114483 | 74816 | 39667 | -1 | 39667 |

Parity is measured from the translation-projected target, not inferred from
PCA bands or hard-coded as an alternating formula. An independent measurement
closes the finite compact product support under translations and reflections
(16, 64, 256 configurations). It needs no full dimer basis or diagonalization.
The target Sy residual at Lx=12 is below 1e-16.

`qdm_character_dimensions.py` counts fixed dimer coverings using exact-cover
dynamic programming with signed electric cut winding zero. Integer character
traces give both Sy multiplicities without constructing the Hilbert space.
The 4x4 and 8x4 multiplicities agree with explicitly constructed sparse sectors;
production context construction checks the 12x4 multiplicity against this count.

The target 12x4 block is 34.65% of the old dimension. At budget 1024, storing
complex128 eigenvectors takes approximately 0.61 GiB rather than 1.75 GiB.
This is a storage/dimension improvement, not a measured solver speedup. Building
the full constrained basis and Hamiltonian still precedes projection. A local
attempt exceeded the 8 GiB memory limit during full model construction; no
12x4 matrix, canonical estimate, or eigensolve was completed locally. The cheap
character/support preflight completed independently.

The fully resolved 4x4 target block has only the target. Its raw A/Z means and
width vanish, and joint-dark deletion leaves no background. It is a symmetry
check, not a thermal comparison or scaling point. The sequence keeps the row
with `scaling_eligible=False`; it produces no three-point fits when only 8x4
and 12x4 are nontrivial. Submission evidence at these sizes supports only a
finite-size claim, not a thermodynamic extrapolation.

## Bounded regenerated evidence

`qdm_parity_evidence.py --stage exact` covers Lx=4,8 and the four positive
phases 0.025, 0.05, 0.075, 0.10. Phi=0 has enhanced symmetry and is deliberately
outside this common positive-phase family. No new phase-zero full-resolution
claim is made.

The complete eigensystems, all-state/background CSVs, audits, target-parity
Fig. 9(a) PDF/PNG, phase figure, and fixed total-energy windows
DeltaE=0.10,0.20,0.25,0.50 are saved. Every eigenpair and the complete Gram matrix
are checked at 1e-8. The original local Q=L†L is projected; it is never replaced
by a product of separately projected L operators. The primary raw ensembles
include the target. Joint-dark-cleaned values are labeled companions; empty
clean windows have no mean or width.

The primary half-width remains the prespecified DeltaE=0.25 used in PR151.
It is not tuned against an observable or a successful solver budget. The old
pilot's heuristic recommendation is not represented as having passed in this
new protocol. All other widths remain mandatory submission-gate controls.
Physical energy density is E/(4Lx); the solver and windows use total energy.

Stripe operators use the complete constrained local Hilbert-Schmidt frame,
quotiented by the kernel of P O P within the **target parity**. The same local
normalization is used at each size; the quotient depends on the sector. The
4x4 quotient is one-dimensional and the 8x4 quotient has 25 directions. Widths
are computed from energy-block covariance, invariant under rotations within
exact degeneracies. The manuscript should state this normalization explicitly.

At Lx=8, phi=0.05:

| Quantity | Target-parity raw value |
| --- | ---: |
| Canonical A | 0.05721170844 |
| Canonical Z | 0.11134491246 |
| Fixed-window A | 0.04931479319 |
| Fixed-window Z | 0.10126465225 |
| Window state count | 47 |
| Raw stripe width | 0.07085592715 |
| Joint-dark-cleaned width | 0.056869 (approximately) |

## Server sequence and submission gate

### Diagnosing PRIMME's linear algebra runtime

PRIMME's extension can link to system BLAS/LAPACK independently of NumPy's and
SciPy's bundled OpenBLAS. A long run sampled in system `zlarfx_`, with the bundled
OpenBLAS workers idle, suggests a dense linear algebra bottleneck but does not
prove how the entire run spent its time or how much convergence remains. Inspect
the actual providers before diagnosing the lattice size as infeasible:

```bash
QDM_CONTAINER=$(cat experimental/data/evidence_jobs/qdm_parity_submission_20261009/L12-refine.container-id)
docker exec "${QDM_CONTAINER}" python scripts/docker/verify_primme_blas_runtime.py
```

The report resolves the extension's BLAS/LAPACK dependencies through `ldd` and
canonical filesystem paths. The PRIMME image now installs Debian's pthread
OpenBLAS implementation and refuses a build if its actual providers remain
reference BLAS/LAPACK. NumPy/SciPy imports alone do not satisfy this gate. Default
images without the PRIMME feature do not acquire this extra system dependency.
The existing numerical smoke test still checks physical residuals and cache
preservation. Provider selection is not a performance benchmark or a promise of
parallel scaling; some LAPACK phases remain serial.

### Collect progress and a current native stack

Use the host checkout's diagnostic script with a running or retained container:

```bash
QDM_CONTAINER=$(cat experimental/data/evidence_jobs/qdm_parity_submission_20261009/L12-refine.container-id)
bash scripts/docker/diagnose_qdm_primme.sh "${QDM_CONTAINER}" --native-stack \
  2>&1 | tee L12-refine-diagnosis.log
```

The script records image identity, container state, resource use, process times,
the latest progress counters and recent non-progress log messages. It computes
the observed seconds per outer iteration over the recent samples, without
treating iteration or matrix-product counts as converged eigenpair counts or
an ETA. It reads at most the last 5000 log lines; absence of a convergence
message in that tail does not establish zero convergence.

`--native-stack` requires host `gdb` and uses `sudo` when needed. GDB briefly
stops Python, reads each thread's stack and detaches to resume it. Library paths
are resolved inside the container, then GDB searches their canonical directories
through `/proc/PID/root` with host-library fallback disabled. Merely prefixing
the sysroot is insufficient when absolute alternatives symlinks escape into
the host filesystem. The Python executable is read through `/proc/PID/exe`.
Omit that
option to collect the other diagnostics without a debugger. The script does
not launch, pause, stop or restart the solver, or write to its evidence cache.

PRIMME 3.2.3's `stats_timeDense` measures `Num_update_VWXR_Sprimme`, not all
dense linear algebra. In particular, the projected eigensolve in
`solve_projection.c` calls the LAPACK wrapper without adding that time to
`timeDense`. The three displayed matrix/orthogonalization/dense clocks do not
account for the full solve. Small values do not rule out a dominant projected
eigensolve. A single native stack identifies the sampled phase, not the
fraction of the whole run spent there.

For a future controlled solver experiment, the launcher forwards explicitly set
`QLINKS_QDM_PRIMME_METHOD`, `QLINKS_QDM_PRIMME_MAX_BASIS_SIZE`,
`QLINKS_QDM_PRIMME_MAX_BLOCK_SIZE`,
`QLINKS_QDM_PRIMME_MIN_RESTART_SIZE` and `QLINKS_QDM_PRIMME_MAX_PREV_RETAIN`.
Previously these overrides worked only inside the Python process and were
dropped at the Docker boundary. Unset variables retain PRIMME's defaults.
The start log and checkpoint solver metadata record the requested settings;
`null` means no explicit override, not a queried effective PRIMME default.
No block size or method is promoted as faster without a measured comparison.
These controls do not alter the spectral problem, the `1e-11` refinement
tolerance or the `1e-6` physical-residual acceptance threshold. They take effect
only for a new solver process; they cannot retune an already running solve.

### Bounded trial for the large projected-eigensolve bottleneck

The 2026-10-23 server sample showed `solve_H_RR_zprimme` diagonalizing a
6176-dimensional projected problem, with `numConverged=52`, after approximately
171 hours. The recent rate was 391.5 seconds per outer iteration and one matrix
product per iteration. The wrapper's default refinement basis cap was 8397.
The debugger reported mismatched system-library symbols, so the leaf routine
name and distorted low-level arguments must not be treated as reliable.
The consistent PRIMME frames establish the sampled projected-eigensolve path;
the internal convergence count is not physical-H acceptance or a linear ETA.

An explicit `QLINKS_QDM_PRIMME_MAX_BASIS_SIZE` overrides the refinement cap.
For a cap smaller than the requested eigenpair count, the wrapper enables hard
locking. PRIMME retains the complete supplied warm-start matrix, initializes
only part of it into the active basis, and brings in remaining guesses during
restarts as completed pairs are deflated. This follows PRIMME 3.2.3's `init.c`
and `main_iter.c`; it does not request fewer eigenpairs or loosen tolerance.
The runtime smoke test exercises four cached guesses with a three-vector basis
and verifies the complete spectrum, residuals, orthogonality and source bytes.

Use a bounded trial before adopting this configuration for production. Pause
the previous writer first; retain its process state and original checkpoint.
Do not unpause it while a replacement writes to the same run/cache.
After updating the checkout, the existing tested PRIMME image can run:

```bash
QLINKS_EVIDENCE_RUN_ID=qdm_parity_submission_20261009 \
QLINKS_DOCKER_IMAGE=tanlin2013/qlinks:notebook-primme \
QLINKS_DOCKER_DETACH=1 \
QLINKS_NUM_THREADS=4 \
QLINKS_DOCKER_MEMORY_LIMIT=128g \
QLINKS_QDM_PARITY_BUDGETS=4096 \
QLINKS_QDM_PRIMME_METHOD=PRIMME_GD_plusK \
QLINKS_QDM_PRIMME_MAX_BASIS_SIZE=512 \
QLINKS_QDM_PRIMME_MAX_BLOCK_SIZE=32 \
QLINKS_QDM_PRIMME_MIN_RESTART_SIZE=256 \
QLINKS_QDM_PRIMME_MAX_PREV_RETAIN=32 \
QLINKS_QDM_PRIMME_WALLTIME_SECONDS=7200 \
scripts/docker/docker_run_qdm_parity_evidence.sh L12-refine
```

The walltime limit runs **inside** the container and covers context construction,
the eigensolve and reconstruction. It sends TERM at 7200 seconds and KILL after
another 30 seconds if needed. An ordinary timeout exits 124 and is not evidence
of convergence. Unset the variable for an unrestricted production run only
after evaluating the trial. A forced interruption does not preserve in-solve
progress; previously completed checkpoints remain reusable. No production
12x4 speedup or completion time has yet been established for these settings.

Progress now includes `stats_numRestarts`, the effective basis/block limits,
and `primme_init_size_last_restart`. The last field is PRIMME's `initSize`
snapshot at restart boundaries, not a continuously updated convergence count.
Compare completed restart snapshots and physical acceptance when the solve
returns; a faster outer-iteration rate alone does not certify a better solve.

### Pull the tested PRIMME image from CI

The `PRIMME smoke` workflow publishes the **same** linux/amd64 image that passed
its provider audit, numerical cache/refinement smoke test and physical QDM
comparison. Publication runs only on successful matching pushes to `main`, or a
manual dispatch on `main`. Pull-request runs test locally without publishing or
accessing Docker Hub credentials. Dockerfile and dependency-lock changes also
trigger the workflow. The ordinary notebook image is not a PRIMME runtime.

After that workflow succeeds on the updated `main`, pull and verify the runtime
on the server instead of rebuilding it:

```bash
export QLINKS_DOCKER_IMAGE=tanlin2013/qlinks:notebook-primme
docker pull "${QLINKS_DOCKER_IMAGE}" &&
docker run --rm "${QLINKS_DOCKER_IMAGE}" \
  python scripts/docker/verify_primme_blas_runtime.py --require-openblas &&
docker run --rm "${QLINKS_DOCKER_IMAGE}" \
  python scripts/docker/verify_primme_evidence_runtime.py
```

The launcher uses this exported image with the existing run ID and cache:

```bash
QLINKS_EVIDENCE_RUN_ID=qdm_parity_submission_20261009 \
QLINKS_QDM_PARITY_BUDGETS=4096 \
QLINKS_NUM_THREADS=4 \
QLINKS_DOCKER_MEMORY_LIMIT=128g \
QLINKS_DOCKER_DETACH=1 \
scripts/docker/docker_run_qdm_parity_evidence.sh L12-refine
```

Retire any old writer of this run/cache before launching its replacement.
The workflow summary also records the commit-specific tag
`tanlin2013/qlinks:sha-<FULL_MAIN_COMMIT_SHA>-notebook-primme`; it can be used as
`QLINKS_DOCKER_IMAGE` instead of the moving `notebook-primme` tag. For an exact
runtime pin, use the summary's `tanlin2013/qlinks@sha256:...` digest reference;
rerunning CI can rebuild a commit-specific tag with updated base packages.
The tag names a runtime build, not a completed 12x4 solve. Pulling a new tag does not change an
existing container. Earlier CI runs with `push: false` did not export a reusable
image, so they cannot be pulled retroactively.

### Build a local candidate when needed

For Dockerfile experiments or when a published runtime is unavailable, build and
verify a candidate under a separate tag, retaining the current container and all
checkpoints:

```bash
QLINKS_PRIMME_DOCKER_IMAGE=qlinks:primme-openblas \
scripts/docker/build_primme_evidence_image.sh
docker run --rm qlinks:primme-openblas \
  python scripts/docker/verify_primme_blas_runtime.py --require-openblas
docker run --rm qlinks:primme-openblas \
  python scripts/docker/verify_primme_evidence_runtime.py
```

Do not modify shared-library files or alternatives inside the running solver.
Changing the checkout/image does not instrument an already executing solve.
Only after examining the provider report and validating the replacement image
should a retry be considered; stop only the selected parity container if doing
so, preserve its logs, and retain both original checkpoint backends. No new
production retry is launched by the build or validation commands above.

Future solves print their dimension, budget, raw tolerance, initial-vector count,
and basis cap before entering PRIMME, and print a return marker before full-Ritz
reconstruction. Native `printLevel=2` reports converged/locked folded pairs and
their residuals; these are not physical-H acceptance results. Level 3 also
prints outer iterations and may produce substantial output. Set
`QLINKS_QDM_PRIMME_PRINT_LEVEL` to adjust verbosity; the Docker runner forwards it.
Matrix callbacks emit elapsed time, matrix-product/outer-iteration counters and
PRIMME's accumulated matrix-product/orthogonalization/dense timings every 300
seconds (`QLINKS_QDM_PRIMME_PROGRESS_SECONDS`, forwarded by the runner, zero disables). This callback
reporting is not an independent timer: no callbacks occur during a long native
LAPACK operation. It does not save intermediate vectors or predict an ETA.

If 12x4 remains outside the production budget, the exact 8x4 target-parity
results remain usable for a finite-size thermal comparison. The 4x4 target
sector is one-dimensional, so it is not a second thermal-background scaling
point. Do not close the existing 12x4 submission gate or claim thermodynamic
extrapolation by dropping its missing rows.

Set `QLINKS_EVIDENCE_RUN_ID` again after opening a new SSH session; shell exports
do not carry over to another session. Only `exact` may generate a new timestamped
run ID. All follow-up stages require an explicit ID, and spectrum/observables/
sequence check the window-protocol file on the host before launching Docker.
The runner prints the export command to reuse its exact-stage folder.

Use a single explicit run id across all stages, after this PR is merged:

```bash
export QLINKS_EVIDENCE_RUN_ID=qdm_parity_submission_20261009
scripts/docker/docker_run_qdm_parity_evidence.sh exact
scripts/docker/docker_run_qdm_parity_evidence.sh L12-preflight
scripts/docker/docker_run_qdm_parity_evidence.sh L12-canonical
scripts/docker/docker_run_qdm_parity_evidence.sh L12-spectrum
scripts/docker/docker_run_qdm_parity_evidence.sh L12-observables
scripts/docker/docker_run_qdm_parity_evidence.sh sequence
scripts/docker/docker_run_qdm_parity_evidence.sh gate
```

For a long stage, launch it in the background with redirected input/output:

```bash
QLINKS_EVIDENCE_RUN_ID=qdm_parity_submission_20261009 \
nohup scripts/docker/docker_run_qdm_parity_evidence.sh L12-spectrum \
  > L12-spectrum-launch.log 2>&1 < /dev/null &
```

This runs the existing foreground Docker wrapper under `nohup`. The wrapper's
stage log also persists in its evidence folder. The same inline run-ID assignment
works for later stages; wait for each prerequisite to finish successfully.
If a protocol file is missing, use the existing parity run folder rather than
copying a legacy pilot's recommendation into a new folder.

The exact and cheap preflight stages default to the existing notebook image
and 32g memory. Matrix/canonical/spectrum/observable stages default to 128g;
the spectral stage uses the existing `notebook-primme` image. Override
`QLINKS_DOCKER_IMAGE`, `QLINKS_DOCKER_MEMORY_LIMIT`, `QLINKS_DOCKER_CPUS`, and
`QLINKS_NUM_THREADS` for the server. No image rebuild is needed when these
images already contain their existing dependencies. Docker was not run locally.
`QLINKS_DOCKER_DRY_RUN=1` prints the resolved command. Stage logs persist on
the host even if a container exits. This runner is foreground: use the server's
normal persistent terminal/session for a long stage.

The separate cache root is `experimental/data/evidence_cache/qdm_target_parity_v1`.
It also uses the reduced matrix fingerprint in its scientific compatibility key.
The old two-week run and its caches are not stopped, relabeled, or overwritten.
Legacy partial vectors are not silently reinterpreted as resolved vectors.
The initial budget schedule is 768,1024,1536,2048,3072,4096, stopping after
acceptance. Set `QLINKS_QDM_PARITY_BUDGETS` to extend it up to 8192 if needed.
Completed compatible budgets are checkpointed and reused. The primary window
remains DeltaE=0.25; the spectral solve covers the widest control up front.

### Recovering inaccurate folded-spectrum eigenvectors

A small residual for `(H-E*)^2` does not ensure a small residual for `H`.
The folded problem can mix states on opposite sides of the target energy,
including near-degenerate pairs whose splitting exceeds the old scalar-cluster
threshold. PRIMME's `tol` is relative to the folded operator's estimated norm.
The new PRIMME postprocessing solves the full generalized projected problem
`(V† H V)c = E (V† V)c`; the Gram metric also handles small input orthogonality
errors. It retains every returned direction and rejects rank deficiency rather
than silently deleting states. Actual H and folded residuals are recomputed for
every reconstructed vector; full output Gram error is measured in Frobenius norm,
which bounds the spectral norm. Solver and evidence tolerances are unchanged.

For an existing failed 4096-vector parity checkpoint, after merging the recovery
PR and updating the checkout, use:

```bash
QLINKS_EVIDENCE_RUN_ID=qdm_parity_submission_20261009 \
nohup scripts/docker/docker_run_qdm_parity_evidence.sh L12-reconstruct \
  > L12-reconstruct-launch.log 2>&1 < /dev/null &
```

This stage defaults to **only budget 4096**. `QLINKS_QDM_PARITY_BUDGETS` can select
other saved original PRIMME budgets. It rebuilds the same parity Hamiltonian
and verifies its matrix fingerprint, but calls no iterative eigensolver. Sparse
Hamiltonian products and dense k-by-k projected diagonalization still take time;
there is no measured 12x4 reconstruction runtime yet. Matrix construction still
requires the server's production resources. The routine has O(n k^2 + k^3)
work and O(n k + k^2) storage; it never diagonalizes the full n-by-n Hamiltonian.

Original `primme/budget_*` checkpoints are preserved byte for byte. Reconstructed
payloads are saved under `primme_full_ritz_v1/budget_*`, with original source
metadata, reconstruction timing, and full Gram error recorded. The original
convergence row is archived under `reconstruction_source_rows/` before its
replacement. Existing spectral and observable acceptance reports are archived
and invalidated before replacing diagnostics, so downstream evidence must be
regenerated and an interrupted repair cannot retain a stale pass. Repeating recovery reuses the versioned result for the same source.
Future PRIMME solves also use full-subspace reconstruction and the new backend;
cache reuse, warm starts, and observable processing recognize those results.

Read `qdm_checkerboard_L12_fixed_O1_reconstruction_acceptance.json` in the same
run folder. A successful recovery requires **all** returned H residuals and full
Gram error at most 1e-6, matching the cache quality required by observables.
Window residuals and edge coverage are also reported separately. Recovery can
fail when the saved subspace has missing components; a small folded residual is
not a promise that reconstruction will pass. Failed candidates remain available
for diagnosis and the original checkpoint remains intact.

A recovery success with one covered budget exits successfully while reporting
`spectral_acceptance_closed=false`: it is not another independent budget.
Only after examining the recovered residuals and count should a new budget be
chosen (for example 5120 if coverage remains adequate), followed by observables,
sequence, and the submission gate. No residual tolerance or window is relaxed.

### Iterative refinement of a saved budget

When reconstruction passes full-Gram quality but H residuals remain too large,
rotation within the saved span cannot remove the remaining off-subspace error.
`L12-refine` explicitly revisits the same budget with a tighter raw folded
tolerance (default 1e-11 instead of 1e-8). All H/Gram acceptance thresholds remain
1e-6 and the prescribed windows remain unchanged.

```bash
QLINKS_EVIDENCE_RUN_ID=qdm_parity_submission_20261009 \
QLINKS_QDM_PARITY_BUDGETS=4096 \
QLINKS_DOCKER_DETACH=1 \
scripts/docker/docker_run_qdm_parity_evidence.sh L12-refine
```

For long SSH runs, `QLINKS_DOCKER_DETACH=1` uses Docker's detached lifecycle;
no `nohup` or shell backgrounding is needed. Launch success does **not** mean
the numerical job has finished. The named container is retained after exit,
including its logs, exit code, and OOM flag. The runner prints monitoring commands
and records its ID in the evidence folder. After reconnecting:

```bash
QDM_OUTPUT=experimental/data/evidence_jobs/qdm_parity_submission_20261009
QDM_CONTAINER=$(cat "${QDM_OUTPUT}/L12-refine.container-id")
docker logs -f "${QDM_CONTAINER}"
docker inspect --format '{{json .State}}' "${QDM_CONTAINER}"
```

PRIMME is silent during its iterative solve; a quiet log after `warm_start`
does not establish completion. Check `.State.Running`, `.State.ExitCode`, and
`.State.OOMKilled`, then the refinement report. Exit 129 is consistent with
SIGHUP (128+1), but does not identify who sent the signal. The foreground runner
can proxy signals to the container, so `nohup` alone is not the preferred SSH
lifecycle. Foreground remains the default and now prints its Docker exit code.
Detached runs do not automatically copy Docker logs into the stage `.log` file.
After exit, save logs and inspect the reports before removing the container:

```bash
docker logs "${QDM_CONTAINER}" > "${QDM_OUTPUT}/L12-refine.log" 2>&1
# Only after preserving diagnostics:
docker rm "${QDM_CONTAINER}"
```

Use the same run ID and cache for a retry. An interrupted iterative solve has no
new result checkpoint until it returns; the original reconstructed 4096-vector
checkpoint remains available. Docker detach prevents terminal-dependent signal
forwarding; it does not protect against explicit signals or host/daemon failures.

The stage bypasses the completed-row skip only for the explicitly requested
budgets. It requires a compatible reconstructed checkpoint with every requested
vector and a recorded full-Gram error at most 1e-6. It uses **all 4096 vectors**
as PRIMME initial guesses, rather than the ordinary 512-vector cap. These are
guesses, not locked approximate eigenvectors. The PRIMME maximum basis is
explicitly sized to accommodate the full initial subspace (2.05 times the budget,
capped by the sector dimension). Missing/incompatible/incomplete or poorly
orthogonal checkpoints fail before any cold iterative solve starts.

Raw `primme` and reconstructed `primme_full_ritz_v1` checkpoints stay untouched.
Refinement results use a separate backend keyed by the exact raw tolerance;
distinct tolerances cannot overwrite each other's candidates. Source checkpoint
metadata and full warm-start vector count are recorded. Existing spectral and
observable reports are invalidated, and replaced CSV rows are archived. Repeated
refinement at the same tolerance can update that refinement candidate or reuse
it if its stored tolerance, H residuals, and Gram quality already pass.

Inspect `qdm_checkerboard_L12_fixed_O1_refinement_acceptance.json` afterward.
Passing requires all requested eigenpairs to be returned, all H residuals at
most 1e-6, and complete Gram error at most 1e-6. A successful same-budget
refinement is one budget point; it does not supply the second independent
coverage/count check. Tighter folded tolerance is not a guarantee of physical-H
acceptance, and production refinement runtime has not been measured.

After inspecting a successful 4096 refinement, a subsequent new budget must
also use the tighter raw tolerance, for example:

```bash
QLINKS_EVIDENCE_RUN_ID=qdm_parity_submission_20261009 \
QLINKS_QDM_PARITY_BUDGETS=5120 \
QLINKS_QDM_PRIMME_WARM_START_VECTORS=4096 \
QLINKS_DOCKER_DETACH=1 \
scripts/docker/docker_run_qdm_parity_evidence.sh L12-spectrum --tolerance 1e-11
```

The runner now honors the optional warm-start cap override; its normal default
remains 512. Use a new budget only after evaluating the refinement report and
its coverage/count. Observable, sequence, and submission gates remain mandatory.

The canonical stage regenerates the 12x4 Sy=-1 trace rather than importing
translation-union canonical means. It compares independent 8/16-sample scans
and a same-seed 41/81-point beta-grid refinement on [0,0.25]. Its recorded
conditional ratio jackknife errors are not claimed to propagate beta-matching
uncertainty. Independent-sample and grid refinement provide additional
sensitivity checks. Default conditional-error tolerance is 5e-4, grid-change
tolerance 5e-5, and independent means must agree within combined 3-sigma plus
grid tolerance. Increase `--samples` if needed; a failed gate remains open.
For example:

```bash
scripts/docker/docker_run_qdm_parity_evidence.sh L12-canonical --samples 16
```

The spectrum stage retains the existing acceptance requirements: at least
two distinct budgets reach beyond both edges of the widest prescribed control
(DeltaE=0.50), have stable state counts,
and have residuals <=1e-6. This is a budget-convergence coverage test, not a
rigorous mathematical eigenvalue-count certificate. The observable stage
recomputes every window eigenpair residual and all Gram blocks with bounded
memory, requires complete target-projector weight, and requires raw A/Z and
stripe-width changes <=5e-5 between budgets. Compressed degeneracies use the
existing residual-aware energy-block tolerance.

The final gate requires:

- complete positive-phase 4x4/8x4 regeneration and target-irrep preflight;
- accepted fresh 12x4 parity-conditioned canonical estimate;
- accepted two-budget primary spectral and observable coverage;
- stable 12x4 primary and all three control windows, including all-vector
  residual, Gram, and target-projector validation;
- a parity-labeled size sequence with the trivial 4x4 row excluded from fits.

`gate` writes `qdm_parity_submission_gate.json` and exits nonzero while any
requirement is open. If a wider control remains uncovered or unstable after
primary acceptance, extend the explicit budgets and rerun spectrum, observables,
sequence, and gate. The gate is numerical evidence only; final manuscript
wording and figure review remain with the authors. It does not submit anything.

The old monolithic notebook and old artifacts retain legacy status. This
focused batch workflow is the authoritative parity-aware regeneration path;
old translation-union figures should not be substituted into the new sequence.
