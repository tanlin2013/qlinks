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
