# Square-QDM torus symmetry follow-up to PR #150

## Literature and convention map

The following are primary papers; the checkerboard conclusions below are our
own group-theory derivation and numerical checks, not claims made by these papers.

| Reference | Relevant location | Consequence for this audit |
| --- | --- | --- |
| D. Banerjee et al., *Finite-volume energy spectrum, fractionalized strings, and low-energy effective field theory for the quantum dimer model on the square lattice*, PRB **94**, 115120 (2016), [arXiv:1511.00881](https://arxiv.org/abs/1511.00881), DOI [10.1103/PhysRevB.94.115120](https://doi.org/10.1103/PhysRevB.94.115120) | Sec. 2.2; Appendix A | Separate translations, reflections, and rotations, with their actual coordinate origins. In the electric-flux representation ordinary dimer translations are CT_x and CT_y. Standalone charge conjugation does not preserve the staggered Gauss law. |
| Z. Lan and S. Powell, *Eigenstate thermalization hypothesis in quantum dimer models*, PRB **96**, 115140 (2017), [arXiv:1706.02601](https://arxiv.org/abs/1706.02601), DOI [10.1103/PhysRevB.96.115140](https://doi.org/10.1103/PhysRevB.96.115140) | Sec. II.C; Sec. III | Mixing topological sectors can produce separate expectation-value bands. Zero winding is already fixed here, so this mechanism motivates an audit but does not explain these bands. |
| S. Biswas, D. Banerjee, and A. Sen, *Scars from protected zero modes and beyond in U(1) quantum link and quantum dimer models*, SciPost Phys. **12**, 148 (2022), [arXiv:2202.03451](https://arxiv.org/abs/2202.03451), DOI [10.21468/SciPostPhys.12.5.148](https://doi.org/10.21468/SciPostPhys.12.5.148) | Sec. 3; Sec. 4 | Distinguish QDM from QLM charge conjugation. Not all spatial symmetries commute; reflection resolution matters for spectral statistics. Quarter rotations require a square torus, whereas an 8x4 torus supports half rotations. The lambda=0 chiral anticommutation is not an extra commuting quantum number at lambda=1. |

Our code stores dimer occupancy, not the staggered electric-flux variables.
Thus its geometric `Tx` is an ordinary translation of dimers, corresponding
to the papers' `CTx` in flux language. Do not add a separate global bit-flip
charge-conjugation parity to this QDM Hilbert space. Gauge generators are
already fixed by the dimer constraint, and the wrapping fluxes are fixed by
our electric-winding (0,0) selection. Approximate low-energy SO(2) symmetry is
not an exact block label for the high-energy ETH population.

## Derived checkerboard little group

At a generic positive checkerboard phase the pattern-preserving translations
have a+b even. Set D=T_x T_y and Y=T_y^2. The shifted reflections act on sites as

- S_x:(x,y)->(1-x,y), with S_x D S_x=D^-1 Y;
- S_y:(x,y)->(x,1-y), with S_y D S_y=D Y^-1;
- C2=S_x S_y, with C2 D C2=D^-1.

For the selected characters D=i, Y=+1, S_y preserves the character; S_x and
C2 exchange i with -i. Hence the rectangular-torus little group is Z2 generated
by S_y. Momentum being nonzero does not imply a trivial little group.
Bare reflections reverse the checkerboard phase. In conjunction with complex
conjugation, R_x K preserves the selected sector and commutes with H; its
square is +1. For antiunitary projection the matrix is B^dagger U B*, not
B^dagger U B. PR #150's antiunitary diagnostic used the latter expression and
is corrected in this follow-up.

## Local bounded validation (2026-10-08 UTC)

The original 1125-dimensional translation sector splits exactly into
S_y=+1 (875) and S_y=-1 (250). The target has weight one in +1 and zero in -1.
Its excluded background therefore has 874+250 states. Maximum full eigenpair
residuals are below 6.2e-14; parity-projector completeness residual is 1.1e-14,
and the Hamiltonian reconstruction residual is 2.0e-13.

The two parity populations occupy predominantly different scatter bands. The
residual-PCA labels agree with parity up to relabelling on 94.66% of states;
this is a heuristic agreement measure, not an exact classifier or a claim
about the as-yet-unreconciled original manuscript scatter.

For lambda=1, phi=0.05 and target E=8, using the full raw populations:

| Population | Dimension | Energy-matched beta | Canonical Q_A | Canonical Q_Z |
| --- | ---: | ---: | ---: | ---: |
| Translation union | 1125 | 0.0747284 | 0.0507306 | 0.0989030 |
| S_y=+1, target sector | 875 | 0.0791437 | 0.0572117 | 0.1113449 |
| S_y=-1 | 250 | 0.0511925 | 0.0288249 | 0.0567151 |

In the bounded |E-8|<=0.25 window, the same local-HS stripe-operator frame
(25 operators after the legacy projected quotient) gives worst widths 0.0752311
for the union, 0.0708559 for +1, and 0.0254768 for -1. These are fixed-window
diagnostics, not a replacement for the production concentration/phase scan.

Legacy and parity-resolved reconstructions have identical spectra (maximum
sorted energy difference 3.2e-14) and matching energy-block A/Z witness traces
(maximum difference 1.3e-12), despite individual-row differences up to 0.01345.
There are 14 degenerate background energy blocks. This demonstrates why
row-by-row matching alone cannot determine physical provenance.

The old staged source scatter is not included in the uploaded PR #150 audit
bundle. Its reported 0.620958 discrepancy remains unresolved locally. The job
compares that source against both reconstructions when executed beside the
original server evidence; missing source bytes are explicitly reported as
`source_missing`, never as a numerical mismatch. Source rows never select or
trim the candidate eigenstate population.

## Run on the server

From the repository root after merging this PR:

```bash
scripts/docker/docker_run_qdm_sy_followup.sh
```

The runner uses the mounted checkout, so no image rebuild is needed when the
existing notebook image already has the repository's dependencies. It writes a
new timestamped `experimental/data/evidence_jobs/qdm_sy_followup_*/` directory,
including a host `run.log`, exact parity eigenstate/scatter tables, sector-colored
PDF/PNG, symmetry audit, provenance report, means, fixed-window concentration,
verdict, and manifest. The job validates complete caches; the only permitted
fallback is the 8x4 1125-dimensional dense solve. It persists the validated
legacy eigensystem under its own output directory for subsequent reuse.

To reuse that eigensystem in a later attempt:

```bash
scripts/docker/docker_run_qdm_sy_followup.sh \
  --cache-root experimental/data/evidence_jobs/<previous-qdm-sy-run>
```

To select another old evidence directory, use `--data-dir PATH`. With
`--no-allow-small-dense`, missing complete caches cause an explicit failure.
No 12x4 solve, production-cache overwrite, or manuscript replacement is performed.

## Provisioning decision

The symmetry-resolved ETH comparison must use the target's S_y=+1 sector. The
union can remain a clearly labelled companion. The canonical means and stripe
widths differ after conditioning, so Fig. 9 and its sector-dependent companion
evidence require parity-aware regeneration once old-source provenance is
reconciled. A union trace is basis-invariant and need not be declared numerically
wrong merely because its irrep description was wrong.

The fixed-window context retains old translation dimensions for traceability
and records that S_y is unresolved. The full-resolution constructor now requires
an explicit +/- parity and raises a descriptive error for omitted parity. This
intentionally prevents the old notebook from silently exporting a falsely
resolved sector; its production migration is deferred until source reconciliation.
Use this diagnostic to settle source reconciliation first, then provision a
single consistent parity across lengths/phases. Existing 12x4 calculations are
translation-union evidence; do not discard or restart them automatically.
