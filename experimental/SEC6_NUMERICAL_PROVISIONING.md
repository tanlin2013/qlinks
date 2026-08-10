# Section VI numerical provisioning cache

**Updated:** 2026-08-09
**Authoritative production job:** `data/evidence_jobs/spin1_production_20260806T074051Z/`

## Purpose

This is the short-term qlinks handoff for the remaining Sec. VI work. The exact
caged family, representative point, direct microcanonical sequence through a
fully covered narrow window at `L=14`, and the two-site concentration data
through `L=12` are already available. The remaining tasks are specifically to
separate true finite-size physics from the partial-spectrum solver at `L=14`,
obtain the missing larger-size concentration diagnostic, and finish the
thermodynamic/family-wide extrapolation.

## Locked Hamiltonian and representative point

Use

\[
K_d(t_d)=\sum_r\left[t_dS_r^+S_{r+d}^-+t_d^*S_r^-S_{r+d}^+\right],
\]

\[
H_\kappa=K_1(J)+K_3(0.1J)+K_2(i\kappa),
\qquad \kappa_\star/J=0.1.
\]

Use even `L`, PBC, `M=-2`, the tower momentum sector, and `h=D=0`. The exact
compatibility rule is

\[
t_d^*+(-1)^dt_d=0.
\]

The representative point breaks ordinary inversion and the unitary `C_A`
anticommutation but retains the antiunitary spectral reflection
`Theta=C_A K`. Keep exact-energy blocks basis independent.

The principal sampled positive grid remains

`kappa/J in {0.05,0.10,0.15,0.20}`.

Treat `kappa=0` as a symmetry-enhanced endpoint control.

## Completed representative-point evidence

### Direct microcanonical sequence

At `kappa_star/J=0.1`, the translated joint-dark projector has rank one through
`L=14`. For `L=14` the spectrum was obtained with `sparse_shift_invert` and
8192 eigenpairs in a resolved sector of dimension 35925. The computed spectrum
covers

\[
|E|\lesssim 2.08384.
\]

The original `Delta E proportional to L^(1/2)` window is therefore **not**
covered at `L=14`, and must not be used there. The narrower windows below are
fully covered.

For the `L^(1/4)`, prefactor-1 window,

\[
\Delta E_{14}=1.93351<2.08384,
\]

with 7615 raw and 7614 retained states. The clean microcanonical values are

\[
(\tau_A,\tau_Z,\tau_Y)_{L=14}
=(0.113212,0.220243,0.328219).
\]

For the same window convention, the clean--clean matching sequence is

\[
\Delta_L^{\rm cc}
=0.026664,\ 0.020872,\ 0.012796,\ 0.012246
\quad (L=8,10,12,14).
\]

A fixed-width `Delta E about 1` window is also fully covered at `L=14` (4011
raw states) and gives `Delta_14^cc=0.012882`, so the apparent `L=12 -> 14`
plateau is not explained simply by using the largest available window.

The exact fixed-`M`, `beta=0` limits remain `(1/9,2/9,1/3)`, but they are an
auxiliary reference only until local microcanonical--trace equivalence is
settled.

### Two-site concentration

The complete magnetization-preserving two-site Hermitian algebra has dimension
19. At `kappa_star/J=0.1`,

\[
w_8=0.1415,\qquad w_{10}=0.0619,\qquad w_{12}=0.0359.
\]

No `L=14` covariance result is present because the current implementation tied
that calculation to the uncovered primary `L^(1/2)` window.

## P0 tasks

### P0.1 -- sparse-solver convergence at `L=14`

Before interpreting the `L=12 -> 14` matching plateau physically, rerun only
the representative `L=14` sector with a larger shift-invert eigenpair budget.
Use at least one of

`k in {10000, 12000}`

in addition to the existing `k=8192`. If practical, a smaller `k=6144` run can
serve as a lower-budget cross-check.

Compare **the same deliberately safe windows**, not progressively enlarged
windows:

- `Delta E = 1`;
- `Delta E = L^(1/4)` with prefactor 1;
- optionally `Delta E = 0.75` as a stricter nested check.

For every budget export:

- covered spectral half-width;
- raw/retained state counts;
- `tau_A,tau_Z,tau_Y` raw and clean;
- clean--clean `delta_A,delta_Z,delta_Y,Delta`;
- tower and joint-dark residuals.

Acceptance target: changes in each clean witness and in `Delta` should be
`<=1e-4` (or otherwise reported explicitly) when the eigenpair budget is
increased. If this passes, treat the current `O(10^-2)` residual as a physical
finite-size/window effect rather than sparse-spectrum truncation.

Suggested output:

- `spin1_xy_kappa0p1_L14_sparse_convergence.csv`.

### P0.2 -- `L=14` complete two-site concentration

Compute the same block-invariant 19-operator covariance at `L=14`, but use a
window that is demonstrably fully covered by the sparse spectrum (prefer
`Delta E=1` and/or `L^(1/4)` after P0.1). Do not require the uncovered
`L^(1/2)` window merely for protocol continuity.

Export the raw-window block-invariant covariance and, if cleaning is used in
the existing implementation, the cleaned companion or an explicit
removed-fraction bound relating the two.  Also export:

- `w_14`;
- largest covariance eigenvalue;
- median nonidentity width;
- worst eigenoperator coefficients;
- exact-energy-block tolerance audit;
- raw/retained state counts and spectral coverage;
- the finite-size window-entropy estimator `log(N_win)/L` for the same safe
  windows, reported only as a bulk-consistency trend rather than a standalone
  thermodynamic proof.

Suggested outputs:

- `spin1_xy_kappa0p1_concentration_L14.csv`;
- update `spin1_xy_kappa0p1_concentration.csv` and its fit file.

### P0.3 -- direct microcanonical extrapolation

Treat the direct **raw** microcanonical sequence as the defining ETH evidence,
and retain the joint-dark-cleaned sequence as a finite-size diagnostic.  Fit
both `tau_A,tau_Z,tau_Y` sequences using `L>=8`, with window choice as a
systematic dimension.  The raw and clean extrapolations should agree if the
removed rank fraction continues to vanish. Compare zero/nonzero finite-size
corrections without forcing the asymptote to the exact `beta=0` values.

In parallel, keep the microcanonical--`beta=0` distance as a separate local-
equivalence test. Compare

\[
\Delta_L=c/L,\quad c/L^2,\quad \Delta_\infty+c/L,
\]

but do not promote any model solely by lower RMSE with four sizes. Report
bootstrap/systematic intervals and the solver-convergence result from P0.1.

### P0.4 -- family-wide larger-size check

After P0.1--P0.2, add at least one `L=14` interior/end-of-grid point (prefer
`kappa/J=0.2`) or provide a justified uniform bound. Update both
`Delta_L^max` and `w_L^max`. Negative `kappa` remains an inversion-related
implementation check rather than a second production sequence.

## P1 tasks

1. **Grid refinement / continuity:** needed for an open-interval ICQMBS claim.
2. **Larger local region:** only if the paper seeks background ETH beyond the
   complete two-site algebra.
3. **Finite-beta deformation grid:** optional generality evidence; not needed
   for the primary `D=0` storyline.

## Fig. 6 update contract

The current production figure from `spin1_production_20260806T074051Z` is the
correct provisional artwork at `kappa_star/J=0.1`:

- panel (a): representative-point ETH scatter, `L=12`;
- panel (b): direct microcanonical sequence through `L=12` with `beta=0`
  auxiliary references;
- panel (c): family-wide matching diagnostic through `L=12`;
- panel (d): 19-operator concentration through `L=12`.

Do not add `L=14` to panel (d) until P0.2 is complete. Add the `L=14` point to
panel (b) only after P0.1 confirms sparse-solver stability, and label it as a
fully covered narrow-window sparse point if its window differs from the dense
sequence. The final caption must not imply that `Delta_L -> 0` is established.

## Claim boundary

Already established:

- exact compatible caged family and bounded `A,Z,Y` construction;
- positive same-Hamiltonian microcanonical witness values through a fully
  covered `L=14` narrow window at the representative point;
- rank-one translated-joint-dark inventory through `L=14`;
- strong complete two-site concentration through `L=12`.

Still provisioned:

- sparse-solver independence and complete two-site concentration at `L=14`;
- controlled microcanonical and microcanonical--`beta=0` thermodynamic fits;
- a larger-size family-wide envelope;
- grid refinement/continuity for an open interval.
