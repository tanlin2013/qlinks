# Section VI numerical provisioning cache

## Purpose

This file is the short-term handoff for the spin-1 XY evidence used in Sec. VI
and Fig. 6. It is intentionally separate from `data/EVIDENCE_SUMMARY.md`:
this cache records the locked Hamiltonian family, completed calculations,
unresolved extrapolations, acceptance criteria, and the next numerical jobs.

The target logic is

1. recover the known \(Q=\pi\) bimagnon tower;
2. define a generic thermal reference Hamiltonian;
3. continue the same exact tower along a complex-Hermitian compatible path;
4. expose the transverse obstruction in the ambient complex plane;
5. compare the compatible-family microcanonical ensemble with the exact
   \(\beta=0\) reference;
6. test concentration of the complete local operator algebra;
7. classify the result only at the strength supported by the extrapolations.

## Authoritative production job

Use

`data/evidence_jobs/spin1_production_20260805T021029Z/`

as the current authoritative spin-1 evidence folder.

## Hamiltonian family locked for implementation

Use manuscript ladder-operator conventions

\[
K_d(t_d)=\sum_r\left[t_dS_r^+S_{r+d}^-+t_d^*S_r^-S_{r+d}^+\right].
\]

The thermal reference point is

\[
H_{\rm ref}=K_1(J)+K_3(J_3^\star),\qquad J_3^\star/J=0.1.
\]

The continuous compatible family is

\[
H_\kappa=K_1(J)+K_3(J_3^\star)+K_2(i\kappa),
\]

or explicitly

\[
H_\kappa=J\sum_r(S_r^+S_{r+1}^-+\mathrm{h.c.})
+J_3^\star\sum_r(S_r^+S_{r+3}^-+\mathrm{h.c.})
+i\kappa\sum_r(S_r^+S_{r+2}^- -\mathrm{h.c.}).
\]

The staggered tower phase is \(\eta_r=(-1)^r\). Real odd-range and purely
imaginary even-range exchanges obey

\[
t_d^*+(-1)^dt_d=0,
\]

so the tower remains an exact zero-energy eigenstate for every real \(\kappa\).
No diagonal deformation is included in the primary protocol.

## Common symmetry resolution

Use only symmetries common to the full family:

- fixed total magnetization \(M\);
- translation momentum \(k\).

Do not resolve ordinary inversion at nonzero \(\kappa\), because inversion maps
\(H_\kappa\) to \(H_{-\kappa}\). Record fixed-\(M\), resolved-\((M,k)\), raw
microcanonical, and joint-dark-cleaned microcanonical values separately.

## Joint-dark projector

Define

\[
Q_{\rm all}=\sum_R(Q_{A,R}+Q_{Z,R}+Q_{Y,R}).
\]

Within each exact energy block, diagonalize \(P_EQ_{\rm all}P_E\) and retain
its numerical kernel. The cleaned microcanonical ensemble removes the full
translated joint-dark kernel, including the target tower. Subtract the target
projector only when reporting the residual non-target dark rank.

Current production status:

- the translated joint-dark projector has rank one for
  \(L=6,8,10,12\) and every sampled \(\kappa/J\in[-0.2,0.2]\);
- this rank is consistent with the selected tower alone;
- the reference Type-1 search finds two additional non-target states only at
  \(L=6,\kappa=0\); both are lifted at \(\kappa/J=0.1\) and \(0.2\);
- for \(L=8,10,12\), the reference Type-1 inventory contains only the target.

The translated joint-dark kernel is complete only relative to the declared
translated \(A,Z,Y\) witness family; do not identify it with the set of all
possible caged states.

## Current production conclusions

### Exact obstruction geometry

At \(L=8\), the complex plane

\[
t_2/J=u+iv
\]

shows an exact zero-residual line at \(u=0\). The obstruction Jacobian has
zero derivative norm along the imaginary direction and nonzero derivative norm
along the real direction. This establishes the compatible line and its
transverse obstruction at finite size.

### Positive finite-size thermal separation

For the cleaned microcanonical ensemble, the witness values remain positive
throughout the sampled path. At \(L=12\), their variation over
\(\kappa/J\in[-0.2,0.2]\) is small:

\[
\tau_A^{\rm mc}\simeq0.1122\text{--}0.1129,\qquad
\tau_Z^{\rm mc}\simeq0.2192\text{--}0.2197,
\]

\[
\tau_Y^{\rm mc}\simeq0.3278\text{--}0.3282,
\]

while all cage values remain exactly zero.

### Matching distance is not yet extrapolated

At the reference point \(\kappa=0\),

\[
\Delta_L\equiv\max_{\alpha\in\{A,Z,Y\}}
\left|\tau_{\alpha,L}^{\rm mc,clean}
-\tau_{\alpha,L}^{\beta=0,(M,k)}\right|
\]

is

\[
\Delta_6=0.0821,\quad
\Delta_8=0.0188,\quad
\Delta_{10}=0.0179,\quad
\Delta_{12}=0.0130.
\]

The interval-wide envelope is

\[
\Delta_L^{\max}=0.0821,\ 0.0190,\ 0.0179,\ 0.0130
\quad (L=6,8,10,12).
\]

The \(L=6\) window is pre-asymptotic: it contains only six states before
cleaning, five after cleaning, and collapses onto the zero-energy block.
Exclude it from thermodynamic fits.

For \(L=8,10,12\), the approximate constancy of \(L\Delta_L^{\max}\) is
consistent with an \(O(1/L)\) finite-window correction, but the present three
useful sizes do not distinguish zero limit from a small positive intercept.
Therefore:

- do not claim that \(\Delta_L\to0\) has been established;
- do not quote the existing unconstrained `a+b/L` or `a+b/L^2` fits as
  thermodynamic extrapolations;
- treat a zero-constrained \(c/L\) curve only as a guide to the eye until
  additional window scalings or a larger size are available.

### Complete local-algebra concentration

Using the block-invariant covariance matrix of a Hilbert--Schmidt-orthonormal
basis of the complete 19-dimensional two-site algebra, the interval-wide
maximum width is

\[
w_L^{\max}=0.2091,\ 0.1457,\ 0.0619,\ 0.0359
\quad (L=6,8,10,12).
\]

For the useful sizes \(L=8,10,12\), this is a strong decreasing trend. The
median nonidentity widths are at most

\[
0.0352,\quad0.0179,\quad0.00941,
\]

respectively. The present diagnostic fits produce unstable or negative free
intercepts and must not be quoted as controlled extrapolations. The data
support strong finite-size concentration across the sampled family, not yet a
proved uniform limit \(w_L^{\max}\to0\).

## Panel (a): reference ETH scatter

Completed at \(H_{\rm ref}\) in the largest reliable resolved \((M,k)\)
sector, currently \(L=12\). Plot normalized fixed witnesses \(Q_A,Q_Z,Q_Y\),
use the analytically constructed tower vector for the star, and shade the
selected window. State explicitly that the displayed background is
joint-dark cleaned.

Current file:

- `spin1_xy_cage_excised_eth_scatter.csv`.

## Panel (b): finite-size matching

The main axes should show the cleaned microcanonical values together with the
exact \(\beta=0\) reference values. The analytical fixed-\(M\) thermodynamic
limits may be shown as horizontal targets.

The inset should show the individual distances

\[
\delta_{A,L},\quad\delta_{Z,L},\quad\delta_{Y,L},
\]

rather than only their maximum. Show \(L=6\) as a gray pre-asymptotic point or
omit it from the inset fit.

Do not display a fitted zero intercept as an established result. If a
zero-constrained \(c/L\) line is retained, label it “guide to the eye.”

Current files:

- `spin1_xy_beta0_cage_excised_overlap.csv`;
- `spin1_xy_beta0_matching_distance_fit.csv`;
- `spin1_xy_beta0_shared_asymptote_fit.csv`.

The last file is diagnostic only and must be regenerated before manuscript use.

## Panel (c): complex-\(t_2\) obstruction plane

Use

\[
t_2/J=u+iv
\]

at fixed \(J_3^\star/J=0.1\). The main axes show the normalized tower residual
and the exact compatible line \(u=0\).

The inset may show either:

1. the unscaled \(\Delta_L(\kappa)\), with no extrapolation claim; or
2. both \(\Delta_L(\kappa)\) and \(L\Delta_L(\kappa)\), where approximate
   collapse of the latter is presented only as evidence for an \(O(1/L)\)
   correction.

The established conclusion is that matching quality is nearly uniform in
\(\kappa\) over the sampled interval; the inset does not yet establish that
its thermodynamic limit is zero.

Current files:

- `spin1_xy_complex_t2_obstruction_grid.csv`;
- `spin1_xy_complex_t2_obstruction_jacobian.csv`;
- `spin1_xy_kappa_matching_grid.csv`.

## Panel (d): complete local-algebra concentration

Plot

\[
w_L(\kappa)=\sqrt{\lambda_{\max}\Gamma_L(\kappa)}
\]

as a heatmap with \(\kappa/J\) on the horizontal axis and integer \(L\) on the
vertical axis. The existing heatmap is scientifically usable as a finite-size
result. Its caption must state that the interval-wide zero-limit remains
unresolved.

Current files:

- `spin1_xy_kappa_concentration_grid.csv`;
- `spin1_xy_kappa_worst_eigenoperator.csv`;
- `spin1_xy_kappa_uniform_concentration_fit.csv`;
- `spin1_xy_kappa_uniform_envelope.csv`.

## Revised short-term numerical tasks

### P0. Consistent cleaned \(\beta=0\) comparison

The microcanonical ensemble is joint-dark cleaned. Regenerate the resolved
\((M,k)\) \(\beta=0\) trace with the same joint-dark projector removed:

\[
\rho_{\beta=0}^{\rm clean}
=\frac{P_{M,k}-P_{\rm jd}}
{\operatorname{Tr}(P_{M,k}-P_{\rm jd})}.
\]

Export raw--raw and clean--clean matching distances separately. The correction
is expected to be small at large \(L\), but the definitions should be
consistent.

### P0. Window-scaling study

Repeat the matching calculation using at least

\[
\Delta E_L=cL^{1/2},\qquad
\Delta E_L=cL^{1/4},\qquad
\Delta E_L=c,
\]

with several prefactors \(c\). Record state counts, energy-density widths,
individual witness distances, and \(\Delta_L\). Require increasing retained
state count while \(\Delta e_L\to0\).

Suggested output:

- `spin1_xy_beta0_window_scaling.csv`.

### P0. Revised extrapolation

Fit only \(L\ge8\) and compare

\[
\Delta_L=c/L,\qquad
\Delta_L=c/L^2,\qquad
\Delta_L=\Delta_\infty+c/L.
\]

For the free-intercept model, export uncertainty on \(\Delta_\infty\). Use
window choice as a systematic uncertainty or bootstrap dimension. Do not pool
\(L=6\) with the asymptotic sequence.

Suggested outputs:

- `spin1_xy_beta0_matching_fit_revised.csv`;
- `spin1_xy_beta0_matching_window_bootstrap.csv`;
- `spin1_xy_kappa_matching_scaled.csv` containing \(L\Delta_L(\kappa)\).

### P0. One larger-size point

Add at least one larger size, preferably \(L=14\), using shift-invert,
filtered typicality, or another method that does not require a full dense
spectrum. A single reliable larger-size point is more valuable than adding
more unconstrained fit forms to \(L=8,10,12\).

### P1. Concentration extrapolation

Repeat the interval-wide envelope analysis after adding the larger size. Fit
\(w_L^{\max}\) using both a zero-constrained decay and a free-intercept model.
Report the fit sensitivity to excluding \(L=6\) and to the energy-block
tolerance.

Suggested output:

- `spin1_xy_kappa_uniform_concentration_fit_revised.csv`.

### P1. Degeneracy tolerance audit

Export the energy-block tolerance used to construct \(P_EO_aP_E\), the number
of blocks, and the change in \(w_L(\kappa)\) under reasonable tolerance
variation.

### P2. Optional extensions

The following remain secondary:

- finite diagonal-shell parameter \(\Delta\);
- a two-dimensional \((J_3,\kappa)\) concentration grid;
- finite-\(\beta\) deformation matching;
- the earlier \(Q=3\pi/4\) complex odd-range example.

## Interval-wide acceptance criteria

For a sampled set \(\mathcal G_\kappa\), define

\[
\Delta_L^{\max}=\max_{\kappa\in\mathcal G_\kappa}\Delta_L(\kappa),
\qquad
w_L^{\max}=\max_{\kappa\in\mathcal G_\kappa}w_L(\kappa).
\]

The exact statements already established are:

- exact tower continuation over the compatible line;
- exact zero cage values for the fixed witnesses;
- uniformly positive exact fixed-\(M\), \(\beta=0\) witness targets;
- positive finite-size cleaned microcanonical activities across the sampled
  path;
- a rank-one translated joint-dark kernel across the sampled path;
- strongly decreasing finite-size covariance widths.

An interval-wide microcanonical deformation-stable ICQMBS claim additionally
requires controlled evidence that

\[
\Delta_L^{\max}\to0,
\qquad
w_L^{\max}\to0.
\]

The current data do not yet establish the first limit and provide strong but
not controlled evidence for the second. Until the revised extrapolations are
available, describe the result as an exact compatible caged family with
uniformly positive \(\beta=0\) targets and strong finite-size ICQMBS evidence
throughout the sampled interval.

## Plotting contract

- Render at final REVTeX physical dimensions.
- Use 9 pt base typography; inset and colorbar text must remain readable.
- Axes representing system size must use integer-only major ticks.
- Panel (b) must not imply that the current three useful sizes prove a zero
  matching-distance intercept.
- Panel (c) must distinguish exact obstruction geometry from the unresolved
  thermodynamic matching extrapolation.
- Panel (d) must be captioned as a finite-size concentration grid.
- Keep PDF and SVG exports and write the figure-dimension/font audit.
