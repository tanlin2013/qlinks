# Section VI numerical provisioning cache

## Purpose

This file is the short-term handoff for the spin-1 XY evidence used in Sec. VI
and Fig. 6. It is intentionally separate from `data/EVIDENCE_SUMMARY.md`:
this cache records the locked Hamiltonian family, completed calculations,
unresolved extrapolations, acceptance criteria, and the next numerical jobs.

The target logic is

1. recover the known \(Q=\pi\) bimagnon tower;
2. define the compatible Hamiltonian family and choose a nonzero interior representative point;
3. continue the same exact tower along the complex-Hermitian compatible path;
4. compare the compatible-family microcanonical ensemble with the exact \(\beta=0\) reference;
5. test concentration of the complete magnetization-preserving two-site operator algebra;
6. classify the result only at the strength supported by the extrapolations.

## Authoritative production job

Use

`data/evidence_jobs/spin1_production_20260805T021029Z/`

as the current authoritative spin-1 evidence folder.

## Hamiltonian family locked for implementation

Use manuscript ladder-operator conventions

\[
K_d(t_d)=\sum_r\left[t_dS_r^+S_{r+d}^-+t_d^*S_r^-S_{r+d}^+\right].
\]

Define the odd-range base Hamiltonian

\[
H_0=K_1(J)+K_3(J_3^\star),\qquad J_3^\star/J=0.1.
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

The primary representative thermal point is locked to

\[
H_{\rm rep}=H_{\kappa_\star},\qquad \kappa_\star/J=0.1.
\]

Use \(\kappa=0\) only as the symmetry-enhanced chiral endpoint control. The
nonzero representative point preserves the exact tower and the exact
\(\beta=0\) energy match, breaks ordinary inversion and the unitary
\(C_A\) anticommutation, but retains the antiunitary spectral reflection
\(\Theta=C_A\mathcal K\), with \(\Theta H_\kappa\Theta^{-1}=-H_\kappa\).

## Common symmetry resolution

Use even \(L\) under PBC and set \(h=D=0\) in the primary thermodynamic
sequence; subtracting the constant \(hM\) in the fixed-\(M\) sector is
equivalent. Use only unitary symmetries common to the full family:

- fixed total magnetization \(M\);
- translation momentum \(k\).

Do not resolve ordinary inversion at nonzero \(\kappa\), because inversion maps
\(H_\kappa\) to \(H_{-\kappa}\). Record fixed-\(M\), resolved-\((M,k)\), raw
microcanonical, and joint-dark-cleaned microcanonical values separately. For
the selected \(k=0\) or \(\pi\) sector, explicitly verify the antiunitary
spectral reflection \(C_A\mathcal K\), rather than assuming that nonzero
\(\kappa\) removes zero-energy pairing.

## Normalized diagonal witness convention

Use

\[
D\big[(S_r^z)^2-\mathbf 1\big]=D Y_r,
\qquad
Y_r=(S_r^z)^2-\mathbf 1.
\]

For \(D\neq0\), \(Y_r\) is the shifted single-ion operator normalized by
\(D\). Remove the coupling before specializing the primary thermal protocol to
\(D=0\); there is no numerical division by zero. Use the same dimensionless
\(Y_r\) and \(Q_r^Y=|0\rangle\langle0|\) at all \(D\) values.

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

### Exact compatible family

At \(L=8\), the ambient complex plane

\[
t_2/J=u+iv
\]

has an exact zero-residual line at \(u=0\). The obstruction Jacobian has zero
derivative norm along the imaginary direction and nonzero derivative norm
along the real direction. This establishes the compatible line and its
transverse obstruction at finite size. This plot is not required in the final
main-text Fig. 6 because it largely visualizes the analytical compatibility
condition.

### Positive finite-size thermal separation

For the cleaned microcanonical ensemble, the witness values remain positive
throughout the sampled positive interval. At \(L=12\), their variation over the
principal positive interval \(\kappa/J\in\{0.05,0.10,0.15,0.20\}\) is small:

\[
\tau_A^{\rm mc}\simeq0.1122\text{--}0.1129,\qquad
\tau_Z^{\rm mc}\simeq0.2192\text{--}0.2197,
\]

\[
\tau_Y^{\rm mc}\simeq0.3278\text{--}0.3282,
\]

while all cage values remain exactly zero.

At the representative point \(\kappa_\star/J=0.1\), the current grid gives

\[
(\tau_A,\tau_Z,\tau_Y)_{L=12}^{\rm mc}
=(0.112355,0.219336,0.328077),
\]

with tower residual \(1.3\times10^{-17}\) and joint-dark rank one.

### Matching distance is not yet extrapolated

At \(\kappa_\star/J=0.1\), define

\[
\Delta_L=\max_{\alpha\in\{A,Z,Y\}}
\left|\tau_{\alpha,L}^{\rm mc,clean}
-\tau_{\alpha,L}^{\beta=0,(M,k)}\right|.
\]

The current clean--full values are

\[
\Delta_6=0.0571,\quad
\Delta_8=0.0186,\quad
\Delta_{10}=0.0171,\quad
\Delta_{12}=0.0129.
\]

For the principal positive interval
\(\mathcal I_\kappa/J=\{0.05,0.10,0.15,0.20\}\), the envelope is

\[
\Delta_L^{\max}=0.0592,\ 0.0190,\ 0.0175,\ 0.0130
\quad (L=6,8,10,12).
\]

The larger \(L=6\) value \(0.0821\) from the full symmetric grid occurs at
the symmetry-enhanced endpoint \(\kappa=0\) and is retained only as a control.

The \(L=6\) point is pre-asymptotic: at \(\kappa_\star/J=0.1\), its window
contains only eight states before cleaning and seven afterward. Exclude it from
thermodynamic fits.

For \(L=8,10,12\), the data are compatible with an \(O(1/L)\) finite-window
correction, but the present three useful sizes do not distinguish zero limit
from a small positive intercept. Therefore:

- do not claim that \(\Delta_L\to0\) has been established;
- do not quote the existing unconstrained `a+b/L` or `a+b/L^2` fits as
  thermodynamic extrapolations;
- treat a zero-constrained \(c/L\) curve only as a guide to the eye until
  additional window scalings or a larger size are available.

### Complete two-site concentration

At \(\kappa_\star/J=0.1\), the block-invariant covariance widths are

\[
w_6=0.1826,\quad
w_8=0.1415,\quad
w_{10}=0.0619,\quad
w_{12}=0.0359.
\]

Over the principal positive interval, the maximum widths are

\[
w_L^{\max}=0.1836,\ 0.1457,\ 0.0619,\ 0.0359
\quad (L=6,8,10,12).
\]

The full-grid \(L=6\) maximum \(0.2091\) occurs at \(\kappa=0\) and is an
endpoint control rather than part of the principal deformation interval.

For the useful sizes \(L=8,10,12\), this is a strong decreasing trend. The
current fit forms are not controlled extrapolations. The data support strong
finite-size two-site concentration across the sampled positive interval, not yet a proved
uniform limit \(w_L^{\max}\to0\).

## Fig. 6 contract

### Panel (a): representative-point ETH scatter

The existing scatter is the legacy \(\kappa=0\) endpoint panel. Regenerate or
re-export it at \(H_{\rm rep}=H_{\kappa_\star}\), first for \(L=12\) and then
for \(L=14\) if the larger-size method provides the required eigenstate data.
Plot normalized fixed witnesses \(Q_A,Q_Z,Q_Y\), use the analytically
constructed tower vector for the star, shade the selected microcanonical
window, and state that the background is joint-dark cleaned.

Required replacement:

- `spin1_xy_kappa0p1_eth_scatter_Lmax.csv`.

### Panel (b): representative-point finite-size matching

Use \(\kappa_\star/J=0.1\). The main axes should show the cleaned
microcanonical values together with the resolved-\((M,k)\) \(\beta=0\) values;
the exact fixed-\(M\) thermodynamic limits may be horizontal targets.

The inset should show the individual distances

\[
\delta_{A,L},\quad\delta_{Z,L},\quad\delta_{Y,L},
\]

rather than only their maximum. Show \(L=6\) as a gray pre-asymptotic point or
omit it from the fit. Do not display a fitted zero intercept as established.
If a zero-constrained \(c/L\) line is retained, label it as a guide to the eye.

### Panel (c): family-wide thermal matching

Replace the ambient obstruction heatmap by a family-wide thermal diagnostic.
Preferred choices are

\[
\Delta_L(\kappa)
\quad\text{and/or}\quad
L\Delta_L(\kappa).
\]

The established conclusion is that matching quality is nearly uniform in
\(\kappa\) over the principal positive interval; the panel must not imply that the
thermodynamic limit is already known.

### Panel (d): complete two-site concentration

Plot

\[
w_L(\kappa)=\sqrt{\lambda_{\max}\Gamma_L(\kappa)}
\]

as a heatmap with \(\kappa/J\) on the horizontal axis and integer \(L\) on the
vertical axis. Caption it as a finite-size two-site concentration grid whose
interval-wide zero limit remains unresolved.

## Revised short-term numerical tasks

### P0. Lock and regenerate the representative point

Use

\[
\kappa_\star/J=0.1
\]

for the detailed Sec. VI and Fig. 6(a,b) analysis. The existing family-grid
files already contain the \(L=6,8,10,12\) witness, matching, concentration, and
joint-dark data at this point, so reuse or re-export them where possible rather
than recomputing blindly.

Regenerate the following representative-point products:

- joint-dark-cleaned ETH scatter at \(L=12\), and at \(L=14\) if feasible;
- analytical tower marker and residual at the same \(\kappa_\star\);
- raw--raw and clean--clean microcanonical--trace comparison;
- complete two-site covariance width and worst eigenoperator;
- exact window counts, energy blocks, and tolerance metadata.

Suggested outputs:

- `spin1_xy_kappa0p1_eth_scatter_Lmax.csv`;
- `spin1_xy_kappa0p1_exact_tower.csv`;
- `spin1_xy_kappa0p1_beta0_overlap.csv`;
- `spin1_xy_kappa0p1_concentration.csv`.

Retain \(\kappa=0\) as a chiral endpoint/control dataset. Do not use its
scatter as the principal representative panel once the \(\kappa_\star\) export
exists.

### P0. Consistent cleaned \(\beta=0\) comparison

The microcanonical ensemble is joint-dark cleaned. Regenerate the resolved
\((M,k)\) \(\beta=0\) trace with the same joint-dark projector removed:

\[
\rho_{\beta=0}^{\rm clean}
=\frac{P_{M,k}-P_{\rm jd}}
{\operatorname{Tr}(P_{M,k}-P_{\rm jd})}.
\]

Export raw--raw and clean--clean matching distances separately. The correction
is expected to be small at large \(L\), but the definitions must be consistent.

### P0. Window-scaling study at \(\kappa_\star\)

Repeat the representative-point matching calculation using at least

\[
\Delta E_L=cL^{1/2},\qquad
\Delta E_L=cL^{1/4},\qquad
\Delta E_L=c,
\]

with several prefactors \(c\). Record state counts, energy-density widths,
individual witness distances, and \(\Delta_L\). Require increasing retained
state count while \(\Delta e_L\to0\).

Suggested output:

- `spin1_xy_kappa0p1_beta0_window_scaling.csv`.

### P0. Revised representative-point extrapolation

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

- `spin1_xy_kappa0p1_matching_fit_revised.csv`;
- `spin1_xy_kappa0p1_matching_window_bootstrap.csv`;
- `spin1_xy_kappa_matching_scaled.csv` containing \(L\Delta_L(\kappa)\).

### P0. One larger-size point at the representative coupling

Run the primary \(L=14\) calculation at \(\kappa_\star/J=0.1\), using
shift-invert, filtered typicality, or another method that does not require a
full dense spectrum. This point must use the same \((M,k)\) sector, joint-dark
cleaning, window definitions, and complete two-site concentration diagnostic
as the smaller sizes.

Keep any already-running \(\kappa=0\) job as an endpoint control; it does not
replace the representative-point run. If resources permit, add \(L=14\) checks
at \(\kappa/J=0\) and \(0.2\) to constrain the interval-wide envelope.
Negative \(\kappa\) values may be used as an inversion-related implementation
check rather than a separate full production sequence.

### P1. Concentration extrapolation

Repeat the representative-point and interval-wide envelope analyses after
adding the larger size. Fit \(w_L(\kappa_\star)\) and \(w_L^{\max}\) using both
a zero-constrained decay and a free-intercept model. Report sensitivity to
excluding \(L=6\) and to the energy-block tolerance.

Suggested outputs:

- `spin1_xy_kappa0p1_concentration_fit_revised.csv`;
- `spin1_xy_kappa_uniform_concentration_fit_revised.csv`.

### P1. Degeneracy and symmetry audit

At \(\kappa_\star/J=0.1\), verify numerically that ordinary inversion and the
unitary \(C_A\) anticommutation are absent while \((M,k)\) remain exact. Also
verify the antiunitary relation
\(C_A\mathcal K H_\kappa(C_A\mathcal K)^{-1}=-H_\kappa\) in the selected
\(k=0\) or \(\pi\) sector. Export the zero-energy block multiplicity, the
energy-block tolerance used to construct \(P_EO_aP_E\), the number of blocks,
and the change in \(w_L\) under reasonable tolerance variation.

### P2. Optional extensions

The following remain secondary:

- finite diagonal-shell parameter \(\Delta\);
- a two-dimensional \((J_3,\kappa)\) concentration grid;
- finite-\(\beta\) deformation matching;
- the earlier \(Q=3\pi/4\) complex odd-range example;
- a larger-support concentration test beyond the complete two-site algebra.

## Acceptance criteria

Use the principal sampled interior interval

\[
\mathcal I_\kappa/J=\{0.05,0.10,0.15,0.20\},
\]

and define

\[
\Delta_L^{\max}=\max_{\kappa\in\mathcal I_\kappa}\Delta_L(\kappa),
\qquad
w_L^{\max}=\max_{\kappa\in\mathcal I_\kappa}w_L(\kappa).
\]

Treat \(\kappa=0\) as a separately resolved symmetry-enhanced endpoint control
and negative \(\kappa\) as inversion-related implementation checks.

The exact or finite-size statements already established are:

- exact tower continuation over the compatible line;
- exact zero cage values for the fixed witnesses;
- uniformly positive exact fixed-\(M\), \(\beta=0\) witness targets;
- positive finite-size cleaned microcanonical witness values across the sampled positive interval;
- a rank-one translated joint-dark kernel across the sampled positive interval;
- strongly decreasing finite-size two-site covariance widths.

A representative-point microcanonical ICQMBS conclusion requires controlled
evidence that

\[
\Delta_L(\kappa_\star)\to0,
\qquad
w_L(\kappa_\star)\to0.
\]

An interval-wide deformation-stable ICQMBS claim additionally requires
controlled evidence that

\[
\Delta_L^{\max}\to0,
\qquad
w_L^{\max}\to0,
\]

together with grid refinement or a continuity bound controlling the couplings
between the sampled points.

Until the revised extrapolations are available, describe the result as an exact
compatible caged family with uniformly positive \(\beta=0\) targets and strong
finite-size two-site ICQMBS evidence throughout the sampled positive interval.

## Plotting contract

- Render at final REVTeX physical dimensions.
- Use 9 pt base typography; inset and colorbar text must remain readable.
- Axes representing system size must use integer-only major ticks.
- Panels (a) and (b) must use \(\kappa_\star/J=0.1\) in the final export;
  keep \(\kappa=0\) as an endpoint control.
- Panel (b) must not imply that the current useful sizes prove a zero matching-distance intercept.
- Panel (c) must show family-wide thermal matching rather than the analytically trivial obstruction plane.
- Panel (d) must be captioned as a finite-size two-site concentration grid.
- Keep PDF and SVG exports and write the figure-dimension/font audit.
