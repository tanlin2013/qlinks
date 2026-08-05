# Section VI numerical provisioning cache

## Purpose

This file is the short-term handoff for the restructured spin-1 XY storyline.
It is intentionally separate from established evidence summaries: this cache
records calculations, acceptance criteria, failed attempts, and figure choices
that remain active for Sec. VI and Fig. 6.

The target logic is

1. recover the known \(Q=\pi\) bimagnon tower;
2. define a generic thermal reference Hamiltonian;
3. establish reference-point microcanonical--\(\beta=0\) matching;
4. continue the same exact caged family along a complex-Hermitian compatible
   deformation;
5. expose the transverse obstruction in the ambient parameter plane;
6. establish matching and complete local-algebra concentration uniformly along
   the compatible path;
7. classify the result as a deformation-stable ICQMBS family, subject to the
   exceptional-subspace inventory and finite-size extrapolation.

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
H(J_3,\kappa)=K_1(J)+K_3(J_3)+K_2(i\kappa),
\]

or explicitly

\[
H(J_3,\kappa)=J\sum_r(S_r^+S_{r+1}^-+\mathrm{h.c.})
+J_3\sum_r(S_r^+S_{r+3}^-+\mathrm{h.c.})
+i\kappa\sum_r(S_r^+S_{r+2}^- -\mathrm{h.c.}).
\]

For the main deformation protocol, hold \(J_3/J=0.1\) fixed and vary
\(\kappa/J\).  The staggered tower phase is \(\eta_r=(-1)^r\).  Real odd-range
and purely imaginary even-range exchanges separately obey

\[
t_d^*+(-1)^dt_d=0,
\]

so the entire tower remains at exactly zero energy along the compatible line.
No diagonal deformation is included in the main protocol.

## Common symmetry resolution

Use only symmetries common to the complete family:

- fixed total magnetization \(M\);
- translation momentum \(k\).

Do not resolve inversion in the primary reference or deformation sequences,
because the imaginary second-neighbor exchange breaks ordinary inversion.  The
central tower momentum must be computed from its number of raised bimagnons.
Record fixed-\(M\), resolved-\((M,k)\), and microcanonical values separately.

## Exceptional projector

Define the primary exceptional projector basis-independently through the
translated joint-dark positive operator

\[
Q_{\rm all}=\sum_R(Q_{A,R}+Q_{Z,R}+Q_{Y,R}).
\]

Within every exact energy block, diagonalize \(P_EQ_{\rm all}P_E\) and retain
its numerical kernel.  Use the resulting joint-dark subspace for cage excision.
Run the direct Type-1 search only at the bipartite reference point
\(\kappa=0\).  The imaginary even-range exchange invalidates the ordinary
bipartite Fock-graph grading used by that searcher, so away from the reference
point continue and retest the reference Type-1 states rather than rerunning the
search.  New exceptional states are detected by the translated joint-dark
kernel.  Subtract the target-tower projector and report any remaining dark
rank.

## Panel (a): reference ETH scatter

At \(H_{\rm ref}\), use the largest reliable fully diagonalized \((M,k)\)
sector.  Plot normalized \(Q_A,Q_Z,Q_Y\) against energy density.  Use the
analytically constructed tower vector for the star and shade the selected
microcanonical energy window.

Required output:

- `spin1_xy_cage_excised_eth_scatter.csv`;
- sector labels, tower residual, window definition, and joint-dark labels.

## Panel (b): reference matching and extrapolation

For each available \(L\), retain distinct columns for

- cage-excised microcanonical values;
- exact resolved-\((M,k)\) \(\beta=0\) traces;
- exact fixed-\(M\) \(\beta=0\) counting values;
- matching distances and window-prefactor variants.

Compare at least \(1/L\) and \(1/L^2\) diagnostic fits.  Do not identify the
finite-size resolved and fixed-\(M\) traces; only their common thermodynamic
target is claimed.

Required outputs:

- `spin1_xy_beta0_cage_excised_overlap.csv`;
- `spin1_xy_beta0_shared_asymptote_fit.csv`;
- `spin1_xy_beta0_matching_distance_fit.csv`.

## Panel (c): complex-\(t_2\) obstruction plane

Embed the compatible line in

\[
t_2/J=u+iv,
\]

at fixed \(J_3/J=0.1\).  The main axes are the finite-size normalized tower
residual over the \((u,v)\) plane.  Overlay the exact compatible line \(u=0\).
The inset shows

\[
\delta_L(\kappa)=\max_{\alpha\in\{A,Z,Y\}}
|\tau_{\alpha,L}^{\rm mc}(\kappa)-
\tau_{\alpha,L}^{\beta=0,(M,k)}(\kappa)|
\]

along \(v=\kappa/J\).

Required outputs:

- `spin1_xy_complex_t2_obstruction_grid.csv`;
- `spin1_xy_complex_t2_obstruction_jacobian.csv`;
- `spin1_xy_kappa_matching_grid.csv`.

## Panel (d): complete local-algebra concentration

Use a Hilbert--Schmidt-orthonormal basis of the complete two-site Hermitian
algebra preserving two-site total \(S^z\).  Its block dimensions are
\((1,2,3,2,1)\), hence its dimension is 19.  Include the normalized identity
only as a zero-width completeness check.

Treat exact degeneracies as projectors.  After joint-dark excision, define the
block-invariant covariance quadratic form from compressed operators
\(P_EO_aP_E\).  Plot

\[
w_L(\kappa)=\sqrt{\lambda_{\max}\Gamma_L(\kappa)}
\]

as a heatmap with \(\kappa/J\) on the horizontal axis and integer system size
\(L\) on the vertical axis.  Export the median nonidentity covariance width and
the coefficients of the worst-concentrated eigenoperator.

Required outputs:

- `spin1_xy_kappa_concentration_grid.csv`;
- `spin1_xy_kappa_worst_eigenoperator.csv`;
- `spin1_xy_kappa_uniform_concentration_fit.csv`.

## Initial grids

Use the pilot compatible path

`kappa/J in {-0.20,-0.15,-0.10,-0.05,0,0.05,0.10,0.15,0.20}`.

Both signs are retained as an inversion-related implementation check, although
the final manuscript may display only one half.  Search the Type-1 inventory at
`kappa/J=0`, then continuation-test those reference states at
`kappa/J in {0.10,0.20}`.  The complex residual plane
initially uses `Re t2/J, Im t2/J in [-0.20,0.20]`.

## Interval-wide acceptance criteria

For a sampled interval \(I\), export

\[
\Delta_L^{\max}=\max_{\kappa\in I}\delta_L(\kappa),\qquad
w_L^{\max}=\max_{\kappa\in I}w_L(\kappa).
\]

The numerical support required for an interval-wide deformation-stable ICQMBS
claim is

\[
\Delta_L^{\max}\to0,\qquad w_L^{\max}\to0,
\]

while the exact fixed-\(M\) thermal witness targets remain uniformly positive.
Until this extrapolation and the remaining joint-dark ranks are satisfactory,
state the result as a sampled compatible-family test rather than a proven open
parameter neighborhood.

## Plotting contract

- Render at final REVTeX physical dimensions.
- Multi-panel base typography should be 9 pt rather than the previous 8 pt;
  inset and colorbar text should remain readable at final printed size.
- Axes representing manifestly integral variables, especially system size
  \(L\), must use integer-only major ticks.
- Panel (c) reserves the residual heatmap for the main axes and
  \(\delta_L(\kappa)\) for the inset.
- Keep PDF and SVG exports and write the figure-dimension/font audit.

## Deferred tracks

The following are not prerequisites for the present implementation:

- finite diagonal-shell parameter \(\Delta\);
- a two-dimensional \((J_3,\kappa)\) concentration grid;
- finite-\(\beta\) deformation matching;
- the earlier \(Q=3\pi/4\) complex odd-range example.
