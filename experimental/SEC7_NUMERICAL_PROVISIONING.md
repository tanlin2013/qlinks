# Section VII numerical provisioning cache

## Purpose

This file is the short-term qlinks handoff for the square-QDM fixed-width
storyline.  It is intentionally separate from `data/EVIDENCE_SUMMARY.md`:
the evidence summary records established results, whereas this file specifies
active scientific validations, acceptance criteria, failed alternatives, and
the reserved main-figure panels.

The manuscript now follows the same cumulative order as Sec. VI:

1. recover the compact cage and bounded symmetry-resolved kinetic witnesses
   `A_R,Z_R`;
2. promote the `4 x 4` non-gauge phase pilot to a size-independent
   circumference-four family and continue the same bounded local construction;
3. perform one fixed-width thermal test for the whole candidate family.

The first step is established.  The second and third are provisioning targets.
Do not describe the checkerboard Hamiltonian as an established fixed-width
family until the larger-strip compatibility, gauge-quotient, and common-sector
checks below pass.

The corrected shifted-shell `Y_R` is not part of the primary
symmetry-resolved family.  It belongs to the localized, translation-breaking
potential route retained in Appendix E.

## Scientific gates before the expensive phase scan

### Gate 1: fixed-width energy-density matching

The primary candidate retains a nonzero uniform flippability potential,
`lambda_star = 1`, to avoid the purely kinetic spectral-reflection structure.
The compact-cage and `beta=0` trace energies are independent of phase, but they
are not equal at finite length.  Before launching the full energy-resolved
checkerboard scan, determine whether

\[
e_{\psi,L_x}-e_{\beta=0,L_x}\to0
\]

on the `L_x x 4` sequence.

Use transfer counting, a transfer matrix, or another exact fixed-width method
to extend both flippability densities to the largest inexpensive `L_x`.
Export the finite-size sequence and compare constant, `1/L_x`, and `1/L_x^2`
limits.  If the difference approaches a nonzero constant, stop the `beta=0`
checkerboard plan and switch to an energy-matched finite-`beta` protocol or a
different cage-preserving potential.

Required output:

- `qdm_checkerboard_energy_density_match.csv`;
- `qdm_checkerboard_energy_density_fit.csv`.

### Gate 2: larger-strip transport of the checkerboard rule

For `L_x in {4,8,12}` initially, verify that the size-independent local pattern

\[
\chi_{x,y}=(-1)^{x+y}
\]

preserves the repeated compact cage and the fixed `A_R,Z_R` operators.  The old
`4 x 4` implementation used the Euclidean-unit-normalized vector `chi/4`; do
not tile that normalized vector, because its local entries would shrink with
system size.

For each strip:

1. construct the repeated compact cage analytically;
2. check every active boundary row and export the paired equal-phase
   constraints;
3. verify cage residual and energy;
4. verify fixed `A,Z` darkness with size-independent normalization;
5. compute the physical link-gauge image and the checkerboard distance from it;
6. identify a symmetry sector common to every sampled nonzero phase.

Required output:

- `qdm_checkerboard_fixed_width_family.csv`;
- `qdm_checkerboard_compatibility_constraints.csv`;
- `qdm_checkerboard_gauge_quotient.csv`;
- `qdm_checkerboard_common_symmetry_sector.csv`.

The family-level equations in Sec. VII become established only after this gate
passes.

### Gate 3: resolved-sector versus transfer target

The exact transfer targets are fixed-winding/fixed-width values.  The thermal
calculation will use a reduced-symmetry sector common to the checkerboard
family.  Keep these two traces distinct at finite size and verify that their
local `A,Z` values approach the same fixed-width limit.

Required output:

- `qdm_checkerboard_resolved_beta0_trace.csv`;
- `qdm_checkerboard_transfer_sector_overlap.csv`.

## Primary candidate Hamiltonian family

Label plaquettes on the `L_x x 4` strip by `p=(x,y)` with
`x in Z_{L_x}` and `y in Z_4`.  The candidate family is

\[
H_{\varphi}
=-J\sum_{p=(x,y)}\left[
 e^{i\varphi\chi_{x,y}}U_p
 +e^{-i\varphi\chi_{x,y}}U_p^\dagger
\right]
+\lambda_\star\sum_p F_p,
\qquad \lambda_\star=1.
\]

The paired cancellation plaquettes differ by two sites around the
circumference and therefore have equal `chi`.  This is a fixed-state
compatibility candidate, not by itself a proof of strip-wide continuation.

## Common symmetry resolution

The checkerboard phase breaks one-step translations but preserves period-two
translations.  Use only exact symmetries common to the full sampled nonzero
phase family:

- winding numbers `(W_x,W_y)`;
- `T_x^2` and `T_y^2` reduced momenta where useful;
- any additional symmetry only after explicitly verifying commutation with
  every `H_varphi` in the sampled family.

Do not use the full parent-model momentum labels at `varphi != 0`.  Construct
and retain separately:

- the localized compact cage;
- the cage projected into the chosen common reduced-symmetry sector.

Treat `varphi=0` as a symmetry-enhanced endpoint control, not automatically as
the representative thermal point.

## Choosing the representative phase and sampled grid

After Gates 1--3 and a small pilot scan, choose one nonzero interior
`varphi_star` where:

- the exact cage and `A,Z` residuals pass;
- the common reduced-symmetry window is well populated;
- no accidental extra jointly dark states appear;
- the stripe-local concentration diagnostic is smallest or otherwise regular;
- no additional symmetry is restored.

Initially scan positive phases, for example

`varphi in {0.025, 0.05, 0.075, 0.10}`.

Retain `varphi=0` as an endpoint control.  Retain negative phases only as
translation/complex-conjugation implementation checks when the corresponding
relation is verified.  Do not call a discrete grid an open interval.

## Primary bounded witnesses

The symmetry-resolved route uses only

\[
A_{R_x},\qquad Z_{R_x}=A_{R_x}+A_{R_x}^\dagger.
\]

`R_x` is one plaquette column but spans the adjacent physical vertex/link
columns `x,x+1` and wraps the circumference-four direction.  Verify exact
darkness on both the localized and reduced-symmetry cages at every transported
length and phase.

The localized shell

\[
Y_{R_x}^{\rm shell}
=\frac12\left(F_{(x,0)}+F_{(x,2)}-2\mathbf1\right)
\]

is dark on the localized cage but not after symmetry projection.  Exclude it
from the primary resolved-sector joint-dark inventory and thermal panels.

## Translated-witness inventory and implementation machinery

For the resolved kinetic route define

\[
Q_{\rm all}^{AZ}=\sum_x\left(Q_{R_x}^A+Q_{R_x}^Z\right).
\]

Inside each exact energy block, diagonalize
`P_E Q_all^{AZ} P_E` and retain its numerical kernel.  Treat exact
degeneracies as projectors.  Subtract the target compact-cage projector only
when reporting the residual non-target dark rank.  The cleaned microcanonical
ensemble removes the full declared joint-dark kernel.

Export raw and cleaned values, exact-energy tolerances, removed ranks and
fractions, and clean--clean comparisons.  These are Appendix A/E implementation
details; the Sec. VII main text should report only the physical inventory
result.

Required output:

- `qdm_checkerboard_joint_dark_kernel.csv`;
- `qdm_checkerboard_type1_continuation.csv`;
- `qdm_checkerboard_cleaning_audit.csv`.

The kernel is complete only relative to the translated `A,Z` witness family;
do not call it a projector onto all possible caged eigenstates.

## Energy-resolved matching sequence

After the scientific gates pass, for every available `L_x,varphi` use a
microcanonical window of the same Hamiltonian and common reduced-symmetry
sector.  Export:

- cage and `beta=0` trace energy densities;
- energy-density mismatch;
- raw and cleaned window counts;
- window half-width and alternative scaling/prefactor choices;
- raw and cleaned `tau_A,tau_Z`;
- reduced-sector `beta=0` traces;
- exact fixed-winding transfer targets;
- individual matching distances `delta_A,delta_Z`.

At least one third energy-resolved strip is required.  If full ED is
prohibitive, use filtered diagonalization or a controlled typicality method.
Two lengths must not be presented as a thermodynamic extrapolation.

Required output:

- `qdm_checkerboard_beta0_overlap.csv`;
- `qdm_checkerboard_fixed_width_shared_fit.csv`;
- `qdm_checkerboard_matching_distance_fit.csv`;
- `qdm_checkerboard_window_systematics.csv`.

## Complete stripe-local concentration

The witness stripe is constrained and has boundary-flux superselection
sectors.  Construct either:

1. a Hilbert--Schmidt-orthonormal basis of the complete projected operator
   algebra on the two-column stripe after quotienting null combinations; or
2. an equivalent block-resolved reduced-density-matrix distance.

Record the operator-space dimension and all boundary-flux block dimensions.
Do not reuse the current nine-operator basis as if it were complete.

After declared joint-dark removal, define the degenerate-block-invariant
covariance matrix and

\[
w_{L_x}(\varphi)=\sqrt{\lambda_{\max}\Gamma_{L_x}(\varphi)}.
\]

This directly tests the worst normalized observable on the witness stripe, not
all bounded longitudinal regions.  Export the median nonidentity width, worst
eigenoperator coefficients, exact-degeneracy tolerance audit, and reduced-
density-matrix cross-check where feasible.

Required output:

- `qdm_checkerboard_concentration_grid.csv`;
- `qdm_checkerboard_worst_eigenoperator.csv`;
- `qdm_checkerboard_uniform_concentration_fit.csv`.

## Family-wide acceptance criteria

For a sampled positive grid `P_varphi`, define

\[
\Delta_{L_x}^{\max}
=\max_{\varphi\in\mathcal P_\varphi}
\max_{\alpha\in\{A,Z\}}\delta_{\alpha,L_x}(\varphi),
\]

\[
w_{L_x}^{\max}
=\max_{\varphi\in\mathcal P_\varphi}w_{L_x}(\varphi).
\]

The primary numerical support required for strong sampled-family fixed-width
ICQMBS evidence on the witness stripe is

\[
\Delta_{L_x}^{\max}\to0,\qquad
w_{L_x}^{\max}\to0,
\]

with positive transfer targets and vanishing or thermodynamically negligible
residual non-target dark rank.  An open-phase-interval claim additionally
requires grid refinement or a continuity/error bound.  A broader statement
about all bounded local regions should be supported by at least one larger
longitudinal region or an explicit argument reducing those regions to the
witness-stripe algebra.

Until these criteria pass, use:

> candidate checkerboard family; sampled fixed-width compatibility and thermal
> test remain provisioned.

## Reserved main figure

### Panel (a): representative-point ETH scatter

At `varphi_star`, plot normalized `Q_A,Q_Z` against energy density in the
largest reliable common reduced-symmetry sector.  Use the analytically
constructed cage for the star and shade the matched window.

### Panel (b): fixed-width matching

Plot microcanonical `tau_A,tau_Z` versus integer `L_x`, the exact transfer
targets, and the controlled finite-size extrapolation.  Inset: individual
matching distances and window systematics.

### Panel (c): family-wide ensemble matching

Plot

\[
\Delta_{L_x}(\varphi)
=\max_{\alpha\in\{A,Z\}}\delta_{\alpha,L_x}(\varphi)
\]

across the sampled compatible phase grid.  Show the proposed finite-size
scaling without calling it established until a third length is available.
The obstruction plane belongs in Appendix E rather than the primary thermal
figure.

### Panel (d): stripe-local concentration

Plot `w_{L_x}(varphi)` with phase on the horizontal axis and integer `L_x` on
the vertical axis.

## Complementary localized potential route

For a translation-breaking potential pattern `v_p(g)`, retain the localized
cage and the adapted shell

\[
Y_R(g)=v_{p_1}(g)F_{p_1}+v_{p_2}(g)F_{p_2}
-[v_{p_1}(g)+v_{p_2}(g)]\mathbf1.
\]

Transport the current `4 x 4` path only as a complementary three-witness route.
Its larger-strip validation and thermal data do not define the primary
symmetry-resolved checkerboard claim.

Required output if pursued:

- `qdm_nonuniform_potential_fixed_width.csv`.

## Secondary controls

- Uniform Peierls phase: gauge/background-cleaning control only.
- Manual excision: protocol check only; all estimator details stay in the
  appendices.
- Collective quotient: the current negative result is relative to the tested
  plaquette-linear kinetic class; do not claim a general no-go theorem.
- True two-dimensional sequence: deferred until the fixed-width construction
  is complete.

## Plotting contract

- Render at final REVTeX dimensions.
- Use 9 pt base typography in the combined figure.
- Use integer-only major ticks for `L_x`.
- Use the same `A,Z` order and normalization in every panel.
- Export PDF and SVG plus figure-dimension/font audits.

## Implementation status (2026-08-06)

The qlinks workflow now implements the three gates as explicit, machine-readable
products and prevents the reserved thermal figure from being interpreted as a
thermodynamic result when fewer than three energy-resolved lengths are present.

- Gate 1 uses the exact circumference-four transfer calculation.  A direct
  preflight through `L_x=256` gives a preferred `Delta_inf+c/L_x` mismatch
  `Delta_inf ~= 5.98e-3`, above the locked `1e-3` tolerance.  Therefore the
  `auto` protocol switches the thermal pilot to an energy-matched finite-beta
  comparison.  The beta-zero comparison remains a separately labelled failed-
  gate diagnostic rather than being reused as the primary thermal reference.
- Gate 2 now exports the size-independent checkerboard transport certificates,
  paired phase constraints, gauge quotient, and common reduced-symmetry sector
  for `4x4`, `8x4`, and `12x4`.  Full sector projection is performed only at
  energy-resolved sizes; the `12x4` projection remains pending.
- Gate 3 and the reserved Fig. 7 pipeline run for the available full-ED lengths.
  The representative phase is explicitly labelled provisional until the
  production joint-dark and regularity filters pass.
- The primary inventory uses translated `A,Z` only.  The complete projected
  stripe algebra is constructed from its boundary-flux blocks and null
  combinations are removed before the covariance calculation.
- Dense repeat-3 (`12x4`) ED remains disabled because it can exceed the 400 GiB
  container limit.  The command-line `--run-large-strip` flag fails explicitly
  until a controlled partial-spectrum or typicality implementation is supplied;
  it never silently falls back to dense ED.

The current smoke validation is not claim-level evidence.  In particular, its
`4x4` pilot has residual non-target translated-`A,Z` dark rank, so the requested
phase is not yet a locked manuscript representative point.  Production at
`4x4` and `8x4`, followed by a controlled third-length method, is still required.
