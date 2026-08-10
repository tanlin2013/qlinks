# Section VII numerical provisioning cache

**Updated:** 2026-08-09
**Authoritative production job:** `data/evidence_jobs/qdm_checkerboard_finite_beta_20260807T171706Z/`

## Purpose

This is the remaining qlinks handoff for the square-QDM checkerboard family.
The local deformation, non-gauge character, compact dark-manifold
classification, and the complete `L_x=4,8` finite-beta/stripe-local diagnostics
are now established on the tested sizes. The unresolved work is the large-
strip thermal limit.

## Locked checkerboard family

On an `L_x x 4` strip,

\[
H_\varphi=-J\sum_{p=(x,y)}\left[
 e^{i\varphi(-1)^{x+y}}U_p+e^{-i\varphi(-1)^{x+y}}U_p^\dagger
\right]+\sum_p F_p.
\]

Use the size-independent local phase amplitude `varphi`. The primary resolved
witnesses are the kinetic `A_R,Z_R`; the localized shifted-shell `Y_R` remains
an Appendix-E complementary route.

Use the representative phase

\[
\varphi_\star=0.05
\]

unless the third-size calculation reveals an anomaly.

## Completed scientific gates

### Checkerboard transport and gauge quotient

Verified at `L_x=4,8,12` over `varphi in {0,0.025,0.05,0.075,0.10}`:

- compact-cage residuals are `O(1e-15)`;
- `E_psi=L_x`, so `e_psi=1/4`;
- fixed `A_R,Z_R` are exactly dark;
- every active paired boundary row satisfies the equal-phase rule;
- the checkerboard direction has relative distance one from the tested
  physical link-gauge image.

The common `(T_x^2,T_y^2)` sector is explicitly constructed at `L_x=4,8`;
`L_x=12` sector projection remains the large-strip task.

### Selective cleanup and dark-manifold classification

At nonzero checkerboard phase, the locally generated compact `(0,4)` cages
survive while the collective/secondary finite-size records are lifted or
project outside the selected compact sector. At `varphi_star=0.05`, the
projected compact Type-I span and translated `A,Z` joint-dark kernel both have
rank four at `L_x=4` and `8`, with unexplained norm below `4e-12`. Thus the
surviving dark manifold is classified on the two energy-resolved strips.

### `beta=0` route rejected

For `lambda_star=1`, exact transfer counting reaches `L_x=256` and gives

\[
e_{\beta=0}-e_\psi\to 5.98\times10^{-3}\neq0.
\]

Use `beta=0` only as an exact baseline/control. The primary thermal protocol is
energy-matched finite beta.

### Finite-beta evidence at `L_x=4,8`

In the common reduced sector the raw finite-size energy matches are

\[
\beta_4=0.143892,\qquad \beta_8=0.074532.
\]

For the clean--clean comparison, `L_x=8` uses `beta_clean=0.074956`; at
`L_x=4` raw and clean values coincide at the quoted precision. The canonical
targets are phase independent within numerical precision over the sampled
checkerboard grid.

At `varphi_star=0.05` and window prefactor `0.75`,

\[
(\tau_A,\tau_Z)_{\rm mc}=(0.071781,0.136407)\quad (L_x=4),
\]

\[
(\tau_A,\tau_Z)_{\rm mc}=(0.048463,0.095158)\quad (L_x=8),
\]

while the matched clean canonical values are

\[
(0.078125,0.145846),\qquad (0.051011,0.099275).
\]

The representative clean--clean maximum mismatch decreases

\[
0.009439\to0.004116.
\]

Over the full sampled phase grid at the same window convention,

\[
\Delta_4^{\max}\simeq9.50\times10^{-3},\qquad
\Delta_8^{\max}\simeq4.14\times10^{-3}.
\]

### Complete stripe-local concentration

The complete projected Hermitian algebra on the declared witness stripe has
dimension 50. At `varphi_star=0.05`,

\[
w_4=0.06538,\qquad w_8=0.01943.
\]

The positive-grid envelope is essentially the same (`about 0.0654 -> 0.0194`).
Two lengths are not sufficient for a thermodynamic fit.

## P0 tasks

### P0.1 -- `12x4` common-sector projection

Complete the `(T_x^2,T_y^2)=(1,1)` sector construction at `L_x=12` and export:

- sector dimension;
- compact-cage projection norm;
- projected `Q_A,Q_Z` residuals;
- projected compact Type-I rank;
- translated joint-dark rank and unexplained norm.

Extend:

- `qdm_checkerboard_common_symmetry_sector.csv`;
- `qdm_checkerboard_joint_dark_vs_type1.csv`.

### P0.2 -- third energy-resolved strip

At `varphi_star=0.05`, obtain the `L_x=12` same-Hamiltonian finite-beta thermal
comparison. If full ED is prohibitive, use a controlled filtered/shift-invert,
Krylov/typicality, or transfer-assisted method.  The raw microcanonical window
is the defining ETH reference; cleaned quantities remain diagnostic. It must
provide:

- energy-matched beta (raw and, if cleaning changes the target, clean);
- raw and, where used, cleaned microcanonical `A,Z` values around `E_psi`;
- matched canonical `A,Z` values;
- clean--clean matching distances;
- window coverage/state counts and method systematics;
- compact dark-manifold cleaning in the same common sector.

Update:

- `qdm_checkerboard_finite_beta_energy_match.csv`;
- `qdm_checkerboard_thermal_overlap.csv`;
- `qdm_checkerboard_window_systematics.csv`.

### P0.3 -- `12x4` complete 50-dimensional stripe concentration

Compute the same block-invariant covariance width at `L_x=12` in a controlled
raw window around the cage energy.  If the present implementation removes the
compact joint-dark manifold before forming the covariance, also export the raw
companion or a controlled removed-fraction bound. Export the worst
eigenoperator, tolerance audit, exact window coverage, and the resolved-sector
dimension needed for the positive-entropy-density check. Update

- `qdm_checkerboard_concentration_grid.csv`;
- `qdm_checkerboard_worst_eigenoperator.csv`.

### P0.4 -- thermodynamic finite-beta target

Use a fixed-width transfer/canonical method at large `L_x`.  In the same
calculation, certify that the resolved common reference space has positive
entropy density by extracting the exponential growth rate of its sector
count/dimension.  Then determine the thermodynamic inverse temperature
`beta_*` satisfying

\[
e_{\rm can}(\beta_*)=1/4,
\]

and the corresponding

\[
\tau_A^{\rm can}(\beta_*),\qquad \tau_Z^{\rm can}(\beta_*).
\]

The existing `qdm_checkerboard_finite_beta_transfer_target.csv` contains only
exact finite-size `L_x=4,8` common-sector targets; it is **not** yet the
thermodynamic transfer result.

Verify whether the thermodynamic target is phase independent; if not, export
it phase by phase.

### P0.5 -- controlled fixed-width extrapolation and final figure

After P0.1--P0.4, fit representative and phase-wide sequences with justified
zero/nonzero-intercept forms and window/method systematics. With a third
regular point, freeze `varphi_star=0.05` and regenerate the four-panel figure:

- (a) representative ETH scatter;
- (b) finite-beta microcanonical/canonical sequence plus thermodynamic target;
- (c) phase-wide finite-beta matching distance;
- (d) complete 50-dimensional stripe-local concentration.

## P1 tasks

1. **Phase-grid refinement / continuity** for an open phase-interval claim.
2. **Larger bounded longitudinal region** only if a broader background-ETH
   statement than stripe-local concentration is needed.
3. **Nonuniform-potential / adapted `Y_R` route** as a complementary Appendix-E
   realization.
4. **True 2D sequence** only after the fixed-width family is complete.

## Claim boundary

Already established on the tested sequence:

- exact non-gauge checkerboard compact-cage and `A,Z` continuation through
  `12x4`;
- classification of the surviving joint-dark manifold at `L_x=4,8`;
- rejection of the `beta=0` matched-thermal route at `lambda_star=1`;
- positive finite-beta `A,Z` thermal values and decreasing microcanonical--
  canonical mismatch at `L_x=4,8`;
- strong narrowing of the complete 50-dimensional stripe-local concentration
  width at `L_x=4,8`.

Still provisioned:

- `12x4` common-sector thermal and concentration data;
- the thermodynamic finite-beta target;
- controlled fixed-width matching/concentration extrapolation;
- grid refinement/continuity for an open phase interval.

No fixed-width result implies a true 2D ICQMBS.
