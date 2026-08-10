# Section VII numerical provisioning cache

**Updated:** 2026-08-10
**Authoritative production job:** `data/evidence_jobs/qdm_checkerboard_finite_beta_20260807T171706Z/`

## Purpose

This is the remaining qlinks handoff for the square-QDM checkerboard family
after the Sec. VII skeptical-referee audit.  The exact local construction is
stronger than the present thermal evidence: the four-covering compact motif,
its repeated periodic-product cage, the bounded kinetic `A_R,Z_R` operators,
and checkerboard transport are verified on the tested strips.  The unresolved
work is the **fully symmetry-resolved raw fixed-width thermal limit**.

Two changes from the previous handoff are important.

1. The `L_x=4,8` thermal data were formed in the common
   `(T_x^2,T_y^2)=(1,1)` sector.  For nonzero checkerboard phase,
   `T_x T_y` is also an exact translation symmetry because
   `(-1)^(x+y)` is unchanged by `(x,y)->(x+1,y+1)`.  The current spectra are
   therefore only partially symmetry resolved under the Sec. III protocol.
2. The exported 50-dimensional stripe covariance uses joint-dark deletion.
   The defining ICQMBS background test is the **raw** window; a raw covariance
   companion is required.  Cleaned data remain diagnostics only.

## Locked checkerboard family

On an `L_x x 4` strip,

\[
H_\varphi=-J\sum_{p=(x,y)}\left[
 e^{i\varphi(-1)^{x+y}}U_p+e^{-i\varphi(-1)^{x+y}}U_p^\dagger
\right]+\lambda_\star\sum_p F_p,
\qquad \lambda_\star/J=1.
\]

Use the size-independent local phase amplitude `varphi`.  The primary resolved
witnesses are the kinetic `A_R,Z_R`; the localized shifted-shell `Y_R` remains
an Appendix-E complementary route.

The `4x4` four-covering state is the **compact motif**.  The fixed-width sequence
repeats this motif and has growing global configuration support (`4,16,64,256`
on the tested undeformed `4N x 4` sequence); call it the **periodic-product
caged sequence**, not a globally compact four-covering cage.

Use the representative phase

\[
\varphi_\star=0.05
\]

unless the fully resolved third-size calculation reveals an anomaly.  Treat
`varphi=0` as a symmetry-enhanced endpoint/control.  The principal deformation
interval uses the positive grid `0.025,0.05,0.075,0.10`.

## Gauge qualification

The numerical quotient uses local physical link rephasings

\[
G(\theta)=\exp\!\left(i\sum_\ell \theta_\ell n_\ell\right),
\]

and the induced link-to-plaquette phase map.  On `L_x=4,8,12` the checkerboard
phase vector has relative distance one from this image.  In manuscript language,
**non-gauge means outside this declared local link-rephasing image**.  Do not
silently upgrade this to a statement about arbitrary configuration-dependent
diagonal unitaries.

## Completed scientific gates

### Local transport and checkerboard compatibility

Verified at `L_x=4,8,12` over `varphi in {0,0.025,0.05,0.075,0.10}`:

- periodic-product cage residuals are `O(1e-15)`;
- `E_psi=lambda_star L_x`, hence `e_psi/J=1/4` at `lambda_star/J=1`;
- fixed `A_R,Z_R` residuals are at machine precision on the tested construction;
- every active paired boundary row satisfies the equal-phase rule;
- the checkerboard direction has relative distance one from the tested local
  link-rephasing image.

The four-covering compact motif itself has an explicit algebraic boundary
certificate.  The larger-strip checkerboard continuation is verified on the
listed sizes; an analytic all-`L_x` theorem remains stronger than this tested
sequence.

### Selective cleanup and dark-manifold classification

At nonzero checkerboard phase on the `4x4` reference inventory, the eight
compact `(0,4)` representatives survive, while the collective ninth `(0,4)`
representative and the `(0,6)` cage are lifted.  In the **partially resolved**
common `(T_x^2,T_y^2)` sector at `varphi_star=0.05`, the projected compact
Type-I span and translated `A,Z` joint-dark kernel both have rank four at
`L_x=4,8`, with unexplained norm below `4e-12`.

This classification must be repeated after full checkerboard-translation
resolution before it is used as the thermodynamic caged-subspace count.

### `beta=0` route rejected

Exact transfer counting reaches `L_x=256`.  Fitting the large sizes to
`Delta_e(L_x)=Delta_e_inf+c/L_x` gives

\[
e_{\beta=0}-e_\psi\to 5.98\times10^{-3},
\]

well above the `1e-3` matching gate.  Alternative constant and `1/L_x^2` fits
give positive limits of the same order.  Use `beta=0` only as an exact
baseline/control.  The primary thermal protocol is energy-matched finite beta.

### Present finite-beta evidence in the partially resolved common sector

At `varphi_star=0.05`, raw finite-size energy matching gives

\[
\beta_4J=0.143892,\qquad \beta_8J=0.074532.
\]

At window prefactor `0.75`, the raw microcanonical values are

\[
(\tau_A,\tau_Z)_{mc,raw}=(0.06188,0.11759)\quad (L_x=4),
\]

\[
(\tau_A,\tau_Z)_{mc,raw}=(0.04836,0.09498)\quad (L_x=8),
\]

while the raw canonical values are

\[
(0.07046,0.13154),\qquad (0.05091,0.09911).
\]

The representative raw--raw maximum mismatch decreases

\[
1.395\times10^{-2}\to4.13\times10^{-3}.
\]

Across the positive phase grid, the raw matching envelope decreases from about
`1.40e-2` to `4.15e-3`.

The joint-dark rank is four at both completed sizes.  In the central raw window,
the removed fraction decreases from `4/29 = 0.138` to
`4/1723 = 2.32e-3`.  The clean--clean representative mismatch decreases from
`9.44e-3` to `4.12e-3`.  These cleaned values are diagnostic only.

### Present stripe-local concentration

The complete projected Hermitian algebra on the declared two-column witness
stripe has dimension 50.  The currently exported covariance is **joint-dark
cleaned**, not raw.  At `varphi_star=0.05`,

\[
w_4^{clean}=0.06538,\qquad w_8^{clean}=0.01943.
\]

The positive-grid envelope is essentially the same.  This is strong finite-size
stripe-local evidence, but it is not yet the raw concentration condition of
Definition III.2 and, even if it vanishes, it establishes concentration only on
this stripe rather than every bounded region.

## P0 tasks

### P0.1 -- fully resolve the checkerboard symmetry sector

For every **positive** checkerboard phase used in the primary family, resolve
the full translation subgroup preserving the pattern.  At minimum this means
including the exact diagonal translation `T_x T_y` in addition to the existing
`T_x^2,T_y^2` information.  Construct one fixed irrep containing the projected
periodic-product cage and use the same irrep convention at `L_x=4,8,12`.

Required outputs:

- explicit commuting translation generators and their orders/relations;
- selected irrep labels;
- fully resolved sector dimension at `L_x=4,8,12`;
- cage projection norm and projected `Q_A,Q_Z` residuals;
- audit of any additional point-group symmetry commuting within the selected
  translation irrep; resolve it or document why it does not create a further
  block for the thermal calculation.

Do **not** use `varphi=0` to define the deformation-wide reference sector; it is
an enhanced-symmetry endpoint and should be analyzed separately.

Update or supersede:

- `qdm_checkerboard_common_symmetry_sector.csv`.

### P0.2 -- fully resolved dark-manifold classification

Repeat the projected compact-Type-I versus translated-`A,Z` joint-dark
comparison in the fully resolved checkerboard sector at `L_x=4,8,12`.
Export:

- projected periodic-product caged-subspace rank;
- joint-dark rank;
- unexplained joint-dark norm;
- raw-window caged fraction at the central window.

Update:

- `qdm_checkerboard_joint_dark_vs_type1.csv`;
- `qdm_checkerboard_compact_dark_manifold.csv`.

### P0.3 -- third raw energy-resolved strip and overlap recheck

In the fully resolved sector, recompute `L_x=4,8` as overlap checks and obtain
the `L_x=12` same-Hamiltonian finite-beta comparison at
`varphi_star=0.05`.  If full ED is prohibitive at `12x4`, use a controlled
filtered/shift-invert, Krylov/typicality, or transfer-assisted method.

The **raw microcanonical window is the defining ETH reference**.  Export:

- raw energy-matched beta;
- raw microcanonical `A,Z` values around `E_psi`;
- raw canonical `A,Z` values in the same fully resolved sector;
- raw--raw matching distances;
- cleaned companions only as diagnostics;
- raw/clean window counts and removed fractions;
- exact window coverage or a solver convergence/coverage audit;
- eigenpair-budget/method systematics where partial spectrum methods are used.

Update:

- `qdm_checkerboard_finite_beta_energy_match.csv`;
- `qdm_checkerboard_thermal_overlap.csv`;
- `qdm_checkerboard_window_systematics.csv`.

### P0.4 -- raw 50-dimensional stripe concentration

Compute the complete 50-dimensional stripe covariance on the **raw** fully
resolved window at `L_x=4,8,12`.  Keep the current cleaned covariance in
parallel.  Export:

- `w_raw` and `w_clean`;
- worst raw and cleaned eigenoperators;
- energy-block tolerance audit;
- exact window coverage/state count;
- raw/clean difference and removed fraction.

Update:

- `qdm_checkerboard_concentration_grid.csv`;
- `qdm_checkerboard_worst_eigenoperator.csv`.

### P0.5 -- thermodynamic finite-beta target and entropy density

Use a fixed-width transfer/canonical method at large `L_x`.  The target must be
consistent with the fully resolved positive-phase symmetry sector.  In the same
calculation:

1. certify positive entropy density of the resolved reference space by
   extracting the exponential growth rate of its sector count/dimension;
2. determine the thermodynamic inverse temperature `beta_*` satisfying

\[
e_{can}(\beta_*)/J=1/4;
\]

3. obtain

\[
\tau_A^{can}(\beta_*),\qquad \tau_Z^{can}(\beta_*);
\]

4. verify whether the thermodynamic local target is phase independent on the
   positive checkerboard family; if not, export it phase by phase.

The existing `qdm_checkerboard_finite_beta_transfer_target.csv` contains only
finite-size `L_x=4,8` partially resolved targets.  It is **not** the final
thermodynamic transfer result.

### P0.6 -- controlled fixed-width extrapolation and final figure

After P0.1--P0.5, fit the representative and positive-phase sequences with
justified zero/nonzero-intercept forms and window/method systematics.  The final
four-panel figure should use:

- (a) fully resolved raw ETH scatter, with any cleaned view clearly identified
  as a secondary overlay/inset;
- (b) raw microcanonical/canonical sequence plus thermodynamic finite-beta
  target;
- (c) positive-phase **raw--raw** matching envelope;
- (d) **raw** 50-dimensional stripe-local concentration, with cleaned width
  optionally shown as a diagnostic.

## P1 tasks

1. **Phase-grid refinement / continuity** for an open positive-`varphi`
   interval claim.
2. **Larger bounded longitudinal regions** or an analytic upgrade argument if
   the paper wants the full all-bounded-region background condition of
   Definition III.2 rather than stripe-local evidence.
3. **Gauge-quotient analytic note:** derive the local link-rephasing incidence
   map and an analytic invariant showing why the checkerboard vector lies
   outside its image.  Optionally audit whether a broader nonlocal
   configuration-basis diagonal unitary makes the family isospectral; do not
   call such a transformation a local physical gauge without proof.
4. **Nonuniform-potential / adapted `Y_R` route** as a complementary Appendix-E
   realization.
5. **True 2D sequence** only after the fixed-width family is complete.

## Claim boundary

Already established on the tested sequence:

- exact `4x4` compact-motif boundary certificate;
- machine-precision periodic-product cage and bounded `A,Z` transport residuals through `12x4`;
- checkerboard phase lies outside the declared local link-rephasing image on
  `4x4,8x4,12x4`;
- selective lifting of the collective/secondary `4x4` cages at nonzero phase;
- rejection of the `beta=0` matched-thermal route at `lambda_star/J=1`;
- in the partially resolved common sector at `L_x=4,8`, positive raw `A,Z`
  values and decreasing raw microcanonical--canonical mismatch;
- strong narrowing of the **cleaned** complete 50-dimensional stripe width.

Still provisioned before a deformation-stable fixed-width ICQMBS claim:

- fully resolved positive-phase checkerboard symmetry sector;
- fully resolved caged/joint-dark rank and vanishing caged fraction;
- `12x4` raw thermal and raw concentration data;
- thermodynamic finite-beta target and positive entropy density;
- controlled raw matching/concentration extrapolation;
- concentration beyond the single tested stripe or an upgrade argument;
- grid refinement/continuity for an open phase interval.

No fixed-width result implies a true 2D ICQMBS.

---

## qlinks implementation status (2026-08-10)

The skeptical-referee P0 corrections are now encoded in the evidence workflow.

### Exact all-`4N x 4` periodic-product theorem

The periodic-product sequence is no longer treated merely as a machine-precision
cross-size eigenpair check.  A dedicated algebraic certificate verifies the
four-column motif using only binary plaquette patterns, exact integer relative
signs, and formal checkerboard phase monomials.  It establishes:

- only plaquette columns `x=0,2 (mod 4)` are active;
- columns `x=1,3 (mod 4)` are inactive on every support configuration, including
  the plaquette column crossing one four-column motif into the next;
- the two active local routes have opposite exact coefficients and equal
  checkerboard phase because `chi(x,y+2)=chi(x,y)`;
- the kinetic action therefore cancels identically for arbitrary real `varphi`;
- every support configuration has exactly four flippable plaquettes per motif.

Hence for every positive integer `N`, on the `4N x 4` torus,

\[
H_\varphi |\Psi_N\rangle = 4N\lambda_\star |\Psi_N\rangle
= L_x\lambda_\star |\Psi_N\rangle,
\]

with no finite-size tolerance.  The generated evidence products are
`qdm_checkerboard_exact_periodic_product_certificate.csv` and
`qdm_checkerboard_exact_periodic_product_proof.md`.

### P0.1 full positive-phase translation irrep

The pattern-preserving translation subgroup is

\[
\mathcal T_\chi=\{T_x^aT_y^b: a+b\equiv0\pmod2\},
\]

of order `2 L_x`, generated by `Tdiag=T_x T_y` and `Ty2=T_y^2`.  The evidence
workflow uses the deformation-wide generic character

\[
T_{\rm diag}=e^{i\pi/2},\qquad T_y^2=+1,
\]

for every `L_x=4N`.  The projected periodic-product state has nonzero norm in
this irrep.  At `L_x=4,8` the fully resolved dimensions are `15` and `1125`,
respectively; `L_x=12` is constructed by the sparse large-strip lane.

At fixed positive phase the shifted reflections and `C2` are exact point-group
symmetries, but they exchange the `kdiag=+pi/2` and `-pi/2` translation irreps.
Thus the selected generic translation irrep has a trivial unitary point-group
little group and needs no additional point-group block.  Bare reflections map
`varphi -> -varphi` and are not fixed-positive-phase symmetries.

### P0.2--P0.4 raw-background convention

All finite-beta thermal and 50-operator concentration products are regenerated
in the fully resolved checkerboard translation irrep.  The raw microcanonical
window and raw covariance are the defining background diagnostics.  Joint-dark
cleaning remains a companion finite-size diagnostic only.  Large-strip
execution is gated on the corrected small-size symmetry/dark/concentration
checks before the `12x4` sparse lane is entered.

### Concentration quotient audit

The full symmetry correction exposes additional null combinations in the
localized stripe algebra.  The constrained two-column algebra still has 51
Hermitian basis directions including the identity (50 nonidentity directions),
but after restricting to the selected fully resolved translation irrep the map
`O -> P_irrep O P_irrep` can have a nontrivial kernel.  The implementation now
quotients this *linear* kernel using the projected-action Gram matrix while
retaining the Hilbert--Schmidt norm inherited from the original local algebra.
It does not globally renormalize `P O P`.

In the `4x4`, `varphi=0.05` smoke validation, the 51-dimensional ambient
algebra has projected quotient rank 20 (19 nonidentity directions).  The raw
and cleaned worst widths in this corrected quotient are approximately
`0.12245` and `0.10820`, respectively.  These values supersede concentration
numbers obtained from the partially symmetry-resolved sector or from a
linearly dependent projected operator list.  The production `8x4` and `12x4`
results must determine the corresponding quotient ranks before any scaling
claim is made.
