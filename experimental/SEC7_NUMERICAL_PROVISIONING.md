# Section VII numerical provisioning cache

**Updated:** 2026-08-11
**Authoritative production job:** `data/evidence_jobs/qdm_checkerboard_fullsym_finite_beta_20260810T164206Z/`

## Purpose

This is the remaining qlinks handoff for the square-QDM checkerboard family.
The conceptual deformation problem and the positive-phase symmetry-resolution
problem are now closed.  The remaining work is concentrated in the
**large-strip thermal limit**: persist the already-computed `12x4` canonical
target, replace the memory-heavy direct-LU shift-invert step, obtain the third
raw microcanonical/concentration point, and connect the finite-size sequence to
a thermodynamic finite-beta target.

## Locked family and representative point

On an `L_x x 4` torus with `L_x=4N`,

\[
H_\varphi=-J\sum_{p=(x,y)}\left[
 e^{i\varphi(-1)^{x+y}}U_p+e^{-i\varphi(-1)^{x+y}}U_p^\dagger
\right]+\lambda_\star\sum_p F_p,
\qquad \lambda_\star/J=1.
\]

Use the size-independent local phase amplitude `varphi`.  The primary local
witnesses are `A_R,Z_R` on the two-column circumference-four stripe.  Keep the
localized shifted-shell `Y_R` route in Appendix E only.

Use

\[
\varphi_\star=0.05
\]

as the representative positive phase, and the principal grid
`{0.025,0.05,0.075,0.10}`.  `varphi=0` is an enhanced-symmetry endpoint/control.

## Newly closed gates

### 1. Exact periodic-product cage for all `4N x 4`

`qdm_checkerboard_exact_periodic_product_proof.md` and
`qdm_checkerboard_exact_periodic_product_certificate.csv` establish the
all-size cage analytically:

- in each four-column motif only plaquette columns `x=0,2` are active;
- `x=1,3` are inactive on every support configuration, so adjacent motifs do
  not couple kinetically;
- active parents have opposite exact amplitudes;
- their checkerboard phases are equal because `chi(x,y+2)=chi(x,y)`;
- every support configuration has four flippable plaquettes per motif.

Hence for arbitrary real `varphi`,

\[
H_\varphi|\psi_{L_x}\rangle=\lambda_\star L_x|\psi_{L_x}\rangle,
\qquad L_x=4N,
\]

and `e_psi/J=1/4` at `lambda_star/J=1`.

The `L_x=4,8,12` cage residuals are implementation checks of this exact result.
The fixed `A_R,Z_R` residuals remain at machine precision on those three sizes.

### 2. Non-gauge checkerboard direction

Within the declared physical local link-rephasing class,

\[
G(\theta)=\exp\!\left(i\sum_\ell \theta_\ell n_\ell\right),
\]

the checkerboard vector obeys

\[
\chi^T M_{\rm lg}=0,
\qquad
\chi^T\chi>0.
\]

Thus it is analytically outside this tangent image.  The `L_x=4,8,12` numerical
quotients give unit relative distance.  Do not upgrade this scoped statement to
arbitrary configuration-dependent diagonal unitaries.

### 3. Full positive-phase checkerboard symmetry sector

For `varphi>0`, resolve the even-parity translation subgroup using

\[
T_{\rm diag}=T_xT_y,
\qquad T_y^2,
\]

and select

\[
k_{\rm diag}=\pi/2,
\qquad T_y^2=+1.
\]

`qdm_checkerboard_common_symmetry_sector.csv` gives

| `L_x` | group size | selected-sector dimension | cage projection norm |
|---:|---:|---:|---:|
| 4 | 8 | 15 | 0.5 |
| 8 | 16 | 1125 | 0.5 |
| 12 | 24 | 114483 | 0.5 |

Projected `Q_A,Q_Z` residuals are at numerical zero.  The point-group audit
shows that the remaining positive-phase operations exchange
`k_diag=+pi/2` and `-pi/2`; they do not make a further block inside the selected
irrep.  This gate is closed.

### 4. Fully resolved dark-manifold classification at `L_x=4,8`

At `varphi_star=0.05`,

\[
(r_{\rm TypeI},r_{\rm jd})=(1,1)\quad (L_x=4),
\qquad
(2,2)\quad (L_x=8),
\]

with unexplained norm below `1.2e-12`.  Thus the translated `A,Z` joint-dark
space is fully explained by the projected compact Type-I manifold at both
completed energy-resolved sizes.

The target-energy `L_x=12` classification remains coupled to the missing
large-strip partial spectrum.

### 5. `beta=0` remains rejected

Transfer counting through `L_x=256` gives a preferred large-size fit

\[
e_{\beta=0}-e_\psi\to 5.98\times10^{-3},
\]

well above the `1e-3` gate.  Use `beta=0` only as an exact baseline/control.
The primary protocol is finite-beta energy matching.

### 6. Fully resolved finite-beta evidence at `L_x=4,8`

Exact raw energy matching gives

\[
\beta_4J=0.158605,
\qquad
\beta_8J=0.074728.
\]

The canonical targets are

\[
(\tau_A,\tau_Z)_{\rm can}
=(0.065801,0.131097)\quad(L_x=4),
\]

\[
(0.050731,0.098903)\quad(L_x=8).
\]

At window prefactor `0.75` and `varphi_star=0.05`, raw MC values are

\[
(0.054535,0.113590),
\qquad
(0.048255,0.094878),
\]

and the raw maximum mismatch decreases

\[
1.7507\times10^{-2}\to4.025\times10^{-3}.
\]

Across the positive phase grid, the prefactor-`0.75` envelope changes from
`1.7507e-2` to `4.069e-3`.

Window systematics are not negligible at `L_x=4`: prefactor `1.0` gives
`2.834e-3 -> 1.616e-3`.  Do not fit the two-size sequence thermodynamically.

### 7. Fully resolved raw stripe concentration at `L_x=4,8`

The formal constrained Hermitian two-column stripe space has 51 directions
including identity, i.e. 50 nonidentity directions.  After symmetry projection
and null quotient, the covariance basis has

- 20 directions at `L_x=4` (19 nonidentity),
- 25 directions at `L_x=8` (24 nonidentity).

The defining raw widths at `varphi_star=0.05` are

\[
w_4^{\rm raw}=0.122454,
\qquad
w_8^{\rm raw}=0.061366,
\]

with cleaned companions `0.108204, 0.061045`.  The positive-grid raw envelope
is `0.122465 -> 0.061493`.

This is a complete test on the declared stripe only, not every bounded
fixed-width region.

## `12x4` failure diagnosis

The production run successfully completed:

- zero-winding basis construction;
- full checkerboard symmetry projection;
- selected sector dimension `114483`;
- projected exact cage with residual `2.2e-15`;
- projected `A_R,Z_R` darkness;
- the canonical-typicality computation stage.

It then attempted

```text
scipy.sparse.linalg.eigsh(..., sigma=..., k=1024)
```

which invokes a direct SuperLU factorization of `H-sigma I`.  SuperLU failed
with `MemoryError` (`Can't expand MemType 0: jcol 93306`).  The cgroup memory
peak was about `105 GB`; the Docker limit was `400 GB`.  This is a sparse-LU
fill-in bottleneck, not evidence that the `114483`-dimensional sector itself is
infeasible.

The notebook currently appends the `12x4` canonical-typicality row only after
the later eigensolve.  Because the eigensolve failed, the successful canonical
result was not persisted.

## Remaining provisioning

### P0.1 -- checkpoint the `12x4` canonical target immediately

Change the large-strip workflow so that, immediately after canonical typicality
and energy matching, it writes/updates:

- `qdm_checkerboard_finite_beta_transfer_target.csv`;
- `qdm_checkerboard_finite_beta_energy_match.csv`;
- `qdm_checkerboard_finite_beta_transfer_phase_check.csv`.

Required `L_x=12` fields:

- `beta_star` and `beta_stderr`;
- `tau_A_target`, `tau_Z_target` and stochastic errors;
- sector dimension and entropy-density diagnostic;
- typicality sample count and seed;
- representative phase and at least one phase-independence check.

This write must happen **before** any microcanonical eigensolve.

### P0.2 -- replace direct-LU shift-invert

Do not simply retry the same `eigsh(..., sigma=...)` path with a larger memory
limit as the primary solution.  Use a memory-controlled method suitable for a
local window around `E_psi=12`, for example:

- polynomial/Chebyshev filtering plus Rayleigh--Ritz;
- spectrum slicing/contour filtering if available;
- iterative shift-invert with an explicit iterative `OPinv` and preconditioner;
- stochastic microcanonical filtering if only witness means are required.

For the full covariance and dark-manifold classification, an explicit filtered
spectral subspace is preferable.

Any partial-spectrum method must export:

- requested/returned subspace size;
- energy range and exact requested-window coverage;
- maximum eigenpair residual;
- method/budget convergence using at least two independent budgets or filters;
- runtime and peak memory.

### P0.3 -- obtain `L_x=12` raw microcanonical `A,Z`

At `varphi_star=0.05`, compute in the same fully resolved irrep:

- raw microcanonical `tau_A,tau_Z`;
- matched canonical `tau_A,tau_Z` from P0.1;
- raw matching distance;
- raw and cleaned window counts;
- joint-dark rank in the target window and caged fraction;
- window prefactors `0.5,0.75,1.0` at minimum;
- target-energy compact-Type-I versus joint-dark classification.

Primary outputs:

- `qdm_checkerboard_thermal_overlap.csv`;
- `qdm_checkerboard_window_systematics.csv`;
- `qdm_checkerboard_joint_dark_kernel.csv`;
- `qdm_checkerboard_joint_dark_vs_type1.csv`;
- a dedicated `qdm_checkerboard_L12_sparse_convergence.csv` (or renamed method-
  agnostic convergence file).

### P0.4 -- obtain `L_x=12` raw stripe concentration

In the same covered raw window, rebuild the complete two-column constrained
Hermitian stripe algebra after full symmetry projection and report:

- projected quotient dimension and nonidentity dimension;
- `w_raw` and `w_clean`;
- worst raw/clean eigenoperators;
- window state count and removed fraction;
- energy-block tolerance audit;
- method/window convergence.

Update:

- `qdm_checkerboard_concentration_grid.csv`;
- `qdm_checkerboard_worst_eigenoperator.csv`;
- `qdm_checkerboard_concentration_L12_raw_clean.csv`.

### P0.5 -- thermodynamic finite-beta target

Compute a large-fixed-width target satisfying

\[
e_{\rm can}(\beta_\star)=1/4,
\]

and export

\[
\beta_\star,
\qquad
\tau_A(\beta_\star),
\qquad
\tau_Z(\beta_\star).
\]

If the large-`L_x` transfer/typicality method works in a coarser winding or
checkerboard symmetry space, establish a bounded-local resolved-to-coarse
ensemble-equivalence bridge before identifying those values with the selected
`(k_diag,T_y^2)=(pi/2,+1)` target.

### P1.1 -- positive entropy density and vanishing caged fraction

The selected full-symmetry dimensions `15,1125,114483` imply finite-size
entropy densities increasing toward the bulk fixed-width scale.  Turn this into
a thermodynamic statement using transfer/group-character counting or a longer
selected-sector sequence.  Combine it with the finite compact-cage count to
show a vanishing exceptional fraction.

### P1.2 -- controlled fixed-width extrapolation

After the `L_x=12` thermal and concentration rows exist:

- compare constant, `1/L_x`, and other motivated finite-size forms;
- bootstrap or otherwise quantify fit/systematic uncertainty;
- require consistency across the prescribed window family;
- do not force the thermodynamic canonical target as an asymptote unless the
  data support it.

### P1.3 -- positive-phase family upgrade

The cage is exact for arbitrary real `varphi`, so no phase-grid refinement is
needed for the **caging** statement.  For deformation-stable thermal behavior,
add a third-size phase scan or a continuity/error bound before promoting the
sampled grid to an open interval.

### P2 -- broader bounded-region concentration

The current complete algebra test is only the selected two-column stripe.
For a literal Definition-III.2 claim over arbitrary bounded fixed-width
regions, either test at least one larger longitudinal region or provide an
analytical reduction/covering argument.

## Figure 7 handoff

The current full-symmetry figure is now the correct interim protocol:

1. raw `L_x=8`, `varphi_star=0.05` ETH scatter in the selected checkerboard
   irrep;
2. raw MC versus matched canonical `A,Z` at `L_x=4,8`;
3. positive-phase raw matching and raw stripe-width diagnostics;
4. raw stripe width over the phase grid.

Final figure requirements:

- add `L_x=12` to panels (b)--(d);
- add the thermodynamic finite-beta target or fit band;
- retain explicit raw-window and full-symmetry labels;
- do not label the projected local algebra as rank 50: 50 is the formal
  nonidentity local dimension, while the projected quotient ranks are size
  dependent (`19,24,...` nonidentity directions).

## Current claim boundary

Already safe to state in Sec. VII:

- exact checkerboard periodic-product cage on every `4N x 4` torus;
- exact cage energy `E_psi=lambda_star L_x` and `e_psi/J=1/4` for
  `lambda_star/J=1`;
- checkerboard non-gauge character in the declared local link-rephasing class;
- bounded `A_R,Z_R` darkness through `L_x=12`;
- full positive-phase checkerboard symmetry resolution through `L_x=12`;
- complete joint-dark/compact-Type-I classification at `L_x=4,8`;
- fully resolved raw finite-beta witness and stripe-concentration evidence at
  `L_x=4,8`;
- failure of `beta=0` energy matching.

Do **not** yet claim:

- a thermodynamic finite-beta target in the selected irrep;
- `L_x=12` raw microcanonical or stripe concentration;
- vanishing raw matching distance or stripe width in the fixed-width limit;
- a deformation-stable fixed-width ICQMBS under the full Definition III.2;
- a true two-dimensional ICQMBS.
