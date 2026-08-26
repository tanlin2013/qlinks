# Section VI numerical provisioning cache

**Updated:** 2026-08-26
**Authoritative production base:** `data/evidence_jobs/spin1_production_20260806T074051Z/`
**Sparse-convergence addendum:** `data/evidence_jobs/spin1_production_20260810T082123Z/`
**Sec. VI provisioning addendum:** `data/evidence_jobs/spin1_sec6_provisioning_20260820T052954Z/`
**Common-window integration addendum:** `data/evidence_jobs/spin1_sec6_integration_20260825T073925Z/`

## Purpose

This is the short-term qlinks handoff for the **remaining** Sec. VI work.  Do
not rerun evidence that is already cached and validated.  The 20260825
integration addendum closes the homogeneous representative concentration
sequence under both the primary `W_L(gamma=1/4,c=1)` window and the fixed
`Delta E=1` control, and it supplies final common-protocol source data for
Fig. 6(a,b), the Appendix-D beta-zero bridge figure, and the complex-`t2`
obstruction figure.  The remaining P0 numerical work is now restricted to the
common-window **deformation grid** needed for Fig. 6(c) and the family band in
Fig. 6(d), followed by stable CSV-driven figure rendering.  Larger-size
family-wide and open-interval upgrades remain P1.

The file `spin1_xy_sec6_integration_audit.json` inside the 20260825 folder is
stale: it was written before the later common-window covariance stage and still
marks that stage pending.  The completed CSV/summary products in the same folder
are authoritative.  Regenerate the audit manifest before launching any new
work so future resumptions do not misclassify completed checkpoints.

## Locked Hamiltonian and representative point

Use

\[
K_d(t_d)=\sum_r\left[t_dS_r^+S_{r+d}^-+t_d^*S_r^-S_{r+d}^+\right],
\]

\[
H_\kappa=K_1(J)+K_3(0.1J)+K_2(i\kappa),
\qquad \kappa_\star/J=0.1.
\]

Use even `L>=8`, PBC, `M=-2`, the tower momentum sector, and `h=D=0` for
thermodynamic fits.  `L=6` may remain as a visual/pre-asymptotic control only:
for that ring the range-three bond coincides with its reverse under the current
translation-invariant sum convention. The exact
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
covered at `L=14`, and must not be used there. The narrower windows below lie inside the returned spectral range.  Their
microcanonical observables are now certified under the tested eigenpair-budget
increase described below.

For the `L^(1/4)`, prefactor-1 window,

\[
\Delta E_{14}=1.93351<2.08384,
\]

with 7615 raw and 7614 retained states. The defining raw `L=14` values are

\[
(\tau_A,\tau_Z,\tau_Y)_{L=14}^{\rm raw}
=(0.113204,0.220109,0.328044),
\]

while the cleaned companion is `(0.113212,0.220243,0.328219)`. Across
`L=8,10,12,14`, the raw state counts are `28,157,1083,7615`, giving
`log(N_win)/L = 0.4165,0.5056,0.5823,0.6384`. Treat this only as a
positive-entropy consistency trend, not a proof of the limiting entropy density.

For the same window convention, the raw--raw and clean--clean matching
sequences are

\[
\Delta_L^{\rm rr}
=0.031654,\ 0.021878,\ 0.013029,\ 0.012411,
\]

\[
\Delta_L^{\rm cc}
=0.026664,\ 0.020872,\ 0.012796,\ 0.012246
\quad (L=8,10,12,14).
\]

A fixed-width `Delta E about 1` window is also contained inside the returned `L=14` range (4011
raw states) and gives `Delta_14^cc=0.012882`, so the apparent `L=12 -> 14`
plateau is not explained simply by using the largest available window.

The convergence addendum reran the same `L=14` sector with 10000 requested
shift-invert eigenpairs.  The covered half-width increased from `2.08384` to
`2.55396`, while the raw state counts in the `L^(1/4)`, `Delta E=1`, and
`Delta E=0.75` windows remained `7615`, `4011`, and `3063`.  Across those
windows, increasing the budget changes each cleaned microcanonical witness and
`Delta^cc` by at most `2.6e-11`; all exported `converged_vs_previous` flags are
true.  The `O(10^-2)` matching plateau is therefore not caused by the earlier
8192-eigenpair cutoff.  It remains a finite-size/window/ensemble-resolution
question, not a sparse-budget question.

The exact fixed-`M`, `beta=0` limits remain `(1/9,2/9,1/3)`, but they are an
auxiliary reference only until local microcanonical--trace equivalence is
settled.

### Two-site concentration

The complete magnetization-preserving two-site Hermitian algebra has dimension
19.  The 20260825 integration addendum closes the homogeneous representative
sequence.  Under the primary `W_L(gamma=1/4,c=1)` protocol, the **raw** widths
for `L=8,10,12,14` are

\[
0.1685908,\quad 0.0763339,\quad 0.0469195,\quad 0.0174573,
\]

and are strictly decreasing.  The independent fixed-width `Delta E=1` control
is also strictly decreasing,

\[
0.1760308,\quad 0.0927004,\quad 0.0616927,\quad 0.0237316.
\]

The primary-window raw state counts are `28,157,1083,7615`; the fixed-width
counts are `24,101,609,4011`.  Raw and joint-dark-cleaned widths remain close,
and the exact-energy grouping tolerance audit is stable.  No concentration
power-law exponent is fitted or required.

## Mandatory checkpoint / resume discipline

**qlinks: read this block before launching any Sec. VI integration job.**

1. **Regenerate the integration audit first.**  The current
   `spin1_xy_sec6_integration_audit.json` is stale and predates the completed
   common-window covariance stage.  Re-scan the run folder, record every
   existing product checksum, and mark P0-A/common-window representative
   concentration as closed before deciding what to run.
2. **Validate and reuse before computing.**  Reuse the validated dense caches
   at `L=8,10,12` and the certified `L=14` 8192-eigenpair checkpoint.  Never
   rerun the 10000-eigenpair convergence calculation or any completed
   representative covariance merely to rebuild figures.
3. **No implicit eigensolver fallback.**  Missing cache/data must cause an
   explicit `PENDING`/`MISSING_CHECKPOINT` status, not an automatic heavy solve.
   A heavy solve may run only when the requested task below explicitly requires
   it.
4. **Checkpoint every `(L,kappa)` unit immediately.**  For the remaining
   deformation grid, persist the validated eigensystem metadata and derived raw
   witness/covariance row after each `kappa` and each size.  Do not wait until
   the full grid or notebook finishes.
5. **Write aggregate CSVs incrementally and atomically.**  After each completed
   unit, update the panel-C deformation CSV and panel-D family-band source table
   (or a resumable row cache) and flush a progress manifest.  A plotting or
   pandas failure must never invalidate completed numerical rows.
6. **Checkpoint before rendering.**  Figure scripts must consume stable exported
   CSVs only.  Record the exact CSV checksums/source manifest before generating
   SVG/PDF/PNG.  Rendering failure must not trigger numerical recomputation.
7. **On every resumed run, validate checkpoints rather than trusting filenames.**
   Check sector labels, `L`, `kappa`, window definition, state count, spectral
   coverage, residuals, and operator-basis/version metadata.
8. **Preserve completed products.**  Do not overwrite the authoritative
   20260825 common-window CSVs with partial/legacy-window data.  New products
   should either append compatible rows or carry a new run id/version.

## P0 tasks

### P0.1 -- close bookkeeping and freeze completed representative products

Regenerate the stale integration audit so it recognizes the following products
as complete and reusable:

- `spin1_xy_kappa0p1_concentration_common_windows.csv`;
- `spin1_xy_kappa0p1_common_window_summary.json`;
- `spin1_xy_figure6_panel_a_scatter.csv`;
- `spin1_xy_figure6_panel_b_witness_sequence.csv`;
- `spin1_xy_appendix_beta0_bridges_data.csv`;
- `spin1_xy_appendix_complex_t2_obstruction_data.csv`;
- common-window checkpoint and tolerance audits.

Acceptance: the regenerated audit must report the primary and fixed-width
representative concentration sequences as complete at `L=8,10,12,14`, with no
missing-size list and no request to recompute them.

### P0.2 -- common-window deformation data for Fig. 6(c)

At `L=12`, compute or validate/reuse raw microcanonical `tau_A,tau_Z,tau_Y` on
the positive grid

`kappa/J in {0.05,0.10,0.15,0.20}`

using exactly the primary `W_L(gamma=1/4,c=1)` protocol.  `kappa=0` remains a
symmetry-enhanced endpoint control and should not define the positive-interior
curve.

Export/update:

- `spin1_xy_figure6_panel_c_deformation.csv`;
- per-`kappa` checkpoint rows containing sector labels, window half-width, raw
  state count, joint-dark rank, tower residual, spectral coverage, and maximum
  in-window eigenpair residual.

**Checkpoint after every kappa value.**  If one point fails, preserve the other
completed points and report only that point pending.

### P0.3 -- common-window family concentration band for Fig. 6(d)

For `L=8,10,12` and the same positive grid, compute/validate the complete
19-operator **raw** covariance using `W_L(1/4,1)`.  The representative
`kappa_star/J=0.1` line through `L=14` is already complete and must be reused.

Export:

- one resumable row per `(L,kappa)` with `w_L^raw`, state count, joint-dark
  rank/fraction, energy-block count, spectral coverage, residual audit, and
  worst eigenoperator metadata;
- `spin1_xy_figure6_panel_d_family_band.csv` containing the positive-grid
  `min/max` envelope for each `L=8,10,12` plus the representative line through
  `L=14`.

**Checkpoint after every `(L,kappa)` covariance.**  Do not recompute completed
representative rows or older sizes if validation passes.

### P0.4 -- render final PRX figures from stable CSVs

Once P0.2--P0.3 source tables are complete, render the main figure independently
of the evidence notebook:

- `spin1_xy_figure6_prx.svg`;
- `spin1_xy_figure6_prx.pdf`;
- preview PNG;
- source-data/checksum manifest and physical-dimension/font audit.

Also render Appendix-D support figures:

1. homogeneous-window concentration systematics (source data already complete);
2. two-bridge `beta=0` RDM distances/witness differences (source data complete);
3. complex-`t2` obstruction plane (source data complete).

Rendering must be a pure postprocessing step.  A figure-generation failure must
never invoke or repeat an eigensolver/covariance calculation.

## Closed by the 20260810--20260825 evidence chain

- `L=14` sparse-budget certification under `8192 -> 10000`;
- post-convergence checkpoint/postprocessing repair;
- representative raw microcanonical witness sequence through `L=14`;
- representative complete two-site covariance under **two homogeneous** window
  protocols at `L=8,10,12,14`;
- common primary-window Fig. 6(a) scatter and Fig. 6(b) witness-sequence source
  data;
- two-bridge local RDM decomposition;
- resolved `(M,k)` to fixed-`M` `beta=0` bridge, which falls to `2.78e-5` at
  `L=14`;
- residual-operator spectrum/coefficients;
- Appendix-D complex-`t2` obstruction source grid;
- exact-energy tolerance and common-window checkpoint audits.

The remaining auxiliary `beta=0` uncertainty is the first bridge
`rho_mc^(M,k) <-> rho_beta0^(M,k)`, which remains `O(10^-2)` at `L=14`.
Do not schedule `L=16` merely to resolve this before the deformation-grid and
figure P0 tasks are complete.

## P1 tasks

1. **Nonrepresentative `L=14` family point:** the 20260820 job has
   `run_family_large_size=false`; preferably compute `kappa/J=0.20` after P0 if
   a stronger larger-size family-wide claim is desired.  One extra point is not
   a full `L=14` kappa envelope.
2. **Grid refinement / continuity:** required for an open-interval ICQMBS claim.
3. **Larger local region / upgrade argument:** required for a literal
   Definition III.2 certification over every fixed bounded region.  Otherwise
   retain the wording “complete two-site concentration.”
4. **Finite-beta deformation grid:** optional generality evidence only.

## Fig. 6 PRX design contract

The final main figure should communicate one physical sentence: the exact caged
tower is spectrally embedded but locally separated from an increasingly
concentrated thermal background, and this separation persists under the
compatible deformation.

- **(a) Representative ETH scatter:** raw background at `L=12`,
  `kappa_star/J=0.1`; three A/Z/Y mini-axes are acceptable; shade the primary
  window; show the tower prominently; avoid a large legend.
- **(b) Representative local separation:** raw A/Z/Y microcanonical values
  versus `L`; optional thin fixed-`M` `beta=0` asymptotes as auxiliary guides;
  no cleaned curves or matching-distance subpanel.
- **(c) Deformation persistence:** raw A/Z/Y microcanonical values versus
  positive `kappa/J` at `L=12`; mark `kappa_star/J=0.1`; zero baseline denotes
  the exact caged values.  Move `Delta_L(kappa)` to Appendix D.
- **(d) Background concentration:** line/band plot rather than the current
  sparse heatmap.  Plot representative raw `w_L(kappa_star)` through `L=14`
  plus a light sampled-positive-kappa min--max band through `L=12`.  No fitted
  critical/power-law exponent in the main figure.

Use final REVTeX two-column dimensions, base typography about 8.5--9 pt,
integer-only `L` ticks, marker shapes in addition to color, and SVG/PDF exports.

## Appendix D figure contract

- **Window concentration:** `w_L^raw` for homogeneous `W_L(1/4,1)` and fixed
  `Delta E=1`; pair with state-count/entropy-density trend or removed-fraction
  control.
- **Two beta-zero bridges:** plot the two RDM trace distances versus `L` (log-y
  is appropriate because the second bridge falls by orders of magnitude) and
  the first-bridge A/Z/Y differences.  Residual-operator coefficients can
  remain tabular unless a stable direction emerges.
- **Complex-t2 obstruction:** 2D residual map in `Re(t_2/J), Im(t_2/J)`, with
  the exact compatible line `Re(t_2)=0` and `t_2/J=i0.1` marked.  This supports
  the analytic compatibility rule of Sec. VI.C and belongs in the appendix.

## Claim boundary

Already established:

- exact compatible caged family and bounded `A,Z,Y` construction;
- positive raw same-Hamiltonian microcanonical witness values through the
  solver-certified `L=14` contained-window sequence at the representative point;
- rank-one translated-joint-dark inventory through `L=14`;
- complete two-site concentration at the representative point under two
  homogeneous `L=8,10,12,14` window protocols; the primary `L^(1/4)` raw
  sequence narrows `0.1685908 -> 0.0763339 -> 0.0469195 -> 0.0174573`;
- resolved-to-fixed-`M` `beta=0` local equivalence is numerically negligible by
  `L=14`; the remaining matching issue lies in the microcanonical-to-resolved
  bridge.

Still provisioned:

- common-window deformation-grid data for Fig. 6(c) and the family band in
  Fig. 6(d);
- controlled direct raw-microcanonical thermodynamic lower bound and positive
  raw-window entropy-density control;
- a larger-size family-wide point/envelope;
- grid refinement/continuity for an open interval;
- a larger-region concentration test or independent upgrade argument before a
  literal all-bounded-region ICQMBS certification;
- final Fig. 6 and Appendix-D figure regeneration under the common protocol.
