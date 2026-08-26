# Spin-1 Sec. VI: evidence integration, Fig. 6 redesign, and qlinks provisioning
**Date:** 2026-08-25
**New evidence base:** `data/evidence_jobs/spin1_sec6_provisioning_20260820T052954Z/`
**Common-window integration:** `data/evidence_jobs/spin1_sec6_integration_20260825T073925Z/`

**2026-08-26 status:** representative homogeneous-window concentration, Fig. 6(a,b) source data, the Appendix-D beta-zero bridge data, and the complex-`t2` obstruction grid are now complete.  Remaining P0 numerical work is only the common-window positive-`kappa` deformation grid for Fig. 6(c) and the family concentration band for Fig. 6(d), followed by rendering.  The integration audit JSON is stale and must be regenerated before any new compute.

## 1. Evidence status after the 2026-08-20 provisioning run

### Newly established
- Representative `L=14`, `kappa/J=0.1` complete two-site concentration is available in a solver-certified contained fixed-width window `Delta E=1`:
  - `w_14^raw = 0.0237316428`
  - `w_14^clean = 0.0236713087`
  - raw states: `4011`
  - joint-dark rank: `1`
  - removed fraction: `2.493e-4`
- The exact-energy grouping tolerance audit is stable.
- The representative sparse-budget certification remains passed.
- The two-bridge beta-zero decomposition is now available:
  - `rho_mc^(M,k) <-> rho_beta0^(M,k)` remains `O(10^-2)` at `L=14`.
  - `rho_beta0^(M,k) <-> rho_beta0^M` falls to `2.78e-5` at `L=14`.
  - Therefore momentum-sector resolution is not the origin of the residual beta-zero mismatch.
- Residual-operator spectra and coefficients were exported.

### Not yet established
- The existing `L=8,10,12` concentration sequence and the new `L=14` point do not yet share one common window prescription.
- The planned nonrepresentative `L=14`, `kappa/J=0.20` family point did not run (`run_family_large_size=false`).
- Family-wide complete two-site concentration therefore remains established only through `L=12`.

## 2. Manuscript integration plan

### Sec. VI.D main text
Keep the PRX storyline physics-first.

1. Retain the representative raw microcanonical witness sequence as the local ETH mismatch.
2. Update the concentration statement to say that complete two-site narrowing now extends to a solver-certified `L=14` point:
   `w_14^raw = 0.0237` in a contained `Delta E=1` window.
3. Do **not** quote a power-law exponent from the mixed-window four-size sequence.
4. Until the common-window calculation below is complete, state explicitly that the `L=14` concentration point uses a different contained-window convention from the older `L<=12` sequence.
5. Keep the beta-zero comparison auxiliary. Do not add a numerical two-bridge table to the main text. At most add one short conceptual sentence:
   the resolved-to-fixed-M beta-zero bridge is already locally negligible by `L=14`, so the residual auxiliary mismatch lies in the microcanonical-to-resolved-beta-zero bridge.
6. Preserve the claim boundary:
   - representative complete two-site concentration: now through `L=14`;
   - family-wide concentration across sampled `kappa`: through `L=12`;
   - arbitrary bounded-region concentration and open-interval thermodynamic control remain outstanding.

### Appendix D
1. Replace the obsolete statement that no `L=14` covariance is available.
2. Record the raw/clean `L=14` covariance values, state counts, removed fraction, residual audit, exact-energy tolerance audit, and solver certification.
3. Add the two-bridge decomposition and explain that the resolved-to-fixed-M bridge is already negligible compared with the first bridge.
4. Keep residual-operator coefficients/spectra as an audit of the remaining finite-size mismatch; do not infer a new conserved local operator from the present data.
5. Once the common-window concentration is available, promote it to the primary concentration sequence and demote the older mixed-window sequence to a systematic/control.

### Claim ledger / evidence summary
Update:
- `L=14` representative two-site concentration: **established in a contained safe window**.
- solver-budget independence of representative `L=14`: **closed**.
- two-bridge beta-zero diagnosis: **computed; no momentum-sector obstruction found**.
- common-window concentration sequence: **pending**.
- nonrepresentative `L=14` deformation point: **pending / optional strengthening**.

## 3. Main Fig. 6: recommended PRX design

### Overall purpose
Fig. 6 should visually prove one sentence:

> The exact caged tower remains locally separated from an increasingly concentrated thermal background, and this separation survives the compatible deformation.

Use a two-column `2 x 2` layout. Avoid turning the figure into a diagnostics dashboard.

### Common protocol
Wherever possible, panels (a)-(d) should use the same primary microcanonical window
`W_L(gamma=1/4,c=1)`.
This is already the manuscript's preferred contained `L=14` sequence.

### Panel (a): representative ETH scatter
Purpose: show **spectrally embedded but locally separated**.

- `L=12`, `kappa_star/J=0.1`.
- Three vertically stacked mini-axes for `Q^A`, `Q^Z`, `Q^Y` are acceptable.
- Plot the **raw** same-Hamiltonian background as the defining comparison.
- Shade the primary `L^(1/4)` microcanonical window.
- Mark the selected tower at `E/L=0`, witness value `0`.
- Draw a subtle horizontal line/marker for the raw microcanonical mean inside the window.
- Remove the large in-panel legend. Explain the star/background/window in the caption.
- Use light, semitransparent background points so the tower marker dominates visually.

### Panel (b): representative finite-size local separation
Purpose: show the witness values stay finite as size grows.

- x-axis: `L = 8,10,12,14`.
- y-axis: normalized local witness value.
- Solid curves/markers: raw microcanonical `tau_A`, `tau_Z`, `tau_Y`.
- Use consistent colors/markers for A/Z/Y across panels (b) and (c).
- Optional thin dashed horizontal lines: exact fixed-M beta-zero thermodynamic counting targets `(1/9,2/9,1/3)`.
  These are auxiliary guides only; do not plot resolved beta-zero curves or a second delta subpanel.
- Do not show cleaned curves in the main figure.
- Let the caption state that raw/clean and two-bridge comparisons are in Appendix D.

### Panel (c): persistence under compatible deformation
Purpose: show the representative point is not isolated.

Recommended main version:
- x-axis: positive compatible deformation `kappa/J`.
- Use the largest fully dense size, preferably `L=12`.
- y-axis: raw microcanonical witness values `tau_A`, `tau_Z`, `tau_Y`.
- Use the same primary `L^(1/4)` window as panel (b).
- The exact caged values are identically zero; a thin zero baseline is enough.
- Mark `kappa_star/J=0.1` with a subtle vertical guide.
- Treat `kappa=0` as a symmetry-enhanced endpoint control (open/gray marker or omit from the main positive-interior grid).
- Move the current auxiliary `Delta_L(kappa)` matching plot to Appendix D.

This panel directly shows:
`exact local zero + finite thermal value` persists under deformation.

### Panel (d): background concentration
Purpose: show that the surrounding local background becomes structureless.

Replace the current sparse heatmap by a finite-size line/band plot.

- x-axis: `L`.
- y-axis: `w_L`.
- Filled markers/solid line: representative `kappa_star/J=0.1` raw concentration width using the **same primary `L^(1/4)` window**, through `L=14`.
- For `L=8,10,12`, add a light band or whiskers spanning the sampled positive-kappa range (`min_kappa w_L` to `max_kappa w_L`) under the same window convention.
- The family-wide band stops at `L=12` unless a full `L=14` kappa grid is computed.
- Do not fit or display a power-law exponent in the main figure.
- A linear y-axis is preferred; log scale can visually overstate a four-point power-law interpretation.

### Main Fig. 6 visual style
- Final size: full two-column REVTeX width.
- Base text at final size: about 8.5-9 pt.
- Panel letters bold, aligned consistently at top-left.
- Line widths about 0.9-1.1 pt; markers about 4-5 pt.
- Use both color and marker shape so curves remain distinguishable in grayscale.
- No panel titles; let axes and caption carry the meaning.
- Integer-only ticks for `L`.
- Keep legends outside dense data regions or use a compact shared legend.
- Export SVG + PDF; also produce a PNG preview and a dimension/font audit.

## 4. Appendix D figure plan

Appendix figures should answer referee/method questions rather than repeat the main physics figure.

### Fig. D1: concentration window systematics
Purpose: validate that the decreasing `w_L` trend is not a window artifact.

Recommended two-panel design:
- (a) `w_L^raw` versus `L` for at least:
  - primary `L^(1/4), c=1`;
  - fixed `Delta E=1`.
  Optionally include `L^(1/4), c=0.75` if cheap.
- (b) corresponding raw window state counts or `log N_win / L` versus `L`, plus raw-clean difference / removed fraction if space permits.

Do not combine the old broad `L^(1/2)` values with the new `L=14` fixed-window point into a fitted exponent.

### Fig. D2: beta-zero bridge decomposition
Purpose: show why the residual beta-zero mismatch is not a momentum-sector artifact.

- (a) two-site RDM trace distance versus `L`:
  - `D[rho_mc^(M,k), rho_beta0^(M,k)]`;
  - `D[rho_beta0^(M,k), rho_beta0^M]`.
  Use log y-axis because the second bridge falls by orders of magnitude.
- (b) `|Delta tau_A|`, `|Delta tau_Z|`, `|Delta tau_Y|` for the first bridge versus `L`.
- Primary window highlighted; other window choices shown as light bands or faint curves.
- Residual-operator coefficient details can stay in a compact table rather than another panel unless one fixed operator direction clearly emerges.

### Fig. D3: compatible deformation / obstruction geometry
Purpose: visually support the exact phase-compatibility condition.

Replace or supplement the current one-dimensional residual figure with the new complex-`t2` obstruction data:
- x-axis: `Re(t_2/J)`;
- y-axis: `Im(t_2/J)`;
- color: normalized tower residual, preferably `log10` with a numerical floor;
- overlay the exact compatible line `Re(t_2)=0`;
- mark the representative point `t_2/J=i0.1`;
- optional inset: a horizontal cut showing residual versus `Re(t_2/J)` at fixed `Im(t_2/J)=0.1`.

This figure belongs in Appendix D because Sec. VI.C already gives the exact analytic compatibility equation.

### Other Appendix D material
- Keep finite-D matched-temperature data as a supplementary control, preferably as a compact figure only if it adds visual information beyond the table/prose.
- Additional exact cages remain scope tests and need not be promoted into the main Sec. VI visual narrative.

## 5. qlinks provisioning tasks

The authoritative remaining-work list is `data/SEC6_NUMERICAL_PROVISIONING.md`;
its checkpoint/resume block must be followed before any compute.  In particular,
regenerate the stale integration audit first, validate/reuse all completed
representative products, forbid implicit eigensolver fallback, and checkpoint
after every `kappa` and `(L,kappa)` unit.

### CLOSED: representative common-window products

Already complete in `spin1_sec6_integration_20260825T073925Z`:

- homogeneous `W_L(1/4,1)` and fixed `Delta E=1` complete two-site covariance
  at `L=8,10,12,14`;
- Fig. 6(a) raw representative scatter;
- Fig. 6(b) raw witness sequence;
- Appendix-D beta-zero bridge source data;
- Appendix-D complex-`t2` obstruction grid.

Do **not** rerun these products merely because the stale audit JSON says they
are pending.

### P0-A: common-window deformation grid for Fig. 6(c)

At `L=12`, positive `kappa/J in {0.05,0.10,0.15,0.20}`, compute/validate raw
`tau_A/Z/Y(kappa)` under `W_L(1/4,1)`.  Checkpoint and flush the source row after
each kappa.

Output: `spin1_xy_figure6_panel_c_deformation.csv`.

### P0-B: common-window family concentration band for Fig. 6(d)

For `L=8,10,12` on the same positive grid, compute/validate the complete
19-operator raw covariance under `W_L(1/4,1)`.  Reuse the already-complete
representative `kappa/J=0.1` line through `L=14`.  Checkpoint after every
`(L,kappa)` covariance and write aggregate rows incrementally.

Output: `spin1_xy_figure6_panel_d_family_band.csv`.

### P0-C: render PRX Fig. 6

Generate from frozen CSVs only:
- `spin1_xy_figure6_prx.svg`;
- `spin1_xy_figure6_prx.pdf`;
- `spin1_xy_figure6_prx_preview.png`;
- a JSON/Markdown figure audit with physical dimensions, fonts, line/marker
  sizes, and source-data checksums.

Rendering failure must not call any numerical solver.

### P0-D: render Appendix D support figures

Source data are already complete. Generate:
- `spin1_xy_appendix_concentration_windows.svg/pdf`;
- `spin1_xy_appendix_beta0_bridges.svg/pdf`;
- `spin1_xy_appendix_complex_t2_obstruction.svg/pdf`.

Use the same typography and A/Z/Y color/marker dictionary as Fig. 6.

### P1: nonrepresentative `L=14` family point

Run `L=14`, preferably `kappa/J=0.20`, only after P0 is complete if stronger
larger-size family-wide evidence is desired.  One extra point is not a full
`L=14` kappa envelope.

### P1/P2: open-interval strengthening

Only if needed for the final claim:
- kappa-grid refinement or continuity bound;
- concentration on a larger bounded region, or an independent argument
  upgrading the complete two-site result.

## 6. Manuscript/figure claim discipline

Main text may claim:
- exact state and local-caging continuation over the analytic compatible family;
- positive raw microcanonical local mismatch through solver-converged `L=14` at the representative point;
- complete representative two-site background concentration through `L=14` under two homogeneous contained-window protocols;
- sampled deformation-wide two-site concentration through `L=12`.

Main text should not yet claim:
- a measured concentration critical exponent;
- that microcanonical-to-beta-zero trace distance is proven to vanish;
- a full `L=14` deformation-wide envelope;
- concentration for every bounded region;
- a proven open kappa interval without grid refinement/continuity control.
