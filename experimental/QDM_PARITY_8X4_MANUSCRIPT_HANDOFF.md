# Fig. 9: primary 8x4 target-parity comparison

The main manuscript figure uses the frozen exact target-parity 8x4 results.
The ongoing 12x4 computation is optional and is not an input to this renderer.
No new eigensolve, canonical typicality calculation, or window selection is needed.

Source bundle: `qdm_parity_evidence_20261009.zip` (PR #152). Its exact stage
contains the complete eigensystems at phases 0.025, 0.05, 0.075 and 0.10,
and total-energy half-widths 0.10, 0.20, 0.25 and 0.50. The primary half-width
remains the prespecified 0.25 and the representative phase remains 0.05.

```bash
python experimental/jobs/render_qdm_parity_figure9.py \
  --data experimental/data/evidence_jobs/qdm_parity_submission_20261009 \
  --output output/qdm_figure9_parity_8x4
```

The renderer requires the exact all-eigenstate CSV, phase/window CSV and exact
audit JSON. It checks their protocol, parity, complete phase/window grid, and
stored residual/Gram certificates. It independently recomputes all raw counts
and raw/canonical witness means from the eigenstate table. It does not repeat
Hamiltonian residual actions or the covariance calculation. Source hashes and
the primary values are written beside the plot. Legacy or opposite-parity
inputs fail before a figure is written.

| Panel | Content |
| --- | --- |
| (a) | A/Z eigenstate scatter at 8x4, phi=0.05, Sy=+1; target star and primary window |
| (b) | Raw microcanonical versus matched canonical A/Z means across positive phase |
| (c) | Maximum A/Z raw-canonical mismatch versus total-energy window half-width |
| (d) | Complete-stripe raw fluctuation width versus the same half-width |

In (c,d), circles give phi=0.05 and whiskers the exact sampled phase min/max.
They are parameter ranges, not errors. Connecting segments are visual guides;
there is no fitted/interpolated numerical evidence or size extrapolation.
All primary means and covariances retain the caged eigenstate.

At the primary point the block dimension is 875, the window contains 47 states,
beta J=0.07914368, raw A/Z means are 0.04931479/0.10126465, matched canonical
means are 0.05721171/0.11134491, maximum ensemble mismatch is 0.01008026,
and raw stripe width is 0.07085593. The 4x4 target-parity block has dimension
one and is a symmetry check, not a thermal size point. In particular, the
former translation-only two-size narrowing and ensemble-mismatch trends must
not survive in the new caption, Sec. VII, Appendix E or conclusion.

The manuscript retains the analytic all-size eigenstate and bounded-local
annihilation construction. Its QDM thermal evidence is a finite-size local
comparison with window/deformation controls. The thermodynamic ICQMBS
classification remains open.

The existing 12x4 spectral/observable gates are unchanged. A future upgrade
requires physical-H residuals, full Gram quality, two distinct covered budgets,
stable counts/observables across all prescribed windows, and matched-parity
canonical evidence. Neither an unfinished run nor a folded residual alone can
add a 12x4 thermal point. The optimized runtime in PR #157 should be validated
before any separately authorized production retry; do not modify live libraries.
