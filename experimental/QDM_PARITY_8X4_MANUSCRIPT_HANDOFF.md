# Fig. 9: original presentation with target-parity data

The figure uses the completed exact target-parity 8x4 evidence and reuses the
pre-PR157 `prx_main_thermal_figure_polish.py` presentation. The original 2x2
panel roles, stacked A/Z axes, 7.05 by 5.75 inch canvas, REVTeX font setup,
blue/green witness colors, orange star, filled raw/open canonical circles,
solid/dashed phase curves, legends and literal phase-range glyphs are retained.
The figure is generated from code for a separate Overleaf upload. This command
does not edit the manuscript, launch an eigensolve or depend on PR #157.

## Render on the server

After merging PR #158 and updating the checkout, run from the qlinks root:

```bash
mkdir -p output/qdm_figure9_parity_8x4

docker run --rm --init \
  --env MPLBACKEND=Agg \
  --volume "$PWD:/workspace/qlinks:ro" \
  --volume "$PWD/output/qdm_figure9_parity_8x4:/workspace/output" \
  --workdir /workspace/qlinks \
  "${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook-primme}" \
  python experimental/jobs/render_qdm_parity_figure9.py \
  --data experimental/data/evidence_jobs/qdm_parity_submission_20261009 \
  --output /workspace/output --use-tex
```

The existing notebook/PRIMME image supplies the plotting/LaTeX dependencies;
the current source is bind-mounted, so this render does not require an image
rebuild. If data are stored outside the checkout, mount that data directory
read-only and give its container path to `--data`.

Upload `output/qdm_figure9_parity_8x4/qdm_checkerboard_figure9_prx.pdf`
to Overleaf. The same directory contains SVG/PNG, each panel's source CSV,
`figure_manifest.json` and `figure9_8x4_provenance.json`.

Without Docker, in a qlinks notebook environment with LaTeX:

```bash
python experimental/jobs/render_qdm_parity_figure9.py \
  --data experimental/data/evidence_jobs/qdm_parity_submission_20261009 \
  --output output/qdm_figure9_parity_8x4 --use-tex
```

`--use-tex` refuses a font fallback if LaTeX is unavailable. Omitting it uses
Computer Modern mathtext for a lightweight preview.

## Panel contents and scope

| Panel | Original role, with updated data |
| --- | --- |
| (a) | Stacked A/Z eigenstate scatter at 8x4, phi=0.05, Sy=+1; target star and primary window |
| (b) | Stacked raw/canonical A/Z means versus strip length, with only Lx=8 and literal positive-phase ranges |
| (c) | Stacked raw/canonical A/Z means versus the four compatible positive phases at Lx=8 |
| (d) | Raw stripe width versus strip length, with only Lx=8 and its literal positive-phase range |

The size axes in (b,d) contain one completed thermal point. There is no size
fit or connecting segment, and no 4x4 or 12x4 marker. The 4x4 target-parity
sector is one-dimensional, so its raw/canonical equality is a trivial symmetry
check rather than a thermal comparison. No translation-only data are mixed
with the resolved 8x4 block. This is the necessary data change relative to the
old size plots; it does not repurpose panels (b,c,d) as window diagnostics.
Ranges smaller than the rendering resolution remain small; no display floor
is added. The source CSVs retain all prescribed window controls.

Source bundle: `qdm_parity_evidence_20261009.zip` (PR #152), with complete exact
4x4/8x4 eigensystems at phases 0.025, 0.05, 0.075 and 0.10, and total-energy
half-widths 0.10, 0.20, 0.25 and 0.50. The primary half-width remains the
prespecified 0.25 and the representative phase remains 0.05. The 8x4 block
has dimension 875. At the primary point: 47 raw-window states, beta J=0.07914368,
raw A/Z means 0.04931479/0.10126465, matched canonical means
0.05721171/0.11134491, maximum mismatch 0.01008026, stripe width 0.07085593.

The renderer checks protocol, target parity, complete phase/window grid and
frozen residual/Gram certificates. It independently reconciles raw counts,
canonical energy matching, all raw/canonical witness means and maximum
mismatches against the full eigenstate table. It does not repeat Hamiltonian
residual actions or stripe covariance calculations. Hashes and plotted tables
are exported beside the figure. Wrong-parity or inconsistent inputs fail before
plotting. The raw populations retain the target.

The old manuscript caption/prose still require corresponding parity and
numerical corrections; a figure replacement alone does not update them.
The earlier window-panel manuscript patch describes a different figure and
must not be applied as the caption for this restored presentation.

## Optional 12x4 continuation after PR #157

Merging PR #157 does not modify the paused PRIMME process or its loaded native
libraries. `docker unpause` resumes that same process. To use the optimized
provider and logging, update the checkout, build and verify a separately tagged
PRIMME image, then launch a new refinement container with that image. Reuse the
same run ID and cache root to warm-start from the saved 4096-vector cache.
Unsaved progress inside the interrupted PRIMME call is not checkpointed, so a
new launch repeats that refinement call. Rebuilding a tag does not update an
existing container, and `docker restart` retains the old image. Do not change
live libraries. No current job is stopped by this renderer.

A future 12x4 figure upgrade still requires physical-H residuals, full Gram
quality, two distinct covered budgets, stable counts/observables across all
prescribed windows and matched-parity canonical evidence. It is independent
of the present finite-size figure.
