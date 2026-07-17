Install the latest
===================

To install the latest version of qlinks simply run:

`pip3 install qlinks`

OR

`poetry add qlinks`

OR

`pipenv install qlinks`


Changelog
=========
## 1.0.0 - TBD
- Make the tensor-network Docker interpreter multi-architecture, install only the explicit ``tn`` extra, and pin ``quimb``, ``numba``, and ``llvmlite`` as direct optional dependencies.
- Add bi-periodic square-QDM product-tile certification, seam/corner diagnostics, and direct finite-tile search.
- Add exact arbitrary-repeat square-QDM stripe cage certificates, normalized cage-derived ETH witnesses, and beta-zero energy-density matching.
- Add Fourier-projected square-QDM winding-sector strip contractions for local ETH witnesses.
- Initial API stable release.
- Add square-QDM two-plaquette singlet-product leakage diagnostics, exact-cover
  scans, a boundary-resolved halo tile basis, an optional quimb MPS handoff,
  and an interactive singlet/TN section in ``experimental/notebooks/cage_padding.ipynb``.
