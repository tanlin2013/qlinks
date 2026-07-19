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
- Add type-1 chiral PEPS objectives, tile-periodic chiral-parity inference, native Z2 charge-resolved tensors, and separated kinetic/potential visual diagnostics.
- Pin the optional tensor-network environment to Python 3.11--3.13, NumPy below 2.4, Numba 0.62, and llvmlite 0.45 so Intel macOS can install binary wheels without compiling LLVM.
- Add Autograd to the ``tn`` extra and use a compact 108-parameter quimb ``PTensor`` for exact finite-cluster PEPS variance optimization.
- Add tensor-network graph, local-entry, amplitude, and optimization-history visualizers together with a dedicated ``experimental/notebooks/tensor_network.ipynb`` demonstration.
- Add bi-periodic square-QDM product-tile certification, seam/corner diagnostics, and direct finite-tile search.
- Add exact arbitrary-repeat square-QDM stripe cage certificates, normalized cage-derived ETH witnesses, and beta-zero energy-density matching.
- Add Fourier-projected square-QDM winding-sector strip contractions for local ETH witnesses.
- Initial API stable release.
- Add square-QDM two-plaquette singlet-product leakage diagnostics, exact-cover
  scans, a boundary-resolved halo tile basis, an optional quimb MPS handoff,
  and a handoff from ``experimental/notebooks/cage_padding.ipynb`` to the dedicated tensor-network notebook.
