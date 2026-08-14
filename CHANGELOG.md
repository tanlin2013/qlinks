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
- Promote the stabilized caging local-search and stability families into nested `qlinks.caging.local_search` and `qlinks.caging.stability` subpackages, remove the temporary flat module paths, and contract the top-level caging API from 499 to 299 exports.
- Stabilize the local cage-search architecture into an acyclic dependency DAG, separating proposal generation/execution, global QDM primitives, exterior-padding search, factorized certification, and residual result assembly.
- Decompose cage stability, local cage search, and dark-manifold detector god modules into responsibility-specific implementation modules and migrate first-party callers to focused modules.
- Add exact checkerboard 4N x 4 periodic-product cage certification and fully resolve the positive-phase checkerboard translation irrep before raw finite-beta and stripe-concentration evidence.
- Add the gated square-QDM checkerboard fixed-width evidence workflow, including transfer-energy matching, size-independent compatibility and gauge tests, reduced-symmetry thermal pilots, translated A/Z joint-dark cleaning, and complete stripe-algebra concentration.
- Add reduced Fredholm-candidate diagnostics for compact square-QDM cage sequences, showing a rectangular state complement and a constant zero-winding coupling symbol across fixed width.
- Add Laurent/Fredholm domain-wall diagnostics and incidence-module interface tests separating genuine index-bound defects from critical or locally glued cage modes.
- Add many-body CLS-completeness quotients, translation-sector resolution, and finite-size persistence reports comparing square-QDM collective cages with the spin-1 XY tower.
- Add directed local-transfer ETH witnesses, local-channel spectra, cage-Jacobian conditioning, thermal-activity margins, and the spin-1 XY deformation-evidence notebook.
- Add plaquette/seam-resolved type-1 PEPS interference diagnostics and targeted period-two tensor enlargements selected by boundary-sector sensitivity.
- Add native chiral-block type-1 PEPS contractions and separated multi-cluster kinetic/potential objectives with cross-size validation.
- Add weighted boundary-cancellation matroid diagnostics and relative dependency scans that separate regional cage circuits from collective cage classes.
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
