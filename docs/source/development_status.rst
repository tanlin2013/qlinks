Current development status
==========================

``qlinks`` is usable for small- and medium-scale exact-diagonalization studies,
but several layers are still evolving quickly.  This page summarizes the current
state of the project so that examples and notebooks can avoid relying on unstable
internals.

Stable or mostly stable
-----------------------

Model-level build workflow
   The recommended public entry point is to instantiate a model and call
   ``model.build()``.  The returned ``ModelBuildResult`` keeps the basis,
   Hamiltonian, kinetic term, potential term, and model metadata aligned.

Lattice and variable abstractions
   Square, triangular, honeycomb, and chain geometries are available through the
   lattice layer.  Link-based and site-based variable layouts are shared by the
   model and operator layers.

Sparse Hamiltonian construction
   The generic sparse builder, optimized sparse builder, and bitmask builder are
   part of the supported build workflow.  The bitmask builder is intended for
   compatible binary local spaces.

Basis solvers
   DFS and brute-force solvers are core dependencies.  The CP-SAT solver is an
   optional extra because it depends on OR-Tools.

Visualization entry points
   Real-space basis visualizers and Hamiltonian graph visualizers are public
   APIs.  Some rendering backends, such as igraph, pycairo, plotly, and pyvis,
   are optional extras.

Active research layers
----------------------

Interference-cage classification
   Classification reports, reduced interference-zero operators, collective
   cancellation diagnostics, and region-support extraction are under active
   refinement.  Prefer high-level helpers from ``qlinks.caging`` over internal
   helper modules.

Local cage search and padding
   Local QDM/QDM-style cage searches, padding diagnostics, robust certification,
   and multi-block workflows are designed for research iteration.  Naming and
   configuration details may still change as more lattices and models are added.

Cage Lindblad construction
   The open-system construction for dark-state engineering is experimental.
   Jump grouping, monitor choices, frustration-free decompositions, and
   recycling terms should be treated as configurable research choices rather
   than final defaults.

Distributed and storage workflows
   Ray and HDF5 helpers exist for sweeps and large result sets.  They are useful
   in project workflows, but the high-level organization of sweep metadata and
   output schemas is still being improved.

Planned documentation work
--------------------------

The next documentation steps are:

#. Split the API reference into public, high-level pages and lower-level module
   pages.
#. Add narrative tutorials for QDM, QLM, PXP, caging, visualization, and
   open-system workflows.
#. Add example notebooks that are executed in CI when optional dependencies are
   available.
#. Add a glossary for project terms such as cage signature, interference zero,
   reduced IZ operator, regional cage, local cage, monitor, and recycler.
#. Add diagrams for the model-build pipeline and the caging/classification
   pipeline.

Backward-compatibility expectations
-----------------------------------

The core model-build API should remain relatively stable.  Experimental caging
and open-system configuration options can change more often, especially when a
new diagnostic makes an older option redundant.  When in doubt, prefer APIs that
are exported from package-level ``__init__.py`` files and documented in this
user guide.
