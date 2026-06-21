Overview
========

``qlinks`` provides building blocks for exact-diagonalization studies of
constrained quantum lattice models.  Its main use cases are quantum link models
(QLM), quantum dimer models (QDM), constrained spin models, Fock-space graph
visualization, and interference-cage diagnostics.

The package is organized in layers rather than around one monolithic solver:

``qlinks.lattice``
   Geometry objects for chains and two-dimensional lattices.  Lattices provide
   sites, links, plaquettes, boundary conditions, orientations, and helper
   methods used by model and visualizer code.

``qlinks.variables``
   Local Hilbert spaces and variable layouts.  A layout says which physical
   degree of freedom lives on each site or link and how those variables are
   encoded as basis configurations.

``qlinks.constraints`` and ``qlinks.basis``
   Constraint objects and basis solvers.  The common workflow is to enumerate
   only configurations that obey Gauss-law, dimer-covering, blockade, winding,
   or other sector constraints.

``qlinks.operators`` and ``qlinks.builders``
   Local operators and sparse Hamiltonian builders.  Builders apply local
   updates to constrained basis states and assemble sparse matrices.

``qlinks.models``
   High-level model classes that package a lattice, layout, constraints,
   sectors, local terms, and build options behind a compact user API.

``qlinks.caging``
   Interference-cage search, classification, local reduced-density-matrix
   diagnostics, region-support extraction, local cage searches, and Lindblad
   construction helpers.

``qlinks.open_system``
   Lindblad operators, Liouvillian solvers, Monte Carlo wavefunction sampling,
   random state helpers, and dark-state diagnostics.

``qlinks.visualizer``
   Real-space basis visualizers and Fock-space graph visualizers for basis
   states, cage support, Hamiltonian graphs, Liouvillian graphs, and stochastic
   trajectories.

Design principles
-----------------

The central design idea is to keep the model-specific pieces local and reusable.
A model defines its lattice, constraints, sectors, and local Hamiltonian terms;
the shared basis solvers and sparse builders then handle enumeration and matrix
assembly.  Downstream tools consume the resulting :class:`qlinks.models.ModelBuildResult`
so that they can reuse the same basis, Hamiltonian, kinetic term, and potential
term without rebuilding them.

This separation is especially important for caging workflows.  The search layer
works on sparse matrices and self-loop values, while the classification,
visualizer, and open-system layers can optionally use model-level geometric
metadata to interpret the same cage in real space.
