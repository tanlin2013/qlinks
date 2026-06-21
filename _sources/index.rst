qlinks
======

``qlinks`` is a research-oriented Python package for constrained lattice
Hamiltonians, Fock-space graph diagnostics, interference-cage searches, and
open-system constructions around many-body scarred states.

The package is designed around a typical exact-diagonalization workflow:
choose a lattice model, build the constrained basis and Hamiltonian, analyze
its Fock-space structure, and then use visualization or diagnostic tools to
understand compact and regional cage states.

.. warning::

   The public API is becoming more stable, but the caging and open-system
   layers are still active research code.  Prefer the high-level APIs shown in
   this guide when writing notebooks or examples.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   overview
   installation
   workflow
   examples
   development_status

.. toctree::
   :maxdepth: 2
   :caption: API reference

   qlinks

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing
