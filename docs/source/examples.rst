Minimal examples
================

This page collects short examples that are useful in notebooks and tests.  They
avoid optional plotting or distributed dependencies unless explicitly noted.

Square QDM Hamiltonian
---------------------

.. code-block:: python

   from qlinks.models import SquareQDMModel

   model = SquareQDMModel(
       lx=4,
       ly=4,
       boundary_condition="periodic",
       winding_x=0,
       winding_y=0,
       coup_kin=1.0,
       coup_pot=1.0,
   )

   result = model.build(builder="sparse", basis_solver="dfs")

   print(result.basis.n_states)
   print(result.hamiltonian.shape)
   print(result.kinetic.nnz)

Square QLM with staggered background charges
--------------------------------------------

Honeycomb and some QLM setups can have an empty zero-charge constrained Hilbert
space.  For square QLM examples that are meant to compare with QDM-like physics,
staggered charges are often useful:

.. code-block:: python

   from qlinks.models import SquareQLMModel

   model = SquareQLMModel.from_staggered_background(
       lx=4,
       ly=4,
       boundary_condition="periodic",
       charge_magnitude=1,
       winding_x=0,
       winding_y=0,
       coup_kin=1.0,
       coup_pot=0.0,
   )

   result = model.build(builder="optimized", basis_solver="dfs")
   print(result.hamiltonian.shape)

Plaquette-dependent couplings and Peierls phases
------------------------------------------------

Kinetic and potential couplings may be uniform values, dictionaries, or helper
objects.  Directed plaquette couplings are useful when a clockwise and
anticlockwise move should carry complex-conjugate amplitudes:

.. code-block:: python

   from qlinks.models import DirectedPlaquetteCoupling, SquareQDMModel

   coup_kin = {
       0: DirectedPlaquetteCoupling(clockwise=1.0j, anticlockwise=-1.0j),
       1: 1.0,
   }

   model = SquareQDMModel(
       lx=2,
       ly=2,
       boundary_condition="periodic",
       coup_kin=coup_kin,
       coup_pot=0.0,
   )

   result = model.build(builder="sparse")
   assert result.hamiltonian.shape[0] == result.basis.n_states

Reuse a basis across builders
-----------------------------

When checking two builders against one another, reuse the basis to make the row
and column order identical:

.. code-block:: python

   sparse_result = model.build(builder="sparse", sort_basis=True)

   bitmask_result = model.build(
       builder="bitmask",
       basis=sparse_result.basis,
       sort_basis=False,
   )

   difference = sparse_result.hamiltonian - bitmask_result.hamiltonian
   print(difference.nnz)

Alternatively, build both with sorting enabled and explicitly assert that the
basis configurations match before comparing matrices.

Cage search and first record
----------------------------

.. code-block:: python

   from qlinks.caging import CageSearchConfig, CageSearcher

   build_result = model.build(builder="sparse", basis_solver="dfs")

   search_result = CageSearcher.from_model_build_result(
       build_result,
       config=CageSearchConfig(search_type="type1", type1_kappas=(0,)),
   ).run()

   if len(search_result) > 0:
       record = search_result.first()
       print(record.signature)
       print(record.support)

Reduced density matrix readout
------------------------------

The caging diagnostics layer can inspect a state on a chosen set of variables:

.. code-block:: python

   from qlinks.caging import local_reduced_density_matrix_readout_from_state

   readout = local_reduced_density_matrix_readout_from_state(
       basis_configs=build_result.basis.states,
       state=record.full_state,
       variable_indices=(0, 1),
   )

   print(readout.local_basis)
   print(readout.density_matrix)

The exact choice of ``variable_indices`` depends on the physical diagnostic you
want.  For interference-zero reports, prefer the local supports returned by the
classification layer rather than manually selecting variables.

Open-system time evolution
--------------------------

For a small dense or sparse problem, the open-system module provides helpers for
initial states and Lindblad evolution:

.. code-block:: python

   import numpy as np

   from qlinks.open_system import density_matrix_from_state, solve_lindblad

   state = np.zeros(build_result.hamiltonian.shape[0], dtype=np.complex128)
   state[0] = 1.0
   rho0 = density_matrix_from_state(state)

   times = np.linspace(0.0, 1.0, 51)
   trajectory = solve_lindblad(
       hamiltonian=build_result.hamiltonian,
       jumps=(),
       density_matrix_initial=rho0,
       times=times,
       method="rk4_liouville",
   )

   print(trajectory.density_matrices[-1].shape)
