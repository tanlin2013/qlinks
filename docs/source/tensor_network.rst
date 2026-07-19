Tensor-network cage ansatzes
============================

Install the optional backend with:

.. code-block:: bash

   pip install "qlinks[tn]"

The ``tn`` extra installs ``quimb`` for tensor-network objects and contractions,
``autograd`` for the default differentiable optimizer, and the compatible
``numba``/``llvmlite`` runtime.  The tensor extra currently supports Python
3.11--3.13.  This upper bound is deliberate: Python 3.13 can use binary
``llvmlite`` wheels on Intel macOS, whereas Python 3.14 would require a custom
LLVM source build.

qlinks remains responsible for the constrained local basis, winding-sector
Hamiltonian, and exact finite-cluster residual.  quimb supplies parametrized
tensors and optimization infrastructure.

Rectangular vertex tensor
-------------------------

The initial square-QDM tensor uses a rectangular tile.  The tile owns every
``+x`` and ``+y`` link whose source site lies inside the rectangle.  Translated
tiles therefore have disjoint physical degrees of freedom.  Incoming left and
down dimer occupations form virtual indices, while the owned outgoing right and
up links determine the opposite virtual indices.

For a horizontal two-plaquette singlet, a ``3 x 2`` tile is natural:

.. code-block:: python

   from qlinks.caging import (
       build_square_qdm_rectangular_tile_tensor_basis,
       build_square_qdm_singlet_peps_ansatz,
       square_qdm_two_plaquette_singlet_blocks,
   )
   from qlinks.models import SquareQDMModel

   host = SquareQDMModel(
       lx=8,
       ly=8,
       boundary_condition="periodic",
       coup_kin=1.0,
       coup_pot=0.0,
   )
   singlet = next(
       block
       for block in square_qdm_two_plaquette_singlet_blocks(
           host,
           directions=("x",),
       )
       if set(block.anchor_cells) == {(2, 2), (3, 2)}
   )
   tile_basis = build_square_qdm_rectangular_tile_tensor_basis(
       host,
       tile_shape=(3, 2),
       origin=(2, 2),
   )
   ansatz = build_square_qdm_singlet_peps_ansatz(
       host,
       singlet,
       origin=(2, 2),
   )

The ``3 x 2`` tile owns 12 links.  The structural constraint reduces the 4096
binary configurations to 71 locally completable physical states and 108
allowed tensor entries.  The resulting tensor shape, in quimb's ``urdlp``
convention, is ``(8, 4, 8, 4, 71)``.

Quimb networks
--------------

For short periodic directions, use the generic constructor, which preserves
parallel periodic bonds explicitly:

.. code-block:: python

   tn = ansatz.to_quimb_tensor_network(
       n_tiles_x=2,
       n_tiles_y=2,
   )

For at least three tiles in each direction, a structured quimb PEPS is
available:

.. code-block:: python

   peps = ansatz.to_quimb_peps(
       n_tiles_x=3,
       n_tiles_y=3,
   )

Compact Autograd optimization
-----------------------------

Candidate tensors can first be optimized against an exact qlinks Hamiltonian on
a small torus:

.. code-block:: python

   from qlinks.caging import build_square_qdm_peps_finite_cluster_problem

   model = SquareQDMModel(
       lx=6,
       ly=4,
       boundary_condition="periodic",
       winding_x=0,
       winding_y=0,
       coup_kin=1.0,
       coup_pot=0.0,
   )
   problem = build_square_qdm_peps_finite_cluster_problem(
       model,
       tile_basis,
   )

The loss is the normalized energy variance,

.. math::

   \mathcal V_H(A)
   = \frac{\langle\Psi(A)|(H-\langle H\rangle_A)^2|\Psi(A)\rangle}
           {\langle\Psi(A)|\Psi(A)\rangle}.

Only the 108 structurally allowed tensor entries are varied.  The dense tensor
contains many more entries, but quimb's parametrized tensor maps the compact
vector into that masked array before contraction.

The sparse singlet initialization is a stationary point, so activate the
boundary-compatible sectors with a small perturbation:

.. code-block:: python

   initial = problem.perturb_parameters(
       ansatz.parameters,
       scale=1.0e-2,
       seed=0,
   )
   loss, gradient = problem.loss_and_gradient_autograd(initial)

A short optimization is then:

.. code-block:: python

   result = problem.optimize_with_quimb(
       ansatz.parameters,
       max_steps=20,
       noise_scale=1.0e-2,
       seed=0,
       autodiff_backend="autograd",
   )

``result`` contains the initial and final exact residual reports, compact
parameters, and the complete function-evaluation history.  A reduced variance
is only a discovery signal.  An exact cage candidate must reach zero residual,
remain stable across larger clusters, and admit a local PEPS eigenstate
certificate.

Visualization
-------------

The tensor visualizer draws the repeated graph, local boundary-resolved dimer
entries, parameter magnitudes, and optimization history:

.. code-block:: python

   from qlinks.visualizer import SquareQDMTensorNetworkVisualizer

   visualizer = SquareQDMTensorNetworkVisualizer(tile_basis)
   visualizer.plot_network(n_tiles_x=3, n_tiles_y=2)
   visualizer.plot_entry(0)
   visualizer.plot_parameter_magnitudes(result.optimized_parameters)
   visualizer.plot_optimization_history(result)

A complete interactive demonstration is provided in
``experimental/notebooks/tensor_network.ipynb``.  The padding notebook now ends
at the boundary-resolved handoff and does not duplicate the PEPS optimization.

Type-1 chiral specialization
---------------------------

A generic variance search asks the optimizer to rediscover the defining cage
mechanism.  The type-1 PEPS problem instead encodes the Fock-space chiral
structure explicitly.  If ``C`` anticommutes with the kinetic term and commutes
with the diagonal potential, a state on one chiral subset obeys

.. math::

   \operatorname{Var}(H)
   = \frac{\|K|\Psi_+\rangle\|^2}{\langle\Psi_+|\Psi_+\rangle}
   + \operatorname{Var}_{\Psi_+}(V).

The two non-negative terms are precisely the type-1 conditions: destructive
kinetic interference on the empty bipartite subset and uniform potential on
the occupied support.

Build the separated objective with:

.. code-block:: python

   from qlinks.caging import build_square_qdm_type1_peps_problem

   type1_problem = build_square_qdm_type1_peps_problem(
       model,
       tile_basis,
       reference_parameters=ansatz.parameters,
   )
   report = type1_problem.diagnose(ansatz.parameters)

   print(report.kinetic_interference_density)
   print(report.potential_variance_density)
   print(report.discarded_chiral_weight)

The PEPS amplitudes are projected exactly onto one kinetic-graph bipartite
subset before the objective is evaluated.  Existing finite type-1 cage records
can select the same subset and potential value directly:

.. code-block:: python

   type1_problem = build_square_qdm_type1_peps_problem(
       model,
       tile_basis,
       cage_record=record,
   )

The chiral coloring is also reconstructed as a linear link-occupation parity.
For the ``3 x 2`` square-QDM tile, qlinks finds a tile-periodic rule with 12
local link coefficients.  This assigns a ``Z2`` charge to each of the 71
compressed physical states.

Native chiral PEPS
------------------

The finite-basis projector has a native tensor-network counterpart.  Each
virtual leg is augmented by one ``Z2`` charge bit, and tensor entries obey local
charge conservation.  Contracting a closed torus cancels all virtual charges,
so only the selected global chiral subset survives.  The charge-resolved tensor
shares the original 108 variational amplitudes:

.. code-block:: python

   from qlinks.caging import SquareQDMChiralPEPSAnsatz

   chiral_ansatz = SquareQDMChiralPEPSAnsatz.from_type1_problem(
       type1_problem,
       ansatz.parameters,
   )
   chiral_tn = chiral_ansatz.to_quimb_tensor_network(
       n_tiles_x=2,
       n_tiles_y=2,
   )

For the ``3 x 2`` tile, the charge-augmented tensor has shape
``(16, 8, 16, 8, 71)`` and 864 nonzero structural entries, but still only 108
independent parameters.

The dedicated visualizer separates the physical mechanisms:

.. code-block:: python

   visualizer.plot_type1_components(report)
   visualizer.plot_chiral_physical_charges(
       type1_problem.parity_rule,
       model,
   )

The next analytical target is a local tensor equation equivalent to
``B psi_+(A) = 0`` together with zero potential variance, valid independently
of the finite torus used during optimization.
