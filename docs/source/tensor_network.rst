Tensor-network cage ansatzes
============================

Install the optional backend with:

.. code-block:: bash

   pip install "qlinks[tn]"

``quimb`` is used for tensor representation, contraction, and automatic-
differentiation optimization.  qlinks remains responsible for constructing the
constrained local basis and the exact finite-cluster Hamiltonian objective.

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

Exact finite-cluster objective
------------------------------

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
   report = problem.diagnose(ansatz.parameters)

The loss is the energy variance,

.. math::

   \mathcal L(A)
   = \left\|\left(H-\langle H\rangle_A\right)|\Psi(A)\rangle\right\|^2.

A quimb optimizer can be constructed with:

.. code-block:: python

   optimizer = problem.make_quimb_optimizer(
       ansatz.parameters,
       autodiff_backend="AUTO",
       optimizer="L-BFGS-B",
   )

The automatic-differentiation backend is selected by quimb.  JAX or PyTorch can
be installed separately when optimization, rather than contraction alone, is
required.
