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
