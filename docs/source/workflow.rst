Typical workflow
================

The high-level workflow is:

#. Create a model.
#. Build the constrained basis and Hamiltonian.
#. Inspect the model sectors or sparse terms when needed.
#. Search for cage states.
#. Classify, visualize, or use the cage in an open-system construction.

Build a model
-------------

A model object packages the lattice, variable layout, constraints, sectors, and
local Hamiltonian terms.  For example, a square QDM in the zero electric winding
sector can be built as follows:

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

Build the constrained Hamiltonian
---------------------------------

Use :meth:`model.build() <qlinks.models.HamiltonianModelBase.build>` when you
need the basis and named terms, not only the total Hamiltonian:

.. code-block:: python

   build_result = model.build(
       basis_solver="dfs",
       builder="sparse",
       backend="scipy",
       sort_basis=True,
       on_missing="raise",
   )

   hamiltonian = build_result.hamiltonian
   kinetic = build_result.kinetic
   potential = build_result.potential
   basis = build_result.basis

The returned :class:`qlinks.models.ModelBuildResult` is the preferred object to
pass into higher-level tools because it keeps the matrix, basis, and model
metadata aligned.

Builder choices
---------------

``builder="sparse"``
   General sparse builder.  This is the easiest builder to use while developing
   new models or operators.

``builder="optimized"``
   Sparse builder with additional internal optimizations for repeated local
   updates.

``builder="bitmask"``
   Encoded builder for binary local spaces.  This is useful for larger QDM/QLM
   workflows but requires compatible binary operator implementations.

When comparing matrices from two builders, always compare them in the same basis
order.  Either reuse a precomputed basis or assert that both build results have
identical basis configurations before comparing matrices.

Inspect sectors
---------------

Many periodic models expose available or nonempty sector labels:

.. code-block:: python

   print(model.allowed_sector_labels())
   print(model.nonempty_sector_labels())

The meaning of a sector label is model- and convention-dependent.  For example,
square QDM winding labels can use either cut-count or staggered electric-flux
conventions depending on ``winding_convention``.

Search for cages
----------------

A cage search consumes the built Hamiltonian, kinetic term, and potential term:

.. code-block:: python

   from qlinks.caging import CageSearchConfig, CageSearcher

   config = CageSearchConfig(
       search_type="type1_and_type2",
       tolerance=1e-10,
       degenerate_basis_strategy="ipr",
       ipr_n_restarts=256,
       ipr_candidate_count=128,
       ipr_random_seed=1234,
   )

   searcher = CageSearcher.from_model_build_result(
       build_result,
       config=config,
   )
   search_result = searcher.run()

   print(search_result.counts_by_signature)

A signature is usually interpreted as ``(kappa, z)``, where ``kappa`` is the
kinetic eigenvalue target and ``z`` is the inferred uniform potential/self-loop
value on the cage support.

Classify a cage
---------------

The classification layer explains how a cage avoids leaking into neighboring
Fock-space states.  It is most useful after selecting one record from a search
result:

.. code-block:: python

   from qlinks.caging import CageClassificationConfig, classify_cage_state

   record = search_result.first()

   report = classify_cage_state(
       record.cage_state,
       kinetic_matrix=build_result.kinetic,
       basis_configs=build_result.basis.states,
       config=CageClassificationConfig(),
   )

The report contains interference-zero records, reduced local operator supports,
classification labels, and diagnostics that can feed visualizers or open-system
construction routines.

Visualize
---------

Use real-space visualizers to inspect basis configurations and support patterns:

.. code-block:: python

   from qlinks.visualizer import BasisGridVisualizer

   grid = BasisGridVisualizer(
       lattice=model.lattice,
       layout=model.layout,
       periodic_image_mode="positive_patch",
   )

   grid.plot_cage_support(
       search_result,
       basis_configs=build_result.basis.states,
       signature=(0, 4),
       record_index=0,
       max_states=16,
   )

Use Fock-space graph visualizers to inspect the Hamiltonian graph:

.. code-block:: python

   from qlinks.visualizer import HamiltonianGraphVisualizer

   graph = HamiltonianGraphVisualizer.from_sparse_matrix(
       build_result.kinetic,
       include_self_loops=False,
   )

   graph.plot(layout="kk")

Open-system construction
------------------------

The open-system layer is intended for Lindblad dark-state engineering and
verification around cage states.  The high-level caging entry point constructs a
problem from a type-1 cage, its classification report, and model metadata:

.. code-block:: python

   from qlinks.open_system.constructions import build_type1_cage_lindblad_construction

   construction = build_type1_cage_lindblad_construction(
       model=model,
       build_result=build_result,
       cage_state=record.full_state,
       classification_report=report,
       z_value=record.potential_value,
   )

   print(construction.n_jumps)
   print(construction.max_jump_residual)

This part of the API is still under active development, so treat construction
options as research controls rather than fixed application-level defaults.
