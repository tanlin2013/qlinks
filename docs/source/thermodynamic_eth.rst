Thermodynamic ETH diagnostics
============================

The thermodynamic caging workflow separates two finite-size questions:

#. whether the same bounded local row operator :math:`L_R` annihilates a cage
   state at every size, and
#. whether :math:`Q_R=L_R^\dagger L_R` has a nonzero expectation value in a
   thermal ensemble from the same constrained symmetry sector.

The APIs in :mod:`qlinks.caging.thermodynamic` keep the local transition pattern
independent of global variable labels, evaluate the projected constrained-basis
operator exactly, and store finite-size scaling data without interpreting a
short fit as a proof of a thermodynamic limit.

Match one local witness across sizes
------------------------------------

After classifying one cage state at each size, identify reduced-IZ patterns that
are exactly common to all reports::

   from qlinks.caging import common_local_witness_families

   families = common_local_witness_families(
       {
           "4x4": report_4x4,
           "4x6": report_4x6,
           "4x8": report_4x8,
       }
   )

   family = families[0]
   witness_4x4 = family.witnesses_for("4x4")[0]

Translations are matched automatically because global variable indices are not
part of the template key.  Rotated or reflected copies must currently use a
consistent local variable ordering before classification.

Evaluate cage and thermal expectations
--------------------------------------

For a cage vector and the exact infinite-temperature ensemble in a constrained
basis::

   from qlinks.caging import make_eth_scaling_point

   point_4x4 = make_eth_scaling_point(
       system_size=model_4x4.lattice.num_plaquettes,
       witness=witness_4x4,
       basis_configs=build_4x4.basis.states,
       cage_state=cage_vector_4x4,
       energy=cage_energy_4x4,
       system_label="4x4",
   )

If ``thermal_states`` is supplied, the same helper evaluates an equal-weight or
weighted state ensemble.  A finite-size microcanonical shell can be selected
with :func:`qlinks.caging.evaluate_local_witness_microcanonical`.

Collect points built from the same template in an
:class:`qlinks.caging.ETHScalingReport`::

   from qlinks.caging import ETHScalingReport

   scaling = ETHScalingReport(
       template=family.template,
       points=(point_4x4, point_4x6, point_4x8),
   )
   fit = scaling.fit_expectation_gap(order=1)

``fit.thermodynamic_limit`` is the intercept of a descriptive
:math:`c_0+c_1/N` fit.  ``tail_liminf_lower_bound`` reports the minimum absolute
gap among the largest supplied sizes and is deliberately named as a finite-data
bound rather than a thermodynamic proof.

Factorized multi-block certification
------------------------------------

The old multi-block padding path materializes a global support of size
:math:`\prod_i d_i`, where :math:`d_i` is the support size of block ``i``.  The
factorized path stores only the shared exterior assignment and contracts the
Hamiltonian action as a sum of product vectors::

   from qlinks.caging import (
       certify_qdm_factorized_product_state,
       find_factorized_qdm_block_paddings,
   )

   paddings = find_factorized_qdm_block_paddings(
       model,
       block_pool,
       config=padding_config,
   )
   certificate = certify_qdm_factorized_product_state(
       model,
       selected_blocks,
       paddings[0],
       config=padding_config,
   )

The factorized certificate is exact when each plaquette touches at most one
selected block, corresponding to ``require_kinetic_separation=True``.  Its cost
is controlled by the largest individual block support and the number of local
Hamiltonian terms, not by the Cartesian product support.  Sector validation is
performed on a reference configuration and every single-block support
variation; the report records this validation mode explicitly.

Square-QDM strip transfer matrix
--------------------------------

For the square QDM, the constrained infinite-temperature expectation can be
computed without constructing the global dimer basis.  The y direction is a
periodic cylinder of fixed circumference ``Ly``.  A transfer state is the bit
mask of horizontal dimers entering one site column.  Each allowed column
transition specifies the outgoing horizontal dimers and the vertical dimers
inside that column.

A local witness is first converted from finite-system variable indices to
size-independent strip coordinates::

   from qlinks.caging import (
       SquareQDMStripTransferMatrix,
       SquareQDMWitnessPlacement,
   )

   placement = SquareQDMWitnessPlacement.from_local_witness(
       reference_model,
       witness,
   )
   transfer = SquareQDMStripTransferMatrix(
       circumference=reference_model.ly,
   )

The conversion automatically unwraps a witness that crosses the x seam of a
periodic reference torus.  The local-variable order is preserved exactly.

Evaluate a finite torus with periodic x boundaries::

   finite_torus = transfer.evaluate_witness(
       placement,
       length=reference_model.lx,
       boundary_x="periodic",
   )

``finite_torus.expectation`` is

.. math::

   \frac{\operatorname{Tr}_{\mathcal H_{\rm dimer}} Q_R}
        {\dim \mathcal H_{\rm dimer}}.

The contraction reproduces the projected constrained-basis operator, not a
naive local trace.  For each source local pattern it tests whether replacing it
by a target pattern leaves every incident dimer constraint satisfied while the
exterior configuration is kept fixed.  Consequently, a one-link lowering
operator has zero projected weight, whereas a plaquette flip has the expected
flippability probability.

For an infinite-cylinder sequence, keep ``Ly`` fixed and increase the x
length::

   scaling = transfer.scan_witness(
       placement,
       lengths=(8, 12, 16, 24, 32),
       boundary_x="open",
   )
   scaling.tail_estimate(tail_points=3)

The open-strip contraction places the witness in the center by default and
normalizes the left and right transfer vectors at every step, so lengths can be
increased without overflowing the raw number of coverings.  Periodic-x
contractions use a symmetric transfer-matrix eigendecomposition and are
currently limited to at most 2048 boundary states.

Winding-resolved periodic contractions
--------------------------------------

For a strict ETH comparison, the finite torus can be projected into the same
electric winding sector as the cage state.  The labels use the same convention
as ``SquareQDMModel(winding_convention="electric")``::

   from qlinks.caging import SquareQDMStripWindingSector

   sector = SquareQDMStripWindingSector(
       winding_x=0,
       winding_y=0,
   )
   w00 = transfer.evaluate_witness(
       placement,
       length=reference_model.lx,
       boundary_x="periodic",
       winding_sector=sector,
   )

The x winding is selected from the transfer boundary mask.  The y winding is
accumulated as an integer charge on each column transition.  The witness
insertion is resolved by the same charge before the numerator is contracted.
Thus both the partition count and ``Tr(Q_R)`` are evaluated inside one exact
winding sector.

All nonempty sectors can be counted without constructing the global basis::

   counts = transfer.periodic_winding_sector_counts(
       length=reference_model.lx,
   )

For the square ``4x4`` QDM, this reproduces the known decomposition, including
``counts[(0, 0)] = 132`` after indexing by ``sector.label``.  Sector-resolved
periodic contractions currently use charge-resolved dynamic programming and
are limited to ``Ly <= 9``.  The local insertion must not cross the canonical x
seam; choose ``insertion_x`` explicitly when necessary.

Evaluate a common cage witness family
-------------------------------------

The output of :func:`qlinks.caging.common_local_witness_families` can be sent
directly to the strip backend::

   from qlinks.caging import evaluate_square_qdm_witness_family_on_strips

   strip_family = evaluate_square_qdm_witness_family_on_strips(
       family,
       models={
           "4x4": model_4x4,
           "6x4": model_6x4,
           "8x4": model_8x4,
       },
       lengths={
           "4x4": (4,),
           "6x4": (6,),
           "8x4": (8,),
       },
       boundary_x="periodic",
       winding_sector=(0, 0),
   )

Each record retains the finite-system embedding, the normalized strip
placement, and its scaling report.  This closes the loop between a reduced-IZ
classification report and a same-sector thermal witness calculation.
