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
