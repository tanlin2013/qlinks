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
``counts[(0, 0)] = 132`` after indexing by ``sector.label``.

Two winding-projection backends are available.  The charge-resolved dynamic
program is the small-width reference implementation.  The Fourier backend
introduces a phase for every positive y-winding charge, evaluates the twisted
transfer product at roots of unity, and extracts the requested coefficient by
a discrete Fourier transform::

   w00_fourier = transfer.evaluate_witness(
       placement,
       length=reference_model.lx,
       boundary_x="periodic",
       winding_sector=(0, 0),
       winding_projection="fourier",
   )

``winding_projection="auto"`` uses dynamic programming for at most 512
boundary states and Fourier projection beyond that point.  Exact projection
uses ``length + 1`` roots of unity by default.  A smaller ``fourier_points``
value gives an explicitly labelled aliased approximation, useful for exploratory
long-strip scans.  The metadata fields ``exact_fourier_projection`` and
``aliased_fourier_projection`` distinguish these cases.

The dense Fourier backend supports at most 1024 states in one x-winding block,
which reaches the central sector through ``Ly=12``.  Unlike the dynamic program,
it also permits a local insertion that crosses the canonical x seam.  The
current Fourier implementation assumes even ``Lx`` and ``Ly``.

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

Normalize and evaluate the actual cage-derived witness
-------------------------------------------------------

A reduced-IZ row has an arbitrary overall coefficient.  For thermodynamic
comparisons use operator-norm normalization, which fixes
``||Q_R|| = ||L_R||^2 = 1``::

   witnesses = local_witnesses_from_classification_report(
       classification,
       normalization="operator_norm",
   )

The whole classification report can be sent directly to the strip backend::

   actual_witnesses = evaluate_square_qdm_classification_witnesses_on_strips(
       classification,
       model=model_4x4,
       lengths=(4, 6, 8, 10, 16),
       winding_sector=(0, 0),
       normalization="operator_norm",
   )

The returned records retain the original interference-zero indices and local
matrix-unit pattern.  This is the preferred API for replacing the pedagogical
directed-plaquette witness by the operator that actually annihilates a cage.
A bounded placement can also be reused on a wider cylinder with
``placement.with_circumference(new_ly)``.

Exact arbitrary-repeat cage sequences
-------------------------------------

For the square-QDM stripe cages, the global product support need not be formed.
Construct a coordinate-level unit cell from one certified multi-block padding::

   from qlinks.caging import (
       SquareQDMPeriodicProductUnitCell,
       certify_square_qdm_periodic_product_sequence,
   )

   unit_cell = SquareQDMPeriodicProductUnitCell.from_padding(
       model_4x4,
       search_context.blocks,
       certified.reports[0].padding,
       repeat_axis="y",
   )
   sequence = certify_square_qdm_periodic_product_sequence(unit_cell)

The certificate checks the following local identities.

* Every block contributes a support-independent dimer count at each site.
* Plaquettes touching more than one coherent factor are nonflippable for every
  combination of the local support patterns.
* All plaquettes touching exactly one factor close on that factor with a fixed
  kinetic eigenvalue and a constant potential value.
* The unit cell is repeated by an exact coordinate translation and the
  couplings are translation invariant.

Three repeats expose both neighboring cells for a range-one plaquette
Hamiltonian.  Once these local action classes pass, the same identities hold
for every larger repeat count without diagonalizing or even materializing the
formal product support.  The smaller one- and two-cell rings are checked
separately, so a successful report certifies every positive integer ``n``.
The electric winding label is evaluated without materializing the product
support and propagated to every repeat.  For the rotated current ``4x4`` stripe unit cell, all members lie in
``(w_x,w_y)=(0,0)`` and the result is an exact sequence on ``(4 n) x 4``
tori with

.. math::

   E_n = 4 n,
   \qquad
   \frac{E_n}{N_n} = \frac{1}{4},
   \qquad
   |\operatorname{supp}\Psi_n| = 4^n,

while the stored representation grows only linearly in ``n``.

Propagate a local ETH witness to the whole sequence::

   sequence_witness = certify_local_witness_on_square_qdm_periodic_sequence(
       sequence,
       witnesses[0],
       normalization="operator_norm",
   )
   assert sequence_witness.is_infinite_sequence_witness

This proves ``L_R |Psi_n> = 0`` for every certified repeat, not merely for the
finite reference state.

This is a rigorous fixed-width thermodynamic sequence: ``N=16 n`` diverges and
the local ETH discrepancy can remain order one, but the transverse width is
still four.  It should therefore be described as a strip or quasi-one-dimensional
thermodynamic theorem.  Establishing a genuine two-dimensional sequence with
both linear dimensions diverging remains a separate construction problem.

Match the cage energy density to beta zero
------------------------------------------

For a uniform square QDM, the kinetic term is off-diagonal in the dimer basis
and has zero infinite-temperature trace.  The potential energy density is the
potential coupling times the probability that a plaquette is flippable in
either orientation.

For the fixed-width sequence with a nonzero potential coupling, the cage value
``e_cage = 1/4`` does not coincide exactly with the fixed-width beta-zero value.
That Hamiltonian therefore requires a finite-temperature calculation at
``beta_*`` determined by ``e_th(beta_*) = e_cage``; the beta-zero API alone must
not be used to claim energy-shell matching.

An exact beta-zero match is already available for the pure kinetic square QDM.
Reuse the certified geometry while setting the potential coupling to zero::

   kinetic_cell = unit_cell.with_couplings(coup_pot=0.0)
   kinetic_sequence = certify_square_qdm_periodic_product_sequence(kinetic_cell)

Then ``E_n = 0`` and ``e_cage = 0`` exactly.  Because the pure kinetic
Hamiltonian is off-diagonal in the dimer basis, ``Tr H = 0`` in every finite
winding sector.  Hence the beta-zero energy density is also exactly zero at
every size::

   beta_zero = scan_square_qdm_beta_zero_energy_density(
       ((4, 4), (8, 4), (12, 4), (16, 4)),
       potential_coupling=0.0,
       winding_sector=(0, 0),
   )
   match = kinetic_sequence.match_energy_density(
       beta_zero.evaluations[-1].energy_density,
       tolerance=1.0e-12,
   )
   assert match.is_matched

Thus the pure kinetic family supplies the cleanest current strong-ETH test:
an exact infinite cage sequence, an exactly matched beta-zero energy density,
and an operator-norm-normalized local cage witness with nonzero same-sector
thermal weight.  The nonzero-potential model remains a finite-temperature
extension rather than an established beta-zero result.

Bi-periodic product-tile certification
--------------------------------------

A one-axis sequence such as ``(4n) x 4`` is only a fixed-width thermodynamic
limit.  The bi-periodic API tests whether one finite product tile can be
repeated with two independent integers ``nx`` and ``ny``::

   from qlinks.caging import (
       SquareQDMBiperiodicProductTile,
       diagnose_square_qdm_biperiodic_repeatability,
   )

   tile = SquareQDMBiperiodicProductTile.from_padding(
       model,
       block_pool,
       padding,
   )
   diagnosis = diagnose_square_qdm_biperiodic_repeatability(tile)

The default certificate checks all arrays with ``1 <= nx, ny <= 3``.  The
``3 x 3`` array contains every range-one local environment: tile interiors,
x seams, y seams, and four-tile corners.  The one- and two-tile tori are checked
separately because their periodic identifications can create exceptional short
rings.  If all checks pass, coordinate translation and uniform couplings prove
an exact family for every pair of positive repeat counts.

The returned seam diagnostics distinguish:

* support-dependent dimer charge at x/y seams or corners;
* nominally inert seam plaquettes that become flippable;
* leakage or nonclosure of a single coherent block;
* support-dependent winding labels.

This distinction is essential for the known square-QDM stripe cage: its tile
interior closes, but transverse repetition changes the boundary dimer charge.
The result is therefore a precise local obstruction, rather than a large-system
residual.

Direct finite-tile search
-------------------------

A pool of already discovered local cage blocks can be searched directly for a
periodic exterior and then subjected to the bi-periodic certificate::

   from qlinks.caging import (
       SquareQDMBiperiodicTileSearchConfig,
       search_square_qdm_biperiodic_product_tiles,
   )

   search = search_square_qdm_biperiodic_product_tiles(
       model,
       block_pool,
       config=SquareQDMBiperiodicTileSearchConfig(
           min_blocks=2,
           max_blocks=4,
           require_kinetic_separation=False,
           max_tile_support_size=4096,
       ),
   )

Only the finite reference-tile product support is materialized, with an
explicit ``max_tile_support_size`` bound.  The exponentially larger support of
the repeated two-dimensional state is never formed.  Failed candidates are
retained and grouped by failure mechanism, allowing a systematic finite-period
no-go scan instead of repeatedly enlarging a global active region.

This product-tile layer is intentionally independent of a tensor-network
backend.  If finite static gluing fails at moderate periods, the boundary
signatures and local action maps produced here should become the physical and
virtual data supplied to an established tensor-network library rather than a
new contraction engine implemented inside qlinks.
