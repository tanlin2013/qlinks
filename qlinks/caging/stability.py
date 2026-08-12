"""Temporary compatibility facade for cage-stability diagnostics.

The implementation is decomposed by scientific responsibility. New package code must import
from the focused ``stability_*`` modules. This module preserves the historical
``qlinks.caging.stability`` path only while first-party and downstream callers migrate; it is
intended to be removed once the refactored API is declared stable.
"""

from __future__ import annotations

from types import ModuleType

from qlinks.caging import stability_boundary as _boundary
from qlinks.caging import stability_core as _core
from qlinks.caging import stability_laurent as _laurent
from qlinks.caging import stability_qdm as _qdm
from qlinks.caging import stability_symmetry as _symmetry
from qlinks.caging import stability_topology as _topology
from qlinks.caging import stability_types as _types

_FORWARD_MODULES: tuple[ModuleType, ...] = (
    _types,
    _core,
    _topology,
    _boundary,
    _qdm,
    _laurent,
    _symmetry,
)

# ``__all__`` mirrors the public surface of the former eager facade. Private compatibility
# names remain available to explicit imports through ``__getattr__`` during migration.
_PUBLIC_EXPORTS = (
    "BoundaryCancellationCircuitEntry",
    "BoundaryCancellationMatroidBranchPoint",
    "BoundaryCancellationMatroidBranchReport",
    "BoundaryCancellationMatroidReport",
    "BoundaryCancellationMomentumPoint",
    "BoundaryCancellationScalingPoint",
    "BoundaryIncidenceCohomologyReport",
    "CageBranchPoint",
    "CageBranchReport",
    "CageCompatibilityHierarchyReport",
    "CageHamiltonianBlocks",
    "CageJacobianConditioningReport",
    "CageRecordStabilitySummary",
    "CageStabilityDiagnostic",
    "ChiralIndexReport",
    "CoefficientField",
    "CyclicAmplitudeBondProfile",
    "FixedCageManifoldCompatibilityReport",
    "FixedCageStateCompatibilityReport",
    "HardCoreLaurentLiftReport",
    "LaurentDimensionDivisibilityViolation",
    "LaurentPeriodicDimensionConsistencyReport",
    "LaurentPolynomialConstraintModuleReport",
    "LaurentPolynomialPeriodicPoint",
    "LaurentPolynomialRootMode",
    "LaurentPolynomialTorsionOrder",
    "LinearizedCageObstructionReport",
    "LocalityRestrictedChiralProfileReport",
    "ManyBodyCLSCompletenessReport",
    "ManyBodyCLSCompletenessSequencePoint",
    "ManyBodyCLSCompletenessSequenceReport",
    "ManyBodyCLSGeneratorOrbitEntry",
    "ManyBodyCLSTranslationSector",
    "ManyBodyTopologicalLocalizationReport",
    "PeriodicBoundaryCancellationScalingReport",
    "PerturbationCompatibilityDiagnostic",
    "QDMCompactCageReducedWindingPoint",
    "QDMCompactCageReducedWindingReport",
    "QDMCyclicColumnGrammar",
    "QDMCyclicGrammarSupport",
    "QDMExplicitProductSupport",
    "QDMExplicitSupportBoundaryMap",
    "QDMLocalGrammarExtensionPoint",
    "QDMLocalGrammarExtensionReport",
    "QDMLocalKineticCompatibilityReport",
    "QDMLocalPotentialCompatibilityReport",
    "QDMPhysicalCancellationScalingPoint",
    "QDMPhysicalCancellationScalingReport",
    "RandomCageStabilityAggregate",
    "RandomCageStabilityReport",
    "RandomCageStabilitySample",
    "RealLocalSignObstructionReport",
    "ReducedConstraintFredholmCandidateReport",
    "RegionalCageQuotientReport",
    "RegionalChiralIndexEntry",
    "RegionalChiralKernelSpanReport",
    "RelativeMod2CycleReport",
    "SignedBoundaryCycle",
    "SignedBoundaryHolonomyReport",
    "SquareQDMColumnSymbol",
    "SquareQDMColumnWord",
    "SquareQDMFiniteBondTransferInvariantReport",
    "SquareQDMTransferSectorMultiplicity",
    "SupportEigenstateBranchPoint",
    "SupportEigenstateBranchReport",
    "boundary_cancellation_matroid_from_hamiltonian",
    "build_qdm_explicit_support_boundary",
    "cage_compatibility_hierarchy_from_hamiltonians",
    "cage_jacobian_conditioning",
    "cage_jacobian_conditioning_from_hamiltonian",
    "combine_perturbations_from_coefficients",
    "diagnose_boundary_cancellation_matroid",
    "diagnose_boundary_incidence_cohomology",
    "diagnose_cage_stability",
    "diagnose_chiral_index",
    "diagnose_cyclic_amplitude_bond_profile",
    "diagnose_hard_core_laurent_lift",
    "diagnose_laurent_periodic_dimension_consistency",
    "diagnose_laurent_polynomial_constraint_module",
    "diagnose_locality_restricted_chiral_profile",
    "diagnose_many_body_cls_completeness",
    "diagnose_many_body_topological_localization",
    "diagnose_qdm_local_kinetic_compatibility",
    "diagnose_qdm_local_potential_compatibility",
    "diagnose_real_local_sign_obstruction",
    "diagnose_reduced_constraint_fredholm_candidate",
    "diagnose_relative_mod2_cycles",
    "diagnose_signed_boundary_holonomy",
    "diagnose_square_qdm_compact_cage_reduced_winding",
    "diagnose_square_qdm_finite_bond_transfer_invariant",
    "enumerate_qdm_cyclic_column_grammar_words",
    "estimate_power_law_exponent",
    "fixed_cage_manifold_compatibility",
    "fixed_cage_manifold_compatibility_from_hamiltonians",
    "fixed_cage_state_compatibility",
    "fixed_cage_state_compatibility_from_hamiltonians",
    "infer_square_qdm_cyclic_column_grammar",
    "laurent_polynomial_constraint_symbol",
    "linearized_cage_obstruction",
    "linearized_cage_obstruction_from_hamiltonians",
    "materialize_square_qdm_cyclic_grammar_support",
    "materialize_square_qdm_periodic_product_support",
    "partition_cage_hamiltonian",
    "periodic_boundary_cancellation_symbol",
    "random_cage_stability_ensemble",
    "regional_cage_quotient",
    "regional_chiral_kernel_span",
    "scan_boundary_cancellation_matroid",
    "scan_cage_stability_branch",
    "scan_periodic_boundary_cancellation_scaling",
    "scan_square_qdm_collective_locality_extension",
    "scan_square_qdm_periodic_product_cancellation_scaling",
    "scan_support_eigenstate_branch",
    "square_qdm_column_words",
    "subspace_complement_basis",
    "subspace_principal_overlaps",
    "subspace_projector_distance",
    "summarize_cage_record_stability",
)

# Keep ``__all__`` indirect so pyflakes does not treat lazy compatibility exports as
# missing eager module bindings. Runtime ``import *`` still resolves these names
# through ``__getattr__``. Remove this facade once migration is complete.
__all__ = _PUBLIC_EXPORTS


def __getattr__(name: str) -> object:
    """Resolve a legacy attribute from its focused implementation module."""
    for module in _FORWARD_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose forwarded compatibility names to interactive introspection."""
    forwarded = {name for module in _FORWARD_MODULES for name in vars(module)}
    return sorted(set(globals()) | forwarded)
