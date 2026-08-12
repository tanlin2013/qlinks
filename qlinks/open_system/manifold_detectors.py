"""Temporary compatibility facade for dark-manifold detector workflows.

Implementation lives in responsibility-specific ``manifold_*`` modules. New package code must
import those focused modules directly. This historical path remains only as migration
scaffolding and is intended to be removed when the refactored open-system API stabilizes.
"""

from __future__ import annotations

from types import ModuleType

from qlinks.open_system import manifold_dark as _dark
from qlinks.open_system import manifold_detector_types as _types
from qlinks.open_system import manifold_recycling as _recycling
from qlinks.open_system import manifold_residual as _residual

_FORWARD_MODULES: tuple[ModuleType, ...] = (
    _types,
    _dark,
    _recycling,
    _residual,
)

# ``__all__`` mirrors the public surface of the former eager facade. Private compatibility
# names remain available to explicit imports through ``__getattr__`` during migration.
_PUBLIC_EXPORTS = (
    "DarkDetectorMatrixReadout",
    "DarkOperatorTerm",
    "DressedManifoldDarkDetectorCandidate",
    "DressedManifoldDarkDetectorReport",
    "LocalOperatorMatrixReadout",
    "ManifoldDarkOperatorBasisReport",
    "ManifoldDarkOperatorCandidate",
    "RecycledFamilyKernelDiagnostics",
    "RecycledManifoldCandidateFamilyKernelReport",
    "RecycledManifoldCollectiveRecyclerGroup",
    "RecycledManifoldDarkDetectorCandidate",
    "RecycledManifoldDarkDetectorReport",
    "RecycledManifoldJumpSelectionReport",
    "RecycledManifoldJumpSelectionStep",
    "RecycledManifoldResidualKernelReport",
    "ResidualKernelLocalSupportEntry",
    "ResidualKernelOperatorActionEntry",
    "ResidualKernelOperatorActionReport",
    "TargetedResidualKernelJumpSelectionReport",
    "TargetedResidualKernelJumpSelectionStep",
    "TargetedResidualKernelLinearCandidate",
    "TargetedResidualKernelLinearSearchReport",
    "TargetedResidualKernelLinearTerm",
    "diagnose_dressed_manifold_dark_detectors",
    "diagnose_manifold_dark_operator_basis",
    "diagnose_recycled_manifold_candidate_family_kernel",
    "diagnose_recycled_manifold_dark_detectors",
    "diagnose_recycled_manifold_residual_kernel",
    "diagnose_targeted_residual_kernel_linear_search",
    "expand_local_regions_to_cluster_unions",
    "expand_local_regions_to_pair_unions",
    "select_recycled_manifold_dark_detector_jumps",
    "select_targeted_residual_kernel_jumps",
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
