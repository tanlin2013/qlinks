"""Open-system diagnostics grouped by numerical responsibility.

Use the focused child modules inside qlinks. This package-level API is the curated
public diagnostics surface for external callers.
"""

from qlinks.open_system.diagnostics.absorbing import (
    AbsorbingProjectorJumpDiagnostics,
    AbsorbingProjectorSymmetryDiagnostics,
    diagnose_absorbing_projector_symmetry,
)
from qlinks.open_system.diagnostics.attractive import (
    AttractiveSubspaceDiagnostics,
    diagnose_attractive_subspace,
)
from qlinks.open_system.diagnostics.dark import (
    CommonKernelHamiltonianInvariantSectorReport,
    DarkManifoldDiagnostics,
    DarkSubspaceDiagnostics,
    bad_h_invariant_common_kernel_basis,
    diagnose_common_kernel_h_invariant_sector,
    diagnose_dark_manifold,
    diagnose_dark_subspace,
)
from qlinks.open_system.diagnostics.evolution import (
    EvolutionDiagnostics,
    analyze_lindblad_evolution,
)
from qlinks.open_system.diagnostics.jumps import (
    JumpSpanDiagnostics,
    diagnose_jump_span,
    jump_activity,
    jump_activity_series,
)
from qlinks.open_system.diagnostics.monitor import (
    MonitorKernelClosureDiagnostics,
    diagnose_monitor_kernel_closure,
)
from qlinks.open_system.diagnostics.target_manifold import (
    target_manifold_coherence_series,
    target_manifold_density_matrix,
    target_manifold_density_matrix_series,
    target_manifold_entropy_series,
    target_manifold_populations_series,
    target_manifold_projector,
    target_manifold_purity_series,
    target_manifold_weight,
    target_manifold_weight_series,
)
from qlinks.open_system.diagnostics.verification import (
    DensityMatrixVerification,
    LindbladFinalStateVerification,
    verify_density_matrix,
    verify_lindblad_final_state,
)

__all__ = [
    "AbsorbingProjectorJumpDiagnostics",
    "AbsorbingProjectorSymmetryDiagnostics",
    "AttractiveSubspaceDiagnostics",
    "CommonKernelHamiltonianInvariantSectorReport",
    "DarkManifoldDiagnostics",
    "DarkSubspaceDiagnostics",
    "DensityMatrixVerification",
    "EvolutionDiagnostics",
    "JumpSpanDiagnostics",
    "LindbladFinalStateVerification",
    "MonitorKernelClosureDiagnostics",
    "analyze_lindblad_evolution",
    "bad_h_invariant_common_kernel_basis",
    "diagnose_absorbing_projector_symmetry",
    "diagnose_attractive_subspace",
    "diagnose_common_kernel_h_invariant_sector",
    "diagnose_dark_manifold",
    "diagnose_dark_subspace",
    "diagnose_jump_span",
    "diagnose_monitor_kernel_closure",
    "jump_activity",
    "jump_activity_series",
    "target_manifold_coherence_series",
    "target_manifold_density_matrix",
    "target_manifold_density_matrix_series",
    "target_manifold_entropy_series",
    "target_manifold_populations_series",
    "target_manifold_projector",
    "target_manifold_purity_series",
    "target_manifold_weight",
    "target_manifold_weight_series",
    "verify_density_matrix",
    "verify_lindblad_final_state",
]
