from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.linalg as scipy_linalg

from qlinks.open_system.backend import (
    OpenSystemBackend,
    OpenSystemBackendName,
    as_backend_dense_array,
    get_open_system_backend,
)
from qlinks.open_system.operators import (
    build_liouvillian,
    lindblad_rhs_density_matrix,
)


@dataclass(frozen=True, slots=True)
class EvolutionDiagnostics:
    trace_errors: np.ndarray
    hermiticity_errors: np.ndarray
    min_eigenvalues: np.ndarray
    purities: np.ndarray
    fidelities: np.ndarray | None
    lindblad_residuals: np.ndarray | None


def analyze_lindblad_evolution(
    density_matrices: list[np.ndarray],
    *,
    target_state: np.ndarray | None = None,
    hamiltonian=None,
    jumps=None,
    atol: float = 1e-10,
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
) -> EvolutionDiagnostics:
    density_diagnostics = [
        verify_density_matrix(
            density_matrix,
            target_state=target_state,
            atol=atol,
        )
        for density_matrix in density_matrices
    ]

    lindblad_residuals = None
    if hamiltonian is not None and jumps is not None:
        lindblad_residuals = np.array(
            [
                np.linalg.norm(
                    lindblad_rhs_density_matrix(
                        density_matrix,
                        hamiltonian=hamiltonian,
                        jumps=jumps,
                        backend=backend,
                    )
                )
                for density_matrix in density_matrices
            ],
            dtype=np.float64,
        )

    fidelities = None
    if target_state is not None:
        fidelities = np.array(
            [diagnostic.fidelity_with_target for diagnostic in density_diagnostics],
            dtype=np.float64,
        )

    return EvolutionDiagnostics(
        trace_errors=np.array([d.trace_error for d in density_diagnostics]),
        hermiticity_errors=np.array([d.hermiticity_error for d in density_diagnostics]),
        min_eigenvalues=np.array([d.min_eigenvalue for d in density_diagnostics]),
        purities=np.array([d.purity for d in density_diagnostics]),
        fidelities=fidelities,
        lindblad_residuals=lindblad_residuals,
    )


@dataclass(frozen=True, slots=True)
class DensityMatrixVerification:
    trace: complex
    trace_error: float
    hermiticity_error: float
    min_eigenvalue: float
    purity: float
    fidelity_with_target: float | None
    is_hermitian: bool
    is_trace_one: bool
    is_positive_semidefinite: bool
    is_density_matrix: bool


def verify_density_matrix(
    rho: npt.ArrayLike,
    *,
    target_state: npt.ArrayLike | None = None,
    atol: float = 1e-10,
) -> DensityMatrixVerification:
    rho_array = np.asarray(rho, dtype=np.complex128)

    if rho_array.ndim != 2 or rho_array.shape[0] != rho_array.shape[1]:
        raise ValueError("rho must be a square matrix.")

    trace = np.trace(rho_array)
    trace_error = float(abs(trace - 1.0))

    hermitian_part = 0.5 * (rho_array + rho_array.conj().T)
    hermiticity_error = float(np.linalg.norm(rho_array - rho_array.conj().T))

    eigenvalues = np.linalg.eigvalsh(hermitian_part)
    min_eigenvalue = float(np.min(eigenvalues))

    purity_value = float(np.real(np.trace(rho_array @ rho_array)))

    fidelity = None
    if target_state is not None:
        psi = np.asarray(target_state, dtype=np.complex128)

        if psi.ndim != 1:
            raise ValueError("target_state must be one-dimensional.")

        norm = np.linalg.norm(psi)
        if norm == 0:
            raise ValueError("target_state must be nonzero.")

        psi = psi / norm
        fidelity = float(np.real(np.vdot(psi, rho_array @ psi)))

    is_hermitian = hermiticity_error <= atol
    is_trace_one = trace_error <= atol
    is_positive = min_eigenvalue >= -atol

    return DensityMatrixVerification(
        trace=complex(trace),
        trace_error=trace_error,
        hermiticity_error=hermiticity_error,
        min_eigenvalue=min_eigenvalue,
        purity=purity_value,
        fidelity_with_target=fidelity,
        is_hermitian=is_hermitian,
        is_trace_one=is_trace_one,
        is_positive_semidefinite=is_positive,
        is_density_matrix=(is_hermitian and is_trace_one and is_positive),
    )


@dataclass(frozen=True, slots=True)
class LindbladFinalStateVerification:
    density_matrix: DensityMatrixVerification
    lindblad_residual: float
    relative_lindblad_residual: float


def verify_lindblad_final_state(
    rho: npt.ArrayLike,
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    target_state: npt.ArrayLike | None = None,
    atol: float = 1e-10,
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
) -> LindbladFinalStateVerification:
    rho_array = np.asarray(rho, dtype=np.complex128)

    density = verify_density_matrix(
        rho_array,
        target_state=target_state,
        atol=atol,
    )

    rhs = lindblad_rhs_density_matrix(
        rho_array,
        hamiltonian=hamiltonian,
        jumps=jumps,
        backend=backend,
    )

    residual = float(np.linalg.norm(rhs))
    relative = residual / max(1.0, float(np.linalg.norm(rho_array)))

    return LindbladFinalStateVerification(
        density_matrix=density,
        lindblad_residual=residual,
        relative_lindblad_residual=relative,
    )


@dataclass(frozen=True, slots=True)
class DarkSubspaceDiagnostics:
    """Diagnostics for whether a dark target is unique/attractive."""

    dim: int
    n_jumps: int

    target_norm: float
    target_jump_residuals: tuple[float, ...]
    max_target_jump_residual: float
    target_liouvillian_residual: float

    common_jump_kernel_dimension: int
    target_projection_onto_common_kernel: float
    target_distance_from_common_kernel: float
    target_in_common_jump_kernel: bool
    bad_common_jump_kernel_dimension: int
    bad_common_jump_kernel_iprs: tuple[float, ...]

    liouvillian_zero_mode_count: int | None
    liouvillian_spectral_gap: float | None
    liouvillian_eigenvalues: tuple[complex, ...]

    likely_unique_dark_state: bool | None

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "dim": self.dim,
            "n_jumps": self.n_jumps,
            "max_target_jump_residual": self.max_target_jump_residual,
            "target_liouvillian_residual": self.target_liouvillian_residual,
            "common_jump_kernel_dimension": self.common_jump_kernel_dimension,
            "target_projection_onto_common_kernel": (self.target_projection_onto_common_kernel),
            "target_distance_from_common_kernel": (self.target_distance_from_common_kernel),
            "target_in_common_jump_kernel": self.target_in_common_jump_kernel,
            "bad_common_jump_kernel_dimension": (self.bad_common_jump_kernel_dimension),
            "bad_common_jump_kernel_iprs": self.bad_common_jump_kernel_iprs,
            "liouvillian_zero_mode_count": self.liouvillian_zero_mode_count,
            "liouvillian_spectral_gap": self.liouvillian_spectral_gap,
            "likely_unique_dark_state": self.likely_unique_dark_state,
        }

    def to_rich(self):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "DarkSubspaceDiagnostics.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()

        overview.add_row("Hilbert dimension", str(self.dim))
        overview.add_row("number of jumps", str(self.n_jumps))
        overview.add_row(
            "likely unique dark state",
            str(self.likely_unique_dark_state),
        )

        target = Table(title="Target checks")
        target.add_column("quantity", style="bold")
        target.add_column("value", justify="right")
        target.add_column("status", justify="center")

        target.add_row(
            "max ||J_mu psi||",
            _format_float(self.max_target_jump_residual),
            _status_for_residual(self.max_target_jump_residual),
        )
        target.add_row(
            "||L(rho_psi)||",
            _format_float(self.target_liouvillian_residual),
            _status_for_residual(self.target_liouvillian_residual),
        )

        jump_kernel = Table(title="Common jump kernel")
        jump_kernel.add_column("quantity", style="bold")
        jump_kernel.add_column("value", justify="right")

        jump_kernel.add_row(
            "dim intersection ker J_mu",
            str(self.common_jump_kernel_dimension),
        )
        jump_kernel.add_row(
            "projection of psi onto kernel",
            _format_float(self.target_projection_onto_common_kernel),
        )
        jump_kernel.add_row(
            "distance of psi from kernel",
            _format_float(self.target_distance_from_common_kernel),
        )
        jump_kernel.add_row(
            "target in common kernel",
            str(self.target_in_common_jump_kernel),
        )
        jump_kernel.add_row(
            "bad common-kernel dimension",
            str(self.bad_common_jump_kernel_dimension),
        )
        jump_kernel.add_row(
            "bad-kernel IPRs",
            _format_float_tuple(self.bad_common_jump_kernel_iprs),
        )

        liouvillian = Table(title="Liouvillian zero modes")
        liouvillian.add_column("quantity", style="bold")
        liouvillian.add_column("value", justify="right")

        liouvillian.add_row(
            "zero-mode count",
            (
                "not checked"
                if self.liouvillian_zero_mode_count is None
                else str(self.liouvillian_zero_mode_count)
            ),
        )
        liouvillian.add_row(
            "spectral gap",
            _format_float_or_none(self.liouvillian_spectral_gap),
        )

        return Panel(
            Group(overview, target, jump_kernel, liouvillian),
            title=Text("Dark-subspace diagnostics", style="bold cyan"),
            border_style="cyan",
        )


def diagnose_dark_subspace(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    target_state: npt.ArrayLike,
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
    kernel_tolerance: float = 1e-10,
    liouvillian_zero_tolerance: float = 1e-9,
    check_liouvillian_spectrum: bool = True,
    max_liouvillian_dense_dimension: int = 4096,
) -> DarkSubspaceDiagnostics:
    """Diagnose whether a dark target is likely unique/attractive.

    This is intended for small systems. It computes:

        1. target jump residuals ||J_mu psi||;
        2. common jump kernel dim intersection_mu ker J_mu;
        3. bad common-kernel dimension after removing the target direction;
        4. target Liouvillian residual ||L(|psi><psi|)||;
        5. optional Liouvillian zero-mode count.

    The Liouvillian spectrum check is dense and should only be used for small
    Hilbert spaces.
    """
    backend_obj = get_open_system_backend(backend)

    hamiltonian_dense = as_backend_dense_array(
        hamiltonian,
        backend=backend_obj,
        dtype=np.complex128,
    )
    jump_dense = tuple(
        as_backend_dense_array(
            jump,
            backend=backend_obj,
            dtype=np.complex128,
        )
        for jump in jumps
    )

    hamiltonian_numpy = np.asarray(
        backend_obj.to_numpy(hamiltonian_dense),
        dtype=np.complex128,
    )
    jumps_numpy = tuple(
        np.asarray(backend_obj.to_numpy(jump), dtype=np.complex128) for jump in jump_dense
    )

    target = np.asarray(target_state, dtype=np.complex128)
    if target.ndim != 1:
        raise ValueError("target_state must be one-dimensional.")

    target_norm = float(np.linalg.norm(target))
    if target_norm == 0.0:
        raise ValueError("target_state must be nonzero.")

    target = target / target_norm
    dim = int(target.size)

    if hamiltonian_numpy.shape != (dim, dim):
        raise ValueError("hamiltonian shape must be compatible with target_state.")

    for jump in jumps_numpy:
        if jump.shape != (dim, dim):
            raise ValueError(
                "Every jump operator must have shape " "(len(target_state), len(target_state))."
            )

    target_jump_residuals = tuple(float(np.linalg.norm(jump @ target)) for jump in jumps_numpy)
    max_target_jump_residual = max(target_jump_residuals) if target_jump_residuals else 0.0

    common_kernel_basis = _common_jump_kernel_basis(
        jumps=jumps_numpy,
        dim=dim,
        tolerance=kernel_tolerance,
    )

    common_jump_kernel_dimension = int(common_kernel_basis.shape[1])

    target_projection_onto_common_kernel = _projection_norm_onto_basis(
        vector=target,
        basis=common_kernel_basis,
    )
    target_distance_from_common_kernel = float(
        np.sqrt(
            max(
                0.0,
                1.0 - target_projection_onto_common_kernel**2,
            )
        )
    )
    target_in_common_jump_kernel = (
        target_distance_from_common_kernel <= np.sqrt(kernel_tolerance)
        or max_target_jump_residual <= kernel_tolerance
    )

    bad_common_kernel_basis = _kernel_basis_orthogonal_to_target(
        basis=common_kernel_basis,
        target=target,
        tolerance=kernel_tolerance,
    )
    bad_common_jump_kernel_dimension = int(bad_common_kernel_basis.shape[1])
    bad_common_jump_kernel_iprs = tuple(
        _state_ipr(bad_common_kernel_basis[:, index])
        for index in range(bad_common_kernel_basis.shape[1])
    )

    target_density_matrix = np.outer(target, target.conj())
    target_rhs = lindblad_rhs_density_matrix(
        target_density_matrix,
        hamiltonian=hamiltonian_numpy,
        jumps=list(jumps_numpy),
        backend="scipy",
    )
    target_liouvillian_residual = float(np.linalg.norm(target_rhs))

    liouvillian_zero_mode_count: int | None = None
    liouvillian_spectral_gap: float | None = None
    liouvillian_eigenvalues: tuple[complex, ...] = ()

    if check_liouvillian_spectrum:
        liouvillian_dimension = dim * dim

        if liouvillian_dimension > max_liouvillian_dense_dimension:
            raise ValueError(
                "Dense Liouvillian spectrum check is too expensive: "
                f"dim^2={liouvillian_dimension}, "
                f"max_liouvillian_dense_dimension="
                f"{max_liouvillian_dense_dimension}. "
                "Set check_liouvillian_spectrum=False or increase the limit."
            )

        liouvillian = build_liouvillian(
            hamiltonian_numpy,
            list(jumps_numpy),
            backend="scipy",
            sparse_format="csr",
        )
        liouvillian_dense = liouvillian.toarray()
        eigenvalues = scipy_linalg.eigvals(liouvillian_dense)

        eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
        eigenvalue_abs = np.abs(eigenvalues)

        liouvillian_zero_mode_count = int(
            np.count_nonzero(eigenvalue_abs <= liouvillian_zero_tolerance)
        )

        nonzero_abs = eigenvalue_abs[eigenvalue_abs > liouvillian_zero_tolerance]
        if nonzero_abs.size == 0:
            liouvillian_spectral_gap = None
        else:
            liouvillian_spectral_gap = float(np.min(nonzero_abs))

        # Store the smallest few eigenvalues for inspection.
        order = np.argsort(eigenvalue_abs)
        liouvillian_eigenvalues = tuple(
            complex(eigenvalues[index]) for index in order[: min(16, eigenvalues.size)]
        )

    likely_unique_dark_state: bool | None
    if liouvillian_zero_mode_count is None:
        likely_unique_dark_state = None
    else:
        likely_unique_dark_state = (
            liouvillian_zero_mode_count == 1
            and target_liouvillian_residual <= liouvillian_zero_tolerance
        )

    return DarkSubspaceDiagnostics(
        dim=dim,
        n_jumps=len(jumps_numpy),
        target_norm=target_norm,
        target_jump_residuals=target_jump_residuals,
        max_target_jump_residual=max_target_jump_residual,
        target_liouvillian_residual=target_liouvillian_residual,
        common_jump_kernel_dimension=common_jump_kernel_dimension,
        target_projection_onto_common_kernel=target_projection_onto_common_kernel,
        target_distance_from_common_kernel=target_distance_from_common_kernel,
        target_in_common_jump_kernel=target_in_common_jump_kernel,
        bad_common_jump_kernel_dimension=bad_common_jump_kernel_dimension,
        bad_common_jump_kernel_iprs=bad_common_jump_kernel_iprs,
        liouvillian_zero_mode_count=liouvillian_zero_mode_count,
        liouvillian_spectral_gap=liouvillian_spectral_gap,
        liouvillian_eigenvalues=liouvillian_eigenvalues,
        likely_unique_dark_state=likely_unique_dark_state,
    )


@dataclass(frozen=True, slots=True)
class AbsorbingProjectorJumpDiagnostics:
    """Diagnostics for one jump relative to a target projector."""

    jump_index: int
    target_residual: float
    outflow_norm: float
    inflow_norm: float
    commutator_norm: float
    dissipator_adjoint_projector_norm: float

    @property
    def is_dark_on_target(self) -> bool:
        return self.target_residual < 1e-10

    @property
    def has_inflow(self) -> bool:
        return self.inflow_norm > 1e-10


@dataclass(frozen=True, slots=True)
class AbsorbingProjectorSymmetryDiagnostics:
    """Diagnostics for the absorbing-state projector symmetry P_psi."""

    dim: int
    n_jumps: int
    hamiltonian_commutator_norm: float
    liouvillian_adjoint_projector_norm: float
    max_target_residual: float
    max_outflow_norm: float
    max_inflow_norm: float
    max_jump_projector_commutator_norm: float
    jump_diagnostics: tuple[AbsorbingProjectorJumpDiagnostics, ...]

    absorbing_projector_is_conserved: bool
    target_is_dark: bool
    has_recycling_inflow: bool
    has_absorbing_projector_symmetry: bool

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "dim": self.dim,
            "n_jumps": self.n_jumps,
            "hamiltonian_commutator_norm": self.hamiltonian_commutator_norm,
            "liouvillian_adjoint_projector_norm": (self.liouvillian_adjoint_projector_norm),
            "max_target_residual": self.max_target_residual,
            "max_outflow_norm": self.max_outflow_norm,
            "max_inflow_norm": self.max_inflow_norm,
            "max_jump_projector_commutator_norm": (self.max_jump_projector_commutator_norm),
            "absorbing_projector_is_conserved": (self.absorbing_projector_is_conserved),
            "target_is_dark": self.target_is_dark,
            "has_recycling_inflow": self.has_recycling_inflow,
            "has_absorbing_projector_symmetry": (self.has_absorbing_projector_symmetry),
            "jump_diagnostics": tuple(
                {
                    "jump_index": diagnostic.jump_index,
                    "target_residual": diagnostic.target_residual,
                    "outflow_norm": diagnostic.outflow_norm,
                    "inflow_norm": diagnostic.inflow_norm,
                    "commutator_norm": diagnostic.commutator_norm,
                    "dissipator_adjoint_projector_norm": (
                        diagnostic.dissipator_adjoint_projector_norm
                    ),
                }
                for diagnostic in self.jump_diagnostics
            ),
        }

    def to_rich(self):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "AbsorbingProjectorSymmetryDiagnostics.to_rich() "
                "requires rich. Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()

        overview.add_row("Hilbert dimension", str(self.dim))
        overview.add_row("number of jumps", str(self.n_jumps))
        overview.add_row(
            "target is dark",
            str(self.target_is_dark),
        )
        overview.add_row(
            "has recycling inflow",
            str(self.has_recycling_inflow),
        )
        overview.add_row(
            "P_psi conserved",
            str(self.absorbing_projector_is_conserved),
        )
        overview.add_row(
            "absorbing-projector symmetry",
            str(self.has_absorbing_projector_symmetry),
        )

        global_table = Table(title="Global projector diagnostics")
        global_table.add_column("quantity", style="bold")
        global_table.add_column("value", justify="right")

        global_table.add_row(
            "||[H, P_psi]||",
            _format_float(self.hamiltonian_commutator_norm),
        )
        global_table.add_row(
            "||L†(P_psi)||",
            _format_float(self.liouvillian_adjoint_projector_norm),
        )
        global_table.add_row(
            "max ||J psi||",
            _format_float(self.max_target_residual),
        )
        global_table.add_row(
            "max ||(I-P) J P||",
            _format_float(self.max_outflow_norm),
        )
        global_table.add_row(
            "max ||P J (I-P)||",
            _format_float(self.max_inflow_norm),
        )
        global_table.add_row(
            "max ||[J, P]||",
            _format_float(self.max_jump_projector_commutator_norm),
        )

        jumps = Table(title="Jump-by-jump projector diagnostics")
        jumps.add_column("jump", justify="right")
        jumps.add_column("||J psi||", justify="right")
        jumps.add_column("outflow", justify="right")
        jumps.add_column("inflow", justify="right")
        jumps.add_column("||[J,P]||", justify="right")
        jumps.add_column("||D†_J(P)||", justify="right")

        for diagnostic in self.jump_diagnostics:
            jumps.add_row(
                str(diagnostic.jump_index),
                _format_float(diagnostic.target_residual),
                _format_float(diagnostic.outflow_norm),
                _format_float(diagnostic.inflow_norm),
                _format_float(diagnostic.commutator_norm),
                _format_float(diagnostic.dissipator_adjoint_projector_norm),
            )

        return Panel(
            Group(overview, global_table, jumps),
            title=Text(
                "Absorbing-projector symmetry diagnostics",
                style="bold cyan",
            ),
            border_style="cyan",
        )


def diagnose_absorbing_projector_symmetry(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    target_state: npt.ArrayLike,
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
    tolerance: float = 1e-10,
) -> AbsorbingProjectorSymmetryDiagnostics:
    """Diagnose whether P_psi is an absorbing-state projector symmetry.

    The target projector is

        P_psi = |psi><psi|.

    The relevant obstruction to attraction is:

        J_mu |psi> = 0
        and
        P_psi J_mu (I - P_psi) = 0

    for all jumps. Then the target is dark, but there is no jump-induced
    inflow from psi_perp into psi. Equivalently, P_psi is conserved by the
    Heisenberg-picture Lindbladian.
    """
    backend_obj = get_open_system_backend(backend)

    hamiltonian_dense = as_backend_dense_array(
        hamiltonian,
        backend=backend_obj,
        dtype=np.complex128,
    )
    jump_dense = tuple(
        as_backend_dense_array(
            jump,
            backend=backend_obj,
            dtype=np.complex128,
        )
        for jump in jumps
    )

    hamiltonian_numpy = np.asarray(
        backend_obj.to_numpy(hamiltonian_dense),
        dtype=np.complex128,
    )
    jumps_numpy = tuple(
        np.asarray(backend_obj.to_numpy(jump), dtype=np.complex128) for jump in jump_dense
    )

    target = np.asarray(target_state, dtype=np.complex128)
    if target.ndim != 1:
        raise ValueError("target_state must be one-dimensional.")

    target_norm = float(np.linalg.norm(target))
    if target_norm == 0.0:
        raise ValueError("target_state must be nonzero.")

    target = target / target_norm
    dim = int(target.size)

    if hamiltonian_numpy.shape != (dim, dim):
        raise ValueError("hamiltonian shape must be compatible with target_state.")

    for jump in jumps_numpy:
        if jump.shape != (dim, dim):
            raise ValueError(
                "Every jump operator must have shape " "(len(target_state), len(target_state))."
            )

    projector_target = np.outer(target, target.conj())
    identity = np.eye(dim, dtype=np.complex128)
    projector_orthogonal = identity - projector_target

    hamiltonian_commutator = (
        hamiltonian_numpy @ projector_target - projector_target @ hamiltonian_numpy
    )
    hamiltonian_commutator_norm = float(np.linalg.norm(hamiltonian_commutator))

    jump_diagnostics: list[AbsorbingProjectorJumpDiagnostics] = []

    liouvillian_adjoint_projector = 1j * (
        hamiltonian_numpy @ projector_target - projector_target @ hamiltonian_numpy
    )

    for jump_index, jump in enumerate(jumps_numpy):
        target_residual = float(np.linalg.norm(jump @ target))

        outflow = projector_orthogonal @ jump @ projector_target
        inflow = projector_target @ jump @ projector_orthogonal
        commutator = jump @ projector_target - projector_target @ jump

        jump_dagger = jump.conj().T
        jump_dagger_jump = jump_dagger @ jump

        dissipator_adjoint_projector = jump_dagger @ projector_target @ jump - 0.5 * (
            jump_dagger_jump @ projector_target + projector_target @ jump_dagger_jump
        )

        liouvillian_adjoint_projector = liouvillian_adjoint_projector + dissipator_adjoint_projector

        jump_diagnostics.append(
            AbsorbingProjectorJumpDiagnostics(
                jump_index=jump_index,
                target_residual=target_residual,
                outflow_norm=float(np.linalg.norm(outflow)),
                inflow_norm=float(np.linalg.norm(inflow)),
                commutator_norm=float(np.linalg.norm(commutator)),
                dissipator_adjoint_projector_norm=float(
                    np.linalg.norm(dissipator_adjoint_projector)
                ),
            )
        )

    max_target_residual = max(
        (diagnostic.target_residual for diagnostic in jump_diagnostics),
        default=0.0,
    )
    max_outflow_norm = max(
        (diagnostic.outflow_norm for diagnostic in jump_diagnostics),
        default=0.0,
    )
    max_inflow_norm = max(
        (diagnostic.inflow_norm for diagnostic in jump_diagnostics),
        default=0.0,
    )
    max_jump_projector_commutator_norm = max(
        (diagnostic.commutator_norm for diagnostic in jump_diagnostics),
        default=0.0,
    )

    liouvillian_adjoint_projector_norm = float(np.linalg.norm(liouvillian_adjoint_projector))

    target_is_dark = max_target_residual <= tolerance
    has_recycling_inflow = max_inflow_norm > tolerance
    absorbing_projector_is_conserved = liouvillian_adjoint_projector_norm <= tolerance

    has_absorbing_projector_symmetry = (
        target_is_dark and not has_recycling_inflow and absorbing_projector_is_conserved
    )

    return AbsorbingProjectorSymmetryDiagnostics(
        dim=dim,
        n_jumps=len(jumps_numpy),
        hamiltonian_commutator_norm=hamiltonian_commutator_norm,
        liouvillian_adjoint_projector_norm=liouvillian_adjoint_projector_norm,
        max_target_residual=max_target_residual,
        max_outflow_norm=max_outflow_norm,
        max_inflow_norm=max_inflow_norm,
        max_jump_projector_commutator_norm=(max_jump_projector_commutator_norm),
        jump_diagnostics=tuple(jump_diagnostics),
        absorbing_projector_is_conserved=absorbing_projector_is_conserved,
        target_is_dark=target_is_dark,
        has_recycling_inflow=has_recycling_inflow,
        has_absorbing_projector_symmetry=(has_absorbing_projector_symmetry),
    )


def _common_jump_kernel_basis(
    *,
    jumps: tuple[np.ndarray, ...],
    dim: int,
    tolerance: float,
) -> np.ndarray:
    if len(jumps) == 0:
        return np.eye(dim, dtype=np.complex128)

    stacked = np.vstack(jumps)
    return _nullspace_basis(stacked, tolerance=tolerance)


def _nullspace_basis(
    matrix: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    if matrix.size == 0:
        return np.eye(matrix.shape[1], dtype=np.complex128)

    _left_vectors, singular_values, right_vectors_dagger = np.linalg.svd(
        matrix,
        full_matrices=True,
    )

    n_columns = matrix.shape[1]
    rank = int(np.count_nonzero(singular_values > tolerance))

    if rank >= n_columns:
        return np.zeros((n_columns, 0), dtype=np.complex128)

    return (
        right_vectors_dagger.conj()
        .T[:, rank:]
        .astype(
            np.complex128,
            copy=False,
        )
    )


def _projection_norm_onto_basis(
    *,
    vector: np.ndarray,
    basis: np.ndarray,
) -> float:
    if basis.shape[1] == 0:
        return 0.0

    coefficients = basis.conj().T @ vector
    return float(np.linalg.norm(coefficients))


def _kernel_basis_orthogonal_to_target(
    *,
    basis: np.ndarray,
    target: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.zeros((target.size, 0), dtype=np.complex128)

    target = target / np.linalg.norm(target)

    projected = basis - np.outer(target, target.conj() @ basis)

    # Remove numerically zero columns before QR/SVD.
    column_norms = np.linalg.norm(projected, axis=0)
    keep = column_norms > tolerance

    if not np.any(keep):
        return np.zeros((target.size, 0), dtype=np.complex128)

    projected = projected[:, keep]

    return _orthonormal_column_basis(
        projected,
        tolerance=tolerance,
    )


def _orthonormal_column_basis(
    matrix: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.complex128)

    left_vectors, singular_values, _right_vectors_dagger = np.linalg.svd(
        matrix,
        full_matrices=False,
    )

    rank = int(np.count_nonzero(singular_values > tolerance))

    if rank == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.complex128)

    return left_vectors[:, :rank].astype(np.complex128, copy=False)


def _state_ipr(state: np.ndarray) -> float:
    norm = float(np.linalg.norm(state))

    if norm == 0.0:
        return 0.0

    normalized = state / norm
    probabilities = np.abs(normalized) ** 2
    return float(np.sum(probabilities**2))


def _format_float(value: float) -> str:
    return f"{value:.3e}"


def _format_float_or_none(value: float | None) -> str:
    if value is None:
        return "not checked"

    return _format_float(float(value))


def _format_float_tuple(
    values: tuple[float, ...],
    *,
    max_items: int = 8,
) -> str:
    if len(values) == 0:
        return "∅"

    if len(values) <= max_items:
        return ", ".join(_format_float(value) for value in values)

    head = ", ".join(_format_float(value) for value in values[:max_items])
    return f"{head}, ... ({len(values)} total)"


def _status_for_residual(
    value: float | None,
    *,
    excellent: float = 1e-12,
    acceptable: float = 1e-8,
) -> str:
    if value is None:
        return "[dim]n/a[/dim]"

    if value <= excellent:
        return "[green]ok[/green]"

    if value <= acceptable:
        return "[yellow]warn[/yellow]"

    return "[red]large[/red]"
