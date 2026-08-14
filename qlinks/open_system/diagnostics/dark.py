from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import scipy.linalg as scipy_linalg
import scipy.sparse as scipy_sparse
import scipy.sparse.linalg as scipy_sparse_linalg

from qlinks.open_system._subspace import (
    _as_scipy_csr_matrix,
    _common_kernel_basis_from_sparse_operators,
    _kernel_basis_orthogonal_to_manifold,
    _kernel_basis_orthogonal_to_target,
    _orthonormal_column_basis,
    _orthonormal_target_state_matrix,
    _projection_norm_onto_basis,
    _subspace_projection_and_distance,
)
from qlinks.open_system.backend import OpenSystemBackend, OpenSystemBackendName
from qlinks.open_system.diagnostics._formatting import (
    _format_float,
    _format_float_or_none,
    _format_float_tuple,
    _format_optional_int,
    _state_ipr,
    _status_for_residual,
)
from qlinks.open_system.diagnostics._linalg import (
    _external_decay_gap_from_spectrum,
    _internal_liouvillian_eigenvalues,
    _largest_h_invariant_subspace_inside_leakage_kernel,
    _manifold_inflow_norm,
    _match_expected_internal_nondecaying_modes,
    _rank_one_lindblad_rhs_norm,
)
from qlinks.open_system.operators import build_liouvillian, lindblad_rhs_density_matrix


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
    liouvillian_zero_mode_count_is_lower_bound: bool
    liouvillian_spectral_gap: float | None
    liouvillian_decay_gap: float | None
    liouvillian_peripheral_mode_count: int | None
    liouvillian_spectrum_method: str
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
            "liouvillian_zero_mode_count_is_lower_bound": (
                self.liouvillian_zero_mode_count_is_lower_bound
            ),
            "liouvillian_spectral_gap": self.liouvillian_spectral_gap,
            "liouvillian_decay_gap": self.liouvillian_decay_gap,
            "liouvillian_peripheral_mode_count": self.liouvillian_peripheral_mode_count,
            "liouvillian_spectrum_method": self.liouvillian_spectrum_method,
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
                else (
                    str(self.liouvillian_zero_mode_count)
                    + ("+" if self.liouvillian_zero_mode_count_is_lower_bound else "")
                )
            ),
        )
        liouvillian.add_row(
            "spectrum method",
            self.liouvillian_spectrum_method,
        )
        liouvillian.add_row(
            "absolute spectral gap",
            _format_float_or_none(self.liouvillian_spectral_gap),
        )
        liouvillian.add_row(
            "decay gap",
            _format_float_or_none(self.liouvillian_decay_gap),
        )
        liouvillian.add_row(
            "peripheral mode count",
            (
                "not checked"
                if self.liouvillian_peripheral_mode_count is None
                else str(self.liouvillian_peripheral_mode_count)
            ),
        )

        return Panel(
            Group(overview, target, jump_kernel, liouvillian),
            title=Text("Dark-subspace diagnostics", style="bold cyan"),
            border_style="cyan",
        )


@dataclass(frozen=True, slots=True)
class DarkManifoldDiagnostics:
    """Diagnostics for an attractive dark manifold/DFS target.

    The target is a column-orthonormal basis ``M`` and the target projector is
    ``P_M = M M†``.  Unlike :class:`DarkSubspaceDiagnostics`, this report does
    not expect a unique pure steady state.  Internal zero or imaginary-axis
    Liouvillian modes generated by the projected Hamiltonian on the target
    manifold are treated as expected modes; only additional non-decaying modes
    outside the target manifold are flagged as obstructions.
    """

    dim: int
    n_jumps: int
    manifold_dimension: int

    hamiltonian_closure_residual: float
    target_jump_residuals: tuple[float, ...]
    max_target_jump_residual: float
    target_density_liouvillian_residual: float
    inflow_norm: float

    common_jump_kernel_dimension: int
    target_projection_onto_common_kernel: float
    target_distance_from_common_kernel: float
    target_in_common_jump_kernel: bool
    bad_common_jump_kernel_dimension: int
    bad_common_jump_kernel_iprs: tuple[float, ...]

    internal_hamiltonian_eigenvalues: tuple[complex, ...]
    expected_internal_liouvillian_eigenvalues: tuple[complex, ...]
    expected_internal_zero_mode_count: int
    expected_internal_peripheral_mode_count: int

    liouvillian_zero_mode_count: int | None
    liouvillian_zero_mode_count_is_lower_bound: bool
    liouvillian_spectral_gap: float | None
    liouvillian_decay_gap: float | None
    liouvillian_peripheral_mode_count: int | None
    liouvillian_spectrum_method: str
    liouvillian_eigenvalues: tuple[complex, ...]

    matched_internal_nondecaying_mode_count: int | None
    missing_internal_nondecaying_mode_count: int | None
    extra_nondecaying_mode_count: int | None
    extra_zero_mode_count: int | None
    external_decay_gap: float | None

    likely_attractive_dark_manifold: bool | None

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "dim": self.dim,
            "n_jumps": self.n_jumps,
            "manifold_dimension": self.manifold_dimension,
            "h_closure_residual": self.hamiltonian_closure_residual,
            "max_target_jump_residual": self.max_target_jump_residual,
            "target_density_liouvillian_residual": self.target_density_liouvillian_residual,
            "inflow_norm": self.inflow_norm,
            "common_jump_kernel_dimension": self.common_jump_kernel_dimension,
            "target_projection_onto_common_kernel": self.target_projection_onto_common_kernel,
            "target_distance_from_common_kernel": self.target_distance_from_common_kernel,
            "target_in_common_jump_kernel": self.target_in_common_jump_kernel,
            "bad_common_jump_kernel_dimension": self.bad_common_jump_kernel_dimension,
            "bad_common_jump_kernel_iprs": self.bad_common_jump_kernel_iprs,
            "internal_hamiltonian_eigenvalues": [
                complex(value) for value in self.internal_hamiltonian_eigenvalues
            ],
            "expected_internal_zero_mode_count": self.expected_internal_zero_mode_count,
            "expected_internal_peripheral_mode_count": (
                self.expected_internal_peripheral_mode_count
            ),
            "liouvillian_zero_mode_count": self.liouvillian_zero_mode_count,
            "liouvillian_zero_mode_count_is_lower_bound": (
                self.liouvillian_zero_mode_count_is_lower_bound
            ),
            "liouvillian_spectral_gap": self.liouvillian_spectral_gap,
            "liouvillian_decay_gap": self.liouvillian_decay_gap,
            "liouvillian_peripheral_mode_count": self.liouvillian_peripheral_mode_count,
            "liouvillian_spectrum_method": self.liouvillian_spectrum_method,
            "matched_internal_nondecaying_mode_count": (
                self.matched_internal_nondecaying_mode_count
            ),
            "missing_internal_nondecaying_mode_count": (
                self.missing_internal_nondecaying_mode_count
            ),
            "extra_nondecaying_mode_count": self.extra_nondecaying_mode_count,
            "extra_zero_mode_count": self.extra_zero_mode_count,
            "external_decay_gap": self.external_decay_gap,
            "likely_attractive_dark_manifold": self.likely_attractive_dark_manifold,
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "DarkManifoldDiagnostics.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.dim))
        overview.add_row("number of jumps", str(self.n_jumps))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row(
            "likely attractive dark manifold",
            str(self.likely_attractive_dark_manifold),
        )

        target = Table(title="Target manifold checks")
        target.add_column("quantity", style="bold")
        target.add_column("value", justify="right")
        target.add_column("status", justify="center")
        target.add_row(
            "||(I-P_M) H P_M||",
            _format_float(self.hamiltonian_closure_residual),
            _status_for_residual(self.hamiltonian_closure_residual),
        )
        target.add_row(
            "max ||J_mu P_M||",
            _format_float(self.max_target_jump_residual),
            _status_for_residual(self.max_target_jump_residual),
        )
        target.add_row(
            "||L(P_M/m)||",
            _format_float(self.target_density_liouvillian_residual),
            _status_for_residual(self.target_density_liouvillian_residual),
        )
        target.add_row(
            "inflow ||P_M J Q_M||",
            _format_float(self.inflow_norm),
            "[green]yes[/green]" if self.inflow_norm > 1e-12 else "[yellow]none[/yellow]",
        )

        jump_kernel = Table(title="Common jump kernel")
        jump_kernel.add_column("quantity", style="bold")
        jump_kernel.add_column("value", justify="right")
        jump_kernel.add_row("dim intersection ker J_mu", str(self.common_jump_kernel_dimension))
        jump_kernel.add_row(
            "target projection onto kernel",
            _format_float(self.target_projection_onto_common_kernel),
        )
        jump_kernel.add_row(
            "target distance from kernel",
            _format_float(self.target_distance_from_common_kernel),
        )
        jump_kernel.add_row("target in kernel", str(self.target_in_common_jump_kernel))
        jump_kernel.add_row(
            "bad complement kernel dim",
            str(self.bad_common_jump_kernel_dimension),
        )
        jump_kernel.add_row(
            "bad-kernel IPRs",
            _format_float_tuple(self.bad_common_jump_kernel_iprs),
        )

        internal = Table(title="Internal non-decaying modes")
        internal.add_column("quantity", style="bold")
        internal.add_column("value", justify="right")
        internal.add_row(
            "expected zero modes",
            str(self.expected_internal_zero_mode_count),
        )
        internal.add_row(
            "expected peripheral modes",
            str(self.expected_internal_peripheral_mode_count),
        )
        internal.add_row(
            "matched internal modes",
            _format_optional_int(self.matched_internal_nondecaying_mode_count),
        )
        internal.add_row(
            "missing internal modes",
            _format_optional_int(self.missing_internal_nondecaying_mode_count),
        )

        liouvillian = Table(title="Liouvillian spectrum")
        liouvillian.add_column("quantity", style="bold")
        liouvillian.add_column("value", justify="right")
        liouvillian.add_row("spectrum method", self.liouvillian_spectrum_method)
        liouvillian.add_row(
            "zero-mode count",
            _format_optional_int(
                self.liouvillian_zero_mode_count,
                lower_bound=self.liouvillian_zero_mode_count_is_lower_bound,
            ),
        )
        liouvillian.add_row(
            "peripheral mode count",
            _format_optional_int(self.liouvillian_peripheral_mode_count),
        )
        liouvillian.add_row(
            "extra non-decaying modes",
            _format_optional_int(self.extra_nondecaying_mode_count),
        )
        liouvillian.add_row("extra zero modes", _format_optional_int(self.extra_zero_mode_count))
        liouvillian.add_row(
            "absolute spectral gap",
            _format_float_or_none(self.liouvillian_spectral_gap),
        )
        liouvillian.add_row("decay gap", _format_float_or_none(self.liouvillian_decay_gap))
        liouvillian.add_row("external decay gap", _format_float_or_none(self.external_decay_gap))

        return Panel(
            Group(overview, target, jump_kernel, internal, liouvillian),
            title=Text("Dark-manifold diagnostics", style="bold cyan"),
            border_style="cyan",
        )


@dataclass(frozen=True, slots=True)
class CommonKernelHamiltonianInvariantSectorReport:
    """Cheap obstruction diagnostic inside the common jump kernel.

    The common jump-kernel condition ``cap_mu ker J_mu = M`` is sufficient but
    stronger than necessary.  A complement vector in the common kernel is only a
    Hamiltonian-stable dark obstruction if its whole Krylov orbit under ``H``
    remains inside the common jump kernel.  This report computes the largest
    such subspace inside ``(cap_mu ker J_mu) cap M^perp`` using a small dense
    nullspace problem.
    """

    dim: int
    n_jumps: int
    manifold_dimension: int
    common_jump_kernel_dimension: int
    bad_common_jump_kernel_dimension: int
    bad_h_invariant_kernel_dimension: int
    h_leakage_norm_from_bad_kernel: float
    h_leakage_norm_from_invariant_kernel: float
    h_target_coupling_norm_from_bad_kernel: float
    h_bad_block_norm: float
    h_invariant_block_eigenvalues: tuple[complex, ...]
    bad_h_invariant_kernel_iprs: tuple[float, ...]
    target_in_common_jump_kernel: bool
    kernel_tolerance: float

    @property
    def has_bad_common_kernel(self) -> bool:
        return self.bad_common_jump_kernel_dimension > 0

    @property
    def has_bad_h_invariant_kernel(self) -> bool:
        return self.bad_h_invariant_kernel_dimension > 0

    @property
    def likely_attractive_by_h_invariant_kernel(self) -> bool:
        return self.target_in_common_jump_kernel and not self.has_bad_h_invariant_kernel

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "dim": self.dim,
            "n_jumps": self.n_jumps,
            "manifold_dimension": self.manifold_dimension,
            "common_jump_kernel_dimension": self.common_jump_kernel_dimension,
            "bad_common_jump_kernel_dimension": self.bad_common_jump_kernel_dimension,
            "bad_h_invariant_kernel_dimension": self.bad_h_invariant_kernel_dimension,
            "h_leakage_norm_from_bad_kernel": self.h_leakage_norm_from_bad_kernel,
            "h_leakage_norm_from_invariant_kernel": (self.h_leakage_norm_from_invariant_kernel),
            "h_target_coupling_norm_from_bad_kernel": (self.h_target_coupling_norm_from_bad_kernel),
            "h_bad_block_norm": self.h_bad_block_norm,
            "h_invariant_block_eigenvalues": tuple(
                complex(value) for value in self.h_invariant_block_eigenvalues
            ),
            "bad_h_invariant_kernel_iprs": self.bad_h_invariant_kernel_iprs,
            "target_in_common_jump_kernel": self.target_in_common_jump_kernel,
            "kernel_tolerance": self.kernel_tolerance,
            "has_bad_common_kernel": self.has_bad_common_kernel,
            "has_bad_h_invariant_kernel": self.has_bad_h_invariant_kernel,
            "likely_attractive_by_h_invariant_kernel": (
                self.likely_attractive_by_h_invariant_kernel
            ),
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "CommonKernelHamiltonianInvariantSectorReport.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.dim))
        overview.add_row("number of jumps", str(self.n_jumps))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row("common kernel dimension", str(self.common_jump_kernel_dimension))
        overview.add_row("bad common kernel dimension", str(self.bad_common_jump_kernel_dimension))
        overview.add_row(
            "bad H-invariant kernel dimension",
            str(self.bad_h_invariant_kernel_dimension),
        )
        overview.add_row(
            "likely attractive by H-invariant kernel",
            str(self.likely_attractive_by_h_invariant_kernel),
        )

        leakage = Table(title="Hamiltonian leakage")
        leakage.add_column("quantity", style="bold")
        leakage.add_column("value", justify="right")
        leakage.add_row(
            "||(I-P_K) H B_bad||",
            _format_float(self.h_leakage_norm_from_bad_kernel),
        )
        leakage.add_row(
            "||(I-P_K) H B_inv||",
            _format_float(self.h_leakage_norm_from_invariant_kernel),
        )
        leakage.add_row(
            "||P_M H B_bad||",
            _format_float(self.h_target_coupling_norm_from_bad_kernel),
        )
        leakage.add_row("||B_bad† H B_bad||", _format_float(self.h_bad_block_norm))
        leakage.add_row(
            "bad invariant IPRs",
            _format_float_tuple(self.bad_h_invariant_kernel_iprs),
        )

        return Panel(
            Group(overview, leakage),
            title=Text("Common-kernel H-invariant sector", style="bold cyan"),
            border_style=("green" if self.likely_attractive_by_h_invariant_kernel else "yellow"),
        )


def bad_h_invariant_common_kernel_basis(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    target_states: npt.ArrayLike,
    kernel_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return the bad H-invariant sector inside the common jump kernel.

    The returned columns form an orthonormal basis for the largest subspace of
    ``(cap_mu ker J_mu) cap M^perp`` whose Hamiltonian orbit stays inside the
    common jump kernel.  An empty ``(dim, 0)`` array means the selected jumps
    have no Hamiltonian-stable dark obstruction outside the target manifold.

    This helper exposes the obstruction basis used internally by
    :func:`diagnose_common_kernel_h_invariant_sector`, so jump-design routines
    can add a targeted completion stage without recomputing or interpreting a
    Liouvillian spectrum.
    """
    hamiltonian_sparse = _as_scipy_csr_matrix(hamiltonian)
    dim = int(hamiltonian_sparse.shape[0])
    if hamiltonian_sparse.shape != (dim, dim):
        raise ValueError("hamiltonian must be a square matrix.")

    manifold_basis = _orthonormal_target_state_matrix(
        target_states,
        dim=dim,
        tolerance=kernel_tolerance,
    )

    jumps_sparse = tuple(_as_scipy_csr_matrix(jump) for jump in jumps)
    for jump in jumps_sparse:
        if jump.shape != (dim, dim):
            raise ValueError("Every jump operator must have shape (dim, dim).")

    common_kernel_basis = _common_kernel_basis_from_sparse_operators(
        operators=jumps_sparse,
        dim=dim,
        tolerance=kernel_tolerance,
    )
    bad_basis = _kernel_basis_orthogonal_to_manifold(
        basis=common_kernel_basis,
        manifold_basis=manifold_basis,
        tolerance=kernel_tolerance,
    )
    bad_dimension = int(bad_basis.shape[1])
    if bad_dimension == 0:
        return np.zeros((dim, 0), dtype=np.complex128)

    h_bad = np.asarray(hamiltonian_sparse @ bad_basis, dtype=np.complex128)
    if common_kernel_basis.shape[1] == 0:
        projected_to_common = np.zeros_like(h_bad)
    else:
        projected_to_common = common_kernel_basis @ (common_kernel_basis.conj().T @ h_bad)
    leakage = h_bad - projected_to_common

    bad_block = bad_basis.conj().T @ h_bad
    bad_block = 0.5 * (bad_block + bad_block.conj().T)

    invariant_coefficients = _largest_h_invariant_subspace_inside_leakage_kernel(
        leakage=leakage,
        bad_block=bad_block,
        tolerance=kernel_tolerance,
    )

    invariant_basis = bad_basis @ invariant_coefficients
    return _orthonormal_column_basis(invariant_basis, tolerance=kernel_tolerance)


def diagnose_common_kernel_h_invariant_sector(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    target_states: npt.ArrayLike,
    kernel_tolerance: float = 1.0e-10,
) -> CommonKernelHamiltonianInvariantSectorReport:
    """Diagnose Hamiltonian-stable obstructions inside the common jump kernel.

    This is a cheap alternative to a Liouvillian spectrum check.  It first
    computes the common jump kernel ``K = cap_mu ker J_mu`` and the bad
    complement ``B = K cap M^perp``.  It then computes the largest subspace of
    ``B`` whose Hamiltonian Krylov orbit stays inside ``K``.  Only this
    H-invariant part is a purely dark Hamiltonian-stable complement sector.
    """
    hamiltonian_sparse = _as_scipy_csr_matrix(hamiltonian)
    dim = int(hamiltonian_sparse.shape[0])
    if hamiltonian_sparse.shape != (dim, dim):
        raise ValueError("hamiltonian must be a square matrix.")

    manifold_basis = _orthonormal_target_state_matrix(
        target_states,
        dim=dim,
        tolerance=kernel_tolerance,
    )
    manifold_dimension = int(manifold_basis.shape[1])

    jumps_sparse = tuple(_as_scipy_csr_matrix(jump) for jump in jumps)
    for jump in jumps_sparse:
        if jump.shape != (dim, dim):
            raise ValueError("Every jump operator must have shape (dim, dim).")

    common_kernel_basis = _common_kernel_basis_from_sparse_operators(
        operators=jumps_sparse,
        dim=dim,
        tolerance=kernel_tolerance,
    )
    common_jump_kernel_dimension = int(common_kernel_basis.shape[1])

    target_projection_onto_common_kernel, target_distance_from_common_kernel = (
        _subspace_projection_and_distance(
            subspace_basis=manifold_basis,
            containing_basis=common_kernel_basis,
        )
    )
    max_target_jump_residual = max(
        (float(np.linalg.norm(jump @ manifold_basis)) for jump in jumps_sparse),
        default=0.0,
    )
    target_in_common_jump_kernel = (
        target_distance_from_common_kernel <= np.sqrt(kernel_tolerance)
        or max_target_jump_residual <= kernel_tolerance
    )

    bad_basis = _kernel_basis_orthogonal_to_manifold(
        basis=common_kernel_basis,
        manifold_basis=manifold_basis,
        tolerance=kernel_tolerance,
    )
    bad_dimension = int(bad_basis.shape[1])

    if bad_dimension == 0:
        return CommonKernelHamiltonianInvariantSectorReport(
            dim=dim,
            n_jumps=len(jumps_sparse),
            manifold_dimension=manifold_dimension,
            common_jump_kernel_dimension=common_jump_kernel_dimension,
            bad_common_jump_kernel_dimension=0,
            bad_h_invariant_kernel_dimension=0,
            h_leakage_norm_from_bad_kernel=0.0,
            h_leakage_norm_from_invariant_kernel=0.0,
            h_target_coupling_norm_from_bad_kernel=0.0,
            h_bad_block_norm=0.0,
            h_invariant_block_eigenvalues=(),
            bad_h_invariant_kernel_iprs=(),
            target_in_common_jump_kernel=bool(target_in_common_jump_kernel),
            kernel_tolerance=float(kernel_tolerance),
        )

    h_bad = np.asarray(hamiltonian_sparse @ bad_basis, dtype=np.complex128)
    if common_jump_kernel_dimension == 0:
        projected_to_common = np.zeros_like(h_bad)
    else:
        projected_to_common = common_kernel_basis @ (common_kernel_basis.conj().T @ h_bad)
    leakage = h_bad - projected_to_common
    h_leakage_norm_from_bad_kernel = float(np.linalg.norm(leakage))
    h_target_coupling_norm_from_bad_kernel = float(np.linalg.norm(manifold_basis.conj().T @ h_bad))

    bad_block = bad_basis.conj().T @ h_bad
    bad_block = 0.5 * (bad_block + bad_block.conj().T)
    h_bad_block_norm = float(np.linalg.norm(bad_block))

    invariant_coefficients = _largest_h_invariant_subspace_inside_leakage_kernel(
        leakage=leakage,
        bad_block=bad_block,
        tolerance=kernel_tolerance,
    )

    invariant_basis = bad_basis @ invariant_coefficients
    invariant_basis = _orthonormal_column_basis(invariant_basis, tolerance=kernel_tolerance)
    invariant_dimension = int(invariant_basis.shape[1])

    if invariant_dimension == 0:
        h_leakage_norm_from_invariant_kernel = 0.0
        h_invariant_block_eigenvalues: tuple[complex, ...] = ()
        bad_h_invariant_kernel_iprs: tuple[float, ...] = ()
    else:
        h_invariant = np.asarray(hamiltonian_sparse @ invariant_basis, dtype=np.complex128)
        projected_h_invariant = common_kernel_basis @ (common_kernel_basis.conj().T @ h_invariant)
        h_leakage_norm_from_invariant_kernel = float(
            np.linalg.norm(h_invariant - projected_h_invariant)
        )
        invariant_block = invariant_basis.conj().T @ h_invariant
        invariant_block = 0.5 * (invariant_block + invariant_block.conj().T)
        h_invariant_block_eigenvalues = tuple(
            complex(value) for value in np.linalg.eigvalsh(invariant_block)
        )
        bad_h_invariant_kernel_iprs = tuple(
            _state_ipr(invariant_basis[:, index]) for index in range(invariant_dimension)
        )

    return CommonKernelHamiltonianInvariantSectorReport(
        dim=dim,
        n_jumps=len(jumps_sparse),
        manifold_dimension=manifold_dimension,
        common_jump_kernel_dimension=common_jump_kernel_dimension,
        bad_common_jump_kernel_dimension=bad_dimension,
        bad_h_invariant_kernel_dimension=invariant_dimension,
        h_leakage_norm_from_bad_kernel=h_leakage_norm_from_bad_kernel,
        h_leakage_norm_from_invariant_kernel=h_leakage_norm_from_invariant_kernel,
        h_target_coupling_norm_from_bad_kernel=h_target_coupling_norm_from_bad_kernel,
        h_bad_block_norm=h_bad_block_norm,
        h_invariant_block_eigenvalues=h_invariant_block_eigenvalues,
        bad_h_invariant_kernel_iprs=bad_h_invariant_kernel_iprs,
        target_in_common_jump_kernel=bool(target_in_common_jump_kernel),
        kernel_tolerance=float(kernel_tolerance),
    )


def diagnose_dark_manifold(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    target_states: npt.ArrayLike,
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
    kernel_tolerance: float = 1e-10,
    liouvillian_zero_tolerance: float = 1e-9,
    check_liouvillian_spectrum: bool = True,
    max_liouvillian_dense_dimension: int = 4096,
    liouvillian_spectrum_method: Literal["auto", "dense", "sparse", "none"] = "auto",
    sparse_liouvillian_eigenvalue_count: int = 32,
) -> DarkManifoldDiagnostics:
    """Diagnose whether a target manifold is an attractive dark manifold.

    The columns of ``target_states`` span the target manifold.  They need not be
    orthonormal; this function orthonormalizes them and uses the target
    projector ``P_M``.  The diagnostic accepts the internal non-decaying
    Liouvillian modes generated by the projected Hamiltonian ``M† H M`` and
    reports additional zero/peripheral modes as possible complement obstructions.
    """
    # _backend_obj = get_open_system_backend(backend)

    hamiltonian_sparse = _as_scipy_csr_matrix(hamiltonian)
    dim = int(hamiltonian_sparse.shape[0])
    if hamiltonian_sparse.shape != (dim, dim):
        raise ValueError("hamiltonian must be a square matrix.")

    manifold_basis = _orthonormal_target_state_matrix(
        target_states,
        dim=dim,
        tolerance=kernel_tolerance,
    )
    manifold_dimension = int(manifold_basis.shape[1])
    jumps_sparse = tuple(_as_scipy_csr_matrix(jump) for jump in jumps)
    for jump in jumps_sparse:
        if jump.shape != (dim, dim):
            raise ValueError("Every jump operator must have shape (dim, dim).")

    hamiltonian_action = np.asarray(hamiltonian_sparse @ manifold_basis, dtype=np.complex128)
    internal_hamiltonian = manifold_basis.conj().T @ hamiltonian_action
    projected_hamiltonian_action = manifold_basis @ internal_hamiltonian
    hamiltonian_closure_residual = float(
        np.linalg.norm(hamiltonian_action - projected_hamiltonian_action)
    )

    target_jump_matrices = tuple(jump @ manifold_basis for jump in jumps_sparse)
    target_jump_residuals = tuple(float(np.linalg.norm(matrix)) for matrix in target_jump_matrices)
    max_target_jump_residual = max(target_jump_residuals) if target_jump_residuals else 0.0

    target_density = (manifold_basis @ manifold_basis.conj().T) / float(manifold_dimension)
    target_density_liouvillian_residual = float(
        np.linalg.norm(
            lindblad_rhs_density_matrix(
                target_density,
                hamiltonian=hamiltonian_sparse,
                jumps=list(jumps_sparse),
                backend=backend,
            )
        )
    )

    inflow_norm = _manifold_inflow_norm(
        jumps=jumps_sparse,
        manifold_basis=manifold_basis,
    )

    common_kernel_basis = _common_kernel_basis_from_sparse_operators(
        operators=jumps_sparse,
        dim=dim,
        tolerance=kernel_tolerance,
    )
    common_jump_kernel_dimension = int(common_kernel_basis.shape[1])

    target_projection_onto_common_kernel, target_distance_from_common_kernel = (
        _subspace_projection_and_distance(
            subspace_basis=manifold_basis,
            containing_basis=common_kernel_basis,
        )
    )
    target_in_common_jump_kernel = (
        target_distance_from_common_kernel <= np.sqrt(kernel_tolerance)
        or max_target_jump_residual <= kernel_tolerance
    )

    bad_common_kernel_basis = _kernel_basis_orthogonal_to_manifold(
        basis=common_kernel_basis,
        manifold_basis=manifold_basis,
        tolerance=kernel_tolerance,
    )
    bad_common_jump_kernel_dimension = int(bad_common_kernel_basis.shape[1])
    bad_common_jump_kernel_iprs = tuple(
        _state_ipr(bad_common_kernel_basis[:, index])
        for index in range(bad_common_kernel_basis.shape[1])
    )

    internal_hamiltonian = 0.5 * (internal_hamiltonian + internal_hamiltonian.conj().T)
    internal_hamiltonian_eigenvalues = tuple(
        complex(value) for value in np.linalg.eigvalsh(internal_hamiltonian)
    )
    expected_internal_liouvillian_eigenvalues = _internal_liouvillian_eigenvalues(
        internal_hamiltonian_eigenvalues
    )
    expected_internal_zero_mode_count = int(
        sum(
            abs(value) <= liouvillian_zero_tolerance
            for value in expected_internal_liouvillian_eigenvalues
        )
    )
    expected_internal_peripheral_mode_count = (
        len(expected_internal_liouvillian_eigenvalues) - expected_internal_zero_mode_count
    )

    liouvillian_zero_mode_count: int | None = None
    liouvillian_zero_mode_count_is_lower_bound = False
    liouvillian_spectral_gap: float | None = None
    liouvillian_decay_gap: float | None = None
    liouvillian_peripheral_mode_count: int | None = None
    liouvillian_eigenvalues: tuple[complex, ...] = ()
    actual_liouvillian_spectrum_method = "none"
    matched_internal_nondecaying_mode_count: int | None = None
    missing_internal_nondecaying_mode_count: int | None = None
    extra_nondecaying_mode_count: int | None = None
    extra_zero_mode_count: int | None = None
    external_decay_gap: float | None = None

    if check_liouvillian_spectrum and liouvillian_spectrum_method != "none":
        liouvillian_dimension = dim * dim
        liouvillian = build_liouvillian(
            hamiltonian_sparse,
            list(jumps_sparse),
            backend="scipy",
            sparse_format="csr",
        )

        if liouvillian_spectrum_method == "auto":
            actual_liouvillian_spectrum_method = (
                "dense" if liouvillian_dimension <= max_liouvillian_dense_dimension else "sparse"
            )
        else:
            actual_liouvillian_spectrum_method = liouvillian_spectrum_method

        if actual_liouvillian_spectrum_method == "dense":
            if liouvillian_dimension > max_liouvillian_dense_dimension:
                raise ValueError(
                    "Dense Liouvillian spectrum check is too expensive: "
                    f"dim^2={liouvillian_dimension}, "
                    f"max_liouvillian_dense_dimension={max_liouvillian_dense_dimension}. "
                    "Use liouvillian_spectrum_method='sparse' or 'auto', "
                    "or set check_liouvillian_spectrum=False."
                )
            eigenvalues = scipy_linalg.eigvals(liouvillian.toarray())
            eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
            is_partial_spectrum = False
        elif actual_liouvillian_spectrum_method == "sparse":
            eigenvalues = _sparse_liouvillian_near_zero_eigenvalues(
                liouvillian,
                n_eigenvalues=sparse_liouvillian_eigenvalue_count,
                zero_tolerance=liouvillian_zero_tolerance,
            )
            is_partial_spectrum = True
        else:
            raise ValueError(
                "liouvillian_spectrum_method must be 'auto', 'dense', 'sparse', or 'none'."
            )

        (
            liouvillian_zero_mode_count,
            liouvillian_zero_mode_count_is_lower_bound,
            liouvillian_spectral_gap,
            liouvillian_decay_gap,
            liouvillian_peripheral_mode_count,
            liouvillian_eigenvalues,
        ) = _summarize_liouvillian_eigenvalues(
            eigenvalues,
            zero_tolerance=liouvillian_zero_tolerance,
            is_partial_spectrum=is_partial_spectrum,
            requested_count=sparse_liouvillian_eigenvalue_count,
        )

        if not is_partial_spectrum:
            nondecaying_values = tuple(
                complex(value)
                for value in eigenvalues
                if abs(complex(value).real) <= liouvillian_zero_tolerance
            )
            match = _match_expected_internal_nondecaying_modes(
                observed=nondecaying_values,
                expected=expected_internal_liouvillian_eigenvalues,
                tolerance=liouvillian_zero_tolerance,
            )
            matched_internal_nondecaying_mode_count = match["matched"]
            missing_internal_nondecaying_mode_count = match["missing"]
            extra_nondecaying_mode_count = match["extra"]
            extra_zero_mode_count = max(
                0,
                int(liouvillian_zero_mode_count) - expected_internal_zero_mode_count,
            )
            external_decay_gap = _external_decay_gap_from_spectrum(
                eigenvalues=eigenvalues,
                matched_observed_indices=match["matched_observed_indices"],
                zero_tolerance=liouvillian_zero_tolerance,
            )

    likely_attractive_dark_manifold: bool | None
    if extra_nondecaying_mode_count is None:
        likely_attractive_dark_manifold = None
    else:
        likely_attractive_dark_manifold = (
            hamiltonian_closure_residual <= liouvillian_zero_tolerance
            and max_target_jump_residual <= liouvillian_zero_tolerance
            and target_density_liouvillian_residual <= liouvillian_zero_tolerance
            and extra_nondecaying_mode_count == 0
        )

    return DarkManifoldDiagnostics(
        dim=dim,
        n_jumps=len(jumps_sparse),
        manifold_dimension=manifold_dimension,
        hamiltonian_closure_residual=hamiltonian_closure_residual,
        target_jump_residuals=target_jump_residuals,
        max_target_jump_residual=max_target_jump_residual,
        target_density_liouvillian_residual=target_density_liouvillian_residual,
        inflow_norm=inflow_norm,
        common_jump_kernel_dimension=common_jump_kernel_dimension,
        target_projection_onto_common_kernel=target_projection_onto_common_kernel,
        target_distance_from_common_kernel=target_distance_from_common_kernel,
        target_in_common_jump_kernel=target_in_common_jump_kernel,
        bad_common_jump_kernel_dimension=bad_common_jump_kernel_dimension,
        bad_common_jump_kernel_iprs=bad_common_jump_kernel_iprs,
        internal_hamiltonian_eigenvalues=internal_hamiltonian_eigenvalues,
        expected_internal_liouvillian_eigenvalues=expected_internal_liouvillian_eigenvalues,
        expected_internal_zero_mode_count=expected_internal_zero_mode_count,
        expected_internal_peripheral_mode_count=expected_internal_peripheral_mode_count,
        liouvillian_zero_mode_count=liouvillian_zero_mode_count,
        liouvillian_zero_mode_count_is_lower_bound=bool(liouvillian_zero_mode_count_is_lower_bound),
        liouvillian_spectral_gap=liouvillian_spectral_gap,
        liouvillian_decay_gap=liouvillian_decay_gap,
        liouvillian_peripheral_mode_count=liouvillian_peripheral_mode_count,
        liouvillian_spectrum_method=actual_liouvillian_spectrum_method,
        liouvillian_eigenvalues=liouvillian_eigenvalues,
        matched_internal_nondecaying_mode_count=matched_internal_nondecaying_mode_count,
        missing_internal_nondecaying_mode_count=missing_internal_nondecaying_mode_count,
        extra_nondecaying_mode_count=extra_nondecaying_mode_count,
        extra_zero_mode_count=extra_zero_mode_count,
        external_decay_gap=external_decay_gap,
        likely_attractive_dark_manifold=likely_attractive_dark_manifold,
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
    liouvillian_spectrum_method: Literal["auto", "dense", "sparse", "none"] = "auto",
    sparse_liouvillian_eigenvalue_count: int = 16,
) -> DarkSubspaceDiagnostics:
    """Diagnose whether a dark target is likely unique/attractive.

    This is intended for small systems. It computes:

        1. target jump residuals ||J_mu psi||;
        2. common jump kernel dim intersection_mu ker J_mu;
        3. bad common-kernel dimension after removing the target direction;
        4. target Liouvillian residual ||L(|psi><psi|)||;
        5. optional Liouvillian zero-mode count.

    The Liouvillian spectrum check uses a dense solver for small Liouvillians and
    a sparse shift-invert Arnoldi solver for larger ones when
    ``liouvillian_spectrum_method="auto"``.  The sparse zero-mode count is a
    lower bound if all requested eigenvalues are numerically zero; increase
    ``sparse_liouvillian_eigenvalue_count`` to resolve more zero modes.
    """
    # _backend_obj = get_open_system_backend(backend)

    hamiltonian_sparse = _as_scipy_csr_matrix(hamiltonian)
    jumps_sparse = tuple(_as_scipy_csr_matrix(jump) for jump in jumps)

    target = np.asarray(target_state, dtype=np.complex128)
    if target.ndim != 1:
        raise ValueError("target_state must be one-dimensional.")

    target_norm = float(np.linalg.norm(target))
    if target_norm == 0.0:
        raise ValueError("target_state must be nonzero.")

    target = target / target_norm
    dim = int(target.size)

    if hamiltonian_sparse.shape != (dim, dim):
        raise ValueError("hamiltonian shape must be compatible with target_state.")

    for jump in jumps_sparse:
        if jump.shape != (dim, dim):
            raise ValueError(
                "Every jump operator must have shape " "(len(target_state), len(target_state))."
            )

    target_jump_vectors = tuple(jump @ target for jump in jumps_sparse)
    target_jump_residuals = tuple(float(np.linalg.norm(vector)) for vector in target_jump_vectors)
    max_target_jump_residual = max(target_jump_residuals) if target_jump_residuals else 0.0

    common_kernel_basis = _common_kernel_basis_from_sparse_operators(
        operators=jumps_sparse,
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

    target_liouvillian_residual = _rank_one_lindblad_rhs_norm(
        hamiltonian=hamiltonian_sparse,
        jumps=jumps_sparse,
        target=target,
        precomputed_jump_targets=target_jump_vectors,
    )

    liouvillian_zero_mode_count: int | None = None
    liouvillian_zero_mode_count_is_lower_bound = False
    liouvillian_spectral_gap: float | None = None
    liouvillian_decay_gap: float | None = None
    liouvillian_peripheral_mode_count: int | None = None
    liouvillian_eigenvalues: tuple[complex, ...] = ()
    actual_liouvillian_spectrum_method = "none"

    if check_liouvillian_spectrum and liouvillian_spectrum_method != "none":
        liouvillian_dimension = dim * dim
        liouvillian = build_liouvillian(
            hamiltonian_sparse,
            list(jumps_sparse),
            backend="scipy",
            sparse_format="csr",
        )

        if liouvillian_spectrum_method == "auto":
            actual_liouvillian_spectrum_method = (
                "dense" if liouvillian_dimension <= max_liouvillian_dense_dimension else "sparse"
            )
        else:
            actual_liouvillian_spectrum_method = liouvillian_spectrum_method

        if actual_liouvillian_spectrum_method == "dense":
            if liouvillian_dimension > max_liouvillian_dense_dimension:
                raise ValueError(
                    "Dense Liouvillian spectrum check is too expensive: "
                    f"dim^2={liouvillian_dimension}, "
                    f"max_liouvillian_dense_dimension="
                    f"{max_liouvillian_dense_dimension}. "
                    "Use liouvillian_spectrum_method='sparse' or 'auto', "
                    "or set check_liouvillian_spectrum=False."
                )

            eigenvalues = scipy_linalg.eigvals(liouvillian.toarray())
            eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
        elif actual_liouvillian_spectrum_method == "sparse":
            eigenvalues = _sparse_liouvillian_near_zero_eigenvalues(
                liouvillian,
                n_eigenvalues=sparse_liouvillian_eigenvalue_count,
                zero_tolerance=liouvillian_zero_tolerance,
            )
        else:
            raise ValueError(
                "liouvillian_spectrum_method must be 'auto', 'dense', 'sparse', or 'none'."
            )

        (
            liouvillian_zero_mode_count,
            liouvillian_zero_mode_count_is_lower_bound,
            liouvillian_spectral_gap,
            liouvillian_decay_gap,
            liouvillian_peripheral_mode_count,
            liouvillian_eigenvalues,
        ) = _summarize_liouvillian_eigenvalues(
            eigenvalues,
            zero_tolerance=liouvillian_zero_tolerance,
            is_partial_spectrum=(actual_liouvillian_spectrum_method == "sparse"),
            requested_count=sparse_liouvillian_eigenvalue_count,
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
        n_jumps=len(jumps_sparse),
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
        liouvillian_zero_mode_count_is_lower_bound=bool(liouvillian_zero_mode_count_is_lower_bound),
        liouvillian_spectral_gap=liouvillian_spectral_gap,
        liouvillian_decay_gap=liouvillian_decay_gap,
        liouvillian_peripheral_mode_count=liouvillian_peripheral_mode_count,
        liouvillian_spectrum_method=actual_liouvillian_spectrum_method,
        liouvillian_eigenvalues=liouvillian_eigenvalues,
        likely_unique_dark_state=likely_unique_dark_state,
    )


def _sparse_liouvillian_near_zero_eigenvalues(
    liouvillian: Any,
    *,
    n_eigenvalues: int,
    zero_tolerance: float,
) -> np.ndarray:
    """Return a partial spectrum close to zero for a sparse Liouvillian."""
    matrix = (
        liouvillian.tocsr()
        if hasattr(liouvillian, "tocsr")
        else scipy_sparse.csr_array(liouvillian)
    )
    dimension = int(matrix.shape[0])
    if dimension <= 2:
        return scipy_linalg.eigvals(matrix.toarray())

    k = max(1, min(int(n_eigenvalues), dimension - 2))

    # Shift-invert close to zero is usually far more reliable than ``which='SM'``
    # for non-Hermitian Liouvillians, but sigma=0 can fail because the
    # Liouvillian is singular.  Use a tiny positive real shift and fall back to
    # smallest-magnitude Arnoldi if the factorization is ill-conditioned.
    sigma = max(float(zero_tolerance) * 0.1, 1.0e-14)
    try:
        values = scipy_sparse_linalg.eigs(
            matrix,
            k=k,
            sigma=sigma,
            which="LM",
            return_eigenvectors=False,
        )
    except Exception:
        values = scipy_sparse_linalg.eigs(
            matrix,
            k=k,
            which="SM",
            return_eigenvectors=False,
        )

    return np.asarray(values, dtype=np.complex128)


def _summarize_liouvillian_eigenvalues(
    eigenvalues: npt.ArrayLike,
    *,
    zero_tolerance: float,
    is_partial_spectrum: bool,
    requested_count: int,
) -> tuple[int, bool, float | None, float | None, int | None, tuple[complex, ...]]:
    values = np.asarray(eigenvalues, dtype=np.complex128)
    if values.size == 0:
        return 0, False, None, None, None, ()

    abs_values = np.abs(values)
    zero_mask = abs_values <= zero_tolerance
    zero_count = int(np.count_nonzero(zero_mask))
    zero_count_is_lower_bound = bool(
        is_partial_spectrum and zero_count >= min(int(requested_count), values.size)
    )

    nonzero_abs = abs_values[~zero_mask]
    absolute_gap = float(np.min(nonzero_abs)) if nonzero_abs.size else None

    nonzero_real_parts = np.real(values[~zero_mask])
    decaying = nonzero_real_parts < -zero_tolerance
    decay_gap = float(-np.max(nonzero_real_parts[decaying])) if np.any(decaying) else None

    peripheral_mask = (~zero_mask) & (np.abs(np.real(values)) <= zero_tolerance)
    peripheral_count = int(np.count_nonzero(peripheral_mask))

    order = np.lexsort((np.real(values), abs_values))
    shown = tuple(complex(values[index]) for index in order[: min(16, values.size)])

    return (
        zero_count,
        zero_count_is_lower_bound,
        absolute_gap,
        decay_gap,
        peripheral_count,
        shown,
    )
