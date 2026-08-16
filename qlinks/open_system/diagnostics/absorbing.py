from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from qlinks.open_system._subspace import _as_scipy_csr_matrix
from qlinks.open_system.backend import OpenSystemBackend, OpenSystemBackendName
from qlinks.open_system.diagnostics._formatting import _format_float
from qlinks.open_system.diagnostics._linalg import (
    _low_rank_operator_frobenius_norm,
    _orthogonal_component_norm,
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
                "Every jump operator must have shape (len(target_state), len(target_state))."
            )

    hamiltonian_target = hamiltonian_sparse @ target
    hamiltonian_commutator_norm = _low_rank_operator_frobenius_norm(
        (
            (1.0, hamiltonian_target, target),
            (-1.0, target, hamiltonian_target),
        )
    )

    jump_diagnostics: list[AbsorbingProjectorJumpDiagnostics] = []

    liouvillian_adjoint_terms: list[tuple[complex, np.ndarray, np.ndarray]] = [
        (1j, hamiltonian_target, target),
        (-1j, target, hamiltonian_target),
    ]

    for jump_index, jump in enumerate(jumps_sparse):
        jump_target = jump @ target
        jump_dagger_target = jump.conj().T @ target
        jump_dagger_jump_target = jump.conj().T @ jump_target

        target_residual = float(np.linalg.norm(jump_target))
        outflow_norm = _orthogonal_component_norm(jump_target, target)
        inflow_norm = _orthogonal_component_norm(jump_dagger_target, target)
        commutator_norm = _low_rank_operator_frobenius_norm(
            (
                (1.0, jump_target, target),
                (-1.0, target, jump_dagger_target),
            )
        )

        dissipator_terms = (
            (1.0, jump_dagger_target, jump_dagger_target),
            (-0.5, jump_dagger_jump_target, target),
            (-0.5, target, jump_dagger_jump_target),
        )
        dissipator_adjoint_projector_norm = _low_rank_operator_frobenius_norm(dissipator_terms)

        liouvillian_adjoint_terms.extend(dissipator_terms)

        jump_diagnostics.append(
            AbsorbingProjectorJumpDiagnostics(
                jump_index=jump_index,
                target_residual=target_residual,
                outflow_norm=outflow_norm,
                inflow_norm=inflow_norm,
                commutator_norm=commutator_norm,
                dissipator_adjoint_projector_norm=dissipator_adjoint_projector_norm,
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

    liouvillian_adjoint_projector_norm = _low_rank_operator_frobenius_norm(
        tuple(liouvillian_adjoint_terms)
    )

    target_is_dark = max_target_residual <= tolerance
    has_recycling_inflow = max_inflow_norm > tolerance
    absorbing_projector_is_conserved = liouvillian_adjoint_projector_norm <= tolerance

    has_absorbing_projector_symmetry = (
        target_is_dark and not has_recycling_inflow and absorbing_projector_is_conserved
    )

    return AbsorbingProjectorSymmetryDiagnostics(
        dim=dim,
        n_jumps=len(jumps_sparse),
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
