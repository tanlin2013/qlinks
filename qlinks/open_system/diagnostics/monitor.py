from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from qlinks.open_system._subspace import (
    _as_scipy_csr_matrix,
    _common_kernel_basis_from_sparse_operators,
    _kernel_basis_orthogonal_to_target,
    _projection_norm_onto_basis,
)
from qlinks.open_system.diagnostics._formatting import (
    _format_float,
    _format_float_or_none,
    _state_ipr,
)
from qlinks.open_system.diagnostics._linalg import _monitor_hamiltonian_leakage_norms


@dataclass(frozen=True, slots=True)
class MonitorKernelClosureDiagnostics:
    """Diagnostics for monitor-kernel closure under Hamiltonian mixing.

    The monitor kernel is ``intersection_i ker(M_i)``.  For a monitor-recycler
    design ``L_i = V_i M_i``, this kernel is always contained in the jump
    kernel and therefore measures what the recyclers cannot see directly.

    The first Hamiltonian-closure layer appends the constraints ``M_i H``.  If
    this sharply reduces the bad kernel, attraction is possible but can be slow
    because the Hamiltonian must rotate bad monitor-kernel states into the
    monitored subspace before dissipation acts.
    """

    dim: int
    n_monitors: int
    closure_order: int

    max_target_monitor_residual: float
    target_monitor_residuals: tuple[float, ...]

    monitor_kernel_dimension: int
    target_projection_onto_monitor_kernel: float
    target_distance_from_monitor_kernel: float
    target_in_monitor_kernel: bool
    bad_monitor_kernel_dimension: int
    bad_monitor_kernel_iprs: tuple[float, ...]

    bad_kernel_hamiltonian_leakage_norms: tuple[float, ...]
    min_bad_kernel_hamiltonian_leakage_norm: float | None
    mean_bad_kernel_hamiltonian_leakage_norm: float | None
    max_bad_kernel_hamiltonian_leakage_norm: float | None

    closure_kernel_dimension: int
    target_projection_onto_closure_kernel: float
    target_distance_from_closure_kernel: float
    target_in_closure_kernel: bool
    bad_closure_kernel_dimension: int
    bad_closure_kernel_iprs: tuple[float, ...]

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "dim": self.dim,
            "n_monitors": self.n_monitors,
            "closure_order": self.closure_order,
            "max_target_monitor_residual": self.max_target_monitor_residual,
            "monitor_kernel_dimension": self.monitor_kernel_dimension,
            "target_projection_onto_monitor_kernel": (self.target_projection_onto_monitor_kernel),
            "target_distance_from_monitor_kernel": (self.target_distance_from_monitor_kernel),
            "target_in_monitor_kernel": self.target_in_monitor_kernel,
            "bad_monitor_kernel_dimension": self.bad_monitor_kernel_dimension,
            "bad_monitor_kernel_iprs": self.bad_monitor_kernel_iprs,
            "bad_kernel_hamiltonian_leakage_norms": (self.bad_kernel_hamiltonian_leakage_norms),
            "min_bad_kernel_hamiltonian_leakage_norm": (
                self.min_bad_kernel_hamiltonian_leakage_norm
            ),
            "mean_bad_kernel_hamiltonian_leakage_norm": (
                self.mean_bad_kernel_hamiltonian_leakage_norm
            ),
            "max_bad_kernel_hamiltonian_leakage_norm": (
                self.max_bad_kernel_hamiltonian_leakage_norm
            ),
            "closure_kernel_dimension": self.closure_kernel_dimension,
            "target_projection_onto_closure_kernel": (self.target_projection_onto_closure_kernel),
            "target_distance_from_closure_kernel": (self.target_distance_from_closure_kernel),
            "target_in_closure_kernel": self.target_in_closure_kernel,
            "bad_closure_kernel_dimension": self.bad_closure_kernel_dimension,
            "bad_closure_kernel_iprs": self.bad_closure_kernel_iprs,
        }

    def to_rich(self):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "MonitorKernelClosureDiagnostics.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.dim))
        overview.add_row("number of monitors", str(self.n_monitors))
        overview.add_row("closure order", str(self.closure_order))

        monitor_table = Table(title="Monitor kernel")
        monitor_table.add_column("quantity", style="bold")
        monitor_table.add_column("value", justify="right")
        monitor_table.add_row("max ||M_i psi||", _format_float(self.max_target_monitor_residual))
        monitor_table.add_row("dim intersection ker M_i", str(self.monitor_kernel_dimension))
        monitor_table.add_row(
            "bad monitor-kernel dimension",
            str(self.bad_monitor_kernel_dimension),
        )
        monitor_table.add_row(
            "min H-leakage from bad kernel",
            _format_float_or_none(self.min_bad_kernel_hamiltonian_leakage_norm),
        )
        monitor_table.add_row(
            "mean H-leakage from bad kernel",
            _format_float_or_none(self.mean_bad_kernel_hamiltonian_leakage_norm),
        )
        monitor_table.add_row(
            "max H-leakage from bad kernel",
            _format_float_or_none(self.max_bad_kernel_hamiltonian_leakage_norm),
        )

        closure_table = Table(title="Hamiltonian closure")
        closure_table.add_column("quantity", style="bold")
        closure_table.add_column("value", justify="right")
        closure_table.add_row("dim ker{M_i, M_i H}", str(self.closure_kernel_dimension))
        closure_table.add_row(
            "bad closure-kernel dimension",
            str(self.bad_closure_kernel_dimension),
        )
        closure_table.add_row(
            "target distance from closure kernel",
            _format_float(self.target_distance_from_closure_kernel),
        )

        return Panel(
            Group(overview, monitor_table, closure_table),
            title=Text("Monitor-kernel closure diagnostics", style="bold cyan"),
            border_style="cyan",
        )


def diagnose_monitor_kernel_closure(
    *,
    hamiltonian: Any,
    monitors: Sequence[Any],
    target_state: npt.ArrayLike,
    closure_order: int = 1,
    tolerance: float = 1e-10,
) -> MonitorKernelClosureDiagnostics:
    """Diagnose whether local monitors are closed by Hamiltonian mixing.

    This is designed for monitor-recycler jumps ``L_i = V_i M_i``.  Recyclers
    cannot act on states in ``intersection_i ker(M_i)``, so the size of this
    kernel and its leakage under ``H`` are the first diagnostics to inspect.

    Currently ``closure_order`` supports ``0`` or ``1``.  Order 1 appends the
    constraints ``M_i H`` and computes the common kernel of ``{M_i, M_i H}``.
    """
    if closure_order not in (0, 1):
        raise ValueError("closure_order currently supports only 0 or 1.")

    hamiltonian_sparse = _as_scipy_csr_matrix(hamiltonian)
    monitor_sparse = tuple(_as_scipy_csr_matrix(monitor) for monitor in monitors)

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

    for monitor in monitor_sparse:
        if monitor.shape != (dim, dim):
            raise ValueError(
                "Every monitor must have shape (len(target_state), len(target_state))."
            )

    target_monitor_vectors = tuple(monitor @ target for monitor in monitor_sparse)
    target_monitor_residuals = tuple(
        float(np.linalg.norm(vector)) for vector in target_monitor_vectors
    )
    max_target_monitor_residual = max(target_monitor_residuals) if target_monitor_residuals else 0.0

    monitor_kernel_basis = _common_kernel_basis_from_sparse_operators(
        operators=monitor_sparse,
        dim=dim,
        tolerance=tolerance,
    )
    monitor_kernel_dimension = int(monitor_kernel_basis.shape[1])
    target_projection_onto_monitor_kernel = _projection_norm_onto_basis(
        vector=target,
        basis=monitor_kernel_basis,
    )
    target_distance_from_monitor_kernel = float(
        np.sqrt(max(0.0, 1.0 - target_projection_onto_monitor_kernel**2))
    )
    target_in_monitor_kernel = (
        target_distance_from_monitor_kernel <= np.sqrt(tolerance)
        or max_target_monitor_residual <= tolerance
    )

    bad_monitor_kernel_basis = _kernel_basis_orthogonal_to_target(
        basis=monitor_kernel_basis,
        target=target,
        tolerance=tolerance,
    )
    bad_monitor_kernel_dimension = int(bad_monitor_kernel_basis.shape[1])
    bad_monitor_kernel_iprs = tuple(
        _state_ipr(bad_monitor_kernel_basis[:, index])
        for index in range(bad_monitor_kernel_basis.shape[1])
    )
    bad_leakages = _monitor_hamiltonian_leakage_norms(
        hamiltonian=hamiltonian_sparse,
        monitors=monitor_sparse,
        basis=bad_monitor_kernel_basis,
    )

    if bad_leakages.size:
        min_bad_leakage = float(np.min(bad_leakages))
        mean_bad_leakage = float(np.mean(bad_leakages))
        max_bad_leakage = float(np.max(bad_leakages))
    else:
        min_bad_leakage = None
        mean_bad_leakage = None
        max_bad_leakage = None

    if closure_order == 0:
        closure_operators = monitor_sparse
    else:
        closure_operators = monitor_sparse + tuple(
            (monitor @ hamiltonian_sparse).tocsr() for monitor in monitor_sparse
        )

    closure_kernel_basis = _common_kernel_basis_from_sparse_operators(
        operators=closure_operators,
        dim=dim,
        tolerance=tolerance,
    )
    closure_kernel_dimension = int(closure_kernel_basis.shape[1])
    target_projection_onto_closure_kernel = _projection_norm_onto_basis(
        vector=target,
        basis=closure_kernel_basis,
    )
    target_distance_from_closure_kernel = float(
        np.sqrt(max(0.0, 1.0 - target_projection_onto_closure_kernel**2))
    )
    target_in_closure_kernel = target_distance_from_closure_kernel <= np.sqrt(tolerance)

    bad_closure_kernel_basis = _kernel_basis_orthogonal_to_target(
        basis=closure_kernel_basis,
        target=target,
        tolerance=tolerance,
    )
    bad_closure_kernel_dimension = int(bad_closure_kernel_basis.shape[1])
    bad_closure_kernel_iprs = tuple(
        _state_ipr(bad_closure_kernel_basis[:, index])
        for index in range(bad_closure_kernel_basis.shape[1])
    )

    return MonitorKernelClosureDiagnostics(
        dim=dim,
        n_monitors=len(monitor_sparse),
        closure_order=int(closure_order),
        max_target_monitor_residual=max_target_monitor_residual,
        target_monitor_residuals=target_monitor_residuals,
        monitor_kernel_dimension=monitor_kernel_dimension,
        target_projection_onto_monitor_kernel=target_projection_onto_monitor_kernel,
        target_distance_from_monitor_kernel=target_distance_from_monitor_kernel,
        target_in_monitor_kernel=target_in_monitor_kernel,
        bad_monitor_kernel_dimension=bad_monitor_kernel_dimension,
        bad_monitor_kernel_iprs=bad_monitor_kernel_iprs,
        bad_kernel_hamiltonian_leakage_norms=tuple(float(value) for value in bad_leakages),
        min_bad_kernel_hamiltonian_leakage_norm=min_bad_leakage,
        mean_bad_kernel_hamiltonian_leakage_norm=mean_bad_leakage,
        max_bad_kernel_hamiltonian_leakage_norm=max_bad_leakage,
        closure_kernel_dimension=closure_kernel_dimension,
        target_projection_onto_closure_kernel=target_projection_onto_closure_kernel,
        target_distance_from_closure_kernel=target_distance_from_closure_kernel,
        target_in_closure_kernel=target_in_closure_kernel,
        bad_closure_kernel_dimension=bad_closure_kernel_dimension,
        bad_closure_kernel_iprs=bad_closure_kernel_iprs,
    )
