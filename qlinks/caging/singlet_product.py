"""Diagnostics and tensor-network handoff for products of local QDM singlets.

The routines in this module answer a deliberately sharp question: can an exact
square-QDM cage be built inside the tensor-product span of independently solved
two-plaquette singlets?  The answer is decided by the leakage map from that
finite product support to configurations outside it.  A full-column-rank
leakage map is a rigorous no-go statement for *every* coefficient tensor in the
chosen support, including arbitrary MPS or PEPS correlations over the singlet
labels.

When the restricted support is insufficient, :class:`QDMSingletTNProblem`
provides a small, backend-neutral handoff object for tensor-network experiments.
The optional quimb bridge is intentionally isolated so qlinks does not depend on
a tensor-network package unless the ``tn`` extra is installed.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt
from scipy import linalg as scipy_linalg
from scipy import sparse as scipy_sparse

from qlinks.caging.local_search_certification import (
    _config_key,
    _make_qdm_multi_padding_from_exterior,
    _qdm_flip_transition_from_action,
    _qdm_global_plaquette_actions,
    _qdm_global_self_loop_values_from_actions,
    iter_multi_qdm_block_paddings,
    make_qdm_cage_block,
)
from qlinks.caging.local_search_proposals import (
    StripeMotifRegionProposal,
    run_local_region_proposal,
)
from qlinks.caging.local_search_qdm import (
    build_qdm_local_kinetic_matrix,
    build_qdm_local_region_from_plaquettes,
    enumerate_qdm_local_basis,
    qdm_local_self_loop_values,
)
from qlinks.caging.local_search_types import (
    LocalQDMCageBlock,
    LocalQDMCageSearchConfig,
    LocalQDMMultiPaddingConfig,
    LocalQDMRegion,
    MultiLocalQDMPadding,
)

SquareQDMSingletDirection = Literal["x", "y"]


@dataclass(frozen=True, slots=True)
class QDMProductSubspaceEigenstate:
    """One exact eigenvector found inside a selected block-product support."""

    energy: complex
    coefficients: npt.NDArray[np.complex128]
    residual: float

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.complex128)
        if coefficients.ndim != 1:
            raise ValueError("coefficients must be one-dimensional.")
        norm = float(np.linalg.norm(coefficients))
        if norm == 0.0:
            raise ValueError("coefficients must have nonzero norm.")
        object.__setattr__(self, "coefficients", coefficients / norm)
        object.__setattr__(self, "energy", complex(self.energy))
        object.__setattr__(self, "residual", float(self.residual))


@dataclass(frozen=True, slots=True)
class QDMBlockProductSubspaceReport:
    """Hamiltonian closure diagnostic for one finite block-product support.

    ``leakage_matrix`` maps coefficient vectors on ``global_support_configs``
    to all one-hop configurations outside that support.  Therefore
    ``leakage_nullity == 0`` rules out not only the independent product state,
    but every correlated coefficient tensor on the same local singlet labels.
    """

    block_ids: tuple[int, ...]
    block_support_sizes: tuple[int, ...]
    padding: MultiLocalQDMPadding
    support_hamiltonian: npt.NDArray[np.complex128]
    leakage_matrix: npt.NDArray[np.complex128]
    leakage_output_configs: npt.NDArray[np.int64]
    leakage_rank: int
    leakage_nullity: int
    product_energy: complex
    product_residual: float
    exact_states: tuple[QDMProductSubspaceEigenstate, ...]
    tolerance: float

    def __post_init__(self) -> None:
        support_hamiltonian = np.asarray(self.support_hamiltonian, dtype=np.complex128)
        leakage_matrix = np.asarray(self.leakage_matrix, dtype=np.complex128)
        leakage_output_configs = np.asarray(self.leakage_output_configs, dtype=np.int64)
        support_size = int(self.padding.global_support_configs.shape[0])
        if support_hamiltonian.shape != (support_size, support_size):
            raise ValueError("support_hamiltonian has the wrong shape.")
        if leakage_matrix.ndim != 2 or leakage_matrix.shape[1] != support_size:
            raise ValueError("leakage_matrix width must match the product support size.")
        if leakage_output_configs.ndim != 2:
            raise ValueError("leakage_output_configs must be two-dimensional.")
        if leakage_output_configs.shape[0] != leakage_matrix.shape[0]:
            raise ValueError("leakage output rows must align with leakage_matrix.")
        object.__setattr__(self, "support_hamiltonian", support_hamiltonian.copy())
        object.__setattr__(self, "leakage_matrix", leakage_matrix.copy())
        object.__setattr__(self, "leakage_output_configs", leakage_output_configs.copy())
        object.__setattr__(self, "leakage_rank", int(self.leakage_rank))
        object.__setattr__(self, "leakage_nullity", int(self.leakage_nullity))
        object.__setattr__(self, "product_energy", complex(self.product_energy))
        object.__setattr__(self, "product_residual", float(self.product_residual))
        object.__setattr__(self, "tolerance", float(self.tolerance))

    @property
    def support_size(self) -> int:
        return int(self.padding.global_support_configs.shape[0])

    @property
    def n_leakage_outputs(self) -> int:
        return int(self.leakage_matrix.shape[0])

    @property
    def product_state_is_exact(self) -> bool:
        return self.product_residual <= self.tolerance

    @property
    def has_exact_state(self) -> bool:
        return bool(self.exact_states)

    @property
    def is_ruled_out_within_product_support(self) -> bool:
        """Whether no coefficient tensor can cancel the external leakage."""
        return self.leakage_nullity == 0

    @property
    def requires_enlarged_local_basis(self) -> bool:
        """Whether an MPS/PEPS over the current singlet labels cannot work."""
        return self.is_ruled_out_within_product_support


@dataclass(frozen=True, slots=True)
class SquareQDMTwoPlaquetteSingletBlock:
    """A translated two-plaquette antisymmetric square-QDM cage block."""

    block: LocalQDMCageBlock
    plaquette_ids: tuple[int, int]
    anchor_cells: tuple[tuple[int, int], tuple[int, int]]
    direction: SquareQDMSingletDirection

    @property
    def block_id(self) -> int:
        return int(self.block.block_id)

    @property
    def support_size(self) -> int:
        return int(self.block.support_size)

    @property
    def covered_site_ids(self) -> tuple[int, ...]:
        return tuple(int(i) for i in np.flatnonzero(self.block.site_counts))


@dataclass(frozen=True, slots=True)
class SquareQDMSingletStripeProductReport:
    """All shared-exterior tests for one regularly spaced singlet stripe."""

    direction: SquareQDMSingletDirection
    transverse_coordinate: int
    selected_blocks: tuple[SquareQDMTwoPlaquetteSingletBlock, ...]
    subspace_reports: tuple[QDMBlockProductSubspaceReport, ...]
    failure_reason: str | None = None

    @property
    def n_blocks(self) -> int:
        return len(self.selected_blocks)

    @property
    def has_padding(self) -> bool:
        return bool(self.subspace_reports)

    @property
    def has_exact_state(self) -> bool:
        return any(report.has_exact_state for report in self.subspace_reports)

    @property
    def is_full_rank_no_go(self) -> bool:
        return bool(self.subspace_reports) and all(
            report.is_ruled_out_within_product_support for report in self.subspace_reports
        )


@dataclass(frozen=True, slots=True)
class SquareQDMSingletProductTiling:
    """One exact cover of all square-lattice sites by singlet rectangles."""

    blocks: tuple[SquareQDMTwoPlaquetteSingletBlock, ...]
    n_horizontal: int
    n_vertical: int

    @property
    def support_size(self) -> int:
        result = 1
        for block in self.blocks:
            result *= int(block.support_size)
        return result


@dataclass(frozen=True, slots=True)
class SquareQDMSingletTilingRecord:
    """Subspace analysis for one exact-cover singlet tiling."""

    tiling: SquareQDMSingletProductTiling
    report: QDMBlockProductSubspaceReport


@dataclass(frozen=True, slots=True)
class SquareQDMSingletTilingSearchResult:
    """Collection of exact-cover singlet-product diagnostics."""

    records: tuple[SquareQDMSingletTilingRecord, ...]
    n_tilings_enumerated: int
    truncated: bool

    @property
    def n_exact_states(self) -> int:
        return sum(len(record.report.exact_states) for record in self.records)

    @property
    def n_full_rank_no_go(self) -> int:
        return sum(record.report.is_ruled_out_within_product_support for record in self.records)

    @property
    def all_full_rank_no_go(self) -> bool:
        return bool(self.records) and self.n_full_rank_no_go == len(self.records)


@dataclass(frozen=True, slots=True)
class QDMBoundaryResolvedTileBasis:
    """Finite local QDM basis resolved by the dimer deficit at boundary sites.

    A signature entry is the number of dimers that must be supplied by links
    outside the tile.  These signatures are the natural virtual labels for a
    constrained tensor network.
    """

    region: LocalQDMRegion
    configurations: npt.NDArray[np.int64]
    boundary_site_ids: npt.NDArray[np.int64]
    boundary_deficits: npt.NDArray[np.int64]
    unique_boundary_signatures: npt.NDArray[np.int64]
    signature_indices: npt.NDArray[np.int64]
    local_hamiltonian: scipy_sparse.csr_array

    def __post_init__(self) -> None:
        configurations = np.asarray(self.configurations, dtype=np.int64)
        boundary_site_ids = np.asarray(self.boundary_site_ids, dtype=np.int64)
        boundary_deficits = np.asarray(self.boundary_deficits, dtype=np.int64)
        signatures = np.asarray(self.unique_boundary_signatures, dtype=np.int64)
        signature_indices = np.asarray(self.signature_indices, dtype=np.int64)
        if configurations.ndim != 2:
            raise ValueError("configurations must be two-dimensional.")
        if boundary_site_ids.ndim != 1:
            raise ValueError("boundary_site_ids must be one-dimensional.")
        if boundary_deficits.shape != (configurations.shape[0], boundary_site_ids.size):
            raise ValueError("boundary_deficits has the wrong shape.")
        if signatures.ndim != 2 or signatures.shape[1] != boundary_site_ids.size:
            raise ValueError("unique_boundary_signatures has the wrong shape.")
        if signature_indices.shape != (configurations.shape[0],):
            raise ValueError("signature_indices has the wrong shape.")
        local_hamiltonian = scipy_sparse.csr_array(self.local_hamiltonian)
        if local_hamiltonian.shape != (configurations.shape[0], configurations.shape[0]):
            raise ValueError("local_hamiltonian has the wrong shape.")
        object.__setattr__(self, "configurations", configurations.copy())
        object.__setattr__(self, "boundary_site_ids", boundary_site_ids.copy())
        object.__setattr__(self, "boundary_deficits", boundary_deficits.copy())
        object.__setattr__(self, "unique_boundary_signatures", signatures.copy())
        object.__setattr__(self, "signature_indices", signature_indices.copy())
        object.__setattr__(self, "local_hamiltonian", local_hamiltonian)

    @property
    def dimension(self) -> int:
        return int(self.configurations.shape[0])

    @property
    def n_boundary_signatures(self) -> int:
        return int(self.unique_boundary_signatures.shape[0])

    def indices_for_signature(self, signature_index: int) -> npt.NDArray[np.int64]:
        return np.flatnonzero(self.signature_indices == int(signature_index)).astype(np.int64)


@dataclass(frozen=True, slots=True)
class SquareQDMSingletBoundaryTile:
    """One singlet core embedded in an enlarged boundary-resolved halo basis."""

    singlet: SquareQDMTwoPlaquetteSingletBlock
    basis: QDMBoundaryResolvedTileBasis
    core_sector_indices: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]

    def __post_init__(self) -> None:
        sectors = tuple(np.asarray(indices, dtype=np.int64) for indices in self.core_sector_indices)
        if len(sectors) != 2:
            raise ValueError("core_sector_indices must contain the two singlet core sectors.")
        object.__setattr__(self, "core_sector_indices", sectors)

    @property
    def enlarged_dimension(self) -> int:
        return self.basis.dimension

    @property
    def core_compatible_dimension(self) -> int:
        return int(sum(indices.size for indices in self.core_sector_indices))

    @property
    def virtual_signature_count(self) -> int:
        return self.basis.n_boundary_signatures


@dataclass(frozen=True, slots=True)
class QDMSingletTNProblem:
    """Backend-neutral finite objective for a singlet-label TN ansatz.

    This object is useful for prototyping MPS/PEPS optimizers with an external
    tensor-network package.  It also states a rigorous limitation: when
    ``leakage_nullity == 0``, increasing the bond dimension while keeping the
    same physical singlet label cannot yield an exact state.  The local physical
    basis must first be enlarged, e.g. by adding halo/boundary configurations.
    """

    physical_dimensions: tuple[int, ...]
    support_hamiltonian: npt.NDArray[np.complex128]
    leakage_matrix: npt.NDArray[np.complex128]
    reference_coefficients: npt.NDArray[np.complex128]
    tolerance: float = 1.0e-10
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_report(cls, report: QDMBlockProductSubspaceReport) -> QDMSingletTNProblem:
        return cls(
            physical_dimensions=report.block_support_sizes,
            support_hamiltonian=report.support_hamiltonian,
            leakage_matrix=report.leakage_matrix,
            reference_coefficients=np.asarray(
                report.padding.global_amplitudes,
                dtype=np.complex128,
            ),
            tolerance=report.tolerance,
            metadata={
                "block_ids": report.block_ids,
                "leakage_rank": report.leakage_rank,
                "leakage_nullity": report.leakage_nullity,
                "requires_enlarged_local_basis": report.requires_enlarged_local_basis,
            },
        )

    @property
    def support_size(self) -> int:
        return int(np.prod(self.physical_dimensions, dtype=np.int64))

    @property
    def leakage_nullity(self) -> int:
        rank = int(np.linalg.matrix_rank(self.leakage_matrix, tol=self.tolerance))
        return self.support_size - rank

    @property
    def requires_enlarged_local_basis(self) -> bool:
        return self.leakage_nullity == 0

    def loss(self, coefficients: npt.ArrayLike, *, energy: complex | None = None) -> float:
        """Return ``||Kc||² + ||Hc-Ec||²`` for a normalized dense vector."""
        vector = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
        if vector.size != self.support_size:
            raise ValueError("coefficient vector size does not match physical_dimensions.")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("coefficients must have nonzero norm.")
        vector = vector / norm
        h_vector = self.support_hamiltonian @ vector
        effective_energy = complex(np.vdot(vector, h_vector)) if energy is None else complex(energy)
        leakage = self.leakage_matrix @ vector
        internal = h_vector - effective_energy * vector
        return float(np.vdot(leakage, leakage).real + np.vdot(internal, internal).real)

    def to_quimb_mps(
        self,
        coefficients: npt.ArrayLike | None = None,
        *,
        max_bond: int | None = None,
        cutoff: float = 0.0,
    ) -> object:
        """Convert a dense coefficient tensor to a quimb MPS.

        This bridge is intended for diagnostics and initialization.  It does not
        imply that the current physical support is sufficient for an exact
        state; check :attr:`requires_enlarged_local_basis` first.
        """
        if not quimb_available():
            raise ImportError(
                "quimb is not installed. Install qlinks with the 'tn' extra or "
                "install quimb>=1.14.0."
            )
        import quimb.tensor as qtn  # type: ignore[import-not-found]

        vector = (
            np.asarray(self.reference_coefficients, dtype=np.complex128)
            if coefficients is None
            else np.asarray(coefficients, dtype=np.complex128)
        ).reshape(-1)
        if vector.size != self.support_size:
            raise ValueError("coefficient vector size does not match physical_dimensions.")
        split_opts: dict[str, object] = {"cutoff": float(cutoff)}
        if max_bond is not None:
            split_opts["max_bond"] = int(max_bond)
        return qtn.MatrixProductState.from_dense(
            vector,
            dims=self.physical_dimensions,
            **split_opts,
        )


def quimb_available() -> bool:
    """Return whether the optional quimb tensor-network backend is installed."""
    return importlib.util.find_spec("quimb") is not None


def analyze_qdm_block_product_subspace(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    padding: MultiLocalQDMPadding,
    *,
    tolerance: float = 1.0e-10,
) -> QDMBlockProductSubspaceReport:
    """Build the exact restricted Hamiltonian and external leakage map."""
    fixed_blocks = tuple(blocks)
    if tuple(int(block.block_id) for block in fixed_blocks) != tuple(padding.block_ids):
        by_id = {int(block.block_id): block for block in fixed_blocks}
        try:
            fixed_blocks = tuple(by_id[int(block_id)] for block_id in padding.block_ids)
        except KeyError as error:
            raise ValueError("blocks do not cover padding.block_ids.") from error

    configs = np.asarray(padding.global_support_configs, dtype=np.int64)
    support_size = int(configs.shape[0])
    if support_size == 0:
        raise ValueError("padding product support must be non-empty.")
    support_index = {_config_key(config): index for index, config in enumerate(configs)}
    if len(support_index) != support_size:
        raise ValueError("padding contains duplicate global support configurations.")

    actions = _qdm_global_plaquette_actions(model)
    support_hamiltonian = np.zeros((support_size, support_size), dtype=np.complex128)
    diagonal = _qdm_global_self_loop_values_from_actions(configs, actions)
    support_hamiltonian[np.arange(support_size), np.arange(support_size)] = diagonal

    leakage_rows: dict[tuple[int, ...], npt.NDArray[np.complex128]] = {}
    leakage_configs: dict[tuple[int, ...], npt.NDArray[np.int64]] = {}
    for column, config in enumerate(configs):
        for action in actions:
            transition = _qdm_flip_transition_from_action(config, action)
            if transition is None:
                continue
            final_config, coefficient = transition
            key = _config_key(final_config)
            row = support_index.get(key)
            if row is not None:
                support_hamiltonian[row, column] += coefficient
                continue
            if key not in leakage_rows:
                leakage_rows[key] = np.zeros(support_size, dtype=np.complex128)
                leakage_configs[key] = np.asarray(final_config, dtype=np.int64)
            leakage_rows[key][column] += coefficient

    ordered_keys = tuple(sorted(leakage_rows))
    leakage_matrix = (
        np.vstack([leakage_rows[key] for key in ordered_keys]).astype(np.complex128)
        if ordered_keys
        else np.zeros((0, support_size), dtype=np.complex128)
    )
    leakage_output_configs = (
        np.vstack([leakage_configs[key] for key in ordered_keys]).astype(np.int64)
        if ordered_keys
        else np.zeros((0, configs.shape[1]), dtype=np.int64)
    )

    hermitian_residual = float(np.linalg.norm(support_hamiltonian - support_hamiltonian.conj().T))
    if hermitian_residual > max(tolerance, 1.0e-14) * max(1.0, support_size):
        raise ValueError(
            "Restricted product-space Hamiltonian is not Hermitian; "
            f"residual={hermitian_residual:.3e}."
        )
    support_hamiltonian = 0.5 * (support_hamiltonian + support_hamiltonian.conj().T)

    leakage_rank = int(np.linalg.matrix_rank(leakage_matrix, tol=tolerance))
    leakage_nullity = support_size - leakage_rank

    product_coefficients = np.asarray(padding.global_amplitudes, dtype=np.complex128)
    product_coefficients = product_coefficients / np.linalg.norm(product_coefficients)
    product_h = support_hamiltonian @ product_coefficients
    product_energy = complex(np.vdot(product_coefficients, product_h))
    product_internal = product_h - product_energy * product_coefficients
    product_leakage = leakage_matrix @ product_coefficients
    product_residual = float(
        np.sqrt(
            np.vdot(product_internal, product_internal).real
            + np.vdot(product_leakage, product_leakage).real
        )
    )

    exact_states: list[QDMProductSubspaceEigenstate] = []
    if leakage_nullity == 0:
        eigenvalues = np.empty(0, dtype=np.float64)
        eigenvectors = np.empty((support_size, 0), dtype=np.complex128)
        start = support_size
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(support_hamiltonian)
        start = 0
    while start < support_size:
        stop = start + 1
        while stop < support_size and abs(eigenvalues[stop] - eigenvalues[start]) <= tolerance:
            stop += 1
        eigenspace = eigenvectors[:, start:stop]
        if leakage_matrix.shape[0] == 0:
            kernel_coordinates = np.eye(stop - start, dtype=np.complex128)
        else:
            kernel_coordinates = scipy_linalg.null_space(
                leakage_matrix @ eigenspace,
                rcond=tolerance,
            )
        for coordinate_index in range(kernel_coordinates.shape[1]):
            coefficients = eigenspace @ kernel_coordinates[:, coordinate_index]
            coefficients = coefficients / np.linalg.norm(coefficients)
            energy = complex(eigenvalues[start])
            internal = support_hamiltonian @ coefficients - energy * coefficients
            leakage = leakage_matrix @ coefficients
            residual = float(
                np.sqrt(np.vdot(internal, internal).real + np.vdot(leakage, leakage).real)
            )
            if residual <= tolerance:
                exact_states.append(
                    QDMProductSubspaceEigenstate(
                        energy=energy,
                        coefficients=coefficients,
                        residual=residual,
                    )
                )
        start = stop

    return QDMBlockProductSubspaceReport(
        block_ids=tuple(int(block.block_id) for block in fixed_blocks),
        block_support_sizes=tuple(int(block.support_size) for block in fixed_blocks),
        padding=padding,
        support_hamiltonian=support_hamiltonian,
        leakage_matrix=leakage_matrix,
        leakage_output_configs=leakage_output_configs,
        leakage_rank=leakage_rank,
        leakage_nullity=leakage_nullity,
        product_energy=product_energy,
        product_residual=product_residual,
        exact_states=tuple(exact_states),
        tolerance=tolerance,
    )


def square_qdm_two_plaquette_singlet_blocks(
    model: object,
    *,
    directions: Sequence[SquareQDMSingletDirection] = ("x", "y"),
    tolerance: float = 1.0e-10,
    block_id_start: int = 0,
) -> tuple[SquareQDMTwoPlaquetteSingletBlock, ...]:
    """Find all translated antisymmetric two-plaquette square-QDM blocks."""
    direction_set = {str(direction) for direction in directions}
    if not direction_set or not direction_set.issubset({"x", "y"}):
        raise ValueError("directions must be a non-empty subset of {'x', 'y'}.")
    proposal = StripeMotifRegionProposal(
        model=model,
        config=LocalQDMCageSearchConfig(
            halo_layers=0,
            boundary_mode="relaxed",
            tolerance=tolerance,
            degenerate_basis_strategy="none",
        ),
        motif_sizes=(2,),
        sources=("stripe",),
        subset_mode="windows",
        stripe_directions=(0, 1),
    )
    scan = run_local_region_proposal(proposal)
    wrappers: list[SquareQDMTwoPlaquetteSingletBlock] = []
    next_block_id = int(block_id_start)
    for scan_record in scan.records:
        proposal_record = scan_record.proposal_record
        plaquette_ids = tuple(int(pid) for pid in proposal_record.plaquette_ids)
        if len(plaquette_ids) != 2:
            continue
        anchors = tuple(
            tuple(int(value) for value in model.lattice.plaquette_anchor_cell(pid))
            for pid in plaquette_ids
        )
        if anchors[0][1] == anchors[1][1]:
            direction: SquareQDMSingletDirection = "x"
        elif anchors[0][0] == anchors[1][0]:
            direction = "y"
        else:
            continue
        if direction not in direction_set:
            continue

        for local_record in scan_record.result.records:
            support = np.asarray(local_record.cage_state.support, dtype=np.int64)
            if support.size != 2:
                continue
            amplitudes = np.asarray(local_record.local_state, dtype=np.complex128)
            if not np.isclose(abs(amplitudes[0]), abs(amplitudes[1]), atol=tolerance):
                continue
            ratio = amplitudes[1] / amplitudes[0]
            if not np.isclose(ratio, -1.0, atol=tolerance):
                continue
            try:
                block = make_qdm_cage_block(
                    model,
                    local_record,
                    block_id=next_block_id,
                )
            except ValueError:
                continue
            wrappers.append(
                SquareQDMTwoPlaquetteSingletBlock(
                    block=block,
                    plaquette_ids=(plaquette_ids[0], plaquette_ids[1]),
                    anchor_cells=(anchors[0], anchors[1]),
                    direction=direction,
                )
            )
            next_block_id += 1
            break
    return tuple(wrappers)


def analyze_square_qdm_singlet_stripe_product(
    model: object,
    *,
    direction: SquareQDMSingletDirection = "x",
    transverse_coordinate: int = 0,
    offset: int = 0,
    spacing: int = 3,
    max_paddings: int = 1,
    include_sectors: bool = False,
    require_static_exterior: bool = True,
    tolerance: float = 1.0e-10,
) -> SquareQDMSingletStripeProductReport:
    """Test a regular stripe of independent two-plaquette singlet factors."""
    if direction not in {"x", "y"}:
        raise ValueError("direction must be 'x' or 'y'.")
    if spacing < 2:
        raise ValueError("spacing must be at least two plaquettes.")
    longitudinal_size = int(model.lattice.lx if direction == "x" else model.lattice.ly)
    transverse_size = int(model.lattice.ly if direction == "x" else model.lattice.lx)
    if longitudinal_size % spacing != 0:
        return SquareQDMSingletStripeProductReport(
            direction=direction,
            transverse_coordinate=int(transverse_coordinate),
            selected_blocks=(),
            subspace_reports=(),
            failure_reason="longitudinal_size_not_divisible_by_spacing",
        )
    transverse_coordinate %= transverse_size
    wrappers = square_qdm_two_plaquette_singlet_blocks(
        model,
        directions=(direction,),
        tolerance=tolerance,
    )
    by_anchor_set = {frozenset(wrapper.anchor_cells): wrapper for wrapper in wrappers}
    selected: list[SquareQDMTwoPlaquetteSingletBlock] = []
    for start in range(offset % spacing, longitudinal_size, spacing):
        if direction == "x":
            anchors = frozenset(
                {
                    (start % longitudinal_size, transverse_coordinate),
                    ((start + 1) % longitudinal_size, transverse_coordinate),
                }
            )
        else:
            anchors = frozenset(
                {
                    (transverse_coordinate, start % longitudinal_size),
                    (transverse_coordinate, (start + 1) % longitudinal_size),
                }
            )
        wrapper = by_anchor_set.get(anchors)
        if wrapper is None:
            return SquareQDMSingletStripeProductReport(
                direction=direction,
                transverse_coordinate=transverse_coordinate,
                selected_blocks=tuple(selected),
                subspace_reports=(),
                failure_reason=f"missing_singlet_block_at_{sorted(anchors)}",
            )
        selected.append(wrapper)

    blocks = tuple(wrapper.block for wrapper in selected)
    config = LocalQDMMultiPaddingConfig(
        min_blocks=len(blocks),
        max_blocks=len(blocks),
        max_paddings=max_paddings,
        max_padding_attempts=max_paddings,
        max_paddings_per_packing=max_paddings,
        include_sectors=include_sectors,
        require_static_exterior=require_static_exterior,
        require_kinetic_separation=False,
        max_product_support_size=2 ** len(blocks),
        tolerance=tolerance,
    )
    reports: list[QDMBlockProductSubspaceReport] = []
    for padding in iter_multi_qdm_block_paddings(
        model,
        blocks,
        config=config,
        max_yielded=max_paddings,
    ):
        reports.append(
            analyze_qdm_block_product_subspace(
                model,
                blocks,
                padding,
                tolerance=tolerance,
            )
        )
    return SquareQDMSingletStripeProductReport(
        direction=direction,
        transverse_coordinate=transverse_coordinate,
        selected_blocks=tuple(selected),
        subspace_reports=tuple(reports),
        failure_reason=None if reports else "no_shared_static_exterior_padding",
    )


def enumerate_square_qdm_singlet_exact_covers(
    model: object,
    *,
    singlet_blocks: Sequence[SquareQDMTwoPlaquetteSingletBlock] | None = None,
    max_tilings: int | None = None,
) -> tuple[SquareQDMSingletProductTiling, ...]:
    """Enumerate exact site covers by horizontal/vertical singlet rectangles."""
    if max_tilings is not None and max_tilings < 0:
        raise ValueError("max_tilings must be non-negative or None.")
    wrappers = tuple(
        square_qdm_two_plaquette_singlet_blocks(model) if singlet_blocks is None else singlet_blocks
    )
    n_sites = int(model.lattice.num_sites)
    full_mask = (1 << n_sites) - 1
    block_masks: list[int] = []
    blocks_by_site: list[list[int]] = [[] for _ in range(n_sites)]
    for block_index, wrapper in enumerate(wrappers):
        mask = 0
        for site_id in wrapper.covered_site_ids:
            mask |= 1 << int(site_id)
        block_masks.append(mask)
        for site_id in wrapper.covered_site_ids:
            blocks_by_site[int(site_id)].append(block_index)

    tilings: list[SquareQDMSingletProductTiling] = []
    chosen: list[int] = []

    def dfs(covered_mask: int) -> None:
        if max_tilings is not None and len(tilings) >= max_tilings:
            return
        if covered_mask == full_mask:
            selected = tuple(wrappers[index] for index in chosen)
            tilings.append(
                SquareQDMSingletProductTiling(
                    blocks=selected,
                    n_horizontal=sum(block.direction == "x" for block in selected),
                    n_vertical=sum(block.direction == "y" for block in selected),
                )
            )
            return

        best_site = -1
        best_candidates: list[int] | None = None
        remaining = full_mask ^ covered_mask
        while remaining:
            low_bit = remaining & -remaining
            site_id = low_bit.bit_length() - 1
            candidates = [
                index for index in blocks_by_site[site_id] if block_masks[index] & covered_mask == 0
            ]
            if not candidates:
                return
            if best_candidates is None or len(candidates) < len(best_candidates):
                best_site = site_id
                best_candidates = candidates
                if len(candidates) == 1:
                    break
            remaining ^= low_bit
        if best_site < 0 or best_candidates is None:
            return
        for block_index in best_candidates:
            chosen.append(block_index)
            dfs(covered_mask | block_masks[block_index])
            chosen.pop()
            if max_tilings is not None and len(tilings) >= max_tilings:
                return

    dfs(0)
    return tuple(tilings)


def analyze_square_qdm_singlet_product_tilings(
    model: object,
    *,
    max_tilings: int | None = None,
    tolerance: float = 1.0e-10,
) -> SquareQDMSingletTilingSearchResult:
    """Analyze exact-cover tilings without invoking a global cage search."""
    wrappers = square_qdm_two_plaquette_singlet_blocks(model, tolerance=tolerance)
    tilings = enumerate_square_qdm_singlet_exact_covers(
        model,
        singlet_blocks=wrappers,
        max_tilings=max_tilings,
    )
    records: list[SquareQDMSingletTilingRecord] = []
    n_global_links = int(model.lattice.num_links)
    for tiling in tilings:
        blocks = tuple(wrapper.block for wrapper in tiling.blocks)
        used_links = {
            int(link_id)
            for block in blocks
            for link_id in np.asarray(block.link_ids, dtype=np.int64)
        }
        exterior_link_ids = np.asarray(
            [link_id for link_id in range(n_global_links) if link_id not in used_links],
            dtype=np.int64,
        )
        padding = _make_qdm_multi_padding_from_exterior(
            model,
            blocks,
            exterior_link_ids=exterior_link_ids,
            exterior_config=np.zeros(exterior_link_ids.size, dtype=np.int64),
        )
        report = analyze_qdm_block_product_subspace(
            model,
            blocks,
            padding,
            tolerance=tolerance,
        )
        records.append(SquareQDMSingletTilingRecord(tiling=tiling, report=report))
    truncated = max_tilings is not None and len(tilings) >= max_tilings
    return SquareQDMSingletTilingSearchResult(
        records=tuple(records),
        n_tilings_enumerated=len(tilings),
        truncated=truncated,
    )


def build_qdm_boundary_resolved_tile_basis(
    model: object,
    *,
    plaquette_ids: Sequence[int] | npt.ArrayLike,
    halo_layers: int = 1,
    max_states: int | None = 100_000,
) -> QDMBoundaryResolvedTileBasis:
    """Build the enlarged local basis needed after a singlet-support no-go.

    The basis keeps every locally valid configuration in a finite plaquette
    halo and groups configurations by the outside dimer deficits on boundary
    sites.  No tensor-network backend is required for this preprocessing step.
    """
    if halo_layers < 0:
        raise ValueError("halo_layers must be non-negative.")
    region = build_qdm_local_region_from_plaquettes(
        model,
        plaquette_ids=plaquette_ids,
        halo_layers=halo_layers,
        boundary_mode="relaxed",
        scoring_plaquette_ids=plaquette_ids,
    )
    local_basis = enumerate_qdm_local_basis(
        model,
        region,
        include_sectors_when_full=False,
        max_states=max_states,
    )
    configurations = np.asarray(local_basis.states, dtype=np.int64)
    link_position = {int(link_id): index for index, link_id in enumerate(region.link_ids)}
    required_count = int(getattr(model, "required_count", 1))
    boundary_site_ids = np.asarray(region.boundary_site_ids, dtype=np.int64)
    deficits = np.zeros((configurations.shape[0], boundary_site_ids.size), dtype=np.int64)
    for boundary_position, site_id in enumerate(boundary_site_ids):
        local_incident = [
            link_position[int(link_id)]
            for link_id in model.lattice.incident_links(int(site_id))
            if int(link_id) in link_position
        ]
        local_counts = (
            np.sum(configurations[:, local_incident], axis=1, dtype=np.int64)
            if local_incident
            else np.zeros(configurations.shape[0], dtype=np.int64)
        )
        deficits[:, boundary_position] = required_count - local_counts
    signatures, signature_indices = np.unique(deficits, axis=0, return_inverse=True)
    kinetic = build_qdm_local_kinetic_matrix(model, region, local_basis)
    diagonal = qdm_local_self_loop_values(model, region, local_basis)
    local_hamiltonian = scipy_sparse.csr_array(kinetic, dtype=np.complex128) + scipy_sparse.diags(
        diagonal,
        format="csr",
        dtype=np.complex128,
    )
    return QDMBoundaryResolvedTileBasis(
        region=region,
        configurations=configurations,
        boundary_site_ids=boundary_site_ids,
        boundary_deficits=deficits,
        unique_boundary_signatures=signatures,
        signature_indices=np.asarray(signature_indices, dtype=np.int64),
        local_hamiltonian=local_hamiltonian,
    )


def build_square_qdm_singlet_boundary_tile(
    model: object,
    singlet: SquareQDMTwoPlaquetteSingletBlock,
    *,
    halo_layers: int = 1,
    max_states: int | None = 100_000,
) -> SquareQDMSingletBoundaryTile:
    """Embed a two-state singlet core in a halo basis for TN construction."""
    basis = build_qdm_boundary_resolved_tile_basis(
        model,
        plaquette_ids=singlet.plaquette_ids,
        halo_layers=halo_layers,
        max_states=max_states,
    )
    local_position = {int(link_id): index for index, link_id in enumerate(basis.region.link_ids)}
    core_positions = np.asarray(
        [local_position[int(link_id)] for link_id in singlet.block.link_ids],
        dtype=np.int64,
    )
    sectors: list[npt.NDArray[np.int64]] = []
    for core_config in singlet.block.support_configs:
        matches = np.all(
            basis.configurations[:, core_positions] == np.asarray(core_config, dtype=np.int64),
            axis=1,
        )
        sectors.append(np.flatnonzero(matches).astype(np.int64))
    return SquareQDMSingletBoundaryTile(
        singlet=singlet,
        basis=basis,
        core_sector_indices=(sectors[0], sectors[1]),
    )
