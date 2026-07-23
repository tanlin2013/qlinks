from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import scipy.linalg as scipy_linalg
import scipy.sparse as scipy_sparse

from qlinks.caging.nullspace import as_dense_array, nullspace_svd

ComplexArray = npt.NDArray[np.complex128]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HamiltonianGraphChainComplex:
    """Finite Hamiltonian-graph caging complex ``C2 -> C1 -> C0``.

    ``constraint_map`` is the physical leakage/eigenvalue map ``D_E`` from
    support amplitudes to violated Hamiltonian rows. ``generator_map`` maps a
    chosen library of bounded-support cage motifs into the support-amplitude
    space. The chain condition is ``constraint_map @ generator_map == 0``.

    The basis conventions are:

    * columns of ``generator_map`` are local cage generators;
    * columns of ``ker(constraint_map)`` are all exact cage amplitudes on the
      chosen support shell;
    * ``H_1 = ker(D_E) / im(T_R)`` is the many-body CLS-completeness defect;
    * ``H_2 = ker(T_R)`` records linear relations among translated motifs.
    """

    constraint_map: ComplexArray
    generator_map: ComplexArray
    support_indices: tuple[int, ...] | None = None
    test_indices: tuple[int, ...] | None = None
    generator_labels: tuple[str, ...] = ()

    @property
    def c0_dimension(self) -> int:
        return int(self.constraint_map.shape[0])

    @property
    def c1_dimension(self) -> int:
        return int(self.constraint_map.shape[1])

    @property
    def c2_dimension(self) -> int:
        return int(self.generator_map.shape[1])

    @property
    def chain_residual(self) -> float:
        return float(np.linalg.norm(self.constraint_map @ self.generator_map))


@dataclass(frozen=True, slots=True)
class HamiltonianGraphHomologyReport:
    """Numerical homology/cohomology report for a caging chain complex."""

    c0_dimension: int
    c1_dimension: int
    c2_dimension: int
    constraint_rank: int
    generator_rank: int
    cage_dimension: int
    h1_dimension: int
    h2_dimension: int
    chain_residual: float
    relative_chain_residual: float
    generator_containment_residual: float
    cage_basis: ComplexArray
    local_generator_basis: ComplexArray
    h1_basis: ComplexArray
    h2_basis: ComplexArray
    cocycle_basis: ComplexArray
    hodge_operator: ComplexArray
    hodge_eigenvalues: FloatArray
    hodge_gap: float | None
    tolerance: float

    @property
    def nu_mb(self) -> int:
        """Return the finite-volume many-body CLS-completeness defect."""
        return self.h1_dimension

    @property
    def is_chain_complex(self) -> bool:
        return self.relative_chain_residual <= self.tolerance

    @property
    def is_locally_complete(self) -> bool:
        return self.h1_dimension == 0

    def pairing_matrix(self, cage_representatives: npt.ArrayLike | None = None) -> ComplexArray:
        """Pair dual cocycles with supplied cage representatives.

        With no argument, harmonic ``H_1`` representatives are used. Under the
        Euclidean inner product the returned matrix should be the identity up
        to numerical tolerance.
        """
        representatives = (
            self.h1_basis
            if cage_representatives is None
            else _as_column_matrix(cage_representatives, self.c1_dimension)
        )
        return np.asarray(self.cocycle_basis.conj().T @ representatives, dtype=np.complex128)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "c0_dimension": self.c0_dimension,
            "c1_dimension": self.c1_dimension,
            "c2_dimension": self.c2_dimension,
            "constraint_rank": self.constraint_rank,
            "generator_rank": self.generator_rank,
            "cage_dimension": self.cage_dimension,
            "h1_dimension": self.h1_dimension,
            "nu_mb": self.nu_mb,
            "h2_dimension": self.h2_dimension,
            "chain_residual": self.chain_residual,
            "relative_chain_residual": self.relative_chain_residual,
            "generator_containment_residual": self.generator_containment_residual,
            "hodge_gap": self.hodge_gap,
            "is_chain_complex": self.is_chain_complex,
            "is_locally_complete": self.is_locally_complete,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class TermResolvedCagingReport:
    """Compare separately vanishing local channels with collective cancellation."""

    physical_constraint_map: ComplexArray
    resolved_constraint_map: ComplexArray
    physical_kernel_basis: ComplexArray
    resolved_kernel_basis: ComplexArray
    collective_quotient_basis: ComplexArray
    physical_nullity: int
    resolved_nullity: int
    collective_quotient_dimension: int
    resolved_containment_residual: float
    tolerance: float

    @property
    def has_collective_cancellation(self) -> bool:
        return self.collective_quotient_dimension > 0

    def channel_activity(self, states: npt.ArrayLike) -> FloatArray:
        """Return ``||\\widetilde D_E psi||`` for each supplied state column."""
        vectors = _as_column_matrix(states, self.physical_constraint_map.shape[1])
        values = np.linalg.norm(self.resolved_constraint_map @ vectors, axis=0)
        return np.asarray(values, dtype=np.float64)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "physical_nullity": self.physical_nullity,
            "resolved_nullity": self.resolved_nullity,
            "collective_quotient_dimension": self.collective_quotient_dimension,
            "resolved_containment_residual": self.resolved_containment_residual,
            "has_collective_cancellation": self.has_collective_cancellation,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class MotifRadiusHomologyPoint:
    """One motif-radius point in a local-generator saturation scan."""

    radius: int
    generator_rank: int
    h1_dimension: int
    h2_dimension: int
    chain_residual: float
    hodge_gap: float | None

    @classmethod
    def from_report(
        cls,
        radius: int,
        report: HamiltonianGraphHomologyReport,
    ) -> MotifRadiusHomologyPoint:
        return cls(
            radius=int(radius),
            generator_rank=report.generator_rank,
            h1_dimension=report.h1_dimension,
            h2_dimension=report.h2_dimension,
            chain_residual=report.chain_residual,
            hodge_gap=report.hodge_gap,
        )


@dataclass(frozen=True, slots=True)
class MotifRadiusSaturationReport:
    """Track whether ``nu_MB`` stabilizes as the motif library grows."""

    points: tuple[MotifRadiusHomologyPoint, ...]
    plateau_length: int = 2
    tolerance: float = 1.0e-10

    @property
    def classification(self) -> str:
        if not self.points:
            return "empty"
        if self.tolerance <= 0.0:
            return "invalid_tolerance"
        if any(point.chain_residual > self.tolerance for point in self.points):
            return "invalid_chain_data"
        required = max(1, int(self.plateau_length))
        if len(self.points) < required:
            return "insufficient_radius_range"
        tail = self.points[-required:]
        h1_values = {point.h1_dimension for point in tail}
        generator_ranks = {point.generator_rank for point in tail}
        if len(h1_values) != 1 or len(generator_ranks) != 1:
            return "not_saturated"
        value = tail[-1].h1_dimension
        return "locally_complete" if value == 0 else "saturated_defect_candidate"

    @property
    def saturated_nu_mb(self) -> int | None:
        if self.classification not in {"locally_complete", "saturated_defect_candidate"}:
            return None
        return self.points[-1].h1_dimension

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "classification": self.classification,
            "saturated_nu_mb": self.saturated_nu_mb,
            "plateau_length": self.plateau_length,
            "tolerance": self.tolerance,
            "points": tuple(
                {
                    "radius": point.radius,
                    "generator_rank": point.generator_rank,
                    "h1_dimension": point.h1_dimension,
                    "h2_dimension": point.h2_dimension,
                    "chain_residual": point.chain_residual,
                    "hodge_gap": point.hodge_gap,
                }
                for point in self.points
            ),
        }


@dataclass(frozen=True, slots=True)
class LaurentPeriodicKernelPoint:
    """Kernel diagnostic for one Laurent operator on a twisted finite ring."""

    length: int
    twist: float
    rank: int
    nullity: int
    singular_values: FloatArray
    smallest_positive_singular_value: float | None

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "length": self.length,
            "twist": self.twist,
            "rank": self.rank,
            "nullity": self.nullity,
            "smallest_positive_singular_value": self.smallest_positive_singular_value,
        }


@dataclass(frozen=True, slots=True)
class ScalarLaurentBulkPhaseReport:
    """Fredholm phase data for a scalar first-order Laurent constraint.

    The local relation is ``psi[j + 1] = transport * psi[j]`` with symbol
    ``b(z) = z - transport``.  When ``abs(transport) != 1`` the symbol is
    nonzero on the unit circle and its winding is well defined.  Unit-modulus
    transport lies exactly on the non-Fredholm transition locus.
    """

    transport: complex
    root_modulus: float
    unit_circle_gap: float
    winding_number: int | None
    toeplitz_index: int | None
    is_fredholm: bool
    localization_length: float | None
    tolerance: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "transport": self.transport,
            "root_modulus": self.root_modulus,
            "unit_circle_gap": self.unit_circle_gap,
            "winding_number": self.winding_number,
            "toeplitz_index": self.toeplitz_index,
            "is_fredholm": self.is_fredholm,
            "localization_length": self.localization_length,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class ScalarLaurentDomainWallReport:
    """Finite-chain domain wall between two scalar Laurent constraints.

    Sites are arranged as ``left_length`` bonds, one interface site, and
    ``right_length`` bonds.  The constraint matrix imposes the left transport
    on bonds to the left of the interface and the right transport on bonds to
    the right.  A right-chiral interface mode is topologically predicted only
    when both bulks are Fredholm and their winding numbers differ in the
    appropriate orientation.
    """

    left_bulk: ScalarLaurentBulkPhaseReport
    right_bulk: ScalarLaurentBulkPhaseReport
    left_length: int
    right_length: int
    constraint_matrix: ComplexArray
    kernel_dimension: int
    kernel_basis: ComplexArray
    canonical_mode: ComplexArray
    residual: float
    inverse_participation_ratio: float | None
    interface_site_weight: float | None
    interface_window_weight: float | None
    center_of_mass: float | None
    predicted_right_interface_modes: int | None
    predicted_left_interface_modes: int | None
    is_exponentially_interface_localized: bool
    classification: str
    tolerance: float

    @property
    def site_count(self) -> int:
        return int(self.left_length + self.right_length + 1)

    @property
    def interface_site(self) -> int:
        return int(self.left_length)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "left_transport": self.left_bulk.transport,
            "right_transport": self.right_bulk.transport,
            "left_winding": self.left_bulk.winding_number,
            "right_winding": self.right_bulk.winding_number,
            "left_length": self.left_length,
            "right_length": self.right_length,
            "kernel_dimension": self.kernel_dimension,
            "residual": self.residual,
            "inverse_participation_ratio": self.inverse_participation_ratio,
            "interface_site_weight": self.interface_site_weight,
            "interface_window_weight": self.interface_window_weight,
            "center_of_mass": self.center_of_mass,
            "predicted_right_interface_modes": self.predicted_right_interface_modes,
            "predicted_left_interface_modes": self.predicted_left_interface_modes,
            "is_exponentially_interface_localized": (self.is_exponentially_interface_localized),
            "classification": self.classification,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class IncidenceConstraintInterfaceReport:
    """Interface obtained by gluing two local incidence constraint modules.

    The combined differential contains both complete bulk constraint maps plus
    additional interface rows.  Consequently, any interface kernel lies inside
    the direct sum of the two original bulk kernels.  The report tests whether
    gluing merely merges the local ``H^0`` sectors, frustrates them, or creates
    a higher-arity problem; it cannot create a new quotient mode without
    modifying or removing bulk constraints near the interface.
    """

    left_support_dimension: int
    right_support_dimension: int
    left_kernel_dimension: int
    right_kernel_dimension: int
    decoupled_kernel_dimension: int
    interface_constraint_count: int
    combined_constraint_map: ComplexArray
    combined_kernel_basis: ComplexArray
    combined_kernel_dimension: int
    surviving_bulk_kernel_dimension: int
    interface_created_dimension: int
    interface_removed_dimension: int
    active_row_weight_histogram: tuple[tuple[int, int], ...]
    is_two_channel: bool
    connected_component_count: int | None
    betti_1: int | None
    gauge_flatness_residual: float | None
    kernel_equals_h0: bool
    classification: str
    tolerance: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "left_support_dimension": self.left_support_dimension,
            "right_support_dimension": self.right_support_dimension,
            "left_kernel_dimension": self.left_kernel_dimension,
            "right_kernel_dimension": self.right_kernel_dimension,
            "decoupled_kernel_dimension": self.decoupled_kernel_dimension,
            "interface_constraint_count": self.interface_constraint_count,
            "combined_kernel_dimension": self.combined_kernel_dimension,
            "surviving_bulk_kernel_dimension": self.surviving_bulk_kernel_dimension,
            "interface_created_dimension": self.interface_created_dimension,
            "interface_removed_dimension": self.interface_removed_dimension,
            "active_row_weight_histogram": self.active_row_weight_histogram,
            "is_two_channel": self.is_two_channel,
            "connected_component_count": self.connected_component_count,
            "betti_1": self.betti_1,
            "gauge_flatness_residual": self.gauge_flatness_residual,
            "kernel_equals_h0": self.kernel_equals_h0,
            "classification": self.classification,
            "tolerance": self.tolerance,
        }


def diagnose_scalar_laurent_bulk_phase(
    transport: complex,
    *,
    tolerance: float = 1.0e-10,
) -> ScalarLaurentBulkPhaseReport:
    """Diagnose the scalar symbol ``b(z)=z-transport`` on the unit circle."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    value = complex(transport)
    modulus = float(abs(value))
    gap = float(abs(1.0 - modulus))
    is_fredholm = bool(gap > tolerance)
    winding: int | None = None
    toeplitz_index: int | None = None
    localization_length: float | None = None
    if is_fredholm:
        winding = 1 if modulus < 1.0 else 0
        toeplitz_index = -winding
        if modulus > tolerance:
            localization_length = float(1.0 / abs(np.log(modulus)))
        else:
            localization_length = 0.0
    return ScalarLaurentBulkPhaseReport(
        transport=value,
        root_modulus=modulus,
        unit_circle_gap=gap,
        winding_number=winding,
        toeplitz_index=toeplitz_index,
        is_fredholm=is_fredholm,
        localization_length=localization_length,
        tolerance=tolerance,
    )


def diagnose_scalar_laurent_domain_wall(
    left_transport: complex,
    right_transport: complex,
    *,
    left_length: int = 24,
    right_length: int = 24,
    interface_window_radius: int = 1,
    tolerance: float = 1.0e-10,
) -> ScalarLaurentDomainWallReport:
    """Test the bulk--defect correspondence of two scalar transport modules."""
    if left_length < 1 or right_length < 1:
        raise ValueError("left_length and right_length must be positive.")
    if interface_window_radius < 0:
        raise ValueError("interface_window_radius must be nonnegative.")
    left = diagnose_scalar_laurent_bulk_phase(left_transport, tolerance=tolerance)
    right = diagnose_scalar_laurent_bulk_phase(right_transport, tolerance=tolerance)
    if abs(left.transport) <= tolerance or abs(right.transport) <= tolerance:
        raise ValueError("domain-wall transport factors must be nonzero.")

    site_count = left_length + right_length + 1
    interface_site = left_length
    differential = np.zeros((site_count - 1, site_count), dtype=np.complex128)
    for bond in range(site_count - 1):
        transport = left.transport if bond < interface_site else right.transport
        differential[bond, bond] = -transport
        differential[bond, bond + 1] = 1.0

    kernel = nullspace_svd(differential, tolerance=tolerance)
    mode = np.zeros(site_count, dtype=np.complex128)
    residual = 0.0
    ipr: float | None = None
    interface_site_weight: float | None = None
    interface_window_weight: float | None = None
    center_of_mass: float | None = None
    if kernel.shape[1]:
        # The first-order open-chain differential has a one-dimensional kernel.
        # Use the SVD vector rather than a recurrence so the diagnostic remains
        # stable for large transport contrasts.
        mode = np.asarray(kernel[:, 0], dtype=np.complex128)
        norm = float(np.linalg.norm(mode))
        if norm > tolerance:
            mode /= norm
        probabilities = np.abs(mode) ** 2
        residual = float(np.linalg.norm(differential @ mode))
        ipr = float(np.sum(probabilities**2))
        interface_site_weight = float(probabilities[interface_site])
        start = max(0, interface_site - interface_window_radius)
        stop = min(site_count, interface_site + interface_window_radius + 1)
        interface_window_weight = float(np.sum(probabilities[start:stop]))
        center_of_mass = float(np.dot(np.arange(site_count), probabilities))

    predicted_right: int | None = None
    predicted_left: int | None = None
    if left.is_fredholm and right.is_fredholm:
        assert left.winding_number is not None and right.winding_number is not None
        difference = int(right.winding_number - left.winding_number)
        predicted_right = max(0, difference)
        predicted_left = max(0, -difference)

    localized = bool(
        predicted_right is not None
        and predicted_right > 0
        and abs(left.transport) > 1.0 + tolerance
        and abs(right.transport) < 1.0 - tolerance
        and interface_window_weight is not None
        and interface_window_weight > 0.25
    )
    if not left.is_fredholm or not right.is_fredholm:
        classification = "critical_transport_no_fredholm_index"
    elif predicted_right and localized:
        classification = "fredholm_interface_mode"
    elif predicted_left:
        classification = "opposite_chirality_interface_mode"
    elif left.winding_number == right.winding_number:
        classification = "same_fredholm_phase_no_interface_index"
    else:
        classification = "index_jump_without_resolved_right_mode"

    return ScalarLaurentDomainWallReport(
        left_bulk=left,
        right_bulk=right,
        left_length=int(left_length),
        right_length=int(right_length),
        constraint_matrix=differential,
        kernel_dimension=int(kernel.shape[1]),
        kernel_basis=kernel,
        canonical_mode=mode,
        residual=residual,
        inverse_participation_ratio=ipr,
        interface_site_weight=interface_site_weight,
        interface_window_weight=interface_window_weight,
        center_of_mass=center_of_mass,
        predicted_right_interface_modes=predicted_right,
        predicted_left_interface_modes=predicted_left,
        is_exponentially_interface_localized=localized,
        classification=classification,
        tolerance=tolerance,
    )


def diagnose_incidence_constraint_interface(
    left_constraint_map: object,
    right_constraint_map: object,
    interface_constraint_map: object,
    *,
    tolerance: float = 1.0e-10,
) -> IncidenceConstraintInterfaceReport:
    """Glue two local constraint modules and test for an excess interface kernel."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    left = _as_matrix(left_constraint_map)
    right = _as_matrix(right_constraint_map)
    interface = _as_matrix(interface_constraint_map)
    combined_columns = left.shape[1] + right.shape[1]
    if interface.shape[1] != combined_columns:
        raise ValueError("interface_constraint_map has incompatible column dimension.")

    decoupled = scipy_linalg.block_diag(left, right).astype(np.complex128, copy=False)
    combined = np.vstack([decoupled, interface])
    left_kernel = nullspace_svd(left, tolerance=tolerance)
    right_kernel = nullspace_svd(right, tolerance=tolerance)
    bulk_kernel = scipy_linalg.block_diag(left_kernel, right_kernel).astype(
        np.complex128,
        copy=False,
    )
    combined_kernel = nullspace_svd(combined, tolerance=tolerance)
    overlaps = scipy_linalg.svdvals(bulk_kernel.conj().T @ combined_kernel)
    surviving = int(np.sum(overlaps >= 1.0 - 10.0 * tolerance))
    created = max(0, int(combined_kernel.shape[1]) - surviving)
    removed = max(0, int(bulk_kernel.shape[1]) - surviving)

    active_counts = np.sum(np.abs(combined) > tolerance, axis=1)
    active_counts = active_counts[active_counts > 0]
    unique, counts = np.unique(active_counts, return_counts=True)
    histogram = tuple((int(weight), int(count)) for weight, count in zip(unique, counts))
    is_two_channel = bool(active_counts.size and np.all(active_counts == 2))
    component_count: int | None = None
    betti_1: int | None = None
    flatness_residual: float | None = None
    kernel_equals_h0 = False
    if is_two_channel:
        n_vertices = combined.shape[1]
        adjacency: list[list[tuple[int, complex]]] = [[] for _ in range(n_vertices)]
        edge_count = 0
        for row in range(combined.shape[0]):
            columns = np.flatnonzero(np.abs(combined[row]) > tolerance)
            if columns.size == 0:
                continue
            first, second = int(columns[0]), int(columns[1])
            # A first-column gauge amplitude transports to the second as
            # -c_first/c_second.
            transport = complex(-combined[row, first] / combined[row, second])
            adjacency[first].append((second, transport))
            adjacency[second].append((first, 1.0 / transport))
            edge_count += 1

        labels = np.full(n_vertices, -1, dtype=np.int64)
        gauge = np.zeros(n_vertices, dtype=np.complex128)
        component_count = 0
        flatness_residual = 0.0
        for start in range(n_vertices):
            if labels[start] >= 0:
                continue
            labels[start] = component_count
            gauge[start] = 1.0 + 0.0j
            stack = [start]
            while stack:
                source = stack.pop()
                for target, transport in adjacency[source]:
                    candidate = transport * gauge[source]
                    if labels[target] < 0:
                        labels[target] = component_count
                        gauge[target] = candidate
                        stack.append(target)
                    else:
                        scale = max(abs(candidate), abs(gauge[target]), tolerance)
                        flatness_residual = max(
                            flatness_residual,
                            abs(gauge[target] - candidate) / scale,
                        )
            component_count += 1
        betti_1 = int(edge_count - n_vertices + component_count)
        kernel_equals_h0 = bool(
            flatness_residual <= 10.0 * tolerance and combined_kernel.shape[1] == component_count
        )

    if created > 0:
        classification = "interface_created_excess_kernel"
    elif is_two_channel and kernel_equals_h0 and removed > 0:
        classification = "flat_gluing_merges_local_h0_sectors"
    elif is_two_channel and flatness_residual is not None and flatness_residual > 10.0 * tolerance:
        classification = "frustrated_interface_lifts_local_h0_sector"
    elif interface.shape[0] == 0:
        classification = "decoupled_modules"
    else:
        classification = "interface_restricts_bulk_kernel_without_excess_mode"

    return IncidenceConstraintInterfaceReport(
        left_support_dimension=int(left.shape[1]),
        right_support_dimension=int(right.shape[1]),
        left_kernel_dimension=int(left_kernel.shape[1]),
        right_kernel_dimension=int(right_kernel.shape[1]),
        decoupled_kernel_dimension=int(bulk_kernel.shape[1]),
        interface_constraint_count=int(interface.shape[0]),
        combined_constraint_map=combined,
        combined_kernel_basis=combined_kernel,
        combined_kernel_dimension=int(combined_kernel.shape[1]),
        surviving_bulk_kernel_dimension=surviving,
        interface_created_dimension=created,
        interface_removed_dimension=removed,
        active_row_weight_histogram=histogram,
        is_two_channel=is_two_channel,
        connected_component_count=component_count,
        betti_1=betti_1,
        gauge_flatness_residual=flatness_residual,
        kernel_equals_h0=kernel_equals_h0,
        classification=classification,
        tolerance=tolerance,
    )


def build_hamiltonian_graph_chain_complex(
    hamiltonian: object,
    support_indices: Sequence[int],
    local_generators: npt.ArrayLike,
    *,
    energy: complex = 0.0,
    test_indices: Sequence[int] | None = None,
    generators_are_full_hilbert_vectors: bool = False,
    generator_labels: Sequence[str] = (),
) -> HamiltonianGraphChainComplex:
    """Build ``D_E`` and ``T_R`` from a Hamiltonian and a support shell.

    ``D_E`` is the selected-row block of ``(H - E I) P_support``. Local
    generators may be supplied either in support coordinates or as full
    Hilbert-space vectors.
    """
    shape = getattr(hamiltonian, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("hamiltonian must be a square matrix.")
    hilbert_dimension = int(shape[0])

    support = _validate_indices(support_indices, hilbert_dimension, "support_indices")
    tests = (
        np.arange(hilbert_dimension, dtype=np.int64)
        if test_indices is None
        else _validate_indices(test_indices, hilbert_dimension, "test_indices")
    )
    constraint_map = _hamiltonian_constraint_block(
        hamiltonian,
        support,
        tests,
        energy=energy,
    )

    raw_generators = np.asarray(local_generators, dtype=np.complex128)
    if raw_generators.ndim == 1:
        raw_generators = raw_generators[:, None]
    if raw_generators.ndim != 2:
        raise ValueError("local_generators must be one- or two-dimensional.")
    if generators_are_full_hilbert_vectors:
        if raw_generators.shape[0] != hilbert_dimension:
            raise ValueError("full-Hilbert generators have incompatible dimension.")
        generator_map = raw_generators[support, :]
        outside = np.delete(raw_generators, support, axis=0)
        if np.linalg.norm(outside) > 1.0e-10:
            raise ValueError("full-Hilbert generators must be supported on support_indices.")
    else:
        generator_map = _as_column_matrix(raw_generators, support.size)

    labels = tuple(str(label) for label in generator_labels)
    if labels and len(labels) != generator_map.shape[1]:
        raise ValueError("generator_labels must match the number of generator columns.")

    return HamiltonianGraphChainComplex(
        constraint_map=np.asarray(constraint_map, dtype=np.complex128),
        generator_map=np.asarray(generator_map, dtype=np.complex128),
        support_indices=tuple(int(value) for value in support),
        test_indices=tuple(int(value) for value in tests),
        generator_labels=labels,
    )


def diagnose_hamiltonian_graph_homology(
    complex_: HamiltonianGraphChainComplex,
    *,
    tolerance: float = 1.0e-10,
    require_chain_condition: bool = True,
) -> HamiltonianGraphHomologyReport:
    """Compute finite-volume homology, dual cocycles, and the Hodge gap."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    d1 = _as_matrix(complex_.constraint_map)
    d2 = _as_matrix(complex_.generator_map)
    if d1.shape[1] != d2.shape[0]:
        raise ValueError("constraint_map and generator_map have incompatible dimensions.")

    chain_residual = float(np.linalg.norm(d1 @ d2))
    scale = max(1.0, float(np.linalg.norm(d1) * np.linalg.norm(d2)))
    normalized_chain_residual = chain_residual / scale
    if require_chain_condition and normalized_chain_residual > tolerance:
        raise ValueError(
            "constraint_map @ generator_map is nonzero; the supplied maps do not form "
            "a chain complex."
        )

    cage_basis = nullspace_svd(d1, tolerance=tolerance)
    local_basis = _orthonormal_column_space(d2, tolerance=tolerance)
    h2_basis = nullspace_svd(d2, tolerance=tolerance)

    cage_projector = cage_basis @ cage_basis.conj().T
    containment_residual = float(np.linalg.norm((np.eye(d1.shape[1]) - cage_projector) @ d2))
    h1_raw = (np.eye(d1.shape[1]) - local_basis @ local_basis.conj().T) @ cage_basis
    h1_basis = _orthonormal_column_space(h1_raw, tolerance=tolerance)

    # Use the orthogonal projector onto im(T_R), rather than ``T_R T_R^†``,
    # so the harmonic gap is invariant under changes of generator basis and
    # nonzero rescalings of individual motif columns.
    local_projector = local_basis @ local_basis.conj().T
    hodge = d1.conj().T @ d1 + local_projector
    hodge = np.asarray(0.5 * (hodge + hodge.conj().T), dtype=np.complex128)
    eigenvalues, eigenvectors = scipy_linalg.eigh(hodge)
    eigenvalues = np.asarray(np.real_if_close(eigenvalues), dtype=np.float64)
    zero_mask = np.abs(eigenvalues) <= tolerance
    harmonic_basis = np.asarray(eigenvectors[:, zero_mask], dtype=np.complex128)
    # The harmonic basis is simultaneously a canonical H_1 representative and
    # a dual H^1 cocycle basis under the Euclidean inner product.
    if harmonic_basis.shape[1] == h1_basis.shape[1]:
        h1_basis = harmonic_basis
    positive = eigenvalues[eigenvalues > tolerance]
    hodge_gap = float(positive[0]) if positive.size else None

    constraint_rank = _matrix_rank(d1, tolerance)
    generator_rank = _matrix_rank(d2, tolerance)
    cage_dimension = int(cage_basis.shape[1])
    h1_dimension = int(h1_basis.shape[1])
    h2_dimension = int(h2_basis.shape[1])

    expected_h1 = cage_dimension - generator_rank
    if normalized_chain_residual <= tolerance and h1_dimension != expected_h1:
        raise RuntimeError("inconsistent H_1 dimension; check the numerical tolerance.")

    return HamiltonianGraphHomologyReport(
        c0_dimension=int(d1.shape[0]),
        c1_dimension=int(d1.shape[1]),
        c2_dimension=int(d2.shape[1]),
        constraint_rank=constraint_rank,
        generator_rank=generator_rank,
        cage_dimension=cage_dimension,
        h1_dimension=h1_dimension,
        h2_dimension=h2_dimension,
        chain_residual=chain_residual,
        relative_chain_residual=normalized_chain_residual,
        generator_containment_residual=containment_residual,
        cage_basis=cage_basis,
        local_generator_basis=local_basis,
        h1_basis=h1_basis,
        h2_basis=h2_basis,
        cocycle_basis=h1_basis.copy(),
        hodge_operator=hodge,
        hodge_eigenvalues=eigenvalues,
        hodge_gap=hodge_gap,
        tolerance=tolerance,
    )


def diagnose_term_resolved_caging(
    local_constraint_maps: Sequence[object],
    *,
    coefficients: Sequence[complex] | None = None,
    tolerance: float = 1.0e-10,
) -> TermResolvedCagingReport:
    """Resolve robust channelwise zeros from collectively cancelled cages.

    Every local map must have the same row and column dimensions. The physical
    differential is their coefficient-weighted sum, while the term-resolved
    differential is the vertical stack of the individual weighted maps.
    """
    if not local_constraint_maps:
        raise ValueError("at least one local constraint map is required.")
    maps = tuple(_as_matrix(value) for value in local_constraint_maps)
    shape = maps[0].shape
    if any(value.shape != shape for value in maps):
        raise ValueError("all local constraint maps must have the same shape.")
    weights = (
        np.ones(len(maps), dtype=np.complex128)
        if coefficients is None
        else np.asarray(coefficients, dtype=np.complex128).reshape(-1)
    )
    if weights.size != len(maps):
        raise ValueError("coefficients must match local_constraint_maps.")

    weighted = tuple(weight * value for weight, value in zip(weights, maps, strict=True))
    physical = np.sum(np.stack(weighted, axis=0), axis=0)
    resolved = np.vstack(weighted)
    physical_kernel = nullspace_svd(physical, tolerance=tolerance)
    resolved_kernel = nullspace_svd(resolved, tolerance=tolerance)

    physical_projector = physical_kernel @ physical_kernel.conj().T
    containment = float(np.linalg.norm((np.eye(shape[1]) - physical_projector) @ resolved_kernel))
    collective_raw = (
        np.eye(shape[1]) - resolved_kernel @ resolved_kernel.conj().T
    ) @ physical_kernel
    collective_basis = _orthonormal_column_space(collective_raw, tolerance=tolerance)

    return TermResolvedCagingReport(
        physical_constraint_map=np.asarray(physical, dtype=np.complex128),
        resolved_constraint_map=np.asarray(resolved, dtype=np.complex128),
        physical_kernel_basis=physical_kernel,
        resolved_kernel_basis=resolved_kernel,
        collective_quotient_basis=collective_basis,
        physical_nullity=int(physical_kernel.shape[1]),
        resolved_nullity=int(resolved_kernel.shape[1]),
        collective_quotient_dimension=int(collective_basis.shape[1]),
        resolved_containment_residual=containment,
        tolerance=tolerance,
    )


def twisted_translation_matrix(length: int, twist: float = 0.0) -> ComplexArray:
    """Return the unitary one-site translation with ``T**L = exp(i twist)``."""
    if length <= 0:
        raise ValueError("length must be positive.")
    translation = np.zeros((length, length), dtype=np.complex128)
    for source in range(length - 1):
        translation[source + 1, source] = 1.0
    translation[0, length - 1] = np.exp(1.0j * float(twist))
    return translation


def periodic_laurent_operator(
    coefficients: Mapping[int, complex | npt.ArrayLike],
    length: int,
    *,
    twist: float = 0.0,
) -> ComplexArray:
    """Evaluate a finite Laurent-polynomial matrix at twisted translation.

    A scalar dictionary such as ``{0: 1, 1: 1}`` constructs ``I + T``. Matrix
    coefficients construct a block Laurent operator via Kronecker products.
    """
    if not coefficients:
        raise ValueError("coefficients must not be empty.")
    blocks: dict[int, ComplexArray] = {}
    block_shape: tuple[int, int] | None = None
    for shift, value in coefficients.items():
        array = np.asarray(value, dtype=np.complex128)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        if array.ndim != 2:
            raise ValueError("Laurent coefficients must be scalars or matrices.")
        if block_shape is None:
            block_shape = array.shape
        if array.shape != block_shape:
            raise ValueError("all Laurent coefficient matrices must have the same shape.")
        blocks[int(shift)] = array
    assert block_shape is not None

    translation = twisted_translation_matrix(length, twist)
    result = np.zeros(
        (length * block_shape[0], length * block_shape[1]),
        dtype=np.complex128,
    )
    for shift, block in blocks.items():
        if shift >= 0:
            translated = np.linalg.matrix_power(translation, shift)
        else:
            translated = np.linalg.matrix_power(translation.conj().T, -shift)
        result += np.kron(translated, block)
    return result


def diagnose_periodic_laurent_kernel(
    coefficients: Mapping[int, complex | npt.ArrayLike],
    length: int,
    *,
    twist: float = 0.0,
    tolerance: float = 1.0e-10,
) -> LaurentPeriodicKernelPoint:
    """Compute the finite-ring nullity and smallest nonzero singular value."""
    operator = periodic_laurent_operator(coefficients, length, twist=twist)
    singular_values = np.asarray(scipy_linalg.svdvals(operator), dtype=np.float64)
    rank = int(np.sum(singular_values > tolerance))
    positive = np.sort(singular_values[singular_values > tolerance])
    gap = float(positive[0]) if positive.size else None
    return LaurentPeriodicKernelPoint(
        length=int(length),
        twist=float(twist),
        rank=rank,
        nullity=int(operator.shape[1] - rank),
        singular_values=singular_values,
        smallest_positive_singular_value=gap,
    )


def _hamiltonian_constraint_block(
    hamiltonian: object,
    support: npt.NDArray[np.int64],
    tests: npt.NDArray[np.int64],
    *,
    energy: complex,
) -> ComplexArray:
    """Return ``P_test (H-E) P_support`` without densifying the full matrix."""
    if scipy_sparse.issparse(hamiltonian):
        block = hamiltonian[tests, :][:, support].toarray()
    else:
        dense = np.asarray(hamiltonian, dtype=np.complex128)
        if dense.ndim != 2:
            raise ValueError("hamiltonian must be two-dimensional.")
        block = dense[np.ix_(tests, support)].copy()

    block = np.asarray(block, dtype=np.complex128)
    test_position = {int(index): row for row, index in enumerate(tests)}
    for column, support_index in enumerate(support):
        row = test_position.get(int(support_index))
        if row is not None:
            block[row, column] -= complex(energy)
    return block


def _as_matrix(matrix: object) -> ComplexArray:
    array = as_dense_array(matrix)
    if array.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    return np.asarray(array, dtype=np.complex128)


def _as_column_matrix(vectors: npt.ArrayLike, row_count: int) -> ComplexArray:
    array = np.asarray(vectors, dtype=np.complex128)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != row_count:
        raise ValueError("vectors have incompatible row dimension.")
    return array


def _orthonormal_column_space(matrix: object, *, tolerance: float) -> ComplexArray:
    array = _as_matrix(matrix)
    if array.shape[1] == 0:
        return np.zeros((array.shape[0], 0), dtype=np.complex128)
    return np.asarray(scipy_linalg.orth(array, rcond=tolerance), dtype=np.complex128)


def _matrix_rank(matrix: ComplexArray, tolerance: float) -> int:
    return int(np.sum(scipy_linalg.svdvals(matrix) > tolerance))


def _validate_indices(indices: Sequence[int], upper: int, name: str) -> npt.NDArray[np.int64]:
    values = np.asarray(tuple(indices), dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(values < 0) or np.any(values >= upper) or np.unique(values).size != values.size:
        raise ValueError(f"{name} must contain unique indices in range({upper}).")
    return values
