"""Unified cage Lindblad jump-design API.

This module is the preferred public entry point for cage-state Lindblad
engineering.  It treats a single cage state as a one-dimensional dark manifold
and a degenerate cage multiplet as a higher-dimensional dark manifold, then
runs the same detector/recycler workflow for both cases.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from qlinks.basis import basis_configs_from_build_result
from qlinks.caging.search import CageRecord
from qlinks.models.base import ModelBuildResult
from qlinks.models.local_terms import LocalTermDescriptor, LocalTermKind
from qlinks.open_system.backend import OpenSystemBackendName
from qlinks.open_system.constructions.degenerate_cage import (
    DegenerateCageJumpDesignWorkflowReport,
    DegenerateCageLindbladConstruction,
    LocalRegionSource,
    build_degenerate_cage_lindblad_construction,
)
from qlinks.open_system.solvers import LindbladProblem

DetectorOperatorKind = Literal["kinetic", "potential", "hamiltonian"]


@dataclass(frozen=True, slots=True)
class CageLindbladDetectorOperators:
    """Named local operators used to build dark detector combinations.

    ``operators`` are the matrices ``O_i`` supplied to the dark-detector solver,
    while ``names`` are the corresponding labels used in workflow/readout reports.
    """

    operators: tuple[Any, ...]
    names: tuple[str, ...]
    terms: tuple[LocalTermDescriptor, ...] = ()

    def __post_init__(self) -> None:
        if len(self.operators) == 0:
            raise ValueError("operators must contain at least one detector operator.")
        if len(self.operators) != len(self.names):
            raise ValueError("operators and names must have the same length.")
        if self.terms and len(self.terms) != len(self.operators):
            raise ValueError("terms and operators must have the same length when provided.")

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_operators": len(self.operators),
            "names": self.names,
            "n_terms": len(self.terms),
        }


def _resolve_target_states(
    *,
    target_state: NDArray[np.complex128] | None,
    target_states: NDArray[np.complex128] | None,
    states: NDArray[np.complex128] | None,
) -> NDArray[np.complex128] | None:
    supplied = [value is not None for value in (target_state, target_states, states)]
    if sum(supplied) > 1:
        raise ValueError("Provide only one of target_state, target_states, or states.")
    if target_state is not None:
        return np.asarray(target_state, dtype=np.complex128)
    if target_states is not None:
        return np.asarray(target_states, dtype=np.complex128)
    if states is not None:
        return np.asarray(states, dtype=np.complex128)
    return None


@dataclass(frozen=True, slots=True)
class CageLindbladDesignProblem:
    """Unified cage-state Lindblad design problem.

    The object stores only the target manifold, basis/configuration metadata, and
    local regions needed by the successful dark-detector workflow.  Use
    :func:`build_cage_lindblad_problem` to construct it from either one state,
    many states, or cage records.
    """

    build_result: ModelBuildResult
    construction: DegenerateCageLindbladConstruction

    @property
    def hamiltonian(self) -> Any:
        return self.build_result.hamiltonian

    @property
    def basis_configs(self) -> NDArray[np.integer]:
        return basis_configs_from_build_result(self.build_result)

    @property
    def manifold_basis(self) -> NDArray[np.complex128]:
        return self.construction.manifold_basis

    @property
    def target_basis(self) -> NDArray[np.complex128]:
        return self.construction.manifold_basis

    @property
    def target_density_matrix(self) -> NDArray[np.complex128]:
        return self.construction.target_density_matrix

    @property
    def hilbert_dimension(self) -> int:
        return self.construction.hilbert_dimension

    @property
    def manifold_dimension(self) -> int:
        return self.construction.manifold_dimension

    @property
    def is_single_cage_target(self) -> bool:
        return self.manifold_dimension == 1

    @property
    def local_regions(self) -> tuple[tuple[int, ...], ...]:
        return self.construction.local_regions

    @property
    def record_signature(self) -> tuple[int, int] | None:
        return self.construction.record_signature

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "hilbert_dimension": self.hilbert_dimension,
            "manifold_dimension": self.manifold_dimension,
            "is_single_cage_target": self.is_single_cage_target,
            "record_signature": self.record_signature,
            "n_local_regions": len(self.local_regions),
            "local_regions": self.local_regions,
            "h_closure_residual": self.construction.hamiltonian_closure_residual,
        }

    def to_lindblad_problem(
        self,
        *,
        jumps: Sequence[Any],
        hamiltonian: Any | None = None,
        backend: str | None = None,
    ) -> LindbladProblem:
        """Package a selected jump set as a solver-ready Lindblad problem."""
        return LindbladProblem(
            hamiltonian=self.hamiltonian if hamiltonian is None else hamiltonian,
            jumps=tuple(jumps),
            backend=self.construction.open_system_backend if backend is None else backend,
        )

    def design_jumps(
        self,
        *,
        detector_operators: CageLindbladDetectorOperators | Sequence[Any],
        detector_operator_names: Sequence[str] | None = None,
        hamiltonian: Any | None = None,
        basis_configs: NDArray[np.integer] | None = None,
        **workflow_kwargs: Any,
    ) -> DegenerateCageJumpDesignWorkflowReport:
        """Run the unified dark-detector/recycler workflow.

        ``detector_operators`` may be a :class:`CageLindbladDetectorOperators`
        bundle returned by :func:`build_cage_lindblad_detector_operators` or a
        raw sequence of matrices with optional ``detector_operator_names``.
        Extra keyword arguments are forwarded to the underlying workflow, e.g.
        ``design_mode``, ``max_detectors``, region modes, compression options,
        and H-invariant certificate settings.
        """
        if isinstance(detector_operators, CageLindbladDetectorOperators):
            operators = detector_operators.operators
            names = detector_operators.names
            if detector_operator_names is not None:
                raise ValueError(
                    "detector_operator_names must be omitted when detector_operators "
                    "is a CageLindbladDetectorOperators bundle."
                )
        else:
            operators = tuple(detector_operators)
            names = None if detector_operator_names is None else tuple(detector_operator_names)

        return self.construction.design_dark_manifold_jumps(
            hamiltonian=self.hamiltonian if hamiltonian is None else hamiltonian,
            basis_configs=(self.basis_configs if basis_configs is None else basis_configs),
            detector_operators=operators,
            detector_operator_names=names,
            **workflow_kwargs,
        )


def build_cage_lindblad_problem(
    *,
    build_result: ModelBuildResult,
    target_state: NDArray[np.complex128] | None = None,
    target_states: NDArray[np.complex128] | None = None,
    states: NDArray[np.complex128] | None = None,
    records: Sequence[CageRecord] | None = None,
    model: Any | None = None,
    local_regions: Sequence[Sequence[int]] | None = None,
    local_term_kind: LocalTermKind | None = None,
    region_source: LocalRegionSource = "kinetic",
    validate_record_signature: bool = True,
    open_system_backend: OpenSystemBackendName = "scipy",
    residual_tolerance: float = 1e-10,
    target_tolerance: float = 1e-10,
) -> CageLindbladDesignProblem:
    """Create a unified cage Lindblad design problem.

    A single cage state is supplied with ``target_state``.  A degenerate cage
    manifold is supplied with ``target_states``/``states`` or ``records``.  The
    returned object uses the same ``design_jumps`` method in both cases.
    """
    resolved_states = _resolve_target_states(
        target_state=target_state,
        target_states=target_states,
        states=states,
    )
    if records is not None and resolved_states is not None:
        raise ValueError("Provide records or target states, but not both.")
    if records is None and resolved_states is None:
        raise ValueError("Provide target_state, target_states, states, or records.")

    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        records=records,
        states=resolved_states,
        model=model,
        local_regions=local_regions,
        local_term_kind=local_term_kind,
        region_source=region_source,
        validate_record_signature=validate_record_signature,
        open_system_backend=open_system_backend,
        check_liouvillian=False,
        residual_tolerance=residual_tolerance,
        recycling_rdm_tolerance=target_tolerance,
        recycling_dark_tolerance=target_tolerance,
    )
    return CageLindbladDesignProblem(
        build_result=build_result,
        construction=construction,
    )


def build_cage_lindblad_detector_operators(
    *,
    model: Any,
    build_result: ModelBuildResult,
    term_kind: LocalTermKind | None = "plaquette",
    operator_kind: DetectorOperatorKind = "potential",
    builder: str = "sparse",
    backend: str = "scipy",
    on_missing: str = "skip",
    name_prefix: str | None = None,
) -> CageLindbladDetectorOperators:
    """Build a named local detector-operator family from model local terms.

    ``operator_kind='hamiltonian'`` includes both kinetic and potential terms
    when the model exposes them.  The returned bundle can be passed directly to
    :meth:`CageLindbladDesignProblem.design_jumps`.
    """
    terms = tuple(
        model.local_term_descriptors(
            term_kind=term_kind,
            operator_kind=operator_kind,
        )
    )
    if len(terms) == 0:
        raise ValueError(
            "model.local_term_descriptors returned no detector terms for "
            f"operator_kind={operator_kind!r}."
        )

    matrices: list[Any] = []
    names: list[str] = []
    kept_terms: list[LocalTermDescriptor] = []
    for term in terms:
        try:
            matrix = model.build_local_term(
                term,
                build_result,
                builder=builder,
                backend=backend,
                on_missing=on_missing,
            )
        except TypeError:
            # Older model implementations do not expose every keyword.  Keep the
            # compatibility path local to this API wrapper.
            matrix = model.build_local_term(
                term,
                build_result,
                builder=builder,
                on_missing=on_missing,
            )
        if matrix is None:
            continue
        matrices.append(matrix)
        kept_terms.append(term)
        if term.label:
            label = str(term.label)
        else:
            label = f"{term.operator_kind}_{term.term_id}"
        names.append(label if name_prefix is None else f"{name_prefix}{label}")

    if len(matrices) == 0:
        raise ValueError("All detector local terms were skipped or missing.")

    return CageLindbladDetectorOperators(
        operators=tuple(matrices),
        names=tuple(names),
        terms=tuple(kept_terms),
    )
