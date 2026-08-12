"""Laurent-polynomial constraint-module diagnostics for periodic cages."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import numpy.typing as npt
import scipy.linalg as scipy_linalg

from qlinks.caging.nullspace import as_dense_array
from qlinks.caging.stability_types import (
    LaurentDimensionDivisibilityViolation,
    LaurentPeriodicDimensionConsistencyReport,
    LaurentPolynomialConstraintModuleReport,
    LaurentPolynomialPeriodicPoint,
    LaurentPolynomialRootMode,
    LaurentPolynomialTorsionOrder,
)


def _normalize_laurent_coefficient_terms(
    coefficient_terms: Sequence[tuple[int, object]],
) -> tuple[tuple[int, npt.NDArray[np.complex128]], ...]:
    if not coefficient_terms:
        raise ValueError("coefficient_terms must not be empty.")
    combined: dict[int, npt.NDArray[np.complex128]] = {}
    shape: tuple[int, int] | None = None
    for raw_displacement, raw_matrix in coefficient_terms:
        displacement = int(raw_displacement)
        matrix = np.asarray(as_dense_array(raw_matrix), dtype=np.complex128)
        if matrix.ndim != 2:
            raise ValueError("every Laurent coefficient must be a matrix.")
        if shape is None:
            shape = matrix.shape
        elif matrix.shape != shape:
            raise ValueError("every Laurent coefficient must have the same shape.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Laurent coefficients must contain finite values.")
        if displacement in combined:
            combined[displacement] = combined[displacement] + matrix
        else:
            combined[displacement] = matrix.copy()
    return tuple(sorted(combined.items()))


def laurent_polynomial_constraint_symbol(
    coefficient_terms: Sequence[tuple[int, object]],
    z: complex,
) -> npt.NDArray[np.complex128]:
    """Evaluate ``sum_d z**d B_d`` for one nonzero complex translation value."""
    point = complex(z)
    if not np.isfinite(point.real) or not np.isfinite(point.imag) or point == 0.0:
        raise ValueError("z must be a finite nonzero complex number.")
    normalized = _normalize_laurent_coefficient_terms(coefficient_terms)
    symbol = np.zeros_like(normalized[0][1], dtype=np.complex128)
    for displacement, matrix in normalized:
        symbol += point**displacement * matrix
    return symbol


def _laurent_symbol_rank_gap(
    symbol: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> tuple[int, int, float | None]:
    singular_values = scipy_linalg.svdvals(symbol)
    rank = int(np.sum(singular_values > tolerance))
    nullity = int(symbol.shape[1] - rank)
    positive = singular_values[singular_values > tolerance]
    gap = None if positive.size == 0 else float(np.min(positive))
    return rank, nullity, gap


def _primitive_root_key(momentum_index: int, repeat_count: int) -> tuple[int, int]:
    divisor = int(np.gcd(momentum_index, repeat_count))
    order = repeat_count // divisor
    numerator = (momentum_index // divisor) % order
    return order, numerator


def diagnose_laurent_polynomial_constraint_module(
    coefficient_terms: Sequence[tuple[int, object]],
    repeat_counts: Sequence[int] | npt.NDArray[np.integer],
    *,
    generic_sample_count: int = 12,
    tolerance: float = 1e-10,
) -> LaurentPolynomialConstraintModuleReport:
    """Diagnose free and root-of-unity torsion sectors of ``B(z)``.

    ``generic_sample_count`` non-root-of-unity complex points are used to find
    the maximum symbol rank, which equals the rank over ``C(z)`` away from a
    nongeneric algebraic set.  Periodic kernels are then evaluated exactly at
    every root of unity for the requested repetition counts.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if generic_sample_count < 3:
        raise ValueError("generic_sample_count must be at least three.")
    normalized = _normalize_laurent_coefficient_terms(coefficient_terms)
    counts = tuple(int(value) for value in np.asarray(repeat_counts).reshape(-1))
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("repeat_counts must contain positive values.")
    if len(set(counts)) != len(counts):
        raise ValueError("repeat_counts must not contain duplicates.")

    sample_ranks: list[int] = []
    for sample_index in range(generic_sample_count):
        radius = 0.71 if sample_index % 2 == 0 else 1.37
        angle = np.sqrt(2.0) * (sample_index + 1)
        z = radius * np.exp(1.0j * angle)
        symbol = laurent_polynomial_constraint_symbol(normalized, z)
        rank, _nullity, _gap = _laurent_symbol_rank_gap(symbol, tolerance=tolerance)
        sample_ranks.append(rank)
    generic_rank = max(sample_ranks)
    n_rows, n_columns = normalized[0][1].shape
    free_rank = n_columns - generic_rank
    generic_rank_is_stable = sample_ranks.count(generic_rank) >= generic_sample_count // 2

    periodic_points: list[LaurentPolynomialPeriodicPoint] = []
    root_torsion: dict[tuple[int, int], int] = {}
    for repeat_count in counts:
        root_modes: list[LaurentPolynomialRootMode] = []
        total_kernel = 0
        for momentum_index in range(repeat_count):
            root = np.exp(2.0j * np.pi * momentum_index / repeat_count)
            symbol = laurent_polynomial_constraint_symbol(normalized, root)
            _rank, nullity, gap = _laurent_symbol_rank_gap(symbol, tolerance=tolerance)
            torsion = max(0, nullity - free_rank)
            order, numerator = _primitive_root_key(momentum_index, repeat_count)
            key = (order, numerator)
            previous = root_torsion.get(key)
            if previous is not None and previous != torsion:
                raise ValueError(
                    "inconsistent numerical nullity for the same primitive root; "
                    "increase tolerance or inspect symbol conditioning."
                )
            root_torsion[key] = torsion
            total_kernel += nullity
            root_modes.append(
                LaurentPolynomialRootMode(
                    repeat_count=repeat_count,
                    momentum_index=momentum_index,
                    primitive_order=order,
                    root=complex(root),
                    kernel_dimension=nullity,
                    free_dimension=free_rank,
                    torsion_dimension=torsion,
                    singular_gap=gap,
                )
            )
        periodic_points.append(
            LaurentPolynomialPeriodicPoint(
                repeat_count=repeat_count,
                total_kernel_dimension=total_kernel,
                free_kernel_dimension=repeat_count * free_rank,
                torsion_kernel_dimension=total_kernel - repeat_count * free_rank,
                root_modes=tuple(root_modes),
            )
        )

    by_order: dict[int, list[int]] = defaultdict(list)
    for (order, _numerator), multiplicity in root_torsion.items():
        if multiplicity > 0:
            by_order[order].append(multiplicity)
    torsion_orders = tuple(
        LaurentPolynomialTorsionOrder(
            primitive_order=order,
            multiplicity=int(sum(multiplicities)),
            primitive_root_count=len(multiplicities),
        )
        for order, multiplicities in sorted(by_order.items())
    )
    return LaurentPolynomialConstraintModuleReport(
        n_rows=n_rows,
        n_columns=n_columns,
        displacements=tuple(displacement for displacement, _matrix in normalized),
        generic_rank=generic_rank,
        free_kernel_rank=free_rank,
        generic_rank_sample_count=generic_sample_count,
        generic_rank_is_stable=generic_rank_is_stable,
        periodic_points=tuple(periodic_points),
        torsion_orders=torsion_orders,
        tolerance=tolerance,
    )


def diagnose_laurent_periodic_dimension_consistency(
    repeat_counts: Sequence[int] | npt.NDArray[np.integer],
    observed_dimensions: Sequence[int] | npt.NDArray[np.integer],
    *,
    assumed_free_rank: int = 0,
) -> LaurentPeriodicDimensionConsistencyReport:
    """Test necessary fixed-symbol constraints on observed periodic dimensions."""
    counts = tuple(int(value) for value in np.asarray(repeat_counts).reshape(-1))
    dimensions = tuple(int(value) for value in np.asarray(observed_dimensions).reshape(-1))
    if not counts or len(counts) != len(dimensions):
        raise ValueError("repeat_counts and observed_dimensions must have equal nonzero length.")
    if any(value <= 0 for value in counts):
        raise ValueError("repeat_counts must be positive.")
    if any(value < 0 for value in dimensions):
        raise ValueError("observed_dimensions must be non-negative.")
    if len(set(counts)) != len(counts):
        raise ValueError("repeat_counts must not contain duplicates.")
    if assumed_free_rank < 0:
        raise ValueError("assumed_free_rank must be non-negative.")

    ordered = sorted(zip(counts, dimensions, strict=True))
    counts = tuple(item[0] for item in ordered)
    dimensions = tuple(item[1] for item in ordered)
    torsion = tuple(
        dimension - repeat_count * assumed_free_rank
        for repeat_count, dimension in zip(counts, dimensions, strict=True)
    )
    torsion_by_count = dict(zip(counts, torsion, strict=True))

    violations: list[LaurentDimensionDivisibilityViolation] = []
    for divisor_count, divisor_torsion in zip(counts, torsion, strict=True):
        for multiple_count, multiple_torsion in zip(counts, torsion, strict=True):
            if multiple_count <= divisor_count or multiple_count % divisor_count:
                continue
            if multiple_torsion < divisor_torsion:
                violations.append(
                    LaurentDimensionDivisibilityViolation(
                        divisor_repeat_count=divisor_count,
                        multiple_repeat_count=multiple_count,
                        divisor_torsion_dimension=divisor_torsion,
                        multiple_torsion_dimension=multiple_torsion,
                    )
                )

    primitive: dict[int, int] = {}
    incomplete: list[int] = []
    for repeat_count in counts:
        proper_divisors = tuple(
            divisor for divisor in range(1, repeat_count) if repeat_count % divisor == 0
        )
        if any(divisor not in torsion_by_count for divisor in proper_divisors):
            incomplete.append(repeat_count)
            continue
        primitive[repeat_count] = torsion_by_count[repeat_count] - sum(
            primitive[divisor] for divisor in proper_divisors
        )

    return LaurentPeriodicDimensionConsistencyReport(
        repeat_counts=counts,
        observed_dimensions=dimensions,
        assumed_free_rank=int(assumed_free_rank),
        torsion_dimensions=torsion,
        divisibility_violations=tuple(violations),
        primitive_order_multiplicities=tuple(sorted(primitive.items())),
        incomplete_primitive_orders=tuple(incomplete),
    )
