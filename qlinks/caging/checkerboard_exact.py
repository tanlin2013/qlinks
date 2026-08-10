"""Exact local theorem for the 4N x 4 periodic-product checkerboard cage.

The certificate here is deliberately algebraic rather than spectral.  It uses
only binary plaquette patterns, integer relative signs, and the symbolic
checkerboard phase exponent.  No Hamiltonian diagonalization and no floating
point eigenpair residual enter the proof.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .periodic_sequence import SquareQDMPeriodicProductUnitCell

_FLIP_A = np.asarray([1, 0, 1, 0], dtype=np.int64)
_FLIP_B = np.asarray([0, 1, 0, 1], dtype=np.int64)


@dataclass(frozen=True, slots=True)
class CheckerboardExactPeriodicProductCertificate:
    period_x: int
    circumference: int
    active_plaquette_columns: tuple[int, ...]
    inactive_plaquette_columns: tuple[int, ...]
    support_size_per_period: int
    exact_relative_signs: tuple[tuple[int, ...], ...]
    flippable_plaquettes_per_support_state: int
    kinetic_symbolic_residual_terms: int
    boundary_inactive_exact: bool
    checkerboard_phase_pairs_exact: bool
    exact_for_all_positive_repeats: bool
    unit_cell_energy_per_lambda: int
    proof_statement: str

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "period_x": self.period_x,
            "circumference": self.circumference,
            "active_plaquette_columns": self.active_plaquette_columns,
            "inactive_plaquette_columns": self.inactive_plaquette_columns,
            "support_size_per_period": self.support_size_per_period,
            "exact_relative_signs": self.exact_relative_signs,
            "flippable_plaquettes_per_support_state": self.flippable_plaquettes_per_support_state,
            "kinetic_symbolic_residual_terms": self.kinetic_symbolic_residual_terms,
            "boundary_inactive_exact": self.boundary_inactive_exact,
            "checkerboard_phase_pairs_exact": self.checkerboard_phase_pairs_exact,
            "exact_for_all_positive_repeats": self.exact_for_all_positive_repeats,
            "unit_cell_energy_per_lambda": self.unit_cell_energy_per_lambda,
            "proof_statement": self.proof_statement,
        }


def _complete_support_configs(
    unit_cell: SquareQDMPeriodicProductUnitCell, block_signs: tuple[tuple[int, ...], ...]
):
    exterior = {
        int(link_id): int(value)
        for link_id, value in zip(
            unit_cell.padding.exterior_link_ids,
            unit_cell.padding.exterior_config,
            strict=True,
        )
    }
    rows: list[tuple[np.ndarray, int]] = []
    for support_indices in itertools.product(
        *(range(block.support_size) for block in unit_cell.blocks)
    ):
        config = np.zeros(unit_cell.model.lattice.num_links, dtype=np.int64)
        for link_id, value in exterior.items():
            config[int(link_id)] = int(value)
        sign = 1
        for block_index, (block, support_index) in enumerate(
            zip(unit_cell.blocks, support_indices, strict=True)
        ):
            config[np.asarray(block.link_ids, dtype=np.int64)] = block.support_configs[
                support_index
            ]
            sign *= int(block_signs[block_index][support_index])
        rows.append((config, sign))
    return rows


def _checkerboard_sign(model: object, plaquette_id: int) -> int:
    x, y = model.lattice.plaquette_anchor_cell(int(plaquette_id))
    return 1 if (int(x) + int(y)) % 2 == 0 else -1


def _symbolic_kinetic_residual(
    unit_cell: SquareQDMPeriodicProductUnitCell, block_signs: tuple[tuple[int, ...], ...]
):
    """Return exact integer coefficients keyed by (child configuration, phase exponent)."""

    residual: dict[tuple[tuple[int, ...], int], int] = {}
    flippable_counts: list[int] = []
    flippable_columns: set[int] = set()
    for config, source_sign in _complete_support_configs(unit_cell, block_signs):
        count = 0
        for plaquette_id in unit_cell.model.plaquette_ids():
            links = unit_cell.model.lattice.plaquette_links(int(plaquette_id))
            variables = np.asarray(
                [unit_cell.model.layout.link_variable_index(int(link_id)) for link_id in links],
                dtype=np.int64,
            )
            local = config[variables]
            if np.array_equal(local, _FLIP_A):
                child = config.copy()
                child[variables] = _FLIP_B
                exponent = _checkerboard_sign(unit_cell.model, int(plaquette_id))
            elif np.array_equal(local, _FLIP_B):
                child = config.copy()
                child[variables] = _FLIP_A
                exponent = -_checkerboard_sign(unit_cell.model, int(plaquette_id))
            else:
                continue
            count += 1
            x, _ = unit_cell.model.lattice.plaquette_anchor_cell(int(plaquette_id))
            flippable_columns.add(int(x) % int(unit_cell.model.lx))
            key = (tuple(int(value) for value in child), int(exponent))
            residual[key] = residual.get(key, 0) + int(source_sign)
        flippable_counts.append(count)
    return residual, flippable_counts, flippable_columns


def certify_checkerboard_periodic_product_exact(
    unit_cell: SquareQDMPeriodicProductUnitCell,
) -> CheckerboardExactPeriodicProductCertificate:
    """Certify the discovered four-column motif for every ``4N x 4`` ring.

    The proof searches only the discrete relative signs of the two-state local
    factors.  A successful certificate establishes cancellation as an identity
    in the formal phase monomials ``exp(+/- i phi)``.  Hence the conclusion is
    exact for arbitrary real checkerboard phase and every positive repeat count.
    """

    model = unit_cell.model
    if int(model.lx) != 4 or int(model.ly) != 4 or unit_cell.repeat_axis != "x":
        raise ValueError("exact checkerboard theorem expects the discovered 4x4 x-repeat motif")
    if any(block.support_size != 2 for block in unit_cell.blocks):
        raise ValueError("exact checkerboard theorem expects two-state coherent factors")

    chosen_signs: tuple[tuple[int, ...], ...] | None = None
    chosen_counts: list[int] | None = None
    chosen_columns: set[int] | None = None
    for relative in itertools.product((-1, 1), repeat=len(unit_cell.blocks)):
        signs = tuple((1, int(value)) for value in relative)
        residual, counts, columns = _symbolic_kinetic_residual(unit_cell, signs)
        if all(int(value) == 0 for value in residual.values()) and len(set(counts)) == 1:
            chosen_signs = signs
            chosen_counts = counts
            chosen_columns = columns
            break
    if chosen_signs is None or chosen_counts is None or chosen_columns is None:
        raise RuntimeError("no exact signed local product cancels the checkerboard kinetic action")

    active_columns = tuple(sorted(int(value) for value in chosen_columns))
    inactive_columns = tuple(value for value in range(4) if value not in chosen_columns)
    boundary_inactive = 3 in inactive_columns and 1 in inactive_columns
    checkerboard_pairs = all(
        _checkerboard_sign(model, model.lattice.plaquette_id_from_cell(x, y))
        == _checkerboard_sign(model, model.lattice.plaquette_id_from_cell(x, (y + 2) % 4))
        for x in active_columns
        for y in (0, 1)
    )
    unit_energy = int(chosen_counts[0])
    exact = bool(
        active_columns == (0, 2)
        and inactive_columns == (1, 3)
        and boundary_inactive
        and checkerboard_pairs
        and unit_energy == 4
    )
    statement = (
        "In one four-column motif only plaquette columns x=0 and x=2 are flippable. "
        "The x=1 and x=3 columns are inactive for every product-support configuration, "
        "so no plaquette term couples neighboring four-column motifs.  On each active "
        "column the two local support configurations have opposite exact signs; every "
        "kinetic child receives paired contributions with the same checkerboard phase "
        "because chi(x,y+2)=chi(x,y), and their integer coefficients cancel identically. "
        "Every support configuration has exactly four flippable plaquettes, so the "
        "potential is 4*lambda on one motif.  Therefore the N-fold periodic product is "
        "an exact eigenstate on every 4N x 4 torus, for arbitrary real phi, with "
        "E_N=4N*lambda=L_x*lambda and e=E/(4L_x)=lambda/4."
    )
    return CheckerboardExactPeriodicProductCertificate(
        period_x=4,
        circumference=4,
        active_plaquette_columns=active_columns,
        inactive_plaquette_columns=inactive_columns,
        support_size_per_period=4,
        exact_relative_signs=chosen_signs,
        flippable_plaquettes_per_support_state=unit_energy,
        kinetic_symbolic_residual_terms=0,
        boundary_inactive_exact=boundary_inactive,
        checkerboard_phase_pairs_exact=checkerboard_pairs,
        exact_for_all_positive_repeats=exact,
        unit_cell_energy_per_lambda=unit_energy,
        proof_statement=statement,
    )
