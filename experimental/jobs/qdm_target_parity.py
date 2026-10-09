"""Measure compact-target parity without enumerating the complete dimer Hilbert space."""

from __future__ import annotations

from itertools import product

import numpy as np
from qdm_checkerboard_large_strip import (
    materialize_periodic_product_state_from_basis,
    packed_binary_basis_index,
)
from qdm_checkerboard_symmetry import (
    checkerboard_target_sector,
    square_qdm_variable_permutation_from_site_map,
)

from qlinks.basis import Basis


def compact_target_parity(instance, *, repeats):
    """Close only the finite product support under translations and reflections.

    This basis is sufficient to measure the translation-projected target irrep.
    Its size is not the full Hamiltonian sector dimension or a spectral estimate.
    """
    model = instance.model
    support = []
    for indices in product(*(range(block.support_size) for block in instance.blocks)):
        config = np.zeros(model.layout.n_variables, dtype=int)
        config[np.asarray(instance.padding.exterior_link_ids, dtype=int)] = (
            instance.padding.exterior_config
        )
        for block, index in zip(instance.blocks, indices, strict=True):
            config[np.asarray(block.link_ids, dtype=int)] = block.support_configs[index]
        support.append(config)
    support = np.asarray(support)
    orbit = []
    for dx, dy, sign_x, sign_y in product(range(model.lx), range(4), (-1, 1), (-1, 1)):
        permutation = square_qdm_variable_permutation_from_site_map(
            model,
            lambda x, y, dx=dx, dy=dy, sign_x=sign_x, sign_y=sign_y: (
                sign_x * x + dx,
                sign_y * y + dy,
            ),
        )
        orbit.append(support[:, permutation])
    basis = Basis.from_states(model.layout, np.unique(np.vstack(orbit), axis=0), sort=True)
    target = materialize_periodic_product_state_from_basis(instance, basis)
    resolved, _ = checkerboard_target_sector(
        model,
        basis,
        packed_index=packed_binary_basis_index(basis),
        repeats=repeats,
        target_state=target,
    )
    return {
        "Lx": model.lx,
        "Sy": resolved.sector.labels["Sy_character"],
        "target_Sy_residual": resolved.sector.labels["target_Sy_residual"],
        "target_Sy_expectation": resolved.sector.labels["target_Sy_expectation"],
        "support_orbit_size": basis.n_states,
        "scope": "exact target irrep on closed support orbit; no full spectral dimension",
    }
