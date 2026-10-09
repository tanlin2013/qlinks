"""Exact torus irrep dimensions from fixed dimer coverings and character traces.

Tr P_chi = sum_g chi(g)* N_fixed(g)/|G|. In the selected translation
irrep, dim(Sy=+/-) = (Tr P_chi +/- Tr Sy P_chi)/2. Coverings fixed by
an operation occupy entire link orbits. Dynamic programming counts exact
covers of sites with signed electric cut winding zero; no basis or H is built.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from qdm_checkerboard_symmetry import square_qdm_variable_permutation_from_site_map

from qlinks.constraints.winding import SquareQDMElectricWindingSector
from qlinks.models import SquareQDMModel


def _fixed_coverings(model, permutation):
    weights = []
    for direction in ("x", "y"):
        cut = SquareQDMElectricWindingSector.cut_data(
            layout=model.layout, lattice=model.lattice, direction=direction
        )
        weight = np.zeros(model.layout.n_variables, dtype=int)
        weight[cut.variable_indices] = cut.signs
        # W = sum eta(2n-1); balanced cuts make W=0 equivalent to sum eta*n=0.
        if weight.sum() != 0:
            raise ValueError("character counter requires balanced staggered winding cuts")
        weights.append(weight)
    masks = [0] * model.layout.n_variables
    for link in model.lattice.links:
        mask = 0
        for site_id in (link.source, link.target):
            x, y = model.lattice.sites[site_id].cell
            mask |= 1 << (int(x) * 4 + int(y))
        masks[model.layout.link_variable_index(link.id)] = mask
    seen, orbits = set(), []
    for i in range(len(masks)):
        if i in seen:
            continue
        orbit, j = [], i
        while j not in seen:
            seen.add(j)
            orbit.append(j)
            j = int(permutation[j])
        mask = 0
        for j in orbit:
            if mask & masks[j]:
                break  # This occupied orbit would touch a site twice.
            mask |= masks[j]
        else:
            orbits.append((mask, *(int(weight[orbit].sum()) for weight in weights)))
    options = [[] for _ in range(model.lx * 4)]
    for orbit in orbits:
        for vertex in range(model.lx * 4):
            if orbit[0] & (1 << vertex):
                options[vertex].append(orbit)
    all_sites = (1 << (model.lx * 4)) - 1

    @lru_cache(None)
    def visit(mask, wx, wy):
        if mask == all_sites:
            return int(wx == wy == 0)
        remaining = all_sites ^ mask
        vertex = (remaining & -remaining).bit_length() - 1
        return sum(
            visit(mask | bits, wx + a, wy + b) for bits, a, b in options[vertex] if not mask & bits
        )

    return visit(0, 0, 0)


def checkerboard_character_dimensions(lx):
    """Count both Sy dimensions exactly, independently of sparse basis construction."""
    if lx not in (4, 8, 12):
        raise ValueError("bounded character counter supports Lx=4,8,12 and Ly=4")
    model = SquareQDMModel(
        lx=lx,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
    )
    real, imag, details = [0, 0], [0, 0], []
    for mirror in (False, True):
        for a in range(lx):
            for b in (0, 1):
                permutation = square_qdm_variable_permutation_from_site_map(
                    model,
                    lambda x, y, a=a, b=b, mirror=mirror: (
                        (x + a, 1 - (y + a + 2 * b)) if mirror else (x + a, y + a + 2 * b)
                    ),
                )
                count = _fixed_coverings(model, permutation)
                real[int(mirror)] += (1, 0, -1, 0)[a % 4] * count
                imag[int(mirror)] += (0, -1, 0, 1)[a % 4] * count
                details.append({"mirror": mirror, "a": a, "b": b, "fixed_count": count})
    denominator = 2 * lx
    if any(imag) or any(value % denominator for value in real):
        raise RuntimeError("character trace is not a real integer")
    legacy, trace = (value // denominator for value in real)
    if (legacy + trace) % 2 or abs(trace) > legacy:
        raise RuntimeError("invalid Sy character multiplicities")
    return {
        "Lx": lx,
        "translation_dimension": legacy,
        "Sy_projected_trace": trace,
        "Sy_dimensions": {"1": (legacy + trace) // 2, "-1": (legacy - trace) // 2},
        "fixed_counts": details,
        "method": "exact_cover_integer_character_trace",
    }
