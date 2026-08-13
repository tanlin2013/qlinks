"""Exact factorized-product certification for separated QDM cage blocks."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from qlinks.caging.local_search_global import _config_key, _qdm_global_plaquette_actions
from qlinks.caging.local_search_padding import (
    _factorized_padding_validation_reason,
    factorized_qdm_padding_from_multi_padding,
)
from qlinks.caging.local_search_qdm import _infer_potential_unit_from_model
from qlinks.caging.local_search_types import (
    FactorizedLocalQDMPadding,
    LocalQDMCageBlock,
    LocalQDMMultiPaddingConfig,
    MultiLocalQDMPadding,
    QDMFactorizedProductCertificationReport,
    _FactorizedProductTerm,
    _QDMGlobalPlaquetteAction,
)
from qlinks.caging.search import signature_from_energy_and_self_loop


def _sparse_factor_inner_product(
    left: dict[tuple[int, ...], complex],
    right: dict[tuple[int, ...], complex],
) -> complex:
    if len(left) > len(right):
        left, right = right, left
        return np.conj(_sparse_factor_inner_product(left, right))
    return sum(np.conj(value) * right.get(key, 0.0 + 0.0j) for key, value in left.items())


def _factorized_product_inner_product(
    left: _FactorizedProductTerm,
    right: _FactorizedProductTerm,
) -> complex:
    if len(left.factors) != len(right.factors):
        raise ValueError("factorized product terms must have the same number of factors.")
    value = np.conj(left.coefficient) * right.coefficient
    for left_factor, right_factor in zip(left.factors, right.factors, strict=True):
        value *= _sparse_factor_inner_product(left_factor, right_factor)
        if value == 0.0:
            break
    return complex(value)


def _factorized_sum_norm(terms: Sequence[_FactorizedProductTerm]) -> float:
    contributions: list[complex] = []
    for left_index, left in enumerate(terms):
        contributions.append(_factorized_product_inner_product(left, left))
        for right in terms[left_index + 1 :]:
            overlap = _factorized_product_inner_product(left, right)
            contributions.extend((overlap, np.conj(overlap)))

    real_value = math.fsum(float(np.real(value)) for value in contributions)
    imaginary_value = math.fsum(float(np.imag(value)) for value in contributions)
    absolute_scale = math.fsum(abs(value) for value in contributions)
    roundoff_bound = 128.0 * np.finfo(np.float64).eps * max(absolute_scale, 1.0)
    if abs(real_value) <= roundoff_bound and abs(imaginary_value) <= roundoff_bound:
        return 0.0
    if real_value < -roundoff_bound:
        raise ArithmeticError("factorized norm contraction produced a negative norm square.")
    return float(np.sqrt(max(real_value, 0.0)))


def _factorized_sum_expectation(
    reference: _FactorizedProductTerm,
    terms: Sequence[_FactorizedProductTerm],
) -> complex:
    return complex(sum(_factorized_product_inner_product(reference, term) for term in terms))


def _factorized_block_state_factor(
    block: LocalQDMCageBlock,
) -> dict[tuple[int, ...], complex]:
    factor: dict[tuple[int, ...], complex] = defaultdict(complex)
    for config, amplitude in zip(block.support_configs, block.amplitudes, strict=True):
        factor[_config_key(config)] += complex(amplitude)
    return {key: value for key, value in factor.items() if value != 0.0}


def _factorized_reference_term(
    blocks: Sequence[LocalQDMCageBlock],
    padding: FactorizedLocalQDMPadding,
) -> _FactorizedProductTerm:
    block_factors = tuple(_factorized_block_state_factor(block) for block in blocks)
    exterior_factor = {_config_key(padding.exterior_config): 1.0 + 0.0j}
    return _FactorizedProductTerm(
        coefficient=1.0 + 0.0j,
        factors=block_factors + (exterior_factor,),
    )


def _factorized_action_context(
    blocks: Sequence[LocalQDMCageBlock],
    padding: FactorizedLocalQDMPadding,
) -> tuple[dict[int, tuple[int, int]], dict[int, int]]:
    block_position_by_link: dict[int, tuple[int, int]] = {}
    for block_index, block in enumerate(blocks):
        for local_index, link_id in enumerate(block.link_ids):
            block_position_by_link[int(link_id)] = (int(block_index), int(local_index))
    exterior_position_by_link = {
        int(link_id): int(index) for index, link_id in enumerate(padding.exterior_link_ids)
    }
    return block_position_by_link, exterior_position_by_link


def _factorized_pattern_matches(
    *,
    action: _QDMGlobalPlaquetteAction,
    pattern: npt.NDArray[np.int64],
    block_config: tuple[int, ...] | None,
    block_index: int | None,
    block_position_by_link: dict[int, tuple[int, int]],
    exterior_config: npt.NDArray[np.int64],
    exterior_position_by_link: dict[int, int],
) -> bool:
    for action_position, link_id_raw in enumerate(action.links):
        link_id = int(link_id_raw)
        owner = block_position_by_link.get(link_id)
        if owner is None:
            value = int(exterior_config[exterior_position_by_link[link_id]])
        else:
            owner_index, local_index = owner
            if block_index is None or owner_index != block_index or block_config is None:
                raise ValueError("inconsistent block ownership in factorized action.")
            value = int(block_config[local_index])
        if value != int(pattern[action_position]):
            return False
    return True


def _factorized_updated_outputs(
    *,
    action: _QDMGlobalPlaquetteAction,
    target_pattern: npt.NDArray[np.int64],
    block_config: tuple[int, ...] | None,
    block_index: int | None,
    block_position_by_link: dict[int, tuple[int, int]],
    exterior_config: npt.NDArray[np.int64],
    exterior_position_by_link: dict[int, int],
) -> tuple[tuple[int, ...] | None, tuple[int, ...]]:
    updated_block = None if block_config is None else list(block_config)
    updated_exterior = np.asarray(exterior_config, dtype=np.int64).copy()
    for action_position, link_id_raw in enumerate(action.links):
        link_id = int(link_id_raw)
        owner = block_position_by_link.get(link_id)
        target_value = int(target_pattern[action_position])
        if owner is None:
            updated_exterior[exterior_position_by_link[link_id]] = target_value
        else:
            owner_index, local_index = owner
            if block_index is None or owner_index != block_index or updated_block is None:
                raise ValueError("inconsistent block ownership in factorized output.")
            updated_block[local_index] = target_value
    return (
        None if updated_block is None else tuple(int(value) for value in updated_block),
        _config_key(updated_exterior),
    )


def _factorized_kinetic_terms_for_action(
    action: _QDMGlobalPlaquetteAction,
    *,
    padding: FactorizedLocalQDMPadding,
    reference: _FactorizedProductTerm,
    block_position_by_link: dict[int, tuple[int, int]],
    exterior_position_by_link: dict[int, int],
) -> list[_FactorizedProductTerm]:
    owners = {
        block_position_by_link[int(link_id)][0]
        for link_id in action.links
        if int(link_id) in block_position_by_link
    }
    if len(owners) > 1:
        raise ValueError("factorized certification requires kinetic separation.")
    block_index = next(iter(owners)) if owners else None

    terms: list[_FactorizedProductTerm] = []
    directions = (
        (action.pattern0, action.pattern1, action.forward),
        (action.pattern1, action.pattern0, action.backward),
    )
    if block_index is None:
        for source_pattern, target_pattern, coefficient in directions:
            if coefficient == 0.0:
                continue
            if not _factorized_pattern_matches(
                action=action,
                pattern=source_pattern,
                block_config=None,
                block_index=None,
                block_position_by_link=block_position_by_link,
                exterior_config=padding.exterior_config,
                exterior_position_by_link=exterior_position_by_link,
            ):
                continue
            _unused_block, exterior_output = _factorized_updated_outputs(
                action=action,
                target_pattern=target_pattern,
                block_config=None,
                block_index=None,
                block_position_by_link=block_position_by_link,
                exterior_config=padding.exterior_config,
                exterior_position_by_link=exterior_position_by_link,
            )
            factors = reference.factors[:-1] + ({exterior_output: 1.0 + 0.0j},)
            terms.append(_FactorizedProductTerm(coefficient=coefficient, factors=factors))
        return terms

    source_factor = reference.factors[block_index]
    for source_pattern, target_pattern, coefficient in directions:
        if coefficient == 0.0:
            continue
        output_factor: dict[tuple[int, ...], complex] = defaultdict(complex)
        exterior_output: tuple[int, ...] | None = None
        for block_config, amplitude in source_factor.items():
            if not _factorized_pattern_matches(
                action=action,
                pattern=source_pattern,
                block_config=block_config,
                block_index=block_index,
                block_position_by_link=block_position_by_link,
                exterior_config=padding.exterior_config,
                exterior_position_by_link=exterior_position_by_link,
            ):
                continue
            block_output, current_exterior = _factorized_updated_outputs(
                action=action,
                target_pattern=target_pattern,
                block_config=block_config,
                block_index=block_index,
                block_position_by_link=block_position_by_link,
                exterior_config=padding.exterior_config,
                exterior_position_by_link=exterior_position_by_link,
            )
            if block_output is None:
                raise ValueError("missing block output for a block-touching action.")
            output_factor[block_output] += amplitude
            if exterior_output is None:
                exterior_output = current_exterior
            elif exterior_output != current_exterior:
                raise ValueError("one kinetic direction produced inconsistent exterior outputs.")
        if not output_factor or exterior_output is None:
            continue
        factors = list(reference.factors)
        factors[block_index] = dict(output_factor)
        factors[-1] = {exterior_output: 1.0 + 0.0j}
        terms.append(
            _FactorizedProductTerm(
                coefficient=coefficient,
                factors=tuple(factors),
            )
        )
    return terms


def _factorized_potential_term_for_action(
    action: _QDMGlobalPlaquetteAction,
    *,
    padding: FactorizedLocalQDMPadding,
    reference: _FactorizedProductTerm,
    block_position_by_link: dict[int, tuple[int, int]],
    exterior_position_by_link: dict[int, int],
) -> _FactorizedProductTerm | None:
    if action.potential == 0.0:
        return None
    owners = {
        block_position_by_link[int(link_id)][0]
        for link_id in action.links
        if int(link_id) in block_position_by_link
    }
    if len(owners) > 1:
        raise ValueError("factorized certification requires kinetic separation.")
    block_index = next(iter(owners)) if owners else None

    if block_index is None:
        flippable = any(
            _factorized_pattern_matches(
                action=action,
                pattern=pattern,
                block_config=None,
                block_index=None,
                block_position_by_link=block_position_by_link,
                exterior_config=padding.exterior_config,
                exterior_position_by_link=exterior_position_by_link,
            )
            for pattern in (action.pattern0, action.pattern1)
        )
        if not flippable:
            return None
        return _FactorizedProductTerm(
            coefficient=action.potential,
            factors=reference.factors,
        )

    source_factor = reference.factors[block_index]
    output_factor: dict[tuple[int, ...], complex] = {}
    for block_config, amplitude in source_factor.items():
        flippable = any(
            _factorized_pattern_matches(
                action=action,
                pattern=pattern,
                block_config=block_config,
                block_index=block_index,
                block_position_by_link=block_position_by_link,
                exterior_config=padding.exterior_config,
                exterior_position_by_link=exterior_position_by_link,
            )
            for pattern in (action.pattern0, action.pattern1)
        )
        if flippable:
            output_factor[block_config] = amplitude
    if not output_factor:
        return None
    factors = list(reference.factors)
    factors[block_index] = output_factor
    return _FactorizedProductTerm(
        coefficient=action.potential,
        factors=tuple(factors),
    )


def _factorized_eigen_residual(
    action_terms: Sequence[_FactorizedProductTerm],
    *,
    reference: _FactorizedProductTerm,
    eigenvalue: complex,
) -> float:
    residual_terms = list(action_terms)
    residual_terms.append(
        _FactorizedProductTerm(
            coefficient=-complex(eigenvalue),
            factors=reference.factors,
        )
    )
    return _factorized_sum_norm(residual_terms)


def certify_qdm_factorized_product_state(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    padding: FactorizedLocalQDMPadding | MultiLocalQDMPadding,
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
) -> QDMFactorizedProductCertificationReport:
    """Certify a separated product cage without forming its global support.

    The Hamiltonian action is represented as a sum of tensor-product vectors.
    Norms and expectation values are evaluated by factor contractions.  The
    cost is polynomial in the number of blocks and plaquettes and exponential
    only in the largest *single-block* support, rather than in the product of
    all block support sizes.

    Exact factorization currently requires every plaquette to touch at most one
    selected block.  This is precisely the kinetic-separation condition used by
    the strict multi-padding workflow.
    """
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    factorized_padding = (
        factorized_qdm_padding_from_multi_padding(padding)
        if isinstance(padding, MultiLocalQDMPadding)
        else padding
    )
    fixed_blocks = tuple(blocks)
    failure_reason, sector_validation, max_touched = _factorized_padding_validation_reason(
        model,
        fixed_blocks,
        factorized_padding,
        multi_config,
    )
    support_size = int(np.prod([block.support_size for block in fixed_blocks], dtype=object))
    if failure_reason is not None:
        return QDMFactorizedProductCertificationReport(
            block_ids=tuple(int(block.block_id) for block in fixed_blocks),
            padding=factorized_padding,
            support_size=support_size,
            kinetic_eigenvalue=0.0 + 0.0j,
            self_loop_value=0.0 + 0.0j,
            energy=0.0 + 0.0j,
            kinetic_residual=float("inf"),
            potential_residual=float("inf"),
            hamiltonian_residual=float("inf"),
            signature=None,
            n_kinetic_product_terms=0,
            n_potential_product_terms=0,
            max_blocks_touched_by_plaquette=max_touched,
            sector_validation=sector_validation,
            failure_reason=failure_reason,
        )

    reference = _factorized_reference_term(fixed_blocks, factorized_padding)
    block_position_by_link, exterior_position_by_link = _factorized_action_context(
        fixed_blocks,
        factorized_padding,
    )
    kinetic_terms: list[_FactorizedProductTerm] = []
    potential_terms: list[_FactorizedProductTerm] = []
    for action in _qdm_global_plaquette_actions(model):
        kinetic_terms.extend(
            _factorized_kinetic_terms_for_action(
                action,
                padding=factorized_padding,
                reference=reference,
                block_position_by_link=block_position_by_link,
                exterior_position_by_link=exterior_position_by_link,
            )
        )
        potential_term = _factorized_potential_term_for_action(
            action,
            padding=factorized_padding,
            reference=reference,
            block_position_by_link=block_position_by_link,
            exterior_position_by_link=exterior_position_by_link,
        )
        if potential_term is not None:
            potential_terms.append(potential_term)

    kinetic_eigenvalue = _factorized_sum_expectation(reference, kinetic_terms)
    self_loop_value = _factorized_sum_expectation(reference, potential_terms)
    energy = kinetic_eigenvalue + self_loop_value
    kinetic_residual = _factorized_eigen_residual(
        kinetic_terms,
        reference=reference,
        eigenvalue=kinetic_eigenvalue,
    )
    potential_residual = _factorized_eigen_residual(
        potential_terms,
        reference=reference,
        eigenvalue=self_loop_value,
    )
    hamiltonian_residual = _factorized_eigen_residual(
        tuple(kinetic_terms) + tuple(potential_terms),
        reference=reference,
        eigenvalue=energy,
    )

    signature = signature_from_energy_and_self_loop(
        energy,
        self_loop_value,
        tolerance=max(multi_config.tolerance, 1.0e-15) * 10.0,
        potential_unit=_infer_potential_unit_from_model(model),
    )
    residual_failure = None
    if kinetic_residual > multi_config.tolerance:
        residual_failure = "kinetic_residual"
    elif potential_residual > multi_config.tolerance:
        residual_failure = "potential_residual"
    elif hamiltonian_residual > multi_config.tolerance:
        residual_failure = "hamiltonian_residual"
    elif signature is None:
        residual_failure = "signature_inference_failed"

    return QDMFactorizedProductCertificationReport(
        block_ids=tuple(int(block.block_id) for block in fixed_blocks),
        padding=factorized_padding,
        support_size=support_size,
        kinetic_eigenvalue=kinetic_eigenvalue,
        self_loop_value=self_loop_value,
        energy=energy,
        kinetic_residual=kinetic_residual,
        potential_residual=potential_residual,
        hamiltonian_residual=hamiltonian_residual,
        signature=signature,
        n_kinetic_product_terms=len(kinetic_terms),
        n_potential_product_terms=len(potential_terms),
        max_blocks_touched_by_plaquette=max_touched,
        sector_validation=sector_validation,
        failure_reason=residual_failure,
    )
