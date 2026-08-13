"""QDM padding, global certification, and factorized-product validation."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt
import scipy.sparse as scipy_sparse

from qlinks.basis import Basis
from qlinks.caging.candidate import CandidateSubgraph
from qlinks.caging.local_search_core import LocalQDMCageSearchResult
from qlinks.caging.local_search_geometry import _unique_int_array, _validate_plaquette_ids
from qlinks.caging.local_search_qdm import (
    _backward_coefficient,
    _forward_coefficient,
    _infer_potential_unit_from_model,
)
from qlinks.caging.local_search_types import (
    CertifiedLocalQDMCageSearchResult,
    FactorizedLocalQDMPadding,
    LocalQDMCageBlock,
    LocalQDMCageRecord,
    LocalQDMCageSearchConfig,
    LocalQDMCertificationReport,
    LocalQDMMultiPaddingConfig,
    LocalQDMPadding,
    LocalQDMPaddingConfig,
    MultiLocalQDMCertificationReport,
    MultiLocalQDMPadding,
    QDMFactorizedProductCertificationReport,
    QDMMultiPaddingDiagnostics,
    QDMMultiPaddingFailureReport,
    _FactorizedProductTerm,
    _QDMExteriorFlippabilityPreference,
    _QDMExteriorStaticPlaquette,
    _QDMGlobalPlaquetteAction,
)
from qlinks.caging.results import CageState
from qlinks.caging.search import (
    CageRecord,
    CageSearchConfig,
    CageSearchResult,
    signature_from_energy_and_self_loop,
)
from qlinks.operators.plaquette import alternating_binary_patterns


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


def certify_qdm_local_result(
    model: object,
    local_result: LocalQDMCageSearchResult,
    *,
    config: LocalQDMPaddingConfig | None = None,
) -> CertifiedLocalQDMCageSearchResult:
    """Pad and certify all local QDM records without a full basis/Hamiltonian.

    The certification uses a limited global basis made from the union of each
    certified support and its one-hop kinetic shell.  It returns ordinary
    ``CageRecord`` objects inside a ``CageSearchResult`` so downstream code that
    only depends on the cage-result protocol can consume the output.
    """
    padding_config = LocalQDMPaddingConfig() if config is None else config

    certified_items: list[tuple[LocalQDMCageRecord, LocalQDMCertificationReport]] = []
    limited_config_keys: set[tuple[int, ...]] = set()

    for local_record_index, local_record in enumerate(local_result.records):
        reports = certify_qdm_local_record(
            model,
            local_record,
            local_record_index=local_record_index,
            config=padding_config,
        )
        for report in reports:
            certified_items.append((local_record, report))
            for config_row in report.padding.global_support_configs:
                limited_config_keys.add(_config_key(config_row))
            for config_row in report.leakage_configs:
                limited_config_keys.add(_config_key(config_row))

    layout = model.layout

    if not certified_items:
        limited_basis = Basis.empty(layout)
        empty_matrix = scipy_sparse.csr_array((0, 0), dtype=np.complex128)
        search_config = _cage_search_config_from_local_and_padding(
            local_result.config,
            padding_config,
        )
        return CertifiedLocalQDMCageSearchResult(
            cage_search_result=CageSearchResult(
                records=[],
                hilbert_size=0,
                config=search_config,
                type1_candidates=[],
                type2_candidates=[],
                search_stage_seconds={},
            ),
            basis=limited_basis,
            kinetic_matrix=empty_matrix,
            self_loop_values=np.zeros(0, dtype=np.complex128),
            reports=[],
            padding_config=padding_config,
        )

    limited_configs = np.asarray([list(key) for key in limited_config_keys], dtype=np.int64)
    if padding_config.sort_limited_basis:
        order = np.lexsort(limited_configs.T[::-1])
        limited_configs = limited_configs[order]

    limited_basis = Basis.from_states(layout, limited_configs)
    limited_index = {_config_key(row): i for i, row in enumerate(limited_basis.states)}
    limited_kinetic = build_qdm_global_limited_kinetic_matrix(model, limited_basis)
    limited_self_loops = qdm_global_self_loop_values(model, limited_basis.states)

    search_config = _cage_search_config_from_local_and_padding(
        local_result.config,
        padding_config,
    )

    cage_records: list[CageRecord] = []
    candidate_by_signature: dict[tuple[int, int], list[CandidateSubgraph]] = defaultdict(list)

    for item_index, (local_record, report) in enumerate(certified_items):
        support_indices: list[int] = []
        support_amplitudes: list[complex] = []
        for config_row, amplitude in zip(
            report.padding.global_support_configs,
            local_record.local_state,
            strict=True,
        ):
            support_indices.append(int(limited_index[_config_key(config_row)]))
            support_amplitudes.append(complex(amplitude))

        support_arr = np.asarray(support_indices, dtype=np.int64)
        amplitude_arr = np.asarray(support_amplitudes, dtype=np.complex128)
        support_order = np.argsort(support_arr)
        support_arr = support_arr[support_order]
        amplitude_arr = amplitude_arr[support_order]

        norm = float(np.linalg.norm(amplitude_arr))
        if norm == 0.0:
            continue
        amplitude_arr = amplitude_arr / norm

        candidate = CandidateSubgraph(
            vertices=support_arr,
            label=f"local_qdm_certified_{item_index}",
            metadata={
                "source": "LocalQDMCageSearcher",
                "local_signature": local_record.signature,
                "local_link_ids": local_record.local_link_ids.copy(),
                "active_plaquette_ids": local_record.active_plaquette_ids.copy(),
                "scoring_plaquette_ids": local_record.scoring_plaquette_ids.copy(),
                "unresolved_boundary_plaquette_ids": (
                    local_record.unresolved_boundary_plaquette_ids.copy()
                ),
                "padding_exterior_link_ids": report.padding.exterior_link_ids.copy(),
                "padding_index": report.padding_index,
                "one_hop_shell_size": report.one_hop_shell_size,
            },
        )
        candidate_by_signature[report.signature].append(candidate)

        cage_state = CageState(
            energy=complex(report.energy),
            local_state=amplitude_arr,
            support=support_arr,
            boundary_residual=float(report.leakage_residual),
            eigen_residual=float(report.support_hamiltonian_residual),
            full_residual=float(report.full_residual),
            metadata={
                "source": "certify_qdm_local_result",
                "local_record_index": report.local_record_index,
                "padding_index": report.padding_index,
                "kinetic_eigenvalue": report.kinetic_eigenvalue,
                "self_loop_value": report.self_loop_value,
                "support_kinetic_residual": report.support_kinetic_residual,
                "support_hamiltonian_residual": report.support_hamiltonian_residual,
                "one_hop_shell_size": report.one_hop_shell_size,
            },
        )

        full_state = None
        if padding_config.store_full_states:
            full_state = np.zeros(int(limited_basis.n_states), dtype=np.complex128)
            full_state[support_arr] = amplitude_arr

        cage_records.append(
            CageRecord(
                cage_state=cage_state,
                signature=report.signature,
                candidate=candidate,
                full_state=full_state,
            )
        )

    return CertifiedLocalQDMCageSearchResult(
        cage_search_result=CageSearchResult(
            records=cage_records,
            hilbert_size=int(limited_basis.n_states),
            config=search_config,
            type1_candidates=[
                candidate
                for signature in sorted(candidate_by_signature)
                for candidate in candidate_by_signature[signature]
            ],
            type2_candidates=[],
            search_stage_seconds={},
        ),
        basis=limited_basis,
        kinetic_matrix=limited_kinetic,
        self_loop_values=limited_self_loops,
        reports=[report for _record, report in certified_items],
        padding_config=padding_config,
    )


def certify_qdm_local_record(
    model: object,
    local_record: LocalQDMCageRecord,
    *,
    local_record_index: int = 0,
    config: LocalQDMPaddingConfig | None = None,
) -> list[LocalQDMCertificationReport]:
    """Return certified shared-exterior paddings for one local QDM record."""
    padding_config = LocalQDMPaddingConfig() if config is None else config
    if padding_config.max_paddings_per_record == 0:
        return []

    paddings = find_shared_qdm_exterior_paddings(
        model,
        local_record,
        config=padding_config,
    )

    reports: list[LocalQDMCertificationReport] = []
    for padding_index, padding in enumerate(paddings):
        report = _certify_qdm_padding(
            model,
            local_record,
            padding,
            local_record_index=local_record_index,
            padding_index=padding_index,
            config=padding_config,
        )
        if report is not None:
            reports.append(report)

    return reports


def make_qdm_cage_block(
    model: object,
    local_record: LocalQDMCageRecord,
    *,
    block_id: int = 0,
    guard_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
) -> LocalQDMCageBlock:
    """Create a constant-boundary Lego block from a local QDM cage record.

    Independent product padding requires the number of dimers contributed by
    the block at every global site to be independent of the local support
    configuration.  If this fails, one shared exterior cannot tensor with the
    entire block support, so this function raises ``ValueError``.
    """
    link_ids = np.asarray(local_record.local_link_ids, dtype=np.int64)
    support_configs = np.asarray(local_record.support_configs, dtype=np.int64)
    if support_configs.ndim != 2:
        raise ValueError("local_record.support_configs must have shape (support, n_local_links).")
    if support_configs.shape[1] != link_ids.size:
        raise ValueError("local_record support width must match local_link_ids size.")

    site_counts = _constant_qdm_block_site_counts(model, link_ids, support_configs)
    if site_counts is None:
        raise ValueError(
            "Local cage record is not an independent padding block: "
            "its site occupation contribution changes across support configs."
        )

    if guard_plaquette_ids is None:
        guard = np.unique(
            np.concatenate(
                [
                    np.asarray(local_record.active_plaquette_ids, dtype=np.int64),
                    np.asarray(local_record.unresolved_boundary_plaquette_ids, dtype=np.int64),
                ]
            )
        ).astype(np.int64)
    else:
        guard = _unique_int_array(guard_plaquette_ids, name="guard_plaquette_ids")
        _validate_plaquette_ids(model, guard)

    return LocalQDMCageBlock(
        block_id=int(block_id),
        record=local_record,
        link_ids=link_ids.copy(),
        active_plaquette_ids=np.asarray(local_record.active_plaquette_ids, dtype=np.int64).copy(),
        guard_plaquette_ids=guard,
        support_configs=support_configs.copy(),
        amplitudes=np.asarray(local_record.local_state, dtype=np.complex128).copy(),
        site_counts=site_counts,
    )


def iter_multi_qdm_block_paddings(
    model: object,
    block_pool: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
    max_yielded: int | None = None,
) -> Iterator[MultiLocalQDMPadding]:
    """Yield shared-exterior paddings built from a pool of QDM blocks.

    This is the streaming counterpart of :func:`find_multi_qdm_block_paddings`.
    It is intended for certification-in-the-loop workflows, where a caller may
    want to keep trying raw exterior completions until enough *certified* cages
    are found.  ``max_yielded`` limits the number of raw candidate paddings
    yielded by this iterator; if omitted, ``config.max_padding_attempts`` is
    used.
    """
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    yielded_limit = multi_config.max_padding_attempts if max_yielded is None else max_yielded
    if yielded_limit is not None and yielded_limit <= 0:
        return

    blocks = tuple(block_pool)
    if not blocks:
        return

    block_ids = [int(block.block_id) for block in blocks]
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("block_pool contains duplicate block_id values.")

    required_count = int(getattr(model, "required_count", 1))
    max_blocks = multi_config.max_blocks if multi_config.max_blocks is not None else len(blocks)
    max_blocks = min(int(max_blocks), len(blocks))

    selected: list[LocalQDMCageBlock] = []
    used_links: set[int] = set()
    site_counts = np.zeros(int(model.lattice.num_sites), dtype=np.int64)
    product_support_size = 1
    yielded_count = 0

    def can_yield_more() -> bool:
        return yielded_limit is None or yielded_count < yielded_limit

    def can_add(block: LocalQDMCageBlock) -> bool:
        block_link_set = set(int(link_id) for link_id in block.link_ids)
        if used_links.intersection(block_link_set):
            return False
        if np.any(site_counts + block.site_counts > required_count):
            return False
        if multi_config.max_product_support_size is not None:
            next_size = int(product_support_size) * int(block.support_size)
            if next_size > multi_config.max_product_support_size:
                return False
        if multi_config.require_kinetic_separation and not _qdm_block_is_kinetically_separated(
            model,
            tuple(selected),
            block,
        ):
            return False
        return True

    def dfs(start: int) -> Iterator[MultiLocalQDMPadding]:
        nonlocal product_support_size, site_counts, yielded_count
        if not can_yield_more():
            return
        if len(selected) >= multi_config.min_blocks:
            fixed_blocks = tuple(selected)
            for padding in _iter_qdm_exterior_paddings_for_blocks(
                model,
                fixed_blocks,
                config=multi_config,
            ):
                if not can_yield_more():
                    return
                yielded_count += 1
                yield padding
            if not can_yield_more():
                return
        if len(selected) >= max_blocks:
            return

        for block_index in range(start, len(blocks)):
            block = blocks[block_index]
            if not can_add(block):
                continue
            block_link_set = set(int(link_id) for link_id in block.link_ids)
            selected.append(block)
            used_links.update(block_link_set)
            old_site_counts = site_counts.copy()
            site_counts = site_counts + block.site_counts
            old_product_support_size = product_support_size
            product_support_size *= int(block.support_size)
            try:
                yield from dfs(block_index + 1)
            finally:
                product_support_size = old_product_support_size
                site_counts = old_site_counts
                used_links.difference_update(block_link_set)
                selected.pop()
            if not can_yield_more():
                return

    yield from dfs(0)


def find_multi_qdm_block_paddings(
    model: object,
    block_pool: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
) -> list[MultiLocalQDMPadding]:
    """Find shared-exterior paddings built from a pool of local QDM blocks.

    This materialized API keeps the original raw-padding semantics:
    ``config.max_paddings`` is the maximum number of candidate paddings returned.
    Certification helpers use :func:`iter_multi_qdm_block_paddings` directly so
    they can keep trying candidates until enough certified cages are found.
    """
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    if multi_config.max_paddings == 0:
        return []
    return list(
        iter_multi_qdm_block_paddings(
            model,
            block_pool,
            config=multi_config,
            max_yielded=multi_config.max_paddings,
        )
    )


def certify_qdm_multi_block_padding(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    padding: MultiLocalQDMPadding,
    *,
    padding_index: int = 0,
    config: LocalQDMMultiPaddingConfig | None = None,
) -> MultiLocalQDMCertificationReport | None:
    """Certify one multi-block QDM padding by explicit global one-hop action."""
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    return _certify_qdm_multi_padding(
        model,
        tuple(blocks),
        padding,
        padding_index=padding_index,
        config=multi_config,
    )


def _qdm_multi_padding_attempt_limit(config: LocalQDMMultiPaddingConfig) -> int | None:
    """Return the raw-padding attempt cap used by certification loops.

    ``max_paddings`` caps certified successes.  ``max_padding_attempts`` is the
    only raw-attempt cap; ``None`` means the finite padding iterator is allowed
    to run until enough certified reports are found or the search space is
    exhausted.
    """
    if config.max_padding_attempts is None:
        return None
    return int(config.max_padding_attempts)


def certify_qdm_multi_block_paddings(
    model: object,
    block_pool: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
) -> list[MultiLocalQDMCertificationReport]:
    """Find and certify Lego-style multi-block QDM paddings from a block pool.

    Candidate padding generation is interleaved with certification.  The search
    stops after ``config.max_paddings`` certified reports or after
    ``config.max_padding_attempts`` raw padding attempts.  If
    ``max_padding_attempts`` is ``None``, there is no separate raw-attempt cap.
    """
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    if multi_config.max_paddings == 0:
        return []

    block_by_id = {int(block.block_id): block for block in block_pool}
    reports: list[MultiLocalQDMCertificationReport] = []
    for padding_index, padding in enumerate(
        iter_multi_qdm_block_paddings(
            model,
            block_pool,
            config=multi_config,
            max_yielded=_qdm_multi_padding_attempt_limit(multi_config),
        )
    ):
        blocks = tuple(block_by_id[int(block_id)] for block_id in padding.block_ids)
        report = _certify_qdm_multi_padding(
            model,
            blocks,
            padding,
            padding_index=padding_index,
            config=multi_config,
        )
        if report is not None:
            reports.append(report)
            if len(reports) >= multi_config.max_paddings:
                break
    return reports


def diagnose_qdm_multi_block_paddings(
    model: object,
    block_pool: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
) -> QDMMultiPaddingDiagnostics:
    """Find multi-block paddings and report both successes and failures.

    This diagnostic path uses the same interleaved padding/certification loop as
    :func:`certify_qdm_multi_block_paddings`.  ``paddings`` stores the raw
    candidates actually attempted, while ``n_padding_attempts`` records that
    count explicitly for notebook/debug summaries.
    """
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    block_by_id = {int(block.block_id): block for block in block_pool}

    paddings: list[MultiLocalQDMPadding] = []
    reports: list[MultiLocalQDMCertificationReport] = []
    failures: list[QDMMultiPaddingFailureReport] = []
    first_certified_padding_index: int | None = None

    if multi_config.max_paddings == 0:
        return QDMMultiPaddingDiagnostics(
            paddings=[],
            reports=[],
            failures=[],
            config=multi_config,
            padding_attempts=0,
            first_certified_padding_index=None,
        )

    for padding_index, padding in enumerate(
        iter_multi_qdm_block_paddings(
            model,
            block_pool,
            config=multi_config,
            max_yielded=_qdm_multi_padding_attempt_limit(multi_config),
        )
    ):
        paddings.append(padding)
        blocks = tuple(block_by_id[int(block_id)] for block_id in padding.block_ids)
        report = _certify_qdm_multi_padding(
            model,
            blocks,
            padding,
            padding_index=padding_index,
            config=multi_config,
        )
        if report is not None:
            reports.append(report)
            if first_certified_padding_index is None:
                first_certified_padding_index = int(padding_index)
            if len(reports) >= multi_config.max_paddings:
                break
            continue
        failures.append(
            _qdm_multi_padding_failure_report(
                model,
                blocks,
                padding,
                padding_index=padding_index,
                config=multi_config,
            )
        )

    return QDMMultiPaddingDiagnostics(
        paddings=paddings,
        reports=reports,
        failures=failures,
        config=multi_config,
        padding_attempts=len(paddings),
        first_certified_padding_index=first_certified_padding_index,
    )


def qdm_multi_padding_config_schedule(
    config: LocalQDMMultiPaddingConfig | None = None,
    *,
    stages: Sequence[str] = ("loose", "static", "strict"),
) -> list[tuple[str, LocalQDMMultiPaddingConfig]]:
    """Return a permissive-to-strict schedule of multi-padding configs."""
    base = LocalQDMMultiPaddingConfig() if config is None else config
    scheduled: list[tuple[str, LocalQDMMultiPaddingConfig]] = []
    for stage in stages:
        stage = str(stage)
        if stage == "base":
            stage_config = base
        elif stage == "loose":
            stage_config = replace(
                base,
                require_static_exterior=False,
                require_kinetic_separation=False,
            )
        elif stage == "static":
            stage_config = replace(
                base,
                require_static_exterior=True,
                require_kinetic_separation=False,
            )
        elif stage == "strict":
            stage_config = replace(
                base,
                require_static_exterior=True,
                require_kinetic_separation=True,
            )
        else:
            raise ValueError(f"Unsupported multi-padding stage: {stage!r}.")
        scheduled.append((stage, stage_config))
    return scheduled


def _qdm_multi_block_report_key(
    report: MultiLocalQDMCertificationReport,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], tuple[int, int]]:
    support_key = tuple(
        sorted(_config_key(config_row) for config_row in report.padding.global_support_configs)
    )
    return (
        tuple(int(block_id) for block_id in report.block_ids),
        support_key,
        report.signature,
    )


def _deduplicate_qdm_multi_block_reports(
    reports: Sequence[MultiLocalQDMCertificationReport],
) -> list[MultiLocalQDMCertificationReport]:
    deduplicated: list[MultiLocalQDMCertificationReport] = []
    seen: set[tuple[tuple[int, ...], tuple[tuple[int, ...], ...], tuple[int, int]]] = set()
    for report in reports:
        key = _qdm_multi_block_report_key(report)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(report)
    return deduplicated


def robust_certify_qdm_multi_block_result(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
    stages: Sequence[str] = ("loose", "static", "strict"),
) -> CertifiedLocalQDMCageSearchResult:
    """Certify a block pool with a multi-stage padding schedule.

    The early stages are deliberately permissive; exact global certification is
    still the only acceptance criterion.  Duplicate certified supports found at
    multiple stages are deduplicated before wrapping into a limited-basis result.
    """
    base_config = LocalQDMMultiPaddingConfig() if config is None else config
    all_reports: list[MultiLocalQDMCertificationReport] = []

    for _stage_name, stage_config in qdm_multi_padding_config_schedule(
        base_config,
        stages=stages,
    ):
        all_reports.extend(certify_qdm_multi_block_paddings(model, blocks, config=stage_config))

    return certified_qdm_result_from_multi_block_reports(
        model,
        _deduplicate_qdm_multi_block_reports(all_reports),
        config=base_config,
    )


def certified_qdm_result_from_multi_block_reports(
    model: object,
    reports: Sequence[MultiLocalQDMCertificationReport],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
) -> CertifiedLocalQDMCageSearchResult:
    """Wrap multi-block QDM certificates as a limited-basis cage result.

    The returned object uses the same ``CertifiedLocalQDMCageSearchResult``
    container as the single-block local-padding path. Its basis is the limited
    union of certified support configurations and their one-hop kinetic shell,
    so downstream classification and visualization tools can consume it without
    enumerating the full global Hilbert space.
    """
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    report_list = list(reports)
    limited_config_keys: set[tuple[int, ...]] = set()

    for report in report_list:
        for config_row in report.padding.global_support_configs:
            limited_config_keys.add(_config_key(config_row))
        for config_row in report.leakage_configs:
            limited_config_keys.add(_config_key(config_row))

    search_config = _cage_search_config_from_multi_padding(
        model,
        multi_config,
        report_list,
    )

    if not limited_config_keys:
        limited_basis = Basis.empty(model.layout)
        empty_matrix = scipy_sparse.csr_array((0, 0), dtype=np.complex128)
        return CertifiedLocalQDMCageSearchResult(
            cage_search_result=CageSearchResult(
                records=[],
                hilbert_size=0,
                config=search_config,
                type1_candidates=[],
                type2_candidates=[],
                search_stage_seconds={},
            ),
            basis=limited_basis,
            kinetic_matrix=empty_matrix,
            self_loop_values=np.zeros(0, dtype=np.complex128),
            reports=[],
            padding_config=multi_config,
        )

    limited_configs = np.asarray([list(key) for key in limited_config_keys], dtype=np.int64)
    if multi_config.sort_limited_basis:
        order = np.lexsort(limited_configs.T[::-1])
        limited_configs = limited_configs[order]

    limited_basis = Basis.from_states(model.layout, limited_configs)
    limited_index = {_config_key(row): i for i, row in enumerate(limited_basis.states)}
    limited_kinetic = build_qdm_global_limited_kinetic_matrix(model, limited_basis)
    limited_self_loops = qdm_global_self_loop_values(model, limited_basis.states)

    cage_records: list[CageRecord] = []
    type1_candidates: list[CandidateSubgraph] = []

    for report_index, report in enumerate(report_list):
        support_indices: list[int] = []
        support_amplitudes: list[complex] = []
        for config_row, amplitude in zip(
            report.padding.global_support_configs,
            report.padding.global_amplitudes,
            strict=True,
        ):
            support_indices.append(int(limited_index[_config_key(config_row)]))
            support_amplitudes.append(complex(amplitude))

        support_arr = np.asarray(support_indices, dtype=np.int64)
        amplitude_arr = np.asarray(support_amplitudes, dtype=np.complex128)
        support_order = np.argsort(support_arr)
        support_arr = support_arr[support_order]
        amplitude_arr = amplitude_arr[support_order]

        norm = float(np.linalg.norm(amplitude_arr))
        if norm == 0.0:
            continue
        amplitude_arr = amplitude_arr / norm

        candidate = CandidateSubgraph(
            vertices=support_arr,
            label=f"multi_qdm_certified_{report_index}",
            metadata={
                "source": "certified_qdm_result_from_multi_block_reports",
                "block_ids": tuple(int(block_id) for block_id in report.block_ids),
                "padding_index": report.padding_index,
                "kinetic_eigenvalue": report.kinetic_eigenvalue,
                "self_loop_value": report.self_loop_value,
                "padding_exterior_link_ids": report.padding.exterior_link_ids.copy(),
                "one_hop_shell_size": report.one_hop_shell_size,
            },
        )
        type1_candidates.append(candidate)

        cage_state = CageState(
            energy=complex(report.energy),
            local_state=amplitude_arr,
            support=support_arr,
            boundary_residual=float(report.leakage_residual),
            eigen_residual=float(report.support_hamiltonian_residual),
            full_residual=float(report.full_residual),
            metadata={
                "source": "certify_qdm_multi_block_result",
                "block_ids": tuple(int(block_id) for block_id in report.block_ids),
                "padding_index": report.padding_index,
                "kinetic_eigenvalue": report.kinetic_eigenvalue,
                "self_loop_value": report.self_loop_value,
                "support_kinetic_residual": report.support_kinetic_residual,
                "support_hamiltonian_residual": report.support_hamiltonian_residual,
                "one_hop_shell_size": report.one_hop_shell_size,
            },
        )

        full_state = None
        if multi_config.store_full_states:
            full_state = np.zeros(int(limited_basis.n_states), dtype=np.complex128)
            full_state[support_arr] = amplitude_arr

        cage_records.append(
            CageRecord(
                cage_state=cage_state,
                signature=report.signature,
                candidate=candidate,
                full_state=full_state,
            )
        )

    return CertifiedLocalQDMCageSearchResult(
        cage_search_result=CageSearchResult(
            records=cage_records,
            hilbert_size=int(limited_basis.n_states),
            config=search_config,
            type1_candidates=type1_candidates,
            type2_candidates=[],
            search_stage_seconds={},
        ),
        basis=limited_basis,
        kinetic_matrix=limited_kinetic,
        self_loop_values=limited_self_loops,
        reports=report_list,
        padding_config=multi_config,
    )


def certify_qdm_multi_block_result(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
) -> CertifiedLocalQDMCageSearchResult:
    """Find/certify multi-block QDM paddings and return a certified result.

    This is the multi-block analogue of ``certify_qdm_local_result``: it keeps
    the basis limited to the certified product support plus one-hop shell, but
    exposes ordinary ``CageRecord`` entries for existing tools.
    """
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    reports = certify_qdm_multi_block_paddings(model, blocks, config=multi_config)
    return certified_qdm_result_from_multi_block_reports(
        model,
        reports,
        config=multi_config,
    )


def _qdm_multi_padding_failure_report(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    padding: MultiLocalQDMPadding,
    *,
    padding_index: int,
    config: LocalQDMMultiPaddingConfig,
) -> QDMMultiPaddingFailureReport:
    fixed_blocks = tuple(blocks)
    if config.require_static_exterior and not _multi_padding_has_static_exterior(
        model,
        padding,
        fixed_blocks,
    ):
        return QDMMultiPaddingFailureReport(
            block_ids=tuple(int(block_id) for block_id in padding.block_ids),
            padding_index=int(padding_index),
            reason="static_exterior",
            padding=padding,
        )

    amplitudes = np.asarray(padding.global_amplitudes, dtype=np.complex128)
    norm = float(np.linalg.norm(amplitudes))
    if norm == 0.0:
        return QDMMultiPaddingFailureReport(
            block_ids=tuple(int(block_id) for block_id in padding.block_ids),
            padding_index=int(padding_index),
            reason="zero_norm",
            padding=padding,
        )
    amplitudes = amplitudes / norm

    support_configs = np.asarray(padding.global_support_configs, dtype=np.int64)
    plaquette_actions = _qdm_multi_block_certification_actions(model, fixed_blocks, config)
    support_keys = [_config_key(config_row) for config_row in support_configs]
    support_amplitude_by_key = {
        key: complex(amplitude) for key, amplitude in zip(support_keys, amplitudes, strict=True)
    }

    action_by_key: dict[tuple[int, ...], complex] = defaultdict(complex)
    action_by_key_and_class: dict[str, dict[tuple[int, ...], complex]] = defaultdict(
        lambda: defaultdict(complex)
    )
    touched_keys: set[tuple[int, ...]] = set(support_keys)
    for source_config, source_amplitude in zip(support_configs, amplitudes, strict=True):
        for action in plaquette_actions:
            transition = _qdm_flip_transition_from_action(source_config, action)
            if transition is None:
                continue
            final_config, coefficient = transition
            final_key = _config_key(final_config)
            contribution = complex(coefficient) * complex(source_amplitude)
            action_by_key[final_key] += contribution
            action_class = _qdm_action_plaquette_class(action, fixed_blocks)
            action_by_key_and_class[action_class][final_key] += contribution
            touched_keys.add(final_key)

    kappa = complex(sum(int(block.kappa) for block in fixed_blocks))
    support_kinetic_residuals: list[complex] = []
    leakage_values: list[complex] = []
    leakage_values_by_class: dict[str, list[complex]] = defaultdict(list)
    leakage_counts_by_class: dict[str, int] = defaultdict(int)
    for key in sorted(touched_keys):
        action_value = complex(action_by_key.get(key, 0.0 + 0.0j))
        if key in support_amplitude_by_key:
            support_kinetic_residuals.append(action_value - kappa * support_amplitude_by_key[key])
        else:
            leakage_values.append(action_value)
            for action_class, class_action_by_key in action_by_key_and_class.items():
                class_value = complex(class_action_by_key.get(key, 0.0 + 0.0j))
                if abs(class_value) <= config.tolerance:
                    continue
                leakage_values_by_class[action_class].append(class_value)
                leakage_counts_by_class[action_class] += 1

    support_kinetic_residual = float(
        np.linalg.norm(np.asarray(support_kinetic_residuals, dtype=np.complex128))
    )
    leakage_residual = float(np.linalg.norm(np.asarray(leakage_values, dtype=np.complex128)))
    leakage_norms_by_class = {
        action_class: float(np.linalg.norm(np.asarray(values, dtype=np.complex128)))
        for action_class, values in leakage_values_by_class.items()
    }
    if leakage_residual > config.tolerance:
        return QDMMultiPaddingFailureReport(
            block_ids=tuple(int(block_id) for block_id in padding.block_ids),
            padding_index=int(padding_index),
            reason="leakage_residual",
            padding=padding,
            leakage_residual=leakage_residual,
            support_kinetic_residual=support_kinetic_residual,
            leakage_counts_by_class=dict(leakage_counts_by_class),
            leakage_norms_by_class=leakage_norms_by_class,
        )
    if support_kinetic_residual > config.tolerance:
        return QDMMultiPaddingFailureReport(
            block_ids=tuple(int(block_id) for block_id in padding.block_ids),
            padding_index=int(padding_index),
            reason="support_kinetic_residual",
            padding=padding,
            leakage_residual=leakage_residual,
            support_kinetic_residual=support_kinetic_residual,
        )

    support_self_loops = _qdm_global_self_loop_values_from_actions(
        support_configs,
        plaquette_actions,
    )
    self_loop_value = complex(support_self_loops[0]) if support_self_loops.size else 0.0 + 0.0j
    if np.linalg.norm(support_self_loops - self_loop_value) > config.tolerance:
        return QDMMultiPaddingFailureReport(
            block_ids=tuple(int(block_id) for block_id in padding.block_ids),
            padding_index=int(padding_index),
            reason="nonuniform_self_loop",
            padding=padding,
            leakage_residual=leakage_residual,
            support_kinetic_residual=support_kinetic_residual,
        )

    energy = self_loop_value + kappa
    support_h_residuals = []
    for key, amplitude, self_loop in zip(
        support_keys,
        amplitudes,
        support_self_loops,
        strict=True,
    ):
        kinetic_action = complex(action_by_key.get(key, 0.0 + 0.0j))
        support_h_residuals.append(
            kinetic_action + complex(self_loop) * amplitude - energy * amplitude
        )
    support_hamiltonian_residual = float(
        np.linalg.norm(np.asarray(support_h_residuals, dtype=np.complex128))
    )
    full_residual = float(np.hypot(support_hamiltonian_residual, leakage_residual))
    if full_residual > config.tolerance:
        return QDMMultiPaddingFailureReport(
            block_ids=tuple(int(block_id) for block_id in padding.block_ids),
            padding_index=int(padding_index),
            reason="full_residual",
            padding=padding,
            leakage_residual=leakage_residual,
            support_kinetic_residual=support_kinetic_residual,
            support_hamiltonian_residual=support_hamiltonian_residual,
            full_residual=full_residual,
        )

    signature = signature_from_energy_and_self_loop(
        energy,
        self_loop_value,
        tolerance=max(config.tolerance, 1.0e-15) * 10.0,
        potential_unit=_infer_potential_unit_from_model(model),
    )
    if signature is None:
        return QDMMultiPaddingFailureReport(
            block_ids=tuple(int(block_id) for block_id in padding.block_ids),
            padding_index=int(padding_index),
            reason="signature_inference_failed",
            padding=padding,
            leakage_residual=leakage_residual,
            support_kinetic_residual=support_kinetic_residual,
            support_hamiltonian_residual=support_hamiltonian_residual,
            full_residual=full_residual,
        )

    return QDMMultiPaddingFailureReport(
        block_ids=tuple(int(block_id) for block_id in padding.block_ids),
        padding_index=int(padding_index),
        reason="unknown",
        padding=padding,
        leakage_residual=leakage_residual,
        support_kinetic_residual=support_kinetic_residual,
        support_hamiltonian_residual=support_hamiltonian_residual,
        full_residual=full_residual,
    )


def _qdm_action_plaquette_class(
    action: _QDMGlobalPlaquetteAction,
    blocks: Sequence[LocalQDMCageBlock],
) -> str:
    """Classify a plaquette action relative to selected local blocks."""
    action_link_set = {int(link_id) for link_id in action.links}
    owner_link_sets = [set(int(link_id) for link_id in block.link_ids) for block in blocks]
    owners = {
        owner
        for owner, link_set in enumerate(owner_link_sets)
        if action_link_set.intersection(link_set)
    }
    if len(owners) > 1:
        return "multi_block_spacer"
    if not owners:
        return "pure_exterior"

    owner = next(iter(owners))
    if action_link_set.issubset(owner_link_sets[owner]):
        active_ids = {int(pid) for pid in blocks[owner].active_plaquette_ids}
        if int(action.plaquette_id) in active_ids:
            return "single_block_active"
        return "single_block_internal"
    return "single_block_boundary"


def _qdm_pattern_compatible_with_block_support(
    block: LocalQDMCageBlock,
    action: _QDMGlobalPlaquetteAction,
    pattern: npt.NDArray[np.int64],
) -> bool:
    """Return whether a plaquette pattern can occur on one block support."""
    local_index_by_link = {int(link_id): i for i, link_id in enumerate(block.link_ids)}
    local_indices: list[int] = []
    required_values: list[int] = []
    for position, link_id in enumerate(action.links):
        local_index = local_index_by_link.get(int(link_id))
        if local_index is None:
            continue
        local_indices.append(int(local_index))
        required_values.append(int(pattern[int(position)]))

    if not local_indices:
        return True

    support_values = np.asarray(block.support_configs, dtype=np.int64)[:, local_indices]
    required = np.asarray(required_values, dtype=np.int64)
    return bool(np.any(np.all(support_values == required, axis=1)))


def _qdm_exterior_flippability_preferences_by_variable(
    model: object,
    exterior_link_ids: npt.NDArray[np.int64],
    blocks: Sequence[LocalQDMCageBlock],
    *,
    include_exterior_only: bool,
) -> list[list[_QDMExteriorFlippabilityPreference]]:
    """Return plaquette-flippability preferences touched by each exterior variable.

    A preference stores exterior-link patterns that would allow a plaquette to be
    flippable for at least one product-support configuration of the selected
    blocks.  The DFS value ordering can then prefer assignments that destroy
    these dangerous patterns early, especially on spacer/boundary plaquettes.
    """
    n_exterior = int(exterior_link_ids.size)
    exterior_index_by_link = {
        int(link_id): int(exterior_index)
        for exterior_index, link_id in enumerate(exterior_link_ids)
    }
    preferences_by_variable: list[list[_QDMExteriorFlippabilityPreference]] = [
        [] for _ in range(n_exterior)
    ]

    weight_by_class = {
        "multi_block_spacer": 256,
        "single_block_boundary": 96,
        "pure_exterior": 16,
        "single_block_active": 8,
        "single_block_internal": 4,
    }

    for action in _qdm_global_plaquette_actions(model):
        exterior_positions: list[int] = []
        exterior_indices: list[int] = []
        for position, link_id in enumerate(action.links):
            exterior_index = exterior_index_by_link.get(int(link_id))
            if exterior_index is None:
                continue
            exterior_positions.append(int(position))
            exterior_indices.append(int(exterior_index))
        if not exterior_indices:
            continue

        plaquette_class = _qdm_action_plaquette_class(action, blocks)
        if plaquette_class == "pure_exterior" and not include_exterior_only:
            continue

        dangerous_patterns: list[tuple[int, ...]] = []
        for pattern in (action.pattern0, action.pattern1):
            if not all(
                _qdm_pattern_compatible_with_block_support(block, action, pattern)
                for block in blocks
            ):
                continue
            dangerous_patterns.append(
                tuple(int(pattern[position]) for position in exterior_positions)
            )

        if not dangerous_patterns:
            continue
        unique_patterns = tuple(
            np.asarray(pattern, dtype=np.int64) for pattern in sorted(set(dangerous_patterns))
        )
        preference = _QDMExteriorFlippabilityPreference(
            plaquette_id=int(action.plaquette_id),
            plaquette_class=plaquette_class,
            exterior_indices=np.asarray(exterior_indices, dtype=np.int64),
            dangerous_patterns=unique_patterns,
            weight=int(weight_by_class.get(plaquette_class, 1)),
        )
        for exterior_index in exterior_indices:
            preferences_by_variable[int(exterior_index)].append(preference)

    return preferences_by_variable


def _qdm_count_compatible_dangerous_patterns(
    preference: _QDMExteriorFlippabilityPreference,
    *,
    exterior_config: npt.NDArray[np.int64],
    assigned: npt.NDArray[np.bool_],
    trial_variable: int | None = None,
    trial_value: int | None = None,
) -> int:
    """Count dangerous patterns still compatible with the current partial branch."""
    count = 0
    for pattern in preference.dangerous_patterns:
        compatible = True
        for exterior_index, required_value in zip(
            preference.exterior_indices,
            pattern,
            strict=True,
        ):
            index = int(exterior_index)
            if trial_variable is not None and index == int(trial_variable):
                value = int(trial_value)  # type: ignore[arg-type]
            elif bool(assigned[index]):
                value = int(exterior_config[index])
            else:
                continue
            if value != int(required_value):
                compatible = False
                break
        if compatible:
            count += 1
    return count


def _qdm_exterior_variable_order(
    model: object,
    exterior_link_ids: npt.NDArray[np.int64],
    site_exterior_links: dict[int, npt.NDArray[np.int64]],
    site_targets: dict[int, int],
    *,
    fixed_link_sets: Sequence[set[int]],
    require_static_exterior: bool,
) -> npt.NDArray[np.int64]:
    """Return a deterministic DFS order for exterior QDM padding links.

    The first padding implementation used only local site-constraint scores.
    That is correct, but it may enumerate many globally legal exterior
    completions before touching the boundary/spacer links that decide whether a
    candidate certifies.  This order prioritizes links on plaquettes touching
    selected blocks, then links on exterior-only plaquettes when a static
    exterior is requested, while preserving the old site-constraint preference
    as a secondary signal.
    """
    n_exterior = int(exterior_link_ids.size)
    exterior_index_by_link = {
        int(link_id): int(exterior_index)
        for exterior_index, link_id in enumerate(exterior_link_ids)
    }
    link_owner: dict[int, int] = {}
    for owner, link_set in enumerate(fixed_link_sets):
        for link_id in link_set:
            link_owner[int(link_id)] = int(owner)

    scores = np.zeros(n_exterior, dtype=np.int64)

    for site_id, exterior_indices in site_exterior_links.items():
        n_site_exterior = int(exterior_indices.size)
        target = int(site_targets[int(site_id)])
        if n_site_exterior == 0:
            continue
        if target in {0, n_site_exterior}:
            weight = 256
        elif target in {1, n_site_exterior - 1}:
            weight = 96
        else:
            weight = 32
        for exterior_index in exterior_indices:
            scores[int(exterior_index)] += weight

    for action in _qdm_global_plaquette_actions(model):
        exterior_indices = [
            exterior_index_by_link[int(link_id)]
            for link_id in action.links
            if int(link_id) in exterior_index_by_link
        ]
        if not exterior_indices:
            continue

        owners = {
            link_owner[int(link_id)] for link_id in action.links if int(link_id) in link_owner
        }
        if len(owners) > 1:
            # Spacer plaquettes between independent blocks are the most useful
            # early decisions when kinetic separation is relaxed.
            plaquette_weight = 4096
        elif owners:
            # Boundary plaquettes touching one selected block determine the
            # one-hop leakage/certification pattern.
            plaquette_weight = 2048
        elif require_static_exterior:
            # Exterior-only plaquettes must be frozen; decide their links before
            # unrelated bulk variables so static branches are pruned earlier.
            plaquette_weight = 512
        else:
            plaquette_weight = 16

        for exterior_index in exterior_indices:
            scores[int(exterior_index)] += plaquette_weight

    # Use the physical link id, not the exterior-array position, as the final
    # tie-breaker so the order is stable under equivalent array construction.
    return np.lexsort((exterior_link_ids, -scores)).astype(np.int64)


def _qdm_static_exterior_plaquettes_by_variable(
    model: object,
    exterior_link_ids: npt.NDArray[np.int64],
    *,
    fixed_link_set: set[int],
) -> list[list[_QDMExteriorStaticPlaquette]]:
    """Return exterior-only static plaquette checks touched by each variable."""
    n_exterior = int(exterior_link_ids.size)
    exterior_index_by_link = {
        int(link_id): int(exterior_index)
        for exterior_index, link_id in enumerate(exterior_link_ids)
    }
    by_variable: list[list[_QDMExteriorStaticPlaquette]] = [[] for _ in range(n_exterior)]

    for action in _qdm_global_plaquette_actions(model):
        action_links = [int(link_id) for link_id in action.links]
        if any(link_id in fixed_link_set for link_id in action_links):
            continue
        if any(link_id not in exterior_index_by_link for link_id in action_links):
            continue
        exterior_indices = np.asarray(
            [exterior_index_by_link[link_id] for link_id in action_links],
            dtype=np.int64,
        )
        static_plaquette = _QDMExteriorStaticPlaquette(
            plaquette_id=int(action.plaquette_id),
            exterior_indices=exterior_indices,
            pattern0=action.pattern0,
            pattern1=action.pattern1,
        )
        for exterior_index in exterior_indices:
            by_variable[int(exterior_index)].append(static_plaquette)

    return by_variable


def _qdm_static_exterior_checks_pass(
    static_plaquettes: Sequence[_QDMExteriorStaticPlaquette],
    *,
    exterior_config: npt.NDArray[np.int64],
    assigned: npt.NDArray[np.bool_],
) -> bool:
    """Reject a branch once a required-static exterior plaquette is flippable."""
    for static_plaquette in static_plaquettes:
        exterior_indices = static_plaquette.exterior_indices
        if not bool(np.all(assigned[exterior_indices])):
            continue
        values = exterior_config[exterior_indices]
        if np.array_equal(values, static_plaquette.pattern0) or np.array_equal(
            values,
            static_plaquette.pattern1,
        ):
            return False
    return True


def _qdm_exterior_value_order(
    exterior_variable: int,
    *,
    exterior_config: npt.NDArray[np.int64],
    assigned: npt.NDArray[np.bool_],
    sites_by_exterior_variable: Sequence[Sequence[int]],
    site_exterior_links: dict[int, npt.NDArray[np.int64]],
    site_targets: dict[int, int],
    flippability_preferences_by_variable: (
        Sequence[Sequence[_QDMExteriorFlippabilityPreference]] | None
    ) = None,
) -> tuple[int, ...]:
    """Order binary choices by site constraints and spacer flippability risk."""
    scored_values: list[tuple[int, int]] = []
    preferences = (
        ()
        if flippability_preferences_by_variable is None
        else flippability_preferences_by_variable[int(exterior_variable)]
    )

    for value in (0, 1):
        score = 0
        feasible = True
        for site_id in sites_by_exterior_variable[int(exterior_variable)]:
            exterior_indices = site_exterior_links[int(site_id)]
            assigned_local = assigned[exterior_indices]
            occupied = int(np.sum(exterior_config[exterior_indices[assigned_local]]))
            unassigned = int(exterior_indices.size - np.count_nonzero(assigned_local))
            remaining_need = int(site_targets[int(site_id)]) - occupied
            remaining_after = unassigned - 1
            next_need = remaining_need - int(value)
            if next_need < 0 or next_need > remaining_after:
                feasible = False
                break
            if next_need in {0, remaining_after}:
                score += 4
            if remaining_after == 0:
                score += 8
        if not feasible:
            continue

        for preference in preferences:
            before = _qdm_count_compatible_dangerous_patterns(
                preference,
                exterior_config=exterior_config,
                assigned=assigned,
            )
            if before == 0:
                continue
            after = _qdm_count_compatible_dangerous_patterns(
                preference,
                exterior_config=exterior_config,
                assigned=assigned,
                trial_variable=int(exterior_variable),
                trial_value=int(value),
            )
            killed = before - after
            score += int(preference.weight) * int(killed)
            if after == 0:
                score += 2 * int(preference.weight)

        scored_values.append((score, int(value)))

    if not scored_values:
        return (0, 1)
    scored_values.sort(key=lambda item: (-item[0], item[1]))
    return tuple(value for _, value in scored_values)


def _iter_qdm_exterior_paddings_for_blocks(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig,
    factorized: bool = False,
) -> Iterator[MultiLocalQDMPadding | FactorizedLocalQDMPadding]:
    fixed_blocks = tuple(blocks)
    if not fixed_blocks:
        return
    if config.max_paddings_per_packing == 0:
        return
    if not _qdm_blocks_are_pairwise_link_disjoint(fixed_blocks):
        return
    if config.require_kinetic_separation and not _qdm_blocks_are_kinetically_separated(
        model,
        fixed_blocks,
    ):
        return

    required_count = int(getattr(model, "required_count", 1))
    total_site_counts = np.zeros(int(model.lattice.num_sites), dtype=np.int64)
    block_link_set: set[int] = set()
    for block in fixed_blocks:
        total_site_counts += np.asarray(block.site_counts, dtype=np.int64)
        block_link_set.update(int(link_id) for link_id in block.link_ids)
    if np.any(total_site_counts > required_count):
        return

    n_global_links = int(model.lattice.num_links)
    exterior_link_ids = np.asarray(
        [link_id for link_id in range(n_global_links) if link_id not in block_link_set],
        dtype=np.int64,
    )
    exterior_index_by_link = {int(link_id): i for i, link_id in enumerate(exterior_link_ids)}
    n_exterior = int(exterior_link_ids.size)

    site_targets: dict[int, int] = {}
    site_exterior_links: dict[int, npt.NDArray[np.int64]] = {}
    for site_id in range(int(model.lattice.num_sites)):
        incident = [int(link_id) for link_id in model.lattice.incident_links(int(site_id))]
        exterior_incident = [
            exterior_index_by_link[link_id]
            for link_id in incident
            if link_id in exterior_index_by_link
        ]
        target = required_count - int(total_site_counts[int(site_id)])
        if target < 0 or target > len(exterior_incident):
            return
        site_targets[int(site_id)] = int(target)
        site_exterior_links[int(site_id)] = np.asarray(exterior_incident, dtype=np.int64)

    if n_exterior == 0:
        exterior_config = np.zeros(0, dtype=np.int64)
        if factorized:
            padding = FactorizedLocalQDMPadding(
                block_ids=tuple(int(block.block_id) for block in fixed_blocks),
                exterior_link_ids=exterior_link_ids,
                exterior_config=exterior_config,
            )
            reason, _sector_validation, _max_touched = _factorized_padding_validation_reason(
                model,
                fixed_blocks,
                padding,
                config,
            )
            if reason is None:
                yield padding
        else:
            padding = _make_qdm_multi_padding_from_exterior(
                model,
                fixed_blocks,
                exterior_link_ids=exterior_link_ids,
                exterior_config=exterior_config,
            )
            if _multi_padding_passes_global_filters(model, padding, fixed_blocks, config):
                yield padding
        return

    variable_order = _qdm_exterior_variable_order(
        model,
        exterior_link_ids,
        site_exterior_links,
        site_targets,
        fixed_link_sets=[set(int(link_id) for link_id in block.link_ids) for block in fixed_blocks],
        require_static_exterior=config.require_static_exterior,
    )

    exterior_config = np.zeros(n_exterior, dtype=np.int64)
    assigned = np.zeros(n_exterior, dtype=bool)
    sites_by_exterior_variable: list[list[int]] = [[] for _ in range(n_exterior)]
    for site_id, exterior_indices in site_exterior_links.items():
        for exterior_index in exterior_indices:
            sites_by_exterior_variable[int(exterior_index)].append(int(site_id))

    static_exterior_plaquettes_by_variable = (
        _qdm_static_exterior_plaquettes_by_variable(
            model,
            exterior_link_ids,
            fixed_link_set=block_link_set,
        )
        if config.require_static_exterior
        else [[] for _ in range(n_exterior)]
    )
    flippability_preferences_by_variable = _qdm_exterior_flippability_preferences_by_variable(
        model,
        exterior_link_ids,
        fixed_blocks,
        include_exterior_only=config.require_static_exterior,
    )

    nodes_visited = 0
    yielded_count = 0

    def partial_site_check(site_id: int) -> bool:
        exterior_indices = site_exterior_links[site_id]
        target = site_targets[site_id]
        if exterior_indices.size == 0:
            return target == 0
        assigned_local = assigned[exterior_indices]
        occupied = int(np.sum(exterior_config[exterior_indices[assigned_local]]))
        unassigned = int(exterior_indices.size - np.count_nonzero(assigned_local))
        if occupied > target:
            return False
        if occupied + unassigned < target:
            return False
        if unassigned == 0 and occupied != target:
            return False
        return True

    def full_check() -> bool:
        for site_id in range(int(model.lattice.num_sites)):
            if not partial_site_check(int(site_id)):
                return False
        return True

    def dfs(depth: int) -> Iterator[MultiLocalQDMPadding]:
        nonlocal nodes_visited, yielded_count
        if yielded_count >= config.max_paddings_per_packing:
            return
        if config.max_dfs_nodes is not None and nodes_visited >= config.max_dfs_nodes:
            return
        nodes_visited += 1

        if depth == n_exterior:
            if full_check():
                if factorized:
                    padding = FactorizedLocalQDMPadding(
                        block_ids=tuple(int(block.block_id) for block in fixed_blocks),
                        exterior_link_ids=exterior_link_ids,
                        exterior_config=exterior_config.copy(),
                    )
                    reason, _sector_validation, _max_touched = (
                        _factorized_padding_validation_reason(
                            model,
                            fixed_blocks,
                            padding,
                            config,
                        )
                    )
                    passes_filters = reason is None
                else:
                    padding = _make_qdm_multi_padding_from_exterior(
                        model,
                        fixed_blocks,
                        exterior_link_ids=exterior_link_ids,
                        exterior_config=exterior_config.copy(),
                    )
                    passes_filters = _multi_padding_passes_global_filters(
                        model,
                        padding,
                        fixed_blocks,
                        config,
                    )
                if passes_filters:
                    yielded_count += 1
                    yield padding
            return

        exterior_variable = int(variable_order[depth])
        for value in _qdm_exterior_value_order(
            exterior_variable,
            exterior_config=exterior_config,
            assigned=assigned,
            sites_by_exterior_variable=sites_by_exterior_variable,
            site_exterior_links=site_exterior_links,
            site_targets=site_targets,
            flippability_preferences_by_variable=flippability_preferences_by_variable,
        ):
            if yielded_count >= config.max_paddings_per_packing:
                return
            exterior_config[exterior_variable] = value
            assigned[exterior_variable] = True
            touched_sites = sites_by_exterior_variable[exterior_variable]
            touched_static_plaquettes = static_exterior_plaquettes_by_variable[exterior_variable]
            if all(partial_site_check(site_id) for site_id in touched_sites) and (
                not touched_static_plaquettes
                or _qdm_static_exterior_checks_pass(
                    touched_static_plaquettes,
                    exterior_config=exterior_config,
                    assigned=assigned,
                )
            ):
                yield from dfs(depth + 1)
            assigned[exterior_variable] = False
            exterior_config[exterior_variable] = 0

    yield from dfs(0)


def _find_qdm_exterior_paddings_for_blocks(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig,
) -> list[MultiLocalQDMPadding]:
    return list(_iter_qdm_exterior_paddings_for_blocks(model, blocks, config=config))


def _certify_qdm_multi_padding(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    padding: MultiLocalQDMPadding,
    *,
    padding_index: int,
    config: LocalQDMMultiPaddingConfig,
) -> MultiLocalQDMCertificationReport | None:
    fixed_blocks = tuple(blocks)
    if tuple(int(block.block_id) for block in fixed_blocks) != tuple(
        int(x) for x in padding.block_ids
    ):
        raise ValueError("blocks must match padding.block_ids and order.")

    amplitudes = np.asarray(padding.global_amplitudes, dtype=np.complex128)
    norm = float(np.linalg.norm(amplitudes))
    if norm == 0.0:
        return None
    amplitudes = amplitudes / norm

    support_configs = np.asarray(padding.global_support_configs, dtype=np.int64)
    if config.require_static_exterior and not _multi_padding_has_static_exterior(
        model,
        padding,
        fixed_blocks,
    ):
        return None

    plaquette_actions = _qdm_multi_block_certification_actions(model, fixed_blocks, config)
    support_keys = [_config_key(config_row) for config_row in support_configs]
    support_amplitude_by_key = {
        key: complex(amplitude) for key, amplitude in zip(support_keys, amplitudes, strict=True)
    }

    action_by_key: dict[tuple[int, ...], complex] = defaultdict(complex)
    touched_keys: set[tuple[int, ...]] = set(support_keys)

    for source_config, source_amplitude in zip(support_configs, amplitudes, strict=True):
        for action in plaquette_actions:
            transition = _qdm_flip_transition_from_action(source_config, action)
            if transition is None:
                continue
            final_config, coefficient = transition
            final_key = _config_key(final_config)
            action_by_key[final_key] += complex(coefficient) * complex(source_amplitude)
            touched_keys.add(final_key)

    kappa = complex(sum(int(block.kappa) for block in fixed_blocks))
    support_kinetic_residuals: list[complex] = []
    leakage_values: list[complex] = []
    leakage_configs: list[npt.NDArray[np.int64]] = []

    for key in sorted(touched_keys):
        action = complex(action_by_key.get(key, 0.0 + 0.0j))
        if key in support_amplitude_by_key:
            expected = kappa * support_amplitude_by_key[key]
            support_kinetic_residuals.append(action - expected)
        else:
            leakage_values.append(action)
            leakage_configs.append(np.asarray(key, dtype=np.int64))

    support_kinetic_residual = float(np.linalg.norm(np.asarray(support_kinetic_residuals)))
    leakage_residual = float(np.linalg.norm(np.asarray(leakage_values, dtype=np.complex128)))

    if leakage_residual > config.tolerance:
        return None
    if support_kinetic_residual > config.tolerance:
        return None

    support_self_loops = _qdm_global_self_loop_values_from_actions(
        support_configs,
        plaquette_actions,
    )
    self_loop_value = complex(support_self_loops[0]) if support_self_loops.size else 0.0 + 0.0j
    if np.linalg.norm(support_self_loops - self_loop_value) > config.tolerance:
        return None

    energy = self_loop_value + kappa
    support_h_residuals = []
    for key, amplitude, self_loop in zip(
        support_keys,
        amplitudes,
        support_self_loops,
        strict=True,
    ):
        kinetic_action = complex(action_by_key.get(key, 0.0 + 0.0j))
        support_h_residuals.append(
            kinetic_action + complex(self_loop) * amplitude - energy * amplitude
        )
    support_hamiltonian_residual = float(
        np.linalg.norm(np.asarray(support_h_residuals, dtype=np.complex128))
    )
    full_residual = float(np.hypot(support_hamiltonian_residual, leakage_residual))
    if full_residual > config.tolerance:
        return None

    signature = signature_from_energy_and_self_loop(
        energy,
        self_loop_value,
        tolerance=max(config.tolerance, 1.0e-15) * 10.0,
        potential_unit=_infer_potential_unit_from_model(model),
    )
    if signature is None:
        return None

    leakage_arr = (
        np.asarray(leakage_configs, dtype=np.int64)
        if leakage_configs
        else np.empty((0, int(model.lattice.num_links)), dtype=np.int64)
    )

    return MultiLocalQDMCertificationReport(
        block_ids=tuple(int(block.block_id) for block in fixed_blocks),
        padding_index=int(padding_index),
        signature=signature,
        energy=energy,
        kinetic_eigenvalue=kappa,
        self_loop_value=self_loop_value,
        support_size=int(support_configs.shape[0]),
        one_hop_shell_size=int(len(touched_keys)),
        leakage_residual=leakage_residual,
        support_kinetic_residual=support_kinetic_residual,
        support_hamiltonian_residual=support_hamiltonian_residual,
        full_residual=full_residual,
        padding=padding,
        leakage_configs=leakage_arr,
    )


def _constant_qdm_block_site_counts(
    model: object,
    link_ids: npt.ArrayLike,
    support_configs: npt.ArrayLike,
) -> npt.NDArray[np.int64] | None:
    local_link_ids = np.asarray(link_ids, dtype=np.int64)
    support_arr = np.asarray(support_configs, dtype=np.int64)
    local_index_by_link = {int(link_id): i for i, link_id in enumerate(local_link_ids)}
    site_counts = np.zeros(int(model.lattice.num_sites), dtype=np.int64)

    for site_id in range(int(model.lattice.num_sites)):
        local_incident = [
            local_index_by_link[int(link_id)]
            for link_id in model.lattice.incident_links(int(site_id))
            if int(link_id) in local_index_by_link
        ]
        if local_incident:
            counts = np.sum(support_arr[:, local_incident], axis=1).astype(np.int64)
        else:
            counts = np.zeros(support_arr.shape[0], dtype=np.int64)
        unique_counts = np.unique(counts)
        if unique_counts.size != 1:
            return None
        site_counts[int(site_id)] = int(unique_counts[0])

    return site_counts


def _qdm_blocks_are_pairwise_link_disjoint(blocks: Sequence[LocalQDMCageBlock]) -> bool:
    used: set[int] = set()
    for block in blocks:
        block_links = set(int(link_id) for link_id in block.link_ids)
        if used.intersection(block_links):
            return False
        used.update(block_links)
    return True


def _qdm_block_is_kinetically_separated(
    model: object,
    existing_blocks: Sequence[LocalQDMCageBlock],
    new_block: LocalQDMCageBlock,
) -> bool:
    return _qdm_blocks_are_kinetically_separated(model, tuple(existing_blocks) + (new_block,))


def _qdm_blocks_are_kinetically_separated(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
) -> bool:
    link_owner: dict[int, int] = {}
    for block_position, block in enumerate(blocks):
        for link_id in block.link_ids:
            link_owner[int(link_id)] = int(block_position)

    for plaquette_id in model.plaquette_ids():
        owners = {
            link_owner[int(link_id)]
            for link_id in model.lattice.plaquette_links(int(plaquette_id))
            if int(link_id) in link_owner
        }
        if len(owners) > 1:
            return False
    return True


def factorized_qdm_padding_from_multi_padding(
    padding: MultiLocalQDMPadding,
) -> FactorizedLocalQDMPadding:
    """Drop the materialized Cartesian-product support from an old padding."""
    return FactorizedLocalQDMPadding(
        block_ids=padding.block_ids,
        exterior_link_ids=padding.exterior_link_ids,
        exterior_config=padding.exterior_config,
    )


def iter_factorized_qdm_block_paddings(
    model: object,
    block_pool: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
    max_yielded: int | None = None,
) -> Iterator[FactorizedLocalQDMPadding]:
    """Yield exterior assignments without materializing block support products.

    This mirrors :func:`iter_multi_qdm_block_paddings`, but the returned object
    contains only the block ids and shared exterior configuration.  The search
    therefore remains usable when ``prod(block.support_size)`` is too large to
    enumerate.  ``max_product_support_size`` is intentionally not applied on
    this path because the Cartesian-product support is never materialized.
    """
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    yielded_limit = multi_config.max_padding_attempts if max_yielded is None else max_yielded
    if yielded_limit is not None and yielded_limit <= 0:
        return

    blocks_tuple = tuple(block_pool)
    block_ids = [int(block.block_id) for block in blocks_tuple]
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("block_pool contains duplicate block_id values.")
    max_blocks = (
        len(blocks_tuple)
        if multi_config.max_blocks is None
        else min(int(multi_config.max_blocks), len(blocks_tuple))
    )
    yielded = 0

    for block_count in range(multi_config.min_blocks, max_blocks + 1):
        for blocks in itertools.combinations(blocks_tuple, block_count):
            if yielded_limit is not None and yielded >= yielded_limit:
                return
            if not _qdm_blocks_are_pairwise_link_disjoint(blocks):
                continue
            separated = _qdm_blocks_are_kinetically_separated(model, blocks)
            if multi_config.require_kinetic_separation and not separated:
                continue
            for padding in _iter_qdm_exterior_paddings_for_blocks(
                model,
                blocks,
                config=multi_config,
                factorized=True,
            ):
                if not isinstance(padding, FactorizedLocalQDMPadding):
                    raise TypeError("factorized padding iterator returned an unexpected object.")
                yield padding
                yielded += 1
                if yielded_limit is not None and yielded >= yielded_limit:
                    return


def find_factorized_qdm_block_paddings(
    model: object,
    block_pool: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
) -> list[FactorizedLocalQDMPadding]:
    """Materialize a bounded list of factorized QDM exterior assignments."""
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    if multi_config.max_paddings == 0:
        return []
    return list(
        itertools.islice(
            iter_factorized_qdm_block_paddings(
                model,
                block_pool,
                config=multi_config,
                max_yielded=multi_config.max_paddings,
            ),
            multi_config.max_paddings,
        )
    )


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


def _factorized_padding_reference_config(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    padding: FactorizedLocalQDMPadding,
    *,
    support_indices: Sequence[int] | None = None,
) -> npt.NDArray[np.int64]:
    config = np.zeros(int(model.lattice.num_links), dtype=np.int64)
    config[padding.exterior_link_ids] = padding.exterior_config
    indices = [0] * len(blocks) if support_indices is None else list(support_indices)
    if len(indices) != len(blocks):
        raise ValueError("support_indices must have one entry per block.")
    for block, support_index in zip(blocks, indices, strict=True):
        config[block.link_ids] = block.support_configs[int(support_index)]
    return config


def _factorized_padding_validation_reason(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    padding: FactorizedLocalQDMPadding,
    config: LocalQDMMultiPaddingConfig,
) -> tuple[str | None, str, int]:
    fixed_blocks = tuple(blocks)
    if not fixed_blocks:
        return "no_blocks", "not_checked", 0
    if tuple(int(block.block_id) for block in fixed_blocks) != padding.block_ids:
        return "block_id_mismatch", "not_checked", 0
    if not _qdm_blocks_are_pairwise_link_disjoint(fixed_blocks):
        return "overlapping_block_links", "not_checked", 0

    owner_by_link: dict[int, int] = {}
    for block_index, block in enumerate(fixed_blocks):
        for link_id in block.link_ids:
            owner_by_link[int(link_id)] = int(block_index)
    exterior_ids = set(int(link_id) for link_id in padding.exterior_link_ids)
    expected_exterior = set(range(int(model.lattice.num_links))) - set(owner_by_link)
    if exterior_ids != expected_exterior:
        return "incomplete_link_partition", "not_checked", 0

    max_touched = 0
    for action in _qdm_global_plaquette_actions(model):
        owners = {
            owner_by_link[int(link_id)] for link_id in action.links if int(link_id) in owner_by_link
        }
        max_touched = max(max_touched, len(owners))
    if max_touched > 1:
        return "plaquette_touches_multiple_blocks", "not_checked", max_touched

    reference = _factorized_padding_reference_config(model, fixed_blocks, padding)
    if not _global_configs_satisfy_qdm_constraints(model, reference):
        return "constraint_violation", "not_checked", max_touched

    sector_validation = "disabled"
    if config.include_sectors:
        sector_validation = "reference_and_single_block_variations"
        if not _global_configs_satisfy_model_sectors(model, reference):
            return "sector_violation", sector_validation, max_touched
        for block_index, block in enumerate(fixed_blocks):
            for support_index in range(block.support_size):
                support_indices = [0] * len(fixed_blocks)
                support_indices[block_index] = int(support_index)
                varied = _factorized_padding_reference_config(
                    model,
                    fixed_blocks,
                    padding,
                    support_indices=support_indices,
                )
                if not _global_configs_satisfy_model_sectors(model, varied):
                    return "sector_variation", sector_validation, max_touched

    if config.require_static_exterior:
        block_link_set = set(owner_by_link)
        for action in _qdm_global_plaquette_actions(model):
            if any(int(link_id) in block_link_set for link_id in action.links):
                continue
            if _qdm_plaquette_is_flippable_from_action(reference, action):
                return "nonstatic_exterior", sector_validation, max_touched

    return None, sector_validation, max_touched


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


def _make_qdm_multi_padding_from_exterior(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    *,
    exterior_link_ids: npt.NDArray[np.int64],
    exterior_config: npt.NDArray[np.int64],
) -> MultiLocalQDMPadding:
    fixed_blocks = tuple(blocks)
    support_ranges = [range(int(block.support_size)) for block in fixed_blocks]
    support_tuples = list(itertools.product(*support_ranges))
    n_support = len(support_tuples)
    n_global_links = int(model.lattice.num_links)

    full_configs = np.zeros((n_support, n_global_links), dtype=np.int64)
    amplitudes = np.ones(n_support, dtype=np.complex128)
    block_support_indices = np.zeros((n_support, len(fixed_blocks)), dtype=np.int64)
    exterior_link_ids = np.asarray(exterior_link_ids, dtype=np.int64)
    exterior_config = np.asarray(exterior_config, dtype=np.int64)

    for row_index, support_tuple in enumerate(support_tuples):
        if exterior_link_ids.size:
            full_configs[row_index, exterior_link_ids] = exterior_config
        for block_position, (block, support_index) in enumerate(
            zip(fixed_blocks, support_tuple, strict=True)
        ):
            support_index = int(support_index)
            full_configs[row_index, np.asarray(block.link_ids, dtype=np.int64)] = (
                block.support_configs[support_index]
            )
            amplitudes[row_index] *= complex(block.amplitudes[support_index])
            block_support_indices[row_index, block_position] = support_index

    return MultiLocalQDMPadding(
        block_ids=tuple(int(block.block_id) for block in fixed_blocks),
        exterior_link_ids=exterior_link_ids.copy(),
        exterior_config=exterior_config.copy(),
        global_support_configs=full_configs,
        global_amplitudes=amplitudes,
        block_support_indices=block_support_indices,
    )


def _multi_padding_passes_global_filters(
    model: object,
    padding: MultiLocalQDMPadding,
    blocks: Sequence[LocalQDMCageBlock],
    config: LocalQDMMultiPaddingConfig,
) -> bool:
    if not _global_configs_satisfy_qdm_constraints(model, padding.global_support_configs):
        return False
    if config.include_sectors and not _global_configs_satisfy_model_sectors(
        model,
        padding.global_support_configs,
    ):
        return False
    if config.require_static_exterior and not _multi_padding_has_static_exterior(
        model,
        padding,
        blocks,
    ):
        return False
    return True


def _global_configs_satisfy_qdm_constraints(
    model: object,
    configs: npt.ArrayLike,
) -> bool:
    required_count = int(getattr(model, "required_count", 1))
    arr = np.asarray(configs, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    for config_row in arr:
        for site_id in range(int(model.lattice.num_sites)):
            incident = np.asarray(model.lattice.incident_links(int(site_id)), dtype=np.int64)
            if int(np.sum(config_row[incident])) != required_count:
                return False
    return True


def _global_configs_satisfy_model_sectors(
    model: object,
    configs: npt.ArrayLike,
) -> bool:
    sectors = tuple(model.make_sectors())
    if not sectors:
        return True
    arr = np.asarray(configs, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    for config_row in arr:
        for sector in sectors:
            if not sector.is_satisfied(config_row):
                return False
    return True


def _qdm_multi_block_certification_actions(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    config: LocalQDMMultiPaddingConfig,
) -> tuple[_QDMGlobalPlaquetteAction, ...]:
    actions = _qdm_global_plaquette_actions(model)
    if not config.require_static_exterior:
        return actions

    block_link_set = {
        int(link_id) for block in blocks for link_id in np.asarray(block.link_ids, dtype=np.int64)
    }
    return tuple(
        action
        for action in actions
        if any(int(link_id) in block_link_set for link_id in action.links)
    )


def _multi_padding_has_static_exterior(
    model: object,
    padding: MultiLocalQDMPadding,
    blocks: Sequence[LocalQDMCageBlock],
) -> bool:
    block_link_set = {
        int(link_id) for block in blocks for link_id in np.asarray(block.link_ids, dtype=np.int64)
    }
    if padding.global_support_configs.shape[0] == 0:
        return True

    # Plaquettes disjoint from every block only see the shared exterior config,
    # so one support row is enough.  Avoid constructing flipped configs here;
    # we only need to know whether an exterior plaquette is flippable.
    reference_config = padding.global_support_configs[0]
    for action in _qdm_global_plaquette_actions(model):
        if any(int(link_id) in block_link_set for link_id in action.links):
            continue
        if _qdm_plaquette_is_flippable_from_action(reference_config, action):
            return False
    return True


def find_shared_qdm_exterior_paddings(
    model: object,
    local_record: LocalQDMCageRecord,
    *,
    config: LocalQDMPaddingConfig | None = None,
) -> list[LocalQDMPadding]:
    """Find shared exterior configurations compatible with a local QDM cage.

    A shared exterior is a single assignment on all nonlocal links such that
    every local support configuration becomes a full valid dimer covering.  This
    is the simplest product padding that preserves the local superposition.
    """
    padding_config = LocalQDMPaddingConfig() if config is None else config
    local_link_ids = np.asarray(local_record.local_link_ids, dtype=np.int64)
    local_link_set = set(int(link_id) for link_id in local_link_ids)
    local_index_by_link = {int(link_id): i for i, link_id in enumerate(local_link_ids)}

    n_global_links = int(model.lattice.num_links)
    exterior_link_ids = np.asarray(
        [link_id for link_id in range(n_global_links) if link_id not in local_link_set],
        dtype=np.int64,
    )
    exterior_index_by_link = {int(link_id): i for i, link_id in enumerate(exterior_link_ids)}
    n_exterior = int(exterior_link_ids.size)

    support_configs = np.asarray(local_record.support_configs, dtype=np.int64)
    if support_configs.ndim != 2:
        raise ValueError("local_record.support_configs must have shape (support, n_local_links).")

    required_count = int(getattr(model, "required_count", 1))
    site_targets: dict[int, int] = {}
    site_exterior_links: dict[int, npt.NDArray[np.int64]] = {}

    for site_id in range(int(model.lattice.num_sites)):
        incident = [int(link_id) for link_id in model.lattice.incident_links(int(site_id))]
        local_incident = [
            local_index_by_link[link_id] for link_id in incident if link_id in local_index_by_link
        ]
        exterior_incident = [
            exterior_index_by_link[link_id]
            for link_id in incident
            if link_id in exterior_index_by_link
        ]

        if local_incident:
            local_counts = np.sum(support_configs[:, local_incident], axis=1).astype(np.int64)
        else:
            local_counts = np.zeros(support_configs.shape[0], dtype=np.int64)

        if np.unique(local_counts).size != 1:
            return []

        target = required_count - int(local_counts[0])
        if target < 0 or target > len(exterior_incident):
            return []

        site_targets[int(site_id)] = int(target)
        site_exterior_links[int(site_id)] = np.asarray(exterior_incident, dtype=np.int64)

    if n_exterior == 0:
        exterior_config = np.zeros(0, dtype=np.int64)
        padding = _make_qdm_padding_from_exterior(
            model,
            local_record,
            exterior_link_ids=exterior_link_ids,
            exterior_config=exterior_config,
        )
        if _padding_passes_global_filters(model, padding, local_record, padding_config):
            return [padding]
        return []

    variable_order = _qdm_exterior_variable_order(
        model,
        exterior_link_ids,
        site_exterior_links,
        site_targets,
        fixed_link_sets=[local_link_set],
        require_static_exterior=padding_config.require_static_exterior,
    )

    exterior_config = np.zeros(n_exterior, dtype=np.int64)
    assigned = np.zeros(n_exterior, dtype=bool)
    sites_by_exterior_variable: list[list[int]] = [[] for _ in range(n_exterior)]
    for site_id, exterior_indices in site_exterior_links.items():
        for exterior_index in exterior_indices:
            sites_by_exterior_variable[int(exterior_index)].append(int(site_id))

    static_exterior_plaquettes_by_variable = (
        _qdm_static_exterior_plaquettes_by_variable(
            model,
            exterior_link_ids,
            fixed_link_set=local_link_set,
        )
        if padding_config.require_static_exterior
        else [[] for _ in range(n_exterior)]
    )

    paddings: list[LocalQDMPadding] = []
    nodes_visited = 0

    def partial_site_check(site_id: int) -> bool:
        exterior_indices = site_exterior_links[site_id]
        target = site_targets[site_id]
        if exterior_indices.size == 0:
            return target == 0
        assigned_local = assigned[exterior_indices]
        occupied = int(np.sum(exterior_config[exterior_indices[assigned_local]]))
        unassigned = int(exterior_indices.size - np.count_nonzero(assigned_local))
        if occupied > target:
            return False
        if occupied + unassigned < target:
            return False
        if unassigned == 0 and occupied != target:
            return False
        return True

    def full_check() -> bool:
        for site_id in range(int(model.lattice.num_sites)):
            if not partial_site_check(int(site_id)):
                return False
        return True

    def dfs(depth: int) -> None:
        nonlocal nodes_visited
        if len(paddings) >= padding_config.max_paddings_per_record:
            return
        if (
            padding_config.max_dfs_nodes is not None
            and nodes_visited >= padding_config.max_dfs_nodes
        ):
            return
        nodes_visited += 1

        if depth == n_exterior:
            if full_check():
                padding = _make_qdm_padding_from_exterior(
                    model,
                    local_record,
                    exterior_link_ids=exterior_link_ids,
                    exterior_config=exterior_config.copy(),
                )
                if _padding_passes_global_filters(model, padding, local_record, padding_config):
                    paddings.append(padding)
            return

        exterior_variable = int(variable_order[depth])
        for value in _qdm_exterior_value_order(
            exterior_variable,
            exterior_config=exterior_config,
            assigned=assigned,
            sites_by_exterior_variable=sites_by_exterior_variable,
            site_exterior_links=site_exterior_links,
            site_targets=site_targets,
        ):
            if len(paddings) >= padding_config.max_paddings_per_record:
                return
            exterior_config[exterior_variable] = value
            assigned[exterior_variable] = True
            touched_sites = sites_by_exterior_variable[exterior_variable]
            touched_static_plaquettes = static_exterior_plaquettes_by_variable[exterior_variable]
            if all(partial_site_check(site_id) for site_id in touched_sites) and (
                not touched_static_plaquettes
                or _qdm_static_exterior_checks_pass(
                    touched_static_plaquettes,
                    exterior_config=exterior_config,
                    assigned=assigned,
                )
            ):
                dfs(depth + 1)
            assigned[exterior_variable] = False
            exterior_config[exterior_variable] = 0

    dfs(0)
    return paddings


def build_qdm_global_limited_kinetic_matrix(
    model: object,
    basis: Basis,
) -> scipy_sparse.csr_array:
    """Build QDM kinetic transitions restricted to an explicitly supplied basis."""
    n = int(basis.n_states)
    if n == 0:
        return scipy_sparse.csr_array((0, 0), dtype=np.complex128)

    config_to_index = {_config_key(config): i for i, config in enumerate(basis.states)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    actions = _qdm_global_plaquette_actions(model)
    for col, config_row in enumerate(basis.states):
        for action in actions:
            transition = _qdm_flip_transition_from_action(config_row, action)
            if transition is None:
                continue
            final_config, coefficient = transition
            row = config_to_index.get(_config_key(final_config))
            if row is None:
                continue
            rows.append(int(row))
            cols.append(int(col))
            data.append(complex(coefficient))

    return scipy_sparse.coo_array(
        (np.asarray(data, dtype=np.complex128), (rows, cols)),
        shape=(n, n),
        dtype=np.complex128,
    ).tocsr()


def qdm_global_self_loop_values(
    model: object,
    configs: npt.ArrayLike,
) -> npt.NDArray[np.complex128]:
    """Compute full QDM potential/self-loop values for explicit configs."""
    return _qdm_global_self_loop_values_from_actions(
        configs,
        _qdm_global_plaquette_actions(model),
    )


def _certify_qdm_padding(
    model: object,
    local_record: LocalQDMCageRecord,
    padding: LocalQDMPadding,
    *,
    local_record_index: int,
    padding_index: int,
    config: LocalQDMPaddingConfig,
) -> LocalQDMCertificationReport | None:
    amplitudes = np.asarray(local_record.local_state, dtype=np.complex128)
    norm = float(np.linalg.norm(amplitudes))
    if norm == 0.0:
        return None
    amplitudes = amplitudes / norm

    support_configs = np.asarray(padding.global_support_configs, dtype=np.int64)
    support_keys = [_config_key(config_row) for config_row in support_configs]
    support_amplitude_by_key = {
        key: complex(amplitude) for key, amplitude in zip(support_keys, amplitudes, strict=True)
    }

    action_by_key: dict[tuple[int, ...], complex] = defaultdict(complex)
    touched_keys: set[tuple[int, ...]] = set(support_keys)

    for source_config, source_amplitude in zip(support_configs, amplitudes, strict=True):
        for plaquette_id in model.plaquette_ids():
            transition = _qdm_flip_transition(model, source_config, int(plaquette_id))
            if transition is None:
                continue
            final_config, coefficient = transition
            final_key = _config_key(final_config)
            action_by_key[final_key] += complex(coefficient) * complex(source_amplitude)
            touched_keys.add(final_key)

    kappa = complex(local_record.kappa)
    support_kinetic_residuals: list[complex] = []
    leakage_values: list[complex] = []
    leakage_configs: list[npt.NDArray[np.int64]] = []

    for key in sorted(touched_keys):
        action = complex(action_by_key.get(key, 0.0 + 0.0j))
        if key in support_amplitude_by_key:
            expected = kappa * support_amplitude_by_key[key]
            support_kinetic_residuals.append(action - expected)
        else:
            leakage_values.append(action)
            leakage_configs.append(np.asarray(key, dtype=np.int64))

    support_kinetic_residual = float(np.linalg.norm(np.asarray(support_kinetic_residuals)))
    leakage_residual = float(np.linalg.norm(np.asarray(leakage_values, dtype=np.complex128)))

    if leakage_residual > config.tolerance:
        return None
    if support_kinetic_residual > config.tolerance:
        return None

    support_self_loops = qdm_global_self_loop_values(model, support_configs)
    self_loop_value = complex(support_self_loops[0]) if support_self_loops.size else 0.0 + 0.0j
    if np.linalg.norm(support_self_loops - self_loop_value) > config.tolerance:
        return None

    energy = self_loop_value + kappa
    support_h_residuals = []
    for key, amplitude, self_loop in zip(
        support_keys,
        amplitudes,
        support_self_loops,
        strict=True,
    ):
        kinetic_action = complex(action_by_key.get(key, 0.0 + 0.0j))
        support_h_residuals.append(
            kinetic_action + complex(self_loop) * amplitude - energy * amplitude
        )
    support_hamiltonian_residual = float(
        np.linalg.norm(np.asarray(support_h_residuals, dtype=np.complex128))
    )
    full_residual = float(np.hypot(support_hamiltonian_residual, leakage_residual))
    if full_residual > config.tolerance:
        return None

    signature = signature_from_energy_and_self_loop(
        energy,
        self_loop_value,
        tolerance=max(config.tolerance, 1.0e-15) * 10.0,
        potential_unit=_infer_potential_unit_from_model(model),
    )
    if signature is None:
        return None

    leakage_arr = (
        np.asarray(leakage_configs, dtype=np.int64)
        if leakage_configs
        else np.empty((0, int(model.lattice.num_links)), dtype=np.int64)
    )

    return LocalQDMCertificationReport(
        local_record_index=int(local_record_index),
        padding_index=int(padding_index),
        signature=signature,
        energy=energy,
        kinetic_eigenvalue=kappa,
        self_loop_value=self_loop_value,
        support_size=int(support_configs.shape[0]),
        one_hop_shell_size=int(len(touched_keys)),
        leakage_residual=leakage_residual,
        support_kinetic_residual=support_kinetic_residual,
        support_hamiltonian_residual=support_hamiltonian_residual,
        full_residual=full_residual,
        padding=padding,
        leakage_configs=leakage_arr,
    )


def _make_qdm_padding_from_exterior(
    model: object,
    local_record: LocalQDMCageRecord,
    *,
    exterior_link_ids: npt.NDArray[np.int64],
    exterior_config: npt.NDArray[np.int64],
) -> LocalQDMPadding:
    local_link_ids = np.asarray(local_record.local_link_ids, dtype=np.int64)
    support_configs = np.asarray(local_record.support_configs, dtype=np.int64)
    full_configs = np.zeros(
        (support_configs.shape[0], int(model.lattice.num_links)),
        dtype=np.int64,
    )
    full_configs[:, local_link_ids] = support_configs
    if exterior_link_ids.size:
        full_configs[:, exterior_link_ids] = np.asarray(exterior_config, dtype=np.int64)
    return LocalQDMPadding(
        exterior_link_ids=np.asarray(exterior_link_ids, dtype=np.int64).copy(),
        exterior_config=np.asarray(exterior_config, dtype=np.int64).copy(),
        global_support_configs=full_configs,
    )


def _padding_passes_global_filters(
    model: object,
    padding: LocalQDMPadding,
    local_record: LocalQDMCageRecord,
    config: LocalQDMPaddingConfig,
) -> bool:
    if not _padding_satisfies_qdm_constraints(model, padding):
        return False
    if config.include_sectors and not _padding_satisfies_model_sectors(model, padding):
        return False
    if config.require_static_exterior and not _padding_has_static_exterior(
        model,
        padding,
        local_record,
    ):
        return False
    return True


def _padding_satisfies_qdm_constraints(model: object, padding: LocalQDMPadding) -> bool:
    required_count = int(getattr(model, "required_count", 1))
    for config_row in padding.global_support_configs:
        for site_id in range(int(model.lattice.num_sites)):
            incident = np.asarray(model.lattice.incident_links(int(site_id)), dtype=np.int64)
            if int(np.sum(config_row[incident])) != required_count:
                return False
    return True


def _padding_satisfies_model_sectors(model: object, padding: LocalQDMPadding) -> bool:
    sectors = tuple(model.make_sectors())
    if not sectors:
        return True
    for config_row in padding.global_support_configs:
        for sector in sectors:
            if not sector.is_satisfied(config_row):
                return False
    return True


def _padding_has_static_exterior(
    model: object,
    padding: LocalQDMPadding,
    local_record: LocalQDMCageRecord,
) -> bool:
    local_link_set = set(int(link_id) for link_id in local_record.local_link_ids)
    if padding.global_support_configs.shape[0] == 0:
        return True

    reference_config = padding.global_support_configs[0]
    for action in _qdm_global_plaquette_actions(model):
        if any(int(link_id) in local_link_set for link_id in action.links):
            continue
        if _qdm_plaquette_is_flippable_from_action(reference_config, action):
            return False
    return True


def _qdm_global_plaquette_actions(
    model: object,
    plaquette_ids: Sequence[int] | None = None,
) -> tuple[_QDMGlobalPlaquetteAction, ...]:
    source_ids = model.plaquette_ids() if plaquette_ids is None else plaquette_ids
    ids = tuple(int(pid) for pid in source_ids)
    actions: list[_QDMGlobalPlaquetteAction] = []
    for plaquette_id in ids:
        links = np.asarray(model.lattice.plaquette_links(int(plaquette_id)), dtype=np.int64)
        pattern0, pattern1 = alternating_binary_patterns(int(links.size))
        coupling = model._coup_kin_at(int(plaquette_id))
        actions.append(
            _QDMGlobalPlaquetteAction(
                plaquette_id=int(plaquette_id),
                links=links,
                pattern0=np.asarray(pattern0, dtype=np.int64),
                pattern1=np.asarray(pattern1, dtype=np.int64),
                forward=complex(_forward_coefficient(coupling)),
                backward=complex(_backward_coefficient(coupling)),
                potential=complex(model._coup_pot_at(int(plaquette_id))),
            )
        )
    return tuple(actions)


def _qdm_flip_transition_from_action(
    config_row: npt.ArrayLike,
    action: _QDMGlobalPlaquetteAction,
) -> tuple[npt.NDArray[np.int64], complex] | None:
    config_arr = np.asarray(config_row, dtype=np.int64)
    values = config_arr[action.links]
    if np.array_equal(values, action.pattern0):
        final = config_arr.copy()
        final[action.links] = action.pattern1
        return final, action.forward
    if np.array_equal(values, action.pattern1):
        final = config_arr.copy()
        final[action.links] = action.pattern0
        return final, action.backward
    return None


def _qdm_plaquette_is_flippable_from_action(
    config_row: npt.ArrayLike,
    action: _QDMGlobalPlaquetteAction,
) -> bool:
    config_arr = np.asarray(config_row, dtype=np.int64)
    values = config_arr[action.links]
    return bool(np.array_equal(values, action.pattern0) or np.array_equal(values, action.pattern1))


def _qdm_flip_transition(
    model: object,
    config_row: npt.ArrayLike,
    plaquette_id: int,
) -> tuple[npt.NDArray[np.int64], complex] | None:
    action = _qdm_global_plaquette_actions(model, (int(plaquette_id),))[0]
    return _qdm_flip_transition_from_action(config_row, action)


def _qdm_global_self_loop_values_from_actions(
    configs: npt.ArrayLike,
    actions: Sequence[_QDMGlobalPlaquetteAction],
) -> npt.NDArray[np.complex128]:
    arr = np.asarray(configs, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    values = np.zeros(arr.shape[0], dtype=np.complex128)
    for action in actions:
        local_values = arr[:, action.links]
        flippable = np.all(local_values == action.pattern0, axis=1) | np.all(
            local_values == action.pattern1,
            axis=1,
        )
        if np.any(flippable):
            values[flippable] += action.potential
    return values


def _qdm_global_self_loop_value(model: object, config_row: npt.ArrayLike) -> complex:
    return complex(
        _qdm_global_self_loop_values_from_actions(
            config_row,
            _qdm_global_plaquette_actions(model),
        )[0]
    )


def _config_key(config_row: npt.ArrayLike) -> tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(config_row, dtype=np.int64))


def _cage_search_config_from_local_and_padding(
    local_config: LocalQDMCageSearchConfig,
    padding_config: LocalQDMPaddingConfig,
) -> CageSearchConfig:
    return CageSearchConfig(
        search_type="type1",
        tolerance=min(local_config.tolerance, padding_config.tolerance),
        min_component_size=local_config.min_component_size,
        validate_full_residual=local_config.validate_full_residual,
        type1_kappas=local_config.allowed_kappas,
        deduplicate_by_rank=False,
        potential_signature_unit=local_config.potential_signature_unit,
        store_full_states=padding_config.store_full_states,
    )


def _cage_search_config_from_multi_padding(
    model: object,
    padding_config: LocalQDMMultiPaddingConfig,
    reports: Sequence[MultiLocalQDMCertificationReport],
) -> CageSearchConfig:
    kappas = tuple(sorted({int(report.signature[0]) for report in reports})) or (0,)
    return CageSearchConfig(
        search_type="type1",
        tolerance=padding_config.tolerance,
        min_component_size=1,
        validate_full_residual=True,
        type1_kappas=kappas,
        deduplicate_by_rank=False,
        potential_signature_unit=_infer_potential_unit_from_model(model),
        store_full_states=padding_config.store_full_states,
    )


find_qdm_multi_block_paddings = find_multi_qdm_block_paddings
