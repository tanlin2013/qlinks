"""QDM local and multi-block cage certification.

This module owns residual-based certification and result assembly. Exterior-padding search,
global QDM action primitives, and factorized-product contraction live in focused lower layers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt
import scipy.sparse as scipy_sparse

from qlinks.basis import Basis
from qlinks.caging.candidate import CandidateSubgraph
from qlinks.caging.local_search_global import (
    _config_key,
    _qdm_flip_transition,
    _qdm_flip_transition_from_action,
    _qdm_global_self_loop_values_from_actions,
    build_qdm_global_limited_kinetic_matrix,
    qdm_global_self_loop_values,
)
from qlinks.caging.local_search_padding import (
    _multi_padding_has_static_exterior,
    _qdm_action_plaquette_class,
    _qdm_multi_block_certification_actions,
    find_shared_qdm_exterior_paddings,
    iter_multi_qdm_block_paddings,
)
from qlinks.caging.local_search_qdm import _infer_potential_unit_from_model
from qlinks.caging.local_search_types import (
    CertifiedLocalQDMCageSearchResult,
    LocalQDMCageBlock,
    LocalQDMCageRecord,
    LocalQDMCageSearchConfig,
    LocalQDMCageSearchResult,
    LocalQDMCertificationReport,
    LocalQDMMultiPaddingConfig,
    LocalQDMPadding,
    LocalQDMPaddingConfig,
    MultiLocalQDMCertificationReport,
    MultiLocalQDMPadding,
    QDMMultiPaddingDiagnostics,
    QDMMultiPaddingFailureReport,
)
from qlinks.caging.results import CageState
from qlinks.caging.search import (
    CageRecord,
    CageSearchConfig,
    CageSearchResult,
    signature_from_energy_and_self_loop,
)


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
