"""High-level robust local-search workflows."""

from __future__ import annotations

from qlinks.caging.local_search_certification import (
    _deduplicate_qdm_multi_block_reports,
    certified_qdm_result_from_multi_block_reports,
    diagnose_qdm_multi_block_paddings,
    qdm_multi_padding_config_schedule,
    robust_certify_qdm_multi_block_result,
)
from qlinks.caging.local_search_proposals import (
    _robust_qdm_region_proposals,
    collect_qdm_cage_blocks_from_region_proposals,
    collect_qdm_cage_blocks_with_scan_from_region_proposals,
)
from qlinks.caging.local_search_types import (
    CertifiedLocalQDMCageSearchResult,
    LocalCageModelAdapter,
    MultiLocalQDMCertificationReport,
    QDMMultiPaddingDiagnostics,
    RobustQDMLocalCageSearchConfig,
    RobustQDMLocalCageSearchContext,
)


def robust_qdm_local_cage_search(
    model: object,
    *,
    config: RobustQDMLocalCageSearchConfig | None = None,
    adapter: LocalCageModelAdapter | None = None,
    return_context: bool = False,
) -> (
    CertifiedLocalQDMCageSearchResult
    | tuple[
        CertifiedLocalQDMCageSearchResult,
        RobustQDMLocalCageSearchContext,
    ]
):
    """Run a budgeted robust local QDM cage search.

    The search builds a portfolio of region proposals, converts successful local
    records into independent Lego blocks, then certifies the block pool with a
    permissive-to-strict padding schedule.  By default, the return value is the
    existing ``CertifiedLocalQDMCageSearchResult`` container used by downstream
    tools.  Pass ``return_context=True`` to also receive the intermediate scan,
    block pool, and per-stage diagnostics for debugging.
    """
    robust_config = RobustQDMLocalCageSearchConfig() if config is None else config
    proposals = _robust_qdm_region_proposals(model, robust_config, adapter=adapter)
    padding_config = robust_config.as_multi_padding_config()

    if not return_context:
        blocks = collect_qdm_cage_blocks_from_region_proposals(
            proposals,
            model=model,
            config=robust_config.local_config,
            adapter=adapter,
            signatures=robust_config.block_signatures,
            max_regions=None,
            max_records_per_region=robust_config.max_records_per_region,
            max_blocks=robust_config.max_blocks,
            skip_incompatible_blocks=robust_config.skip_incompatible_blocks,
        )
        if not blocks:
            return certified_qdm_result_from_multi_block_reports(
                model,
                [],
                config=padding_config,
            )
        return robust_certify_qdm_multi_block_result(
            model,
            blocks,
            config=padding_config,
            stages=robust_config.padding_stages,
        )

    scan, blocks = collect_qdm_cage_blocks_with_scan_from_region_proposals(
        proposals,
        model=model,
        config=robust_config.local_config,
        adapter=adapter,
        signatures=robust_config.block_signatures,
        max_records_per_region=robust_config.max_records_per_region,
        max_blocks=robust_config.max_blocks,
        skip_incompatible_blocks=robust_config.skip_incompatible_blocks,
    )

    diagnostics_by_stage: dict[str, QDMMultiPaddingDiagnostics] = {}
    all_reports: list[MultiLocalQDMCertificationReport] = []
    for stage_name, stage_config in qdm_multi_padding_config_schedule(
        padding_config,
        stages=robust_config.padding_stages,
    ):
        if blocks:
            diagnostics = diagnose_qdm_multi_block_paddings(model, blocks, config=stage_config)
        else:
            diagnostics = QDMMultiPaddingDiagnostics(
                paddings=[],
                reports=[],
                failures=[],
                config=stage_config,
            )
        diagnostics_by_stage[stage_name] = diagnostics
        all_reports.extend(diagnostics.reports)

    certified = certified_qdm_result_from_multi_block_reports(
        model,
        _deduplicate_qdm_multi_block_reports(all_reports),
        config=padding_config,
    )
    context = RobustQDMLocalCageSearchContext(
        config=robust_config,
        scan=scan,
        blocks=blocks,
        padding_config=padding_config,
        diagnostics_by_stage=diagnostics_by_stage,
    )
    return certified, context


robust_local_qdm_cage_search = robust_qdm_local_cage_search
