"""Execution helpers for proposal-driven local cage scans.

Proposal objects only describe candidate regions.  This module owns the side that executes
local searches over those regions and converts successful QDM records into certification
blocks.  Keeping execution separate prevents proposal generation from depending on the
certification layer.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from qlinks.caging.local_search.core import LocalCageSearcher
from qlinks.caging.local_search.padding import make_qdm_cage_block
from qlinks.caging.local_search.types import (
    LocalCageModelAdapter,
    LocalQDMCageBlock,
    LocalQDMCageSearchConfig,
    LocalQDMRegion,
    LocalRegionProposal,
    LocalRegionProposalSearchRecord,
    LocalRegionProposalSearchResult,
)


def run_local_region_proposal(
    proposal: LocalRegionProposal,
    *,
    model: object | None = None,
    config: LocalQDMCageSearchConfig | None = None,
    adapter: LocalCageModelAdapter | None = None,
    max_regions: int | None = None,
) -> LocalRegionProposalSearchResult:
    """Run the local cage searcher over every region emitted by one proposal."""
    return run_local_region_proposals(
        [proposal],
        model=model,
        config=config,
        adapter=adapter,
        max_regions=max_regions,
    )


def run_local_region_proposals(
    proposals: Sequence[LocalRegionProposal],
    *,
    model: object | None = None,
    config: LocalQDMCageSearchConfig | None = None,
    adapter: LocalCageModelAdapter | None = None,
    max_regions: int | None = None,
) -> LocalRegionProposalSearchResult:
    """Run local cage searches over a stream of proposal-generated regions.

    The helper is intentionally lightweight: proposal objects only need to
    provide ``iter_regions()``.  If they provide richer ``iter_records()``
    records with a ``region`` attribute, that metadata is retained in the scan
    result.  ``StripeRegionProposal`` follows this richer path.
    """
    if max_regions is not None and max_regions < 0:
        raise ValueError("max_regions must be non-negative or None.")

    search_records: list[LocalRegionProposalSearchRecord] = []
    emitted = 0
    for proposal_index, proposal in enumerate(proposals):
        proposal_model = _model_for_region_proposal(proposal, model)
        proposal_adapter = _adapter_for_region_proposal(proposal, adapter)
        proposal_config = _config_for_region_proposal(proposal, config)

        for region_index, proposal_record, region in _iter_region_proposal_records(proposal):
            if max_regions is not None and emitted >= max_regions:
                return LocalRegionProposalSearchResult(records=search_records)
            result = LocalCageSearcher(
                model=proposal_model,
                region=region,
                config=proposal_config,
                adapter=proposal_adapter,
            ).run()
            search_records.append(
                LocalRegionProposalSearchRecord(
                    proposal_index=proposal_index,
                    region_index=region_index,
                    region=region,
                    result=result,
                    proposal_record=proposal_record,
                )
            )
            emitted += 1

    return LocalRegionProposalSearchResult(records=search_records)


def collect_qdm_cage_blocks_with_scan_from_region_proposals(
    proposals: Sequence[LocalRegionProposal],
    *,
    model: object | None = None,
    config: LocalQDMCageSearchConfig | None = None,
    adapter: LocalCageModelAdapter | None = None,
    signatures: Sequence[tuple[int, int]] | None = None,
    max_regions: int | None = None,
    max_records_per_region: int | None = None,
    max_blocks: int | None = None,
    block_id_start: int = 0,
    skip_incompatible_blocks: bool = True,
) -> tuple[LocalRegionProposalSearchResult, list[LocalQDMCageBlock]]:
    """Run proposal searches and stream compatible QDM blocks.

    This is the block-oriented counterpart of :func:`run_local_region_proposals`.
    It converts records into ``LocalQDMCageBlock`` objects immediately after each
    region is searched and stops as soon as ``max_blocks`` is reached.  This is
    important for expensive proposal portfolios: the older two-stage workflow
    searched every proposed region first and only then applied the block cap, so
    robust scans could spend most of their time in local DFS branches that would
    never contribute to the requested block pool.
    """
    if block_id_start < 0:
        raise ValueError("block_id_start must be non-negative.")
    if max_regions is not None and max_regions < 0:
        raise ValueError("max_regions must be non-negative or None.")
    if max_records_per_region is not None and max_records_per_region < 0:
        raise ValueError("max_records_per_region must be non-negative or None.")
    if max_blocks is not None and max_blocks < 0:
        raise ValueError("max_blocks must be non-negative or None.")

    signature_filter = None
    if signatures is not None:
        signature_filter = {(int(kappa), int(potential)) for kappa, potential in signatures}

    search_records: list[LocalRegionProposalSearchRecord] = []
    blocks: list[LocalQDMCageBlock] = []
    emitted_regions = 0
    next_block_id = int(block_id_start)

    if max_blocks == 0:
        return LocalRegionProposalSearchResult(records=[]), []

    for proposal_index, proposal in enumerate(proposals):
        proposal_model = _model_for_region_proposal(proposal, model)
        proposal_adapter = _adapter_for_region_proposal(proposal, adapter)
        proposal_config = _config_for_region_proposal(proposal, config)

        for region_index, proposal_record, region in _iter_region_proposal_records(proposal):
            if max_regions is not None and emitted_regions >= max_regions:
                return LocalRegionProposalSearchResult(records=search_records), blocks
            if max_blocks is not None and len(blocks) >= max_blocks:
                return LocalRegionProposalSearchResult(records=search_records), blocks

            result = LocalCageSearcher(
                model=proposal_model,
                region=region,
                config=proposal_config,
                adapter=proposal_adapter,
            ).run()
            search_records.append(
                LocalRegionProposalSearchRecord(
                    proposal_index=proposal_index,
                    region_index=region_index,
                    region=region,
                    result=result,
                    proposal_record=proposal_record,
                )
            )
            emitted_regions += 1

            region_records = result.records
            if signature_filter is not None:
                region_records = [
                    record for record in region_records if record.signature in signature_filter
                ]
            if max_records_per_region is not None:
                region_records = region_records[:max_records_per_region]

            for local_record in region_records:
                if max_blocks is not None and len(blocks) >= max_blocks:
                    return LocalRegionProposalSearchResult(records=search_records), blocks
                try:
                    block = make_qdm_cage_block(
                        proposal_model,
                        local_record,
                        block_id=next_block_id,
                    )
                except ValueError:
                    if skip_incompatible_blocks:
                        continue
                    raise
                blocks.append(block)
                next_block_id += 1

    return LocalRegionProposalSearchResult(records=search_records), blocks


def collect_qdm_cage_blocks_from_region_proposals(
    proposals: Sequence[LocalRegionProposal],
    *,
    model: object | None = None,
    config: LocalQDMCageSearchConfig | None = None,
    adapter: LocalCageModelAdapter | None = None,
    signatures: Sequence[tuple[int, int]] | None = None,
    max_regions: int | None = None,
    max_records_per_region: int | None = None,
    max_blocks: int | None = None,
    block_id_start: int = 0,
    skip_incompatible_blocks: bool = True,
) -> list[LocalQDMCageBlock]:
    """Run proposal searches and return a QDM block pool for multi-padding."""
    _, blocks = collect_qdm_cage_blocks_with_scan_from_region_proposals(
        proposals,
        model=model,
        config=config,
        adapter=adapter,
        signatures=signatures,
        max_regions=max_regions,
        max_records_per_region=max_records_per_region,
        max_blocks=max_blocks,
        block_id_start=block_id_start,
        skip_incompatible_blocks=skip_incompatible_blocks,
    )
    return blocks


def _model_for_region_proposal(
    proposal: LocalRegionProposal,
    model: object | None,
) -> object:
    if model is not None:
        return model
    proposal_model = getattr(proposal, "model", None)
    if proposal_model is None:
        raise ValueError("model must be provided when proposal has no model attribute.")
    return proposal_model


def _adapter_for_region_proposal(
    proposal: LocalRegionProposal,
    adapter: LocalCageModelAdapter | None,
) -> LocalCageModelAdapter | None:
    if adapter is not None:
        return adapter
    return getattr(proposal, "adapter", None)


def _config_for_region_proposal(
    proposal: LocalRegionProposal,
    config: LocalQDMCageSearchConfig | None,
) -> LocalQDMCageSearchConfig:
    if config is not None:
        return config
    proposal_config = getattr(proposal, "config", None)
    if proposal_config is None:
        return LocalQDMCageSearchConfig()
    return proposal_config


def _iter_region_proposal_records(
    proposal: LocalRegionProposal,
) -> Iterator[tuple[int, object | None, LocalQDMRegion]]:
    if hasattr(proposal, "iter_records"):
        for region_index, proposal_record in enumerate(proposal.iter_records()):
            region = getattr(proposal_record, "region", None)
            if region is None:
                raise ValueError("proposal iter_records() entries must carry a region attribute.")
            yield region_index, proposal_record, region
        return

    for region_index, region in enumerate(proposal.iter_regions()):
        yield region_index, None, region


collect_qdm_cage_blocks_from_proposals = collect_qdm_cage_blocks_from_region_proposals
collect_qdm_cage_blocks_with_scan_from_proposals = (
    collect_qdm_cage_blocks_with_scan_from_region_proposals
)
