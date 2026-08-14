from __future__ import annotations

import pytest

from qlinks.caging.local_search import (
    RobustQDMLocalCageSearchConfig,
    RobustQDMLocalCageSearchContext,
    robust_qdm_local_cage_search,
)
from qlinks.models import SquareQDMModel
from tests.caging.local_search._helpers import (
    _square_qdm_stripe_pair_robust_config,
    _square_qdm_w00_model,
)


def test_robust_qdm_local_cage_search_accepts_stripe_motif_component_strategy() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
        coup_kin=1.0,
        coup_pot=1.0,
    )
    robust_config = RobustQDMLocalCageSearchConfig(
        region_strategies=("stripe_motif_component",),
        stripe_motif_sources=("stripe",),
        stripe_motif_sizes=(2,),
        stripe_motif_subset_mode="windows",
        stripe_motif_component_subset_mode="full",
        stripe_motif_component_max_seed_motifs_per_stripe=1,
        stripe_motif_component_max_components_per_stripe=1,
        stripe_widths=(1,),
        stripe_directions=(0,),
        max_regions_per_strategy=2,
        block_signatures=((0, 2),),
        max_records_per_region=1,
        min_blocks=1,
        max_blocks=2,
        max_paddings_per_stage=0,
        padding_stages=("static",),
        store_full_states=False,
    )

    _certified, context = robust_qdm_local_cage_search(
        model,
        config=robust_config,
        return_context=True,
    )

    assert context.n_regions <= 2
    assert context.stage_names == ("static",)
    assert context.padding_config == robust_config.as_multi_padding_config()


def test_robust_qdm_local_cage_search_accepts_stripe_motif_strategy() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
        coup_kin=1.0,
        coup_pot=1.0,
    )
    robust_config = RobustQDMLocalCageSearchConfig(
        region_strategies=("stripe_motif",),
        stripe_motif_sources=("stripe",),
        stripe_motif_sizes=(2,),
        stripe_motif_subset_mode="all",
        stripe_widths=(1,),
        stripe_directions=(0,),
        max_regions_per_strategy=4,
        block_signatures=((0, 2),),
        max_records_per_region=2,
        min_blocks=1,
        max_blocks=2,
        max_paddings_per_stage=0,
        padding_stages=("static",),
        store_full_states=False,
    )

    _certified, context = robust_qdm_local_cage_search(
        model,
        config=robust_config,
        return_context=True,
    )

    assert context.n_regions <= 4
    assert context.stage_names == ("static",)
    assert context.padding_config == robust_config.as_multi_padding_config()


def test_robust_config_can_use_snake_stripe_strategy() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
        coup_kin=1.0,
        coup_pot=1.0,
    )
    config = RobustQDMLocalCageSearchConfig(
        region_strategies=("snake_stripe",),
        max_region_plaquettes=4,
        min_region_plaquettes=4,
        max_regions_per_strategy=4,
        max_records_per_region=0,
        padding_stages=("static",),
    )

    certified, context = robust_qdm_local_cage_search(
        model,
        config=config,
        return_context=True,
    )

    assert certified.counts_by_signature == {}
    assert context.n_regions == 4
    assert context.n_blocks == 0


def test_robust_qdm_context_uses_streaming_block_collection() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
        coup_kin=1.0,
        coup_pot=1.0,
    )
    robust_config = RobustQDMLocalCageSearchConfig(
        region_strategies=("stripe",),
        stripe_widths=(1,),
        stripe_directions=(0,),
        max_regions_per_strategy=None,
        block_signatures=((0, 2),),
        max_records_per_region=2,
        min_blocks=2,
        max_blocks=2,
        max_paddings_per_stage=0,
        padding_stages=("static",),
        store_full_states=False,
    )

    _certified, context = robust_qdm_local_cage_search(
        model,
        config=robust_config,
        return_context=True,
    )

    assert context.n_blocks == 2
    assert context.n_regions == 1


def test_robust_qdm_local_cage_search_returns_certified_result_under_small_budget() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
        coup_kin=1.0,
        coup_pot=1.0,
    )

    robust_config = RobustQDMLocalCageSearchConfig(
        region_strategies=("stripe",),
        stripe_widths=(1,),
        stripe_directions=(0,),
        max_regions_per_strategy=2,
        block_signatures=((0, 2),),
        max_records_per_region=2,
        min_blocks=2,
        max_blocks=2,
        max_paddings_per_stage=2,
        max_paddings_per_packing=1,
        max_product_support_size=2048,
        include_sectors=True,
        padding_stages=("static",),
        tolerance=1.0e-9,
        store_full_states=False,
    )

    certified = robust_qdm_local_cage_search(model, config=robust_config)

    assert certified.hilbert_size == certified.basis.n_states
    assert isinstance(certified.counts_by_signature, dict)

    certified_with_context, context = robust_qdm_local_cage_search(
        model,
        config=robust_config,
        return_context=True,
    )

    assert certified_with_context.counts_by_signature == certified.counts_by_signature
    assert isinstance(context, RobustQDMLocalCageSearchContext)
    assert context.n_regions > 0
    assert context.n_blocks >= 0
    assert context.stage_names == ("static",)
    assert set(context.n_paddings_by_stage) == {"static"}
    assert set(context.n_certified_by_stage) == {"static"}
    assert set(context.failure_counts_by_stage) == {"static"}
    assert context.padding_config == robust_config.as_multi_padding_config()


def test_robust_square_qdm_4x4_w00_recovers_known_stripe_pair_cages() -> None:
    """Recover the eight known two-stripe local-padding cages in 4x4 W00 QDM.

    The full exact search has nine ``(0, 4)`` cages.  Empirically, eight of
    them decompose into two separated local ``(0, 2)`` stripe cages with one
    inactive spacer stripe between them.  This test keeps the local-first path
    honest without requiring full-basis ED or relying on the ninth non-stripe
    cage.
    """
    model = _square_qdm_w00_model(4, 4)
    certified, context = robust_qdm_local_cage_search(
        model,
        config=_square_qdm_stripe_pair_robust_config(),
        return_context=True,
    )

    assert certified.counts_by_signature == {(0, 4): 8}
    assert all(record.cage_state.support_size == 4 for record in certified.records)
    assert all(report.one_hop_shell_size == 12 for report in certified.reports)
    assert max(report.full_residual for report in certified.reports) < 1.0e-8
    assert max(report.leakage_residual for report in certified.reports) < 1.0e-8

    assert context.n_regions == 8
    assert context.n_blocks == 16
    assert context.n_paddings_by_stage == {"static": 16}
    assert context.n_certified_by_stage == {"static": 8}
    assert context.failure_counts_by_stage == {"static": {"leakage_residual": 8}}


@pytest.mark.manual
@pytest.mark.skip(
    reason=(
        "Manual diagnostic scaffold for larger square-QDM stripe stacking; "
        "enable locally when tuning robust local/padding budgets."
    )
)
def test_robust_square_qdm_4x8_w00_stripe_stacking_diagnostic_manual() -> None:
    """Diagnostic scaffold for the known 4xLy stripe-stacking family.

    The robust local-first solver is not expected to recover every exact cage,
    but the stripe-pair mechanism should produce non-empty certified results
    under a sufficiently permissive budget.  Keep this manual until the exact
    4x8 target counts and runtime budget are finalized.
    """
    model = _square_qdm_w00_model(4, 8)
    config = _square_qdm_stripe_pair_robust_config(
        max_paddings_per_stage=64,
        max_paddings_per_packing=16,
        max_product_support_size=1024,
        stripe_directions=(0,),
    )

    certified, context = robust_qdm_local_cage_search(
        model,
        config=config,
        return_context=True,
    )

    assert context.n_regions > 0
    assert context.n_blocks > 0
    assert context.n_paddings_by_stage["static"] > 0
    assert len(certified) > 0
    assert all(report.full_residual < 1.0e-8 for report in certified.reports)
