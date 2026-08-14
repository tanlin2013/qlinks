from __future__ import annotations

from qlinks.caging.local_search import LocalCageSearchConfig, LocalCageSearcher, QDMLocalCageAdapter
from qlinks.models import SquareQDMModel


def test_generic_local_cage_searcher_accepts_explicit_qdm_adapter() -> None:
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

    searcher = LocalCageSearcher.from_plaquettes(
        model,
        plaquette_ids=[0],
        config=LocalCageSearchConfig(
            halo_layers=1,
            boundary_mode="relaxed",
            tolerance=1.0e-10,
        ),
        adapter=QDMLocalCageAdapter(model),
    )
    result = searcher.run()

    assert result.local_hilbert_size > 0
    assert result.region.link_ids.size < model.lattice.num_links
