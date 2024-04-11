import numpy as np
import pytest

from qlinks.computation_basis import ComputationBasis


@pytest.fixture(scope="function")
def qlm_2x2_basis() -> ComputationBasis:
    """Zero charge distribution, flux sector (0, 0).

    o◄──────o◄──────o    o◄──────o◄──────o   o◄──────o──────►o
    │       ▲       │    ▲       │       ▲   │       ▲       │
    │       │       │    │       │       │   │       │       │
    ▼       │       ▼    │       ▼       │   ▼       │       ▼
    o──────►o──────►o    o──────►o──────►o   o──────►o◄──────o
    │       ▲       │    ▲       │       ▲   ▲       │       ▲
    │       │       │    │       │       │   │       │       │
    ▼       │       ▼    │       ▼       │   │       ▼       │
    o◄──────o◄──────o    o◄──────o◄──────o   o◄──────o──────►o

            27                   78                 105

    o──────►o◄──────o    o──────►o──────►o   o──────►o──────►o
    ▲       │       ▲    │       ▲       │   ▲       │       ▲
    │       │       │    │       │       │   │       │       │
    │       ▼       │    ▼       │       ▼   │       ▼       │
    o◄──────o──────►o    o◄──────o◄──────o   o◄──────o◄──────o
    │       ▲       │    │       ▲       │   ▲       │       ▲
    │       │       │    │       │       │   │       │       │
    ▼       │       ▼    ▼       │       ▼   │       ▼       │
    o──────►o◄──────o    o──────►o──────►o   o──────►o──────►o

           150                  177                 228
    """
    basis = ComputationBasis(
        np.array(
            [
                [0, 0, 0, 1, 1, 0, 1, 1],  # 27
                [0, 1, 0, 0, 1, 1, 1, 0],  # 78
                [0, 1, 1, 0, 1, 0, 0, 1],  # 105
                [1, 0, 0, 1, 0, 1, 1, 0],  # 150
                [1, 0, 1, 1, 0, 0, 0, 1],  # 177
                [1, 1, 1, 0, 0, 1, 0, 0],  # 228
            ]
        )
    )
    basis.sort()
    return basis


@pytest.fixture(scope="function")
def qdm_4x2_basis() -> ComputationBasis:
    """Staggered charge distribution, flux sector (0, 0).

    .. image:: /docs/source/images/qdm_basis_4x2.png
        :width: 100px
        :align: center
    """
    basis = ComputationBasis(
        np.array(
            [
                [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1],  # 17595
                [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],  # 18867
                [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0],  # 19638
                [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1],  # 36135
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],  # 37947
                [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],  # 39219
                [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],  # 39990
                [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],  # 44307
                [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],  # 50283
                [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1],  # 51555
                [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0],  # 52326
                [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],  # 52773
                [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],  # 55410
                [1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],  # 55857
                [1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],  # 60498
                [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # 60945
            ]
        )
    )
    basis.sort()
    return basis


@pytest.fixture(scope="function")
def qdm_4x4_basis() -> ComputationBasis:
    """Staggered charge distribution, flux sector (0, 0).

    .. image:: /docs/source/images/qdm_basis_4x4.png
        :width: 100px
        :align: center
    """
    basis = ComputationBasis.from_parquet("qdm_4x4_basis.parquet")
    basis.sort()
    return basis
