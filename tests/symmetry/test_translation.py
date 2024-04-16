import numpy as np
import pandas as pd
import pytest

from qlinks.lattice.square_lattice import SquareLattice
from qlinks.symmetry.translation import Translation


class TestTranslation:
    """
    (27) 00011011     (105) 01101001

    o◄────o◄────o     o◄────o────►o
    │     ▲     │     │     ▲     │
    ▼     │     ▼     ▼     │     ▼
    o────►o────►o     o────►o◄────o
    │     ▲     │     ▲     │     ▲
    ▼     │     ▼     │     ▼     │
    o◄────o◄────o     o◄────o────►o


    +-------------+-----+-----+-----+-----+-----+-----+
    | shift/basis | 27  | 78  | 105 | 150 | 177 | 228 |
    +-------------+-----+-----+-----+-----+-----+-----+
    | (0, 0)      |  27 |  78 | 105 | 150 | 177 | 228 |
    | (0, 1)      | 177 | 228 | 150 | 105 |  27 |  78 |
    | (1, 0)      |  78 |  27 | 150 | 105 | 228 | 177 |
    | (1, 1)      | 228 | 177 | 105 | 150 |  78 |  27 |
    +-------------+-----+-----+-----+-----+-----+-----+
    """

    @pytest.fixture(scope="function")
    def lattice(self, request):
        return SquareLattice(*request.param)

    @pytest.fixture(scope="function")
    def translation(self, request) -> Translation:
        model, size = request.param
        fixture = f"{model}_{'x'.join(map(str, size))}_basis"
        basis = request.getfixturevalue(fixture)
        return Translation(SquareLattice(*size), basis)

    @pytest.mark.parametrize(
        "translation, expected",
        [
            (
                ("qlm", (2, 2)),
                pd.DataFrame(
                    np.array(
                        [
                            [27, 78, 105, 150, 177, 228],
                            [177, 228, 150, 105, 27, 78],
                            [78, 27, 150, 105, 228, 177],
                            [228, 177, 105, 150, 78, 27],
                        ]
                    ),
                    columns=np.asarray([27, 78, 105, 150, 177, 228], dtype=object),
                    index=[(0, 0), (0, 1), (1, 0), (1, 1)],
                ),
            ),
            (
                ("qdm", (4, 2)),
                # fmt: off
                pd.DataFrame(
                    np.array(
                        [
                            [17595, 18867, 19638, 36135, 37947, 39219, 39990, 44307,
                             50283, 51555, 52326, 52773, 55410, 55857, 60498, 60945],
                            [47940, 45897, 46668, 10125, 15252, 13209, 13980, 5037,
                             27588, 25545, 26316, 9678, 29400, 12762, 21228, 4590],
                            [4590, 9678, 12762, 13980, 21228, 26316, 29400, 46668,
                             5037, 10125, 13209, 15252, 25545, 27588, 45897, 47940],
                            [60945, 52773, 55857, 39990, 60498, 52326, 55410, 19638,
                             44307, 36135, 39219, 37947, 51555, 50283, 18867, 17595],
                            [17595, 37947, 50283, 55410, 18867, 39219, 51555, 55857,
                             19638, 39990, 52326, 60498, 36135, 44307, 52773, 60945],
                            [47940, 15252, 27588, 29400, 45897, 13209, 25545, 12762,
                             46668, 13980, 26316, 21228, 10125, 5037, 9678, 4590],
                            [4590, 21228, 5037, 25545, 9678, 26316, 10125, 27588,
                             12762, 29400, 13209, 45897, 13980, 46668, 15252, 47940],
                            [60945, 60498, 44307, 51555, 52773, 52326, 36135, 50283,
                             55857, 55410, 39219, 18867, 39990, 19638, 37947, 17595],
                        ]
                    ),
                    columns=np.asarray(
                        [17595, 18867, 19638, 36135, 37947, 39219, 39990, 44307, 50283,
                         51555, 52326, 52773, 55410, 55857, 60498, 60945],
                        dtype=object,
                    ),
                    index=[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)],
                ),
                # fmt: on
            ),
        ],
        indirect=["translation"],
    )
    def test_df(self, translation, expected):
        pd.options.display.max_columns = None
        pd.testing.assert_frame_equal(translation._df, expected)

    def test_representative(self, lattice, lattice_2x2_basis):
        translation = Translation(lattice, lattice_2x2_basis)
        pd.testing.assert_series_equal(
            translation.representatives,
            pd.Series(
                np.array([27, 27, 105, 105, 27, 27]),
                index=np.asarray([27, 78, 105, 150, 177, 228], dtype=object),
            ),
        )

    def test_periodicity(self, lattice, lattice_2x2_basis):
        translation = Translation(lattice, lattice_2x2_basis)
        pd.testing.assert_series_equal(
            translation.periodicity,
            pd.Series(
                np.array([4, 4, 2, 2, 4, 4]),
                index=np.asarray([27, 78, 105, 150, 177, 228], dtype=object),
            ),
        )

    def test_shift(self, lattice, lattice_2x2_basis):
        translation = Translation(lattice, lattice_2x2_basis)
        pd.testing.assert_series_equal(
            translation.shift(pd.Series({27: 177, 105: 150})),
            pd.Series({27: (0, 1), 105: (0, 1)}),
        )
        assert translation.shift(pd.Series({27: 105, 105: 27})).empty

    def test_phase_factor(self, lattice, lattice_2x2_basis):
        translation = Translation(lattice, lattice_2x2_basis)
        ph = translation.phase_factor(0, 0, pd.Series({27: (1, 0), 105: (1, 0)}))
        np.testing.assert_array_equal(ph, np.array([[1, 1], [1, 1]]))
        assert ph.dtype == np.float64

    def test_normalization_factor(self, lattice, lattice_2x2_basis):
        translation = Translation(lattice, lattice_2x2_basis)
        np.testing.assert_allclose(
            translation.normalization_factor(), np.array([[1, np.sqrt(2)], [1 / np.sqrt(2), 1]])
        )

    def test_matrix_element(self, lattice, lattice_2x2_basis):
        translation = Translation(lattice, lattice_2x2_basis)
        operators = list(lattice.iter_plaquettes())
        print(translation[operators[3], (0, 0)])
        ...
