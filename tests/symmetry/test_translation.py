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
    def lattice(self):
        return SquareLattice(2, 2)

    def test_df(self, lattice, lattice_2x2_basis):
        translation = Translation(lattice, lattice_2x2_basis)
        pd.testing.assert_frame_equal(
            translation._df,
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
        )

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
