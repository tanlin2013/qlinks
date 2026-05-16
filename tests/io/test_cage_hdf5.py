import numpy as np

from qlinks.caging import CageState
from qlinks.io import (
    CageStateHDF5Reader,
    CageStateHDF5Writer,
)


def test_cage_state_hdf5_roundtrip(tmp_path) -> None:
    path = tmp_path / "cages.h5"

    cage_states = [
        CageState(
            energy=4.0 + 0.0j,
            local_state=np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2),
            support=np.array([3, 7], dtype=np.int64),
            boundary_residual=1e-13,
            eigen_residual=2e-13,
            full_residual=3e-13,
            metadata={
                "kappa": 0,
                "potential_value": 4,
            },
        ),
        CageState(
            energy=10.0 + 0.0j,
            local_state=np.array([1.0, 0.0, -1.0], dtype=np.complex128) / np.sqrt(2),
            support=np.array([1, 5, 9], dtype=np.int64),
            boundary_residual=1e-12,
            eigen_residual=2e-12,
            full_residual=None,
            metadata={
                "kappa": 2,
                "potential_value": 8,
            },
        ),
    ]

    with CageStateHDF5Writer(path) as writer:
        writer.write_metadata(
            model_name="SquareQLMModel",
            parameters={
                "lx": 4,
                "ly": 4,
                "boundary_condition": "periodic",
            },
            extra={
                "note": "test roundtrip",
            },
        )
        writer.write_cage_states(
            cage_states,
            hilbert_size=10,
            attrs={
                "source": "unit test",
            },
        )

    with CageStateHDF5Reader(path) as reader:
        assert reader.model_name == "SquareQLMModel"
        assert reader.n_cage_states == 2

        metadata = reader.read_metadata()
        assert metadata["parameters"]["lx"] == 4
        assert metadata["parameters"]["ly"] == 4

        loaded_states = reader.read_cage_states()

    with CageStateHDF5Reader(path) as reader:
        assert reader.hilbert_size == 10

        full_vector = reader.read_full_vector(0)
        assert full_vector.shape == (10,)

        np.testing.assert_allclose(
            full_vector[cage_states[0].support],
            cage_states[0].local_state,
        )

        outside_mask = np.ones(10, dtype=bool)
        outside_mask[cage_states[0].support] = False
        np.testing.assert_allclose(full_vector[outside_mask], 0.0)

        full_matrix = reader.read_full_matrix()
        assert full_matrix.shape == (len(cage_states), 10)

    assert len(loaded_states) == len(cage_states)

    for expected_state, actual_state in zip(
        cage_states,
        loaded_states,
        strict=True,
    ):
        np.testing.assert_allclose(actual_state.energy, expected_state.energy)
        np.testing.assert_array_equal(actual_state.support, expected_state.support)
        np.testing.assert_allclose(
            actual_state.local_state,
            expected_state.local_state,
        )
        np.testing.assert_allclose(
            actual_state.boundary_residual,
            expected_state.boundary_residual,
        )
        np.testing.assert_allclose(
            actual_state.eigen_residual,
            expected_state.eigen_residual,
        )

        if expected_state.full_residual is None:
            assert actual_state.full_residual is None
        else:
            np.testing.assert_allclose(
                actual_state.full_residual,
                expected_state.full_residual,
            )


def test_cage_state_hdf5_indexed_reads(tmp_path) -> None:
    path = tmp_path / "cages.h5"

    cage_states = [
        CageState(
            energy=float(index),
            local_state=np.ones(index + 2, dtype=np.complex128),
            support=np.arange(index + 2, dtype=np.int64),
            boundary_residual=0.0,
            eigen_residual=0.0,
            full_residual=None,
            metadata={
                "kappa": index,
                "potential_value": index + 10,
            },
        )
        for index in range(3)
    ]

    with CageStateHDF5Writer(path) as writer:
        writer.write_cage_states(cage_states)

    with CageStateHDF5Reader(path) as reader:
        np.testing.assert_array_equal(
            reader.read_support(2),
            np.array([0, 1, 2, 3], dtype=np.int64),
        )
        np.testing.assert_allclose(
            reader.read_local_state(1),
            np.ones(3, dtype=np.complex128),
        )

        selected_states = reader.read_cage_states([0, 2])

    assert len(selected_states) == 2
    assert selected_states[0].support_size == 2
    assert selected_states[1].support_size == 4
