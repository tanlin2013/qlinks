import numpy as np

from qlinks.io import EigenpairHDF5Reader, EigenpairHDF5Writer


def test_hdf5_write_and_partial_read_eigenvectors(tmp_path) -> None:
    path = tmp_path / "ed_result.h5"

    energies = np.array([-2.0, -1.0, 0.5, 1.0])
    vectors = np.arange(4 * 8, dtype=np.float64).reshape(4, 8)

    with EigenpairHDF5Writer(path) as writer:
        writer.write_metadata(
            model_name="test_model",
            parameters={"length": 4, "boundary_condition": "open"},
        )
        writer.write_energies(energies)
        writer.write_eigenvectors(vectors)
        writer.write_observable("local_z", np.arange(4))

    with EigenpairHDF5Reader(path) as reader:
        assert reader.model_name == "test_model"
        assert reader.n_eigenvectors == 4
        assert reader.n_basis == 8

        np.testing.assert_array_equal(reader.read_energy(2), energies[2])
        np.testing.assert_array_equal(reader.read_eigenvector(2), vectors[2])

        np.testing.assert_array_equal(
            reader.read_eigenvectors(slice(1, 3)),
            vectors[1:3],
        )

        np.testing.assert_array_equal(
            reader.read_energies([0, 3]),
            energies[[0, 3]],
        )

        np.testing.assert_array_equal(
            reader.read_observable("local_z", [1, 3]),
            np.array([1, 3]),
        )


def test_hdf5_incremental_eigenvector_write(tmp_path) -> None:
    path = tmp_path / "incremental.h5"

    with EigenpairHDF5Writer(path) as writer:
        writer.write_energies(np.array([0.0, 1.0]))
        writer.create_eigenvector_dataset(
            n_eigenvectors=2,
            n_basis=5,
            dtype=np.complex128,
        )
        writer.write_eigenvector(0, np.ones(5, dtype=np.complex128))
        writer.write_eigenvector(1, 2.0 * np.ones(5, dtype=np.complex128))

    with EigenpairHDF5Reader(path) as reader:
        np.testing.assert_array_equal(
            reader.read_eigenvector(0),
            np.ones(5, dtype=np.complex128),
        )
        np.testing.assert_array_equal(
            reader.read_eigenvector(1),
            2.0 * np.ones(5, dtype=np.complex128),
        )


def test_hdf5_basis_states_partial_read(tmp_path) -> None:
    path = tmp_path / "basis.h5"

    states = np.array(
        [
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, -1],
        ],
        dtype=np.int64,
    )

    with EigenpairHDF5Writer(path) as writer:
        writer.write_basis_states(
            states,
            attrs={"layout": "spin_one_chain"},
        )
        writer.write_energies(np.array([0.0]))
        writer.write_eigenvectors(np.ones((1, states.shape[0])))

    with EigenpairHDF5Reader(path) as reader:
        np.testing.assert_array_equal(
            reader.read_basis_states(slice(1, 3)),
            states[1:3],
        )
