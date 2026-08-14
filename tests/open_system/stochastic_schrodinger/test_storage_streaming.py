import numpy as np
import pytest

from qlinks.open_system.stochastic_schrodinger import McwfOptions, sample_lindblad_mcwf


def test_sample_lindblad_mcwf_populates_timing_collector(qubit_ops):
    timing: dict[str, float] = {}
    hamiltonian = 0.1 * qubit_ops["sigma_x"]
    jump = np.sqrt(0.2) * qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.1, 4)

    sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=4,
            seed=123,
            store_trajectories=False,
            timing_collector=timing,
        ),
    )

    assert timing["mcwf.operator_preparation"] >= 0.0
    assert timing["mcwf.initial_state_matrix"] >= 0.0
    assert timing["mcwf.rate_evaluation"] >= 0.0
    assert timing["mcwf.no_jump_propagation"] >= 0.0
    assert timing["mcwf.normalization"] >= 0.0
    assert timing["mcwf.density_accumulation"] >= 0.0


def test_sample_lindblad_mcwf_accumulates_existing_timing_collector_values(qubit_ops):
    timing: dict[str, float] = {"mcwf.rate_evaluation": 10.0}
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.sqrt(0.2) * qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.1, 3)

    sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=2,
            seed=123,
            store_trajectories=False,
            timing_collector=timing,
        ),
    )

    assert timing["mcwf.rate_evaluation"] >= 10.0


def test_sample_lindblad_mcwf_can_skip_density_matrices(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.sqrt(0.2) * qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.1, 4)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=4,
            seed=123,
            store_trajectories=False,
            store_density_matrices=False,
        ),
    )

    assert result.rho_t == []
    assert result.state_snapshots is None


def test_density_matrix_from_state_matrix_validates_shape():
    from qlinks.open_system.stochastic_schrodinger import (
        density_matrix_from_state_matrix,
    )

    with pytest.raises(ValueError, match="2D array"):
        density_matrix_from_state_matrix(np.ones(2, dtype=np.complex128))

    with pytest.raises(ValueError, match="at least one trajectory"):
        density_matrix_from_state_matrix(np.zeros((2, 0), dtype=np.complex128))


def test_sample_lindblad_mcwf_state_snapshots_reconstruct_density(qubit_ops):
    from qlinks.open_system.stochastic_schrodinger import (
        density_matrix_from_state_matrix,
    )

    hamiltonian = 0.1 * qubit_ops["sigma_x"]
    jump = np.sqrt(0.2) * qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.1, 4)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=4,
            seed=123,
            store_trajectories=False,
            store_density_matrices=True,
            store_state_snapshots=True,
        ),
    )

    assert result.state_snapshots is not None
    assert len(result.state_snapshots) == len(times)
    for snapshot, density_matrix in zip(result.state_snapshots, result.rho_t, strict=True):
        assert snapshot.shape == (2, 4)
        np.testing.assert_allclose(
            density_matrix_from_state_matrix(snapshot),
            density_matrix,
            atol=1e-14,
        )


def test_sample_lindblad_mcwf_state_snapshots_without_density(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.sqrt(0.2) * qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.1, 3)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=3,
            seed=123,
            store_trajectories=False,
            store_density_matrices=False,
            store_state_snapshots=True,
        ),
    )

    assert result.rho_t == []
    assert result.state_snapshots is not None
    assert len(result.state_snapshots) == len(times)
    assert all(snapshot.shape == (2, 3) for snapshot in result.state_snapshots)


def test_sample_lindblad_mcwf_state_snapshots_nonvectorized(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.sqrt(0.2) * qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.1, 3)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=2,
            seed=123,
            store_trajectories=True,
            store_states=False,
            store_density_matrices=False,
            store_state_snapshots=True,
        ),
    )

    assert result.rho_t == []
    assert result.trajectories is not None
    assert result.state_snapshots is not None
    assert len(result.state_snapshots) == len(times)
    assert all(snapshot.shape == (2, 2) for snapshot in result.state_snapshots)


def test_sample_lindblad_mcwf_chunked_density_matches_unchunked(qubit_ops):
    hamiltonian = 0.05 * qubit_ops["sigma_x"]
    times = np.linspace(0.0, 0.1, 6)

    baseline = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=12,
            seed=123,
            store_trajectories=False,
            store_density_matrices=True,
            trajectory_chunk_size=None,
        ),
    )
    chunked = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=12,
            seed=123,
            store_trajectories=False,
            store_density_matrices=True,
            trajectory_chunk_size=5,
        ),
    )

    assert len(chunked.rho_t) == len(baseline.rho_t)
    for actual, expected in zip(chunked.rho_t, baseline.rho_t, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_sample_lindblad_mcwf_chunked_state_snapshots_have_all_trajectories(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.sqrt(0.2) * qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.1, 4)
    timing = {}

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=10,
            seed=123,
            store_trajectories=False,
            store_density_matrices=False,
            store_state_snapshots=True,
            trajectory_chunk_size=4,
            timing_collector=timing,
        ),
    )

    assert result.rho_t == []
    assert result.state_snapshots is not None
    assert len(result.state_snapshots) == len(times)
    assert all(snapshot.shape == (2, 10) for snapshot in result.state_snapshots)
    assert timing["mcwf.chunk_merge"] >= 0.0


def test_mcwf_options_rejects_nonpositive_trajectory_chunk_size():
    with pytest.raises(ValueError, match="trajectory_chunk_size"):
        McwfOptions(trajectory_chunk_size=0).validate()


def test_sample_lindblad_mcwf_parallel_chunked_density_matches_unchunked(qubit_ops):
    hamiltonian = 0.05 * qubit_ops["sigma_x"]
    times = np.linspace(0.0, 0.1, 5)

    baseline = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=8,
            seed=123,
            store_trajectories=False,
            store_density_matrices=True,
            trajectory_chunk_size=None,
        ),
    )
    chunked = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=8,
            seed=123,
            store_trajectories=False,
            store_density_matrices=True,
            trajectory_chunk_size=4,
            trajectory_chunk_workers=2,
        ),
    )

    assert len(chunked.rho_t) == len(baseline.rho_t)
    for actual, expected in zip(chunked.rho_t, baseline.rho_t, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_mcwf_options_rejects_nonpositive_trajectory_chunk_workers():
    with pytest.raises(ValueError, match="trajectory_chunk_workers"):
        McwfOptions(trajectory_chunk_workers=0).validate()


def test_sample_lindblad_mcwf_streams_target_fidelity_without_snapshots(qubit_ops):
    from qlinks.open_system.stochastic_schrodinger import (
        McwfOptions,
        sample_lindblad_mcwf,
    )

    times = np.linspace(0.0, 0.2, 3)
    options = McwfOptions(
        n_trajectories=4,
        seed=123,
        store_density_matrices=False,
        store_state_snapshots=False,
        fidelity_targets={"target": qubit_ops["ket1"]},
    )

    result = sample_lindblad_mcwf(
        hamiltonian=np.zeros((2, 2), dtype=np.complex128),
        jumps=[],
        state_initial=qubit_ops["ket1"],
        times=times,
        options=options,
    )

    assert result.rho_t == []
    assert result.state_snapshots is None
    assert result.target_fidelities is not None
    np.testing.assert_allclose(result.target_fidelities["target"], np.ones(times.size))


def test_sample_lindblad_mcwf_chunked_streamed_target_fidelity(qubit_ops):
    from qlinks.open_system.stochastic_schrodinger import (
        McwfOptions,
        sample_lindblad_mcwf,
    )

    times = np.linspace(0.0, 0.2, 3)
    options = McwfOptions(
        n_trajectories=6,
        seed=123,
        store_density_matrices=False,
        store_state_snapshots=False,
        trajectory_chunk_size=2,
        fidelity_targets={"target": qubit_ops["ket1"]},
    )

    result = sample_lindblad_mcwf(
        hamiltonian=np.zeros((2, 2), dtype=np.complex128),
        jumps=[],
        state_initial=qubit_ops["ket1"],
        times=times,
        options=options,
    )

    assert result.rho_t == []
    assert result.state_snapshots is None
    assert result.target_fidelities is not None
    np.testing.assert_allclose(result.target_fidelities["target"], np.ones(times.size))


def test_sample_lindblad_mcwf_adaptive_trajectory_blocks_streamed_target_fidelity(qubit_ops):
    from qlinks.open_system.stochastic_schrodinger import (
        McwfOptions,
        sample_lindblad_mcwf,
    )

    times = np.linspace(0.0, 0.2, 3)
    timing_collector: dict[str, float] = {}
    options = McwfOptions(
        n_trajectories=6,
        seed=123,
        store_density_matrices=False,
        store_state_snapshots=False,
        adaptive_time_step=True,
        adaptive_trajectory_block_size=2,
        fidelity_targets={"target": qubit_ops["ket1"]},
        timing_collector=timing_collector,
    )

    result = sample_lindblad_mcwf(
        hamiltonian=np.zeros((2, 2), dtype=np.complex128),
        jumps=[],
        state_initial=qubit_ops["ket1"],
        times=times,
        options=options,
    )

    assert result.rho_t == []
    assert result.state_snapshots is None
    assert result.target_fidelities is not None
    np.testing.assert_allclose(result.target_fidelities["target"], np.ones(times.size))
    assert timing_collector["mcwf.count.adaptive_trajectory_blocks"] == 3.0


def test_mcwf_options_reject_nonpositive_adaptive_trajectory_block_size():
    from qlinks.open_system.stochastic_schrodinger import McwfOptions

    with pytest.raises(ValueError, match="adaptive_trajectory_block_size"):
        McwfOptions(adaptive_trajectory_block_size=0).validate()
