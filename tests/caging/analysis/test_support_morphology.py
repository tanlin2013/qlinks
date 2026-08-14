import numpy as np
import pytest

from qlinks.caging.analysis.environment import diagnose_environment_reduction
from qlinks.caging.analysis.support_morphology import analyze_support_morphology


def test_support_morphology_is_separate_from_environment_reduction(
    environment_reduction_config, pairwise_interference_case
):
    basis_configs, kinetic, indices = pairwise_interference_case

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(2.0)
    state[indices["v2"]] = -1.0 / np.sqrt(2.0)

    potential_diagonal = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    potential_diagonal[indices["v1"]] = 5.0
    potential_diagonal[indices["v2"]] = 5.0

    environment_report = diagnose_environment_reduction(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=environment_reduction_config,
    )
    morphology = analyze_support_morphology(
        full_state=state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        environment_report=environment_report,
        potential_diagonal=potential_diagonal,
    )

    assert environment_report.is_safely_removable
    assert environment_report.removal_summary.n_no_environment_weight_probes == 1

    fock = morphology.fock
    assert fock.label == "finite_size_shell_dense"
    assert fock.support_size == 2
    assert fock.effective_support_size == pytest.approx(2.0)
    assert fock.potential_shell_size == 2
    assert fock.support_shell_fraction == pytest.approx(1.0)
    assert fock.effective_shell_fraction == pytest.approx(1.0)
    assert fock.boundary_size == 1
    assert fock.support_internal_matrix_entries == 0

    real_space = morphology.real_space
    assert real_space.label == "partially_active"
    assert real_space.active_variable_indices == (1, 2)
    assert real_space.active_variable_count == 2
    assert real_space.frozen_variable_count == 1
    assert real_space.reduced_iz_region_variable_indices == (1, 2)
