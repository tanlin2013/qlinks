import numpy as np

from qlinks.basis.configs import basis_configs_from_build_result
from qlinks.caging import (
    ManyBodyCLSCompletenessSequencePoint,
    ManyBodyCLSCompletenessSequenceReport,
    diagnose_many_body_cls_completeness,
    diagnose_many_body_topological_localization,
)
from qlinks.caging.spectral import basis_permutation_from_variable_permutation
from qlinks.models import SpinOneXYChainModel, spin_one_xy_scar_tower_states


def _swap_permutation() -> np.ndarray:
    # U|0> = |1>, U|1> = |0>, with two fixed states.
    return np.asarray([1, 0, 2, 3], dtype=np.int64)


def test_many_body_cls_completeness_finds_translation_quotient() -> None:
    target = np.eye(4, dtype=np.complex128)[:, :3]
    local_seed = np.eye(4, dtype=np.complex128)[:, 0]

    report = diagnose_many_body_cls_completeness(
        target,
        local_seed,
        translation_permutations=(_swap_permutation(),),
        translation_orders=(2,),
        tolerance=1.0e-12,
    )

    assert report.target_dimension == 3
    assert report.generator_seed_count == 1
    assert report.translated_generator_span_dimension == 2
    assert report.intersection_dimension == 2
    assert report.quotient_dimension == 1
    assert report.orbit_entries[0].orbit_dimension == 2
    np.testing.assert_allclose(report.quotient_projector, np.diag([0.0, 0.0, 1.0, 0.0]))


def test_many_body_topological_localization_resolves_momentum() -> None:
    target = np.eye(4, dtype=np.complex128)[:, :3]
    local_seed = np.eye(4, dtype=np.complex128)[:, 0]

    report = diagnose_many_body_topological_localization(
        target,
        local_seed,
        translation_permutations=(_swap_permutation(),),
        translation_orders=(2,),
        tolerance=1.0e-12,
    )

    assert report.completeness.quotient_dimension == 1
    assert report.quotient_sector_signature == (((0,), 1),)
    assert report.quotient_characters is not None
    np.testing.assert_allclose(report.quotient_characters, (1.0 + 0.0j,))
    assert report.has_symmetry_resolved_quotient


def test_many_body_cls_sequence_distinguishes_persistent_and_finite_size_defects() -> None:
    persistent = ManyBodyCLSCompletenessSequenceReport(
        model_label="persistent",
        points=tuple(
            ManyBodyCLSCompletenessSequencePoint(
                size_label=str(size),
                linear_sizes=(size,),
                target_dimension=1,
                local_generator_span_dimension=0,
                quotient_dimension=1,
            )
            for size in (4, 6, 8)
        ),
    )
    finite_size = ManyBodyCLSCompletenessSequenceReport(
        model_label="finite",
        points=(
            ManyBodyCLSCompletenessSequencePoint(
                size_label="4x4",
                linear_sizes=(4, 4),
                target_dimension=9,
                local_generator_span_dimension=8,
                quotient_dimension=1,
            ),
            ManyBodyCLSCompletenessSequencePoint(
                size_label="8x4",
                linear_sizes=(8, 4),
                target_dimension=4,
                local_generator_span_dimension=4,
                quotient_dimension=0,
            ),
        ),
    )

    assert persistent.classification == "persistent_quotient_candidate"
    assert persistent.has_persistent_defect
    assert finite_size.classification == "finite_size_only_quotient"
    assert not finite_size.has_persistent_defect


def test_spin_one_xy_tower_is_a_persistent_quotient_candidate_at_small_size() -> None:
    length = 4
    model = SpinOneXYChainModel(
        length=length,
        boundary_condition="periodic",
        j_xy=1.0,
        total_sz=0,
    )
    build = model.build(
        builder="optimized",
        basis_solver="dfs",
        sort_basis=True,
        on_missing="raise",
    )
    configs = basis_configs_from_build_result(build)
    states, _labels = spin_one_xy_scar_tower_states(
        basis_configs=configs,
        length=length,
    )
    two_site_translation = basis_permutation_from_variable_permutation(
        configs,
        np.roll(np.arange(length, dtype=np.int64), 2),
    )

    report = diagnose_many_body_topological_localization(
        states[:, 0],
        None,
        translation_permutations=(two_site_translation,),
        translation_orders=(length // 2,),
        tolerance=1.0e-10,
    )

    assert report.completeness.target_dimension == 1
    assert report.completeness.translated_generator_span_dimension == 0
    assert report.completeness.quotient_dimension == 1
    assert report.quotient_sector_signature == (((0,), 1),)
