import numpy as np


def test_laurent_polynomial_constraint_module_resolves_free_and_torsion_parts() -> None:
    from qlinks.caging.stability import diagnose_laurent_polynomial_constraint_module

    free_report = diagnose_laurent_polynomial_constraint_module(
        ((0, np.zeros((1, 1))),),
        (1, 2, 3, 4),
    )
    assert free_report.free_kernel_rank == 1
    assert free_report.module_label == "free"
    assert [point.total_kernel_dimension for point in free_report.periodic_points] == [1, 2, 3, 4]

    derivative_report = diagnose_laurent_polynomial_constraint_module(
        ((0, np.ones((1, 1))), (1, -np.ones((1, 1)))),
        (1, 2, 3, 4),
    )
    assert derivative_report.free_kernel_rank == 0
    assert derivative_report.module_label == "root_of_unity_torsion"
    torsion_orders = [
        (entry.primitive_order, entry.multiplicity) for entry in derivative_report.torsion_orders
    ]
    assert torsion_orders == [(1, 1)]
    periodic_dimensions = [
        point.total_kernel_dimension for point in derivative_report.periodic_points
    ]
    assert periodic_dimensions == [1, 1, 1, 1]


def test_laurent_polynomial_constraint_module_detects_even_period_torsion() -> None:
    from qlinks.caging.stability import diagnose_laurent_polynomial_constraint_module

    report = diagnose_laurent_polynomial_constraint_module(
        ((0, np.ones((1, 1))), (1, np.ones((1, 1)))),
        (1, 2, 3, 4, 6),
    )
    assert report.free_kernel_rank == 0
    assert [(entry.primitive_order, entry.multiplicity) for entry in report.torsion_orders] == [
        (2, 1)
    ]
    dimensions = {
        point.repeat_count: point.total_kernel_dimension for point in report.periodic_points
    }
    assert dimensions == {1: 0, 2: 1, 3: 0, 4: 1, 6: 1}


def test_laurent_periodic_dimension_consistency_detects_divisibility_obstruction() -> None:
    from qlinks.caging.stability import diagnose_laurent_periodic_dimension_consistency

    report = diagnose_laurent_periodic_dimension_consistency(
        (1, 2, 3),
        (1, 0, 0),
    )
    assert not report.passes_necessary_conditions
    assert report.obstruction_label == "divisibility_violation"
    assert {
        (entry.divisor_repeat_count, entry.multiple_repeat_count)
        for entry in report.divisibility_violations
    } == {(1, 2), (1, 3)}
    assert report.primitive_order_multiplicities == ((1, 1), (2, -1), (3, -1))


def test_laurent_periodic_dimension_consistency_accepts_order_two_torsion() -> None:
    from qlinks.caging.stability import diagnose_laurent_periodic_dimension_consistency

    report = diagnose_laurent_periodic_dimension_consistency(
        (1, 2, 4),
        (0, 1, 1),
    )
    assert report.passes_necessary_conditions
    assert report.primitive_order_multiplicities == ((1, 0), (2, 1), (4, 0))
