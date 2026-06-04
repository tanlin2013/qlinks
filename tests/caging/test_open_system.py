from qlinks.caging.open_system import (
    LocalTermDescriptor,
    _select_jump_terms,
    _select_monitor_terms,
)


def test_select_jump_terms_can_include_crossing_terms():
    inside = (
        LocalTermDescriptor(
            term_id=0,
            term_kind="plaquette",
            operator_kind="kinetic",
            support_links=(0, 1),
        ),
    )
    outside = ()
    crossing = (
        LocalTermDescriptor(
            term_id=1,
            term_kind="plaquette",
            operator_kind="kinetic",
            support_links=(1, 2),
        ),
    )

    selected = _select_jump_terms(
        inside_terms=inside,
        outside_terms=outside,
        crossing_terms=crossing,
        policy="outside_or_crossing",
    )

    assert tuple(term.term_id for term in selected) == (1,)


def test_select_monitor_terms_stays_strict_inside_by_default():
    inside = (
        LocalTermDescriptor(
            term_id=0,
            term_kind="plaquette",
            operator_kind="kinetic",
            support_links=(0, 1),
        ),
    )
    crossing = (
        LocalTermDescriptor(
            term_id=1,
            term_kind="plaquette",
            operator_kind="kinetic",
            support_links=(1, 2),
        ),
    )

    selected = _select_monitor_terms(
        inside_terms=inside,
        outside_terms=(),
        crossing_terms=crossing,
        policy="strict_inside",
    )

    assert tuple(term.term_id for term in selected) == (0,)
