from __future__ import annotations

import numpy as np


def assert_partial_check_matches_full_check_on_complete_configs(
    condition,
    configs: np.ndarray,
) -> None:
    states = np.asarray(configs, dtype=np.int64)

    if states.ndim == 1:
        states = states.reshape(1, -1)

    for config in states:
        assigned_mask = np.ones(config.shape, dtype=bool)
        assert condition.partial_check(config, assigned_mask) == condition.is_satisfied(config)


def first_allowed_target(sector_cls, **kwargs):
    allowed = sector_cls.allowed_targets(**kwargs)

    assert len(allowed) > 0

    # Prefer zero when it exists, otherwise use the first legal sector.
    if 0 in allowed:
        return 0

    return allowed[0]
