import pytest

from tests.helpers.caging_toys import (
    base_environment_reduction_config,
    pairwise_interference_system,
    two_zero_closed_interference_system,
)


@pytest.fixture
def environment_reduction_config():
    return base_environment_reduction_config()


@pytest.fixture
def pairwise_interference_case():
    return pairwise_interference_system()


@pytest.fixture
def two_zero_closed_interference_case():
    return two_zero_closed_interference_system()
