import pytest

from tests.helpers.caging_toys import (
    base_classification_config,
    pairwise_interference_system,
    two_zero_closed_interference_system,
)


@pytest.fixture
def classification_config():
    return base_classification_config()


@pytest.fixture
def pairwise_interference_case():
    return pairwise_interference_system()


@pytest.fixture
def two_zero_closed_interference_case():
    return two_zero_closed_interference_system()
