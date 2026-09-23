import pytest

import pmedm_vb
from synthetic import make_problem


@pytest.fixture(autouse=True)
def quiet():
    """Progress messages are on by default; tests do not need them."""
    pmedm_vb.set_verbosity("WARNING")
    yield
    pmedm_vb.set_verbosity("INFO")


@pytest.fixture(scope="session")
def problem():
    return make_problem()
