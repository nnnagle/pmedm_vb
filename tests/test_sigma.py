"""Sigma against its dense form, tapered and untapered."""

import numpy as np
import pytest

from pmedm_vb.assemble.sigma import Sigma
from pmedm_vb.data.variance import SDR_FACTOR

ALPHAS = [1.0, 0.5, 0.05]
TAPERS = ["tract", None]


def expected_dense(problem, alpha, taper):
    """D + mask * (1-alpha)(4/80) L L', written out from the definition."""
    low_rank = SDR_FACTOR * (1 - alpha) * problem.sigma_l @ problem.sigma_l.T
    groups = problem.constraint_tracts()
    mask = groups[:, None] == groups[None, :] if taper else np.ones_like(low_rank, bool)
    return np.diag(problem.sigma_v - np.diag(low_rank)) + low_rank * mask


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("taper", TAPERS)
def test_operations_match_dense(problem, alpha, taper):
    sigma = problem.sigma(alpha, taper)
    dense = expected_dense(problem, alpha, taper)
    x = np.random.default_rng(0).normal(size=(problem.n_constraints, 3))
    np.testing.assert_allclose(sigma.to_dense(), dense, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(sigma.matvec(x), dense @ x, rtol=1e-10)
    np.testing.assert_allclose(sigma.matvec(x[:, 0]), dense @ x[:, 0], rtol=1e-10)
    np.testing.assert_allclose(sigma.solve(x), np.linalg.solve(dense, x), rtol=1e-8)
    assert sigma.logdet() == pytest.approx(np.linalg.slogdet(dense)[1], rel=1e-12)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("taper", TAPERS)
def test_diagonal_is_the_published_variance(problem, alpha, taper):
    np.testing.assert_allclose(problem.sigma(alpha, taper).diagonal(), problem.sigma_v, rtol=1e-12)


def test_alpha_one_ignores_the_taper(problem):
    np.testing.assert_array_equal(
        problem.sigma(1.0, "tract").to_dense(), problem.sigma(1.0, None).to_dense()
    )


@pytest.mark.parametrize("taper", TAPERS)
def test_block_dense_is_the_part_a_solver_adds(problem, taper):
    sigma = problem.sigma(0.3, taper)
    rows = np.flatnonzero(problem.constraint_tracts() == 1)
    expected = sigma.to_dense()[np.ix_(rows, rows)] if taper else np.diag(sigma.d[rows])
    np.testing.assert_allclose(sigma.block_dense(rows), expected, rtol=1e-12)
    assert (sigma.global_factor is None) == (taper is not None)


def test_draws_have_the_right_covariance(problem):
    sigma = problem.sigma(0.3, "tract")
    draws = sigma.draw(np.random.default_rng(1), size=100_000)
    dense = sigma.to_dense()
    scale = np.sqrt(np.outer(np.diag(dense), np.diag(dense)))
    # Correlation error of 100,000 draws is about 1/sqrt(100000) = 0.003.
    assert np.abs(np.cov(draws) - dense).max() / scale.max() < 0.02


def test_rejects_alpha_outside_the_open_interval(problem):
    with pytest.raises(ValueError, match="alpha"):
        Sigma(v=problem.sigma_v, l=problem.sigma_l, alpha=0.0)
