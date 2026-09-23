"""The variational family, its torch objective, and the fit."""

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.special import logsumexp  # noqa: E402

from pmedm_vb.solvers.base import dual_gradient, dual_objective  # noqa: E402
from pmedm_vb.solvers.map_dual import solve_map  # noqa: E402
from pmedm_vb.solvers.vb import (  # noqa: E402
    StructuredGaussian,
    _DualTarget,
    posterior_weights,
    solve_vb,
)
from synthetic import dense_hessian, make_problem  # noqa: E402

TAPERS = ["tract", None]


@pytest.mark.parametrize("taper", TAPERS)
def test_laplace_is_a_member_of_the_family(problem, taper):
    result = solve_map(problem, alpha=0.3, taper=taper)
    family = StructuredGaussian.laplace(problem, result)
    precision = problem.n * dense_hessian(problem, result.W / problem.N, problem.sigma(0.3, taper))
    dense = np.column_stack([family.precision_matvec(e) for e in np.eye(problem.n_constraints)])
    np.testing.assert_allclose(dense, precision, rtol=1e-10, atol=1e-10 * np.abs(precision).max())
    assert family.logdet_precision() == pytest.approx(np.linalg.slogdet(precision)[1], abs=1e-9)
    draws = family.sample(np.random.default_rng(0), 100_000)
    covariance = np.linalg.inv(precision)
    assert np.abs(np.cov(draws) - covariance).max() / np.abs(covariance).max() < 0.02


@pytest.mark.parametrize("taper", TAPERS)
def test_torch_objective_and_gradient_match_numpy(problem, taper):
    sigma = problem.sigma(0.3, taper)
    target = _DualTarget(problem, sigma, "cpu")
    lam = np.random.default_rng(1).normal(0, 2, (3, problem.n_constraints))
    batch = torch.tensor(lam, requires_grad=True)
    values = target(batch)
    values.sum().backward()
    for i in range(3):
        assert values[i].item() == pytest.approx(dual_objective(problem, lam[i], sigma), abs=1e-12)
        np.testing.assert_allclose(batch.grad[i].numpy(), dual_gradient(problem, lam[i], sigma),
                                   rtol=1e-10, atol=1e-12)


def exact_mean(inputs, sigma, proposal, draws=50_000):
    """Posterior mean of exp(-n f) by self-normalised importance sampling."""
    rng = np.random.default_rng(5)
    lam = proposal.mean[:, None] + 1.5 * (proposal.sample(rng, draws) - proposal.mean[:, None])
    d = lam - proposal.mean[:, None]
    log_proposal = -0.5 * np.einsum("id,id->d", d, proposal.precision_matvec(d)) / 1.5**2
    with torch.no_grad():
        log_target = -inputs.n * _DualTarget(inputs, sigma, "cpu")(torch.tensor(lam.T)).numpy()
    weights = np.exp(log_target - log_proposal - logsumexp(log_target - log_proposal))
    return lam @ weights, np.sqrt(((lam - (lam @ weights)[:, None]) ** 2) @ weights)


@pytest.mark.parametrize("n, taper", [(150, "tract"), (8, "tract"), (8, None)])
def test_vb_improves_on_its_laplace_start(n, taper):
    inputs = dataclasses.replace(make_problem(), n=n)
    start = solve_map(inputs, alpha=0.3, taper=taper)
    fit = solve_vb(inputs, alpha=0.3, taper=taper, init=start, max_iter=3000, seed=1)
    se = np.hypot(fit.elbo_se, fit.laplace_elbo_se)
    assert fit.converged
    assert fit.elbo > fit.laplace_elbo - 3 * se
    if n == 8:  # a skewed posterior, where the Gaussian at the mode is poor
        assert fit.elbo > fit.laplace_elbo + 3 * se
    mean, sd = exact_mean(inputs, inputs.sigma(0.3, taper), StructuredGaussian.laplace(inputs, start))
    rms = lambda x: np.sqrt(np.mean(((x - mean) / sd) ** 2))  # noqa: E731
    assert rms(fit.q.mean) < rms(start.lam)


def test_init_must_match_the_requested_sigma(problem):
    start = solve_map(problem, alpha=0.3, taper="tract")
    with pytest.raises(ValueError, match="alpha"):
        solve_vb(problem, alpha=0.5, taper="tract", init=start, max_iter=1)


def test_posterior_weights_are_population_weights(problem):
    fit = solve_vb(problem, alpha=0.3, taper="tract", max_iter=100, final_draws=16)
    draws = posterior_weights(problem, fit, n_draws=4, rng=np.random.default_rng(0))
    assert draws.shape == (4, problem.n_zones, problem.n_units)
    np.testing.assert_allclose(draws.sum(axis=(1, 2)), problem.N)
    assert not np.allclose(draws[0], draws[1])


def test_vb_start_must_share_the_floor(problem):
    from test_inputs import with_zero_cells

    inputs, _, _ = with_zero_cells(problem)
    start = solve_map(inputs, alpha=0.3, variance_floor="zero")
    with pytest.raises(ValueError, match="variance_floor"):
        solve_vb(inputs, alpha=0.3, init=start, max_iter=1)
    fit = solve_vb(inputs, alpha=0.3, variance_floor="zero", init=start, max_iter=100, final_draws=16)
    assert fit.variance_floor == "zero"
