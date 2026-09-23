"""The MAP solver and its pieces, against dense linear algebra."""

import dataclasses

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp

from pmedm_vb.solvers.base import (
    ConstraintOperator,
    dual_gradient,
    dual_objective,
    dual_state,
    weights_from_lambda,
)
from pmedm_vb.solvers.map_dual import DualHessian, laplace_precision, solve_map
from synthetic import dense_hessian, kronecker, make_problem

CASES = [(1.0, "tract"), (0.5, "tract"), (0.05, "tract"), (0.5, None), (0.05, None)]


def test_operator_matches_the_kronecker_product(problem):
    op, xt = ConstraintOperator(problem), kronecker(problem)
    rng = np.random.default_rng(0)
    W = rng.random((problem.n_zones, problem.n_units))
    lam = rng.normal(size=problem.n_constraints)
    np.testing.assert_allclose(op.forward(W), xt @ W.ravel(order="F"), rtol=1e-12)
    np.testing.assert_allclose(op.adjoint(lam).ravel(order="F"), xt.T @ lam, rtol=1e-12, atol=1e-12)


def test_weights_are_a_distribution_even_for_huge_multipliers(problem):
    p = weights_from_lambda(problem, np.full(problem.n_constraints, 500.0))
    assert np.isfinite(p).all() and p.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("alpha, taper", CASES)
def test_gradient_matches_finite_differences(problem, alpha, taper):
    sigma = problem.sigma(alpha, taper)
    lam = np.random.default_rng(2).normal(0, 3, problem.n_constraints)
    h, eye = 1e-6, np.eye(problem.n_constraints)
    fd = [(dual_objective(problem, lam + h * e, sigma) - dual_objective(problem, lam - h * e, sigma)) / (2 * h)
          for e in eye]
    np.testing.assert_allclose(dual_gradient(problem, lam, sigma), fd, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("alpha, taper", CASES)
def test_hessian_operator_matches_dense(problem, alpha, taper):
    sigma = problem.sigma(alpha, taper)
    state = dual_state(problem, np.random.default_rng(3).normal(0, 3, problem.n_constraints), sigma)
    hessian, dense = DualHessian(problem, state, sigma), dense_hessian(problem, state.p, sigma)
    x = np.random.default_rng(4).normal(size=problem.n_constraints)
    np.testing.assert_allclose(hessian.matvec(x), dense @ x, rtol=1e-10)
    np.testing.assert_allclose(hessian.solve(x), np.linalg.solve(dense, x), rtol=1e-8)
    assert hessian.logdet() == pytest.approx(np.linalg.slogdet(dense)[1], abs=1e-9)


@pytest.mark.parametrize("alpha, taper", CASES)
def test_solver_finds_the_unique_minimiser(problem, alpha, taper):
    sigma = problem.sigma(alpha, taper)
    result = solve_map(problem, alpha=alpha, taper=taper)
    reference = minimize(
        lambda l: dual_objective(problem, l, sigma),
        np.zeros(problem.n_constraints),
        jac=lambda l: dual_gradient(problem, l, sigma),
        hess=lambda l: dense_hessian(problem, dual_state(problem, l, sigma).p, sigma),
        method="trust-exact",
        options={"gtol": 1e-12},
    )
    assert result.converged and result.newton_decrement <= 1e-10
    assert result.objective == pytest.approx(reference.fun, abs=1e-12)
    assert result.W.sum() == pytest.approx(problem.N)


def test_solver_recovers_from_a_distant_start():
    inputs = make_problem(zone_tract=(0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3), n_units=400,
                          tract_cells=12, block_group_cells=8, noise=0.5, seed=5)
    start = np.random.default_rng(6).normal(0, 20, inputs.n_constraints)
    result = solve_map(inputs, alpha=0.1, taper=None, init=start)
    assert result.converged
    assert result.trace["step"][:-1].min() < 1  # the line search had to work


def test_tapers_agree_at_alpha_one(problem):
    tapered, untapered = (solve_map(problem, alpha=1.0, taper=t) for t in ("tract", None))
    np.testing.assert_allclose(tapered.lam, untapered.lam, rtol=1e-10, atol=1e-12)
    assert tapered.log_evidence == pytest.approx(untapered.log_evidence, rel=1e-12)


@pytest.mark.parametrize("alpha, taper", CASES)
def test_fit_measure_identities_at_the_optimum(problem, alpha, taper):
    sigma = problem.sigma(alpha, taper)
    # The stationarity identity is exact only where the gradient vanishes; at the
    # default tol the leftover gradient (~1e-8), times N, moves it by ~1e-4.
    result = solve_map(problem, alpha=alpha, taper=taper, tol=1e-16)
    residual = ConstraintOperator(problem).forward(result.W) - problem.targets()
    m = problem.n_constraints
    dense = residual @ np.linalg.solve(sigma.to_dense(), residual) / m
    stationarity = (problem.n / problem.N) ** 2 * (result.lam @ sigma.matvec(result.lam)) / m
    assert result.trace["mahalanobis"][-1] == pytest.approx(dense, rel=1e-8)
    assert result.trace["mahalanobis"][-1] == pytest.approx(stationarity, rel=1e-9)
    if alpha == 1.0:
        assert dense == pytest.approx(np.mean(residual**2 / problem.sigma_v), rel=1e-10)


def test_laplace_precision_is_the_converged_hessian(problem):
    result = solve_map(problem, alpha=0.5, taper="tract")
    sigma = problem.sigma(0.5, "tract")
    dense = dense_hessian(problem, result.W / problem.N, sigma)
    x = np.random.default_rng(7).normal(size=problem.n_constraints)
    np.testing.assert_allclose(laplace_precision(problem, result).matvec(x), dense @ x, rtol=1e-8)


def model_problem(alpha_true, seed=11):
    """A problem whose Y is drawn from the model itself, n well above m."""
    base = make_problem(zone_tract=(0, 0, 1, 1), n_units=60, tract_cells=3,
                        block_group_cells=2, seed=seed)
    inputs = dataclasses.replace(base, n=200)
    rng = np.random.default_rng(seed)
    xt, qv = kronecker(inputs), inputs.q.ravel(order="F")
    u = xt @ qv
    sampling_var = inputs.N**2 * (xt**2 @ qv - u**2) / inputs.n
    scale = np.sqrt(sampling_var / inputs.sigma_v)[:, None]
    inputs = dataclasses.replace(inputs, sigma_v=inputs.sigma_v * scale[:, 0] ** 2,
                                 sigma_l=inputs.sigma_l * scale)
    y = (inputs.N / inputs.n) * xt @ rng.multinomial(inputs.n, qv)
    y = y + inputs.sigma(alpha_true, "tract").draw(rng)[:, 0]
    split = inputs.Y_T.size
    return dataclasses.replace(
        inputs,
        Y_T=y[:split].reshape(inputs.Y_T.shape, order="F"),
        Y_B=y[split:].reshape(inputs.Y_B.shape, order="F"),
    )


@pytest.mark.parametrize("alpha", [1.0, 0.1])
def test_log_evidence_matches_the_marginal_likelihood(alpha):
    """Saddlepoint against an importance-sampled p(Y), when m is well below n."""
    inputs = model_problem(alpha_true=0.3)
    result = solve_map(inputs, alpha=alpha, taper="tract")
    sigma, xt, n, N = inputs.sigma(alpha, "tract"), kronecker(inputs), inputs.n, inputs.N
    chol = np.linalg.cholesky(sigma.to_dense())
    p_star = result.W.ravel(order="F") / N
    log_ratio = np.log(inputs.q.ravel(order="F")) - np.log(p_star)
    rng, terms = np.random.default_rng(8), []
    for _ in range(5):
        w = rng.multinomial(n, p_star, size=20_000)
        r = np.linalg.solve(chol, (inputs.targets() - (N / n) * w @ xt.T).T)
        terms.append(-0.5 * (r**2).sum(0) - np.log(np.diag(chol)).sum()
                     - 0.5 * len(r) * np.log(2 * np.pi) + w @ log_ratio)
    terms = np.concatenate(terms)
    exact = logsumexp(terms) - np.log(terms.size)
    assert result.log_evidence == pytest.approx(exact, abs=0.1)
