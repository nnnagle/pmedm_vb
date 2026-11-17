"""The PUMA reference ``q^P(θ_P)`` from the PUMA's own records (step 1, piece
B1 of ``PLAN_latent.md``).

From ``MODEL_latent.qmd`` (notation kept):

* ``q^P_i(θ) = q_i γ^{d(P,i)} exp{θ' h(z_i) - A_P(θ)}``, with
  ``A_P(θ) = log Σ_i q_i γ^{d(P,i)} e^{θ' h(z_i)}``;
* ``ℓ_P(θ) = n_P [Σ_{i∈P} w̃_i θ' h(z_i) - A_P(θ)]``;
* prior ``θ ~ N(0, σ_θ² I)`` on the standardized basis.

:func:`fit_reference` maximizes ``ℓ_P(θ) - ‖θ‖² / 2σ_θ²`` by Newton's method
with the exact Hessian ``n_P Cov_{q^P}(h) + I/σ_θ²`` and a backtracking
(Armijo) line search, from ``θ = 0`` or a warm start. It stops when the
largest gradient component is below ``tol`` relative to the objective's scale
``n_P`` (the multiplier of ``ℓ_P``; at ``σ_θ = ∞`` this is the largest
moment mismatch in standardized units), or after ``max_iter`` iterations.
``σ_θ = ∞`` (no prior) is allowed; a singular Hessian is then solved in the
least-squares sense.

**Support.** The atoms entering ``A_P`` are given by ``rows`` (``None``: every
atom) and the log base weights ``log q_i + d(P,i) log γ``; an atom with base
weight zero (``-inf``) is off the support. ``γ = 0`` is the support mask
``d(P,i) = 0`` (:func:`log_base`), and for it only the PUMA's own rows are
passed, which keeps that fit cheap.

**Outputs** (:class:`ReferenceFit`): ``θ̂``, ``q^P`` over all ``M`` atoms
(zero off the support), iterations, the final largest gradient component
(relative), convergence, the Kish effective sample size ``1/Σ (q^P_i)²`` and
the share of ``q^P`` on atoms outside the PUMA.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.special import logsumexp


@dataclass
class ReferenceFit:
    theta: np.ndarray
    p: np.ndarray
    iterations: int
    grad_max: float
    converged: bool
    objective: float
    log_partition: float
    ess: float
    share_outside: float
    seconds: float
    message: str


def log_base(q: np.ndarray, distance: np.ndarray, gamma: float,
             exclude: np.ndarray | None = None) -> np.ndarray:
    """``log q_i + d(P,i) log γ`` for every atom, ``-inf`` off the support:
    for ``γ = 0`` every atom with ``d > 0``, and always the atoms in
    ``exclude`` (a boolean mask: the held-out fold in the cross-fit)."""
    with np.errstate(divide="ignore"):
        out = np.log(q).astype(float)
    if gamma == 0:
        out = np.where(distance == 0, out, -np.inf)
    elif gamma != 1:
        out = out + distance * np.log(gamma)
    if exclude is not None:
        out = np.where(exclude, -np.inf, out)
    return out


class _Problem:
    """The negative log posterior over the support rows, ``F(θ) =
    n [A(θ) - θ'μ] + ‖θ‖² / 2σ²`` and its derivatives."""

    def __init__(self, H, logb, mu, n, sigma):
        self.H, self.logb, self.mu, self.n = H, logb, mu, float(n)
        self.prec = 0.0 if not np.isfinite(sigma) else 1.0 / float(sigma) ** 2

    def value(self, theta):
        eta = self.logb + self.H @ theta
        A = float(logsumexp(eta))
        return self.n * (A - theta @ self.mu) + 0.5 * self.prec * (theta @ theta), A, eta

    def gradient(self, theta, A, eta):
        p = np.exp(eta - A)
        Eh = p @ self.H
        return self.n * (Eh - self.mu) + self.prec * theta, p, Eh

    def hessian(self, p, Eh):
        X = self.H * np.sqrt(p)[:, None]
        hess = self.n * (X.T @ X - np.outer(Eh, Eh))
        hess[np.diag_indices_from(hess)] += self.prec
        return hess


def _newton_step(hess: np.ndarray, g: np.ndarray) -> np.ndarray:
    try:
        return -cho_solve(cho_factor(hess, lower=True, check_finite=False), g)
    except LinAlgError:
        return -np.linalg.lstsq(hess, g, rcond=1e-12)[0]


def fit_reference(H: np.ndarray, logb: np.ndarray, mu: np.ndarray, n: float, sigma: float, *,
                  rows: np.ndarray | None = None, theta0: np.ndarray | None = None,
                  in_puma: np.ndarray | None = None, tol: float = 1e-8,
                  max_iter: int = 100) -> ReferenceFit:
    """Fit ``θ_P`` (see the module notes).

    ``H`` is the basis over all ``M`` atoms and ``logb`` the log base weights
    over all atoms (:func:`log_base`); ``rows`` restricts both to a subset
    (the support, or a superset of it). ``mu`` is the training records'
    weighted mean of ``h`` and ``n`` their number, ``n_P``. ``in_puma`` (a
    boolean mask over all atoms) gives the share of ``q^P`` outside the PUMA.
    """
    start = time.perf_counter()
    M = H.shape[0]
    Hs = H if rows is None else H[rows]
    lb = logb if rows is None else logb[rows]
    on = np.isfinite(lb)
    if not on.all():
        Hs, lb = Hs[on], lb[on]
        rows = np.flatnonzero(on) if rows is None else np.asarray(rows)[on]
    if Hs.shape[0] == 0:
        raise ValueError("empty support")
    problem = _Problem(Hs, lb, mu, n, sigma)
    theta = np.zeros(H.shape[1]) if theta0 is None else np.array(theta0, dtype=float)
    F, A, eta = problem.value(theta)
    scale = max(float(n), 1.0)
    message, converged, iterations = "max_iter", False, 0
    g, p, Eh = problem.gradient(theta, A, eta)
    for iterations in range(1, max_iter + 1):
        if np.abs(g).max() <= tol * scale:
            iterations -= 1
            converged, message = True, "gradient"
            break
        step = _newton_step(problem.hessian(p, Eh), g)
        slope = float(g @ step)
        if not slope < 0:  # not a descent direction (numerically singular Hessian)
            step, slope = -g, -float(g @ g)
        t = 1.0
        for _ in range(60):
            F_new, A_new, eta_new = problem.value(theta + t * step)
            if F_new <= F + 1e-4 * t * slope:
                break
            t *= 0.5
        else:
            message = "line search failed"
            break
        theta = theta + t * step
        F, A, eta = F_new, A_new, eta_new
        g, p, Eh = problem.gradient(theta, A, eta)
    else:
        if np.abs(g).max() <= tol * scale:
            converged, message = True, "gradient"
    p_full = np.zeros(M)
    p_full[slice(None) if rows is None else rows] = p
    outside = float(p_full[~in_puma].sum()) if in_puma is not None else float("nan")
    return ReferenceFit(theta=theta, p=p_full, iterations=iterations,
                        grad_max=float(np.abs(g).max() / scale), converged=converged,
                        objective=-F, log_partition=A, ess=float(1.0 / np.square(p).sum()),
                        share_outside=outside, seconds=time.perf_counter() - start,
                        message=message)


def objective(H, logb, mu, n, sigma, theta) -> float:
    """``ℓ_P(θ) - ‖θ‖² / 2σ²`` (support where ``logb`` is finite)."""
    on = np.isfinite(logb)
    return -_Problem(H[on], logb[on], mu, n, sigma).value(np.asarray(theta, float))[0]


def gradient(H, logb, mu, n, sigma, theta) -> np.ndarray:
    """The gradient of :func:`objective`."""
    on = np.isfinite(logb)
    problem = _Problem(H[on], logb[on], mu, n, sigma)
    theta = np.asarray(theta, float)
    _, A, eta = problem.value(theta)
    return -problem.gradient(theta, A, eta)[0]
