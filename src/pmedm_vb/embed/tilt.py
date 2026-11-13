"""Exponential tilts of a weighted support of units, fitted to table targets.

A tilt is one distribution over the support's units,

    p_i(β) = q_i exp(−F_i β) / Z(β),

with ``q_i`` the base (design) weights and ``F_i`` the unit's features. Two
fits are provided:

:func:`fit_tilt`
    β by maximum a posteriori under Gaussian measurement error on table cells
    and an i.i.d. Gaussian prior on β:

        f(β) = ½ (y − N X′p)′ Σ⁻¹ (y − N X′p) + ½ ‖β‖² / τ²

    ``X`` holds every unit's counts in the target cells, ``y`` the targets,
    ``Σ`` their (diagonal) sampling variances and ``N`` the total that turns
    the distribution into counts. The same form serves every feature set (the
    constraint cells themselves, positions φ, or both), so tilts on different
    features are compared on one footing. It is a MAP form, **not** the
    PMEDM's penalized maximum-entropy dual: here the prior is on β directly,
    with no entropy weight, and ``Σ`` is diagonal.

:func:`fit_oracle`
    β by maximum entropy with exact moment constraints, E_p[F] = μ: the
    tilt whose feature means equal given values. Its dual is convex,

        g(β) = log Z(β) + β′μ,   ∇g = μ − E_p[F],   ∇²g = Cov_p(F),

    and is solved by Newton's method. A 0/1 feature whose target mean is 0
    cannot be matched by a finite β; the units carrying it are removed from
    the support instead and the feature dropped.

Units outside the support carry ``log q = −inf`` and get ``p = 0``, so the
same feature and count matrices serve every leave-one-out fit without copies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.optimize
import scipy.sparse as sp


@dataclass
class Tilt:
    """A fitted tilt: β, the distribution over units and convergence."""

    beta: np.ndarray
    p: np.ndarray
    objective: float
    iterations: int
    converged: bool
    message: str


def _as_csr(F) -> sp.csr_matrix:
    return F.tocsr() if sp.issparse(F) else sp.csr_matrix(np.asarray(F, dtype=np.float64))


def distribution(log_q: np.ndarray, F, beta: np.ndarray) -> np.ndarray:
    """``p ∝ exp(log_q − F β)``, computed stably; ``−inf`` entries give 0."""
    a = log_q - (F @ beta)
    finite = np.isfinite(a)
    top = a[finite].max()
    e = np.zeros_like(a)
    e[finite] = np.exp(a[finite] - top)
    return e / e.sum()


def fit_tilt(
    log_q: np.ndarray,
    F,
    X: sp.csr_matrix,
    y: np.ndarray,
    se: np.ndarray,
    N: float,
    *,
    tau: float = 3.0,
    max_iter: int = 5000,
    tol: float = 1e-9,
) -> Tilt:
    """MAP tilt on features ``F`` fitted to targets ``y`` (see the module notes).

    ``F`` and ``X`` have one row per unit (support or not); ``log_q`` is
    ``−inf`` off the support.
    """
    F = _as_csr(F)
    Ft = F.T.tocsr()
    X = X.tocsr()
    inv_var = 1.0 / np.square(se)
    prior = 1.0 / tau**2

    def objective(beta):
        p = distribution(log_q, F, beta)
        r = y - N * (X.T @ p)
        g = r * inv_var
        u = X @ g
        v = p * (u - p @ u)
        value = 0.5 * float(r @ g) + 0.5 * prior * float(beta @ beta)
        grad = N * (Ft @ v) + prior * beta
        return value, grad

    beta0 = np.zeros(F.shape[1])
    result = scipy.optimize.minimize(
        objective, beta0, jac=True, method="L-BFGS-B",
        options={"maxiter": max_iter, "maxfun": 2 * max_iter, "ftol": tol, "gtol": 1e-7,
                 "maxcor": 20},
    )
    beta = result.x
    return Tilt(beta=beta, p=distribution(log_q, F, beta), objective=float(result.fun),
                iterations=int(result.nit), converged=bool(result.success),
                message=str(result.message))


def fit_oracle(
    log_q: np.ndarray,
    F: np.ndarray,
    mu: np.ndarray,
    *,
    binary: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[Tilt, np.ndarray]:
    """Maximum-entropy tilt with E_p[F] = ``mu`` (dense ``F``, few columns).

    ``binary`` marks 0/1 columns; one whose ``mu`` is 0 is dropped and its
    units removed from the support (see the module notes). Returns the tilt
    and the final moment error E_p[F] − μ over all columns.
    """
    F = np.asarray(F, dtype=np.float64)
    log_q = log_q.copy()
    keep = np.ones(F.shape[1], dtype=bool)
    if binary is not None:
        for j in np.flatnonzero(binary):
            if mu[j] <= 0:
                log_q[F[:, j] > 0] = -np.inf
                keep[j] = False
    Fk, muk = F[:, keep], mu[keep]
    support = np.isfinite(log_q)
    beta = np.zeros(Fk.shape[1])
    converged, message, step_count = False, "maximum iterations", 0
    for step_count in range(1, max_iter + 1):
        p = distribution(log_q, Fk, beta)
        mean = p @ Fk
        grad = muk - mean
        if np.max(np.abs(grad)) < tol:
            converged, message = True, "moments matched"
            break
        centred = (Fk[support] - mean) * np.sqrt(p[support])[:, None]
        hess = centred.T @ centred + 1e-12 * np.eye(len(beta))
        direction = -np.linalg.solve(hess, grad)

        def dual(b):
            a = log_q[support] - Fk[support] @ b
            top = a.max()
            return top + np.log(np.exp(a - top).sum()) + b @ muk

        current, t = dual(beta), 1.0
        while dual(beta + t * direction) > current + 1e-4 * t * (grad @ direction) and t > 1e-10:
            t *= 0.5
        beta = beta + t * direction
    full = np.zeros(F.shape[1])
    full[keep] = beta
    p = distribution(log_q, Fk, beta)
    error = p @ F - mu
    return Tilt(beta=full, p=p, objective=float("nan"), iterations=step_count,
                converged=converged, message=message), error
