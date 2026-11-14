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

**Feature scale.** Both fits accept ``scale``, a per-feature standard
deviation (``1`` for 0/1 indicators and constraint counts). For
:func:`fit_tilt` it enters the prior as ``½ Σ_j (s_j β_j)² / τ²``, which is
exactly the MAP fit on standardised features ``(F_j − m_j) / s_j``: the shift
``m_j`` changes only ``Z``, and ``β`` on the standardised scale is ``s_j β_j``.
So τ means the same thing for features of any scale (positions, their
products, predicted probabilities) without a standardised copy of ``F``.
:func:`fit_oracle` matches moments exactly, so its distribution does not
depend on the scale at all; it standardises internally only for conditioning
and to state its tolerance in standard deviations.

``F`` may be one matrix or a :class:`Features` of several blocks (sparse
constraint counts beside dense position features), multiplied block by block.
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


class Features:
    """Column blocks of a feature matrix, each kept in its own storage: sparse
    blocks as CSR (with their transpose precomputed), dense blocks as arrays.
    Supports ``F @ beta`` and :meth:`rmatvec` (``F′ v``)."""

    def __init__(self, blocks):
        self.blocks, self.transposed = [], []
        for block in blocks:
            if sp.issparse(block):
                block = block.tocsr()
                self.transposed.append(block.T.tocsr())
            else:
                block = np.asarray(block, dtype=np.float64)
                self.transposed.append(None)
            self.blocks.append(block)
        rows = {b.shape[0] for b in self.blocks}
        if len(rows) != 1:
            raise ValueError(f"feature blocks differ in rows: {sorted(rows)}")
        self.edges = np.cumsum([0, *(b.shape[1] for b in self.blocks)])
        self.shape = (rows.pop(), int(self.edges[-1]))

    def __matmul__(self, beta: np.ndarray) -> np.ndarray:
        out = np.zeros(self.shape[0])
        for block, start, end in zip(self.blocks, self.edges[:-1], self.edges[1:]):
            out += block @ beta[start:end]
        return out

    def rmatvec(self, v: np.ndarray) -> np.ndarray:
        return np.concatenate([
            (t @ v) if t is not None else (b.T @ v)
            for b, t in zip(self.blocks, self.transposed)])


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
    scale: np.ndarray | None = None,
    method: str = "lbfgs",
    max_iter: int = 5000,
    tol: float = 1e-9,
) -> Tilt:
    """MAP tilt on features ``F`` fitted to targets ``y`` (see the module notes).

    ``F`` and ``X`` have one row per unit (support or not); ``log_q`` is
    ``−inf`` off the support. ``scale`` (per feature; default all 1) puts the
    prior on the standardised scale (module notes).

    ``method``: ``"lbfgs"`` (quasi-Newton on f, the default) or ``"lm"``, the
    same objective written as a sum of squares -- the standardised cell
    residuals stacked over the prior terms ``s_j β_j / τ`` -- and solved by
    Levenberg-Marquardt with the exact Jacobian (:func:`_fit_tilt_lm`). The
    minimiser is the same; Levenberg-Marquardt copes far better with many
    nearly collinear features (the decoder's probabilities), where L-BFGS
    stalls.
    """
    F = F if isinstance(F, Features) else Features([_as_csr(F)])
    X = X.tocsr()
    inv_var = 1.0 / np.square(se)
    s2 = np.ones(F.shape[1]) if scale is None else np.square(np.asarray(scale, dtype=np.float64))
    prior = s2 / tau**2

    def objective(beta):
        p = distribution(log_q, F, beta)
        r = y - N * (X.T @ p)
        g = r * inv_var
        u = X @ g
        v = p * (u - p @ u)
        value = 0.5 * float(r @ g) + 0.5 * float(prior @ np.square(beta))
        grad = N * F.rmatvec(v) + prior * beta
        return value, grad

    if method == "lm":
        return _fit_tilt_lm(log_q, F, X, y, se, N, prior, max_iter=max_iter, tol=tol)
    if method != "lbfgs":
        raise ValueError(f"method must be 'lbfgs' or 'lm', got {method!r}")
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


def _fit_tilt_lm(log_q, F: Features, X: sp.csr_matrix, y, se, N, prior, *, max_iter, tol) -> Tilt:
    """:func:`fit_tilt`'s objective by Levenberg-Marquardt.

    Residuals ``e = [(y − N X′p) / se ; √prior ∘ β]``, so ½‖e‖² is f. With
    ∂p_u/∂β_j = −p_u (F_uj − E_p F_j), the cell block of the Jacobian is
    ``(N / se) ∘ X′ diag(p) (F − 1 E_p[F]′)``, built block by block.
    """
    root_prior = np.sqrt(prior)
    support = np.flatnonzero(np.isfinite(log_q))
    Xs = X[support]
    blocks = [b[support] for b in F.blocks]

    def residuals(beta):
        p = distribution(log_q, F, beta)
        return np.concatenate([(y - N * (X.T @ p)) / se, root_prior * beta])

    def jacobian(beta):
        p = distribution(log_q, F, beta)[support]
        cols = []
        for block in blocks:
            mean = p @ block if not sp.issparse(block) else np.asarray(block.T @ p).ravel()
            if sp.issparse(block):
                weighted = np.asarray((Xs.T.multiply(p[None, :]) @ block).todense())
            else:
                weighted = np.asarray(Xs.T @ (p[:, None] * block))
            xp = np.asarray(Xs.T @ p).ravel()
            cols.append(weighted - np.outer(xp, mean))
        cell = (N / se)[:, None] * np.hstack(cols)
        return np.vstack([cell, np.diag(root_prior)])

    result = scipy.optimize.least_squares(
        residuals, np.zeros(F.shape[1]), jac=jacobian, method="lm",
        ftol=tol, xtol=1e-12, gtol=1e-10, max_nfev=max_iter)
    beta = result.x
    return Tilt(beta=beta, p=distribution(log_q, F, beta), objective=float(result.cost),
                iterations=int(result.nfev), converged=bool(result.success),
                message=str(result.message))


def fit_oracle(
    log_q: np.ndarray,
    F: np.ndarray,
    mu: np.ndarray,
    *,
    binary: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-10,
    max_damping: float = 1e2,
) -> tuple[Tilt, np.ndarray]:
    """Maximum-entropy tilt with E_p[F] = ``mu`` (dense ``F``, few columns).

    ``binary`` marks 0/1 columns; one whose ``mu`` is 0 is dropped and its
    units removed from the support (see the module notes). Non-binary columns
    are standardised over the support (base-weighted mean and sd) before the
    Newton iterations, which changes nothing in the solution -- the
    moment-matched distribution is the same for any affine map of ``F`` -- but
    keeps the Hessian well scaled when features differ by orders of magnitude
    (products of positions, small probabilities), and makes ``tol`` a
    tolerance in standard deviations. Columns constant over the support are
    dropped. If a Newton step fails its line search the Hessian is damped,
    ``H + δ diag(H)`` with δ growing tenfold from 1e-6 up to ``max_damping``;
    the largest δ used is reported in the message.

    Returns the tilt (``beta`` on the original feature scale) and the final
    moment error E_p[F] − μ over all columns, on the original scale.
    """
    F = np.asarray(F, dtype=np.float64)
    log_q = log_q.copy()
    keep = np.ones(F.shape[1], dtype=bool)
    if binary is None:
        binary = np.zeros(F.shape[1], dtype=bool)
    for j in np.flatnonzero(binary):
        if mu[j] <= 0:
            log_q[F[:, j] > 0] = -np.inf
            keep[j] = False
    support = np.isfinite(log_q)
    q = np.exp(log_q[support] - log_q[support].max())
    q /= q.sum()
    centre = np.zeros(F.shape[1])
    spread = np.ones(F.shape[1])
    for j in np.flatnonzero(keep & ~binary):
        m = float(q @ F[support, j])
        sd = float(np.sqrt(q @ np.square(F[support, j] - m)))
        if sd <= 1e-12 * max(1.0, abs(m)):
            keep[j] = False
        else:
            centre[j], spread[j] = m, sd
    Fk = (F[:, keep] - centre[keep]) / spread[keep]
    muk = (mu[keep] - centre[keep]) / spread[keep]
    beta = np.zeros(Fk.shape[1])
    converged, message, step_count, damping_used = False, "maximum iterations", 0, 0.0

    def dual(b):
        a = log_q[support] - Fk[support] @ b
        top = a.max()
        return top + np.log(np.exp(a - top).sum()) + b @ muk

    for step_count in range(1, max_iter + 1):
        p = distribution(log_q, Fk, beta)
        mean = p @ Fk
        grad = muk - mean
        if np.max(np.abs(grad)) < tol:
            converged, message = True, "moments matched"
            break
        centred = (Fk[support] - mean) * np.sqrt(p[support])[:, None]
        hess = centred.T @ centred + 1e-12 * np.eye(len(beta))
        current, damping, accepted = dual(beta), 0.0, False
        while not accepted:
            h = hess + damping * np.diag(np.diag(hess)) if damping else hess
            direction = -np.linalg.solve(h, grad)
            t = 1.0
            while dual(beta + t * direction) > current + 1e-4 * t * (grad @ direction) and t > 1e-10:
                t *= 0.5
            if t > 1e-10:
                accepted = True
            elif damping >= max_damping:
                break
            else:
                damping = 1e-6 if damping == 0 else damping * 10
        if not accepted:
            message = "line search failed at maximum damping"
            break
        damping_used = max(damping_used, damping)
        beta = beta + t * direction
    if damping_used:
        message += f"; damping up to {damping_used:g}"
    full = np.zeros(F.shape[1])
    full[keep] = beta / spread[keep]
    p = distribution(log_q, Fk, beta)
    error = p @ F - mu
    return Tilt(beta=full, p=p, objective=float("nan"), iterations=step_count,
                converged=converged, message=message), error
