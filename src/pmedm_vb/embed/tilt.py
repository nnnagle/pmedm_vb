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
    features are compared on one footing. With ``alpha = 0`` (the default)
    it is a MAP form, **not** the PMEDM's penalized maximum-entropy problem:
    the prior is on β directly, with no entropy weight, and ``Σ`` is diagonal.

    **Entropy penalty.** ``alpha > 0`` adds the PMEDM's least-distortion term,

        f(β) + α n KL(p ‖ q),   KL(p ‖ q) = Σ_i p_i log(p_i / q_i) = −(Fβ)′p − log Z,

    with ``q`` normalised over the support and ``Z = Σ_i q_i exp(−F_i β)``.
    Its gradient is ``F′(p ∘ (Fβ − p′Fβ))``, the covariance under ``p`` of
    the features with the exponent. ``n`` is the sample size the PMEDM
    weights its entropy by (``pmedm_derivation.md``: the PMEDM maximises
    ``−n KL(p ‖ q) − ½ e′Σ⁻¹e`` with ``e = N X′p − y`` and ``n`` the number of
    sample units, the ``inputs.n`` of :mod:`pmedm_vb.solvers.base`, whose dual
    carries ``c = n/N²``). So ``alpha = 1`` is the PMEDM's own balance of
    entropy against Gaussian misfit: for ``F = X`` with no prior (τ → ∞) the
    fit is the PMEDM's maximum-entropy solution, whose primal optimum lies in
    this exponential family. With a finite τ the prior acts on β in this
    primal parameterisation, not on the dual multipliers λ as the PMEDM's
    multiplier prior does, so the two agree only as τ → ∞. Among tilts that
    fit the cells equally well the term prefers the one closest to ``q``,
    which is what keeps an ``x`` tilt on one-way margins from changing the
    support's odds ratios; a rich feature map has no such restraint without
    it. These fits use :func:`_fit_tilt_gn`, a damped Gauss-Newton method
    (the KL term is not a sum of squares, so Levenberg-Marquardt as written
    does not apply), or L-BFGS on request.

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
    alpha: float = 0.0,
    n: float = 0.0,
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

    ``alpha`` and ``n`` add the entropy penalty ``alpha * n * KL(p ‖ q)``
    (module notes). With it ``method`` may be ``"gn"`` (damped Gauss-Newton,
    :func:`_fit_tilt_gn`) or ``"lbfgs"``; ``"lm"`` is taken as ``"gn"``.
    """
    F = F if isinstance(F, Features) else Features([_as_csr(F)])
    entropy = float(alpha) * float(n)
    if entropy < 0:
        raise ValueError(f"alpha * n must be non-negative, got {entropy}")
    X = X.tocsr()
    inv_var = 1.0 / np.square(se)
    s2 = np.ones(F.shape[1]) if scale is None else np.square(np.asarray(scale, dtype=np.float64))
    prior = s2 / tau**2

    def objective(beta):
        if entropy:
            return _penalised(log_q, F, X, y, inv_var, N, prior, entropy, beta)[:2]
        p = distribution(log_q, F, beta)
        r = y - N * (X.T @ p)
        g = r * inv_var
        u = X @ g
        v = p * (u - p @ u)
        value = 0.5 * float(r @ g) + 0.5 * float(prior @ np.square(beta))
        grad = N * F.rmatvec(v) + prior * beta
        return value, grad

    if entropy and method in ("gn", "lm"):
        return _fit_tilt_gn(log_q, F, X, y, se, N, prior, entropy,
                            max_iter=min(max_iter, 500), tol=tol)
    if method == "gn":
        if not entropy:
            raise ValueError("method 'gn' is for fits with the entropy penalty; use 'lm'")
    if method == "lm":
        return _fit_tilt_lm(log_q, F, X, y, se, N, prior, max_iter=max_iter, tol=tol)
    if method != "lbfgs":
        raise ValueError(f"method must be 'lbfgs', 'lm' or 'gn', got {method!r}")
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


def log_partition(log_q: np.ndarray, Fb: np.ndarray) -> tuple[np.ndarray, float]:
    """``p ∝ q exp(−Fβ)`` from ``Fb = Fβ``, and ``log Z = log Σ q̃ exp(−Fβ)``
    with ``q̃`` normalised over the support (so ``log Z = 0`` at β = 0)."""
    finite = np.isfinite(log_q)
    a = np.where(finite, log_q - Fb, -np.inf)
    top_a, top_q = a[finite].max(), log_q[finite].max()
    e = np.zeros_like(a)
    e[finite] = np.exp(a[finite] - top_a)
    s_a = e.sum()
    s_q = np.exp(log_q[finite] - top_q).sum()
    log_z = (top_a + np.log(s_a)) - (top_q + np.log(s_q))
    return e / s_a, float(log_z)


def kl_to_base(log_q: np.ndarray, p: np.ndarray) -> float:
    """KL(p ‖ q̃) over the support, ``q̃`` the normalised base weights."""
    finite = np.isfinite(log_q)
    lq = log_q[finite] - log_q[finite].max()
    lq = lq - np.log(np.exp(lq).sum())
    pf = p[finite]
    pos = pf > 0
    return float((pf[pos] * (np.log(pf[pos]) - lq[pos])).sum())


def _penalised(log_q, F, X, y, inv_var, N, prior, entropy, beta):
    """Value and gradient of f(β) + entropy · KL(p ‖ q) (module notes), and
    ``p``. KL = −(Fβ)′p − log Z; ∇KL = F′(p ∘ (Fβ − p′Fβ))."""
    Fb = F @ beta
    p, log_z = log_partition(log_q, Fb)
    r = y - N * (X.T @ p)
    g = r * inv_var
    u = X @ g
    on = p > 0
    fb_mean = float(p[on] @ Fb[on])
    kl = -fb_mean - log_z
    w = np.zeros_like(p)
    w[on] = N * p[on] * (u[on] - p @ u) + entropy * p[on] * (Fb[on] - fb_mean)
    value = 0.5 * float(r @ g) + entropy * kl + 0.5 * float(prior @ np.square(beta))
    grad = F.rmatvec(w) + prior * beta
    return value, grad, p, kl


def _weighted_gram(blocks, p) -> np.ndarray:
    """``F′ diag(p) F`` for column blocks (sparse or dense) of the support rows."""
    k = [b.shape[1] for b in blocks]
    edges = np.cumsum([0, *k])
    out = np.zeros((edges[-1], edges[-1]))
    for i, bi in enumerate(blocks):
        bip = bi.multiply(p[:, None]).tocsr() if sp.issparse(bi) else bi * p[:, None]
        for j in range(i, len(blocks)):
            bj = blocks[j]
            prod = bip.T @ bj
            prod = prod.toarray() if sp.issparse(prod) else np.asarray(prod)
            out[edges[i]:edges[i + 1], edges[j]:edges[j + 1]] = prod
            if j != i:
                out[edges[j]:edges[j + 1], edges[i]:edges[i + 1]] = prod.T
    return out


def _fit_tilt_gn(log_q, F: Features, X: sp.csr_matrix, y, se, N, prior, entropy, *,
                 max_iter, tol) -> Tilt:
    """f(β) + entropy · KL(p ‖ q) by damped Gauss-Newton.

    The curvature is approximated by ``N² A′Σ⁻¹A + entropy · Cov_p(F) +
    diag(prior)``, with ``A = X′ diag(p) (F − 1 E_p[F]′)`` the Jacobian of the
    allocated cells (the Gauss-Newton term of the misfit) and ``Cov_p(F)`` the
    Fisher information of the tilt (the positive part of the KL term's
    Hessian). Steps solve ``(H + μ diag(H)) δ = −∇`` with the exact gradient;
    a step is kept only if it lowers the objective, and μ shrinks after a
    kept step and grows after a refused one (Marquardt's rule). Converged when
    a kept step changes the objective by less than ``tol`` relative, or the
    gradient, scaled by the curvature's diagonal, falls below 1e-8.
    """
    inv_var = 1.0 / np.square(se)
    support = np.flatnonzero(np.isfinite(log_q))
    Xs = X[support]
    blocks = [b[support] for b in F.blocks]
    beta = np.zeros(F.shape[1])
    value, grad, p, _ = _penalised(log_q, F, X, y, inv_var, N, prior, entropy, beta)
    mu, iterations, converged, message = 1e-3, 0, False, "maximum iterations"
    for iterations in range(1, max_iter + 1):
        ps = p[support]
        mean = np.concatenate([
            np.asarray(b.T @ ps).ravel() if sp.issparse(b) else ps @ b for b in blocks])
        xp = np.asarray(Xs.T @ ps).ravel()
        cross = np.hstack([
            np.asarray((Xs.T.multiply(ps[None, :]) @ b).todense()) if sp.issparse(b)
            else np.asarray(Xs.T @ (ps[:, None] * b)) for b in blocks])
        A = cross - np.outer(xp, mean)
        H = (N**2) * (A.T * inv_var) @ A
        if entropy:
            H += entropy * (_weighted_gram(blocks, ps) - np.outer(mean, mean))
        H[np.diag_indices_from(H)] += prior
        diag = np.maximum(np.diag(H), 1e-12)
        if np.max(np.abs(grad) / np.sqrt(diag)) < 1e-8:
            converged, message = True, "gradient below tolerance"
            break
        accepted = False
        while mu < 1e12:
            try:
                step = -np.linalg.solve(H + mu * np.diag(diag), grad)
            except np.linalg.LinAlgError:
                mu *= 10
                continue
            trial = _penalised(log_q, F, X, y, inv_var, N, prior, entropy, beta + step)
            if trial[0] < value:
                accepted = True
                break
            mu *= 10
        if not accepted:
            message = "no decreasing step at maximum damping"
            converged = abs(value) > 0 and np.max(np.abs(grad) / np.sqrt(diag)) < 1e-5
            break
        change = value - trial[0]
        beta = beta + step
        value, grad, p, _ = trial
        mu = max(mu / 3, 1e-9)
        if change <= tol * max(1.0, abs(value)):
            converged, message = True, "relative reduction below tolerance"
            break
    return Tilt(beta=beta, p=p, objective=float(value), iterations=iterations,
                converged=converged, message=message)


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
