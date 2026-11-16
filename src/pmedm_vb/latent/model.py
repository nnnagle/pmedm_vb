"""The latent-space PMEDM over one area: x multipliers shared by the area's
zones, a coefficient vector per zone on features of φ.

For unit i and zone b of an area,

    η_ib = x_i′λ + c_b + h_i′θ_b,      p_ib = q_ib exp(−η_ib) / Σ q exp(−η),

one distribution over (zone, unit) pairs, as in the PMEDM (``q`` is the
PMEDM's design distribution, :func:`~pmedm_vb.assemble.design.design_weights`).
The objective (``HANDOFF_stage2.md``) is the PMEDM's primal restricted to this
family:

    f = ½ r′(Σ + τ²I)⁻¹ r + α n KL(p ‖ q) + Σ_b ‖θ̃_b‖² / 2τ_θ² + Σ_b c_b² / 2τ_c²,
    r = N Ã′p − Y.

**Terms.**

* ``Y``, ``Ã`` (applied by :class:`~pmedm_vb.solvers.base.ConstraintOperator`),
  ``N``, ``n`` and ``Σ`` are those of a :class:`~pmedm_vb.assemble.inputs.PMEDMInputs`.
  ``Σ`` arrives with the PMEDM's multiplier prior already on its diagonal,
  ``Σ + τ²I`` with ``τ = N/(n s)`` (:mod:`pmedm_vb.assemble.prior`), which is
  how the paper baseline applies its prior; there is no separate ridge on λ,
  which would count that prior twice. Its inverse is applied by
  :meth:`~pmedm_vb.assemble.sigma.Sigma.solve` (Woodbury on the
  diagonal-plus-low-rank structure).
* ``x_i`` are the unit's counts in the area's constraint cells: the
  zone-level (block-group) cells, plus any tract-level cells whose category is
  not also a zone-level one (tract-only tables). λ is shared by every zone.
* ``h_i`` are the unit's features of φ (the decoder probabilities and the
  group-quarters indicators; :mod:`pmedm_vb.latent.regions`). Non-indicator
  columns are standardised over the area's units with the design weights, so
  θ̃ is on that standardised scale and ``τ_θ`` means the same for every
  feature. Columns constant over the area are dropped.
* ``c_b`` is a zone intercept: the zone's level, which features without a
  constant column cannot set (a zone with more people needs every unit raised
  alike). It is ``h_i`` with a constant column appended; standardising the
  other columns is then a reparameterisation of ``c_b``. Its prior sd ``τ_c``
  is on the log-weight scale.
* **Sum to zero.** The zone coefficients ``[θ̃_b, c_b]`` satisfy
  ``Σ_b w_b [θ̃_b, c_b] = 0`` for zone weights ``w_b`` (``N_b``: the zones'
  sizes; see :func:`zone_weights_from_inputs`), so λ carries the area's level
  and the zones only their differences. Implemented exactly by
  reparameterisation: the zone coefficients are ``M Γ`` with ``M`` an
  orthonormal basis of the complement of ``w`` (``M′w = 0``, ``M′M = I``),
  and Γ is free. Since ``M′M = I``, ``Σ_b ‖θ̃_b‖² = ‖Γ_θ‖²``, so the prior is
  a plain ridge on Γ.

**Gradient.** With ``E = η`` (zones × units), ``s = (Σ+τ²I)⁻¹ r`` and
``G = N Ã s`` laid out as a weight matrix
(:meth:`~pmedm_vb.solvers.base.ConstraintOperator.adjoint`),

    ∂f/∂E = −p ∘ (G − ⟨p, G⟩) + α n · p ∘ (E − ⟨p, E⟩),

(the first from the misfit through the softmax, the second from
``KL = −⟨p, E⟩ − log Z̃``), then ``∂f/∂λ = X′ Σ_b (∂f/∂E)_b`` and
``∂f/∂Γ = M′ (∂f/∂E) H₁`` plus the prior, ``H₁ = [H̃, 1]``.

**Fitting.** Damped Newton with the exact gradient and Hessian: the
Gauss-Newton curvature ``J′(Σ+τ²I)⁻¹J + α n Cov_p(Φ) + diag(prior)``
(``J = ∂r/∂(λ, Γ)``, ``Φ`` the exponent's derivative) plus the residual
terms it drops (:meth:`LatentModel.residual_curvature`), which matter here
because the family leaves large residuals: Gauss-Newton alone converges
slowly. Steps are kept only when they lower ``f``, with Marquardt's rule for
the damping. The Hessian is dense in the parameters, so problems with more
than :data:`GN_LIMIT` parameters (a PUMA's block groups) use L-BFGS on the
same objective instead. The objective is not convex in (λ, Γ) (``p`` is
a softmax of them), and fits from different starts stop at different
stationary points; :meth:`LatentModel.project` gives a start from the PMEDM's
solution. No uncertainty is computed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse as sp
from scipy.linalg import null_space
from scipy.special import logsumexp

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.prior import prior_variance, with_prior
from pmedm_vb.assemble.sigma import Sigma
from pmedm_vb.progress import logger
from pmedm_vb.solvers.base import ConstraintOperator

#: Above this many parameters the dense Gauss-Newton curvature is not formed.
GN_LIMIT = 4000


def shared_counts(inputs: PMEDMInputs) -> tuple[sp.csr_matrix, list[str]]:
    """The unit's counts in the area's distinct constraint cells: every
    zone-level column, plus tract-level columns whose name no zone-level column
    has (tract-only tables)."""
    bg = list(inputs.bg_constraints)
    extra = [k for k, name in enumerate(inputs.tract_constraints) if name not in set(bg)]
    blocks = [sp.csr_matrix(inputs.X_B)]
    names = bg[:]
    if extra:
        blocks.append(sp.csr_matrix(inputs.X_T)[:, extra])
        names += [inputs.tract_constraints[k] for k in extra]
    return sp.hstack(blocks).tocsr(), names


def sum_to_zero_basis(weights: np.ndarray) -> np.ndarray:
    """``(Z, Z−1)`` orthonormal basis ``M`` of ``{v : w′v = 0}``."""
    w = np.asarray(weights, dtype=np.float64).reshape(1, -1)
    if not np.all(w > 0):
        raise ValueError("zone weights must be positive")
    return null_space(w)


def zone_weights_from_inputs(inputs: PMEDMInputs, columns: list[int] | None = None) -> np.ndarray:
    """``N_b``: each zone's observed size, the sum of its zone-level ``Y``
    over ``columns`` (cells that partition the units, e.g. the unit type);
    equal weights when ``columns`` is None."""
    if columns is None:
        return np.ones(inputs.n_zones)
    w = np.asarray(inputs.Y_B[:, columns], dtype=np.float64).sum(axis=1)
    if not np.all(w > 0):
        raise ValueError("a zone has no observed units in the given columns")
    return w


def standardise(H: np.ndarray, binary: np.ndarray, weight: np.ndarray):
    """Design-weighted standardisation of the non-indicator columns of ``H``;
    columns constant over the units are dropped. Returns the standardised
    matrix, the kept column index, and the means and sds used."""
    H = np.asarray(H, dtype=np.float64)
    w = weight / weight.sum()
    mean = w @ H
    sd = np.sqrt(np.maximum(w @ np.square(H) - np.square(mean), 0.0))
    keep = sd > 1e-12 * np.maximum(1.0, np.abs(mean))
    centre = np.where(binary, 0.0, mean)
    spread = np.where(binary, 1.0, np.where(sd > 0, sd, 1.0))
    out = (H[:, keep] - centre[keep]) / spread[keep]
    return out, np.flatnonzero(keep), centre, spread


@dataclass
class LatentFit:
    """A fitted latent model over one area."""

    lam: np.ndarray            # shared multipliers on the distinct cells
    gamma: np.ndarray          # (Z−1, k+1) free zone parameters
    theta: np.ndarray          # (Z, k+1) zone coefficients [θ̃_b, c_b] = M Γ
    p: np.ndarray              # (Z, U) distribution over pairs
    objective: float
    parts: dict
    iterations: int
    converged: bool
    message: str
    seconds: float
    solver: str
    feature_names: list[str] = field(default_factory=list)


class LatentModel:
    """Objective, gradient and Gauss-Newton curvature of the latent model.

    Parameters
    ----------
    inputs:
        The area's problem (``q``, ``X``, ``A``, ``Y``, ``N``, ``n``).
    sigma:
        ``Σ`` with the multiplier prior on its diagonal (see
        :func:`area_sigma`).
    H, binary, names:
        Units x features of φ (before standardisation) and which columns are
        0/1 indicators. ``H`` may have zero columns (an x-only model with
        zone intercepts).
    zone_weight:
        ``w_b`` of the sum-to-zero constraint.
    alpha:
        Entropy weight; the KL term is ``alpha * n * KL(p ‖ q)``.
    tau_theta, tau_intercept:
        Prior sds of the standardised feature coefficients and of ``c_b``.
    """

    def __init__(self, inputs: PMEDMInputs, sigma: Sigma, H: np.ndarray, binary: np.ndarray,
                 *, zone_weight: np.ndarray, alpha: float, tau_theta: float = 3.0,
                 tau_intercept: float = 3.0, names: list[str] | None = None) -> None:
        self.inputs, self.sigma = inputs, sigma
        self.op = ConstraintOperator(inputs)
        self.N, self.n = float(inputs.N), float(inputs.n)
        self.Y = inputs.targets()
        with np.errstate(divide="ignore"):
            self.log_q = np.log(inputs.q)
        if not np.all(np.isfinite(self.log_q)):
            raise ValueError("q has zeros; the latent model expects every pair possible")
        self.X, self.x_names = shared_counts(inputs)
        self.XT = self.X.T.tocsr()
        unit_weight = inputs.q.sum(axis=0)
        H = np.zeros((inputs.n_units, 0)) if H is None else np.asarray(H, dtype=np.float64)
        Hs, kept, _, _ = standardise(H, np.asarray(binary, bool), unit_weight)
        names = list(names) if names is not None else [f"h{j}" for j in range(H.shape[1])]
        self.feature_names = [names[j] for j in kept] + ["(intercept)"]
        self.H1 = np.column_stack([Hs, np.ones(inputs.n_units)])
        self.Z, self.U = inputs.q.shape
        self.k1 = self.H1.shape[1]
        self.M = sum_to_zero_basis(zone_weight)
        self.zone_weight = np.asarray(zone_weight, dtype=np.float64)
        self.entropy = float(alpha) * self.n
        self.alpha = float(alpha)
        prior = np.full(self.k1, 1.0 / tau_theta**2)
        prior[-1] = 1.0 / tau_intercept**2
        self.prior_gamma = np.tile(prior, (self.Z - 1, 1))
        self.n_lam = self.X.shape[1]
        self.n_params = self.n_lam + (self.Z - 1) * self.k1

    # -- parameters ----------------------------------------------------------

    def split(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lam = params[: self.n_lam]
        gamma = params[self.n_lam:].reshape(self.Z - 1, self.k1)
        return lam, gamma

    def exponent(self, params: np.ndarray) -> np.ndarray:
        """``η``, zones x units."""
        lam, gamma = self.split(params)
        theta = self.M @ gamma
        return (self.X @ lam)[None, :] + theta @ self.H1.T

    def distribution(self, E: np.ndarray) -> tuple[np.ndarray, float]:
        """``p`` and ``log Z̃`` (relative to ``q``, so 0 at η = 0)."""
        a = self.log_q - E
        log_z = float(logsumexp(a)) - float(logsumexp(self.log_q))
        top = a.max()
        p = np.exp(a - top)
        p /= p.sum()
        return p, log_z

    # -- objective -----------------------------------------------------------

    def evaluate(self, params: np.ndarray, gradient: bool = True):
        """Objective (and gradient), with its parts and the state the
        curvature needs."""
        lam, gamma = self.split(params)
        E = self.exponent(params)
        p, log_z = self.distribution(E)
        r = self.N * self.op.forward(p) - self.Y
        s = self.sigma.solve(r)
        mean_e = float((p * E).sum())
        kl = -mean_e - log_z
        misfit = 0.5 * float(r @ s)
        prior = 0.5 * float((self.prior_gamma * np.square(gamma)).sum())
        value = misfit + self.entropy * kl + prior
        parts = {"misfit": misfit, "kl": kl, "entropy_term": self.entropy * kl, "prior": prior}
        state = {"p": p, "E": E, "r": r}
        if not gradient:
            return value, None, parts, state
        G = self.N * self.op.adjoint(s)
        state["G"] = G
        dE = -p * (G - float((p * G).sum())) + self.entropy * p * (E - mean_e)
        g_lam = self.XT @ dE.sum(axis=0)
        g_gamma = self.M.T @ (dE @ self.H1) + self.prior_gamma * gamma
        return value, np.concatenate([g_lam, g_gamma.ravel()]), parts, state

    # -- Gauss-Newton curvature ---------------------------------------------

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        """``J = ∂r/∂(λ, Γ)``, constraint rows x parameters, dense.

        Column by column ``−N [Ã(p∘φ) − ⟨p,φ⟩ Ã p]`` for the exponent
        derivative ``φ`` of each parameter: ``x_c`` in every zone for λ_c,
        ``M_bm h_ij`` for Γ_mj. Assembled per zone from ``X′ diag(p_b) X_L``
        and ``H₁′ diag(p_b) X_L`` for each constraint level L (tract, block
        group), in the stacked order ``[vec(Y_T); vec(Y_B)]``.
        """
        Z, k1, nl = self.Z, self.k1, self.n_lam
        M, H1 = self.M, self.H1
        mean_lam = self.XT @ p.sum(axis=0)
        mean_gamma = (M.T @ (p @ H1)).reshape(Z - 1, k1)
        blocks = []
        levels = ((self.op.A_T, self.op.X_T, self.op.shape_T),
                  (self.op.A_B, self.op.X_B, self.op.shape_B))
        for A, XL, (n_areas, n_cells) in levels:
            if n_areas * n_cells == 0:
                continue
            A = A.toarray()
            K_lam = np.zeros((Z, nl, n_cells))
            K_h = np.zeros((Z, k1, n_cells))
            P_x = np.zeros((Z, n_cells))
            for b in range(Z):
                weighted = XL.multiply(p[b][:, None]).tocsc()   # diag(p_b) X_L
                K_lam[b] = (self.XT @ weighted).toarray()
                K_h[b] = H1.T @ weighted.toarray()
                P_x[b] = np.asarray(XL.T @ p[b]).ravel()
            Ap = A @ P_x                                          # Ã p, (areas, cells)
            J_lam = np.einsum("ab,bcl->lac", A, K_lam) - np.einsum("c,al->lac", mean_lam, Ap)
            J_gam = (np.einsum("ab,bm,bjl->lamj", A, M, K_h)
                     - np.einsum("mj,al->lamj", mean_gamma, Ap))
            blocks.append(-self.N * np.concatenate(
                [J_lam.reshape(n_cells * n_areas, nl),
                 J_gam.reshape(n_cells * n_areas, (Z - 1) * k1)], axis=1))
        return np.vstack(blocks)

    def moments(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``Φ′ diag(w) Φ`` and ``Φ′ w`` for weights ``w`` on the (zone, unit)
        pairs, ``Φ`` the exponent's derivative in (λ, Γ), blockwise."""
        Z, k1, nl = self.Z, self.k1, self.n_lam
        M, H1, X = self.M, self.H1, self.X
        w_unit = w.sum(axis=0)
        mean = np.concatenate([self.XT @ w_unit, (M.T @ (w @ H1)).ravel()])
        S_ll = (self.XT @ X.multiply(w_unit[:, None])).toarray()
        S_lg = np.zeros((nl, Z - 1, k1))
        S_gg = np.zeros((Z - 1, k1, Z - 1, k1))
        for b in range(Z):
            XwH = np.asarray(self.XT @ (H1 * w[b][:, None]))     # (nl, k1)
            S_lg += np.einsum("m,cj->cmj", M[b], XwH)
            HwH = H1.T @ (H1 * w[b][:, None])                   # (k1, k1)
            S_gg += np.einsum("m,n,jk->mjnk", M[b], M[b], HwH)
        S = np.block([
            [S_ll, S_lg.reshape(nl, -1)],
            [S_lg.reshape(nl, -1).T, S_gg.reshape((Z - 1) * k1, (Z - 1) * k1)]])
        return S, mean

    def curvature(self, state: dict, exact: bool = False) -> np.ndarray:
        """Gauss-Newton curvature ``J′(Σ+τ²I)⁻¹J + α n Cov_p(Φ) +
        diag(prior)``, dense (n_params²); with ``exact``, the Hessian (adds
        :meth:`residual_curvature`)."""
        p = state["p"]
        J = self.jacobian(p)
        H = J.T @ self.sigma.solve(J)
        S, mean = self.moments(p)
        H += self.entropy * (S - np.outer(mean, mean))
        H[np.diag_indices_from(H)] += np.concatenate(
            [np.zeros(self.n_lam), self.prior_gamma.ravel()])
        if exact:
            H += self.residual_curvature(state, mean)
        return H

    def residual_curvature(self, state: dict, mean: np.ndarray | None = None) -> np.ndarray:
        """The Hessian less the Gauss-Newton curvature: the second derivatives
        of ``r`` and of the KL that Gauss-Newton drops.

        In ``η`` both are ``diag(a) − a p′ − p a′`` with
        ``a = p ∘ [(G − ⟨p,G⟩) − α n (E − ⟨p,E⟩)]``, ``G`` held at the current
        ``s``. They matter when the residuals are large, which the latent
        family, with fewer parameters than cells, leaves them. Not positive
        definite away from the optimum.
        """
        p, E, G = state["p"], state["E"], state["G"]
        if mean is None:
            _, mean = self.moments(p)
        a = p * ((G - float((p * G).sum())) - self.entropy * (E - float((p * E).sum())))
        S_a, mean_a = self.moments(a)
        return S_a - np.outer(mean_a, mean) - np.outer(mean, mean_a)

    def project(self, p: np.ndarray) -> np.ndarray:
        """Parameters whose exponent best matches another distribution ``p``'s,
        ``η = log(q/p)``: least squares weighted by ``p``, with a free constant
        (the normalisation) and a ridge of 1e-3 times the prior (1e-11 on λ,
        whose all-units direction is unidentified). A start for :meth:`fit`
        from the PMEDM's solution, which lies in a better basin than ``η = 0``
        for this non-convex objective."""
        with np.errstate(divide="ignore"):
            eta = self.log_q - np.log(p)
        keep = p > 0
        w = np.where(keep, p, 0.0)
        eta = np.where(keep, eta, 0.0)
        S, mean = self.moments(w)
        _, rhs = self.moments(w * eta)
        A = S - np.outer(mean, mean) / w.sum()
        rhs = rhs - mean * float((w * eta).sum()) / w.sum()
        A[np.diag_indices_from(A)] += 1e-3 * np.concatenate(
            [np.full(self.n_lam, 1e-8), self.prior_gamma.ravel()])
        return np.linalg.solve(A, rhs)

    # -- fitting -------------------------------------------------------------

    def fit(self, *, solver: str = "auto", max_iter: int = 300, tol: float = 1e-10,
            init: np.ndarray | None = None, label: str = "") -> LatentFit:
        """MAP by damped Newton (``"newton"``, the exact Hessian), damped
        Gauss-Newton (``"gn"``) or L-BFGS (``"lbfgs"``); ``"auto"`` uses Newton
        up to :data:`GN_LIMIT` parameters."""
        if solver == "auto":
            solver = "newton" if self.n_params <= GN_LIMIT else "lbfgs"
        start = time.perf_counter()
        x0 = np.zeros(self.n_params) if init is None else np.asarray(init, float).copy()
        if solver in ("newton", "gn"):
            params, value, parts, p, iterations, converged, message = self._fit_newton(
                x0, max_iter=max_iter, tol=tol, exact=solver == "newton")
        elif solver == "lbfgs":
            params, value, parts, p, iterations, converged, message = self._fit_lbfgs(
                x0, max_iter=max_iter * 50, tol=tol)
        else:
            raise ValueError(f"unknown solver {solver!r}")
        lam, gamma = self.split(params)
        seconds = time.perf_counter() - start
        logger.info("%s latent fit (%s, %d parameters): %s after %d iterations, f %.6g "
                    "(misfit %.6g, KL %.4g), %.1fs", label, solver, self.n_params,
                    "converged" if converged else "NOT converged", iterations, value,
                    parts["misfit"], parts["kl"], seconds)
        return LatentFit(lam=lam, gamma=gamma, theta=self.M @ gamma, p=p, objective=value,
                         parts=parts, iterations=iterations, converged=converged,
                         message=message, seconds=seconds, solver=solver,
                         feature_names=self.feature_names)

    def _fit_newton(self, params, *, max_iter, tol, exact):
        """Damped (Gauss-)Newton: Marquardt damping ``μ diag(GN curvature)``,
        raised until the damped matrix is positive definite and the step lowers
        ``f``, relaxed after each accepted step."""
        value, grad, parts, state = self.evaluate(params)
        mu, converged, message, iterations = 1e-3, False, "maximum iterations", 0
        for iterations in range(1, max_iter + 1):
            H = self.curvature(state)
            diag = np.maximum(np.diag(H), 1e-12)
            if np.max(np.abs(grad) / np.sqrt(diag)) < 1e-8:
                converged, message = True, "gradient below tolerance"
                break
            if exact:
                H = H + self.residual_curvature(state)
            accepted = False
            while mu < 1e12:
                try:
                    factor = scipy.linalg.cho_factor(H + mu * np.diag(diag))
                except np.linalg.LinAlgError:
                    mu *= 10
                    continue
                step = -scipy.linalg.cho_solve(factor, grad)
                trial = self.evaluate(params + step)
                if trial[0] < value:
                    accepted = True
                    break
                mu *= 10
            if not accepted:
                message = "no decreasing step at maximum damping"
                converged = bool(np.max(np.abs(grad) / np.sqrt(diag)) < 1e-5)
                break
            change = value - trial[0]
            params = params + step
            value, grad, parts, state = trial
            mu = max(mu / 3, 1e-12)
            if change <= tol * max(1.0, abs(value)):
                converged, message = True, "relative reduction below tolerance"
                break
        return params, value, parts, state["p"], iterations, converged, message

    def _fit_lbfgs(self, params, *, max_iter, tol):
        def fun(x):
            value, grad, _, _ = self.evaluate(x)
            return value, grad

        result = scipy.optimize.minimize(
            fun, params, jac=True, method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": tol, "gtol": 1e-9, "maxcor": 30})
        value, _, parts, state = self.evaluate(result.x, gradient=False)
        return (result.x, value, parts, state["p"], int(result.nit), bool(result.success),
                str(result.message))


def area_sigma(inputs: PMEDMInputs, *, sigma_alpha: float = 0.01, taper: str | None = None,
               prior_sd: float | None = 3.0, variance_floor=None) -> Sigma:
    """``Σ + τ²I`` as the paper baseline builds it: the replicate covariance
    shrunk by ``sigma_alpha`` (:meth:`~pmedm_vb.assemble.inputs.PMEDMInputs.sigma`)
    and the multiplier prior ``τ = N/(n s)`` on its diagonal."""
    base = inputs.sigma(sigma_alpha, taper=taper, variance_floor=variance_floor)
    return with_prior(base, prior_variance(prior_sd, inputs.n, inputs.N))
