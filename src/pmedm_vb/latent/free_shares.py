"""Free zone shares: the realized level of model version 5 (draft,
``MODEL_latent_v5_draft.md``; step 4d, ``PLAN_latent_step4d.md``).

The area total ``N_A`` is known; the zone shares ``s`` are parameters with a
weak ``Dirichlet(a)`` prior (``a = ½``), written ``s = softmax(ψ)`` with the
last coordinate of ``ψ`` fixed at 0. Given the shares, zone ``b`` holds
``N_b = N_A s_b`` units and its composition is

    C_b | s ~ DirMult(N_b, κ s_b p_b),
    E C_b = N_b p_b,  Cov C_b = φ_b N_b (diag p_b - p_b p_b'),
    φ_b = (N_b + κ s_b)/(1 + κ s_b) = s_b (N_A + κ)/(1 + κ s_b)      (standard),

zones independent. The prior shares ``π⁰`` of version 4 leave the model; the
survey's shares are only a starting value.

**Collapsed likelihood** (:class:`FreeShareLikelihood`; derived here). With
``n_b = N_A s_b`` (the mean's weight), ``c_b = φ_b N_b`` (the covariance's)
and ``μ_b = T0'p_b``:

    m = Σ_b n_b D_b μ_b,
    S = Σ + Σ_b c_b D_b V_b D_b',   V_b = T0'(diag p_b - p_b p_b')T0,

block diagonal by zone plus the survey's low rank, as step 2's engine
(:class:`~pmedm_vb.latent.collapsed.CollapsedLikelihood`) but with separate
weights for the mean and the covariance. Version 4's global ``-c vv'`` term
does not arise: given the shares, the totals are fixed.

**Gradient** (derived here). With ``r = S⁻¹(Y - m)``, ``A_bi = t_i'r_b``,
``ā_b = μ_b'r_b``, ``G_b = [S⁻¹]_bb``, ``Q_bi = t_i'G_b t_i`` and
``σ_bi = t_i'G_b μ_b``:

    ∂L/∂u_b = Cov_{p_b}(g, φ_b),
    φ_bi = n_b A_bi + ½ c_b (A_bi² - 2 A_bi ā_b) - ½ c_b (Q_bi - 2 σ_bi);
    ∂L/∂n_b = ā_b,
    ∂L/∂c_b = ½ (E_{p_b}[A²] - ā_b²) - ½ (E_{p_b}[Q] - μ_b'G_b μ_b);
    ∂L/∂s_b = N_A ∂L/∂n_b + ∂L/∂c_b · dc_b/ds_b,
    dc_b/ds_b = N_A (N_A + κ) s_b (2 + κ s_b)/(1 + κ s_b)²
    (c_b = N_A (N_A + κ) s_b²/(1 + κ s_b));
    ∂L/∂ψ_j = s_j (∂L/∂s_j - Σ_b s_b ∂L/∂s_b).

**Prior and objective** (:class:`FreeShareObjective`), over
``x = (vec U, ψ_1..ψ_{Z-1})``:

    F(x) = -L + ‖U‖²/(2τ²) - a Σ_b log s_b,

the last term the ``Dirichlet(a)`` density in ``ψ`` (Jacobian included);
its gradient is ``-a (1 - Z s_j)`` and its Hessian ``a Z (diag s - ss')``
(free coordinates).

**Gauss–Newton information** (:meth:`FreeShareLikelihood.gn_information`):
``J'S⁻¹J`` with ``J = ∂m/∂x`` dense: ``n_b Cov_{p_b}(t0, g)`` on zone
``b``'s rows for ``u_b``, and ``N_A s_j (δ_bj μ_j - s_b μ_b)`` on zone ``b``'s
rows for ``ψ_j``. The mean part only, as version 4's.

**Posterior** (:func:`free_share_posterior_samples`): ``x ~ N(x̂, H⁻¹)``,
``H`` the information plus the priors' Hessians; for each draw, each zone's
functionals conditioned on ``Y`` (derived here): with ``Eg = g'p_b``,
``E2 = (g∘g)'p_b``, ``k1 = T0'diag(p_b)g`` and ``κ_g = k1 - Eg μ_b``,

    E   = n_b Eg + c_b κ_g'r_b,
    Var = c_b (E2 - Eg²) - c_b² κ_g'G_b κ_g.

The zone total (``g = 1``) has ``κ_g = 0``: given the shares it is
``N_A s_b`` exactly, and all its uncertainty comes from ``ψ``'s draws.

**Realized MAP** (:func:`solve_free_share_map`): the conditional PMEDM with
the shares free, ``min_r W Σ_b π_b KL(r_b ‖ p_b) + ½ e'Σ⁻¹e``
(``π_b = Σ_i r_bi``, ``W = N_A/φ``). Since
``KL(r ‖ ρ⊗p) = KL(π ‖ ρ) + Σ_b π_b KL(r_b ‖ p_b)`` (derived here), it is
``min_ρ`` of version 4's objective with base ``ρ⊗p``: alternate the existing
solver (:func:`~pmedm_vb.latent.overdispersed.solve_conditional_pairs`) with
``ρ ←`` the solution's shares, warm-starting each solve from the previous
multipliers, until the shares stop moving. The ``Dirichlet(a)`` term is of
order one against terms of order ``N_A`` and is left out of this step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.special import logsumexp

from pmedm_vb.latent.area import AreaInputs
from pmedm_vb.latent.collapsed import PairOperator
from pmedm_vb.latent.map_fit import _gn_lbfgs, _lbfgs
from pmedm_vb.latent.overdispersed import solve_conditional_pairs
from pmedm_vb.latent.posterior import Functionals
from pmedm_vb.latent.region_engine import RegionLikelihood

#: The Dirichlet prior's parameter on the shares (MODEL_latent_v5_draft.md §2).
SHARE_PRIOR = 0.5


def softmax_shares(psi_free: np.ndarray) -> np.ndarray:
    """``s = softmax(ψ)``, ``ψ = (ψ_free, 0)``."""
    psi = np.append(np.asarray(psi_free, float), 0.0)
    return np.exp(psi - logsumexp(psi))


def shares_to_psi(shares: np.ndarray) -> np.ndarray:
    """The free coordinates of ``ψ`` for given shares (last fixed at 0)."""
    s = np.maximum(np.asarray(shares, float), 1e-300)
    return np.log(s[:-1]) - np.log(s[-1])


def covariance_weights(shares: np.ndarray, N_A: float, kappa: float
                       ) -> tuple[np.ndarray, np.ndarray]:
    """``c_b = φ_b N_A s_b`` and ``dc_b/ds_b`` (module notes); ``κ = ∞``
    gives ``c_b = N_A s_b``."""
    s = np.asarray(shares, float)
    if not np.isfinite(kappa):
        return N_A * s, np.full_like(s, N_A)
    k = float(kappa)
    c = N_A * (N_A + k) * s * s / (1.0 + k * s)
    dc = N_A * (N_A + k) * s * (2.0 + k * s) / (1.0 + k * s) ** 2
    return c, dc


@dataclass
class FreeShareEvaluation:
    loglik: float
    quad: float
    logdet: float
    m: np.ndarray
    mu: np.ndarray               # (C, Z) T0'p_b
    p: np.ndarray                # (M, Z)
    shares: np.ndarray           # (Z,)
    n: np.ndarray                # N_A s
    c: np.ndarray                # φ_b N_A s_b
    dc: np.ndarray
    factors: list
    X: list
    K_factor: tuple | None
    r: np.ndarray                # S⁻¹(Y - m)
    grad_U: np.ndarray | None = None
    grad_s: np.ndarray | None = None
    grad_psi: np.ndarray | None = None
    timings: dict = field(default_factory=dict)


class FreeShareLikelihood(RegionLikelihood):
    """Model version 5's collapsed likelihood for a region (module notes);
    the deviation basis ``g`` with ``θ_A'h`` folded into the zones' log bases
    (evaluate with ``θ_P = 0``), as version 4's engine."""

    def __init__(self, area: AreaInputs, G: np.ndarray, logb_zones: np.ndarray,
                 pairs: PairOperator, N_A: float, kappa: float,
                 share_prior: float = SHARE_PRIOR):
        super().__init__(area, G, logb_zones, pairs)
        if area.J_T != 0 or any(len(t["members"]) != 1 for t in self.tracts):
            raise ValueError("FreeShareLikelihood expects the regional layout (a zone per tract)")
        self.N_A, self.kappa = float(N_A), float(kappa)
        self.share_prior = float(share_prior)

    @property
    def Z(self) -> int:
        return self.area.B

    @property
    def n_params(self) -> int:
        return self.Z * self.K + self.Z - 1

    def split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, float)
        return x[:self.Z * self.K].reshape(self.Z, self.K), x[self.Z * self.K:]

    def join(self, U: np.ndarray, psi_free: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(U, float).ravel(), np.asarray(psi_free, float)])

    # -- solves -----------------------------------------------------------

    def _solve(self, factors, X, K_factor, x: np.ndarray) -> np.ndarray:
        """``S⁻¹ x`` (stacked; ``x`` a vector or a matrix of columns)."""
        out = np.empty(x.shape, dtype=float)
        Lz = 0.0
        for tract, f in zip(self.tracts, factors):
            z = cho_solve(f, x[tract["rows"]], check_finite=False)
            out[tract["rows"]] = z
            if K_factor is not None:
                Lz = Lz + tract["Lr"].T @ z
        if K_factor is not None:
            corr = cho_solve(K_factor, Lz, check_finite=False)
            for tract, Xt in zip(self.tracts, X):
                out[tract["rows"]] -= Xt @ corr
        return out

    def s_solve(self, ev: FreeShareEvaluation, x: np.ndarray) -> np.ndarray:
        return self._solve(ev.factors, ev.X, ev.K_factor, x)

    def zone_inverse(self, ev: FreeShareEvaluation, b: int) -> np.ndarray:
        """``[S⁻¹]_bb`` in zone ``b``'s local order."""
        return self.tract_inverse(ev.factors[b], ev.X[b], ev.K_factor)

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, theta_P, U, psi_free, *, gradient: bool = True) -> FreeShareEvaluation:
        area, pairs = self.area, self.pairs
        start = time.perf_counter()
        Z, C = self.Z, area.J_B
        s = softmax_shares(psi_free)
        n = self.N_A * s
        c, dc = covariance_weights(s, self.N_A, self.kappa)
        p = self.probabilities(theta_P, U)
        Pu = pairs.aggregate(p)
        mu = pairs.TuT @ Pu                                   # (C, Z)
        Wp = pairs.second_moments(Pu)
        m = (mu * n[None, :]).ravel()                         # rows k*Z + b
        factors = []
        for b, tract in enumerate(self.tracts):
            V = pairs.unpack(Wp[:, b])
            V -= np.outer(mu[:, b], mu[:, b])
            Bb = np.diag(self.d[tract["rows"]]) + c[b] * V[np.ix_(self.src, self.src)]
            factors.append(cho_factor(Bb, lower=True, overwrite_a=True, check_finite=False))
        X, K_factor = [None] * Z, None
        logdet = sum(2.0 * float(np.log(np.diag(f[0])).sum()) for f in factors)
        if self.Lr is not None:
            K = np.eye(self.Lr.shape[1])
            for b, (tract, f) in enumerate(zip(self.tracts, factors)):
                X[b] = cho_solve(f, tract["Lr"], check_finite=False)
                K += tract["Lr"].T @ X[b]
            K_factor = cho_factor(K, lower=True, check_finite=False)
            logdet += 2.0 * float(np.log(np.diag(K_factor[0])).sum())
        e = area.Y - m
        r = self._solve(factors, X, K_factor, e)
        quad = float(e @ r)
        ev = FreeShareEvaluation(loglik=-0.5 * quad - 0.5 * logdet, quad=quad, logdet=logdet,
                                 m=m, mu=mu, p=p, shares=s, n=n, c=c, dc=dc, factors=factors,
                                 X=X, K_factor=K_factor, r=r)
        if gradient:
            self._gradient(ev, Pu)
        ev.timings["total"] = time.perf_counter() - start
        return ev

    def _gradient(self, ev: FreeShareEvaluation, Pu: np.ndarray) -> None:
        pairs, Z, C = self.pairs, self.Z, self.area.J_B
        R = ev.r.reshape(C, Z)
        A = pairs.Tu @ R                                      # (U, Z)
        abar = (ev.mu * R).sum(axis=0)
        packed = np.zeros((pairs.n_pairs, Z))
        Gmu = np.zeros((pairs.J0, Z))
        for b in range(Z):
            G0 = self.fold(self.zone_inverse(ev, b))
            packed[:, b] = pairs.pack(G0)
            Gmu[:, b] = G0 @ ev.mu[:, b]
        Q = pairs.quadratic_forms(packed)                    # (U, Z)
        sig = pairs.Tu @ Gmu                                 # (U, Z)
        n, c = ev.n[None, :], ev.c[None, :]
        phi = n * A + 0.5 * c * (A * A - 2.0 * A * abar[None, :]) - 0.5 * c * (Q - 2.0 * sig)
        ev.grad_U = self.covariance_with_h(ev.p, pairs.expand(phi))
        EA2 = (Pu * A * A).sum(axis=0)
        EQ = (Pu * Q).sum(axis=0)
        muGmu = (Pu * sig).sum(axis=0)
        dL_dc = 0.5 * (EA2 - abar ** 2) - 0.5 * (EQ - muGmu)
        ev.grad_s = self.N_A * abar + dL_dc * ev.dc
        s = ev.shares
        ev.grad_psi = (s * (ev.grad_s - float(s @ ev.grad_s)))[:-1]

    # -- Gauss–Newton information ------------------------------------------

    def jacobian(self, ev: FreeShareEvaluation) -> np.ndarray:
        """``J = ∂m/∂x`` (``n_rows × n_params``, dense; module notes)."""
        Z, C, K = self.Z, self.area.J_B, self.K
        J = np.zeros((Z * C, self.n_params))
        for b, tract in enumerate(self.tracts):
            p = ev.p[:, b]
            gbar = self.H.T @ p
            Cb = (self.pairs.TuT @ self.pairs.aggregate(p[:, None] * self.H)
                  - np.outer(ev.mu[:, b], gbar))
            J[np.ix_(tract["rows"], np.arange(b * K, (b + 1) * K))] = ev.n[b] * Cb[self.src]
        s = ev.shares
        for j in range(Z - 1):
            col = Z * K + j
            for b, tract in enumerate(self.tracts):
                coef = self.N_A * s[j] * ((1.0 if b == j else 0.0) - s[b])
                J[tract["rows"], col] = coef * ev.mu[self.src, b]
        return J

    def gn_information(self, ev: FreeShareEvaluation) -> np.ndarray:
        J = self.jacobian(ev)
        info = J.T @ self.s_solve(ev, J)
        return 0.5 * (info + info.T)

    def prior_hessian(self, shares: np.ndarray, tau: float) -> np.ndarray:
        """The priors' Hessian: ``τ⁻² I`` on ``U``, ``a Z (diag s - ss')`` on
        the free coordinates of ``ψ``."""
        Z, K = self.Z, self.K
        H = np.zeros((self.n_params, self.n_params))
        H[np.arange(Z * K), np.arange(Z * K)] = 1.0 / tau ** 2
        s = np.asarray(shares, float)[:-1]
        H[Z * K:, Z * K:] = self.share_prior * Z * (np.diag(s) - np.outer(s, s))
        return H


def free_share_engine(area: AreaInputs, H: np.ndarray, theta: np.ndarray, G: np.ndarray,
                      logb_zones: np.ndarray, pairs: PairOperator, N_A: float, kappa: float,
                      share_prior: float = SHARE_PRIOR) -> FreeShareLikelihood:
    """:class:`FreeShareLikelihood` with ``θ_A'h`` folded into the zones' log
    bases (evaluate with ``θ_P = 0``)."""
    offset = np.asarray(H, float) @ np.asarray(theta, float)
    return FreeShareLikelihood(area, G, np.asarray(logb_zones, float) + offset[:, None], pairs,
                               N_A, kappa, share_prior)


# --------------------------------------------------------------------------
# The MAP of (U, ψ)
# --------------------------------------------------------------------------


class FreeShareObjective:
    """``F(x)`` (module notes), in the shape :mod:`~pmedm_vb.latent.map_fit`'s
    optimizers expect (a flat vector)."""

    def __init__(self, engine: FreeShareLikelihood, theta: np.ndarray, tau: float):
        self.engine, self.theta, self.tau = engine, np.asarray(theta, float), float(tau)
        self.shape = (engine.n_params,)
        self.evaluations = 0

    def _prior(self, U, s) -> float:
        return (0.5 * float(np.sum(U * U)) / self.tau ** 2
                - self.engine.share_prior * float(np.sum(np.log(s))))

    def value_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        eng = self.engine
        U, psi = eng.split(x)
        ev = eng.evaluate(self.theta, U, psi)
        self.evaluations += 1
        s = ev.shares
        grad_U = -ev.grad_U + U / self.tau ** 2
        grad_psi = -ev.grad_psi - eng.share_prior * (1.0 - eng.Z * s[:-1])
        return -ev.loglik + self._prior(U, s), eng.join(grad_U, grad_psi)

    def value(self, x: np.ndarray) -> float:
        eng = self.engine
        U, psi = eng.split(x)
        ev = eng.evaluate(self.theta, U, psi, gradient=False)
        self.evaluations += 1
        return -ev.loglik + self._prior(U, ev.shares)

    def gn_hessian(self, x: np.ndarray) -> np.ndarray:
        eng = self.engine
        U, psi = eng.split(x)
        ev = eng.evaluate(self.theta, U, psi, gradient=False)
        self.evaluations += 1
        return eng.gn_information(ev) + eng.prior_hessian(ev.shares, self.tau)


@dataclass
class FreeShareFit:
    tau: float
    x: np.ndarray
    U: np.ndarray
    psi: np.ndarray
    shares: np.ndarray
    objective: float
    loglik: float
    grad_max: float
    grad_max_U: float
    grad_max_psi: float
    iterations: int
    evaluations: int
    converged: bool
    message: str
    seconds: float
    start: str
    optimizer: str
    log_evidence: float = float("nan")
    logdet_hessian: float = float("nan")

    def info(self) -> dict:
        return {"start": self.start, "optimizer": self.optimizer, "tau": self.tau,
                "objective": self.objective, "loglik": self.loglik,
                "grad_max": self.grad_max, "grad_max_U": self.grad_max_U,
                "grad_max_psi": self.grad_max_psi, "iterations": self.iterations,
                "evaluations": self.evaluations, "converged": self.converged,
                "message": self.message, "seconds": self.seconds,
                "U_rms": float(np.sqrt(np.mean(self.U ** 2))),
                "log_evidence": self.log_evidence, "logdet_hessian": self.logdet_hessian}


def fit_free_shares(engine: FreeShareLikelihood, theta: np.ndarray, tau: float,
                    x0: np.ndarray, optimizer: str, start: str, *, gtol: float = 1e-4,
                    ftol: float = 1e-14, max_iter: int = 20000) -> FreeShareFit:
    """One fit of ``F`` from ``x0``: ``m10``, ``m20`` (plain L-BFGS) or ``gn``
    (L-BFGS preconditioned by the Gauss–Newton Hessian), as
    :func:`~pmedm_vb.latent.map_fit.run_fit`."""
    t0 = time.perf_counter()
    obj = FreeShareObjective(engine, theta, tau)
    times: list[float] = []
    if optimizer == "gn":
        x, nit, ok, msg = _gn_lbfgs(obj, np.asarray(x0, float), gtol, ftol, max_iter, times)
    elif optimizer in ("m10", "m20"):
        x, nit, ok, msg = _lbfgs(obj, np.asarray(x0, float), 10 if optimizer == "m10" else 20,
                                 gtol, ftol, max_iter, times)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    value, grad = obj.value_grad(x)
    U, psi = engine.split(x)
    ev = engine.evaluate(theta, U, psi, gradient=False)
    k = engine.Z * engine.K
    return FreeShareFit(tau=float(tau), x=x, U=U, psi=psi, shares=ev.shares,
                        objective=float(value), loglik=float(ev.loglik),
                        grad_max=float(np.abs(grad).max()),
                        grad_max_U=float(np.abs(grad[:k]).max()),
                        grad_max_psi=float(np.abs(grad[k:]).max()) if grad.size > k else 0.0,
                        iterations=nit, evaluations=obj.evaluations, converged=ok, message=msg,
                        seconds=time.perf_counter() - t0, start=start, optimizer=optimizer)


def free_share_starts(engine: FreeShareLikelihood, theta: np.ndarray, tau: float,
                      start_shares: np.ndarray, **kwargs) -> tuple[FreeShareFit, list]:
    """The MAP: the lower objective of two starts at ``U = 0``, both with the
    Gauss–Newton-preconditioned L-BFGS: from ``start_shares`` (the survey's)
    and from equal shares. (Plain L-BFGS is left out: the shares' curvature,
    of order ``N_A``, is far larger than ``U``'s, and the preconditioner
    absorbs the difference.)"""
    Z, K = engine.Z, engine.K
    x_survey = engine.join(np.zeros((Z, K)), shares_to_psi(start_shares))
    x_equal = engine.join(np.zeros((Z, K)), np.zeros(Z - 1))
    fits = [fit_free_shares(engine, theta, tau, x_survey, "gn", "survey_gn", **kwargs),
            fit_free_shares(engine, theta, tau, x_equal, "gn", "equal_gn", **kwargs)]
    return min(fits, key=lambda f: f.objective), fits


def free_share_hessian(engine: FreeShareLikelihood, theta: np.ndarray, fit: FreeShareFit
                       ) -> np.ndarray:
    """``H``: the Gauss–Newton information plus the priors' Hessians at the MAP."""
    ev = engine.evaluate(theta, fit.U, fit.psi, gradient=False)
    return engine.gn_information(ev) + engine.prior_hessian(ev.shares, fit.tau)


def free_share_evidence(engine: FreeShareLikelihood, theta: np.ndarray, fit: FreeShareFit
                        ) -> FreeShareFit:
    """Laplace's approximation of ``log p(Y | τ)`` with ``ψ`` integrated too
    (reported only): ``-F(x̂) - ZK log τ - ½ log det H`` plus the Dirichlet's
    normalizing constant (common to every fit, left out)."""
    H = free_share_hessian(engine, theta, fit)
    sign, logdet = np.linalg.slogdet(H)
    if sign <= 0:
        raise np.linalg.LinAlgError("the Gauss-Newton Hessian is not positive definite")
    fit.logdet_hessian = float(logdet)
    fit.log_evidence = float(-fit.objective - fit.U.size * np.log(fit.tau) - 0.5 * logdet)
    return fit


# --------------------------------------------------------------------------
# Gaussian conditioning, posterior draws
# --------------------------------------------------------------------------


def free_share_conditional_moments(engine: FreeShareLikelihood, ev: FreeShareEvaluation,
                                   fun: Functionals, T0: sp.spmatrix,
                                   GG: sp.spmatrix | None = None
                                   ) -> tuple[np.ndarray, np.ndarray, int]:
    """``(Z, F)`` conditional means and variances given the shares (module notes)."""
    G = fun.G
    GG = G.multiply(G).tocsc() if GG is None else GG
    T0 = sp.csr_matrix(T0)
    Z, C = engine.Z, engine.area.J_B
    R = ev.r.reshape(C, Z)
    mean, var = np.zeros((Z, fun.F)), np.zeros((Z, fun.F))
    negative = 0
    for b in range(Z):
        p = ev.p[:, b]
        Eg = np.asarray(G.T @ p).ravel()
        E2 = np.asarray(GG.T @ p).ravel()
        k1 = np.asarray((T0.multiply(p[:, None]).T @ G).todense())   # (C, F)
        kg = k1 - np.outer(ev.mu[:, b], Eg)
        Gbb = engine.zone_inverse(ev, b)
        mean[b] = ev.n[b] * Eg + ev.c[b] * (kg.T @ R[:, b])
        vb = ev.c[b] * (E2 - Eg ** 2) - ev.c[b] ** 2 * np.einsum("ij,ij->j", kg, Gbb @ kg)
        negative += int((vb < 0).sum())
        var[b] = np.maximum(vb, 0.0)
    return mean, var, negative


@dataclass
class FreeSharePosterior:
    samples: np.ndarray        # (S, Z, F) float32
    shares: np.ndarray         # (S, Z) the drawn shares
    negative_variances: int
    seconds: float
    seconds_per_draw: float


def free_share_posterior_samples(engine: FreeShareLikelihood, theta: np.ndarray,
                                 fit: FreeShareFit, fun: Functionals, T0: sp.spmatrix,
                                 n_draws: int, rng: np.random.Generator) -> FreeSharePosterior:
    """Laplace draws of ``(U, ψ)``, one Gaussian draw of every functional per
    draw (module notes)."""
    start = time.perf_counter()
    R = cholesky(free_share_hessian(engine, theta, fit), lower=False)
    xi = rng.standard_normal((fit.x.size, n_draws))
    xs = (fit.x[:, None] + solve_triangular(R, xi, lower=False)).T
    GG = fun.G.multiply(fun.G).tocsc()
    out = np.empty((n_draws, engine.Z, fun.F), dtype=np.float32)
    shares = np.empty((n_draws, engine.Z))
    negative = 0
    for k, x in enumerate(xs):
        U, psi = engine.split(x)
        ev = engine.evaluate(theta, U, psi, gradient=False)
        mean, var, neg = free_share_conditional_moments(engine, ev, fun, T0, GG)
        negative += neg
        shares[k] = ev.shares
        out[k] = mean + np.sqrt(var) * rng.standard_normal(mean.shape)
    seconds = time.perf_counter() - start
    return FreeSharePosterior(out, shares, negative, seconds, seconds / max(n_draws, 1))


# --------------------------------------------------------------------------
# The realized MAP with the shares free
# --------------------------------------------------------------------------


@dataclass
class FreeShareMapResult:
    result: object             # the last PairConditionalResult (base ρ⊗p, ρ its own shares)
    shares: np.ndarray
    outer_iterations: int
    share_change: float
    converged: bool
    seconds: float

    def info(self) -> dict:
        return {**self.result.info(), "outer_iterations": self.outer_iterations,
                "share_change": self.share_change, "outer_converged": self.converged,
                "seconds_total": self.seconds}


def solve_free_share_map(T: sp.spmatrix, p: np.ndarray, start_shares: np.ndarray, N_A: float,
                         W: float, Y: np.ndarray, Sigma: np.ndarray, *, tol: float = 1e-10,
                         max_iter: int = 200, unit_tol: float = 0.01, max_outer: int = 200
                         ) -> FreeShareMapResult:
    """The realized MAP with the shares free (module notes). ``p`` is
    ``M × Z``; ``start_shares`` the first ``ρ``.

    The alternation ``ρ ← shares(solve(ρ))`` converges linearly. It is
    accelerated by SQUAREM (Varadhan and Roland's squared extrapolation, in
    ``log ρ``, step length ``α = -‖r‖/‖v‖`` capped at ``-1``), with a
    safeguard: the extrapolated point is kept only if it does not raise the
    free-shares objective ``W Σ_b π_b KL(r_b ‖ p_b) + ½ e'Σ⁻¹e``, which is the
    solver's primal less ``W KL(π ‖ ρ)``. It stops when one plain step moves
    no zone's total by more than ``unit_tol`` units."""
    t0 = time.perf_counter()
    pT = np.asarray(p, float).T
    state = {"lam": None, "calls": 0}

    def step(rho):
        res = solve_conditional_pairs(T, rho[:, None] * pT, N_A, W, Y, Sigma, tol=tol,
                                      max_iter=max_iter,
                                      init=None if state["lam"] is None else state["lam"].ravel())
        state["lam"], state["calls"] = res.lam, state["calls"] + 1
        pi = res.shares
        kl = float(np.sum(np.where(pi > 0, pi * np.log(np.maximum(pi, 1e-300) / rho), 0.0)))
        return pi, res, res.primal - W * kl

    def norm(x):
        x = np.exp(x - x.max())
        return x / x.sum()

    rho = np.asarray(start_shares, float)
    rho = rho / rho.sum()
    share_tol = unit_tol / N_A
    change, res = np.inf, None
    while state["calls"] < max_outer:
        pi1, res1, f1 = step(rho)
        change = float(np.abs(pi1 - rho).max())
        if change <= share_tol:
            res, rho = res1, pi1
            break
        pi2, res2, f2 = step(pi1)
        r = np.log(pi1) - np.log(rho)
        v = np.log(pi2) - 2.0 * np.log(pi1) + np.log(rho)
        vn = float(np.linalg.norm(v))
        if vn <= 1e-300:
            res, rho = res2, pi2
            continue
        alpha = min(-float(np.linalg.norm(r)) / vn, -1.0)
        candidate = norm(np.log(rho) - 2.0 * alpha * r + alpha ** 2 * v)
        pi3, res3, f3 = step(candidate)
        if f3 <= f2:
            res, rho = res3, pi3
        else:
            res, rho = res2, pi2
    return FreeShareMapResult(result=res, shares=rho, outer_iterations=state["calls"],
                              share_change=change, converged=bool(change <= share_tol),
                              seconds=time.perf_counter() - t0)


__all__ = ["FreeShareLikelihood", "FreeShareObjective", "FreeShareFit", "SHARE_PRIOR",
           "covariance_weights", "fit_free_shares", "free_share_conditional_moments",
           "free_share_engine", "free_share_evidence", "free_share_hessian",
           "free_share_posterior_samples", "free_share_starts", "shares_to_psi",
           "softmax_shares", "solve_free_share_map"]
