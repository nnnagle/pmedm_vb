"""The overdispersed realized level of model version 4 (step 4b,
``PLAN_latent_step4b.md`` version 1; ``MODEL_latent.qmd`` version 4).

The area total ``N_A`` is known; the zone totals are random. Over (zone, atom)
pairs, with prior zone shares ``a_b = π⁰_b`` and the zones' superpopulations
``p_b``, ``q_bi = a_b p_bi`` and

    C ~ DirMult(N_A, κ q),   E C = N_A q,   Cov C = φ N_A (diag q - qq'),
    φ = (N_A + κ)/(1 + κ)                                        (standard).

(:func:`dirmult_phi`, :func:`draw_dirmult`; the zone totals are
``DirMult(N_A, κ a)`` by the Dirichlet's aggregation property.)

**Collapsed likelihood** (:class:`PairLikelihood`; derived here). With
``v = Σ_b a_b D_b T'p_b`` (zone ``b``'s rows: ``a_b μ_b``, ``μ_b = T0'p_b``)
and ``c = φ N_A``:

    m = N_A v,
    S = Σ + c [blockdiag_b(a_b T0'diag(p_b)T0) - vv'] = B̃ - c vv',
    B̃ = blockdiag_b(D_b + c a_b W_b) + LL'      (W_b = T0'diag(p_b)T0).

``B̃⁻¹`` and ``log det B̃`` by step 2's per-zone Cholesky factors and
Woodbury identity; then, with ``w = B̃⁻¹v`` and ``γ = 1 - c v'w``,
``S⁻¹x = B̃⁻¹x + (c/γ) w (w'x)``, ``log det S = log det B̃ + log γ``,
``[S⁻¹]_bb = [B̃⁻¹]_bb + (c/γ) w_b w_b'`` and ``s = S⁻¹v = w/γ``.

**Gradient** (derived here). With ``r = S⁻¹(Y - m)``, ``A_bi = t_i'r_b``,
``Q_bi = t_i'[S⁻¹]_bb t_i`` and ``σ_bi = t_i's_b``,
``∂L/∂u_b = Cov_{p_b}(g, φ_b)`` with

    φ_bi = N_A a_b A_bi + ½ c a_b (A_bi² - 2 (r'v) A_bi) - ½ c a_b (Q_bi - 2 σ_bi):

the mean term ``r'dm``, the quadratic form's ``½ r'dS r`` and the
``-½ tr(S⁻¹dS)`` of the log determinant, with
``dS = c[blockdiag(a_b dW_b) - dv v' - v dv']`` and
``dp_bi = p_bi (g_i - ḡ_b)'du_b``. The engine runs on the deviation basis
``g`` with ``θ_A'h`` folded into the log bases, as
:func:`~pmedm_vb.latent.region_engine.deviation_engine` (evaluate with
``θ_P = 0``).

**Gauss–Newton information** (:meth:`PairLikelihood.gn_information`):
``J_b = ∂m_b/∂u_b = N_A a_b Cov_{p_b}(t, g)`` on zone ``b``'s rows, and
``I = J'S⁻¹J = blockdiag(J_b'B_b⁻¹J_b) - Z'K⁻¹Z + (c/γ) yy'``
(``Z_b = X_b'J_b``, ``y_b = w_b'J_b``).

**Gaussian conditioning over pairs** (:func:`pair_conditional_moments`; the
model's stage 2). For a functional of zone ``b``, ``g`` over pairs equal to
``g_i`` on zone ``b``'s pairs: with ``Eg = g'p_b``, ``E2 = (g∘g)'p_b``,
``k1 = T0'diag(p_b) g``,

    E   = N_A a_b Eg + c a_b (k1'r_b - Eg v'r),
    Var = c (a_b E2 - a_b² Eg²) - c² a_b² [k1'[S⁻¹]_bb k1 - 2 Eg k1's_b + Eg² v's]

(``k = c 𝒟T'(diag q - qq')g = c a_b (k1 on zone b's rows - Eg v)``). The
zone total is the functional ``g = 1``.

**Conditional PMEDM over pairs** (:func:`solve_conditional_pairs`): the MAP
of the realized population, to second order,
``min_r W KL(r ‖ q) + ½ e'Σ⁻¹e``, ``e = N_A Σ_b D_bT'r_b - Y``, ``r`` over
pairs, ``W = N_A/φ``; dual
``f(λ) = λ'Y + W log Σ_{b,i} q_bi exp{-(N_A/W) t_i'λ_b} + ½ λ'Σλ``, by
dual Newton on the dense Hessian
``Σ + (N_A²/W)[blockdiag(T0'diag(r_b)T0) - uu']`` (``u_b = T0'r_b``), the
zone shares free.

**B1 predictive log density** (:func:`predictive_log_density`, for choosing
``c`` in arm K2): the posterior of B1's true tables plus their survey error,
each cell taken as Gaussian with the posterior sample's mean and variance
plus the published variance, cells independent (a choice made here), summed
over cells and zones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from pmedm_vb.latent.area import AreaInputs
from pmedm_vb.latent.collapsed import PairOperator
from pmedm_vb.latent.posterior import Functionals, draw_U
from pmedm_vb.latent.region_engine import RegionLikelihood


def dirmult_phi(N: float, kappa: float) -> float:
    """``φ = (N + κ)/(1 + κ)``; 1 for ``κ = ∞``."""
    return 1.0 if not np.isfinite(kappa) else (N + kappa) / (1.0 + kappa)


def draw_dirmult(N: int, alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One ``DirMult(N, α)`` draw (Dirichlet by Gammas, then multinomial)."""
    alpha = np.asarray(alpha, float)
    shape = alpha.shape
    g = rng.gamma(alpha.ravel())
    return rng.multinomial(int(N), g / g.sum()).reshape(shape)


# --------------------------------------------------------------------------
# The collapsed likelihood over pairs
# --------------------------------------------------------------------------


@dataclass
class PairEvaluation:
    loglik: float
    quad: float
    logdet: float
    m: np.ndarray
    mu: np.ndarray               # (C, Z) T0'p_b
    p: np.ndarray                # (M, Z)
    factors: list                # Cholesky of B_b per zone
    X: list                      # B_b⁻¹ L_b
    K_factor: tuple | None
    v: np.ndarray                # stacked
    w: np.ndarray                # B̃⁻¹ v
    gamma: float
    r: np.ndarray                # S⁻¹(Y - m)
    c: float
    grad_U: np.ndarray | None = None
    timings: dict = field(default_factory=dict)

    @property
    def s(self) -> np.ndarray:
        return self.w / self.gamma


class PairLikelihood(RegionLikelihood):
    """Model version 4's collapsed likelihood for a region (module notes):
    ``RegionLikelihood``'s support and probabilities, its own ``evaluate``."""

    def __init__(self, area: AreaInputs, G: np.ndarray, logb_zones: np.ndarray,
                 pairs: PairOperator, shares: np.ndarray, N_A: float, kappa: float):
        super().__init__(area, G, logb_zones, pairs)
        if area.J_T != 0 or any(len(t["members"]) != 1 for t in self.tracts):
            raise ValueError("PairLikelihood expects the regional layout (a zone per tract)")
        self.a = np.asarray(shares, float)
        self.N_A, self.kappa = float(N_A), float(kappa)
        self.phi = dirmult_phi(self.N_A, self.kappa)
        self.c = self.phi * self.N_A

    # -- B̃ and S solves --------------------------------------------------

    def _btilde_solve(self, ev_parts, x: np.ndarray) -> np.ndarray:
        factors, X, K_factor = ev_parts
        out = np.empty_like(x, dtype=float)
        Lz = 0.0
        for t, (tract, f) in enumerate(zip(self.tracts, factors)):
            z = cho_solve(f, x[tract["rows"]], check_finite=False)
            out[tract["rows"]] = z
            if K_factor is not None:
                Lz = Lz + tract["Lr"].T @ z
        if K_factor is not None:
            corr = cho_solve(K_factor, Lz, check_finite=False)
            for tract, Xt in zip(self.tracts, X):
                out[tract["rows"]] -= Xt @ corr
        return out

    def s_solve(self, ev: PairEvaluation, x: np.ndarray) -> np.ndarray:
        """``S⁻¹ x`` (stacked)."""
        y = self._btilde_solve((ev.factors, ev.X, ev.K_factor), x)
        return y + (ev.c / ev.gamma) * ev.w * float(ev.w @ x)

    def zone_inverse(self, ev: PairEvaluation, b: int) -> np.ndarray:
        """``[S⁻¹]_bb`` in zone ``b``'s local order."""
        tract = self.tracts[b]
        inv = self.tract_inverse(ev.factors[b], ev.X[b], ev.K_factor)
        wb = ev.w[tract["rows"]]
        return inv + (ev.c / ev.gamma) * np.outer(wb, wb)

    # -- evaluation -------------------------------------------------------

    def evaluate(self, theta_P, U=None, *, gradient: bool = True, keep_S: bool = False
                 ) -> PairEvaluation:
        area, pairs = self.area, self.pairs
        clock = time.perf_counter
        start = clock()
        Z = area.B
        p = self.probabilities(theta_P, U)
        Pu = pairs.aggregate(p)
        mu = pairs.TuT @ Pu                                   # (C, Z)
        Wp = pairs.second_moments(Pu)
        v = (mu * self.a[None, :]).ravel()                    # rows k*Z + b
        m = self.N_A * v
        factors, kept = [], []
        for b, tract in enumerate(self.tracts):
            Bb = np.diag(self.d[tract["rows"]])
            Bb += self.c * self.a[b] * pairs.unpack(Wp[:, b])[np.ix_(self.src, self.src)]
            if keep_S:
                kept.append(Bb.copy())
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
        parts = (factors, X, K_factor)
        w = self._btilde_solve(parts, v)
        gamma = 1.0 - self.c * float(v @ w)
        if not gamma > 0:
            raise np.linalg.LinAlgError(f"S is not positive definite (gamma = {gamma:.3g})")
        logdet += np.log(gamma)
        e = area.Y - m
        r = self._btilde_solve(parts, e)
        r = r + (self.c / gamma) * w * float(w @ e)
        quad = float(e @ r)
        ev = PairEvaluation(loglik=-0.5 * quad - 0.5 * logdet, quad=quad, logdet=logdet, m=m,
                            mu=mu, p=p, factors=factors, X=X, K_factor=K_factor, v=v, w=w,
                            gamma=gamma, r=r, c=self.c)
        if keep_S:
            ev.timings["S"] = kept
        if gradient:
            ev.grad_U = self._gradient(ev)
        ev.timings["total"] = clock() - start
        return ev

    def _gradient(self, ev: PairEvaluation) -> np.ndarray:
        pairs, Z, C = self.pairs, self.area.B, self.area.J_B
        R = ev.r.reshape(C, Z)
        Smat = ev.s.reshape(C, Z)
        A = pairs.Tu @ R                                     # (U, Z)
        sig = pairs.Tu @ Smat
        rv = float(ev.r @ ev.v)
        packed = np.zeros((pairs.n_pairs, Z))
        for b in range(Z):
            packed[:, b] = pairs.pack(self.fold(self.zone_inverse(ev, b)))
        Q = pairs.quadratic_forms(packed)
        a, c = self.a[None, :], self.c
        phi = (self.N_A * a * A + 0.5 * c * a * (A * A - 2.0 * rv * A)
               - 0.5 * c * a * (Q - 2.0 * sig))
        return self.covariance_with_h(ev.p, pairs.expand(phi))

    # -- Gauss–Newton information -----------------------------------------

    def jacobians(self, ev: PairEvaluation) -> list[np.ndarray]:
        """``J_b = N_A a_b Cov_{p_b}(t0, g)`` (``C × K``) per zone."""
        out = []
        for b in range(self.area.B):
            p = ev.p[:, b]
            gbar = self.H.T @ p
            Cb = (self.pairs.TuT @ self.pairs.aggregate(p[:, None] * self.H)
                  - np.outer(ev.mu[:, b], gbar))
            out.append(self.N_A * self.a[b] * Cb[self.src])
        return out

    def gn_information(self, ev: PairEvaluation) -> np.ndarray:
        Z, K = self.area.B, self.K
        J = self.jacobians(ev)
        info = np.zeros((Z * K, Z * K))
        Zm = None if ev.K_factor is None else np.zeros((ev.K_factor[0].shape[0], Z * K))
        y = np.zeros(Z * K)
        for b, tract in enumerate(self.tracts):
            cols = slice(b * K, (b + 1) * K)
            info[cols, cols] = J[b].T @ cho_solve(ev.factors[b], J[b], check_finite=False)
            if Zm is not None:
                Zm[:, cols] = ev.X[b].T @ J[b]
            y[cols] = ev.w[tract["rows"]] @ J[b]
        if Zm is not None:
            info -= Zm.T @ cho_solve(ev.K_factor, Zm, check_finite=False)
        info += (ev.c / ev.gamma) * np.outer(y, y)
        return 0.5 * (info + info.T)


def pair_engine(area: AreaInputs, H: np.ndarray, theta: np.ndarray, G: np.ndarray,
                logb_zones: np.ndarray, pairs: PairOperator, shares: np.ndarray, N_A: float,
                kappa: float) -> PairLikelihood:
    """:class:`PairLikelihood` on the deviation basis ``g`` with ``θ_A'h``
    folded into the zones' log bases (evaluate with ``θ_P = 0``), as
    :func:`~pmedm_vb.latent.region_engine.deviation_engine`."""
    offset = np.asarray(H, float) @ np.asarray(theta, float)
    return PairLikelihood(area, G, np.asarray(logb_zones, float) + offset[:, None], pairs,
                          shares, N_A, kappa)


# --------------------------------------------------------------------------
# Gaussian conditioning over pairs
# --------------------------------------------------------------------------


def pair_conditional_moments(engine: PairLikelihood, ev: PairEvaluation, fun: Functionals,
                             T0: sp.spmatrix, GG: sp.spmatrix | None = None
                             ) -> tuple[np.ndarray, np.ndarray, int]:
    """``(Z, F)`` conditional means and variances over pairs (module notes)."""
    G = fun.G
    GG = G.multiply(G).tocsc() if GG is None else GG
    T0 = sp.csr_matrix(T0)
    Z, C = engine.area.B, engine.area.J_B
    R = ev.r.reshape(C, Z)
    Smat = ev.s.reshape(C, Z)
    vr, vs = float(ev.v @ ev.r), float(ev.v @ ev.s)
    c, NA = engine.c, engine.N_A
    mean, var = np.zeros((Z, fun.F)), np.zeros((Z, fun.F))
    negative = 0
    for b in range(Z):
        p, a = ev.p[:, b], engine.a[b]
        Eg = np.asarray(G.T @ p).ravel()
        E2 = np.asarray(GG.T @ p).ravel()
        k1 = np.asarray((T0.multiply(p[:, None]).T @ G).todense())   # (C, F)
        Gbb = engine.zone_inverse(ev, b)
        mean[b] = NA * a * Eg + c * a * (k1.T @ R[:, b] - Eg * vr)
        quad = (np.einsum("ij,ij->j", k1, Gbb @ k1) - 2.0 * Eg * (k1.T @ Smat[:, b])
                + Eg ** 2 * vs)
        vb = c * (a * E2 - a * a * Eg ** 2) - c * c * a * a * quad
        negative += int((vb < 0).sum())
        var[b] = np.maximum(vb, 0.0)
    return mean, var, negative


@dataclass
class PairPosterior:
    samples: np.ndarray        # (S, Z, F) float32
    negative_variances: int
    seconds: float
    seconds_per_draw: float


def pair_posterior_samples(engine: PairLikelihood, theta: np.ndarray, U_hat: np.ndarray,
                           tau: float, fun: Functionals, T0: sp.spmatrix, n_draws: int,
                           rng: np.random.Generator) -> PairPosterior:
    """Laplace draws of ``U`` (``H = I(Û) + τ⁻² I``), one Gaussian draw of every
    functional per draw (as :func:`~pmedm_vb.latent.posterior.posterior_samples`)."""
    from scipy.linalg import cholesky

    start = time.perf_counter()
    ev = engine.evaluate(theta, U_hat, gradient=False)
    H = engine.gn_information(ev)
    H[np.diag_indices_from(H)] += 1.0 / tau ** 2
    Us = draw_U(U_hat, cholesky(H, lower=False), n_draws, rng)
    GG = fun.G.multiply(fun.G).tocsc()
    out = np.empty((n_draws, engine.area.B, fun.F), dtype=np.float32)
    negative = 0
    for s, U in enumerate(Us):
        ev = engine.evaluate(theta, U, gradient=False)
        mean, var, neg = pair_conditional_moments(engine, ev, fun, T0, GG)
        negative += neg
        out[s] = mean + np.sqrt(var) * rng.standard_normal(mean.shape)
    seconds = time.perf_counter() - start
    return PairPosterior(out, negative, seconds, seconds / max(n_draws, 1))


# --------------------------------------------------------------------------
# The conditional PMEDM over pairs
# --------------------------------------------------------------------------


@dataclass
class PairConditionalResult:
    lam: np.ndarray            # (C, Z)
    r: np.ndarray              # (Z, M) over pairs, sums to 1
    fitted: np.ndarray         # N_A Σ_b D_b T' r_b, stacked
    objective: float
    primal: float
    gap: float
    kkt_max: float
    kkt_relative: float
    decrement: float
    iterations: int
    converged: bool
    seconds: float

    @property
    def shares(self) -> np.ndarray:
        return self.r.sum(axis=1)

    def info(self) -> dict:
        return {"objective": self.objective, "primal": self.primal, "gap": self.gap,
                "relative_gap": self.gap / max(1.0, abs(self.primal)),
                "kkt_max": self.kkt_max, "kkt_relative": self.kkt_relative,
                "decrement": self.decrement, "iterations": self.iterations,
                "converged": self.converged, "seconds": self.seconds}


def solve_conditional_pairs(T: sp.spmatrix, q: np.ndarray, N_A: float, W: float, Y: np.ndarray,
                            Sigma: np.ndarray, *, tol: float = 1e-10, max_iter: int = 200,
                            init: np.ndarray | None = None) -> PairConditionalResult:
    """The conditional PMEDM over pairs (module notes). ``q`` is ``Z × M``
    (sums to 1 over all pairs), ``Y`` stacked (cells × zones, zone fastest)."""
    start = time.perf_counter()
    T = sp.csr_matrix(T, dtype=float)
    Tt = T.T.tocsr()
    Z, M = q.shape
    C = T.shape[1]
    with np.errstate(divide="ignore"):
        logq = np.log(q).T                                        # (M, Z)
    ratio = N_A / W
    Sigma = np.asarray(Sigma, float)
    Y = np.asarray(Y, float)

    def state(lam):
        L = lam.reshape(C, Z)
        s = -ratio * (T @ L)                                      # (M, Z)
        logits = logq + s
        logZ = float(logsumexp(logits))
        r = np.exp(logits - logZ)                                 # (M, Z)
        u = Tt @ r                                                # (C, Z)
        fitted = N_A * u.ravel()
        sl = Sigma @ lam
        return {"lam": lam, "r": r, "s": s, "logZ": logZ, "u": u, "fitted": fitted,
                "sl": sl, "value": float(lam @ Y + W * logZ + 0.5 * lam @ sl),
                "grad": Y - fitted + sl}

    def hessian(st):
        H = Sigma.copy()
        for b in range(Z):
            rows = np.arange(C) * Z + b
            rb = st["r"][:, b]
            H[np.ix_(rows, rows)] += (N_A ** 2 / W) * (Tt @ T.multiply(rb[:, None])).toarray()
        u = st["u"].ravel()
        H -= (N_A ** 2 / W) * np.outer(u, u)
        return H

    lam = np.zeros(C * Z) if init is None else np.asarray(init, float).ravel()
    st = state(lam)
    converged, decrement, it = False, np.inf, 0
    for it in range(max_iter + 1):
        direction = -cho_solve(cho_factor(hessian(st), lower=True), st["grad"])
        slope = float(st["grad"] @ direction)
        decrement = -0.5 * slope
        if decrement <= tol:
            converged = True
            break
        if it == max_iter:
            break
        step = 1.0
        for _ in range(50):
            cand = state(st["lam"] + step * direction)
            if cand["value"] <= st["value"] + 0.25 * step * slope:
                break
            step *= 0.5
        else:
            break
        st = cand
    r, s = st["r"], st["s"]
    kl = float(np.sum(np.where(r > 0, r * s, 0.0))) - st["logZ"]
    e = st["fitted"] - Y
    primal = W * kl + 0.5 * float(e @ cho_solve(cho_factor(Sigma, lower=True), e))
    kkt = float(np.abs(e - st["sl"]).max())
    return PairConditionalResult(
        lam=st["lam"].reshape(C, Z), r=r.T.copy(), fitted=st["fitted"], objective=st["value"],
        primal=primal, gap=primal + st["value"], kkt_max=kkt,
        kkt_relative=kkt / max(float(np.abs(e).max()), 1e-300), decrement=decrement,
        iterations=it, converged=converged, seconds=time.perf_counter() - start)


# --------------------------------------------------------------------------
# Choosing c: the B1 predictive log density
# --------------------------------------------------------------------------


def predictive_log_density(samples: np.ndarray, observed: np.ndarray,
                           published_var: np.ndarray) -> float:
    """Σ over zones and cells of ``log N(y; mean, var_post + var_published)``
    (module notes). ``samples`` ``(S, Z, F_B1)``; the others ``(Z, F_B1)``."""
    s = np.asarray(samples, float)
    mean, var = s.mean(axis=0), s.var(axis=0, ddof=1)
    total = var + np.asarray(published_var, float)
    return float(np.sum(-0.5 * np.log(2 * np.pi * total)
                        - 0.5 * (np.asarray(observed) - mean) ** 2 / total))


def prior_shares(Y_fitted: np.ndarray, unit_cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``π⁰_b = N̂_b / Σ N̂``, ``N̂_b`` the zone's estimated unit total (its
    fitted unit-type cells), clipped at a tenth of a unit; also ``N̂``."""
    N_hat = np.maximum(np.asarray(Y_fitted)[:, unit_cells].sum(axis=1), 0.1)
    return N_hat / N_hat.sum(), N_hat


__all__ = ["PairLikelihood", "dirmult_phi", "draw_dirmult", "pair_conditional_moments",
           "pair_posterior_samples", "predictive_log_density", "prior_shares",
           "solve_conditional_pairs", "pair_engine"]
